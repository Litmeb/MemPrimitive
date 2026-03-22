"""Baseline: retrieval primitive."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import ClassVar, Final

from sentence_transformers import SentenceTransformer

from ..core import MemoryStore, ModuleSpec, Packet, RetrievedSet
from ..interfaces import RetrievalModule

from ._trace import copy_trace


class RecencyRetrieval(RetrievalModule):
    """Retrieve up to ``top_k`` records: keyword filter when possible, else by recency.

    Constructor: ``top_k`` must be a positive integer. ``layer`` selects
    ``store.iter_records(layer)``; ``None`` means all layers (order follows
    ``MemoryStore.iter_records``).

    ``run`` requires ``packet.query``. Tokenizes query text on whitespace; if any
    token matches substring-wise in a record, keeps only matches (order: newest
    first among candidates); otherwise takes the ``top_k`` newest records overall.
    Does not mutate the store. Populates ``packet.retrieved`` and score dicts
    (rank/strategy, not dense similarity scores).
    """

    spec = ModuleSpec(
        name="recency_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("RecencyRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("RecencyRetrieval requires packet.query.")

        all_records = store.iter_records(self.layer)
        query_tokens = {token for token in packet.query.text.casefold().split() if token}

        ordered = list(reversed(all_records))
        filtered_records = ordered
        keyword_mode = False
        if query_tokens:
            matching_records = [
                record
                for record in ordered
                if any(token in record.text.casefold() for token in query_tokens)
            ]
            if matching_records:
                filtered_records = matching_records
                keyword_mode = True

        selected_records = filtered_records[: self.top_k]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "strategy": "keyword+recency" if keyword_mode else "recency",
            }
            for rank, record in enumerate(selected_records, start=1)
        ]
        retrieved = RetrievedSet(
            items=selected_records,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "matched_by_keyword": keyword_mode,
                "candidate_count": len(all_records),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class EmbeddingSimilarityRetrieval(RetrievalModule):
    """Retrieve the ``top_k`` records with highest embedding cosine similarity.

    Constructor: ``top_k`` must be a positive integer. ``layer`` selects
    ``store.iter_records(layer)``; ``None`` means all layers. ``embedding_model``
    defaults to the same sentence-transformers model as ``BasicRepresentation``.

    ``run`` requires ``packet.query``. Uses ``query.embedding`` when present;
    otherwise encodes ``query.text`` and returns an updated packet with the cached
    query embedding. Only records with ``record.embedding`` participate in scoring.
    Records with missing or dimension-mismatched embeddings are skipped. Does not
    mutate the store. Populates ``packet.retrieved`` and score dicts with numeric
    similarity values.
    """

    spec = ModuleSpec(
        name="embedding_similarity_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("record.embedding",),
    )
    _embedding_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def __init__(
        self,
        top_k: int = 3,
        layer: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if top_k <= 0:
            raise ValueError("EmbeddingSimilarityRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.embedding_model = embedding_model

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("EmbeddingSimilarityRetrieval requires packet.query.")

        query = packet.query
        reused_query_embedding = query.embedding is not None
        query_embedding = list(query.embedding) if query.embedding is not None else self._embed_text(query.text)
        if query.embedding is None:
            query = replace(query, embedding=query_embedding)

        all_records = store.iter_records(self.layer)
        scored_candidates: list[tuple[float, object]] = []
        skipped_dim_mismatch = 0
        for record in all_records:
            if record.embedding is None:
                continue
            if len(record.embedding) != len(query_embedding):
                skipped_dim_mismatch += 1
                continue
            score = self._cosine_similarity(query_embedding, record.embedding)
            scored_candidates.append((score, record))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_candidates = scored_candidates[: self.top_k]
        selected_records = [record for _, record in selected_candidates]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": score,
                "strategy": "embedding_similarity",
            }
            for rank, (score, record) in enumerate(selected_candidates, start=1)
        ]
        retrieved = RetrievedSet(
            items=selected_records,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "strategy": "embedding_similarity",
                "candidate_count": len(all_records),
                "embedding_candidate_count": len(scored_candidates),
                "reused_query_embedding": reused_query_embedding,
                "skipped_dim_mismatch_count": skipped_dim_mismatch,
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, query=query, retrieved=retrieved, trace=trace), store

    def _embed_text(self, text: str) -> list[float]:
        model = self._embedding_cache.get(self.embedding_model)
        if model is None:
            model = SentenceTransformer(self.embedding_model)
            self._embedding_cache[self.embedding_model] = model
        return [float(value) for value in model.encode(text, normalize_embeddings=True).tolist()]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


BASELINE_SLOT: Final[str] = "retrieval"
BASELINE_CLASSES: Final[tuple[type[RetrievalModule], ...]] = (RecencyRetrieval, EmbeddingSimilarityRetrieval)
