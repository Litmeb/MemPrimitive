"""Baseline: retrieval primitive."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Any, ClassVar, Final

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


class _LayerScopedStore:
    """Proxy ``MemoryStore`` so retrievers only see one logical layer."""

    def __init__(self, base: MemoryStore, layer: str) -> None:
        self._base = base
        self._layer = layer

    def iter_records(self, layer: str | None = None) -> list[Any]:
        if layer is None or layer == self._layer:
            return self._base.iter_records(self._layer)
        return []

    def __getattr__(self, name: str):
        return getattr(self._base, name)


class LayerAwareRetrieval(RetrievalModule):
    """Dispatch retrieval per layer, then merge multi-layer results globally.

    Constructor: ``top_k`` must be positive. ``default_retriever`` is used for
    layers without overrides; when omitted, V1 defaults to ``RecencyRetrieval``
    with the same ``top_k``. ``retriever_by_layer`` maps layer names to concrete
    retrieval modules. ``active_layers=None`` means all topology layers.

    ``run`` requires ``packet.query``. Each active layer is retrieved
    independently against a layer-scoped view of the store. Merge order is
    global: scored results sort ahead of rank-only results, then by descending
    score or ascending rank, with layer order as a stable tie-breaker.
    """

    spec = ModuleSpec(
        name="layer_aware_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        *,
        default_retriever: RetrievalModule | None = None,
        retriever_by_layer: dict[str, RetrievalModule] | None = None,
        active_layers: tuple[str, ...] | None = None,
        top_k: int = 3,
        merge_strategy: str = "global_rank",
    ) -> None:
        if top_k <= 0:
            raise ValueError("LayerAwareRetrieval requires top_k > 0.")
        if merge_strategy != "global_rank":
            raise ValueError("LayerAwareRetrieval only supports merge_strategy='global_rank'.")

        self.top_k = top_k
        self.merge_strategy = merge_strategy
        self.default_retriever = default_retriever if default_retriever is not None else RecencyRetrieval(top_k=top_k)
        if not isinstance(self.default_retriever, RetrievalModule):
            raise TypeError("LayerAwareRetrieval.default_retriever must be a RetrievalModule.")

        raw_overrides = {} if retriever_by_layer is None else dict(retriever_by_layer)
        normalized_overrides: dict[str, RetrievalModule] = {}
        for layer_name, retriever in raw_overrides.items():
            if not isinstance(layer_name, str) or not layer_name.strip():
                raise ValueError("LayerAwareRetrieval retriever_by_layer keys must be non-empty strings.")
            if not isinstance(retriever, RetrievalModule):
                raise TypeError("LayerAwareRetrieval retriever_by_layer values must be RetrievalModule instances.")
            normalized_overrides[layer_name.strip()] = retriever
        self.retriever_by_layer = normalized_overrides
        self.active_layers = None if active_layers is None else tuple(active_layers)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("LayerAwareRetrieval requires packet.query.")

        active_layers = self._resolve_active_layers(store)
        query = packet.query
        layer_results: list[dict[str, Any]] = []
        merged_candidates: list[dict[str, Any]] = []

        for layer_index, layer_name in enumerate(active_layers):
            retriever = self.retriever_by_layer.get(layer_name, self.default_retriever)
            layer_packet = Packet(query=query, trace=packet.trace)
            scoped_store = _LayerScopedStore(store, layer_name)
            layer_packet, _ = retriever.run(layer_packet, scoped_store)
            if layer_packet.query is not None:
                query = layer_packet.query

            retrieved = layer_packet.retrieved if layer_packet.retrieved is not None else RetrievedSet()
            layer_trace = dict(retrieved.trace)
            returned_count = len(retrieved.items)
            candidate_count = int(layer_trace.get("candidate_count", len(store.iter_records(layer_name))))
            layer_results.append(
                {
                    "layer": layer_name,
                    "module": retriever.spec.name,
                    "candidate_count": candidate_count,
                    "returned_count": returned_count,
                    "trace": layer_trace,
                }
            )

            for item_index, record in enumerate(retrieved.items):
                score_info = retrieved.scores[item_index] if item_index < len(retrieved.scores) else {}
                merged_candidates.append(
                    {
                        "record": record,
                        "score_info": dict(score_info),
                        "layer": layer_name,
                        "layer_index": layer_index,
                        "item_index": item_index,
                    }
                )

        merged_candidates.sort(key=self._merge_sort_key)

        merged_items = []
        merged_scores = []
        seen_record_ids: set[str] = set()
        for candidate in merged_candidates:
            record = candidate["record"]
            if record.record_id in seen_record_ids:
                continue
            seen_record_ids.add(record.record_id)
            merged_items.append(record)
            score_info = dict(candidate["score_info"])
            score_info["layer"] = candidate["layer"]
            score_info["source_strategy"] = score_info.get("strategy", "unknown")
            score_info["merge_rank"] = len(merged_items)
            score_info["merge_key_type"] = "score" if self._extract_numeric_score(score_info) is not None else "rank"
            merged_scores.append(score_info)
            if len(merged_items) >= self.top_k:
                break

        retrieved_trace = {
            "module": self.spec.name,
            "active_layers": list(active_layers),
            "merge_strategy": self.merge_strategy,
            "per_layer": layer_results,
            "total_merged_count": len(merged_candidates),
            "final_returned_count": len(merged_items),
        }
        retrieved = RetrievedSet(items=merged_items, scores=merged_scores, trace=retrieved_trace)
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved_trace
        return replace(packet, query=query, retrieved=retrieved, trace=trace), store

    def _resolve_active_layers(self, store: MemoryStore) -> tuple[str, ...]:
        if self.active_layers is None:
            return tuple(store.topology.layer_names)
        missing = [layer for layer in self.active_layers if not store.has_layer(layer)]
        if missing:
            raise ValueError(f"LayerAwareRetrieval active_layers are not declared in the store topology: {missing}")
        return self.active_layers

    @staticmethod
    def _extract_numeric_score(score_info: dict[str, Any]) -> float | None:
        raw_score = score_info.get("score")
        if isinstance(raw_score, bool):
            return None
        if isinstance(raw_score, (int, float)):
            return float(raw_score)
        return None

    @classmethod
    def _merge_sort_key(cls, candidate: dict[str, Any]) -> tuple[Any, ...]:
        score_info = candidate["score_info"]
        numeric_score = cls._extract_numeric_score(score_info)
        rank = score_info.get("rank")
        normalized_rank = rank if isinstance(rank, int) and rank > 0 else 10**9
        if numeric_score is not None:
            return (0, -numeric_score, normalized_rank, candidate["layer_index"], candidate["item_index"])
        return (1, normalized_rank, candidate["layer_index"], candidate["item_index"])


BASELINE_SLOT: Final[str] = "retrieval"
BASELINE_CLASSES: Final[tuple[type[RetrievalModule], ...]] = (
    RecencyRetrieval,
    EmbeddingSimilarityRetrieval,
    LayerAwareRetrieval,
)
