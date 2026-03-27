"""Baseline: retrieval primitive."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Any, ClassVar, Final

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..core import MemoryStore, ModuleSpec, Packet, RetrievedSet
from ..interfaces import RetrievalModule

from ._graph_family import graph_metadata_from_record
from ._reflexion_family import DEFAULT_MEMORY_SIZE, DEFAULT_REFLECTION_LAYER
from ._trace import copy_trace


def _tokenize_text(text: str) -> list[str]:
    return [token for token in text.casefold().split() if token]


def _query_tokens(text: str) -> set[str]:
    return set(_tokenize_text(text))


def _representation(record) -> dict[str, Any]:
    value = record.metadata.get("representation", {})
    return value if isinstance(value, dict) else {}


def _document_tokens(record) -> list[str]:
    tokens = _tokenize_text(record.text)
    keywords = _representation(record).get("keywords", [])
    if isinstance(keywords, list):
        for keyword in keywords:
            tokens.extend(_tokenize_text(str(keyword)))
    return tokens


def _graph_candidate_record_ids(query) -> set[str] | None:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    nested = metadata.get("graph")
    if "graph_candidate_record_ids" in metadata:
        return set(str(value).strip() for value in metadata["graph_candidate_record_ids"] if str(value).strip())
    if isinstance(nested, dict) and "candidate_record_ids" in nested:
        return set(str(value).strip() for value in nested["candidate_record_ids"] if str(value).strip())
    return None


def _graph_seed_record_ids(query) -> list[str]:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    nested = metadata.get("graph")
    if "graph_seed_record_ids" in metadata:
        return [str(value).strip() for value in metadata["graph_seed_record_ids"] if str(value).strip()]
    if isinstance(nested, dict) and "seed_record_ids" in nested:
        return [str(value).strip() for value in nested["seed_record_ids"] if str(value).strip()]
    return []


def _graph_token_haystack(record) -> set[str]:
    graph = graph_metadata_from_record(record)
    tokens = set(_document_tokens(record))
    tokens.update(_tokenize_text(" ".join(graph["entities"])))
    for subject, predicate, obj in graph["triples"]:
        tokens.update(_tokenize_text(subject))
        tokens.update(_tokenize_text(predicate))
        tokens.update(_tokenize_text(obj))
    return tokens


def _score_graph_seed(query_text: str, record) -> float:
    query_tokens = _query_tokens(query_text)
    if not query_tokens:
        return 0.0
    graph = graph_metadata_from_record(record)
    haystack = _graph_token_haystack(record)
    overlap = len(query_tokens & haystack)
    entity_overlap = len(query_tokens & {entity.casefold() for entity in graph["entities"]})
    return float(overlap + (2 * entity_overlap))


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
        query_tokens = _query_tokens(packet.query.text)

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


class KeywordCountRetrieval(RetrievalModule):
    """Rank records by query-token hit count, then break ties by recency."""

    spec = ModuleSpec(
        name="keyword_count_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("KeywordCountRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("KeywordCountRetrieval requires packet.query.")

        tokens = _query_tokens(packet.query.text)
        all_records = list(reversed(store.iter_records(self.layer)))
        scored = []
        for order_index, record in enumerate(all_records):
            representation = _representation(record)
            haystack = set(_query_tokens(record.text))
            haystack.update(str(item).casefold() for item in representation.get("keywords", []))
            overlap = len(tokens & haystack)
            scored.append((overlap, order_index, record))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[: self.top_k]
        items = [record for _, _, record in selected]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": float(overlap),
                "strategy": "keyword_count",
            }
            for rank, (overlap, _, record) in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
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


class TagRetrieval(RetrievalModule):
    """Rank records by overlap between query tokens and representation tags."""

    spec = ModuleSpec(
        name="tag_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("TagRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("TagRetrieval requires packet.query.")

        tokens = _query_tokens(packet.query.text)
        all_records = list(reversed(store.iter_records(self.layer)))
        scored = []
        for order_index, record in enumerate(all_records):
            tags = {str(tag).casefold() for tag in _representation(record).get("tags", [])}
            overlap = len(tokens & tags)
            scored.append((overlap, order_index, record))

        if any(overlap > 0 for overlap, _, _ in scored):
            scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[: self.top_k]
        items = [record for _, _, record in selected]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": float(overlap),
                "strategy": "tag_overlap",
            }
            for rank, (overlap, _, record) in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "candidate_count": len(all_records),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class EntityRetrieval(RetrievalModule):
    """Rank records by overlap between query entities/tokens and record entities."""

    spec = ModuleSpec(
        name="entity_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("EntityRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("EntityRetrieval requires packet.query.")

        query_entities = {
            token
            for token in packet.query.text.split()
            if token and token[:1].isupper()
        }
        fallback_tokens = _query_tokens(packet.query.text)
        all_records = list(reversed(store.iter_records(self.layer)))
        scored = []
        for order_index, record in enumerate(all_records):
            representation = _representation(record)
            record_entities = {str(item) for item in representation.get("entities", [])}
            lowered_entities = {item.casefold() for item in record_entities}
            overlap = len({entity.casefold() for entity in query_entities} & lowered_entities)
            if overlap == 0:
                overlap = len(fallback_tokens & lowered_entities)
            scored.append((overlap, order_index, record))

        if any(overlap > 0 for overlap, _, _ in scored):
            scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[: self.top_k]
        items = [record for _, _, record in selected]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": float(overlap),
                "strategy": "entity_overlap",
            }
            for rank, (overlap, _, record) in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "candidate_count": len(all_records),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class BM25Retrieval(RetrievalModule):
    """Rank records with BM25 over text plus representation keywords, then recency."""

    spec = ModuleSpec(
        name="bm25_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("BM25Retrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("BM25Retrieval requires packet.query.")

        query_tokens = _tokenize_text(packet.query.text)
        all_records = list(reversed(store.iter_records(self.layer)))
        if not all_records:
            retrieved = RetrievedSet(
                items=[],
                scores=[],
                trace={
                    "module": self.spec.name,
                    "top_k": self.top_k,
                    "candidate_count": 0,
                    "scored_count": 0,
                    "used_recency_fallback": False,
                },
            )
            trace = copy_trace(packet)
            trace["retrieval"] = retrieved.trace
            return replace(packet, retrieved=retrieved, trace=trace), store

        document_tokens = [_document_tokens(record) for record in all_records]
        bm25 = BM25Okapi(document_tokens)
        raw_scores = bm25.get_scores(query_tokens).tolist()
        query_token_set = set(query_tokens)

        scored = [
            (float(score), order_index, record, len(query_token_set & set(tokens)))
            for order_index, (score, record, tokens) in enumerate(zip(raw_scores, all_records, document_tokens, strict=True))
        ]

        used_recency_fallback = not any(overlap > 0 for _, _, _, overlap in scored)
        if used_recency_fallback:
            selected = [(0.0, order_index, record) for order_index, record in enumerate(all_records[: self.top_k])]
        else:
            matching_scored = [(score, order_index, record) for score, order_index, record, overlap in scored if overlap > 0]
            matching_scored.sort(key=lambda item: (-item[0], item[1]))
            selected = matching_scored[: self.top_k]

        items = [record for _, _, record in selected]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": score,
                "strategy": "bm25",
            }
            for rank, (score, _, record) in enumerate(selected, start=1)
        ]
        avg_doc_len = sum(len(tokens) for tokens in document_tokens) / len(document_tokens) if document_tokens else 0.0
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "candidate_count": len(all_records),
                "scored_count": len(scored),
                "avg_doc_len": avg_doc_len,
                "used_recency_fallback": used_recency_fallback,
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class GraphNeighborRetrieval(RetrievalModule):
    """Expand explicit graph seed record ids into their linked neighbors.

    Constructor: ``top_k`` must be positive. ``layer`` must refer to a graph
    layer, and the query must provide seed ids via
    ``query.metadata["graph_seed_record_ids"]`` or
    ``query.metadata["graph"]["seed_record_ids"]``. ``include_seed_records``
    controls whether seed records themselves are returned alongside neighbors.

    ``run`` requires ``packet.query``. It does not mutate the store. Results are
    returned in graph-link order and can optionally be constrained to a query-
    supplied candidate set.
    """

    spec = ModuleSpec(
        name="graph_neighbor_retrieval",
        slot="retrieval",
        input_requirements=("query",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
    )

    def __init__(self, top_k: int = 3, layer: str = "knowledge_graph", *, include_seed_records: bool = False) -> None:
        if top_k <= 0:
            raise ValueError("GraphNeighborRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.target_layer = layer
        self.include_seed_records = include_seed_records

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("GraphNeighborRetrieval requires packet.query.")

        all_records = store.iter_records(self.layer)
        by_id = {record.record_id: record for record in all_records}
        seed_ids = [record_id for record_id in _graph_seed_record_ids(packet.query) if record_id in by_id]
        candidate_filter = _graph_candidate_record_ids(packet.query)

        selected_records: list[Any] = []
        selected_scores: list[dict[str, Any]] = []
        seen_record_ids: set[str] = set()
        discovered_neighbor_ids: list[str] = []

        def add_record(record, *, seed_record_id: str | None, hop: int, strategy: str) -> None:
            if record.record_id in seen_record_ids:
                return
            if candidate_filter is not None and record.record_id not in candidate_filter:
                return
            seen_record_ids.add(record.record_id)
            selected_records.append(record)
            selected_scores.append(
                {
                    "record_id": record.record_id,
                    "rank": len(selected_records),
                    "strategy": strategy,
                    "hop": hop,
                    "seed_record_id": seed_record_id,
                }
            )

        for seed_id in seed_ids:
            seed_record = by_id[seed_id]
            if self.include_seed_records:
                add_record(seed_record, seed_record_id=seed_id, hop=0, strategy="graph_seed")
                if len(selected_records) >= self.top_k:
                    break
            for neighbor in store.iter_graph_neighbors(self.layer, seed_id):
                discovered_neighbor_ids.append(neighbor.record_id)
                add_record(neighbor, seed_record_id=seed_id, hop=1, strategy="graph_neighbor")
                if len(selected_records) >= self.top_k:
                    break
            if len(selected_records) >= self.top_k:
                break

        retrieved = RetrievedSet(
            items=selected_records[: self.top_k],
            scores=selected_scores[: self.top_k],
            trace={
                "module": self.spec.name,
                "layer": self.layer,
                "top_k": self.top_k,
                "seed_record_ids": seed_ids,
                "candidate_count": len(all_records),
                "candidate_filter_count": 0 if candidate_filter is None else len(candidate_filter),
                "expanded_neighbor_ids": list(dict.fromkeys(discovered_neighbor_ids)),
                "returned_count": min(len(selected_records), self.top_k),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class GraphSeedAndExpandRetrieval(RetrievalModule):
    """Select graph seed records from query text, then expand through links.

    Constructor: ``top_k`` and ``seed_top_k`` must be positive. ``layer`` must
    refer to a graph layer. The simplified baseline uses token/entity overlap to
    choose seed records before performing one-hop graph expansion; this is an
    inferred engineering decomposition of the classic A-MEM-style motif.

    ``run`` requires ``packet.query`` and does not mutate the store. It returns
    seeds plus expanded neighbors, deduplicated and optionally bounded by a
    query-provided graph candidate set.
    """

    spec = ModuleSpec(
        name="graph_seed_and_expand_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
    )

    def __init__(
        self,
        top_k: int = 3,
        layer: str = "knowledge_graph",
        *,
        seed_top_k: int = 2,
        include_seed_records: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("GraphSeedAndExpandRetrieval requires top_k > 0.")
        if seed_top_k <= 0:
            raise ValueError("GraphSeedAndExpandRetrieval requires seed_top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.target_layer = layer
        self.seed_top_k = seed_top_k
        self.include_seed_records = include_seed_records

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("GraphSeedAndExpandRetrieval requires packet.query.")

        candidate_filter = _graph_candidate_record_ids(packet.query)
        candidate_records = [
            record
            for record in store.iter_records(self.layer)
            if candidate_filter is None or record.record_id in candidate_filter
        ]
        scored_seeds = [
            (_score_graph_seed(packet.query.text, record), order_index, record)
            for order_index, record in enumerate(reversed(candidate_records))
        ]
        scored_seeds = [item for item in scored_seeds if item[0] > 0.0]
        scored_seeds.sort(key=lambda item: (-item[0], item[1]))
        seed_candidates = scored_seeds[: self.seed_top_k]

        merged_records: list[Any] = []
        merged_scores: list[dict[str, Any]] = []
        seen_record_ids: set[str] = set()
        expanded_neighbor_ids: list[str] = []

        def append_result(record, *, strategy: str, score: float, seed_record_id: str, hop: int) -> None:
            if record.record_id in seen_record_ids:
                return
            seen_record_ids.add(record.record_id)
            merged_records.append(record)
            merged_scores.append(
                {
                    "record_id": record.record_id,
                    "rank": len(merged_records),
                    "score": score,
                    "strategy": strategy,
                    "seed_record_id": seed_record_id,
                    "hop": hop,
                }
            )

        for seed_score, _, seed_record in seed_candidates:
            if self.include_seed_records:
                append_result(
                    seed_record,
                    strategy="graph_seed",
                    score=seed_score,
                    seed_record_id=seed_record.record_id,
                    hop=0,
                )
                if len(merged_records) >= self.top_k:
                    break
            for neighbor in store.iter_graph_neighbors(self.layer, seed_record.record_id):
                expanded_neighbor_ids.append(neighbor.record_id)
                neighbor_score = max(seed_score - 0.5, 0.0)
                append_result(
                    neighbor,
                    strategy="graph_expand",
                    score=neighbor_score,
                    seed_record_id=seed_record.record_id,
                    hop=1,
                )
                if len(merged_records) >= self.top_k:
                    break
            if len(merged_records) >= self.top_k:
                break

        retrieved = RetrievedSet(
            items=merged_records[: self.top_k],
            scores=merged_scores[: self.top_k],
            trace={
                "module": self.spec.name,
                "layer": self.layer,
                "top_k": self.top_k,
                "seed_top_k": self.seed_top_k,
                "candidate_count": len(candidate_records),
                "candidate_filter_count": 0 if candidate_filter is None else len(candidate_filter),
                "seed_record_ids": [record.record_id for _, _, record in seed_candidates],
                "expanded_neighbor_ids": list(dict.fromkeys(expanded_neighbor_ids)),
                "returned_count": min(len(merged_records), self.top_k),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


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
        top_k_by_layer: dict[str, int] | None = None,
        merge_weight_by_layer: dict[str, float] | None = None,
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
        self.top_k_by_layer = {
            str(layer).strip(): int(value)
            for layer, value in ({} if top_k_by_layer is None else top_k_by_layer).items()
        }
        self.merge_weight_by_layer = {
            str(layer).strip(): float(value)
            for layer, value in ({} if merge_weight_by_layer is None else merge_weight_by_layer).items()
        }

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("LayerAwareRetrieval requires packet.query.")

        active_layers = self._resolve_active_layers(store)
        query = packet.query
        layer_results: list[dict[str, Any]] = []
        merged_candidates: list[dict[str, Any]] = []

        for layer_index, layer_name in enumerate(active_layers):
            retriever = self._retriever_for_layer(layer_name)
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
                        "merge_weight": self.merge_weight_by_layer.get(layer_name, 1.0),
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

    def _retriever_for_layer(self, layer_name: str) -> RetrievalModule:
        retriever = self.retriever_by_layer.get(layer_name, self.default_retriever)
        layer_top_k = self.top_k_by_layer.get(layer_name)
        if layer_top_k is None:
            return retriever
        if hasattr(retriever, "top_k"):
            params = dict(retriever.__dict__)
            params["top_k"] = layer_top_k
            return type(retriever)(**params)
        return retriever

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
            weighted_score = numeric_score * float(candidate.get("merge_weight", 1.0))
            return (0, -weighted_score, normalized_rank, candidate["layer_index"], candidate["item_index"])
        return (1, normalized_rank, candidate["layer_index"], candidate["item_index"])


class BufferRetrieval(RetrievalModule):
    """Read a bounded recency window from one layer instead of doing query search.

    Constructor: ``top_k`` must be positive. ``layer`` selects the temporal
    buffer to read from. ``chronological`` controls whether the returned window
    is reordered oldest-to-newest after selecting the most recent ``top_k``.

    ``run`` requires ``packet.query`` so the module remains a normal recall-slot
    primitive. It does not use query text for ranking. The store is unchanged.
    """

    spec = ModuleSpec(
        name="buffer_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        top_k: int = DEFAULT_MEMORY_SIZE,
        *,
        layer: str = DEFAULT_REFLECTION_LAYER,
        chronological: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("BufferRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.chronological = chronological

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("BufferRetrieval requires packet.query.")

        reverse_window = list(reversed(store.iter_records(self.layer)))[: self.top_k]
        items = list(reversed(reverse_window)) if self.chronological else reverse_window
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "strategy": "buffer_window",
                "layer": self.layer,
            }
            for rank, record in enumerate(items, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "layer": self.layer,
                "top_k": self.top_k,
                "chronological": self.chronological,
                "candidate_count": len(store.iter_records(self.layer)),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


BASELINE_SLOT: Final[str] = "retrieval"
BASELINE_CLASSES: Final[tuple[type[RetrievalModule], ...]] = (
    RecencyRetrieval,
    KeywordCountRetrieval,
    EmbeddingSimilarityRetrieval,
    TagRetrieval,
    EntityRetrieval,
    BM25Retrieval,
    GraphNeighborRetrieval,
    GraphSeedAndExpandRetrieval,
    LayerAwareRetrieval,
    BufferRetrieval,
)
