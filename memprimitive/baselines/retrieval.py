"""Baseline: retrieval primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from math import sqrt
import re
from typing import Any, ClassVar, Final

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_NOTE_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
    TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
    TOPOLOGY_TAG_INDEX_CONTRACT,
    UNIT_EMBEDDING_CONTRACT,
    UNIT_ENTITIES_CONTRACT,
    UNIT_TAGS_CONTRACT,
    normalize_contracts,
)
from ..core import MemoryStore, ModuleSpec, Packet, Query, RetrievedSet
from ..interfaces import RetrievalModule

from ..utils._amem_family import (
    DEFAULT_CATEGORY,
    DEFAULT_NOTE_NAMESPACE,
    build_enhanced_embedding_text,
    note_payload_from_record,
    repair_note_payload,
)
from ..utils._graph_family import graph_metadata_from_record
from ..utils._reflexion_family import DEFAULT_MEMORY_SIZE, DEFAULT_REFLECTION_LAYER
from ..utils._template import (
    PromptPlan,
    ensure_prompt_plan,
    project_packet_runtime_for_template,
    project_query_for_template,
    render_prompt_plan,
    text_prompt,
)
from ..utils._trace import copy_trace


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


_RETRIEVAL_SOURCES: Final[frozenset[str]] = frozenset({"store", "retrieved"})
_QUERY_REWRITE_STRATEGIES: Final[frozenset[str]] = frozenset({"llm", "regex"})
_QUERY_REWRITE_REGEX_FLAGS: Final[dict[str, int]] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


def _normalize_retrieval_source(source: str) -> str:
    normalized = str(source).strip().casefold()
    if normalized not in _RETRIEVAL_SOURCES:
        raise ValueError("retrieval source must be one of: retrieved, store.")
    return normalized


def _normalize_query_rewrite_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().casefold()
    if normalized not in _QUERY_REWRITE_STRATEGIES:
        raise ValueError("query rewrite strategy must be one of: llm, regex.")
    return normalized


def _prompt_is_template(plan: PromptPlan) -> bool:
    return plan.mode == "structured" or (
        isinstance(plan.template, str)
        and "{{" in plan.template
        and "}}" in plan.template
    )


def _summarize_raw_llm_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        summary: dict[str, Any] = {
            "type": "object",
            "keys": sorted(str(key) for key in output.keys()),
        }
        if "query" in output:
            summary["query"] = output.get("query")
        if "queries" in output and isinstance(output.get("queries"), list):
            summary["queries"] = list(output["queries"])
        return summary
    if isinstance(output, list):
        return {"type": "list", "length": len(output)}
    return {"type": type(output).__name__, "value": output}


def _rewrite_metadata(query: Query, *, source: str, index: int) -> dict[str, Any]:
    return {
        **dict(query.metadata),
        "rewrite": {
            "source": source,
            "from_query_id": query.query_id,
            "index": index,
        },
    }


def _normalize_rewritten_query_texts(
    raw_texts: list[Any],
    *,
    strip_queries: bool,
    drop_empty_queries: bool,
    max_queries: int,
) -> tuple[list[str], dict[str, Any]]:
    normalized: list[str] = []
    seen: set[str] = set()
    dropped_empty_count = 0
    duplicate_count = 0
    over_limit_count = 0
    for raw in raw_texts:
        text = str(raw)
        if strip_queries:
            text = text.strip()
        if not text:
            dropped_empty_count += 1
            if drop_empty_queries:
                continue
        if text in seen:
            duplicate_count += 1
            continue
        if len(normalized) >= max_queries:
            over_limit_count += 1
            continue
        seen.add(text)
        normalized.append(text)

    if not normalized:
        raise ValueError("QueryRewriteRetrieval produced no usable rewritten queries.")

    return normalized, {
        "dropped_empty_count": dropped_empty_count,
        "duplicate_count": duplicate_count,
        "over_limit_count": over_limit_count,
    }


def _dedupe_records_by_id(records: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen_record_ids: set[str] = set()
    for record in records:
        record_id = getattr(record, "record_id", None)
        if not isinstance(record_id, str) or record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        deduped.append(record)
    return deduped


def _candidate_records(packet: Packet, store: MemoryStore, *, source: str, layer: str | None = None) -> list[Any]:
    if source == "store":
        return store.iter_records(layer)

    retrieved = packet.retrieved if packet.retrieved is not None else RetrievedSet()
    records = list(retrieved.items)
    if layer is not None:
        records = [record for record in records if getattr(record, "layer", None) == layer]
    return _dedupe_records_by_id(records)


def _with_retrieved(packet: Packet, retrieved: RetrievedSet, *, query=None) -> Packet:
    trace = copy_trace(packet)
    trace["retrieval"] = retrieved.trace
    if query is None:
        return replace(packet, retrieved=retrieved, trace=trace)
    return replace(packet, query=query, retrieved=retrieved, trace=trace)


def _resolve_retrieval_queries(packet: Packet, *, module_name: str) -> tuple[list[Any], bool]:
    if packet.queries:
        return list(packet.queries), True
    if packet.query is not None:
        return [packet.query], False
    raise ValueError(f"{module_name} requires packet.query.")


def _single_query_packet(packet: Packet, *, query) -> Packet:
    return replace(packet, query=query)


def _merge_retrieval_packets(
    packet: Packet,
    branch_packets: list[Packet],
    *,
    module_name: str,
) -> Packet:
    merged_items: list[Any] = []
    merged_scores: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    per_query: list[dict[str, Any]] = []
    merged_queries: list[Any] = []

    for index, branch_packet in enumerate(branch_packets):
        branch_query = branch_packet.query
        if branch_query is not None:
            merged_queries.append(branch_query)
        retrieved = branch_packet.retrieved if branch_packet.retrieved is not None else RetrievedSet()
        per_query.append(
            {
                "index": index,
                "query_id": getattr(branch_query, "query_id", None),
                "query_text": getattr(branch_query, "text", None),
                "returned_count": len(retrieved.items),
                "trace": dict(retrieved.trace),
            }
        )
        for item_index, record in enumerate(retrieved.items):
            record_id = getattr(record, "record_id", None)
            if not isinstance(record_id, str) or record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)
            merged_items.append(record)
            merged_scores.append(dict(retrieved.scores[item_index] if item_index < len(retrieved.scores) else {}))

    merged_trace = {
        "module": module_name,
        "query_count": len(branch_packets),
        "query_ids": [entry["query_id"] for entry in per_query],
        "merge_strategy": "query_order_dedupe",
        "per_query": per_query,
        "final_returned_count": len(merged_items),
    }
    merged_retrieved = RetrievedSet(
        items=merged_items,
        scores=merged_scores,
        trace=merged_trace,
    )
    trace = copy_trace(packet)
    trace["retrieval"] = merged_trace
    return replace(
        packet,
        queries=merged_queries,
        retrieved=merged_retrieved,
        trace=trace,
    )


class _MultiQueryRetrievalMixin:
    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        queries, used_multi_query = _resolve_retrieval_queries(packet, module_name=self.spec.name)
        if not used_multi_query and len(queries) == 1:
            return self._run_single_query(packet, store)

        branch_packets: list[Packet] = []
        for query in queries:
            branch_packet = _single_query_packet(packet, query=query)
            branch_packet, store = self._run_single_query(branch_packet, store)
            branch_packets.append(branch_packet)
        return _merge_retrieval_packets(packet, branch_packets, module_name=self.spec.name), store


class QueryRewriteRetrieval(RetrievalModule):
    """Rewrite the query first, then delegate retrieval to another retriever."""

    spec = ModuleSpec(
        name="query_rewrite_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        retriever: RetrievalModule,
        *,
        strategy: str = "llm",
        prompt: PromptPlan | str | None = None,
        allow_multi_query: bool = False,
        regex_rules: list[dict[str, Any]] | None = None,
        include_original: bool = False,
        max_queries: int | None = None,
        strip_queries: bool = True,
        drop_empty_queries: bool = True,
    ) -> None:
        if not isinstance(retriever, RetrievalModule):
            raise TypeError("QueryRewriteRetrieval.retriever must be a RetrievalModule instance.")
        self.retriever = retriever
        self.strategy = _normalize_query_rewrite_strategy(strategy)
        self.prompt = prompt
        self.allow_multi_query = bool(allow_multi_query)
        self.regex_rules = [dict(rule) for rule in (regex_rules or [])]
        self.include_original = bool(include_original)
        resolved_max_queries = max_queries if max_queries is not None else (4 if self.allow_multi_query else 1)
        if int(resolved_max_queries) <= 0:
            raise ValueError("QueryRewriteRetrieval max_queries must be positive.")
        self.max_queries = int(resolved_max_queries)
        self.strip_queries = bool(strip_queries)
        self.drop_empty_queries = bool(drop_empty_queries)

        if self.strategy == "llm":
            if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
                raise ValueError("QueryRewriteRetrieval strategy='llm' requires a non-empty prompt.")
        elif not self.regex_rules:
            raise ValueError("QueryRewriteRetrieval strategy='regex' requires regex_rules.")

    def get_requires_contracts(self) -> frozenset[str]:
        return self.retriever.get_requires_contracts()

    def get_produces_contracts(self) -> frozenset[str]:
        return self.retriever.get_produces_contracts()

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("QueryRewriteRetrieval requires packet.query.")

        rewritten_packet, rewrite_trace, store = self._rewrite_packet(packet, store)
        delegated_packet, store = self.retriever.run(rewritten_packet, store)
        retrieval_trace = dict(delegated_packet.trace.get("retrieval", {}))
        merged_trace = {
            **retrieval_trace,
            "module": self.spec.name,
            "wrapped_retriever": self.retriever.spec.name,
            "query_rewrite": rewrite_trace,
        }
        trace = copy_trace(delegated_packet)
        trace["retrieval"] = merged_trace
        retrieved = delegated_packet.retrieved
        if retrieved is not None:
            retrieved = replace(retrieved, trace=merged_trace)
        return replace(delegated_packet, retrieved=retrieved, trace=trace), store

    def _rewrite_packet(self, packet: Packet, store: MemoryStore) -> tuple[Packet, dict[str, Any], MemoryStore]:
        if self.strategy == "llm":
            return self._rewrite_packet_with_llm(packet, store)
        return self._rewrite_packet_with_regex(packet, store)

    def _rewrite_packet_with_llm(
        self,
        packet: Packet,
        store: MemoryStore,
    ) -> tuple[Packet, dict[str, Any], MemoryStore]:
        rewritten_prompt, prompt_trace, store = self._render_prompt(packet, store)
        raw_output = self._llm_json(
            user=json.dumps(
                {
                    "query": packet.query.text,
                    "allow_multi_query": self.allow_multi_query,
                    "max_queries": self.max_queries,
                    "prompt": rewritten_prompt,
                },
                ensure_ascii=False,
            )
        )
        raw_queries = self._extract_llm_query_texts(raw_output)
        query_texts, normalization_trace = _normalize_rewritten_query_texts(
            raw_queries,
            strip_queries=self.strip_queries,
            drop_empty_queries=self.drop_empty_queries,
            max_queries=self.max_queries,
        )
        rewritten_packet = self._packet_with_rewritten_queries(packet, query_texts, source="llm")
        rewrite_trace = {
            "rewrite_strategy": "llm",
            "allow_multi_query": self.allow_multi_query,
            "include_original": self.include_original,
            "used_original_query": False,
            "max_queries": self.max_queries,
            "returned_query_count": len(query_texts),
            "rewritten_query_texts": list(query_texts),
            "original_query_text": packet.query.text,
            "prompt_is_template": _prompt_is_template(ensure_prompt_plan(self.prompt, metadata_mode="prompt")),
            "raw_output_summary": _summarize_raw_llm_output(raw_output),
            **prompt_trace,
            **normalization_trace,
        }
        return rewritten_packet, rewrite_trace, store

    def _rewrite_packet_with_regex(
        self,
        packet: Packet,
        store: MemoryStore,
    ) -> tuple[Packet, dict[str, Any], MemoryStore]:
        current_text = packet.query.text
        rule_summaries: list[dict[str, Any]] = []
        for rule in self.regex_rules:
            pattern = str(rule.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("QueryRewriteRetrieval regex rules require non-empty pattern.")
            repl = str(rule.get("repl", ""))
            count = int(rule.get("count", 0))
            flags_value, normalized_flags = self._regex_flags(rule.get("flags"))
            current_text, changed_count = re.subn(pattern, repl, current_text, count=count, flags=flags_value)
            rule_summaries.append(
                {
                    "pattern": pattern,
                    "repl": repl,
                    "count": count,
                    "flags": normalized_flags,
                    "changed": changed_count > 0,
                    "replacement_count": changed_count,
                }
            )

        query_texts, normalization_trace = _normalize_rewritten_query_texts(
            [current_text],
            strip_queries=self.strip_queries,
            drop_empty_queries=self.drop_empty_queries,
            max_queries=1,
        )
        rewritten_packet = self._packet_with_rewritten_queries(packet, query_texts, source="regex")
        rewrite_trace = {
            "rewrite_strategy": "regex",
            "allow_multi_query": False,
            "include_original": self.include_original,
            "used_original_query": False,
            "rule_count": len(self.regex_rules),
            "original_query_text": packet.query.text,
            "rewritten_query_text": query_texts[0],
            "rewritten_query_texts": list(query_texts),
            "returned_query_count": 1,
            "rules": rule_summaries,
            **normalization_trace,
        }
        return rewritten_packet, rewrite_trace, store

    def _packet_with_rewritten_queries(self, packet: Packet, query_texts: list[str], *, source: str) -> Packet:
        rewritten_queries = [
            Query(
                text=text,
                metadata=_rewrite_metadata(packet.query, source=source, index=index),
            )
            for index, text in enumerate(query_texts)
        ]
        if self.include_original:
            original_query = replace(
                packet.query,
                metadata={
                    **dict(packet.query.metadata),
                    "rewrite_original_preserved": True,
                },
            )
            rewritten_queries.append(original_query)
        primary_query = rewritten_queries[0]
        if len(rewritten_queries) == 1:
            return replace(packet, query=primary_query, queries=None)
        return replace(packet, query=primary_query, queries=rewritten_queries)

    def _render_prompt(self, packet: Packet, store: MemoryStore) -> tuple[str, dict[str, Any], MemoryStore]:
        plan = ensure_prompt_plan(
            self.prompt,
            metadata_mode="prompt",
            context_builder=lambda current_packet, current_store: {
                "query": project_query_for_template(current_packet.query),
                "runtime": project_packet_runtime_for_template(current_packet),
                "retrieval": {
                    "wrapped_retriever": self.retriever.spec.name,
                    "allow_multi_query": self.allow_multi_query,
                    "max_queries": self.max_queries,
                },
            },
        )
        rendered_prompt, prompt_trace, store = render_prompt_plan(plan, packet=packet, store=store)
        return rendered_prompt, prompt_trace, store

    def _llm_json(self, *, user: str) -> Any:
        from ..utils._runtime import get_runtime

        runtime = get_runtime()
        runtime.require_llm(capability="QueryRewriteRetrieval")
        return runtime.json(
            system=(
                "You rewrite retrieval queries. Return strict JSON only with either "
                "{\"query\": \"...\"} or {\"queries\": [\"...\", \"...\"]}. "
                "Do not include any text outside the JSON object."
            ),
            user=user,
        )

    def _extract_llm_query_texts(self, output: Any) -> list[Any]:
        if isinstance(output, dict):
            if self.allow_multi_query and isinstance(output.get("queries"), list):
                return list(output["queries"])
            if isinstance(output.get("query"), str):
                return [output["query"]]
            if isinstance(output.get("queries"), list):
                queries = list(output["queries"])
                return queries[:1] if not self.allow_multi_query else queries
        raise ValueError("QueryRewriteRetrieval LLM rewrite requires JSON object with 'query' or 'queries'.")

    def _regex_flags(self, raw_flags: Any) -> tuple[int, list[str]]:
        if raw_flags is None:
            return 0, []
        if isinstance(raw_flags, str):
            flag_names = [raw_flags]
        elif isinstance(raw_flags, (list, tuple)):
            flag_names = list(raw_flags)
        else:
            raise ValueError("QueryRewriteRetrieval regex rule flags must be a string or list of strings.")

        resolved = 0
        normalized: list[str] = []
        for raw_name in flag_names:
            name = str(raw_name).strip().upper()
            if not name:
                continue
            try:
                resolved |= _QUERY_REWRITE_REGEX_FLAGS[name]
            except KeyError as exc:
                options = ", ".join(sorted(_QUERY_REWRITE_REGEX_FLAGS))
                raise ValueError(f"Unsupported regex flag {name!r}; expected one of: {options}.") from exc
            if name not in normalized:
                normalized.append(name)
        return resolved, normalized


def _empty_retrieved(
    packet: Packet,
    *,
    module_name: str,
    top_k: int,
    source: str,
    query=None,
    **trace_fields: Any,
) -> Packet:
    retrieved = RetrievedSet(
        items=[],
        scores=[],
        trace={
            "module": module_name,
            "top_k": top_k,
            "source": source,
            **trace_fields,
        },
    )
    return _with_retrieved(packet, retrieved, query=query)


class RecencyRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Retrieve up to ``top_k`` latest records by recency only.

    Constructor: ``top_k`` must be a positive integer. ``layer`` selects
    ``store.iter_records(layer)``; ``None`` means all layers (order follows
    ``MemoryStore.iter_records``).

    ``run`` requires ``packet.query``. Returns the ``top_k`` newest records from
    the selected layer scope. Query text is accepted for interface consistency
    but does not affect ranking. Does not mutate the store. Populates
    ``packet.retrieved`` and score dicts (rank/strategy, not dense similarity
    scores).
    """

    spec = ModuleSpec(
        name="recency_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None, *, source: str = "store") -> None:
        if top_k <= 0:
            raise ValueError("RecencyRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.source = _normalize_retrieval_source(source)

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("RecencyRetrieval requires packet.query.")

        all_records = _candidate_records(packet, store, source=self.source, layer=self.layer)
        ordered = list(reversed(all_records))
        selected_records = ordered[: self.top_k]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "strategy": "recency",
            }
            for rank, record in enumerate(selected_records, start=1)
        ]
        retrieved = RetrievedSet(
            items=selected_records,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "source": self.source,
                "candidate_count": len(all_records),
            },
        )
        return _with_retrieved(packet, retrieved), store


class KeywordCountRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Rank records by query-token hit count, then break ties by recency."""

    spec = ModuleSpec(
        name="keyword_count_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None, *, source: str = "store") -> None:
        if top_k <= 0:
            raise ValueError("KeywordCountRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.source = _normalize_retrieval_source(source)

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("KeywordCountRetrieval requires packet.query.")

        tokens = _query_tokens(packet.query.text)
        all_records = list(reversed(_candidate_records(packet, store, source=self.source, layer=self.layer)))
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
                "source": self.source,
                "candidate_count": len(all_records),
            },
        )
        return _with_retrieved(packet, retrieved), store


class EmbeddingSimilarityRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
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
    requires_contracts = frozenset({UNIT_EMBEDDING_CONTRACT})
    _embedding_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def __init__(
        self,
        top_k: int = 3,
        layer: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        source: str = "store",
    ) -> None:
        if top_k <= 0:
            raise ValueError("EmbeddingSimilarityRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.embedding_model = embedding_model
        self.source = _normalize_retrieval_source(source)

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("EmbeddingSimilarityRetrieval requires packet.query.")

        query = packet.query
        reused_query_embedding = query.embedding is not None
        query_embedding = list(query.embedding) if query.embedding is not None else self._embed_text(query.text)
        if query.embedding is None:
            query = replace(query, embedding=query_embedding)

        all_records = _candidate_records(packet, store, source=self.source, layer=self.layer)
        if not all_records:
            packet = _empty_retrieved(
                packet,
                module_name=self.spec.name,
                top_k=self.top_k,
                source=self.source,
                query=query,
                strategy="embedding_similarity",
                candidate_count=0,
                embedding_candidate_count=0,
                reused_query_embedding=reused_query_embedding,
                skipped_dim_mismatch_count=0,
            )
            return packet, store
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
                "source": self.source,
                "candidate_count": len(all_records),
                "embedding_candidate_count": len(scored_candidates),
                "reused_query_embedding": reused_query_embedding,
                "skipped_dim_mismatch_count": skipped_dim_mismatch,
            },
        )
        return _with_retrieved(packet, retrieved, query=query), store

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


class TagRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Rank records by overlap between query tokens and representation tags."""

    spec = ModuleSpec(
        name="tag_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )
    requires_contracts = frozenset({UNIT_TAGS_CONTRACT, TOPOLOGY_TAG_INDEX_CONTRACT})

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("TagRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
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


class EntityRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Rank records by overlap between query entities/tokens and record entities."""

    spec = ModuleSpec(
        name="entity_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )
    requires_contracts = frozenset({UNIT_ENTITIES_CONTRACT})

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("EntityRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
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


class BM25Retrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Rank records with BM25 over text plus representation keywords, then recency."""

    spec = ModuleSpec(
        name="bm25_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None, *, source: str = "store") -> None:
        if top_k <= 0:
            raise ValueError("BM25Retrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.source = _normalize_retrieval_source(source)

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("BM25Retrieval requires packet.query.")

        query_tokens = _tokenize_text(packet.query.text)
        all_records = list(reversed(_candidate_records(packet, store, source=self.source, layer=self.layer)))
        if not all_records:
            packet = _empty_retrieved(
                packet,
                module_name=self.spec.name,
                top_k=self.top_k,
                source=self.source,
                candidate_count=0,
                scored_count=0,
                avg_doc_len=0.0,
                used_recency_fallback=False,
            )
            return packet, store

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
                "source": self.source,
                "candidate_count": len(all_records),
                "scored_count": len(scored),
                "avg_doc_len": avg_doc_len,
                "used_recency_fallback": used_recency_fallback,
            },
        )
        return _with_retrieved(packet, retrieved), store


class GraphNeighborRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
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
    requires_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT})

    def __init__(self, top_k: int = 3, layer: str = "knowledge_graph", *, include_seed_records: bool = False) -> None:
        if top_k <= 0:
            raise ValueError("GraphNeighborRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.target_layer = layer
        self.include_seed_records = include_seed_records

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
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


class ExpandRetrievedGraphNeighbors(RetrievalModule):
    """Expand graph neighbors from ``packet.retrieved.items`` seed records."""

    spec = ModuleSpec(
        name="expand_retrieved_graph_neighbors",
        slot="retrieval",
        input_requirements=("retrieved.items",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
    )
    requires_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT})

    def __init__(
        self,
        top_k: int = 3,
        *,
        layer: str = "knowledge_graph",
        include_seed_records: bool = True,
        per_seed_top_k: int | None = None,
        dedupe: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("ExpandRetrievedGraphNeighbors requires top_k > 0.")
        if per_seed_top_k is not None and per_seed_top_k <= 0:
            raise ValueError("ExpandRetrievedGraphNeighbors requires per_seed_top_k > 0 when provided.")
        self.top_k = top_k
        self.layer = layer
        self.target_layer = layer
        self.include_seed_records = include_seed_records
        self.per_seed_top_k = per_seed_top_k
        self.dedupe = dedupe

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        retrieved_input = packet.retrieved if packet.retrieved is not None else RetrievedSet()
        seed_records = [
            record
            for record in _dedupe_records_by_id(list(retrieved_input.items))
            if getattr(record, "layer", None) == self.layer
        ]
        if not seed_records:
            packet = _empty_retrieved(
                packet,
                module_name=self.spec.name,
                top_k=self.top_k,
                source="retrieved",
                layer=self.layer,
                per_seed_top_k=self.per_seed_top_k,
                include_seed_records=self.include_seed_records,
                dedupe=self.dedupe,
                seed_record_ids=[],
                expanded_neighbor_ids=[],
                returned_count=0,
            )
            return packet, store

        items: list[Any] = []
        scores: list[dict[str, Any]] = []
        seen_record_ids: set[str] = set()
        expanded_neighbor_ids: list[str] = []

        def append_result(record, *, strategy: str, seed_record_id: str, hop: int) -> None:
            if self.dedupe and record.record_id in seen_record_ids:
                return
            seen_record_ids.add(record.record_id)
            items.append(record)
            scores.append(
                {
                    "record_id": record.record_id,
                    "rank": len(items),
                    "strategy": strategy,
                    "seed_record_id": seed_record_id,
                    "hop": hop,
                }
            )

        for seed_record in seed_records:
            if self.include_seed_records:
                append_result(
                    seed_record,
                    strategy="graph_seed",
                    seed_record_id=seed_record.record_id,
                    hop=0,
                )
                if len(items) >= self.top_k:
                    break

            neighbors = store.iter_graph_neighbors(self.layer, seed_record.record_id)
            if self.per_seed_top_k is not None:
                neighbors = neighbors[: self.per_seed_top_k]
            for neighbor in neighbors:
                expanded_neighbor_ids.append(neighbor.record_id)
                append_result(
                    neighbor,
                    strategy="graph_expand_retrieved",
                    seed_record_id=seed_record.record_id,
                    hop=1,
                )
                if len(items) >= self.top_k:
                    break
            if len(items) >= self.top_k:
                break

        retrieved = RetrievedSet(
            items=items[: self.top_k],
            scores=scores[: self.top_k],
            trace={
                "module": self.spec.name,
                "source": "retrieved",
                "layer": self.layer,
                "top_k": self.top_k,
                "per_seed_top_k": self.per_seed_top_k,
                "include_seed_records": self.include_seed_records,
                "dedupe": self.dedupe,
                "seed_record_ids": [record.record_id for record in seed_records],
                "expanded_neighbor_ids": list(dict.fromkeys(expanded_neighbor_ids)),
                "returned_count": min(len(items), self.top_k),
            },
        )
        return _with_retrieved(packet, retrieved), store


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


class LayerAwareRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
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

    def get_requires_contracts(self) -> frozenset[str]:
        return normalize_contracts(
            contract
            for module in (self.default_retriever, *self.retriever_by_layer.values())
            for contract in module.get_requires_contracts()
        )

    def get_produces_contracts(self) -> frozenset[str]:
        return normalize_contracts(
            contract
            for module in (self.default_retriever, *self.retriever_by_layer.values())
            for contract in module.get_produces_contracts()
        )

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
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


class BufferRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
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

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
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


class VectorGraphSeedAndExpandRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Wrap generic seed retrieval plus graph-neighbor expansion for note records.

    Constructor: ``top_k`` and ``candidate_k`` must be positive. ``layer`` must
    refer to a graph layer that supports the vector index. ``neighbor_expansion_k``
    controls one-hop expansion beyond the seed set. ``agentic_search`` enables
    an optional LLM rerank over the merged candidate set, while
    ``query_expand_with_llm`` lets the module build a retrieval-oriented query
    projection before embedding.

    ``run`` requires ``packet.query`` and does not mutate ``store``. The module
    reuses ``query.embedding`` when present; otherwise it embeds the query or
    its LLM-expanded projection and returns the updated query on the packet.
    """

    spec = ModuleSpec(
        name="vector_graph_seed_and_expand_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "index:vector", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph", "target_layer_index:vector"),
    )
    requires_contracts = frozenset({RECORD_NOTE_PAYLOAD_CONTRACT, TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT})

    def __init__(
        self,
        top_k: int = 3,
        *,
        layer: str = "knowledge_graph",
        candidate_k: int = 5,
        neighbor_expansion_k: int = 3,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        default_category: str = DEFAULT_CATEGORY,
        agentic_search: bool = False,
        query_expand_with_llm: bool = False,
        prompt: PromptPlan | str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("VectorGraphSeedAndExpandRetrieval requires top_k > 0.")
        if candidate_k <= 0:
            raise ValueError("VectorGraphSeedAndExpandRetrieval requires candidate_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.target_layer = layer
        self.candidate_k = candidate_k
        self.neighbor_expansion_k = neighbor_expansion_k
        self.note_namespace = note_namespace
        self.default_category = default_category
        self.agentic_search = agentic_search
        self.query_expand_with_llm = query_expand_with_llm
        self.prompt = prompt

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("VectorGraphSeedAndExpandRetrieval requires packet.query.")

        runtime = None
        query = packet.query
        query_payload = repair_note_payload(
            {"content": query.text, "note_text": query.text},
            fallback_content=query.text,
            default_category="query",
        )
        query_expansion_prompt_trace: dict[str, Any] | None = None
        if self.query_expand_with_llm:
            from ..utils._runtime import get_runtime

            runtime = get_runtime()
            runtime.require_llm(capability="Vector graph seed-and-expand query expansion")
            query_expansion_system_prompt, query_expansion_prompt_trace, store = self._render_query_expansion_prompt(packet, store)
            raw = runtime.json(
                system=query_expansion_system_prompt,
                user=json.dumps({"query": query.text}, ensure_ascii=False),
            )
            query_payload = repair_note_payload(raw, fallback_content=query.text, default_category="query")
            query_payload["content"] = query_payload["content"] or query.text
        if query.embedding is not None:
            query = replace(query, embedding=list(query.embedding))
        else:
            if runtime is None:
                from ..utils._runtime import get_runtime

                runtime = get_runtime()
            embedding_text = (
                build_enhanced_embedding_text(
                    content=query_payload["content"],
                    context=query_payload["context"],
                    keywords=query_payload["keywords"],
                    tags=query_payload["tags"],
                )
                if self.query_expand_with_llm
                else query.text
            )
            query = replace(query, embedding=runtime.embed(embedding_text))

        seed_packet, store = EmbeddingSimilarityRetrieval(
            top_k=self.candidate_k,
            layer=self.layer,
            source="store",
        ).run(
            replace(packet, query=query),
            store,
        )
        seed_retrieved = seed_packet.retrieved if seed_packet.retrieved is not None else RetrievedSet()

        combined_packet = seed_packet
        expand_trace: dict[str, Any] | None = None
        if self.neighbor_expansion_k > 0:
            expand_top_k = self.candidate_k + self.neighbor_expansion_k
            combined_packet, store = ExpandRetrievedGraphNeighbors(
                top_k=expand_top_k,
                layer=self.layer,
                include_seed_records=True,
                per_seed_top_k=self.neighbor_expansion_k,
                dedupe=True,
            ).run(
                seed_packet,
                store,
            )
            combined_retrieved = combined_packet.retrieved if combined_packet.retrieved is not None else RetrievedSet()
            expand_trace = dict(combined_retrieved.trace)
        else:
            combined_retrieved = seed_retrieved

        merged_records = list(combined_retrieved.items)
        seed_scores = {
            str(score.get("record_id")): float(score.get("score", 0.0))
            for score in seed_retrieved.scores
            if isinstance(score.get("record_id"), str)
        }

        candidate_payload = []
        for record in merged_records:
            payload = note_payload_from_record(
                record,
                note_namespace=self.note_namespace,
                default_category=self.default_category,
            )
            candidate_payload.append(
                {
                    "id": record.record_id,
                    "content": payload["content"],
                    "note_text": payload["note_text"],
                    "context": payload["context"],
                    "tags": payload["tags"],
                    "keywords": payload["keywords"],
                    "score": seed_scores.get(record.record_id, 0.0),
                }
            )

        if self.agentic_search:
            if runtime is None:
                from ..utils._runtime import get_runtime

                runtime = get_runtime()
            reranked = runtime.rerank(
                query=query.text,
                candidates=candidate_payload,
                task="Perform graph seed-and-expand retrieval over enriched note records.",
                top_k=self.top_k,
            )
            retrieval_mode = "search_agentic"
            record_by_id = {record.record_id: record for record in merged_records}
            items = [record_by_id[item["id"]] for item in reranked if item["id"] in record_by_id][: self.top_k]
            scores = [
                {
                    "record_id": item["id"],
                    "score": float(item.get("score", 0.0)),
                    "rationale": str(item.get("rationale", "")).strip(),
                    "strategy": retrieval_mode,
                }
                for item in reranked
                if item["id"] in record_by_id
            ][: self.top_k]
        else:
            items = merged_records[: self.top_k]
            scores = []
            for rank, score in enumerate(combined_retrieved.scores[: self.top_k], start=1):
                normalized_score = dict(score)
                normalized_score["rank"] = rank
                scores.append(normalized_score)
            retrieval_mode = "embedding_similarity_plus_graph_expand"

        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "retrieval_mode": retrieval_mode,
                "candidate_count": len(merged_records),
                "selected_count": len(items),
                "candidate_record_ids": [record.record_id for record in merged_records],
                "expanded_neighbor_ids": list(
                    dict.fromkeys(
                        expand_trace.get("expanded_neighbor_ids", []) if isinstance(expand_trace, dict) else []
                    )
                ),
                "note_namespace": self.note_namespace,
                "query_expand_with_llm": self.query_expand_with_llm,
                "system_prompt_is_template": bool(query_expansion_prompt_trace and query_expansion_prompt_trace.get("prompt_is_template")),
                "query_expansion_prompt_trace": query_expansion_prompt_trace,
                "seed_trace": dict(seed_retrieved.trace),
                "expand_trace": expand_trace,
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, query=query if query.embedding is not None else seed_packet.query, retrieved=retrieved, trace=trace), store

    def _render_query_expansion_prompt(self, packet: Packet, store: MemoryStore) -> tuple[str, dict[str, Any], MemoryStore]:
        default_template = (
            "Expand the query for enriched graph-memory retrieval. "
            "Return JSON with fields query_text, content, context, keywords, tags, category, attributes."
        )
        plan = ensure_prompt_plan(
            self.prompt or text_prompt(default_template),
            metadata_mode="prompt",
            context_builder=lambda current_packet, current_store: {
                "query": project_query_for_template(current_packet.query),
                "runtime": project_packet_runtime_for_template(current_packet),
                "retrieval": {
                    "layer": self.layer,
                    "candidate_k": self.candidate_k,
                    "neighbor_expansion_k": self.neighbor_expansion_k,
                    "top_k": self.top_k,
                },
            },
        )
        return render_prompt_plan(plan, packet=packet, store=store)


BASELINE_SLOT: Final[str] = "retrieval"
BASELINE_CLASSES: Final[tuple[type[RetrievalModule], ...]] = (
    RecencyRetrieval,
    KeywordCountRetrieval,
    EmbeddingSimilarityRetrieval,
    TagRetrieval,
    EntityRetrieval,
    BM25Retrieval,
    GraphNeighborRetrieval,
    ExpandRetrievedGraphNeighbors,
    VectorGraphSeedAndExpandRetrieval,
    LayerAwareRetrieval,
    BufferRetrieval,
    QueryRewriteRetrieval,
)
