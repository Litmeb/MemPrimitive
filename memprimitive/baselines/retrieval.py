"""Baseline: retrieval primitive."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import sqrt
import re
from typing import Any, ClassVar, Final

from rank_bm25 import BM25Okapi

from ..contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_NOTE_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
    TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
    UNIT_EMBEDDING_CONTRACT,
    UNIT_ENTITIES_CONTRACT,
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


def _normalize_triple_query_part(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text == "*":
        return None
    return text


def _parse_triple_query(query: Query) -> tuple[dict[str, Any], str]:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    candidates = []
    if isinstance(metadata.get("triple_query"), dict):
        candidates.append(("metadata.triple_query", metadata["triple_query"]))
    if isinstance(metadata.get("triple"), dict):
        candidates.append(("metadata.triple", metadata["triple"]))

    for source, payload in candidates:
        subject = _normalize_triple_query_part(payload.get("subject"))
        relation = _normalize_triple_query_part(payload.get("relation"))
        obj = _normalize_triple_query_part(payload.get("object"))
        break
    else:
        parts = [part.strip() for part in str(query.text).split(">>")]
        if len(parts) != 3:
            raise ValueError(
                "TripleMemoryRetrieval requires a structured triple query. "
                "Use Query.metadata['triple_query'] with subject/relation/object "
                "or query.text formatted as 'subject >> relation >> object'."
            )
        subject = _normalize_triple_query_part(parts[0])
        relation = _normalize_triple_query_part(parts[1])
        obj = _normalize_triple_query_part(parts[2])
        source = "query.text"

    grounded_slots = [slot for slot, value in (("subject", subject), ("relation", relation), ("object", obj)) if value is not None]
    if not grounded_slots:
        raise ValueError(
            "TripleMemoryRetrieval requires at least one grounded slot in the structured triple query."
        )

    mode = "_".join(grounded_slots)

    return {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "mode": mode,
        "grounded_slots": grounded_slots,
    }, source


def _record_triples(record) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    representation_triples = _representation(record).get("triples", [])
    graph_triples = graph_metadata_from_record(record).get("triples", [])
    for source in (representation_triples, graph_triples):
        if not isinstance(source, list):
            continue
        for value in source:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                continue
            triple = (str(value[0]).strip(), str(value[1]).strip(), str(value[2]).strip())
            if not all(triple) or triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
    return triples


def _triple_matches_query(triple: tuple[str, str, str], *, query_spec: dict[str, Any]) -> bool:
    subject, relation, obj = triple
    query_subject = query_spec["subject"]
    query_relation = query_spec["relation"]
    query_object = query_spec["object"]

    if query_subject is not None and subject.casefold() != str(query_subject).casefold():
        return False
    if query_relation is not None and relation.casefold() != str(query_relation).casefold():
        return False
    if query_object is not None and obj.casefold() != str(query_object).casefold():
        return False
    return True


def _triple_slot_value(triple: tuple[str, str, str], slot: str) -> str:
    subject, relation, obj = triple
    if slot == "subject":
        return subject
    if slot == "relation":
        return relation
    if slot == "object":
        return obj
    raise ValueError(f"Unknown triple slot {slot!r}.")


_RETRIEVAL_SOURCES: Final[frozenset[str]] = frozenset({"store", "retrieved"})
_QUERY_REWRITE_STRATEGIES: Final[frozenset[str]] = frozenset({"llm", "regex"})
_QUERY_REWRITE_REGEX_FLAGS: Final[dict[str, int]] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}
_METADATA_MATCH_MODES: Final[frozenset[str]] = frozenset({"exact", "regex"})


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


def _normalize_metadata_match_mode(match_mode: str) -> str:
    normalized = str(match_mode).strip().casefold()
    if normalized not in _METADATA_MATCH_MODES:
        raise ValueError("metadata match_mode must be one of: exact, regex.")
    return normalized


def _normalize_metadata_field(field: str) -> str:
    normalized = str(field).strip()
    if not normalized:
        raise ValueError("MetadataRetrieval requires a non-empty field.")
    return normalized


def _metadata_values_for_match(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [str(item).strip() for item in value]
    return [str(value).strip()]


def _metadata_value_matches(*, candidate_values: list[str], target: str, match_mode: str, regex: re.Pattern[str] | None) -> bool:
    if match_mode == "exact":
        normalized_target = target.casefold()
        return any(value.casefold() == normalized_target for value in candidate_values)
    if regex is None:
        raise ValueError("regex pattern must be compiled when match_mode='regex'.")
    return any(bool(regex.search(value)) for value in candidate_values)


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


def _normalize_parent_id_fields(parent_id_fields: Iterable[str]) -> tuple[str, ...]:
    raw_fields = (parent_id_fields,) if isinstance(parent_id_fields, str) else parent_id_fields
    normalized = tuple(str(field).strip() for field in raw_fields if str(field).strip())
    if not normalized:
        raise ValueError("ParentEpisodeExpansionRetrieval requires at least one parent_id_field.")
    return normalized


def _score_dict_at(scores: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index >= len(scores):
        return {}
    raw_score = scores[index]
    return dict(raw_score) if isinstance(raw_score, Mapping) else {}


def _numeric_score_value(score: Mapping[str, Any]) -> float | None:
    value = score.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _explicit_record_id_value(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return str(value)


def _explicit_parent_id_from_record(record, parent_id_fields: tuple[str, ...]) -> tuple[str | None, str | None, str | None]:
    metadata = record.metadata if isinstance(getattr(record, "metadata", None), dict) else {}
    for field in parent_id_fields:
        parent_id = _explicit_record_id_value(metadata.get(field))
        if parent_id is not None:
            return parent_id, field, "metadata"

    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        for field in parent_id_fields:
            parent_id = _explicit_record_id_value(provenance.get(field))
            if parent_id is not None:
                return parent_id, field, "provenance"

    return None, None, None


def _normalize_scope_fields(scope_fields: Iterable[str] | None) -> tuple[str, ...]:
    if scope_fields is None:
        return ()
    raw_fields = (scope_fields,) if isinstance(scope_fields, str) else scope_fields
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_field in raw_fields:
        field = str(raw_field).strip()
        if not field or field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    return tuple(normalized)


def _record_metadata(record) -> dict[str, Any]:
    metadata = getattr(record, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _timestamp_sort_value(record) -> datetime | None:
    raw_timestamp = getattr(record, "timestamp", None)
    if not isinstance(raw_timestamp, str):
        return None
    timestamp = raw_timestamp.strip()
    if not timestamp:
        return None
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ordered_temporal_records(records: list[Any]) -> tuple[list[Any], str]:
    timestamp_keys: list[tuple[datetime, str, int, Any]] = []
    for index, record in enumerate(records):
        timestamp = _timestamp_sort_value(record)
        if timestamp is None:
            return list(records), "store_iteration"
        timestamp_keys.append((timestamp, str(getattr(record, "record_id", "")), index, record))
    timestamp_keys.sort(key=lambda item: (item[0], item[1], item[2]))
    return [record for _, _, _, record in timestamp_keys], "timestamp_record_id"


def _temporal_scope_constraints(record, scope_fields: tuple[str, ...]) -> dict[str, Any]:
    metadata = _record_metadata(record)
    return {field: metadata[field] for field in scope_fields if field in metadata}


def _record_matches_temporal_scope(record, constraints: Mapping[str, Any]) -> bool:
    metadata = _record_metadata(record)
    return all(field in metadata and metadata[field] == value for field, value in constraints.items())


def _record_id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        raw_values = value
    else:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        record_id = _explicit_record_id_value(raw_value)
        if record_id is None or record_id in seen:
            continue
        seen.add(record_id)
        normalized.append(record_id)
    return normalized


def _append_unique(values: list[str], value: str | None) -> None:
    if value is not None and value not in values:
        values.append(value)


def _chronological_record_key(record) -> tuple[bool, datetime, str]:
    timestamp = _timestamp_sort_value(record)
    return (
        timestamp is None,
        timestamp or datetime.max.replace(tzinfo=timezone.utc),
        str(getattr(record, "record_id", "")),
    )


def _score_by_record_id(retrieved: RetrievedSet) -> dict[str, dict[str, Any]]:
    scores_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(retrieved.items):
        score = _score_dict_at(retrieved.scores, index)
        record_id = _explicit_record_id_value(score.get("record_id")) or getattr(record, "record_id", None)
        if isinstance(record_id, str) and record_id not in scores_by_id:
            scores_by_id[record_id] = score
    for raw_score in retrieved.scores:
        score = dict(raw_score) if isinstance(raw_score, Mapping) else {}
        record_id = _explicit_record_id_value(score.get("record_id"))
        if record_id is not None and record_id not in scores_by_id:
            scores_by_id[record_id] = score
    return scores_by_id


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


class MetadataRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Filter records by a metadata field using exact or regex matching."""

    spec = ModuleSpec(
        name="metadata_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        top_k: int = 3,
        field: str = "",
        target: str = "",
        match_mode: str = "exact",
        layer: str | None = None,
        *,
        source: str = "store",
    ) -> None:
        if top_k <= 0:
            raise ValueError("MetadataRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.field = _normalize_metadata_field(field)
        self.target = str(target).strip()
        self.match_mode = _normalize_metadata_match_mode(match_mode)
        self.layer = layer
        self.source = _normalize_retrieval_source(source)
        self._regex = re.compile(self.target, re.IGNORECASE) if self.match_mode == "regex" else None

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("MetadataRetrieval requires packet.query.")

        all_records = list(reversed(_candidate_records(packet, store, source=self.source, layer=self.layer)))
        matched_records: list[Any] = []
        scores: list[dict[str, Any]] = []

        for record in all_records:
            metadata = record.metadata if isinstance(record.metadata, dict) else {}
            if self.field not in metadata:
                continue
            candidate_values = _metadata_values_for_match(metadata.get(self.field))
            if not _metadata_value_matches(
                candidate_values=candidate_values,
                target=self.target,
                match_mode=self.match_mode,
                regex=self._regex,
            ):
                continue
            matched_records.append(record)

        selected_records = matched_records[: self.top_k]
        for rank, record in enumerate(selected_records, start=1):
            scores.append(
                {
                    "record_id": record.record_id,
                    "rank": rank,
                    "strategy": f"metadata_{self.match_mode}",
                    "field": self.field,
                }
            )

        retrieved = RetrievedSet(
            items=selected_records,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "source": self.source,
                "field": self.field,
                "target": self.target,
                "match_mode": self.match_mode,
                "layer": self.layer,
                "candidate_count": len(all_records),
                "matched_count": len(matched_records),
                "returned_count": len(selected_records),
            },
        )
        return _with_retrieved(packet, retrieved), store


class EmbeddingSimilarityRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Retrieve the ``top_k`` records with highest embedding cosine similarity.

    Constructor: ``top_k`` must be a positive integer. ``layer`` selects
    ``store.iter_records(layer)``; ``None`` means all layers. Embedding is
    delegated to ``Runtime.embed()`` so the runtime embedding provider controls
    whether query text is embedded locally or through an API.

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

    def __init__(
        self,
        top_k: int = 3,
        layer: str | None = None,
        embedding_model: str | None = None,
        source: str = "store",
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("EmbeddingSimilarityRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer
        self.embedding_model = embedding_model
        self.source = _normalize_retrieval_source(source)
        self.embedding_provider = embedding_provider
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url

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
        from ..utils._runtime import Runtime

        return Runtime(
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_api_key=self.embedding_api_key,
            embedding_base_url=self.embedding_base_url,
        ).embed(text)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


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


class TripleMemoryRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Retrieve triple-bearing records by exact slot match with fuzzy fallback.

    Queries use the structured format ``subject >> relation >> object`` or
    ``Query.metadata["triple_query"]`` with ``subject`` / ``relation`` /
    ``object`` fields. Empty strings or ``*`` act as wildcards.

    The retriever first searches for exact slot matches across any grounded slot
    combination. When no exact result exists, it falls back to vector-similarity
    matching over grounded query terms and keeps only candidate triples whose
    per-slot similarity clears ``candidate_similarity_threshold`` and whose
    average grounded-slot similarity clears ``final_similarity_threshold``.
    """

    spec = ModuleSpec(
        name="triple_memory_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )
    _term_embedding_cache: ClassVar[dict[tuple[str | None, str | None, str | None, str], list[float]]] = {}

    def __init__(
        self,
        top_k: int = 3,
        layer: str | None = None,
        *,
        source: str = "store",
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        candidate_similarity_threshold: float = 0.7,
        final_similarity_threshold: float = 0.85,
    ) -> None:
        if top_k <= 0:
            raise ValueError("TripleMemoryRetrieval requires top_k > 0.")
        if not 0.0 <= candidate_similarity_threshold <= 1.0:
            raise ValueError("candidate_similarity_threshold must be in [0, 1].")
        if not 0.0 <= final_similarity_threshold <= 1.0:
            raise ValueError("final_similarity_threshold must be in [0, 1].")
        self.top_k = top_k
        self.layer = layer
        self.source = _normalize_retrieval_source(source)
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.candidate_similarity_threshold = float(candidate_similarity_threshold)
        self.final_similarity_threshold = float(final_similarity_threshold)

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("TripleMemoryRetrieval requires packet.query.")

        query_spec, query_source = _parse_triple_query(packet.query)
        all_records = list(reversed(_candidate_records(packet, store, source=self.source, layer=self.layer)))
        query_triple = {
            "subject": query_spec["subject"],
            "relation": query_spec["relation"],
            "object": query_spec["object"],
        }
        if not all_records:
            packet = _empty_retrieved(
                packet,
                module_name=self.spec.name,
                top_k=self.top_k,
                source=self.source,
                query_mode=query_spec["mode"],
                query_source=query_source,
                query_triple=query_triple,
                candidate_count=0,
                triple_candidate_count=0,
                matched_candidate_count=0,
                retrieval_mode="exact",
                fallback_used=False,
                candidate_similarity_threshold=self.candidate_similarity_threshold,
                final_similarity_threshold=self.final_similarity_threshold,
            )
            return packet, store

        exact_scored: list[tuple[int, int, Any, list[tuple[str, str, str]]]] = []
        triple_candidate_count = 0
        for order_index, record in enumerate(all_records):
            record_triples = _record_triples(record)
            if record_triples:
                triple_candidate_count += 1
            matched_triples = [
                triple
                for triple in record_triples
                if _triple_matches_query(triple, query_spec=query_spec)
            ]
            if matched_triples:
                exact_scored.append((len(matched_triples), order_index, record, matched_triples))

        if exact_scored:
            exact_scored.sort(key=lambda item: (-item[0], item[1]))
            selected = exact_scored[: self.top_k]
            items = [record for _, _, record, _ in selected]
            scores = [
                {
                    "record_id": record.record_id,
                    "rank": rank,
                    "score": float(match_count),
                    "strategy": "triple_memory_exact",
                    "retrieval_mode": "exact",
                    "matched_triples": list(matched_triples),
                }
                for rank, (match_count, _, record, matched_triples) in enumerate(selected, start=1)
            ]
            retrieved = RetrievedSet(
                items=items,
                scores=scores,
                trace={
                    "module": self.spec.name,
                    "top_k": self.top_k,
                    "source": self.source,
                    "query_mode": query_spec["mode"],
                    "query_source": query_source,
                    "query_triple": query_triple,
                    "candidate_count": len(all_records),
                    "triple_candidate_count": triple_candidate_count,
                    "matched_candidate_count": len(exact_scored),
                    "retrieval_mode": "exact",
                    "fallback_used": False,
                    "candidate_similarity_threshold": self.candidate_similarity_threshold,
                    "final_similarity_threshold": self.final_similarity_threshold,
                },
            )
            return _with_retrieved(packet, retrieved), store

        fuzzy_scored: list[tuple[float, int, int, Any, list[dict[str, Any]]]] = []
        for order_index, record in enumerate(all_records):
            fuzzy_matches: list[dict[str, Any]] = []
            for triple in _record_triples(record):
                match = self._fuzzy_match(triple, query_spec=query_spec)
                if match is None:
                    continue
                fuzzy_matches.append(match)
            if not fuzzy_matches:
                continue
            best_score = max(float(match["score"]) for match in fuzzy_matches)
            fuzzy_scored.append((best_score, len(fuzzy_matches), order_index, record, fuzzy_matches))

        if not fuzzy_scored:
            packet = _empty_retrieved(
                packet,
                module_name=self.spec.name,
                top_k=self.top_k,
                source=self.source,
                query_mode=query_spec["mode"],
                query_source=query_source,
                query_triple=query_triple,
                candidate_count=len(all_records),
                triple_candidate_count=triple_candidate_count,
                matched_candidate_count=0,
                retrieval_mode="fuzzy",
                fallback_used=True,
                candidate_similarity_threshold=self.candidate_similarity_threshold,
                final_similarity_threshold=self.final_similarity_threshold,
            )
            return packet, store

        fuzzy_scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = fuzzy_scored[: self.top_k]
        items = [record for _, _, _, record, _ in selected]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": best_score,
                "strategy": "triple_memory_fuzzy",
                "retrieval_mode": "fuzzy",
                "matched_triples": [tuple(match["triple"]) for match in fuzzy_matches],
                "matched_triple_scores": [
                    {
                        "triple": tuple(match["triple"]),
                        "score": float(match["score"]),
                        "slot_similarities": dict(match["slot_similarities"]),
                    }
                    for match in fuzzy_matches
                ],
            }
            for rank, (best_score, _, _, record, fuzzy_matches) in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "source": self.source,
                "query_mode": query_spec["mode"],
                "query_source": query_source,
                "query_triple": query_triple,
                "candidate_count": len(all_records),
                "triple_candidate_count": triple_candidate_count,
                "matched_candidate_count": len(fuzzy_scored),
                "retrieval_mode": "fuzzy",
                "fallback_used": True,
                "candidate_similarity_threshold": self.candidate_similarity_threshold,
                "final_similarity_threshold": self.final_similarity_threshold,
            },
        )
        return _with_retrieved(packet, retrieved), store

    def _fuzzy_match(self, triple: tuple[str, str, str], *, query_spec: dict[str, Any]) -> dict[str, Any] | None:
        slot_similarities: dict[str, float] = {}
        grounded_slots = list(query_spec.get("grounded_slots", []))
        if not grounded_slots:
            return None
        for slot in grounded_slots:
            query_value = str(query_spec[slot])
            triple_value = _triple_slot_value(triple, slot)
            similarity = 1.0
            if triple_value.casefold() != query_value.casefold():
                similarity = self._cosine_similarity(self._embed_text(query_value), self._embed_text(triple_value))
            if similarity < self.candidate_similarity_threshold:
                return None
            slot_similarities[slot] = float(similarity)
        score = sum(slot_similarities.values()) / len(slot_similarities)
        if score < self.final_similarity_threshold:
            return None
        return {
            "triple": list(triple),
            "score": float(score),
            "slot_similarities": slot_similarities,
        }

    def _embed_text(self, text: str) -> list[float]:
        normalized = str(text).strip().casefold()
        key = (
            self.embedding_provider,
            self.embedding_base_url,
            self.embedding_model,
            normalized,
        )
        cached = self._term_embedding_cache.get(key)
        if cached is not None:
            return cached
        from ..utils._runtime import Runtime

        embedding = Runtime(
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_api_key=self.embedding_api_key,
            embedding_base_url=self.embedding_base_url,
        ).embed(normalized)
        self._term_embedding_cache[key] = embedding
        return embedding

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


TripleExactMatchRetrieval = TripleMemoryRetrieval


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


class RerankerRetrieval(_MultiQueryRetrievalMixin, RetrievalModule):
    """Rerank candidate records through the dedicated runtime reranker."""

    spec = ModuleSpec(
        name="reranker_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        top_k: int | None = None,
        layer: str | None = None,
        *,
        source: str = "retrieved",
        task: str = "Rerank memory records for retrieval relevance.",
    ) -> None:
        if top_k is not None and top_k <= 0:
            raise ValueError("RerankerRetrieval requires top_k > 0 when provided.")
        self.top_k = top_k
        self.layer = layer
        self.source = _normalize_retrieval_source(source)
        self.task = str(task)

    def _run_single_query(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("RerankerRetrieval requires packet.query.")

        candidates = _candidate_records(packet, store, source=self.source, layer=self.layer)
        effective_top_k = len(candidates) if self.top_k is None else min(self.top_k, len(candidates))
        if not candidates:
            retrieved = RetrievedSet(
                items=[],
                scores=[],
                trace={
                    "module": self.spec.name,
                    "top_k": self.top_k,
                    "effective_top_k": 0,
                    "source": self.source,
                    "layer": self.layer,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "candidate_record_ids": [],
                    "returned_ids": [],
                    "ignored_result_count": 0,
                    "missing_candidate_count": 0,
                    "missing_or_ignored_count": 0,
                },
            )
            return _with_retrieved(packet, retrieved), store

        payload = [{"id": record.record_id, "content": record.text} for record in candidates]
        from ..utils._runtime import get_runtime

        reranked = get_runtime().rerank(
            query=packet.query.text,
            candidates=payload,
            task=self.task,
            top_k=effective_top_k,
        )
        record_by_id = {record.record_id: record for record in candidates}
        selected_records = []
        scores: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        ignored_result_count = 0
        for item in reranked:
            if not isinstance(item, Mapping):
                ignored_result_count += 1
                continue
            record_id = _explicit_record_id_value(item.get("id"))
            if record_id is None or record_id in used_ids or record_id not in record_by_id:
                ignored_result_count += 1
                continue
            record = record_by_id[record_id]
            used_ids.add(record_id)
            selected_records.append(record)
            raw_score = item.get("score", 0.0)
            score = raw_score if not isinstance(raw_score, bool) and isinstance(raw_score, (int, float)) else 0.0
            scores.append(
                {
                    "record_id": record_id,
                    "rank": len(selected_records),
                    "score": float(score),
                    "rationale": str(item.get("rationale", "")).strip(),
                    "strategy": "runtime_rerank",
                }
            )
            if len(selected_records) >= effective_top_k:
                break

        missing_candidate_count = max(0, effective_top_k - len(selected_records))
        retrieved = RetrievedSet(
            items=selected_records,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "effective_top_k": effective_top_k,
                "source": self.source,
                "layer": self.layer,
                "candidate_count": len(candidates),
                "selected_count": len(selected_records),
                "candidate_record_ids": [record.record_id for record in candidates],
                "returned_ids": [record.record_id for record in selected_records],
                "ignored_result_count": ignored_result_count,
                "missing_candidate_count": missing_candidate_count,
                "missing_or_ignored_count": missing_candidate_count + ignored_result_count,
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


class ParentEpisodeExpansionRetrieval(RetrievalModule):
    """Expand sentence/derivative retrieval hits to explicit parent episodes."""

    DEFAULT_PARENT_ID_FIELDS: ClassVar[tuple[str, ...]] = (
        "parent_episode_record_id",
        "episode_record_id",
        "source_episode_record_id",
    )

    spec = ModuleSpec(
        name="parent_episode_expansion_retrieval",
        slot="retrieval",
        input_requirements=("retrieved.items", "record.metadata.parent_episode_record_id"),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        top_k: int = 3,
        *,
        episode_layer: str = "episodic",
        parent_id_fields: Iterable[str] = DEFAULT_PARENT_ID_FIELDS,
        source: str = "retrieved",
    ) -> None:
        if top_k <= 0:
            raise ValueError("ParentEpisodeExpansionRetrieval requires top_k > 0.")
        normalized_episode_layer = str(episode_layer).strip()
        if not normalized_episode_layer:
            raise ValueError("ParentEpisodeExpansionRetrieval requires a non-empty episode_layer.")
        normalized_source = _normalize_retrieval_source(source)
        if normalized_source != "retrieved":
            raise ValueError("ParentEpisodeExpansionRetrieval only supports source='retrieved'.")
        self.top_k = top_k
        self.episode_layer = normalized_episode_layer
        self.parent_id_fields = _normalize_parent_id_fields(parent_id_fields)
        self.source = normalized_source

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        retrieved_input = packet.retrieved
        hit_records = [] if retrieved_input is None else list(retrieved_input.items)
        input_hit_ids = [getattr(record, "record_id", None) for record in hit_records]
        if not hit_records:
            packet = _empty_retrieved(
                packet,
                module_name=self.spec.name,
                top_k=self.top_k,
                source=self.source,
                episode_layer=self.episode_layer,
                parent_id_fields=list(self.parent_id_fields),
                candidate_count=0,
                input_hit_ids=input_hit_ids,
                successful_parent_ids=[],
                missing_parent_count=0,
                hit_to_parent=[],
                duplicate_parent_count=0,
                unresolved_parent_count=0,
                returned_count=0,
            )
            return packet, store

        parent_records_by_id: dict[str, Any] = {}
        for record in store.iter_records(self.episode_layer):
            parent_records_by_id.setdefault(record.record_id, record)

        entries_by_parent_id: dict[str, dict[str, Any]] = {}
        ordered_parent_ids: list[str] = []
        successful_parent_ids: list[str] = []
        seen_successful_parent_ids: set[str] = set()
        hit_to_parent: list[dict[str, Any]] = []
        missing_parent_count = 0
        duplicate_parent_count = 0
        unresolved_parent_count = 0

        input_scores = retrieved_input.scores if retrieved_input is not None else []
        for hit_index, hit_record in enumerate(hit_records):
            hit_record_id = getattr(hit_record, "record_id", None)
            source_score = _score_dict_at(input_scores, hit_index)
            parent_id, parent_id_field, parent_id_source = _explicit_parent_id_from_record(
                hit_record,
                self.parent_id_fields,
            )
            if parent_id is None:
                missing_parent_count += 1
                hit_to_parent.append(
                    {
                        "hit_record_id": hit_record_id,
                        "parent_record_id": None,
                        "parent_id_field": None,
                        "parent_id_source": None,
                        "status": "missing_parent_id",
                    }
                )
                continue

            parent_record = parent_records_by_id.get(parent_id)
            if parent_record is None:
                unresolved_parent_count += 1
                hit_to_parent.append(
                    {
                        "hit_record_id": hit_record_id,
                        "parent_record_id": parent_id,
                        "parent_id_field": parent_id_field,
                        "parent_id_source": parent_id_source,
                        "status": "parent_not_found",
                    }
                )
                continue

            hit_to_parent.append(
                {
                    "hit_record_id": hit_record_id,
                    "parent_record_id": parent_id,
                    "parent_id_field": parent_id_field,
                    "parent_id_source": parent_id_source,
                    "status": "resolved",
                }
            )
            if parent_id not in seen_successful_parent_ids:
                seen_successful_parent_ids.add(parent_id)
                successful_parent_ids.append(parent_id)

            metadata = hit_record.metadata if isinstance(getattr(hit_record, "metadata", None), dict) else {}
            hit_provenance = metadata.get("provenance")
            copied_hit_provenance = dict(hit_provenance) if isinstance(hit_provenance, dict) else None
            numeric_score = _numeric_score_value(source_score)

            entry = entries_by_parent_id.get(parent_id)
            if entry is None:
                entries_by_parent_id[parent_id] = {
                    "record": parent_record,
                    "first_hit_record_id": hit_record_id,
                    "best_score": source_score,
                    "best_numeric_score": numeric_score,
                    "source_hit_record_id": hit_record_id,
                    "parent_id_field": parent_id_field,
                    "parent_id_source": parent_id_source,
                    "hit_provenance": copied_hit_provenance,
                }
                ordered_parent_ids.append(parent_id)
                continue

            duplicate_parent_count += 1
            best_numeric_score = entry["best_numeric_score"]
            if numeric_score is not None and (best_numeric_score is None or numeric_score > best_numeric_score):
                entry["best_score"] = source_score
                entry["best_numeric_score"] = numeric_score
                entry["source_hit_record_id"] = hit_record_id
                entry["parent_id_field"] = parent_id_field
                entry["parent_id_source"] = parent_id_source
                entry["hit_provenance"] = copied_hit_provenance

        selected_parent_ids = ordered_parent_ids[: self.top_k]
        items = [entries_by_parent_id[parent_id]["record"] for parent_id in selected_parent_ids]
        scores: list[dict[str, Any]] = []
        for rank, parent_id in enumerate(selected_parent_ids, start=1):
            entry = entries_by_parent_id[parent_id]
            score_info: dict[str, Any] = {
                "record_id": parent_id,
                "rank": rank,
                "strategy": "parent_episode_expansion",
                "source_score": dict(entry["best_score"]),
                "source_hit_record_id": entry["source_hit_record_id"],
                "first_hit_record_id": entry["first_hit_record_id"],
                "parent_id_field": entry["parent_id_field"],
                "parent_id_source": entry["parent_id_source"],
            }
            numeric_score = _numeric_score_value(score_info["source_score"])
            if numeric_score is not None:
                score_info["score"] = numeric_score
            if entry["hit_provenance"] is not None:
                score_info["hit_provenance"] = dict(entry["hit_provenance"])
            scores.append(score_info)

        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "source": self.source,
                "episode_layer": self.episode_layer,
                "parent_id_fields": list(self.parent_id_fields),
                "candidate_count": len(hit_records),
                "input_hit_ids": input_hit_ids,
                "successful_parent_ids": successful_parent_ids,
                "missing_parent_count": missing_parent_count,
                "hit_to_parent": hit_to_parent,
                "duplicate_parent_count": duplicate_parent_count,
                "unresolved_parent_count": unresolved_parent_count,
                "returned_count": len(items),
            },
        )
        return _with_retrieved(packet, retrieved), store


class TemporalNeighborExpansionRetrieval(RetrievalModule):
    """Expand nucleus episodes to bounded temporal neighbors within the same scope."""

    DEFAULT_SCOPE_FIELDS: ClassVar[tuple[str, ...]] = ("session_id", "user_id", "agent_id")

    spec = ModuleSpec(
        name="temporal_neighbor_expansion_retrieval",
        slot="retrieval",
        input_requirements=("retrieved.items",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        *,
        layer: str = "episodic",
        backward: int = 1,
        forward: int = 2,
        scope_fields: Iterable[str] | None = DEFAULT_SCOPE_FIELDS,
        chronological: bool = True,
    ) -> None:
        normalized_layer = str(layer).strip()
        if not normalized_layer:
            raise ValueError("TemporalNeighborExpansionRetrieval requires a non-empty layer.")
        if backward < 0:
            raise ValueError("TemporalNeighborExpansionRetrieval requires backward >= 0.")
        if forward < 0:
            raise ValueError("TemporalNeighborExpansionRetrieval requires forward >= 0.")
        self.layer = normalized_layer
        self.target_layer = normalized_layer
        self.backward = int(backward)
        self.forward = int(forward)
        self.scope_fields = _normalize_scope_fields(scope_fields)
        self.chronological = bool(chronological)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        retrieved_input = packet.retrieved if packet.retrieved is not None else RetrievedSet()
        input_records = list(retrieved_input.items)
        input_record_ids = [getattr(record, "record_id", None) for record in input_records]

        nucleus_records: list[Any] = []
        seen_nucleus_ids: set[str] = set()
        skipped_non_layer_input_ids: list[str] = []
        duplicate_nucleus_count = 0
        for record in input_records:
            record_id = getattr(record, "record_id", None)
            if getattr(record, "layer", None) != self.layer:
                if isinstance(record_id, str):
                    skipped_non_layer_input_ids.append(record_id)
                continue
            if not isinstance(record_id, str):
                continue
            if record_id in seen_nucleus_ids:
                duplicate_nucleus_count += 1
                continue
            seen_nucleus_ids.add(record_id)
            nucleus_records.append(record)

        if not nucleus_records:
            retrieved = RetrievedSet(
                items=[],
                scores=[],
                trace={
                    "module": self.spec.name,
                    "source": "retrieved",
                    "layer": self.layer,
                    "backward": self.backward,
                    "forward": self.forward,
                    "scope_fields": list(self.scope_fields),
                    "chronological": self.chronological,
                    "ordering_mode": "none",
                    "candidate_count": 0,
                    "input_record_ids": input_record_ids,
                    "nucleus_record_ids": [],
                    "skipped_non_layer_input_ids": skipped_non_layer_input_ids,
                    "duplicate_nucleus_count": duplicate_nucleus_count,
                    "unresolved_nucleus_ids": [],
                    "clusters": [],
                    "returned_ids": [],
                    "returned_count": 0,
                    "total_cluster_candidate_count": 0,
                    "deduped_duplicate_count": 0,
                },
            )
            return _with_retrieved(packet, retrieved), store

        layer_records = store.iter_records(self.layer)
        ordered_records, ordering_mode = _ordered_temporal_records(layer_records)
        order_index_by_id = {
            record.record_id: index
            for index, record in enumerate(ordered_records)
            if isinstance(getattr(record, "record_id", None), str)
        }

        entries_by_id: dict[str, dict[str, Any]] = {}
        discovery_order: list[str] = []
        clusters: list[dict[str, Any]] = []
        unresolved_nucleus_ids: list[str] = []
        total_cluster_candidate_count = 0
        deduped_duplicate_count = 0

        def add_entry(record, *, nucleus_record_id: str, role: str) -> None:
            nonlocal total_cluster_candidate_count, deduped_duplicate_count
            total_cluster_candidate_count += 1
            record_id = record.record_id
            entry = entries_by_id.get(record_id)
            if entry is None:
                entries_by_id[record_id] = {
                    "record": record,
                    "source_nucleus_record_ids": [nucleus_record_id],
                    "roles": [role],
                }
                discovery_order.append(record_id)
                return
            deduped_duplicate_count += 1
            if nucleus_record_id not in entry["source_nucleus_record_ids"]:
                entry["source_nucleus_record_ids"].append(nucleus_record_id)
            if role not in entry["roles"]:
                entry["roles"].append(role)

        for nucleus_record in nucleus_records:
            nucleus_record_id = nucleus_record.record_id
            if nucleus_record_id not in order_index_by_id:
                unresolved_nucleus_ids.append(nucleus_record_id)
                clusters.append(
                    {
                        "nucleus_record_id": nucleus_record_id,
                        "cluster_ids": [],
                        "backward_ids": [],
                        "forward_ids": [],
                        "scope": _temporal_scope_constraints(nucleus_record, self.scope_fields),
                        "status": "nucleus_not_found",
                    }
                )
                continue

            scope_constraints = _temporal_scope_constraints(nucleus_record, self.scope_fields)
            scoped_records = [
                record for record in ordered_records if _record_matches_temporal_scope(record, scope_constraints)
            ]
            scoped_index_by_id = {
                record.record_id: index
                for index, record in enumerate(scoped_records)
                if isinstance(getattr(record, "record_id", None), str)
            }
            scoped_index = scoped_index_by_id.get(nucleus_record_id)
            if scoped_index is None:
                unresolved_nucleus_ids.append(nucleus_record_id)
                clusters.append(
                    {
                        "nucleus_record_id": nucleus_record_id,
                        "cluster_ids": [],
                        "backward_ids": [],
                        "forward_ids": [],
                        "scope": dict(scope_constraints),
                        "status": "nucleus_out_of_scope",
                    }
                )
                continue

            backward_records = scoped_records[max(0, scoped_index - self.backward) : scoped_index]
            nucleus_store_record = scoped_records[scoped_index]
            forward_records = scoped_records[scoped_index + 1 : scoped_index + 1 + self.forward]
            cluster_records = [*backward_records, nucleus_store_record, *forward_records]

            for record in backward_records:
                add_entry(record, nucleus_record_id=nucleus_record_id, role="backward")
            add_entry(nucleus_store_record, nucleus_record_id=nucleus_record_id, role="nucleus")
            for record in forward_records:
                add_entry(record, nucleus_record_id=nucleus_record_id, role="forward")

            clusters.append(
                {
                    "nucleus_record_id": nucleus_record_id,
                    "cluster_ids": [record.record_id for record in cluster_records],
                    "backward_ids": [record.record_id for record in backward_records],
                    "forward_ids": [record.record_id for record in forward_records],
                    "scope": dict(scope_constraints),
                    "status": "resolved",
                }
            )

        if self.chronological:
            selected_ids = sorted(discovery_order, key=lambda record_id: order_index_by_id.get(record_id, 10**12))
        else:
            selected_ids = list(discovery_order)
        items = [entries_by_id[record_id]["record"] for record_id in selected_ids]
        scores = [
            {
                "record_id": record_id,
                "rank": rank,
                "strategy": "temporal_neighbor_expansion",
                "source_nucleus_record_ids": list(entries_by_id[record_id]["source_nucleus_record_ids"]),
                "roles": list(entries_by_id[record_id]["roles"]),
                "is_nucleus": "nucleus" in entries_by_id[record_id]["roles"],
                "order_index": order_index_by_id.get(record_id),
            }
            for rank, record_id in enumerate(selected_ids, start=1)
        ]

        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "source": "retrieved",
                "layer": self.layer,
                "backward": self.backward,
                "forward": self.forward,
                "scope_fields": list(self.scope_fields),
                "chronological": self.chronological,
                "ordering_mode": ordering_mode,
                "candidate_count": len(layer_records),
                "input_record_ids": input_record_ids,
                "nucleus_record_ids": [record.record_id for record in nucleus_records],
                "skipped_non_layer_input_ids": skipped_non_layer_input_ids,
                "duplicate_nucleus_count": duplicate_nucleus_count,
                "unresolved_nucleus_ids": unresolved_nucleus_ids,
                "clusters": clusters,
                "returned_ids": selected_ids,
                "returned_count": len(items),
                "total_cluster_candidate_count": total_cluster_candidate_count,
                "deduped_duplicate_count": deduped_duplicate_count,
            },
        )
        return _with_retrieved(packet, retrieved), store


class EpisodeClusterRerankRetrieval(RetrievalModule):
    """Rerank temporal episode clusters, dedupe them under budget, and return episodes."""

    spec = ModuleSpec(
        name="episode_cluster_rerank_retrieval",
        slot="retrieval",
        input_requirements=("query.text", "retrieved.items", "retrieved.trace.clusters"),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        top_k: int = 20,
        *,
        cluster_top_k: int | None = None,
        rerank: bool = True,
        chronological: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("EpisodeClusterRerankRetrieval requires top_k > 0.")
        if cluster_top_k is not None and cluster_top_k <= 0:
            raise ValueError("EpisodeClusterRerankRetrieval requires cluster_top_k > 0 when provided.")
        self.top_k = int(top_k)
        self.cluster_top_k = None if cluster_top_k is None else int(cluster_top_k)
        self.rerank = bool(rerank)
        self.chronological = bool(chronological)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        retrieved_input = packet.retrieved
        raw_clusters = self._raw_clusters(retrieved_input)
        input_cluster_count = len(raw_clusters)
        if retrieved_input is None:
            return self._empty_result(packet, store, empty_reason="missing_retrieved", input_cluster_count=0)
        if packet.query is None or not getattr(packet.query, "text", None):
            return self._empty_result(
                packet,
                store,
                empty_reason="missing_query",
                input_cluster_count=input_cluster_count,
            )
        if not raw_clusters:
            return self._empty_result(
                packet,
                store,
                empty_reason="no_clusters",
                input_cluster_count=input_cluster_count,
            )

        record_by_id = self._records_by_id(retrieved_input, store)
        scores_by_id = _score_by_record_id(retrieved_input)
        resolved_clusters = self._resolve_clusters(raw_clusters, record_by_id, scores_by_id)
        unresolved_cluster_count = input_cluster_count - len(resolved_clusters)
        if not resolved_clusters:
            return self._empty_result(
                packet,
                store,
                empty_reason="no_resolved_clusters",
                input_cluster_count=input_cluster_count,
                unresolved_cluster_count=unresolved_cluster_count,
            )

        reranked_clusters = self._ordered_clusters(packet.query.text, resolved_clusters)
        selected_entries, selected_order, deduped_duplicate_count, budget_truncated_count = self._merge_clusters(
            reranked_clusters
        )

        returned_ids = self._returned_ids(selected_entries, selected_order)
        items = [selected_entries[record_id]["record"] for record_id in returned_ids]
        scores = [
            {
                "record_id": record_id,
                "rank": rank,
                "strategy": "episode_cluster_rerank",
                "source_nucleus_record_ids": list(selected_entries[record_id]["source_nucleus_record_ids"]),
                "roles": list(selected_entries[record_id]["roles"]),
                "cluster_score": float(selected_entries[record_id]["cluster_score"]),
            }
            for rank, record_id in enumerate(returned_ids, start=1)
        ]

        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "source": "retrieved",
                "top_k": self.top_k,
                "cluster_top_k": self.cluster_top_k,
                "rerank": self.rerank,
                "chronological": self.chronological,
                "input_cluster_count": input_cluster_count,
                "resolved_cluster_count": len(resolved_clusters),
                "unresolved_cluster_count": unresolved_cluster_count,
                "reranked_clusters": self._cluster_trace(reranked_clusters),
                "returned_ids": returned_ids,
                "returned_count": len(items),
                "deduped_duplicate_count": deduped_duplicate_count,
                "budget_truncated_count": budget_truncated_count,
            },
        )
        return _with_retrieved(packet, retrieved), store

    def _empty_result(
        self,
        packet: Packet,
        store: MemoryStore,
        *,
        empty_reason: str,
        input_cluster_count: int,
        unresolved_cluster_count: int = 0,
    ) -> tuple[Packet, MemoryStore]:
        retrieved = RetrievedSet(
            items=[],
            scores=[],
            trace={
                "module": self.spec.name,
                "source": "retrieved",
                "top_k": self.top_k,
                "cluster_top_k": self.cluster_top_k,
                "rerank": self.rerank,
                "chronological": self.chronological,
                "input_cluster_count": input_cluster_count,
                "resolved_cluster_count": 0,
                "unresolved_cluster_count": unresolved_cluster_count,
                "reranked_clusters": [],
                "returned_ids": [],
                "returned_count": 0,
                "deduped_duplicate_count": 0,
                "budget_truncated_count": 0,
                "empty_reason": empty_reason,
            },
        )
        return _with_retrieved(packet, retrieved), store

    @staticmethod
    def _raw_clusters(retrieved: RetrievedSet | None) -> list[Any]:
        if retrieved is None or not isinstance(retrieved.trace, Mapping):
            return []
        clusters = retrieved.trace.get("clusters")
        return list(clusters) if isinstance(clusters, list) else []

    @staticmethod
    def _records_by_id(retrieved: RetrievedSet, store: MemoryStore) -> dict[str, Any]:
        records_by_id: dict[str, Any] = {}
        for record in retrieved.items:
            record_id = getattr(record, "record_id", None)
            if isinstance(record_id, str):
                records_by_id.setdefault(record_id, record)

        source_layer = retrieved.trace.get("layer") if isinstance(retrieved.trace, Mapping) else None
        if isinstance(source_layer, str) and source_layer.strip() and store.has_layer(source_layer.strip()):
            fallback_records = store.iter_records(source_layer.strip())
        else:
            fallback_records = store.iter_records()
        for record in fallback_records:
            record_id = getattr(record, "record_id", None)
            if isinstance(record_id, str):
                records_by_id.setdefault(record_id, record)
        return records_by_id

    def _resolve_clusters(
        self,
        raw_clusters: list[Any],
        record_by_id: Mapping[str, Any],
        scores_by_id: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        nucleus_counts: dict[str, int] = {}
        for raw_cluster in raw_clusters:
            if not isinstance(raw_cluster, Mapping):
                continue
            nucleus_id = _explicit_record_id_value(raw_cluster.get("nucleus_record_id"))
            if nucleus_id is not None:
                nucleus_counts[nucleus_id] = nucleus_counts.get(nucleus_id, 0) + 1

        resolved_clusters: list[dict[str, Any]] = []
        for cluster_index, raw_cluster in enumerate(raw_clusters):
            if not isinstance(raw_cluster, Mapping):
                continue
            nucleus_id = _explicit_record_id_value(raw_cluster.get("nucleus_record_id"))
            backward_ids = _record_id_list(raw_cluster.get("backward_ids"))
            forward_ids = _record_id_list(raw_cluster.get("forward_ids"))
            cluster_ids = _record_id_list(raw_cluster.get("cluster_ids"))
            if not cluster_ids:
                cluster_ids = [record_id for record_id in [*backward_ids, nucleus_id, *forward_ids] if record_id]

            records: list[Any] = []
            roles_by_id: dict[str, list[str]] = {}
            unresolved_ids: list[str] = []
            seen_ids: set[str] = set()
            for record_id in cluster_ids:
                record = record_by_id.get(record_id)
                if record is None:
                    unresolved_ids.append(record_id)
                    continue
                role = self._role_for_cluster_id(
                    record_id,
                    nucleus_id=nucleus_id,
                    backward_ids=backward_ids,
                    forward_ids=forward_ids,
                )
                roles = roles_by_id.setdefault(record_id, [])
                _append_unique(roles, role)
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                records.append(record)

            if not records:
                continue

            source_score = dict(scores_by_id.get(nucleus_id, {})) if nucleus_id is not None else {}
            numeric_score = _numeric_score_value(source_score)
            source_rank = source_score.get("rank")
            rank_score = None
            if not isinstance(source_rank, bool) and isinstance(source_rank, (int, float)) and source_rank >= 0:
                rank_score = 1.0 / (1.0 + float(source_rank))
            fallback_score = numeric_score if numeric_score is not None else rank_score
            if fallback_score is None:
                fallback_score = 1.0 / (1.0 + float(cluster_index))
            if numeric_score is not None:
                score_source = "upstream_score"
            elif rank_score is not None:
                score_source = "upstream_rank"
            else:
                score_source = "input_order"

            candidate_id = self._candidate_id(
                cluster_index,
                nucleus_id=nucleus_id,
                nucleus_counts=nucleus_counts,
            )
            resolved_clusters.append(
                {
                    "input_index": cluster_index,
                    "candidate_id": candidate_id,
                    "nucleus_record_id": nucleus_id,
                    "cluster_ids": [record.record_id for record in records],
                    "backward_ids": backward_ids,
                    "forward_ids": forward_ids,
                    "records": records,
                    "roles_by_id": roles_by_id,
                    "unresolved_ids": unresolved_ids,
                    "fallback_score": float(fallback_score),
                    "cluster_score": float(fallback_score),
                    "rationale": "",
                    "score_source": score_source,
                }
            )
        return resolved_clusters

    @staticmethod
    def _candidate_id(cluster_index: int, *, nucleus_id: str | None, nucleus_counts: Mapping[str, int]) -> str:
        if nucleus_id is not None and nucleus_counts.get(nucleus_id, 0) == 1:
            return nucleus_id
        return f"cluster-{cluster_index}:{nucleus_id or 'unknown'}"

    @staticmethod
    def _role_for_cluster_id(
        record_id: str,
        *,
        nucleus_id: str | None,
        backward_ids: list[str],
        forward_ids: list[str],
    ) -> str:
        if nucleus_id is not None and record_id == nucleus_id:
            return "nucleus"
        if record_id in backward_ids:
            return "backward"
        if record_id in forward_ids:
            return "forward"
        return "context"

    def _ordered_clusters(self, query_text: str, resolved_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cluster_limit = min(self.cluster_top_k or len(resolved_clusters), len(resolved_clusters))
        if not self.rerank:
            clusters = [dict(cluster) for cluster in resolved_clusters[:cluster_limit]]
            for rank, cluster in enumerate(clusters, start=1):
                cluster["rank"] = rank
                cluster["score_source"] = cluster.get("score_source", "input_order")
            return clusters

        candidates = [
            {
                "id": cluster["candidate_id"],
                "content": self._cluster_content(cluster["records"]),
                "nucleus_record_id": cluster["nucleus_record_id"],
                "cluster_ids": list(cluster["cluster_ids"]),
                "score": float(cluster["fallback_score"]),
            }
            for cluster in resolved_clusters
        ]
        from ..utils._runtime import get_runtime

        reranked = get_runtime().rerank(
            query=query_text,
            candidates=candidates,
            task="Rerank contextualized episode clusters for episodic recall.",
            top_k=cluster_limit,
        )
        cluster_by_candidate_id = {cluster["candidate_id"]: cluster for cluster in resolved_clusters}
        ordered: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for rerank_index, item in enumerate(reranked):
            if not isinstance(item, Mapping):
                continue
            candidate_id = _explicit_record_id_value(item.get("id"))
            if candidate_id is None or candidate_id in seen_candidate_ids:
                continue
            cluster = cluster_by_candidate_id.get(candidate_id)
            if cluster is None:
                continue
            score = item.get("score", 0.0)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                score = 0.0
            ranked_cluster = dict(cluster)
            ranked_cluster["cluster_score"] = float(score)
            ranked_cluster["rationale"] = str(item.get("rationale", "")).strip()
            ranked_cluster["score_source"] = "runtime_rerank"
            ranked_cluster["rerank_index"] = rerank_index
            ordered.append(ranked_cluster)
            seen_candidate_ids.add(candidate_id)
            if len(ordered) >= cluster_limit:
                break

        for cluster in resolved_clusters:
            if len(ordered) >= cluster_limit:
                break
            candidate_id = cluster["candidate_id"]
            if candidate_id in seen_candidate_ids:
                continue
            ranked_cluster = dict(cluster)
            ranked_cluster["cluster_score"] = 0.0
            ranked_cluster["rationale"] = ""
            ranked_cluster["score_source"] = "rerank_missing_fallback"
            ranked_cluster["rerank_index"] = len(reranked) + len(ordered)
            ordered.append(ranked_cluster)
            seen_candidate_ids.add(candidate_id)

        ordered.sort(key=lambda cluster: (-float(cluster["cluster_score"]), int(cluster.get("rerank_index", 0))))
        for rank, cluster in enumerate(ordered, start=1):
            cluster["rank"] = rank
        return ordered

    @staticmethod
    def _cluster_content(records: list[Any]) -> str:
        return "\n".join(f"[{record.record_id}] {record.text}" for record in records)

    def _merge_clusters(
        self,
        clusters: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], list[str], int, int]:
        selected_entries: dict[str, dict[str, Any]] = {}
        selected_order: list[str] = []
        deduped_duplicate_count = 0
        budget_truncated_count = 0

        for cluster in clusters:
            if len(selected_order) >= self.top_k:
                break
            records = list(cluster["records"])
            new_records = [record for record in records if record.record_id not in selected_entries]
            remaining_budget = self.top_k - len(selected_order)
            if len(new_records) <= remaining_budget:
                for record in records:
                    added = self._add_or_update_selected(selected_entries, selected_order, record, cluster)
                    if not added:
                        deduped_duplicate_count += 1
                continue

            added_new_count = 0
            new_record_count = len(new_records)
            for record in self._partial_cluster_records(cluster):
                added = self._add_or_update_selected(selected_entries, selected_order, record, cluster)
                if added:
                    added_new_count += 1
                else:
                    deduped_duplicate_count += 1
                if len(selected_order) >= self.top_k:
                    break
            budget_truncated_count += max(0, new_record_count - added_new_count)

        return selected_entries, selected_order, deduped_duplicate_count, budget_truncated_count

    def _partial_cluster_records(self, cluster: Mapping[str, Any]) -> list[Any]:
        records = list(cluster["records"])
        nucleus_id = cluster.get("nucleus_record_id")
        nucleus_index = next(
            (index for index, record in enumerate(records) if record.record_id == nucleus_id),
            0,
        )
        indexed_records = list(enumerate(records))
        return [
            record
            for _, record in sorted(
                indexed_records,
                key=lambda item: (
                    abs(item[0] - nucleus_index),
                    0 if item[0] >= nucleus_index else 1,
                    abs(item[0] - nucleus_index),
                    item[0],
                ),
            )
        ]

    @staticmethod
    def _add_or_update_selected(
        selected_entries: dict[str, dict[str, Any]],
        selected_order: list[str],
        record,
        cluster: Mapping[str, Any],
    ) -> bool:
        record_id = record.record_id
        roles_by_id = cluster["roles_by_id"]
        nucleus_id = cluster.get("nucleus_record_id")
        entry = selected_entries.get(record_id)
        if entry is None:
            selected_entries[record_id] = {
                "record": record,
                "source_nucleus_record_ids": [nucleus_id] if isinstance(nucleus_id, str) else [],
                "roles": list(roles_by_id.get(record_id, ["context"])),
                "cluster_score": float(cluster["cluster_score"]),
            }
            selected_order.append(record_id)
            return True

        _append_unique(entry["source_nucleus_record_ids"], nucleus_id if isinstance(nucleus_id, str) else None)
        for role in roles_by_id.get(record_id, ["context"]):
            _append_unique(entry["roles"], role)
        entry["cluster_score"] = max(float(entry["cluster_score"]), float(cluster["cluster_score"]))
        return False

    def _returned_ids(self, selected_entries: Mapping[str, dict[str, Any]], selected_order: list[str]) -> list[str]:
        if not self.chronological:
            return list(selected_order)
        return sorted(
            selected_order,
            key=lambda record_id: _chronological_record_key(selected_entries[record_id]["record"]),
        )

    @staticmethod
    def _cluster_trace(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rank": int(cluster.get("rank", index)),
                "candidate_id": cluster["candidate_id"],
                "nucleus_record_id": cluster["nucleus_record_id"],
                "cluster_ids": list(cluster["cluster_ids"]),
                "score": float(cluster["cluster_score"]),
                "rationale": str(cluster.get("rationale", "")),
                "score_source": str(cluster.get("score_source", "")),
                "input_index": int(cluster["input_index"]),
                "unresolved_ids": list(cluster.get("unresolved_ids", [])),
            }
            for index, cluster in enumerate(clusters, start=1)
        ]


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
    MetadataRetrieval,
    EmbeddingSimilarityRetrieval,
    EntityRetrieval,
    TripleMemoryRetrieval,
    BM25Retrieval,
    RerankerRetrieval,
    GraphNeighborRetrieval,
    ParentEpisodeExpansionRetrieval,
    TemporalNeighborExpansionRetrieval,
    EpisodeClusterRerankRetrieval,
    ExpandRetrievedGraphNeighbors,
    VectorGraphSeedAndExpandRetrieval,
    LayerAwareRetrieval,
    BufferRetrieval,
    QueryRewriteRetrieval,
)
