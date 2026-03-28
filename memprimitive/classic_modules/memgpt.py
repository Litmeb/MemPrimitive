"""Paper-aligned MemGPT support primitives.

This module no longer exposes a single end-to-end "MemGPT pipeline". Instead it
provides the lower-level building blocks used by the example agent loop:

- five-layer store topology
- keyed upsert organization for core/working memory blocks
- paged retrieval for conversation and archival search tools
- JSON readout for tool-facing payloads
"""

from __future__ import annotations

from dataclasses import replace
import json
from math import sqrt
from typing import Any, Final

from ..utils._trace import copy_trace
from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet, Placement, Query, Readout, RetrievedSet, StoreLayerSpec, StoreTopology
from ..utils.exceptions import IncompatibleCompositionError
from ..interfaces import OrganizationModule, ReadoutModule, RetrievalModule
from ..utils._runtime import get_classic_runtime

MEMGPT_CORE_LAYER: Final[str] = "core_memory"
MEMGPT_WORKING_LAYER: Final[str] = "working_memory"
MEMGPT_QUEUE_LAYER: Final[str] = "conversation_queue"
MEMGPT_RECALL_LAYER: Final[str] = "recall_storage"
MEMGPT_ARCHIVAL_LAYER: Final[str] = "archival_memory"
MEMGPT_MAIN_LAYER: Final[str] = MEMGPT_QUEUE_LAYER

MEMGPT_CORE_BLOCK_PERSONA: Final[str] = "persona"
MEMGPT_CORE_BLOCK_HUMAN: Final[str] = "human"
MEMGPT_WORKING_SUMMARY_KEY: Final[str] = "working_summary"

MEMGPT_REQUIRED_CORE_BLOCKS: Final[tuple[str, ...]] = (
    MEMGPT_CORE_BLOCK_PERSONA,
    MEMGPT_CORE_BLOCK_HUMAN,
)
MEMGPT_LAYER_ORDER: Final[tuple[str, ...]] = (
    MEMGPT_CORE_LAYER,
    MEMGPT_WORKING_LAYER,
    MEMGPT_QUEUE_LAYER,
    MEMGPT_RECALL_LAYER,
    MEMGPT_ARCHIVAL_LAYER,
)

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "for",
        "with",
        "is",
        "are",
        "be",
        "this",
        "that",
        "it",
        "as",
        "at",
        "by",
        "from",
    }
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _clean_text(text).casefold().replace("\n", " ").split()
        if token and token not in _STOPWORDS
    ]


def _record_keywords(record: MemoryRecord) -> set[str]:
    tokens = set(_tokenize(record.text))
    representation = record.metadata.get("representation", {})
    if isinstance(representation, dict):
        keywords = representation.get("keywords", [])
        if isinstance(keywords, list):
            tokens.update(str(item).casefold().strip() for item in keywords if str(item).strip())
    return {token for token in tokens if token}


def _memgpt_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("memgpt")
    return dict(nested) if isinstance(nested, dict) else {}


def build_memgpt_store() -> MemoryStore:
    """Build the five-region MemGPT store topology."""

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name=MEMGPT_CORE_LAYER, theme="profile", indices=("keyword",)),
            StoreLayerSpec(name=MEMGPT_WORKING_LAYER, theme="working", indices=("keyword", "temporal")),
            StoreLayerSpec(name=MEMGPT_QUEUE_LAYER, theme="working", indices=("keyword", "temporal")),
            StoreLayerSpec(name=MEMGPT_RECALL_LAYER, theme="episodic", indices=("keyword", "temporal")),
            StoreLayerSpec(name=MEMGPT_ARCHIVAL_LAYER, theme="semantic", indices=("keyword", "temporal", "vector")),
        ]
    )
    return MemoryStore(topology=topology)


def get_block_record(store: MemoryStore, *, layer: str, key_name: str, key_value: str) -> MemoryRecord | None:
    matches = store.find_records_by_key(key_name, key_value, layer=layer)
    return matches[-1] if matches else None


def get_core_block(store: MemoryStore, block_key: str) -> str:
    record = get_block_record(store, layer=MEMGPT_CORE_LAYER, key_name="memgpt_key", key_value=block_key)
    return "" if record is None else record.text


def get_working_summary(store: MemoryStore) -> str:
    record = get_block_record(
        store,
        layer=MEMGPT_WORKING_LAYER,
        key_name="memgpt_key",
        key_value=MEMGPT_WORKING_SUMMARY_KEY,
    )
    return "" if record is None else record.text


class MemGPTKeyedUpsertOrganization(OrganizationModule):
    """Write units into one layer and upsert by a metadata key."""

    spec = ModuleSpec(
        name="memgpt_keyed_upsert_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records", "replace_records"),
    )

    def __init__(self, *, target_layer: str, key_name: str) -> None:
        self.target_layer = target_layer
        self.key_name = key_name

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(
                f"MemGPTKeyedUpsertOrganization requires declared layer {self.target_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemGPTKeyedUpsertOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("MemGPTKeyedUpsertOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("MemGPTKeyedUpsertOrganization requires decisions aligned with units.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        effects: list[dict[str, Any]] = []

        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            if not decision:
                effects.append({"unit_id": unit.unit_id, "effect_type": "skipped"})
                continue

            key_value = str(unit.metadata.get(self.key_name, "")).strip()
            if not key_value:
                raise ValueError(
                    f"MemGPTKeyedUpsertOrganization requires unit.metadata[{self.key_name!r}] for all written units."
                )

            existing = get_block_record(store, layer=self.target_layer, key_name=self.key_name, key_value=key_value)
            if existing is None:
                record = MemoryRecord.from_unit(unit=unit, layer=self.target_layer, sequence_id=store.next_sequence_id())
                store.append(record)
                effects.append(
                    {
                        "unit_id": unit.unit_id,
                        "effect_type": "inserted",
                        "record_id": record.record_id,
                        "key": key_value,
                    }
                )
                continue

            replacement = MemoryRecord(
                record_id=existing.record_id,
                unit_id=unit.unit_id,
                layer=existing.layer,
                text=unit.text,
                timestamp=unit.timestamp,
                embedding=list(unit.embedding) if unit.embedding is not None else None,
                metadata={
                    **unit.metadata,
                    "unit_type": unit.unit_type,
                    "representation": unit.metadata.get("representation", {}),
                },
            )
            store.replace_record(self.target_layer, existing.record_id, replacement)
            effects.append(
                {
                    "unit_id": unit.unit_id,
                    "effect_type": "updated",
                    "record_id": existing.record_id,
                    "key": key_value,
                }
            )

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "key_name": self.key_name,
            "effects": effects,
        }
        return replace(packet, placements=placements, trace=trace), store


class MemGPTPagedRetrieval(RetrievalModule):
    """Retrieve paged results from one MemGPT layer using embedding similarity."""

    spec = ModuleSpec(
        name="memgpt_paged_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("record.embedding",),
    )

    def __init__(self, *, target_layer: str, page_size: int = 3, tool_name: str) -> None:
        if page_size <= 0:
            raise ValueError("MemGPTPagedRetrieval requires page_size > 0.")
        self.target_layer = target_layer
        self.page_size = int(page_size)
        self.tool_name = tool_name

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(
                f"MemGPTPagedRetrieval requires declared layer {self.target_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("MemGPTPagedRetrieval requires packet.query.")

        page = int(packet.query.metadata.get("page", 1) or 1)
        page_size = int(packet.query.metadata.get("page_size", self.page_size) or self.page_size)
        if page <= 0:
            raise ValueError("MemGPTPagedRetrieval requires page >= 1.")
        if page_size <= 0:
            raise ValueError("MemGPTPagedRetrieval requires page_size >= 1.")

        runtime = get_classic_runtime()
        query = packet.query
        query_embedding = (
            [float(value) for value in query.embedding]
            if query.embedding is not None
            else runtime.embed(query.text)
        )
        if query.embedding is None:
            query = replace(query, embedding=query_embedding)

        ordered = list(reversed(store.iter_records(self.target_layer)))
        scored: list[tuple[float, int, MemoryRecord]] = []
        for order_index, record in enumerate(ordered):
            if record.embedding is None or len(record.embedding) != len(query_embedding):
                continue
            score = self._cosine_similarity(query_embedding, record.embedding)
            scored.append((score, order_index, record))

        candidates = sorted(scored, key=lambda item: (-item[0], item[1]))

        total_matches = len(candidates)
        start = (page - 1) * page_size
        end = start + page_size
        selected = candidates[start:end]
        items = [record for _, _, record in selected]
        scores = [
            {
                "record_id": record.record_id,
                "rank": start + rank,
                "score": float(score),
                "layer": self.target_layer,
                "strategy": f"{self.tool_name}_embedding_search",
            }
            for rank, (score, _, record) in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "tool_name": self.tool_name,
                "target_layer": self.target_layer,
                "page": page,
                "page_size": page_size,
                "total_matches": total_matches,
                "returned_count": len(items),
                "strategy": "embedding_similarity",
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, query=query, retrieved=retrieved, trace=trace), store

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


class MemGPTSearchReadout(ReadoutModule):
    """Render search-tool results as JSON payloads for the agent loop."""

    spec = ModuleSpec(
        name="memgpt_search_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, *, tool_name: str, target_layer: str) -> None:
        self.tool_name = tool_name
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("MemGPTSearchReadout requires packet.retrieved.")

        trace_data = packet.retrieved.trace if isinstance(packet.retrieved.trace, dict) else {}
        source_ids = [record.record_id for record in packet.retrieved.items]
        payload = {
            "tool_name": self.tool_name,
            "target_layer": self.target_layer,
            "page": int(trace_data.get("page", 1)),
            "page_size": int(trace_data.get("page_size", len(packet.retrieved.items))),
            "total_matches": int(trace_data.get("total_matches", len(packet.retrieved.items))),
            "has_more": int(trace_data.get("page", 1)) * int(trace_data.get("page_size", len(packet.retrieved.items)))
            < int(trace_data.get("total_matches", len(packet.retrieved.items))),
            "source_ids": source_ids,
            "items": [
                {
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "text": record.text,
                    "timestamp": record.timestamp,
                    "event_type": _memgpt_metadata(record.metadata).get("event_type"),
                    "memgpt_key": record.metadata.get("memgpt_key"),
                }
                for record in packet.retrieved.items
            ],
        }
        readout = Readout(
            text=json.dumps(payload, ensure_ascii=False),
            source_ids=source_ids,
            metadata=payload,
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "tool_name": self.tool_name,
            "source_ids": source_ids,
        }
        return replace(packet, readout=readout, trace=trace), store


__all__ = [
    "MEMGPT_ARCHIVAL_LAYER",
    "MEMGPT_CORE_BLOCK_HUMAN",
    "MEMGPT_CORE_BLOCK_PERSONA",
    "MEMGPT_CORE_LAYER",
    "MEMGPT_LAYER_ORDER",
    "MEMGPT_MAIN_LAYER",
    "MEMGPT_QUEUE_LAYER",
    "MEMGPT_RECALL_LAYER",
    "MEMGPT_REQUIRED_CORE_BLOCKS",
    "MEMGPT_WORKING_LAYER",
    "MEMGPT_WORKING_SUMMARY_KEY",
    "MemGPTKeyedUpsertOrganization",
    "MemGPTPagedRetrieval",
    "MemGPTSearchReadout",
    "build_memgpt_store",
    "get_core_block",
    "get_working_summary",
]
