from __future__ import annotations

import asyncio
import json
from typing import Any

from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
)
from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    Observation,
    Packet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.pipeline_slots import PRE_EVOLUTION_SLOTS


def _stored_pipeline_packet(text: str, store: MemoryStore) -> tuple[Packet, MemoryStore]:
    """Pre-evolution ingest chain; uses the same default modules as the full pipeline."""
    mods = instantiate_default_baseline_modules(top_k=2)
    packet = Packet(observation=Observation(text=text, source="dialogue"))
    for slot in PRE_EVOLUTION_SLOTS:
        packet, store = mods[slot].run(packet, store)
    return packet, store


def _represented_packet(
    text: str,
    *,
    source: str = "dialogue",
    observation_metadata: dict | None = None,
) -> tuple[Packet, MemoryStore]:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    packet = Packet(observation=Observation(text=text, source=source, metadata=observation_metadata or {}))
    packet, store = PassThroughUnitFormation().run(packet, MemoryStore())
    packet, store = BasicRepresentation().run(packet, store)
    return packet, store


def _graph_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )


def _graph_vector_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity", "vector")),
            ]
        )
    )


def _mixed_graph_vector_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity", "vector")),
                StoreLayerSpec(name="other_graph", shape="Graph", indices=("graph", "entity", "vector")),
            ]
        )
    )


def _budgeted_store(
    *,
    layer_name: str = "episodic",
    record_budget: int | None = None,
    token_budget: int | None = None,
) -> MemoryStore:
    settings: dict[str, int] = {}
    if record_budget is not None:
        settings["record_budget"] = record_budget
    if token_budget is not None:
        settings["token_budget"] = token_budget
    capacity = "unlimited"
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name=layer_name, capacity=capacity, settings=settings),
            ]
        )
    )


def _seed_layer(store: MemoryStore, layer: str, texts: list[str]) -> None:
    for index, text in enumerate(texts, start=1):
        unit = MemoryUnit(
            unit_id=f"seed-{index}",
            text=text,
            timestamp=f"2026-01-01T00:00:{index:02d}Z",
            metadata={},
        )
        store.append(MemoryRecord.from_unit(unit=unit, layer=layer, sequence_id=store.next_sequence_id()))


def _seed_layer_with_metadata(store: MemoryStore, layer: str, units: list[dict[str, object]]) -> None:
    for index, payload in enumerate(units, start=1):
        unit = MemoryUnit(
            unit_id=str(payload.get("unit_id", f"seed-{index}")),
            text=str(payload.get("text", f"seed text {index}")),
            timestamp=f"2026-01-01T00:00:{index:02d}Z",
            metadata=dict(payload.get("metadata", {})),
        )
        store.append(MemoryRecord.from_unit(unit=unit, layer=layer, sequence_id=store.next_sequence_id()))


def _invoke_runtime_tool(tool, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.on_invoke_tool(None, json.dumps(arguments, ensure_ascii=False)))


class _FakeAMEMRuntime:
    def require_llm(self, *, capability: str) -> None:
        return None

    def embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            10.0 if "alice" in lowered else 0.0,
            8.0 if "tea" in lowered else 0.0,
            6.0 if "focus" in lowered else 0.0,
            4.0 if "graph" in lowered else 0.0,
            float(len(lowered)),
        ]

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        lowered_system = system.casefold()
        if "enrich memory notes" in lowered_system or "note generator" in lowered_system:
            unit_text = payload["unit_text"].casefold()
            if "alice likes tea" in unit_text:
                return {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                }
            if "tea routines improve focus" in unit_text:
                return {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                }
            return {
                "content": payload["unit_text"],
                "note_text": "Graph note",
                "context": "Graph memory context.",
                "keywords": ["graph", "memory"],
                "tags": ["graph", "memory"],
                "category": "insight",
                "attributes": {"topic": "graph"},
            }
        if "memory write controller" in lowered_system:
            return {"decision": "write", "reason": "store the note", "confidence": 0.9}
        if "choose which neighbors should receive" in lowered_system:
            return {"connections": [0], "tags": ["focus", "tea", "bridge"]}
        if "update each neighbor note's context and tags" in lowered_system:
            return {
                "updates": [
                    {
                        "context": "Alice's tea habit is now understood as a focus-supporting routine.",
                        "tags": ["preference", "habit", "focus"],
                    }
                ]
            }
        if "expand the query" in lowered_system or (
            "expand" in lowered_system and "knowledge_graph" in lowered_system
        ):
            return {
                "query_text": payload["query"],
                "content": payload["query"],
                "context": "Retrieve the most relevant enriched note.",
                "keywords": ["alice", "tea"] if "alice" in payload["query"].casefold() else ["focus", "graph"],
                "tags": ["query", "memory"],
                "category": "query",
                "attributes": {},
            }
        raise AssertionError(f"Unexpected runtime prompt: {system}")

    def rerank(self, *, query: str, candidates: list[dict[str, object]], task: str, top_k: int):
        return [
            {
                "id": str(candidate["id"]),
                "score": float(candidate.get("score", 0.0)),
                "rationale": f"selected for {query}",
            }
            for candidate in sorted(
                candidates,
                key=lambda item: (-float(item.get("score", 0.0)), str(item.get("id", ""))),
            )[:top_k]
        ]


class _WrapperShapeAMEMRuntime(_FakeAMEMRuntime):
    def json(self, *, system: str, user: str):
        payload = super().json(system=system, user=user)
        lowered_system = system.casefold()
        if "choose which neighbors should receive" in lowered_system:
            return [0]
        if "update each neighbor note's context and tags" in lowered_system:
            return payload["updates"]
        return payload


class _FakeHierarchicalRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        self.calls.append({"system": system, "payload": payload})
        fields = payload["extract_fields"]
        records = payload["records"]
        group_key = payload["group_key"]
        if "CUSTOM HIERARCHICAL PROMPT" in system:
            return {
                field: f"custom::{field}::{group_key.get('session_id', 'all')}::{len(records)}"
                for field in fields
            }
        return {
            field: f"generated::{field}::{group_key.get('session_id', 'all')}::{len(records)}"
            for field in fields
        }



