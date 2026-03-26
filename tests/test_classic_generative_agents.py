from __future__ import annotations

import pytest

from memprimitive.classic_modules.generative_agents import (
    GenerativeAgentsMemoryEvolution,
    GenerativeAgentsReflectionTrigger,
    GenerativeAgentsRepresentation,
    GenerativeAgentsRetrieval,
    GenerativeAgentsWriteTrigger,
    build_generative_agents_pipeline,
    build_generative_agents_topology,
)
from memprimitive.core import MemoryRecord, MemoryStore, Observation, Packet, Query


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _make_store() -> MemoryStore:
    return MemoryStore(topology=build_generative_agents_topology())


def _ga_meta(record: MemoryRecord) -> dict:
    value = record.metadata.get("generative_agents", {})
    return value if isinstance(value, dict) else {}


def _event_record(
    record_id: str,
    *,
    text: str,
    timestamp: str,
    importance: float,
    last_accessed_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        unit_id=f"unit-{record_id}",
        layer="observation_stream",
        text=text,
        timestamp=timestamp,
        metadata={
            "unit_type": "event",
            "importance": importance,
            "representation": {
                "summary": text,
                "keywords": [token.casefold() for token in text.split()[:4]],
                "entities": ["Alice"] if "Alice" in text else [],
                "tags": ["event"],
                "importance": importance,
            },
            "generative_agents": {
                "memory_type": "event",
                "created_at": timestamp,
                "last_accessed_at": last_accessed_at or timestamp,
                "importance": importance,
                "poignancy": importance,
                "keywords": [token.casefold().strip(".,") for token in text.split()[:4]],
                "entities": ["Alice"] if "Alice" in text else [],
                "tags": ["event"],
                "evidence_record_ids": [],
                "depth": 0,
            },
        },
    )


def _thought_record(
    record_id: str,
    *,
    text: str,
    timestamp: str,
    importance: float,
    evidence_record_ids: list[str],
    depth: int,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        unit_id=f"unit-{record_id}",
        layer="reflections",
        text=text,
        timestamp=timestamp,
        metadata={
            "unit_type": "thought",
            "importance": importance,
            "representation": {
                "summary": text,
                "keywords": [token.casefold().strip(".,") for token in text.split()[:5]],
                "entities": ["Alice"] if "Alice" in text else [],
                "tags": ["thought", "reflection"],
                "importance": importance,
            },
            "generative_agents": {
                "memory_type": "thought",
                "created_at": timestamp,
                "last_accessed_at": timestamp,
                "importance": importance,
                "poignancy": importance,
                "keywords": [token.casefold().strip(".,") for token in text.split()[:5]],
                "entities": ["Alice"] if "Alice" in text else [],
                "tags": ["thought", "reflection"],
                "evidence_record_ids": evidence_record_ids,
                "depth": depth,
            },
        },
    )


def test_generative_agents_representation_assigns_event_memory_metadata() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        MemoryStore(),
    )

    packet_out, _ = GenerativeAgentsRepresentation().run(packet, store)

    assert packet_out.units is not None
    unit = packet_out.units[0]
    ga_meta = unit.metadata["generative_agents"]
    assert unit.unit_type == "event"
    assert unit.metadata["importance"] > 0.0
    assert ga_meta["memory_type"] == "event"
    assert ga_meta["last_accessed_at"] == unit.timestamp
    assert ga_meta["depth"] == 0
    assert "summary" in unit.metadata["representation"]
    assert "Alice" in unit.metadata["representation"]["summary"]
    assert packet_out.trace["representation"]["module"] == "generative_agents_representation"


def test_generative_agents_write_trigger_suppresses_recent_duplicate_event() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    store = _make_store()
    store.append(
        _event_record(
            "rec-1",
            text="Alice prefers tea when writing code.",
            timestamp="2026-03-26T10:00:00+00:00",
            importance=0.8,
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        store,
    )
    packet, store = GenerativeAgentsRepresentation().run(packet, store)

    packet_out, _ = GenerativeAgentsWriteTrigger(duplicate_window=3).run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["reason"] == "duplicate_recent_event"


def test_generative_agents_reflection_trigger_uses_cumulative_importance() -> None:
    from memprimitive.baselines import AppendOrganization, PassThroughUnitFormation

    store = _make_store()

    packet_a, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        store,
    )
    packet_a, store = GenerativeAgentsRepresentation().run(packet_a, store)
    packet_a.decisions = [True]
    packet_a, store = AppendOrganization(target_layer="observation_stream").run(packet_a, store)
    packet_a, store = GenerativeAgentsReflectionTrigger(reflection_threshold=1.1, reflection_batch_size=2).run(packet_a, store)

    assert packet_a.evolution_decisions == [False]
    assert packet_a.trace["evolution_trigger"]["triggered_cycle"] is False

    packet_b, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice plans to document the tea ritual.", source="dialogue")),
        store,
    )
    packet_b, store = GenerativeAgentsRepresentation().run(packet_b, store)
    packet_b.decisions = [True]
    packet_b, store = AppendOrganization(target_layer="observation_stream").run(packet_b, store)
    packet_b, store = GenerativeAgentsReflectionTrigger(reflection_threshold=1.1, reflection_batch_size=2).run(packet_b, store)

    assert packet_b.trace["evolution_trigger"]["triggered_cycle"] is True
    assert packet_b.trace["evolution_trigger"]["selected_record_ids"]


def test_generative_agents_memory_evolution_appends_thought_records_with_evidence() -> None:
    from memprimitive.baselines import AppendOrganization, PassThroughUnitFormation

    store = _make_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        store,
    )
    packet, store = GenerativeAgentsRepresentation().run(packet, store)
    packet.decisions = [True]
    packet, store = AppendOrganization(target_layer="observation_stream").run(packet, store)
    packet, store = GenerativeAgentsReflectionTrigger(reflection_threshold=0.1, reflection_batch_size=1).run(packet, store)

    packet_out, updated_store = GenerativeAgentsMemoryEvolution(
        focal_point_count=1,
        insights_per_cycle=1,
    ).run(packet, store)

    assert packet_out.trace["memory_evolution"]["triggered_cycle"] is True
    assert packet_out.trace["memory_evolution"]["focal_points"]
    assert packet_out.trace["memory_evolution"]["thought_record_ids"]
    thought_record = updated_store.iter_records("reflections")[0]
    ga_meta = _ga_meta(thought_record)
    assert thought_record.metadata["unit_type"] == "thought"
    assert ga_meta["memory_type"] == "thought"
    assert ga_meta["evidence_record_ids"]
    assert ga_meta["depth"] >= 1


def test_generative_agents_retrieval_prefers_thoughts_and_updates_last_accessed() -> None:
    store = _make_store()
    store.append(
        _event_record(
            "rec-1",
            text="Alice likes tea and writes careful notes.",
            timestamp="2026-03-26T10:00:00+00:00",
            importance=0.55,
            last_accessed_at="2026-03-26T10:00:00+00:00",
        )
    )
    store.append(
        _thought_record(
            "rec-2",
            text="Insight: Alice consistently links tea to focused coding.",
            timestamp="2026-03-26T10:10:00+00:00",
            importance=0.92,
            evidence_record_ids=["rec-1"],
            depth=1,
        )
    )

    query = Query(text="Alice tea coding", timestamp="2026-03-26T11:00:00+00:00")
    packet_out, _ = GenerativeAgentsRetrieval(top_k=2).run(Packet(query=query), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert packet_out.retrieved.scores[0]["memory_type"] == "thought"
    assert packet_out.retrieved.scores[0]["strategy"] == "normalized_relevance_recency_importance"
    assert _ga_meta(store.iter_records("reflections")[0])["last_accessed_at"] == query.timestamp


def test_generative_agents_pipeline_keeps_public_entrypoints_and_produces_thought_memory() -> None:
    pipeline = build_generative_agents_pipeline(
        top_k=3,
        reflection_threshold=0.1,
        reflection_batch_size=1,
        focal_point_count=1,
        insights_per_cycle=1,
    )

    packet = pipeline.ingest(Observation(text="Alice prefers tea when writing code.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice tea coding"))

    assert packet.trace["memory_evolution"]["thought_record_ids"]
    assert pipeline.store.count("reflections") >= 1
    assert readout.source_ids
    assert readout.text
