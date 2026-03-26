from __future__ import annotations

import pytest

from memprimitive.classic_modules.generative_agents import (
    GenerativeAgentsMemoryEvolution,
    GenerativeAgentsReflectionTrigger,
    GenerativeAgentsRepresentation,
    GenerativeAgentsRetrieval,
    build_generative_agents_pipeline,
    build_generative_agents_topology,
)
from memprimitive.core import MemoryRecord, MemoryStore, Observation, Packet, Query


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _make_store() -> MemoryStore:
    return MemoryStore(topology=build_generative_agents_topology())


def test_generative_agents_representation_assigns_importance_and_summary() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        MemoryStore(),
    )

    packet_out, _ = GenerativeAgentsRepresentation().run(packet, store)

    assert packet_out.units is not None
    unit = packet_out.units[0]
    assert unit.metadata["importance"] > 0.0
    assert "summary" in unit.metadata["representation"]
    assert "Alice" in unit.metadata["representation"]["summary"]
    assert packet_out.trace["representation"]["module"] == "generative_agents_representation"


def test_generative_agents_reflection_pipeline_appends_reflections_once() -> None:
    pipeline = build_generative_agents_pipeline(
        top_k=3,
        reflection_threshold=0.75,
        reflection_batch_size=2,
    )

    pipeline.ingest(Observation(text="Alice prefers tea when writing code.", source="dialogue"))
    first_reflection_count = pipeline.store.count("reflections")
    assert first_reflection_count == 1

    pipeline.ingest(Observation(text="Blue.", source="notes"))
    assert pipeline.store.count("reflections") == first_reflection_count

    reflection_record = pipeline.store.iter_records("reflections")[0]
    assert reflection_record.text.startswith("Reflection:")
    assert reflection_record.metadata["importance"] >= 0.5


def test_generative_agents_reflection_trigger_marks_salient_new_units() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, AppendOrganization, PassThroughUnitFormation

    store = _make_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        store,
    )
    packet, store = GenerativeAgentsRepresentation().run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = AppendOrganization(target_layer="observation_stream").run(packet, store)

    packet_out, _ = GenerativeAgentsReflectionTrigger(
        source_layer="observation_stream",
        reflection_layer="reflections",
        reflection_threshold=0.5,
        reflection_batch_size=2,
    ).run(packet, store)

    assert packet_out.evolution_decisions == [True]
    assert packet_out.trace["evolution_trigger"]["selected_unit_ids"] == [packet.units[0].unit_id]
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["reason"] == "candidate"


def test_generative_agents_weighted_retrieval_prefers_relevance_recency_and_importance() -> None:
    store = _make_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="u1",
            layer="observation_stream",
            text="Alice likes tea and writes careful notes.",
            timestamp="2026-03-26T10:00:00+00:00",
            metadata={
                "unit_type": "observation",
                "importance": 0.58,
                "representation": {
                    "summary": "Alice likes tea and writes careful notes.",
                    "keywords": ["alice", "tea", "notes"],
                    "entities": ["Alice"],
                    "tags": ["observation", "preference"],
                    "importance": 0.58,
                },
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="u2",
            layer="observation_stream",
            text="The desk lamp is blue.",
            timestamp="2026-03-26T10:05:00+00:00",
            metadata={
                "unit_type": "observation",
                "importance": 0.1,
                "representation": {
                    "summary": "The desk lamp is blue.",
                    "keywords": ["desk", "lamp", "blue"],
                    "entities": [],
                    "tags": ["observation"],
                    "importance": 0.1,
                },
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="u3",
            layer="reflections",
            text="Reflection: Alice prefers tea when writing code.",
            timestamp="2026-03-26T10:10:00+00:00",
            metadata={
                "unit_type": "reflection",
                "importance": 0.92,
                "representation": {
                    "summary": "Reflection: Alice prefers tea when writing code.",
                    "keywords": ["alice", "tea", "code"],
                    "entities": ["Alice"],
                    "tags": ["reflection", "pattern"],
                    "importance": 0.92,
                },
            },
        )
    )

    packet_out, _ = GenerativeAgentsRetrieval(top_k=2).run(Packet(query=Query(text="Alice tea")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-3", "rec-1"]
    assert packet_out.retrieved.scores[0]["strategy"] == "weighted_relevance_recency_importance"
    assert packet_out.retrieved.scores[0]["importance"] >= packet_out.retrieved.scores[1]["importance"]
    assert packet_out.retrieved.scores[0]["score"] > packet_out.retrieved.scores[1]["score"]


def test_generative_agents_memory_evolution_appends_reflection_records() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, AppendOrganization, PassThroughUnitFormation

    store = _make_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice prefers tea when writing code.", source="dialogue")),
        store,
    )
    packet, store = GenerativeAgentsRepresentation().run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = AppendOrganization(target_layer="observation_stream").run(packet, store)
    packet, store = GenerativeAgentsReflectionTrigger(
        source_layer="observation_stream",
        reflection_layer="reflections",
        reflection_threshold=0.5,
        reflection_batch_size=2,
    ).run(packet, store)

    packet_out, updated_store = GenerativeAgentsMemoryEvolution(
        source_layer="observation_stream",
        reflection_layer="reflections",
    ).run(packet, store)

    assert packet_out.trace["memory_evolution"]["effects"]
    assert updated_store.count("reflections") == 1
    assert updated_store.iter_records("reflections")[0].metadata["source_record_id"] == "rec-1"
