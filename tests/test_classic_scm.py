from __future__ import annotations

import pytest

from memprimitive import MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.classic_modules.scm import (
    SCMControlledRetrieval,
    SCMEntityProfileUpsert,
    SCMJudgeGateWrite,
    SCMStructuredExtraction,
)
from memprimitive.example.classics.scm_self_controlled_memory import build_scm_pipeline


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _scm_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="semantic", theme="semantic", indices=("entity", "keyword", "vector")),
                StoreLayerSpec(name="profile", theme="profile", indices=("entity", "keyword")),
            ]
        )
    )


def test_scm_structured_extraction_extracts_entities_triples_kv_and_embedding() -> None:
    store = _scm_store()
    packet = Packet(observation=Observation(text="(Alice, works_at, ACME); role: engineer", source="dialogue"))

    packet_out, _ = SCMStructuredExtraction().run(packet, store)

    assert packet_out.units is not None
    unit = packet_out.units[0]
    assert "Alice" in unit.entities
    assert ("Alice", "works_at", "ACME") in unit.triples
    assert unit.kv["role"] == "engineer"
    assert unit.embedding is not None
    assert packet_out.trace["unit_formation"]["entity_count"] >= 1
    assert packet_out.trace["unit_formation"]["triple_count"] >= 1


def test_scm_write_trigger_accepts_structured_units_and_rejects_plain_text() -> None:
    store = _scm_store()
    structured_packet, _ = SCMStructuredExtraction().run(
        Packet(observation=Observation(text="Alice works at ACME.", source="dialogue")),
        store,
    )
    plain_packet, _ = SCMStructuredExtraction().run(
        Packet(observation=Observation(text="nothing remarkable here", source="dialogue")),
        store,
    )

    accepted_packet, _ = SCMJudgeGateWrite(threshold=0.5).run(structured_packet, store)
    rejected_packet, _ = SCMJudgeGateWrite(threshold=0.5).run(plain_packet, store)

    assert accepted_packet.decisions == [True]
    assert accepted_packet.trace["write_trigger"]["scorer"] == "llm_judge"
    assert rejected_packet.decisions == [False]
    assert rejected_packet.trace["write_trigger"]["per_unit"][0]["gate"] is False


def test_scm_entity_profile_upsert_merges_multiple_entity_facts() -> None:
    pipeline = build_scm_pipeline(top_k=2, threshold=0.5)

    pipeline.ingest(Observation(text="Alice works at ACME.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    store = pipeline.store
    assert store.count("semantic") == 2
    assert store.count("profile") == 1

    profile_record = store.iter_records("profile")[0]
    profile = profile_record.metadata["profile"]
    assert profile["entity"] == "Alice"
    assert profile["update_count"] == 2
    assert len(profile["source_unit_ids"]) == 2
    assert any("works at" in fact for fact in profile["facts"])
    assert any("likes tea" in fact.lower() for fact in profile["facts"])


def test_scm_controlled_retrieval_prefers_entity_profiles_before_semantic_facts() -> None:
    pipeline = build_scm_pipeline(top_k=2, threshold=0.5)
    pipeline.ingest(Observation(text="Alice works at ACME.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    packet_out, _ = SCMControlledRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), pipeline.store)

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.trace["control_mode"] == "entity_first"
    assert packet_out.retrieved.items[0].layer == "profile"
    assert packet_out.retrieved.items[0].metadata["profile"]["entity"] == "Alice"


def test_scm_example_pipeline_returns_readout_and_source_ids() -> None:
    pipeline = build_scm_pipeline(top_k=3, threshold=0.5)
    pipeline.ingest(Observation(text="Alice works at ACME.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice likes tea and builds retrieval tools.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert "Alice" in readout.text
    assert readout.source_ids
    assert len(readout.source_ids) <= 3
