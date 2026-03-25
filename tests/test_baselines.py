from __future__ import annotations

from dataclasses import replace
import json
import pytest

from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
    registered_baseline_class_names,
)
from memprimitive.core import MemoryRecord, MemoryStore, Observation, Packet, Query, RetrievedSet, StoreLayerSpec, StoreTopology
from memprimitive.pipeline_slots import PRE_EVOLUTION_SLOTS


def _stored_pipeline_packet(text: str, store: MemoryStore) -> tuple[Packet, MemoryStore]:
    """Pre-evolution ingest chain; uses the same default modules as the full pipeline."""
    mods = instantiate_default_baseline_modules(top_k=2)
    packet = Packet(observation=Observation(text=text, source="dialogue"))
    for slot in PRE_EVOLUTION_SLOTS:
        packet, store = mods[slot].run(packet, store)
    return packet, store


def test_unit_formation_returns_one_unit_with_provenance() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    module = PassThroughUnitFormation()
    packet = Packet(observation=Observation(text="Alice likes tea.", source="dialogue"))

    packet_out, _ = module.run(packet, MemoryStore())

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    assert packet_out.units[0].text == "Alice likes tea."
    assert packet_out.units[0].metadata["provenance"]["observation_id"] == packet.observation.observation_id


def test_unit_formation_requires_observation() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    module = PassThroughUnitFormation()

    with pytest.raises(ValueError, match="packet.observation"):
        module.run(Packet(), MemoryStore())


def test_representation_preserves_identity_and_adds_normalized_text() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="  Alice Likes Tea  ", source="dialogue")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation().run(unit_packet, store)

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    assert packet_out.units[0].unit_id == unit_packet.units[0].unit_id
    assert packet_out.units[0].text == "Alice Likes Tea"
    assert packet_out.units[0].normalized_text == "alice likes tea"
    assert packet_out.units[0].embedding is not None
    assert len(packet_out.units[0].embedding) > 0
    assert packet_out.units[0].representation_elements == ("embedding", "text")
    assert packet_out.trace["representation"]["elements"] == ["text", "embedding"]
    assert packet_out.trace["representation"]["per_unit"][0]["elements"] == ["embedding", "text"]
    assert packet_out.units[0].metadata["representation"]["text"] == "Alice Likes Tea"
    assert packet_out.units[0].metadata["representation"]["normalized_text"] == "alice likes tea"
    assert packet_out.units[0].metadata["representation"]["embedding"]["dim"] == len(packet_out.units[0].embedding)


def test_representation_can_build_structured_element_sets() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea. role: engineer", source="dialogue")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation(elements=("text", "triple", "kv", "entities", "tags")).run(unit_packet, store)

    unit = packet_out.units[0]
    assert ("Alice", "likes", "tea") in unit.triples
    assert unit.kv["role"] == "engineer"
    assert "Alice" in unit.entities
    assert "structured_triple" in unit.tags
    assert "structured_kv" in unit.tags
    assert unit.metadata["representation"]["triples"]
    assert unit.metadata["representation"]["kv"]["role"] == "engineer"
    assert unit.metadata["representation"]["entities"] == unit.entities
    assert unit.metadata["representation"]["tags"] == unit.tags


def test_representation_can_build_hybrid_element_set() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Graph memory helps Alice study code.", source="notes")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation(elements=("text", "embedding", "triple", "tags", "entities")).run(
        unit_packet,
        store,
    )

    unit = packet_out.units[0]
    assert unit.embedding is not None
    assert "embedding" in unit.representation_elements
    assert "text" in unit.representation_elements
    assert "entities" in unit.representation_elements
    assert "tags" in unit.representation_elements
    assert "Alice" in unit.entities
    assert "graph" in unit.tags
    assert "memory" in unit.tags


def test_representation_description_requires_openai_config() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice writes reusable Python code for graph memory tools.", source="notes")),
        MemoryStore(),
    )
    rep = BasicRepresentation(
        elements=("text", "description"),
        api_key="",
        base_url="",
        model="",
    )
    with pytest.raises(ValueError, match="description.*MEMPRIMITIVE"):
        rep.run(unit_packet, store)


def test_representation_can_generate_real_description_via_api() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    probe = BasicRepresentation(elements=("text", "description"))
    if not (probe.api_key and probe.base_url and probe.model):
        pytest.skip("Requires MEMPRIMITIVE_API_KEY, MEMPRIMITIVE_BASE_URL, MEMPRIMITIVE_MODEL for LLM description")

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice writes reusable Python code for graph memory tools.", source="notes")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation(elements=("text", "entities", "tags", "description")).run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.description is not None
    assert len(unit.description) > 10
    assert "alice" in unit.description.casefold() or "python" in unit.description.casefold()
    assert unit.metadata["representation"]["description"] == unit.description


def test_representation_summary_requires_openai_config() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = BasicRepresentation(
        elements=("text", "summary"),
        api_key="",
        base_url="",
        model="",
    )
    with pytest.raises(ValueError, match="summary.*MEMPRIMITIVE"):
        rep.run(unit_packet, store)


def test_representation_can_generate_real_summary_via_api() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    probe = BasicRepresentation(elements=("text", "summary"))
    if not (probe.api_key and probe.base_url and probe.model):
        pytest.skip("Requires MEMPRIMITIVE_API_KEY, MEMPRIMITIVE_BASE_URL, MEMPRIMITIVE_MODEL for LLM summary")

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory and retrieval for long contexts.", source="notes")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation(elements=("text", "entities", "tags", "summary")).run(unit_packet, store)

    unit = packet_out.units[0]
    summary = unit.metadata["representation"].get("summary")
    assert isinstance(summary, str)
    assert len(summary) > 8
    assert "alice" in summary.casefold() or "graph" in summary.casefold() or "memory" in summary.casefold()


def test_write_trigger_aligns_decisions_with_units() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, BasicRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)

    packet_out, _ = AlwaysWriteTrigger().run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["policy"] == "always"
    assert packet_out.trace["write_trigger"]["scorer"] == "identity"
    assert packet_out.trace["write_trigger"]["output_field"] == "decisions"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["signals"] == {"constant": 1.0}
    assert packet_out.trace["write_trigger"]["per_unit"][0]["score"] == 1.0
    assert packet_out.trace["write_trigger"]["per_unit"][0]["gate"] is True
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is True


def test_evolution_trigger_aligns_evolution_decisions_with_units() -> None:
    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        AppendOrganization,
        BasicRepresentation,
        NeverEvolutionTrigger,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = AppendOrganization().run(packet, store)

    packet_out, _ = NeverEvolutionTrigger().run(packet, store)

    assert packet_out.evolution_decisions == [False]
    assert packet_out.trace["evolution_trigger"]["policy"] == "never"
    assert packet_out.trace["evolution_trigger"]["scorer"] == "identity"
    assert packet_out.trace["evolution_trigger"]["evolution_decisions"] == [False]
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["signals"] == {"constant": 1.0}
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["score"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["gate"] is True
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is False


def test_organization_aligns_placements_with_units_and_commits_normal_write() -> None:
    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)

    packet_out, updated_store = AppendOrganization().run(packet, store)

    assert packet_out.placements is not None
    assert len(packet_out.placements) == len(packet_out.units)
    assert packet_out.placements[0].target_layer == "default"
    assert updated_store.count() == 1
    assert packet_out.trace["organization"]["written_record_ids"]
    assert packet_out.trace["organization"]["written_unit_ids"] == [packet_out.units[0].unit_id]
    assert packet_out.trace["organization"]["skipped_unit_count"] == 0


def test_append_only_evolution_is_noop_when_evolution_decisions_are_false() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        evolution_decisions=[False],
        placements=packet.placements,
        trace=packet.trace,
    )

    _, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 1


def test_append_only_evolution_records_active_unit_ids_without_mutating_store() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        evolution_decisions=[True],
        placements=packet.placements,
        trace=packet.trace,
    )

    packet_out, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 1
    assert packet_out.trace["memory_evolution"]["decision_source"] == "evolution_decisions"
    assert packet_out.trace["memory_evolution"]["active_unit_ids"] == [packet.units[0].unit_id]
    assert packet_out.trace["memory_evolution"]["effects"] == []


def test_append_only_evolution_requires_explicit_evolution_decisions() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        trace=packet.trace,
    )

    with pytest.raises(ValueError, match="packet.evolution_decisions"):
        AppendOnlyEvolution().run(packet, store)


def test_append_only_evolution_requires_aligned_inputs() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    with pytest.raises(ValueError, match="aligned units"):
        AppendOnlyEvolution().run(
            Packet(units=[], evolution_decisions=[True], placements=[]),
            MemoryStore(),
        )


def test_write_and_evolution_trigger_are_independent_by_default() -> None:
    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        AppendOrganization,
        BasicRepresentation,
        NeverEvolutionTrigger,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    write_packet, store = AlwaysWriteTrigger().run(packet, store)
    write_packet, store = AppendOrganization().run(write_packet, store)
    evolution_packet, _ = NeverEvolutionTrigger().run(write_packet, store)

    assert write_packet.decisions == [True]
    assert evolution_packet.evolution_decisions == [False]
    assert write_packet.trace["write_trigger"]["family"] == evolution_packet.trace["evolution_trigger"]["family"]


def test_threshold_write_trigger_respects_threshold_policy() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation, ThresholdWriteTrigger

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)

    packet_out, _ = ThresholdWriteTrigger(threshold=0.8, constant=0.7).run(packet, store)
    assert packet_out.decisions == [False]
    assert packet_out.trace["write_trigger"]["policy"] == "threshold"
    assert packet_out.trace["write_trigger"]["scorer"] == "weighted_sum"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["score"] == 0.7
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is False

    packet_out, _ = ThresholdWriteTrigger(threshold=0.7, constant=0.7).run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is True


def test_threshold_evolution_trigger_writes_only_evolution_decisions() -> None:
    from memprimitive.baselines import (
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
        ThresholdEvolutionTrigger,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = ThresholdEvolutionTrigger(threshold=2.0, constant=1.0).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.evolution_decisions == [False]
    assert packet_out.trace["evolution_trigger"]["policy"] == "threshold"
    assert packet_out.trace["evolution_trigger"]["scorer"] == "weighted_sum"
    assert packet_out.trace["evolution_trigger"]["output_field"] == "evolution_decisions"
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is False


def test_composed_write_trigger_validates_input_requirements_at_entry() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation
    from memprimitive.baselines._trigger_family import (
        AlwaysOpenGate,
        ConstantSignal,
        ThresholdPolicy,
        WeightedSumScorer,
    )
    from memprimitive.baselines.write_trigger import compose_write_trigger

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    trigger = compose_write_trigger(
        name="query_aware_write_trigger",
        signal_providers=(ConstantSignal(signal_name="constant", value=1.0),),
        scorer=WeightedSumScorer(weights={"constant": 1.0}),
        gate=AlwaysOpenGate(),
        policy=ThresholdPolicy(threshold=0.5),
        input_requirements=("units", "query"),
    )

    with pytest.raises(ValueError, match="query is required for trigger execution"):
        trigger.run(packet, store)


def test_composed_evolution_trigger_validates_custom_input_requirements_at_entry() -> None:
    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
    )
    from memprimitive.baselines._trigger_family import (
        AlwaysOpenGate,
        ConstantSignal,
        ThresholdPolicy,
        WeightedSumScorer,
    )
    from memprimitive.baselines.evolution_trigger import compose_evolution_trigger

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = AppendOrganization().run(packet, store)
    trigger = compose_evolution_trigger(
        name="query_aware_evolution_trigger",
        signal_providers=(ConstantSignal(signal_name="constant", value=1.0),),
        scorer=WeightedSumScorer(weights={"constant": 1.0}),
        gate=AlwaysOpenGate(),
        policy=ThresholdPolicy(threshold=0.5),
        input_requirements=("units", "placements", "query"),
    )

    with pytest.raises(ValueError, match="query is required for trigger execution"):
        trigger.run(packet, store)


def test_retrieval_honors_top_k() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("one", "two", "three"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="items")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2


def test_retrieval_rejects_non_positive_top_k() -> None:
    from memprimitive.baselines import RecencyRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        RecencyRetrieval(top_k=0)


def test_embedding_similarity_retrieval_rejects_non_positive_top_k() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        EmbeddingSimilarityRetrieval(top_k=0)


def test_retrieval_on_empty_store_returns_empty_retrieved_set() -> None:
    from memprimitive.baselines import RecencyRetrieval

    packet_out, store_out = RecencyRetrieval(top_k=2).run(
        Packet(query=Query(text="alice")),
        MemoryStore(),
    )

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert store_out.count() == 0


def test_readout_formats_deterministic_text_and_source_ids() -> None:
    from memprimitive.baselines import ConcatenateReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    retrieved = RetrievedSet(items=list(reversed(store.iter_records())), scores=[])

    packet_out, _ = ConcatenateReadout().run(Packet(retrieved=retrieved), store)

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Alice likes tea."
    assert packet_out.readout.source_ids == [store.iter_records()[0].record_id]


def test_readout_on_empty_retrieval_returns_valid_empty_output() -> None:
    from memprimitive.baselines import ConcatenateReadout

    packet_out, _ = ConcatenateReadout().run(Packet(retrieved=RetrievedSet()), MemoryStore())

    assert packet_out.readout is not None
    assert packet_out.readout.text == ""
    assert packet_out.readout.source_ids == []


def test_retrieval_prefers_keyword_matches_when_available() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob prefers coffee", "Alice studies graphs"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2
    assert all("alice" in record.text.casefold() for record in packet_out.retrieved.items)


def test_retrieval_returns_latest_records_first_when_falling_back_to_recency() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("first item", "second item", "third item"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="unmatched")), store)

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["third item", "second item"]


def test_retrieval_does_not_mutate_store() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea", store)
    before_ids = [record.record_id for record in store.iter_records()]

    _, store_after = RecencyRetrieval(top_k=1).run(Packet(query=Query(text="Alice")), store)

    assert [record.record_id for record in store_after.iter_records()] == before_ids


def test_embedding_similarity_retrieval_ranks_records_by_query_embedding() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    records = [
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="closest",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        ),
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="second",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[0.8, 0.2],
            metadata={"representation": {"embedding": {"dim": 2}}},
        ),
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="default",
            text="opposite",
            timestamp="2026-01-01T00:00:02+00:00",
            embedding=[-1.0, 0.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        ),
    ]
    for record in records:
        store.append(record)

    packet_out, store_after = EmbeddingSimilarityRetrieval(top_k=2).run(
        Packet(query=Query(text="ignored", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1", "rec-2"]
    assert packet_out.retrieved.scores[0]["strategy"] == "embedding_similarity"
    assert packet_out.retrieved.scores[0]["record_id"] == "rec-1"
    assert packet_out.retrieved.scores[0]["rank"] == 1
    assert packet_out.retrieved.scores[0]["score"] >= packet_out.retrieved.scores[1]["score"]
    assert packet_out.trace["retrieval"]["reused_query_embedding"] is True
    assert [record.record_id for record in store_after.iter_records()] == [record.record_id for record in store.iter_records()]


def test_embedding_similarity_retrieval_computes_and_caches_query_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="alpha",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="beta",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[0.0, 1.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        )
    )

    def _fake_embed_text(self, text: str) -> list[float]:
        assert text == "alpha query"
        return [1.0, 0.0]

    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", _fake_embed_text)

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1).run(Packet(query=Query(text="alpha query")), store)

    assert packet_out.query is not None
    assert packet_out.query.embedding == [1.0, 0.0]
    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.trace["retrieval"]["reused_query_embedding"] is False
    assert packet_out.trace["retrieval"]["embedding_candidate_count"] == 2


def test_embedding_similarity_retrieval_uses_record_embedding_not_metadata_summary() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="metadata-only",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=None,
            metadata={"representation": {"embedding": {"dim": 2}}},
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1).run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.trace["retrieval"]["candidate_count"] == 1
    assert packet_out.trace["retrieval"]["embedding_candidate_count"] == 0


def test_embedding_similarity_retrieval_skips_missing_and_mismatched_embeddings() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="usable",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="missing",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=None,
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="default",
            text="wrong-dim",
            timestamp="2026-01-01T00:00:02+00:00",
            embedding=[1.0, 0.0, 0.0],
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=3).run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.trace["retrieval"]["embedding_candidate_count"] == 1
    assert packet_out.trace["retrieval"]["skipped_dim_mismatch_count"] == 1


def test_embedding_similarity_retrieval_can_target_declared_topology_layer() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episode"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="default",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="episodic",
            text="episodic-best",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[1.0, 0.0],
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1, layer="episodic").run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2"]
    assert packet_out.trace["retrieval"]["candidate_count"] == 1


def test_organization_can_write_into_declared_non_default_topology_layer() -> None:
    from memprimitive.baselines import AppendOrganization

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
        ]
    )
    store = MemoryStore(topology=topology)
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    packet, store = AppendOrganization(target_layer="episodic").run(packet, store)

    assert store.count("episodic") == 1
    assert store.iter_records("episodic")[0].layer == "episodic"


def test_retrieval_can_target_declared_topology_layer() -> None:
    from memprimitive.baselines import AppendOrganization, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episode"),
        ]
    )
    store = MemoryStore(topology=topology)
    for text in ("episodic first", "episodic second"):
        packet, store = _stored_pipeline_packet(text, store)
        packet, store = AppendOrganization(target_layer="episodic").run(packet, store)

    packet_out, _ = RecencyRetrieval(top_k=1, layer="episodic").run(Packet(query=Query(text="episodic")), store)

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["episodic second"]


def test_layer_aware_retrieval_merges_per_layer_results_and_applies_global_top_k() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval, LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="semantic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working hit",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-semantic-1",
            unit_id="unit-semantic-1",
            layer="semantic",
            text="semantic best",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-semantic-2",
            unit_id="unit-semantic-2",
            layer="semantic",
            text="semantic weaker",
            timestamp="2026-01-01T00:00:02+00:00",
            embedding=[0.8, 0.2],
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=2),
        retriever_by_layer={"semantic": EmbeddingSimilarityRetrieval(top_k=2)},
        top_k=2,
    ).run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-semantic-1", "rec-semantic-2"]
    assert packet_out.retrieved.scores[0]["merge_rank"] == 1
    assert packet_out.retrieved.scores[0]["merge_key_type"] == "score"
    assert packet_out.retrieved.scores[0]["layer"] == "semantic"
    assert packet_out.trace["retrieval"]["merge_strategy"] == "global_rank"
    assert packet_out.trace["retrieval"]["total_merged_count"] == 3
    assert packet_out.trace["retrieval"]["final_returned_count"] == 2


def test_layer_aware_retrieval_falls_back_to_default_retriever_for_unconfigured_layers() -> None:
    from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working latest",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-episodic-1",
            unit_id="unit-episodic-1",
            layer="episodic",
            text="episodic latest",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=1),
        retriever_by_layer={"working": RecencyRetrieval(top_k=1)},
        top_k=2,
    ).run(Packet(query=Query(text="latest")), store)

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-working-1", "rec-episodic-1"]
    assert [entry["module"] for entry in packet_out.trace["retrieval"]["per_layer"]] == [
        "recency_retrieval",
        "recency_retrieval",
    ]


def test_layer_aware_retrieval_can_limit_active_layers() -> None:
    from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working memory",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-episodic-1",
            unit_id="unit-episodic-1",
            layer="episodic",
            text="episodic memory",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=1),
        active_layers=("episodic",),
        top_k=2,
    ).run(Packet(query=Query(text="memory")), store)

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-episodic-1"]
    assert packet_out.trace["retrieval"]["active_layers"] == ["episodic"]


def test_layer_aware_retrieval_uses_layer_order_to_break_rank_ties() -> None:
    from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working rank one",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-episodic-1",
            unit_id="unit-episodic-1",
            layer="episodic",
            text="episodic rank one",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=1),
        top_k=2,
    ).run(Packet(query=Query(text="rank")), store)

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-working-1", "rec-episodic-1"]
    assert packet_out.retrieved.scores[0]["merge_key_type"] == "rank"
    assert packet_out.retrieved.scores[1]["merge_key_type"] == "rank"


def test_layer_aware_retrieval_returns_valid_empty_result_for_empty_store() -> None:
    from memprimitive.baselines import LayerAwareRetrieval

    packet_out, store_out = LayerAwareRetrieval(top_k=2).run(
        Packet(query=Query(text="query")),
        MemoryStore(),
    )

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.trace["retrieval"]["per_layer"][0]["candidate_count"] == 0
    assert store_out.count() == 0


def test_layer_aware_retrieval_validates_inputs() -> None:
    from memprimitive.baselines import LayerAwareRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        LayerAwareRetrieval(top_k=0)

    with pytest.raises(ValueError, match="merge_strategy='global_rank'"):
        LayerAwareRetrieval(merge_strategy="round_robin")

    with pytest.raises(TypeError, match="default_retriever"):
        LayerAwareRetrieval(default_retriever=object())

    with pytest.raises(TypeError, match="retriever_by_layer values"):
        LayerAwareRetrieval(retriever_by_layer={"semantic": object()})

    topology = StoreTopology.from_layers([StoreLayerSpec(name="working")])
    store = MemoryStore(topology=topology)
    with pytest.raises(ValueError, match="not declared in the store topology"):
        LayerAwareRetrieval(active_layers=("missing",)).run(Packet(query=Query(text="query")), store)


def test_store_capability_queries_reflect_declared_topology() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working", indices=("keyword",)),
                StoreLayerSpec(name="graph", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )

    assert store.has_graph_layer() is True
    assert store.has_keyword_layer() is True
    assert store.layer_supports_index("graph", "graph") is True


def test_baselines_simple_reexports_match_package_exports() -> None:
    import memprimitive.baselines as pkg
    import memprimitive.baselines.simple as legacy

    assert set(pkg.__all__) == set(legacy.__all__)
    for name in sorted(pkg.__all__):
        assert getattr(pkg, name) is getattr(legacy, name), name


def test_baselines_all_matches_registered_baseline_classes() -> None:
    """``__init__.__all__`` must list exactly the classes registered in per-module ``BASELINE_CLASSES``."""
    import memprimitive.baselines as pkg

    assert set(pkg.__all__) == registered_baseline_class_names()


def test_write_false_skips_normal_write_and_leaves_evolution_noop() -> None:
    from memprimitive.baselines import (
        AppendOnlyEvolution,
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
    )

    store = MemoryStore()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        store,
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[False],
        trace=packet.trace,
    )
    packet, store = AppendOrganization().run(packet, store)
    packet = Packet(
        units=packet.units,
        decisions=packet.decisions,
        evolution_decisions=[False],
        placements=packet.placements,
        trace=packet.trace,
    )
    packet, store = AppendOnlyEvolution().run(packet, store)

    assert store.count() == 0
    assert packet.trace["organization"]["written_record_ids"] == []
    assert packet.trace["memory_evolution"]["effects"] == []


def test_sentence_split_unit_formation_splits_sentences_and_preserves_provenance() -> None:
    from memprimitive.baselines import SentenceSplitUnitFormation

    packet_out, _ = SentenceSplitUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea. Bob prefers coffee!", source="dialogue")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["Alice likes tea.", "Bob prefers coffee!"]
    assert all("provenance" in unit.metadata for unit in packet_out.units)


def test_line_split_unit_formation_filters_empty_lines() -> None:
    from memprimitive.baselines import LineSplitUnitFormation

    packet_out, _ = LineSplitUnitFormation().run(
        Packet(observation=Observation(text="alpha\n\n beta \n", source="notes")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["alpha", "beta"]


def test_windowed_unit_formation_creates_overlapping_windows() -> None:
    from memprimitive.baselines import WindowedUnitFormation

    packet_out, _ = WindowedUnitFormation(window_size=5, stride=3).run(
        Packet(observation=Observation(text="abcdefghij", source="notes")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["abcde", "defgh", "ghij"]
    assert packet_out.units[1].metadata["window_index"] == 1


def test_metadata_hint_unit_formation_prefers_hint_and_can_set_unit_type() -> None:
    from memprimitive.baselines import MetadataHintUnitFormation

    packet_out, _ = MetadataHintUnitFormation().run(
        Packet(
            observation=Observation(
                text="fallback",
                source="notes",
                metadata={"units": [{"text": "Alice likes tea", "unit_type": "fact"}]},
            )
        ),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["Alice likes tea"]
    assert packet_out.units[0].unit_type == "fact"
    assert packet_out.trace["unit_formation"]["mode"] == "metadata"


def test_representation_supports_new_elements_and_persists_them_into_record_metadata() -> None:
    from memprimitive.baselines import AppendOrganization, AlwaysWriteTrigger, BasicRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory on 2026-03-24.", source="notes")),
        MemoryStore(),
    )
    u0 = packet.units[0]
    packet = replace(
        packet,
        units=[
            replace(
                u0,
                metadata={**u0.metadata, "summary": "Alice studies graph memory on 2026-03-24."},
            )
        ],
    )
    packet, store = BasicRepresentation(
        elements=("text", "entities", "tags", "keywords", "summary", "time_anchor", "relation_tags", "source_type")
    ).run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    _, store = AppendOrganization().run(packet, store)

    record = store.iter_records()[0]
    rep = record.metadata["representation"]
    assert "keywords" in rep
    assert "summary" in rep
    assert "time_anchor" in rep
    assert rep["source_type"] == "notes"


def test_keyword_representation_exposes_keyword_summary_without_embedding() -> None:
    from memprimitive.baselines import KeywordRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice builds retrieval tools for memory graphs.", source="notes")),
        MemoryStore(),
    )
    packet, _ = KeywordRepresentation().run(packet, store)

    rep = packet.units[0].metadata["representation"]
    assert "keywords" in rep
    assert packet.units[0].embedding is None


def test_trigger_family_new_components_compute_scores_and_gates() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation
    from memprimitive.baselines._trigger_family import (
        AverageScorer,
        HasEntitySignal,
        QueryOverlapSignal,
        QueryPresentGate,
        ThresholdOrGatePolicy,
    )
    from memprimitive.baselines.write_trigger import compose_write_trigger

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies memory graphs", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation(elements=("text", "entities")).run(packet, store)
    packet = Packet(observation=packet.observation, units=packet.units, query=Query(text="Alice memory"), trace=packet.trace)
    trigger = compose_write_trigger(
        name="query_overlap_gate_trigger",
        signal_providers=(HasEntitySignal(), QueryOverlapSignal()),
        scorer=AverageScorer(sources=("has_entity", "query_overlap")),
        gate=QueryPresentGate(),
        policy=ThresholdOrGatePolicy(threshold=1.5),
        input_requirements=("units", "query"),
    )

    packet_out, _ = trigger.run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["score"] >= 1.0


def test_conditional_layer_organization_routes_entity_rich_units_to_semantic() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, BasicRepresentation, ConditionalLayerOrganization, PassThroughUnitFormation

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working"),
                StoreLayerSpec(name="semantic", theme="semantic", indices=("entity", "keyword")),
            ]
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        store,
    )
    packet, store = BasicRepresentation(elements=("text", "entities", "tags")).run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = ConditionalLayerOrganization(
        default_layer="working",
        rules=({"has_entity": True, "target_layer": "semantic"},),
    ).run(packet, store)

    assert packet.placements[0].target_layer == "semantic"
    assert store.count("semantic") == 1


def test_graph_append_organization_requires_graph_layer_and_writes_graph_metadata() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, BasicRepresentation, GraphAppendOrganization, PassThroughUnitFormation

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = BasicRepresentation(elements=("text", "entities", "triple")).run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    _, store = GraphAppendOrganization(target_layer="knowledge_graph").run(packet, store)

    record = store.iter_records("knowledge_graph")[0]
    assert "graph" in record.metadata
    assert record.metadata["graph"]["triples"]


def test_summary_rewrite_evolution_appends_summary_record() -> None:
    from memprimitive.baselines import SummaryRewriteEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    packet, store = _stored_pipeline_packet("Alice likes jasmine tea.", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        evolution_decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = SummaryRewriteEvolution(target_layer="semantic").run(packet, store)

    assert store.count("semantic") == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["effect_type"] == "summary_append"


def test_layer_move_evolution_copy_appends_unit_to_target_layer() -> None:
    from memprimitive.baselines import LayerMoveEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    packet, store = _stored_pipeline_packet("Alice likes jasmine tea.", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        evolution_decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = LayerMoveEvolution(target_layer="semantic").run(packet, store)

    assert store.count("semantic") == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["move_style"] == "copy_append"


def test_keyword_count_retrieval_prefers_keyword_hits() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob likes coffee", "Alice studies graphs"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = KeywordCountRetrieval(top_k=2).run(Packet(query=Query(text="Alice graphs")), store)

    assert [record.text for record in packet_out.retrieved.items] == ["Alice studies graphs", "Alice likes tea"]


def test_bm25_retrieval_prefers_stronger_lexical_matches() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    for text in ("graph memory retrieval", "graph retrieval", "tea notes"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="graph memory")), store)

    assert [record.text for record in packet_out.retrieved.items] == ["graph memory retrieval", "graph retrieval"]
    assert packet_out.retrieved.scores[0]["strategy"] == "bm25"
    assert packet_out.retrieved.scores[0]["score"] >= packet_out.retrieved.scores[1]["score"]


def test_bm25_retrieval_breaks_ties_by_recency() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    for text in ("graph memory", "graph memory"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="graph memory")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]


def test_bm25_retrieval_uses_representation_keywords() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="u1",
            layer="default",
            text="notes about tea",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"keywords": ["graph", "memory", "graph"]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="u2",
            layer="default",
            text="plain tea notes",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = BM25Retrieval(top_k=1).run(Packet(query=Query(text="graph memory")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]


def test_bm25_retrieval_on_empty_store_returns_empty_retrieved_set() -> None:
    from memprimitive.baselines import BM25Retrieval

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="Alice")), MemoryStore())

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []


def test_bm25_retrieval_requires_query() -> None:
    from memprimitive.baselines import BM25Retrieval

    with pytest.raises(ValueError, match="packet.query"):
        BM25Retrieval(top_k=2).run(Packet(), MemoryStore())


def test_bm25_retrieval_falls_back_to_recency_when_all_scores_are_zero() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    for text in ("old note", "new note"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="graph memory")), store)

    assert [record.text for record in packet_out.retrieved.items] == ["new note", "old note"]
    assert packet_out.retrieved.trace["used_recency_fallback"] is True
    assert all(score["score"] == 0.0 for score in packet_out.retrieved.scores)


def test_tag_retrieval_prefers_matching_tags() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, AppendOrganization, BasicRepresentation, PassThroughUnitFormation, TagRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Alice studies graph memory", "Bob likes coffee"):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = BasicRepresentation(elements=("text", "tags")).run(packet, store)
        packet, store = AlwaysWriteTrigger().run(packet, store)
        _, store = AppendOrganization().run(packet, store)

    packet_out, _ = TagRetrieval(top_k=1).run(Packet(query=Query(text="graph")), store)

    assert packet_out.retrieved.items[0].text == "Alice studies graph memory"


def test_entity_retrieval_prefers_entity_overlap() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger, AppendOrganization, BasicRepresentation, EntityRetrieval, PassThroughUnitFormation

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob likes coffee", "Alice studies graph memory"):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = BasicRepresentation(elements=("text", "entities")).run(packet, store)
        packet, store = AlwaysWriteTrigger().run(packet, store)
        _, store = AppendOrganization().run(packet, store)

    packet_out, _ = EntityRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), store)

    assert all("Alice" in record.text for record in packet_out.retrieved.items)


def test_layer_aware_retrieval_supports_per_layer_top_k_and_merge_weights() -> None:
    from memprimitive.baselines import KeywordCountRetrieval, LayerAwareRetrieval, RecencyRetrieval

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="working"), StoreLayerSpec(name="semantic")])
    )
    store.append(MemoryRecord(record_id="rec-1", unit_id="u1", layer="working", text="recent working", timestamp="2026-01-01T00:00:00+00:00"))
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="u2",
            layer="semantic",
            text="Alice semantic graph",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"representation": {"keywords": ["alice", "semantic", "graph"]}},
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=2),
        retriever_by_layer={"semantic": KeywordCountRetrieval(top_k=2)},
        top_k=2,
        top_k_by_layer={"working": 1, "semantic": 1},
        merge_weight_by_layer={"semantic": 2.0},
    ).run(Packet(query=Query(text="Alice graph")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]


def test_bullet_list_readout_formats_bullets() -> None:
    from memprimitive.baselines import BulletListReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    retrieved = RetrievedSet(items=store.iter_records(), scores=[])

    packet_out, _ = BulletListReadout().run(Packet(retrieved=retrieved), store)

    assert packet_out.readout.text.startswith("- Alice likes tea.")


def test_grouped_by_layer_readout_groups_items() -> None:
    from memprimitive.baselines import GroupedByLayerReadout

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="working"), StoreLayerSpec(name="semantic")])
    )
    store.append(MemoryRecord(record_id="rec-1", unit_id="u1", layer="working", text="working", timestamp="2026-01-01T00:00:00+00:00"))
    store.append(MemoryRecord(record_id="rec-2", unit_id="u2", layer="semantic", text="semantic", timestamp="2026-01-01T00:00:01+00:00"))

    packet_out, _ = GroupedByLayerReadout().run(Packet(retrieved=RetrievedSet(items=store.iter_records(), scores=[])), store)

    assert "[working]" in packet_out.readout.text
    assert packet_out.readout.metadata["group_counts"] == {"working": 1, "semantic": 1}


def test_json_readout_returns_json_string() -> None:
    from memprimitive.baselines import JSONReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)

    packet_out, _ = JSONReadout().run(Packet(retrieved=RetrievedSet(items=store.iter_records(), scores=[])), store)

    payload = json.loads(packet_out.readout.text)
    assert payload["items"][0]["text"] == "Alice likes tea."
