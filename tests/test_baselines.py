from __future__ import annotations

import pytest

from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
    registered_baseline_class_names,
)
from memprimitive.core import MemoryStore, Observation, Packet, Query, RetrievedSet, StoreLayerSpec, StoreTopology
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


def test_representation_can_generate_real_description_via_api() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

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
        AlwaysEvolutionTrigger,
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
    packet, store = AppendOrganization().run(packet, store)

    packet_out, _ = AlwaysEvolutionTrigger().run(packet, store)

    assert packet_out.evolution_decisions == [True]
    assert packet_out.trace["evolution_trigger"]["policy"] == "always"
    assert packet_out.trace["evolution_trigger"]["scorer"] == "identity"
    assert packet_out.trace["evolution_trigger"]["evolution_decisions"] == [True]
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["signals"] == {"constant": 1.0}
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["score"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["gate"] is True
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is True


def test_organization_aligns_placements_with_units() -> None:
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

    packet_out, _ = AppendOrganization().run(packet, store)

    assert packet_out.placements is not None
    assert len(packet_out.placements) == len(packet_out.units)
    assert packet_out.placements[0].target_layer == "default"


def test_append_only_evolution_mutates_store_only_for_true_decisions() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        decisions=[False],
        placements=packet.placements,
        trace=packet.trace,
    )

    _, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 0


def test_append_only_evolution_prefers_evolution_decisions_over_decisions() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        decisions=[True],
        evolution_decisions=[False],
        placements=packet.placements,
        trace=packet.trace,
    )

    _, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 0


def test_append_only_evolution_falls_back_to_decisions_when_evolution_decisions_missing() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())

    _, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 1


def test_append_only_evolution_requires_aligned_inputs() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    with pytest.raises(ValueError, match="aligned units"):
        AppendOnlyEvolution().run(
            Packet(units=[], decisions=[True], placements=[]),
            MemoryStore(),
        )


def test_write_and_evolution_trigger_share_observable_mask_behavior() -> None:
    from memprimitive.baselines import (
        AlwaysEvolutionTrigger,
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
    write_packet, store = AlwaysWriteTrigger().run(packet, store)
    write_packet, store = AppendOrganization().run(write_packet, store)
    evolution_packet, _ = AlwaysEvolutionTrigger().run(write_packet, store)

    assert write_packet.decisions == evolution_packet.evolution_decisions
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
    from memprimitive.baselines import AppendOnlyEvolution, RecencyRetrieval

    store = MemoryStore()
    evolution = AppendOnlyEvolution()
    for text in ("one", "two", "three"):
        packet, store = _stored_pipeline_packet(text, store)
        _, store = evolution.run(packet, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="items")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2


def test_retrieval_rejects_non_positive_top_k() -> None:
    from memprimitive.baselines import RecencyRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        RecencyRetrieval(top_k=0)


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
    from memprimitive.baselines import AppendOnlyEvolution, ConcatenateReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    _, store = AppendOnlyEvolution().run(packet, store)
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
    from memprimitive.baselines import AppendOnlyEvolution, RecencyRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob prefers coffee", "Alice studies graphs"):
        packet, store = _stored_pipeline_packet(text, store)
        _, store = AppendOnlyEvolution().run(packet, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2
    assert all("alice" in record.text.casefold() for record in packet_out.retrieved.items)


def test_retrieval_returns_latest_records_first_when_falling_back_to_recency() -> None:
    from memprimitive.baselines import AppendOnlyEvolution, RecencyRetrieval

    store = MemoryStore()
    for text in ("first item", "second item", "third item"):
        packet, store = _stored_pipeline_packet(text, store)
        _, store = AppendOnlyEvolution().run(packet, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="unmatched")), store)

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["third item", "second item"]


def test_retrieval_does_not_mutate_store() -> None:
    from memprimitive.baselines import AppendOnlyEvolution, RecencyRetrieval

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea", store)
    _, store = AppendOnlyEvolution().run(packet, store)
    before_ids = [record.record_id for record in store.iter_records()]

    _, store_after = RecencyRetrieval(top_k=1).run(Packet(query=Query(text="Alice")), store)

    assert [record.record_id for record in store_after.iter_records()] == before_ids


def test_append_only_evolution_can_write_into_declared_non_default_topology_layer() -> None:
    from memprimitive.baselines import AppendOnlyEvolution, AppendOrganization

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
        ]
    )
    store = MemoryStore(topology=topology)
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    packet, store = AppendOrganization(target_layer="episodic").run(packet, store)

    _, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count("episodic") == 1
    assert updated_store.iter_records("episodic")[0].layer == "episodic"


def test_retrieval_can_target_declared_topology_layer() -> None:
    from memprimitive.baselines import AppendOnlyEvolution, AppendOrganization, RecencyRetrieval

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
        _, store = AppendOnlyEvolution().run(packet, store)

    packet_out, _ = RecencyRetrieval(top_k=1, layer="episodic").run(Packet(query=Query(text="episodic")), store)

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["episodic second"]


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
