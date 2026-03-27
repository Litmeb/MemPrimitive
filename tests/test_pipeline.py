from __future__ import annotations

import pytest

from memprimitive import (
    DispatchOrganization,
    DispatchReadout,
    IncompatibleCompositionError,
    Observation,
    Query,
    create_baseline_pipeline,
)
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOnlyEvolution,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    BulletListReadout,
    EntityRetrieval,
    GraphAppendOrganization,
    LayerAwareRetrieval,
    NeverEvolutionTrigger,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
    iter_baseline_pipeline_instances,
)
from memprimitive.core import MemoryStore, ModuleSpec, Packet, StoreLayerSpec, StoreTopology
from memprimitive.interfaces import RetrievalModule
from memprimitive.pipeline import MemoryPipeline


def test_ingesting_observations_then_recalling_query_produces_non_empty_readout() -> None:
    pipeline = create_baseline_pipeline(top_k=2)
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob likes coffee.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert readout.text
    assert len(readout.source_ids) == 1


def test_full_baseline_pipeline_preserves_trace_fields_across_ingest_stages() -> None:
    pipeline = create_baseline_pipeline(top_k=1)

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    for slot in MemoryPipeline.INGEST_SLOTS:
        assert slot in packet.trace, f"missing trace key for {slot}"


def test_repeated_ingests_accumulate_records_in_store() -> None:
    pipeline = create_baseline_pipeline(top_k=2)
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob likes coffee.", source="dialogue"))
    pipeline.ingest(Observation(text="Charlie likes cocoa.", source="dialogue"))

    assert pipeline.store.count() == 3


def test_create_baseline_pipeline_keeps_recency_retrieval_as_default() -> None:
    pipeline = create_baseline_pipeline(top_k=2)

    assert isinstance(pipeline.retrieval, RecencyRetrieval)
    assert pipeline.retrieval.top_k == 2


def test_round_trip_demo_scenario_works_with_baseline_pipeline() -> None:
    pipeline = create_baseline_pipeline(top_k=2)

    readout = pipeline.run_round(
        Observation(text="Alice started learning graph memory systems.", source="notes"),
        Query(text="Alice"),
    )

    assert "Alice" in readout.text
    assert readout.metadata["item_count"] >= 1
    assert "ingest_trace" in readout.metadata


def test_memory_pipeline_rejects_wrong_abstract_type_at_slot() -> None:
    """Composition rules: each kwarg must match the expected primitive ABC."""
    m = instantiate_default_baseline_modules(top_k=2)
    with pytest.raises(TypeError, match="readout"):
        MemoryPipeline(
            unit_formation=m["unit_formation"],
            representation=m["representation"],
            write_trigger=m["write_trigger"],
            organization=m["organization"],
            evolution_trigger=m["evolution_trigger"],
            memory_evolution=m["memory_evolution"],
            retrieval=m["retrieval"],
            readout=m["retrieval"],
        )


def test_memory_pipeline_rejects_wrong_module_spec_slot() -> None:
    """Even with the correct ABC, ModuleSpec.slot must match the pipeline position."""

    class MislabeledRetrieval(RetrievalModule):
        spec = ModuleSpec(name="mislabeled", slot="readout")

        def run(self, packet: Packet, store):
            return packet, store

    m = instantiate_default_baseline_modules(top_k=2)
    with pytest.raises(ValueError, match=r"expects ModuleSpec\.slot='retrieval'"):
        MemoryPipeline(
            unit_formation=m["unit_formation"],
            representation=m["representation"],
            write_trigger=m["write_trigger"],
            organization=m["organization"],
            evolution_trigger=m["evolution_trigger"],
            memory_evolution=m["memory_evolution"],
            retrieval=MislabeledRetrieval(),
            readout=m["readout"],
        )


def test_memory_pipeline_rejects_write_trigger_instance_in_evolution_trigger_slot() -> None:
    from memprimitive.baselines import AlwaysWriteTrigger

    m = instantiate_default_baseline_modules(top_k=2)
    with pytest.raises(TypeError, match="EvolutionTriggerModule"):
        MemoryPipeline(
            unit_formation=m["unit_formation"],
            representation=m["representation"],
            write_trigger=m["write_trigger"],
            organization=m["organization"],
            evolution_trigger=AlwaysWriteTrigger(),
            memory_evolution=m["memory_evolution"],
            retrieval=m["retrieval"],
            readout=m["readout"],
        )


def test_default_pipeline_includes_evolution_trigger_trace_and_preserves_behavior() -> None:
    pipeline = create_baseline_pipeline(top_k=2)
    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert "evolution_trigger" in packet.trace
    assert packet.trace["evolution_trigger"]["policy"] == "never"
    assert packet.evolution_decisions == [False]
    assert pipeline.store.count() == 1


def test_memory_pipeline_zero_arg_constructor_populates_all_default_modules() -> None:
    pipeline = MemoryPipeline()

    assert isinstance(pipeline.unit_formation, PassThroughUnitFormation)
    assert isinstance(pipeline.representation, BasicRepresentation)
    assert isinstance(pipeline.write_trigger, AlwaysWriteTrigger)
    assert isinstance(pipeline.organization, AppendOrganization)
    assert isinstance(pipeline.evolution_trigger, NeverEvolutionTrigger)
    assert isinstance(pipeline.memory_evolution, AppendOnlyEvolution)
    assert isinstance(pipeline.retrieval, RecencyRetrieval)
    assert pipeline.retrieval.top_k == 3
    assert isinstance(pipeline.readout, ConcatenateReadout)


def test_memory_pipeline_defaults_all_modules_when_only_ingest_side_overrides_are_provided() -> None:
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(),
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.evolution_decisions == [False]
    assert packet.trace["evolution_trigger"]["policy"] == "never"
    assert pipeline.store.count() == 1


def test_memory_pipeline_defaults_ingest_side_when_only_recall_side_overrides_are_provided() -> None:
    pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert "Alice" in readout.text
    assert readout.source_ids


def test_pipeline_can_mask_evolution_without_blocking_write_path_organization() -> None:
    from dataclasses import replace

    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        AppendOnlyEvolution,
        AppendOrganization,
        BasicRepresentation,
        ConcatenateReadout,
        NeverEvolutionTrigger,
        PassThroughUnitFormation,
        RecencyRetrieval,
    )

    class PartialEvolutionTrigger(NeverEvolutionTrigger):
        def run(self, packet: Packet, store):
            packet, store = super().run(packet, store)
            return replace(packet, evolution_decisions=[False for _ in packet.units], trace=packet.trace), store

    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(),
        evolution_trigger=PartialEvolutionTrigger(),
        memory_evolution=AppendOnlyEvolution(),
        retrieval=RecencyRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.decisions == [True]
    assert packet.placements is not None
    assert packet.evolution_decisions == [False]
    assert pipeline.store.count() == 1
    assert packet.trace["memory_evolution"]["effects"] == []


def test_default_ingest_writes_before_optional_evolution_runs() -> None:
    pipeline = create_baseline_pipeline(top_k=2)

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.decisions == [True]
    assert packet.evolution_decisions == [False]
    assert pipeline.store.count() == 1
    assert packet.trace["organization"]["written_record_ids"]
    assert packet.trace["memory_evolution"]["active_unit_ids"] == []
    assert packet.trace["memory_evolution"]["effects"] == []


def test_pipeline_iteration_surfaces_incompatible_graph_layer_compositions() -> None:
    """Iteration remains strict and should expose composition errors distinctly."""
    iterator = iter_baseline_pipeline_instances(top_k=2)

    successful_pipelines = 0
    with pytest.raises(IncompatibleCompositionError, match=r"slot='(organization|retrieval|memory_evolution)'.*graph"):
        for pipeline in iterator:
            successful_pipelines += 1
            pipeline.ingest(Observation(text="combinatorial ingest.", source="dialogue"))
            readout = pipeline.recall(Query(text="combinatorial"))
            assert isinstance(readout.text, str)
            assert pipeline.store.count() >= 1

    assert successful_pipelines >= 1


def test_memory_pipeline_rejects_graph_organization_without_declared_graph_layer() -> None:
    with pytest.raises(IncompatibleCompositionError, match=r"slot='organization'.*declared graph layer.*knowledge_graph"):
        MemoryPipeline(organization=GraphAppendOrganization())


def test_memory_pipeline_accepts_graph_organization_with_compatible_store_topology() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    pipeline = MemoryPipeline(store=store, organization=GraphAppendOrganization())

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="notes"))

    assert packet.trace["organization"]["target_layer"] == "knowledge_graph"
    assert pipeline.store.count("knowledge_graph") == 1


def test_memory_pipeline_accepts_iterable_slot_modules_and_runs_them_in_order() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "entities", "triple", "tags")),
        organization=(
            AppendOrganization(target_layer="working"),
            GraphAppendOrganization(target_layer="knowledge_graph"),
        ),
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=2),
            retriever_by_layer={"knowledge_graph": EntityRetrieval(top_k=2)},
            top_k=3,
        ),
        readout=(ConcatenateReadout(separator="\n\n"), BulletListReadout()),
        store=store,
    )

    pipeline.ingest(Observation(text="Alice likes tea.", source="notes"))
    readout = pipeline.recall(Query(text="Alice"))

    assert store.count("working") == 1
    assert store.count("knowledge_graph") == 1
    assert readout.text.startswith("- ")


def test_memory_pipeline_rejects_empty_iterable_slot() -> None:
    with pytest.raises(ValueError, match="slot iterables must contain at least one module"):
        MemoryPipeline(organization=[])


def test_memory_pipeline_validates_each_module_inside_iterable_slot() -> None:
    with pytest.raises(TypeError, match="OrganizationModule"):
        MemoryPipeline(organization=[AppendOrganization(), object()])


def test_memory_pipeline_checks_graph_compatibility_for_iterable_organization_slot() -> None:
    with pytest.raises(IncompatibleCompositionError, match=r"slot='organization'.*declared graph layer.*knowledge_graph"):
        MemoryPipeline(organization=[AppendOrganization(), GraphAppendOrganization()])


def test_dispatch_organization_fans_out_same_snapshot_and_keeps_primary_packet() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "entities", "triple", "tags")),
        organization=DispatchOrganization(
            (
                AppendOrganization(target_layer="working"),
                GraphAppendOrganization(target_layer="knowledge_graph"),
            ),
            primary_index=0,
        ),
        store=store,
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="notes"))

    assert store.count("working") == 1
    assert store.count("knowledge_graph") == 1
    assert packet.placements[0].target_layer == "working"
    assert packet.trace["dispatch"]["organization"]["children"][1]["module"] == "graph_append_organization"


def test_dispatch_readout_returns_primary_branch_but_records_all_children() -> None:
    store = MemoryStore()
    pipeline = create_baseline_pipeline(top_k=2)
    pipeline.store = store
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.readout = DispatchReadout((ConcatenateReadout(), BulletListReadout()), primary_index=1)

    readout = pipeline.recall(Query(text="Alice"))

    assert readout.text.startswith("- ")


def test_dispatch_organization_validates_child_slots_and_graph_compatibility() -> None:
    with pytest.raises(TypeError, match="OrganizationModule"):
        DispatchOrganization((AppendOrganization(), ConcatenateReadout()))

    with pytest.raises(IncompatibleCompositionError, match=r"slot='organization'.*declared graph layer.*knowledge_graph"):
        MemoryPipeline(
            organization=DispatchOrganization(
                (AppendOrganization(), GraphAppendOrganization()),
                primary_index=0,
            )
        )


def test_pipeline_accepts_custom_topology_store_without_breaking_baseline_flow() -> None:
    modules = instantiate_default_baseline_modules(top_k=2)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
            ]
        )
    )
    pipeline = MemoryPipeline(store=store, **modules)

    pipeline.ingest(Observation(text="Alice likes topology-aware stores.", source="dialogue"))
    readout = pipeline.recall(Query(text="topology"))

    assert readout.text
    assert pipeline.store.topology.layer_count == 2
    assert pipeline.store.count("default") == 1
    assert pipeline.store.count("episodic") == 0
