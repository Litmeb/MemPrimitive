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
    AlwaysTrigger,
    AppendOnlyEvolution,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    BulletListReadout,
    EmbeddingSimilarityRetrieval,
    EntityRetrieval,
    GraphAppendOrganization,
    GraphAppendLinkReadyOrganization,
    GraphLinkEvolution,
    GraphNeighborContextTraceEvolution,
    GraphNeighborRetrieval,
    GraphSeedAndExpandRetrieval,
    LayerAwareRetrieval,
    LinkStrengtheningEvolution,
    NeverTrigger,
    NeighborContextUpdateEvolution,
    PassThroughUnitFormation,
    KeywordRepresentation,
    RecencyRetrieval,
    RetrievalOrientedEmbeddingRepresentation,
    SemanticFieldEnrichmentRepresentation,
    TagRetrieval,
    TripleRepresentation,
    VectorGraphSeedAndExpandRetrieval,
)
from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
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
    assert "Alice likes tea." in readout.text
    assert len(readout.source_ids) == 2


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
    from memprimitive.baselines import AlwaysTrigger

    m = instantiate_default_baseline_modules(top_k=2)
    with pytest.raises(ValueError, match=r"expects ModuleSpec\.slot='evolution_trigger'"):
        MemoryPipeline(
            unit_formation=m["unit_formation"],
            representation=m["representation"],
            write_trigger=m["write_trigger"],
            organization=m["organization"],
            evolution_trigger=AlwaysTrigger(),
            memory_evolution=m["memory_evolution"],
            retrieval=m["retrieval"],
            readout=m["readout"],
        )


def test_default_pipeline_includes_evolution_trigger_trace_and_preserves_behavior() -> None:
    pipeline = create_baseline_pipeline(top_k=2)
    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert "evolution_trigger" in packet.trace
    assert packet.trace["evolution_trigger"]["module"] == "never_evolution_trigger"
    assert packet.trace["evolution_trigger"]["decisions"] == [False]
    assert packet.decisions == [False]
    assert pipeline.store.count() == 1


def test_memory_pipeline_zero_arg_constructor_populates_all_default_modules() -> None:
    pipeline = MemoryPipeline()

    assert isinstance(pipeline.unit_formation, PassThroughUnitFormation)
    assert isinstance(pipeline.representation, BasicRepresentation)
    assert isinstance(pipeline.write_trigger, AlwaysTrigger)
    assert isinstance(pipeline.organization, AppendOrganization)
    assert isinstance(pipeline.evolution_trigger, NeverTrigger)
    assert isinstance(pipeline.memory_evolution, AppendOnlyEvolution)
    assert isinstance(pipeline.retrieval, RecencyRetrieval)
    assert pipeline.retrieval.top_k == 3
    assert isinstance(pipeline.readout, ConcatenateReadout)


def test_memory_pipeline_defaults_all_modules_when_only_ingest_side_overrides_are_provided() -> None:
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(),
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.decisions == [False]
    assert packet.trace["evolution_trigger"]["module"] == "never_evolution_trigger"
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
        AlwaysTrigger,
        AppendOnlyEvolution,
        AppendOrganization,
        BasicRepresentation,
        ConcatenateReadout,
        NeverTrigger,
        PassThroughUnitFormation,
        RecencyRetrieval,
    )

    class PartialEvolutionTrigger(NeverTrigger):
        def run(self, packet: Packet, store):
            packet, store = super().run(packet, store)
            return replace(packet, decisions=[False for _ in packet.units], trace=packet.trace), store

    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(),
        evolution_trigger=PartialEvolutionTrigger(),
        memory_evolution=AppendOnlyEvolution(),
        retrieval=RecencyRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.trace["write_trigger"]["decisions"] == [True]
    assert packet.placements is not None
    assert packet.decisions == [False]
    assert pipeline.store.count() == 1
    assert packet.trace["memory_evolution"]["effects"] == []


def test_default_ingest_writes_before_optional_evolution_runs() -> None:
    pipeline = create_baseline_pipeline(top_k=2)

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.trace["write_trigger"]["decisions"] == [True]
    assert packet.decisions == [False]
    assert pipeline.store.count() == 1
    assert packet.trace["organization"]["written_record_ids"]
    assert packet.trace["memory_evolution"]["active_unit_ids"] == []
    assert packet.trace["memory_evolution"]["effects"] == []


def test_pipeline_supports_unified_trigger_classes_across_write_and_evolution_slots() -> None:
    from memprimitive.baselines import BoundaryEventTrigger, SummaryRewriteEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=BoundaryEventTrigger(accepted_events=("session_end",)),
        evolution_trigger=BoundaryEventTrigger(slot="evolution_trigger", accepted_events=("session_end",)),
        memory_evolution=SummaryRewriteEvolution(target_layer="semantic"),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="Alice likes jasmine tea.",
            source="dialogue",
            metadata={"trigger": {"events": ["session_end"]}},
        )
    )

    assert packet.trace["write_trigger"]["module"] == "boundary_event_write_trigger"
    assert packet.trace["evolution_trigger"]["module"] == "boundary_event_evolution_trigger"
    assert packet.decisions == [True]
    assert pipeline.store.count("default") == 1
    assert pipeline.store.count("semantic") == 1


def test_pipeline_supports_runtime_and_scalar_trigger_combinations() -> None:
    from memprimitive.baselines import RuntimeEventTrigger, ScalarRuleTrigger

    pipeline = MemoryPipeline(
        write_trigger=ScalarRuleTrigger(signal_key="importance", threshold=0.7),
        evolution_trigger=RuntimeEventTrigger(accepted_events=("task_failed",)),
    )

    packet = pipeline.ingest(
        Observation(
            text="Alice likes tea.",
            source="dialogue",
            metadata={"trigger": {"signals": {"importance": 0.9}, "events": ["task_failed"]}},
        )
    )

    assert packet.trace["write_trigger"]["module"] == "scalar_rule_write_trigger"
    assert packet.trace["write_trigger"]["decisions"] == [True]
    assert packet.trace["evolution_trigger"]["module"] == "runtime_event_evolution_trigger"
    assert packet.trace["evolution_trigger"]["decisions"] == [True]


def test_memory_pipeline_allows_graph_organization_without_eager_store_validation() -> None:
    pipeline = MemoryPipeline(organization=GraphAppendOrganization())

    assert isinstance(pipeline.organization, GraphAppendOrganization)


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
        representation=(
            BasicRepresentation(elements=("text",)),
            TripleRepresentation(method="direct"),
            BasicRepresentation(elements=("tags",)),
        ),
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


def test_memory_pipeline_allows_iterable_organization_slot_without_eager_graph_validation() -> None:
    pipeline = MemoryPipeline(organization=[AppendOrganization(), GraphAppendOrganization()])

    assert len(pipeline.organization) == 2


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
        representation=(
            BasicRepresentation(elements=("text",)),
            TripleRepresentation(method="direct"),
            BasicRepresentation(elements=("tags",)),
        ),
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

    pipeline = MemoryPipeline(
        organization=DispatchOrganization(
            (AppendOrganization(), GraphAppendOrganization()),
            primary_index=0,
        )
    )

    assert isinstance(pipeline.organization, DispatchOrganization)


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


def test_memory_pipeline_registers_leaf_module_contracts_on_store() -> None:
    store = MemoryStore()

    MemoryPipeline(
        store=store,
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "tags")),
        retrieval=EntityRetrieval(top_k=2),
    )

    assert "unit.embedding" in store.produced_contracts
    assert "unit.entities" in store.produced_contracts
    assert "unit.tags" in store.produced_contracts
    assert "unit.entities" in store.required_contracts


def test_store_check_fails_when_retrieval_requires_missing_embedding_contract() -> None:
    store = MemoryStore()
    MemoryPipeline(
        store=store,
        representation=KeywordRepresentation(),
        retrieval=EmbeddingSimilarityRetrieval(top_k=2),
    )

    with pytest.raises(IncompatibleCompositionError, match="unit.embedding"):
        store.check()


def test_store_check_accepts_cross_pipeline_shared_store_contract_production() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "vector")),
            ]
        )
    )

    MemoryPipeline(
        store=store,
        representation=(SemanticFieldEnrichmentRepresentation(), RetrievalOrientedEmbeddingRepresentation()),
        organization=GraphAppendLinkReadyOrganization(target_layer="knowledge_graph"),
    )
    MemoryPipeline(
        store=store,
        retrieval=VectorGraphSeedAndExpandRetrieval(layer="knowledge_graph"),
    )

    assert store.check() == frozenset()


def test_store_check_reports_missing_graph_topology_contract() -> None:
    store = MemoryStore()
    MemoryPipeline(store=store, organization=GraphAppendOrganization())

    with pytest.raises(IncompatibleCompositionError, match="topology.graph_layer"):
        store.check()


def test_dispatch_registers_child_contracts_without_double_counting() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "vector")),
            ]
        )
    )

    MemoryPipeline(
        store=store,
        organization=DispatchOrganization(
            (
                AppendOrganization(target_layer="working"),
                GraphAppendOrganization(target_layer="knowledge_graph"),
            )
        ),
    )

    modules = [entry["module"] for entry in store.registered_compositions]
    assert "dispatch_organization" not in modules
    assert modules.count("append_organization") == 1
    assert modules.count("graph_append_organization") == 1


def _register_sparse_pipeline_for_check(
    store: MemoryStore,
    **overrides,
) -> MemoryStore:
    """Build a contract-sparse pipeline so unsupported consumers stay unsupported."""

    kwargs = {
        "unit_formation": PassThroughUnitFormation(),
        "representation": BasicRepresentation(elements=("text",)),
        "write_trigger": AlwaysTrigger(),
        "organization": AppendOrganization(),
        "evolution_trigger": NeverTrigger(),
        "memory_evolution": AppendOnlyEvolution(),
        "retrieval": RecencyRetrieval(top_k=2),
        "readout": ConcatenateReadout(),
    }
    kwargs.update(overrides)
    MemoryPipeline(store=store, **kwargs)
    return store


_OBVIOUSLY_INVALID_SINGLE_SLOT_PIPELINES = (
    pytest.param(
        {"retrieval": EmbeddingSimilarityRetrieval(top_k=2)},
        id="embedding-retrieval-without-embedding-producer",
    ),
    pytest.param(
        {"retrieval": EntityRetrieval(top_k=2)},
        id="entity-retrieval-without-entity-producer",
    ),
    pytest.param(
        {"retrieval": TagRetrieval(top_k=2)},
        id="tag-retrieval-without-tags-or-tag-index",
    ),
    pytest.param(
        {"organization": GraphAppendOrganization()},
        id="graph-organization-without-graph-topology",
    ),
    pytest.param(
        {"organization": GraphAppendLinkReadyOrganization(target_layer="knowledge_graph")},
        id="graph-note-organization-without-note-payload-or-graph-vector-topology",
    ),
    pytest.param(
        {"memory_evolution": GraphLinkEvolution(target_layer="knowledge_graph")},
        id="graph-link-evolution-without-graph-topology",
    ),
    pytest.param(
        {"memory_evolution": GraphNeighborContextTraceEvolution(target_layer="knowledge_graph")},
        id="graph-neighbor-context-without-graph-topology",
    ),
    pytest.param(
        {"memory_evolution": LinkStrengtheningEvolution(target_layer="knowledge_graph")},
        id="link-strengthening-without-note-records-or-graph-vector-topology",
    ),
    pytest.param(
        {"memory_evolution": NeighborContextUpdateEvolution(target_layer="knowledge_graph")},
        id="neighbor-context-update-without-linked-note-records",
    ),
    pytest.param(
        {"retrieval": GraphNeighborRetrieval(layer="knowledge_graph")},
        id="graph-neighbor-retrieval-without-graph-records",
    ),
    pytest.param(
        {"retrieval": GraphSeedAndExpandRetrieval(layer="knowledge_graph")},
        id="graph-seed-expand-retrieval-without-graph-records",
    ),
    pytest.param(
        {"retrieval": VectorGraphSeedAndExpandRetrieval(layer="knowledge_graph")},
        id="vector-graph-retrieval-without-note-records-or-graph-vector-topology",
    ),
)


@pytest.mark.parametrize("overrides", _OBVIOUSLY_INVALID_SINGLE_SLOT_PIPELINES)
def test_store_check_rejects_many_obviously_invalid_single_slot_pipelines(overrides: dict[str, object]) -> None:
    store = _register_sparse_pipeline_for_check(MemoryStore(), **overrides)

    expected_missing = store.required_contracts - store.produced_contracts

    assert expected_missing
    with pytest.raises(IncompatibleCompositionError) as excinfo:
        store.check()

    message = str(excinfo.value)
    for contract in sorted(expected_missing):
        assert contract in message


_OBVIOUSLY_INVALID_COMPOSITE_PIPELINES = (
    pytest.param(
        {
            "organization": GraphAppendOrganization(),
            "memory_evolution": GraphNeighborContextTraceEvolution(target_layer="knowledge_graph"),
            "retrieval": GraphSeedAndExpandRetrieval(layer="knowledge_graph"),
        },
        id="stacked-graph-pipeline-without-graph-topology",
    ),
    pytest.param(
        {
            "organization": GraphAppendLinkReadyOrganization(target_layer="knowledge_graph"),
            "memory_evolution": (
                LinkStrengtheningEvolution(target_layer="knowledge_graph"),
                NeighborContextUpdateEvolution(target_layer="knowledge_graph"),
            ),
            "retrieval": VectorGraphSeedAndExpandRetrieval(layer="knowledge_graph"),
        },
        id="graph-note-pipeline-without-note-representation-or-graph-vector-topology",
    ),
    pytest.param(
        {
            "retrieval": LayerAwareRetrieval(
                default_retriever=EmbeddingSimilarityRetrieval(top_k=2),
                retriever_by_layer={"default": TagRetrieval(top_k=2)},
                top_k=2,
            ),
        },
        id="layer-aware-retrieval-combines-multiple-missing-contracts",
    ),
)


@pytest.mark.parametrize("overrides", _OBVIOUSLY_INVALID_COMPOSITE_PIPELINES)
def test_store_check_rejects_many_obviously_invalid_composite_pipelines(overrides: dict[str, object]) -> None:
    store = _register_sparse_pipeline_for_check(MemoryStore(), **overrides)

    expected_missing = store.required_contracts - store.produced_contracts

    assert expected_missing
    with pytest.raises(IncompatibleCompositionError) as excinfo:
        store.check()

    contracts_meta = store.metadata.get("composition_contracts", {})
    assert sorted(expected_missing) == contracts_meta.get("missing", [])
    message = str(excinfo.value)
    for contract in sorted(expected_missing):
        assert contract in message
