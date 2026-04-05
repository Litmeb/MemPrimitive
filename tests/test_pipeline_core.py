from __future__ import annotations

from dataclasses import replace
import json
import pytest

from memprimitive import (
    Observation,
    Query,
    create_baseline_pipeline,
)
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOnlyEvolution,
    AppendOrganization,
    BasicRepresentation,
    BoundaryEventTrigger,
    ConcatenateReadout,
    BulletListReadout,
    EmbeddingSimilarityRetrieval,
    HierarchicalEvolution,
    LayerAwareRetrieval,
    NeverTrigger,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
)
from memprimitive.core import MemoryStore, ModuleSpec, Packet, StoreLayerSpec, StoreTopology
from memprimitive.interfaces import RetrievalModule
from memprimitive.pipeline import FreeMemoryPipeline, MemoryPipeline

from pipeline_test_helpers import _FreePipelineProbeModule

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


def test_free_memory_pipeline_ingest_runs_only_modules_before_first_retrieval() -> None:
    pipeline = FreeMemoryPipeline(
        modules=(
            _FreePipelineProbeModule(name="write-a", slot="representation", record_text="alpha"),
            _FreePipelineProbeModule(name="write-b", slot="organization", record_text="beta"),
            _FreePipelineProbeModule(name="retrieve-1", slot="retrieval", produce_retrieved=True),
            _FreePipelineProbeModule(name="readout-1", slot="readout", produce_readout=True),
        )
    )

    packet = pipeline.ingest(Observation(text="ignored input", source="notes"))

    assert packet.observation is not None
    assert packet.query is None
    assert pipeline.store.count() == 2
    assert pipeline.store.metadata["free_pipeline_log"] == ["write-a", "write-b"]


def test_free_memory_pipeline_recall_starts_at_first_retrieval_and_runs_remaining_modules_in_order() -> None:
    pipeline = FreeMemoryPipeline(
        modules=(
            _FreePipelineProbeModule(name="write-a", slot="representation", record_text="alpha"),
            _FreePipelineProbeModule(name="write-b", slot="organization", record_text="beta"),
            _FreePipelineProbeModule(name="retrieve-1", slot="retrieval", produce_retrieved=True),
            _FreePipelineProbeModule(name="retrieve-2", slot="retrieval", produce_retrieved=True),
            _FreePipelineProbeModule(name="readout-1", slot="readout", produce_readout=True),
        )
    )

    pipeline.ingest(Observation(text="ignored input", source="notes"))
    readout = pipeline.recall(Query(text="find stored notes"))

    assert readout.text == "alpha | beta"
    assert readout.metadata["free_pipeline_log"] == [
        "write-a",
        "write-b",
        "retrieve-1",
        "retrieve-2",
        "readout-1",
    ]
    assert readout.source_ids == ["rec-1", "rec-2"]


def test_free_memory_pipeline_recall_does_not_run_modules_before_first_retrieval() -> None:
    pipeline = FreeMemoryPipeline(
        modules=(
            _FreePipelineProbeModule(name="write-a", slot="representation", record_text="alpha"),
            _FreePipelineProbeModule(name="write-b", slot="organization", record_text="beta"),
            _FreePipelineProbeModule(name="retrieve-1", slot="retrieval", produce_retrieved=True),
            _FreePipelineProbeModule(name="readout-1", slot="readout", produce_readout=True),
        )
    )

    readout = pipeline.recall(Query(text="find stored notes"))

    assert readout.text == ""
    assert pipeline.store.count() == 0
    assert pipeline.store.metadata["free_pipeline_log"] == ["retrieve-1", "readout-1"]


def test_free_memory_pipeline_run_round_preserves_ingest_trace_metadata() -> None:
    pipeline = FreeMemoryPipeline(
        modules=(
            _FreePipelineProbeModule(name="write-a", slot="representation", record_text="alpha"),
            _FreePipelineProbeModule(name="retrieve-1", slot="retrieval", produce_retrieved=True),
            _FreePipelineProbeModule(name="readout-1", slot="readout", produce_readout=True),
        )
    )

    readout = pipeline.run_round(
        Observation(text="ignored input", source="notes"),
        Query(text="find stored notes"),
    )

    assert readout.text == "alpha"
    assert readout.metadata["ingest_trace"]["ingest_started"] is True


def test_free_memory_pipeline_skips_store_contract_registration_and_validate_store_hooks() -> None:
    class ValidateStoreProbe(_FreePipelineProbeModule):
        def validate_store(self, store: MemoryStore) -> None:
            store.metadata["validate_store_called"] = True

    store = MemoryStore()
    pipeline = FreeMemoryPipeline(
        modules=(
            ValidateStoreProbe(name="write-a", slot="representation", record_text="alpha"),
            _FreePipelineProbeModule(name="retrieve-1", slot="retrieval", produce_retrieved=True),
            _FreePipelineProbeModule(name="readout-1", slot="readout", produce_readout=True),
        ),
        store=store,
    )

    pipeline.ingest(Observation(text="ignored input", source="notes"))

    assert store.metadata.get("validate_store_called") is None
    assert store.registered_compositions == ()


def test_free_memory_pipeline_rejects_empty_module_iterable() -> None:
    with pytest.raises(ValueError, match="must contain at least one module"):
        FreeMemoryPipeline(modules=())


def test_free_memory_pipeline_requires_a_retrieval_boundary() -> None:
    with pytest.raises(ValueError, match="spec.slot='retrieval'"):
        FreeMemoryPipeline(
            modules=(
                _FreePipelineProbeModule(name="write-a", slot="representation", record_text="alpha"),
                _FreePipelineProbeModule(name="write-b", slot="organization", record_text="beta"),
            )
        )


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


def test_hierarchical_session_summary_pipeline_merges_global_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime

    def _fake_embed_text(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float("graph" in lowered) + (0.4 if "retrieval" in lowered else 0.0),
            float("trip" in lowered) + float("hotel" in lowered) + float("train" in lowered),
            1.0 if "summary" in lowered else 0.2,
        ]

    class _FakeHierarchicalRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def json(self, *, system: str, user: str):
            payload = json.loads(user)
            group_key = payload.get("group_key", {})
            session_id = group_key.get("session_id", "unknown")
            texts = [record.get("text", "") for record in payload.get("records", [])]
            return {"summary": f"summary for {session_id}: {' '.join(texts)}"}

    monkeypatch.setattr(BasicRepresentation, "_embed_text", _fake_embed_text)
    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", _fake_embed_text)
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeHierarchicalRuntime())

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="episodic", theme="session_memory", indices=("temporal", "vector")),
                StoreLayerSpec(name="session_summary", theme="semantic", indices=("temporal", "vector")),
            ]
        )
    )
    turn_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "embedding")),
        organization=AppendOrganization(target_layer="episodic"),
        store=store,
    )
    session_close_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=AppendOrganization(target_layer="episodic"),
        evolution_trigger=BoundaryEventTrigger(
            slot="evolution_trigger",
            accepted_events=("session_end",),
        ),
        memory_evolution=HierarchicalEvolution(
            source_layer="episodic",
            extract_mode="generate",
            extract_fields=("summary",),
            group_by=("session_id",),
            target_layer="session_summary",
        ),
        store=store,
    )
    recall_pipeline = MemoryPipeline(
        retrieval=LayerAwareRetrieval(
            default_retriever=EmbeddingSimilarityRetrieval(top_k=3),
            retriever_by_layer={"session_summary": EmbeddingSimilarityRetrieval(top_k=3)},
            active_layers=("session_summary", "episodic"),
            top_k=3,
            top_k_by_layer={"session_summary": 3, "episodic": 3},
        ),
        readout=BulletListReadout(),
        store=store,
    )

    observations = [
        Observation(
            text="Alice is debugging graph retrieval.",
            source="dialogue",
            metadata={"session_id": "sess-1"},
        ),
        Observation(
            text="She wants a clean session summary for retrieval work.",
            source="dialogue",
            metadata={"session_id": "sess-1"},
        ),
        Observation(
            text="Bob is comparing hotel and train options.",
            source="dialogue",
            metadata={"session_id": "sess-2"},
        ),
        Observation(
            text="He is planning a weekend trip.",
            source="dialogue",
            metadata={"session_id": "sess-2"},
        ),
    ]
    for observation in observations:
        turn_pipeline.ingest(observation)

    session_close_pipeline.ingest(
        Observation(
            text="close session sess-1",
            source="session_controller",
            metadata={"session_id": "sess-1", "trigger": {"events": ["session_end"]}},
        )
    )
    session_close_pipeline.ingest(
        Observation(
            text="close session sess-2",
            source="session_controller",
            metadata={"session_id": "sess-2", "trigger": {"events": ["session_end"]}},
        )
    )

    assert store.count("session_summary") == 2
    summary_records = store.iter_records("session_summary")
    assert summary_records[0].metadata["hierarchical"]["group_key"] == {"session_id": "sess-1"}
    assert summary_records[1].metadata["hierarchical"]["group_key"] == {"session_id": "sess-2"}

    query = Query(text="graph retrieval session summary")
    packet, _ = recall_pipeline.retrieval.run(Packet(query=query), recall_pipeline.store)

    assert packet.retrieved is not None
    assert len(packet.retrieved.items) == 3
    assert len(packet.retrieved.scores) == 3
    assert packet.retrieved.scores[0]["merge_rank"] == 1
    assert packet.retrieved.scores[-1]["merge_rank"] == 3
    assert "session_summary" in {record.layer for record in packet.retrieved.items}
    assert packet.trace["retrieval"]["final_returned_count"] == 3
    assert packet.trace["retrieval"]["active_layers"] == ["session_summary", "episodic"]


