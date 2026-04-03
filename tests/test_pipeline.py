from __future__ import annotations

import json
import pytest

from memprimitive import (
    DispatchEvolutionTrigger,
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
    BoundaryEventTrigger,
    BufferRetrieval,
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
    HierarchicalEvolution,
    LayerAwareRetrieval,
    LinkStrengtheningEvolution,
    LLMRepresentation,
    ModelJudgeTrigger,
    NeverTrigger,
    NeighborContextUpdateEvolution,
    PassThroughUnitFormation,
    PromptContextReadout,
    KeywordRepresentation,
    RecencyRetrieval,
    ReflectionGenerationEvolution,
    RetrievalOrientedEmbeddingRepresentation,
    SemanticFieldEnrichmentRepresentation,
    StoreAllTrigger,
    TagRetrieval,
    TripleRepresentation,
    VectorGraphSeedAndExpandRetrieval,
)
from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
)
from memprimitive.core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Packet, StoreLayerSpec, StoreTopology
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
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior default memory", metadata={"session_id": "sess-1"}),
            layer="default",
            sequence_id=store.next_sequence_id(),
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior semantic memory", metadata={"session_id": "sess-1"}),
            layer="semantic",
            sequence_id=store.next_sequence_id(),
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
            metadata={"session_id": "sess-1", "trigger": {"events": ["session_end"], "session_id": "sess-1"}},
        )
    )

    assert packet.trace["write_trigger"]["module"] == "boundary_event_write_trigger"
    assert packet.trace["evolution_trigger"]["module"] == "boundary_event_evolution_trigger"
    assert packet.decisions == [True]
    assert packet.decisions_store is not None
    assert packet.trace["write_trigger"]["decisions_store_counts"] == {"default": 1, "semantic": 1}
    assert packet.trace["evolution_trigger"]["decisions_store_counts"] == {"default": 2, "semantic": 1}
    assert pipeline.store.count("default") == 2
    assert pipeline.store.count("semantic") == 2


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


def test_pipeline_boundary_trigger_can_feed_hierarchical_evolution() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="profile", theme="semantic"),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(
                text="prior session note",
                metadata={"session_id": "sess-1", "doc_id": "doc-a", "subgoal_id": "sg-1"},
            ),
            layer="default",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=BoundaryEventTrigger(accepted_events=("session_end",)),
        organization=AppendOrganization(target_layer="default"),
        evolution_trigger=BoundaryEventTrigger(slot="evolution_trigger", accepted_events=("session_end",)),
        memory_evolution=HierarchicalEvolution(
            source_layer="default",
            extract_mode="copy",
            extract_fields=("doc_id", "subgoal_id"),
            group_by=("session_id",),
            target_layer="profile",
        ),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="new session note",
            source="dialogue",
            metadata={
                "session_id": "sess-1",
                "doc_id": "doc-b",
                "subgoal_id": "sg-2",
                "trigger": {"events": ["session_end"], "session_id": "sess-1"},
            },
        )
    )

    assert packet.decisions_store is not None
    assert packet.trace["evolution_trigger"]["decisions_store_counts"] == {"default": 2}
    assert packet.trace["memory_evolution"]["module"] == "hierarchical_evolution"
    assert packet.trace["memory_evolution"]["group_count"] == 1
    assert packet.trace["memory_evolution"]["write_mode"] == "memory_pipeline_ingest"
    assert packet.trace["memory_evolution"]["writer_pipeline_mode"] == "default_target_layer"
    assert pipeline.store.count("profile") == 1
    profile_record = pipeline.store.iter_records("profile")[0]
    assert profile_record.metadata["hierarchical"]["source_record_ids"] == ["rec-1", "rec-2"]
    assert profile_record.metadata["hierarchical"]["field_payload"]["doc_id"] == ["doc-a", "doc-b"]
    assert profile_record.metadata["hierarchical"]["group_key"] == {"session_id": "sess-1"}


def test_pipeline_hierarchical_evolution_can_write_through_provided_child_pipeline() -> None:
    child_pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="profile"),
        store=MemoryStore(),
    )
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="profile", theme="semantic"),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(
                text="prior session note",
                metadata={"session_id": "sess-1", "doc_id": "doc-a"},
            ),
            layer="default",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=BoundaryEventTrigger(accepted_events=("session_end",)),
        organization=AppendOrganization(target_layer="default"),
        evolution_trigger=BoundaryEventTrigger(slot="evolution_trigger", accepted_events=("session_end",)),
        memory_evolution=HierarchicalEvolution(
            source_layer="default",
            extract_mode="copy",
            extract_fields=("doc_id",),
            memory_pipeline=child_pipeline,
        ),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="new session note",
            source="dialogue",
            metadata={
                "session_id": "sess-1",
                "doc_id": "doc-b",
                "trigger": {"events": ["session_end"], "session_id": "sess-1"},
            },
        )
    )

    assert child_pipeline.store is pipeline.store
    assert packet.trace["memory_evolution"]["writer_pipeline_mode"] == "provided"
    assert pipeline.store.count("profile") == 1
    profile_record = pipeline.store.iter_records("profile")[0]
    assert profile_record.metadata["hierarchical"]["field_payload"]["doc_id"] == ["doc-a", "doc-b"]
    assert packet.trace["memory_evolution"]["effects"][0]["sub_ingest_trace"]["organization"]["target_layer"] == "profile"


def test_pipeline_runtime_memory_pressure_records_decisions_store_summary() -> None:
    from memprimitive.baselines import RuntimeEventTrigger, ScalarRuleTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic", settings={"record_budget": 2}),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior one"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior two"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=ScalarRuleTrigger(signal_key="importance", threshold=0.7),
        organization=AppendOrganization(target_layer="episodic"),
        evolution_trigger=RuntimeEventTrigger(accepted_events=("memory_pressure",), pressure_threshold=1.0),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="Alice likes tea.",
            source="dialogue",
            metadata={"trigger": {"signals": {"importance": 0.9}}},
        )
    )

    assert packet.trace["evolution_trigger"]["module"] == "runtime_event_evolution_trigger"
    assert packet.trace["evolution_trigger"]["matched_events"] == ["memory_pressure"]
    assert packet.trace["evolution_trigger"]["decisions_store_counts"] == {"episodic": 3}
    assert packet.decisions_store is not None
    assert packet.decisions_store["episodic"]["selector"]["kind"] == "layer_all"


def test_pipeline_scalar_memory_pressure_records_decisions_store_summary() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic", settings={"record_budget": 2}),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior one"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior two"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=ScalarRuleTrigger(signal_key="memory_pressure", threshold=1.0, target_layer="episodic"),
        organization=AppendOrganization(target_layer="episodic"),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="Alice likes tea.",
            source="dialogue",
        )
    )

    assert packet.trace["write_trigger"]["module"] == "scalar_rule_write_trigger"
    assert packet.trace["write_trigger"]["decisions_store_counts"] == {"episodic": 2}
    assert packet.decisions_store is not None
    assert packet.decisions_store["episodic"]["selector"]["kind"] == "layer_all"
    assert packet.decisions_store["episodic"]["selector"]["source"] == "scalar_rule"


def test_pipeline_end_to_end_supports_model_judge_trigger_with_reflection_generation() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="reflections", theme="semantic"),
            ]
        )
    )

    def judge_failure(payload: dict[str, object]) -> dict[str, object]:
        observation = payload.get("observation", {})
        metadata = observation.get("metadata", {}) if isinstance(observation, dict) else {}
        reflexion = metadata.get("reflexion", {}) if isinstance(metadata, dict) else {}
        return {
            "decision": False,
            "score": 0.92,
            "label": "trigger",
            "reason": reflexion.get("evaluator_feedback", ""),
        }

    def generate_reflection(payload) -> str:
        return (
            "Reflection: verify the boundary condition first, then cross-check the returned index "
            "against the query before finalizing the answer."
        )

    pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        evolution_trigger=ModelJudgeTrigger(
            slot="evolution_trigger",
            system_prompt="Decide whether this failed trial needs reflection.",
            decision_mode="score",
            threshold=0.8,
            per_unit=False,
            judge_callable=judge_failure,
        ),
        memory_evolution=ReflectionGenerationEvolution(
            target_layer="reflections",
            memory_size=2,
            reflection_generator=generate_reflection,
        ),
        retrieval=BufferRetrieval(top_k=2, layer="reflections"),
        readout=PromptContextReadout(
            memory_layer="reflections",
            default_strategy="last_trial_and_reflexion",
            top_k=2,
        ),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="Tried scanning from the second element and returned index 3, but the expected answer was 2.",
            source="reasoning_trial",
            metadata={
                "reflexion": {
                    "question": "Find the first matching index in the stream.",
                    "last_attempt": "I started from position 1 and skipped the first candidate.",
                    "scratchpad": "I started from position 1 and skipped the first candidate.",
                    "evaluator_feedback": "You ignored the earliest valid match.",
                    "trial_index": 2,
                }
            },
        )
    )

    readout = pipeline.recall(
        Query(
            text="Find the first matching index in the stream.",
            metadata={
                "reflexion": {
                    "strategy": "last_trial_and_reflexion",
                    "last_attempt": "I started from position 1 and skipped the first candidate.",
                }
            },
        )
    )

    assert packet.trace["evolution_trigger"]["module"] == "model_judge_evolution_trigger"
    assert packet.trace["evolution_trigger"]["decisions"] == [True]
    assert packet.trace["evolution_trigger"]["judge_per_unit"] is False
    assert packet.trace["evolution_trigger"]["per_unit"][0]["score"] == 0.92
    assert packet.trace["memory_evolution"]["module"] == "reflection_generation_evolution"
    assert packet.trace["memory_evolution"]["generation_mode"] == "callable_override"
    assert packet.trace["memory_evolution"]["record_ids"]
    assert packet.trace["memory_evolution"]["effects"][0]["effect_type"] == "reflection_append"
    assert pipeline.store.count("default") == 1
    assert pipeline.store.count("reflections") == 1

    reflection_record = pipeline.store.iter_records("reflections")[0]
    assert reflection_record.metadata["reflection"]["source_layer"] == "default"
    assert reflection_record.metadata["reflection"]["trial_index"] == 2
    assert "verify the boundary condition first" in reflection_record.text

    assert "Below is the last trial you attempted" in readout.text
    assert "Reflection 1:" in readout.text
    assert "verify the boundary condition first" in readout.text
    assert readout.source_ids == [reflection_record.record_id]
    assert readout.metadata["reflection_count"] == 1


def test_pipeline_model_judge_can_block_reflection_generation_end_to_end() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="reflections", theme="semantic"),
            ]
        )
    )

    pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        evolution_trigger=ModelJudgeTrigger(
            slot="evolution_trigger",
            system_prompt="Decide whether reflection is needed.",
            decision_mode="score",
            threshold=0.8,
            per_unit=False,
            judge_callable=lambda payload: {"score": 0.2, "label": "skip"},
        ),
        memory_evolution=ReflectionGenerationEvolution(
            target_layer="reflections",
            reflection_generator=lambda payload: "Reflection: this should not be written.",
        ),
        retrieval=BufferRetrieval(top_k=2, layer="reflections"),
        readout=PromptContextReadout(memory_layer="reflections", default_strategy="reflexion", top_k=2),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(
            text="Attempted answer still failed.",
            source="reasoning_trial",
            metadata={
                "reflexion": {
                    "question": "Recover the first valid index.",
                    "last_attempt": "I guessed without checking the earliest candidate.",
                    "evaluator_feedback": "The attempt was weak, but we do not want to store a reflection yet.",
                    "trial_index": 1,
                }
            },
        )
    )
    readout = pipeline.recall(Query(text="Recover the first valid index."))

    assert packet.trace["evolution_trigger"]["module"] == "model_judge_evolution_trigger"
    assert packet.trace["evolution_trigger"]["decisions"] == [False]
    assert packet.trace["memory_evolution"]["active_unit_ids"] == []
    assert packet.trace["memory_evolution"]["effects"] == []
    assert packet.trace["memory_evolution"]["record_ids"] == []
    assert pipeline.store.count("default") == 1
    assert pipeline.store.count("reflections") == 0
    assert readout.text == ""
    assert readout.source_ids == []
    assert readout.metadata["reflection_count"] == 0


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


def test_memory_pipeline_accepts_iterable_llm_representation_modules() -> None:
    store = MemoryStore()
    tag_rep = LLMRepresentation(field="tags", prompt="Extract tags.")
    summary_rep = LLMRepresentation(field="summary", prompt="Extract summary.")
    tag_rep._llm_json = lambda *, user: ["graph", "memory"]  # type: ignore[method-assign]
    summary_rep._llm_text = lambda *, user: "Alice studies graph memory."  # type: ignore[method-assign]

    pipeline = MemoryPipeline(
        representation=(tag_rep, summary_rep),
        store=store,
    )

    packet = pipeline.ingest(Observation(text="Alice studies graph memory.", source="notes"))

    assert packet.units is not None
    assert packet.units[0].tags == ["graph", "memory"]
    assert packet.units[0].metadata["representation"]["summary"] == "Alice studies graph memory."
    assert store.count() == 1


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


def test_pipeline_serial_write_triggers_can_preserve_decisions_and_fill_store_selection() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior one"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=[AlwaysTrigger(), StoreAllTrigger()],
        organization=AppendOrganization(target_layer="episodic"),
        evolution_trigger=StoreAllTrigger(slot="evolution_trigger"),
        store=store,
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.decisions == [True]
    assert packet.trace["write_trigger"]["module"] == "store_all_write_trigger"
    assert packet.trace["evolution_trigger"]["module"] == "store_all_evolution_trigger"
    assert packet.decisions_store is not None
    assert packet.decisions_store["episodic"]["selector"]["kind"] == "store_all"
    assert packet.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2"]


def test_dispatch_store_all_trigger_silently_drops_non_primary_store_selection() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior one"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="episodic"),
        evolution_trigger=DispatchEvolutionTrigger(
            (
                NeverTrigger(),
                StoreAllTrigger(slot="evolution_trigger"),
            ),
            primary_index=0,
        ),
        store=store,
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.decisions == [False]
    assert packet.decisions_store is None
    assert packet.trace["dispatch"]["evolution_trigger"]["children"][1]["module"] == "store_all_evolution_trigger"
    assert (
        packet.trace["dispatch"]["evolution_trigger"]["children"][1]["slot_trace"]["decisions_store_counts"]
        == {"episodic": 2}
    )


def test_dispatch_store_all_trigger_keeps_store_selection_when_primary() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    store.append(
        MemoryRecord.from_unit(
            unit=MemoryUnit(text="prior one"),
            layer="episodic",
            sequence_id=store.next_sequence_id(),
        )
    )
    pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="episodic"),
        evolution_trigger=DispatchEvolutionTrigger(
            (
                StoreAllTrigger(slot="evolution_trigger"),
                NeverTrigger(),
            ),
            primary_index=0,
        ),
        store=store,
    )

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    assert packet.decisions == [True]
    assert packet.decisions_store is not None
    assert packet.decisions_store["episodic"]["selector"]["kind"] == "store_all"
    assert packet.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2"]


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
