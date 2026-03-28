from __future__ import annotations

from dataclasses import replace
import json
import pytest

from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
    registered_baseline_class_names,
)
from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    Observation,
    Packet,
    Placement,
    Query,
    RetrievedSet,
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


def test_metadata_gated_write_trigger_reuses_unit_type_and_metadata_flag_signals() -> None:
    from memprimitive.baselines import MetadataGatedWriteTrigger

    packet = Packet(
        units=[
            MemoryUnit(
                text="Remember Alice prefers jasmine tea.",
                unit_id="unit-write",
                unit_type="tim_thought",
                metadata={"tim": {"write": True}},
            ),
            MemoryUnit(
                text="Skip this disabled thought.",
                unit_id="unit-skip",
                unit_type="tim_thought",
                metadata={"tim": {"write": False}},
            ),
        ]
    )

    packet_out, _ = MetadataGatedWriteTrigger().run(packet, MemoryStore())

    assert packet_out.decisions == [True, False]
    assert packet_out.trace["write_trigger"]["scorer"] == "min"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["signals"]["unit_type_ready"] == 1.0
    assert packet_out.trace["write_trigger"]["per_unit"][0]["signals"]["metadata_write_flag"] == 1.0
    assert packet_out.trace["write_trigger"]["per_unit"][1]["signals"]["metadata_write_flag"] == 0.0


def test_key_ready_write_trigger_writes_only_units_with_required_key() -> None:
    from memprimitive.baselines import KeyReadyWriteTrigger

    packet = Packet(
        units=[
            MemoryUnit(
                text="Working summary block",
                unit_id="unit-key",
                metadata={"memgpt_key": "working_summary"},
            ),
            MemoryUnit(
                text="Unnamed block",
                unit_id="unit-missing",
                metadata={},
            ),
        ]
    )

    packet_out, _ = KeyReadyWriteTrigger().run(packet, MemoryStore())

    assert packet_out.decisions == [True, False]
    assert packet_out.trace["write_trigger"]["policy"] == "threshold"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["signals"]["key_ready"] == 1.0
    assert packet_out.trace["write_trigger"]["per_unit"][1]["signals"]["key_ready"] == 0.0


def test_outcome_conditioned_evolution_trigger_triggers_only_for_failed_trials() -> None:
    from memprimitive.baselines import OutcomeConditionedEvolutionTrigger

    failed_packet = Packet(
        observation=Observation(
            text="Trial scratchpad",
            source="dialogue",
            metadata={"is_correct": False, "feedback": "The answer missed the edge case."},
        ),
        units=[MemoryUnit(text="trial unit", unit_id="unit-failed")],
        placements=[Placement(unit_id="unit-failed", target_layer="trial_buffer")],
    )
    success_packet = Packet(
        observation=Observation(
            text="Trial scratchpad",
            source="dialogue",
            metadata={"is_correct": True, "feedback": "The answer is correct."},
        ),
        units=[MemoryUnit(text="trial unit", unit_id="unit-success")],
        placements=[Placement(unit_id="unit-success", target_layer="trial_buffer")],
    )

    failed_out, _ = OutcomeConditionedEvolutionTrigger().run(failed_packet, MemoryStore())
    success_out, _ = OutcomeConditionedEvolutionTrigger().run(success_packet, MemoryStore())

    assert failed_out.evolution_decisions == [True]
    assert failed_out.trace["evolution_trigger"]["scorer"] == "weighted_sum"
    assert failed_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["trial_failed"] == 1.0
    assert failed_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["feedback_present"] == 1.0
    assert success_out.evolution_decisions == [False]
    assert success_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["trial_failed"] == 0.0


def test_reflection_generation_evolution_skips_success_trial_with_outcome_trigger() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        BasicRepresentation,
        OutcomeConditionedEvolutionTrigger,
        PassThroughUnitFormation,
        PlacementWithoutAppendOrganization,
        ReflectionGenerationEvolution,
    )

    pipeline = MemoryPipeline(
        store=MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")])),
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=AlwaysWriteTrigger(),
        organization=PlacementWithoutAppendOrganization(target_layer="trial_buffer"),
        evolution_trigger=OutcomeConditionedEvolutionTrigger(),
        memory_evolution=ReflectionGenerationEvolution(
            target_layer="reflections",
            reflection_generator=lambda payload: f"Reflection: avoid {payload.evaluator_feedback}",
        ),
    )

    packet = pipeline.ingest(
        Observation(
            text="Trial scratchpad",
            source="dialogue",
            metadata={
                "reflexion": {
                    "question": "Parse the input stream",
                    "scratchpad": "Attempt handled the edge case correctly.",
                    "is_correct": True,
                    "evaluator_feedback": "Correct answer.",
                }
            },
        )
    )

    assert packet.evolution_decisions == [False]
    assert pipeline.store.count("reflections") == 0
    assert packet.trace["memory_evolution"]["effects"] == []


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
        if "expand the query" in lowered_system:
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


def test_reflection_generation_evolution_appends_reflection_for_failed_trial() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import (
        AlwaysWriteTrigger,
        BasicRepresentation,
        OutcomeConditionedEvolutionTrigger,
        PassThroughUnitFormation,
        PlacementWithoutAppendOrganization,
        ReflectionGenerationEvolution,
    )

    pipeline = MemoryPipeline(
        store=MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")])),
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=AlwaysWriteTrigger(),
        organization=PlacementWithoutAppendOrganization(target_layer="trial_buffer"),
        evolution_trigger=OutcomeConditionedEvolutionTrigger(),
        memory_evolution=ReflectionGenerationEvolution(
            target_layer="reflections",
            reflection_generator=lambda payload: f"Reflection: avoid {payload.evaluator_feedback}",
        ),
    )

    packet = pipeline.ingest(
        Observation(
            text="Trial scratchpad",
            source="dialogue",
            metadata={
                "reflexion": {
                    "question": "Parse the input stream",
                    "scratchpad": "Attempt missed the empty-input edge case.",
                    "is_correct": False,
                    "evaluator_feedback": "the empty-input edge case",
                    "trial_index": 2,
                }
            },
        )
    )

    assert packet.evolution_decisions == [True]
    assert pipeline.store.count("reflections") == 1
    reflection_record = pipeline.store.iter_records("reflections")[0]
    assert reflection_record.text == "Reflection: avoid the empty-input edge case"
    assert reflection_record.metadata["reflection"]["question"] == "Parse the input stream"
    assert packet.trace["memory_evolution"]["effects"][0]["effect_type"] == "reflection_append"
    assert packet.trace["memory_evolution"]["residual_boundary"]["skeleton"] == "generic reflection generation evolution"


def test_new_write_evolution_trigger_requires_partition_ready_local_write_context() -> None:
    from memprimitive.baselines import NewWriteEvolutionTrigger

    packet = Packet(
        units=[
            MemoryUnit(
                text="Alice prefers jasmine tea.",
                unit_id="unit-ready",
                unit_type="tim_thought",
                metadata={"tim": {"group_id": "alice-profile"}},
            ),
            MemoryUnit(
                text="Thought without partition key.",
                unit_id="unit-missing-key",
                unit_type="tim_thought",
                metadata={"tim": {}},
            ),
            MemoryUnit(
                text="Thought routed to the wrong layer.",
                unit_id="unit-wrong-layer",
                unit_type="tim_thought",
                metadata={"tim": {"group_id": "alice-profile"}},
            ),
        ],
        placements=[
            Placement(unit_id="unit-ready", target_layer="thought_memory"),
            Placement(unit_id="unit-missing-key", target_layer="thought_memory"),
            Placement(unit_id="unit-wrong-layer", target_layer="default"),
        ],
    )

    packet_out, _ = NewWriteEvolutionTrigger().run(packet, MemoryStore())

    assert packet_out.evolution_decisions == [True, False, False]
    assert packet_out.trace["evolution_trigger"]["scorer"] == "min"
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["partition_key_ready"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][1]["signals"]["partition_key_ready"] == 0.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][2]["gate"] is False


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


def test_reflexion_style_trigger_family_signals_fire_for_failed_trial_feedback() -> None:
    from memprimitive.baselines import PassThroughUnitFormation
    from memprimitive.baselines._trigger_family import (
        FeedbackPresenceSignal,
        FeedbackSchemaGate,
        OutcomeCorrectnessSignal,
        ThresholdPolicy,
        WeightedSumScorer,
    )
    from memprimitive.baselines.evolution_trigger import compose_evolution_trigger

    packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="trial trace",
                source="dialogue",
                metadata={"reflexion": {"is_correct": False, "evaluator_feedback": "missing edge case"}},
            )
        ),
        MemoryStore(),
    )
    trigger = compose_evolution_trigger(
        name="reflexion_failed_trial_trigger",
        signal_providers=(OutcomeCorrectnessSignal(), FeedbackPresenceSignal()),
        scorer=WeightedSumScorer(weights={"trial_failed": 1.0, "feedback_present": 0.1}),
        gate=FeedbackSchemaGate(),
        policy=ThresholdPolicy(threshold=1.0),
        input_requirements=("units", "observation"),
    )

    packet_out, _ = trigger.run(packet, store)

    assert packet_out.evolution_decisions == [True]
    per_unit = packet_out.trace["evolution_trigger"]["per_unit"][0]
    assert per_unit["signals"] == {"trial_failed": 1.0, "feedback_present": 1.0}
    assert per_unit["gate"] is True
    assert per_unit["score"] == pytest.approx(1.1)


def test_reflexion_style_trigger_family_signals_do_not_fire_for_successful_trial() -> None:
    from memprimitive.baselines import PassThroughUnitFormation
    from memprimitive.baselines._trigger_family import (
        FeedbackPresenceSignal,
        FeedbackSchemaGate,
        OutcomeCorrectnessSignal,
        ThresholdPolicy,
        WeightedSumScorer,
    )
    from memprimitive.baselines.evolution_trigger import compose_evolution_trigger

    packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="trial trace",
                source="dialogue",
                metadata={"is_correct": True, "feedback": "answer matches expected output"},
            )
        ),
        MemoryStore(),
    )
    trigger = compose_evolution_trigger(
        name="reflexion_success_trial_trigger",
        signal_providers=(OutcomeCorrectnessSignal(), FeedbackPresenceSignal()),
        scorer=WeightedSumScorer(weights={"trial_failed": 1.0, "feedback_present": 0.1}),
        gate=FeedbackSchemaGate(),
        policy=ThresholdPolicy(threshold=1.0),
        input_requirements=("units", "observation"),
    )

    packet_out, _ = trigger.run(packet, store)

    assert packet_out.evolution_decisions == [False]
    per_unit = packet_out.trace["evolution_trigger"]["per_unit"][0]
    assert per_unit["signals"] == {"trial_failed": 0.0, "feedback_present": 1.0}
    assert per_unit["gate"] is True


def test_feedback_schema_gate_blocks_when_outcome_and_feedback_schema_are_missing() -> None:
    from memprimitive.baselines import PassThroughUnitFormation
    from memprimitive.baselines._trigger_family import (
        FeedbackPresenceSignal,
        FeedbackSchemaGate,
        OutcomeCorrectnessSignal,
        ThresholdPolicy,
        WeightedSumScorer,
    )
    from memprimitive.baselines.evolution_trigger import compose_evolution_trigger

    packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="trial trace",
                source="dialogue",
                metadata={"note": "no outcome schema here"},
            )
        ),
        MemoryStore(),
    )
    trigger = compose_evolution_trigger(
        name="reflexion_schema_guarded_trigger",
        signal_providers=(OutcomeCorrectnessSignal(), FeedbackPresenceSignal()),
        scorer=WeightedSumScorer(weights={"trial_failed": 1.0, "feedback_present": 0.1}),
        gate=FeedbackSchemaGate(),
        policy=ThresholdPolicy(threshold=0.0),
        input_requirements=("units", "observation"),
    )

    packet_out, _ = trigger.run(packet, store)

    assert packet_out.evolution_decisions == [False]
    per_unit = packet_out.trace["evolution_trigger"]["per_unit"][0]
    assert per_unit["signals"] == {"trial_failed": 0.0, "feedback_present": 0.0}
    assert per_unit["gate"] is False


def test_metadata_flag_signal_reads_nested_unit_metadata_flag() -> None:
    from memprimitive.baselines._trigger_family import MetadataFlagSignal, TriggerContext

    packet = Packet(
        units=[MemoryUnit(text="tim thought", unit_type="tim_thought", metadata={"tim": {"write": True}})]
    )
    signals = MetadataFlagSignal(path="tim.write", signal_name="write_flag").provide(
        TriggerContext(packet=packet, store=MemoryStore(), output_field="decisions", trace_key="write_trigger"),
        0,
    )

    assert signals == {"write_flag": 1.0}


def test_metadata_flag_signal_raises_for_missing_required_path() -> None:
    from memprimitive.baselines._trigger_family import MetadataFlagSignal, TriggerContext

    packet = Packet(units=[MemoryUnit(text="tim thought", unit_type="tim_thought", metadata={})])

    with pytest.raises(ValueError, match="required path"):
        MetadataFlagSignal(path="tim.write", signal_name="write_flag").provide(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="decisions", trace_key="write_trigger"),
            0,
        )


def test_unit_type_signal_matches_expected_type() -> None:
    from memprimitive.baselines._trigger_family import TriggerContext, UnitTypeSignal

    packet = Packet(units=[MemoryUnit(text="tim thought", unit_type="tim_thought")])
    signals = UnitTypeSignal(expected_unit_type="tim_thought", signal_name="is_tim_thought").provide(
        TriggerContext(packet=packet, store=MemoryStore(), output_field="decisions", trace_key="write_trigger"),
        0,
    )

    assert signals == {"is_tim_thought": 1.0}


def test_placement_exists_signal_requires_aligned_placements() -> None:
    from memprimitive.baselines._trigger_family import PlacementExistsSignal, TriggerContext

    packet = Packet(units=[MemoryUnit(text="Alice note", unit_id="unit-1")])

    with pytest.raises(ValueError, match="placements is required"):
        PlacementExistsSignal().provide(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="evolution_decisions", trace_key="evolution_trigger"),
            0,
        )


def test_partition_key_present_signal_detects_available_partition_key() -> None:
    from memprimitive.baselines._trigger_family import PartitionKeyPresentSignal, TriggerContext

    packet = Packet(
        units=[MemoryUnit(text="tim thought", metadata={"tim": {"group_id": "bucket-1"}})]
    )
    signals = PartitionKeyPresentSignal(
        paths=("tim.group_id", "tim.hash_index"),
        signal_name="has_partition_key",
    ).provide(
        TriggerContext(packet=packet, store=MemoryStore(), output_field="evolution_decisions", trace_key="evolution_trigger"),
        0,
    )

    assert signals == {"has_partition_key": 1.0}


def test_partition_key_present_signal_can_raise_when_all_paths_are_missing() -> None:
    from memprimitive.baselines._trigger_family import PartitionKeyPresentSignal, TriggerContext

    packet = Packet(units=[MemoryUnit(text="tim thought", metadata={})])

    with pytest.raises(ValueError, match="could not find any configured paths"):
        PartitionKeyPresentSignal(
            paths=("tim.group_id", "tim.hash_index"),
            strict_missing=True,
        ).provide(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="evolution_decisions", trace_key="evolution_trigger"),
            0,
        )


def test_neighbor_count_and_top_similarity_signals_read_graph_vector_candidates() -> None:
    from memprimitive.baselines._trigger_family import NeighborCountSignal, TopNeighborSimilaritySignal, TriggerContext

    store = _graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-existing",
            layer="knowledge_graph",
            text="Alice studies graph memory",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Alice graphs", unit_id="unit-new", embedding=[0.8, 0.2])],
        placements=[Placement(unit_id="unit-new", target_layer="knowledge_graph")],
    )
    context = TriggerContext(packet=packet, store=store, output_field="evolution_decisions", trace_key="evolution_trigger")

    count_signals = NeighborCountSignal(top_k=3).provide(context, 0)
    similarity_signals = TopNeighborSimilaritySignal(top_k=3).provide(context, 0)

    assert count_signals == {"neighbor_count": 1.0}
    assert similarity_signals["top_neighbor_similarity"] > 0.9


def test_neighbor_signals_return_zero_without_target_layer_context() -> None:
    from memprimitive.baselines._trigger_family import NeighborCountSignal, TopNeighborSimilaritySignal, TriggerContext

    packet = Packet(units=[MemoryUnit(text="Alice graphs", embedding=[1.0, 0.0])])
    context = TriggerContext(packet=packet, store=_graph_vector_store(), output_field="evolution_decisions", trace_key="evolution_trigger")

    assert NeighborCountSignal().provide(context, 0) == {"neighbor_count": 0.0}
    assert TopNeighborSimilaritySignal().provide(context, 0) == {"top_neighbor_similarity": 0.0}


def test_neighbor_signals_prefer_explicit_layer_over_packet_placement() -> None:
    from memprimitive.baselines._trigger_family import NeighborCountSignal, TopNeighborSimilaritySignal, TriggerContext

    store = _mixed_graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-knowledge",
            unit_id="unit-knowledge",
            layer="knowledge_graph",
            text="Alice knowledge graph note",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-other",
            unit_id="unit-other",
            layer="other_graph",
            text="Alice other graph note",
            timestamp="2026-03-27T00:01:00+00:00",
            embedding=[0.0, 1.0],
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Alice mixed-layer graph note", unit_id="unit-new", embedding=[0.9, 0.1])],
        placements=[Placement(unit_id="unit-new", target_layer="other_graph")],
    )
    context = TriggerContext(packet=packet, store=store, output_field="evolution_decisions", trace_key="evolution_trigger")

    count_signals = NeighborCountSignal(top_k=3, layer="knowledge_graph").provide(context, 0)
    similarity_signals = TopNeighborSimilaritySignal(top_k=3, layer="knowledge_graph").provide(context, 0)

    assert count_signals == {"neighbor_count": 1.0}
    assert similarity_signals["top_neighbor_similarity"] > 0.9
    assert NeighborCountSignal(top_k=3).provide(context, 0) == {"neighbor_count": 1.0}
    assert TopNeighborSimilaritySignal(top_k=3).provide(context, 0)["top_neighbor_similarity"] < 0.2


def test_schema_present_gate_validates_required_unit_schema() -> None:
    from memprimitive.baselines._trigger_family import SchemaPresentGate, TriggerContext

    packet = Packet(
        units=[
            MemoryUnit(
                text="graph note",
                metadata={"note": {"summary": "Alice studies graph memory", "keywords": ["alice", "graph"]}},
            )
        ]
    )
    gate = SchemaPresentGate(paths=("note.summary", "note.keywords"), source="unit.metadata")

    assert (
        gate.evaluate(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="decisions", trace_key="write_trigger"),
            0,
            signals={},
            score=0.0,
        )
        is True
    )


def test_schema_present_gate_blocks_when_required_schema_is_missing() -> None:
    from memprimitive.baselines._trigger_family import SchemaPresentGate, TriggerContext

    packet = Packet(units=[MemoryUnit(text="graph note", metadata={"note": {"summary": "only summary"}})])
    gate = SchemaPresentGate(paths=("note.summary", "note.keywords"), source="unit.metadata")

    assert (
        gate.evaluate(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="decisions", trace_key="write_trigger"),
            0,
            signals={},
            score=0.0,
        )
        is False
    )


def test_has_embedding_gate_checks_current_unit_embedding() -> None:
    from memprimitive.baselines._trigger_family import HasEmbeddingGate, TriggerContext

    packet = Packet(units=[MemoryUnit(text="embedded", embedding=[1.0, 0.0])])
    gate = HasEmbeddingGate()

    assert (
        gate.evaluate(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="evolution_decisions", trace_key="evolution_trigger"),
            0,
            signals={},
            score=0.0,
        )
        is True
    )


def test_vector_index_ready_gate_blocks_when_target_layer_lacks_vector_index() -> None:
    from memprimitive.baselines._trigger_family import TriggerContext, VectorIndexReadyGate

    packet = Packet(
        units=[MemoryUnit(text="graph note", unit_id="unit-1", embedding=[1.0, 0.0])],
        placements=[Placement(unit_id="unit-1", target_layer="knowledge_graph")],
    )
    gate = VectorIndexReadyGate()

    assert (
        gate.evaluate(
            TriggerContext(packet=packet, store=_graph_store(), output_field="evolution_decisions", trace_key="evolution_trigger"),
            0,
            signals={},
            score=0.0,
        )
        is False
    )


def test_graph_layer_gate_checks_target_layer_shape() -> None:
    from memprimitive.baselines._trigger_family import GraphLayerGate, TriggerContext

    packet = Packet(
        units=[MemoryUnit(text="graph note", unit_id="unit-1", embedding=[1.0, 0.0])],
        placements=[Placement(unit_id="unit-1", target_layer="knowledge_graph")],
    )

    assert (
        GraphLayerGate().evaluate(
            TriggerContext(packet=packet, store=_graph_vector_store(), output_field="evolution_decisions", trace_key="evolution_trigger"),
            0,
            signals={},
            score=0.0,
        )
        is True
    )

    assert (
        GraphLayerGate().evaluate(
            TriggerContext(packet=packet, store=MemoryStore(), output_field="evolution_decisions", trace_key="evolution_trigger"),
            0,
            signals={},
            score=0.0,
        )
        is False
    )


def test_vector_and_graph_readiness_gates_prefer_explicit_layer_over_packet_placement() -> None:
    from memprimitive.baselines._trigger_family import GraphLayerGate, TriggerContext, VectorIndexReadyGate

    packet = Packet(
        units=[MemoryUnit(text="graph note", unit_id="unit-1", embedding=[1.0, 0.0])],
        placements=[Placement(unit_id="unit-1", target_layer="default")],
    )
    context = TriggerContext(packet=packet, store=_graph_vector_store(), output_field="evolution_decisions", trace_key="evolution_trigger")

    assert VectorIndexReadyGate().evaluate(context, 0, signals={}, score=0.0) is False
    assert GraphLayerGate().evaluate(context, 0, signals={}, score=0.0) is False
    assert VectorIndexReadyGate(layer="knowledge_graph").evaluate(context, 0, signals={}, score=0.0) is True
    assert GraphLayerGate(layer="knowledge_graph").evaluate(context, 0, signals={}, score=0.0) is True


def test_amem_style_trigger_family_components_gate_neighbor_trigger_on_vector_graph_readiness() -> None:
    from memprimitive.baselines._trigger_family import (
        BooleanGatePolicy,
        GraphLayerGate,
        HasEmbeddingGate,
        MinScorer,
        NeighborCountSignal,
        TopNeighborSimilaritySignal,
        TriggerContext,
        VectorIndexReadyGate,
    )
    from memprimitive.baselines.evolution_trigger import compose_evolution_trigger

    ready_store = _graph_vector_store()
    ready_store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-existing",
            layer="knowledge_graph",
            text="Alice studies graph memory",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    ready_packet = Packet(
        units=[MemoryUnit(text="Alice graph note", unit_id="unit-new", embedding=[0.9, 0.1])],
        placements=[Placement(unit_id="unit-new", target_layer="knowledge_graph")],
    )
    trigger = compose_evolution_trigger(
        name="amem_neighbor_trigger",
        signal_providers=(NeighborCountSignal(top_k=2), TopNeighborSimilaritySignal(top_k=2)),
        scorer=MinScorer(sources=("neighbor_count", "top_neighbor_similarity")),
        gate=GraphLayerGate(),
        policy=BooleanGatePolicy(),
    )

    packet_out, _ = trigger.run(ready_packet, ready_store)
    assert packet_out.evolution_decisions == [True]

    not_ready_packet = Packet(
        units=[MemoryUnit(text="Alice flat note", unit_id="unit-flat", embedding=[0.9, 0.1])],
        placements=[Placement(unit_id="unit-flat", target_layer="default")],
    )
    not_ready_trigger = compose_evolution_trigger(
        name="amem_neighbor_trigger_with_full_gates",
        signal_providers=(NeighborCountSignal(top_k=2), TopNeighborSimilaritySignal(top_k=2)),
        scorer=MinScorer(sources=("neighbor_count", "top_neighbor_similarity")),
        gate=GraphLayerGate(),
        policy=BooleanGatePolicy(),
    )
    packet_out, _ = not_ready_trigger.run(not_ready_packet, ready_store)
    assert packet_out.evolution_decisions == [False]

    assert HasEmbeddingGate().evaluate(
        TriggerContext(packet=ready_packet, store=ready_store, output_field="evolution_decisions", trace_key="evolution_trigger"),
        0,
        signals={},
        score=0.0,
    )
    assert VectorIndexReadyGate().evaluate(
        TriggerContext(packet=ready_packet, store=ready_store, output_field="evolution_decisions", trace_key="evolution_trigger"),
        0,
        signals={},
        score=0.0,
    )


def test_compose_graph_neighbor_evolution_trigger_fires_when_neighbors_exist() -> None:
    from memprimitive.baselines.evolution_trigger import compose_graph_neighbor_evolution_trigger

    store = _graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice studies graph memory",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=[1.0, 0.0],
            metadata={"graph": {"entities": ["Alice"], "links": []}},
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Alice graph note", unit_id="unit-new", embedding=[0.95, 0.05])],
        placements=[Placement(unit_id="unit-new", target_layer="knowledge_graph")],
    )

    packet_out, _ = compose_graph_neighbor_evolution_trigger(
        name="graph_neighbor_exists_trigger",
        layer="knowledge_graph",
        candidate_top_k=2,
    ).run(packet, store)

    assert packet_out.evolution_decisions == [True]
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["neighbor_count"] == 1.0


def test_neighbor_exists_evolution_trigger_blocks_without_neighbor_candidates() -> None:
    from memprimitive.baselines import NeighborExistsEvolutionTrigger

    store = _graph_vector_store()
    packet = Packet(
        units=[MemoryUnit(text="Alice isolated note", unit_id="unit-new", embedding=[1.0, 0.0])],
        placements=[Placement(unit_id="unit-new", target_layer="knowledge_graph")],
    )

    packet_out, _ = NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=2).run(packet, store)

    assert packet_out.evolution_decisions == [False]
    assert packet_out.trace["evolution_trigger"]["gate"] == "all"


def test_neighbor_exists_evolution_trigger_prefers_configured_target_layer_over_packet_placement() -> None:
    from memprimitive.baselines import NeighborExistsEvolutionTrigger

    store = _mixed_graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-other",
            unit_id="unit-other",
            layer="other_graph",
            text="Alice other graph note",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=[0.99, 0.01],
            metadata={"graph": {"entities": ["Alice"], "links": []}},
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Alice mixed-layer note", unit_id="unit-new", embedding=[1.0, 0.0])],
        placements=[Placement(unit_id="unit-new", target_layer="other_graph")],
    )

    packet_out, _ = NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=2).run(packet, store)
    assert packet_out.evolution_decisions == [False]
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["neighbor_count"] == 0.0

    store.append(
        MemoryRecord(
            record_id="rec-knowledge",
            unit_id="unit-knowledge",
            layer="knowledge_graph",
            text="Alice knowledge graph note",
            timestamp="2026-03-27T00:01:00+00:00",
            embedding=[0.98, 0.02],
            metadata={"graph": {"entities": ["Alice"], "links": []}},
        )
    )

    packet_out, _ = NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=2).run(packet, store)
    assert packet_out.evolution_decisions == [True]
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["signals"]["neighbor_count"] == 1.0


def test_neighbor_exists_evolution_trigger_validates_placements_even_with_explicit_target_layer() -> None:
    from memprimitive.baselines import NeighborExistsEvolutionTrigger

    packet = Packet(
        units=[MemoryUnit(text="Alice mixed-layer note", unit_id="unit-new", embedding=[1.0, 0.0])],
        placements=[Placement(unit_id="unit-wrong", target_layer="other_graph")],
    )

    with pytest.raises(ValueError, match="placements must align"):
        NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=2).run(
            packet,
            _mixed_graph_vector_store(),
        )


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

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = BasicRepresentation(elements=("text", "entities", "triple")).run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = GraphAppendOrganization(target_layer="knowledge_graph").run(packet, store)

    record = store.iter_records("knowledge_graph")[0]
    assert "graph" in record.metadata
    assert record.metadata["graph"]["triples"]
    assert record.metadata["graph"]["links"] == []
    assert record.metadata["graph"]["link_count"] == 0
    assert packet.trace["organization"]["graph_metadata_schema"]


def test_memory_store_graph_link_round_trip_returns_neighbors() -> None:
    store = _graph_store()
    first = MemoryRecord(record_id="rec-1", unit_id="unit-1", layer="knowledge_graph", text="Alice likes tea", timestamp="t1")
    second = MemoryRecord(record_id="rec-2", unit_id="unit-2", layer="knowledge_graph", text="Alice studies graphs", timestamp="t2")
    store.append(first)
    store.append(second)

    merged_links = store.add_graph_links("knowledge_graph", "rec-2", ["rec-1"])
    neighbors = store.iter_graph_neighbors("knowledge_graph", "rec-2")

    assert merged_links == ["rec-1"]
    assert [record.record_id for record in neighbors] == ["rec-1"]


def test_graph_neighbor_retrieval_handles_missing_and_present_links() -> None:
    from memprimitive.baselines import GraphNeighborRetrieval

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-seed",
        unit_id="unit-seed",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    store.append(seed)
    store.append(neighbor)

    empty_packet, _ = GraphNeighborRetrieval(top_k=3).run(
        Packet(query=Query(text="Alice", metadata={"graph_seed_record_ids": ["rec-seed"]})),
        store,
    )
    assert empty_packet.retrieved.items == []

    store.add_graph_links("knowledge_graph", "rec-seed", ["rec-neighbor"])
    linked_packet, _ = GraphNeighborRetrieval(top_k=3).run(
        Packet(query=Query(text="Alice", metadata={"graph_seed_record_ids": ["rec-seed"]})),
        store,
    )

    assert [record.record_id for record in linked_packet.retrieved.items] == ["rec-neighbor"]
    assert linked_packet.trace["retrieval"]["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_graph_seed_and_expand_retrieval_uses_candidate_set_and_neighbor_expansion() -> None:
    from memprimitive.baselines import GraphSeedAndExpandRetrieval

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-seed",
        unit_id="unit-seed",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": ["rec-neighbor"]}},
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    other = MemoryRecord(
        record_id="rec-other",
        unit_id="unit-other",
        layer="knowledge_graph",
        text="Bob studies memory retrieval",
        timestamp="2026-03-27T00:02:00+00:00",
        metadata={"graph": {"entities": ["Bob"], "links": []}},
    )
    store.append(seed)
    store.append(neighbor)
    store.append(other)

    packet_out, _ = GraphSeedAndExpandRetrieval(top_k=3, seed_top_k=1).run(
        Packet(
            query=Query(
                text="Alice graph",
                metadata={"graph_candidate_record_ids": ["rec-seed", "rec-neighbor"]},
            )
        ),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-seed", "rec-neighbor"]
    assert packet_out.trace["retrieval"]["seed_record_ids"] == ["rec-seed"]
    assert packet_out.trace["retrieval"]["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_graph_neighbor_append_evolution_only_modifies_graph_layer() -> None:
    from memprimitive.baselines import GraphNeighborAppendEvolution

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-working",
            unit_id="unit-working",
            layer="default",
            text="Working memory note",
            timestamp="2026-03-27T00:00:00+00:00",
        )
    )
    existing = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    incoming = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    store.append(existing)
    store.append(incoming)

    packet = Packet(
        units=[MemoryUnit(text="Alice studies graph memory", unit_id="unit-2")],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        evolution_decisions=[True],
    )

    packet_out, store = GraphNeighborAppendEvolution(target_layer="knowledge_graph", neighbor_limit=1).run(packet, store)

    updated_graph_records = store.iter_records("knowledge_graph")
    updated_incoming = [record for record in updated_graph_records if record.record_id == "rec-2"][0]
    assert updated_incoming.metadata["graph"]["links"] == ["rec-1"]
    assert store.iter_records("default")[0].record_id == "rec-working"
    assert packet_out.trace["memory_evolution"]["effects"][0]["target_layer"] == "knowledge_graph"


def test_graph_link_evolution_rewrites_only_graph_metadata_namespace() -> None:
    from memprimitive.baselines import GraphLinkEvolution

    store = _graph_vector_store()
    existing = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:00:00+00:00",
        embedding=[1.0, 0.0],
        metadata={"owner": "kept", "graph": {"entities": ["Alice"], "links": []}},
    )
    incoming = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:01:00+00:00",
        embedding=[0.95, 0.05],
        metadata={"owner": "kept", "graph": {"entities": ["Alice"], "links": []}},
    )
    store.append(existing)
    store.append(incoming)

    packet = Packet(
        units=[MemoryUnit(text="Alice studies graph memory", unit_id="unit-2", embedding=[0.95, 0.05])],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        evolution_decisions=[True],
    )

    packet_out, store = GraphLinkEvolution(
        target_layer="knowledge_graph",
        neighbor_limit=1,
        rewrite_neighbor_metadata=True,
    ).run(packet, store)

    updated = [record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-2"][0]
    assert updated.metadata["owner"] == "kept"
    assert updated.metadata["graph"]["links"] == ["rec-1"]
    assert updated.metadata["graph"]["neighbor_context"]["neighbor_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["effects"][0]["candidate_scores"][0]["record_id"] == "rec-1"


def test_graph_neighbor_context_trace_evolution_can_run_trace_only_or_rewrite() -> None:
    from memprimitive.baselines import GraphNeighborContextTraceEvolution

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    current = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": ["rec-1"]}},
    )
    store.append(seed)
    store.append(current)

    packet = Packet(
        units=[MemoryUnit(text="Alice studies graph memory", unit_id="unit-2")],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        evolution_decisions=[True],
    )

    trace_packet, store = GraphNeighborContextTraceEvolution(target_layer="knowledge_graph").run(packet, store)
    assert trace_packet.trace["memory_evolution"]["effects"][0]["neighbor_record_ids"] == ["rec-1"]
    assert "neighbor_context" not in store.iter_records("knowledge_graph")[1].metadata["graph"]

    rewrite_packet, store = GraphNeighborContextTraceEvolution(
        target_layer="knowledge_graph",
        rewrite_metadata=True,
    ).run(packet, store)
    assert rewrite_packet.trace["memory_evolution"]["effects"][0]["rewrite_metadata"] is True
    assert store.iter_records("knowledge_graph")[1].metadata["graph"]["neighbor_context"]["neighbor_record_ids"] == ["rec-1"]


def test_graph_readout_renders_graph_metadata() -> None:
    from memprimitive.baselines import GraphReadout

    record = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": ["rec-0"]}},
    )
    packet_out, _ = GraphReadout().run(Packet(retrieved=RetrievedSet(items=[record], scores=[])), _graph_store())

    assert "entities=Alice" in packet_out.readout.text
    assert "links=rec-0" in packet_out.readout.text
    assert packet_out.readout.metadata["graph_item_count"] == 1


def test_graph_dependent_pipeline_end_to_end_supports_trigger_evolution_retrieval_and_readout() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import (
        BasicRepresentation,
        GraphAppendOrganization,
        GraphLinkEvolution,
        GraphNeighborContextTraceEvolution,
        GraphReadout,
        GraphSeedAndExpandRetrieval,
        NeighborExistsEvolutionTrigger,
        PassThroughUnitFormation,
    )

    store = _graph_vector_store()
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "triple", "tags", "keywords")),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=2),
        memory_evolution=(
            GraphLinkEvolution(target_layer="knowledge_graph", neighbor_limit=2, rewrite_neighbor_metadata=True),
            GraphNeighborContextTraceEvolution(target_layer="knowledge_graph", rewrite_metadata=True),
        ),
        retrieval=GraphSeedAndExpandRetrieval(top_k=4, layer="knowledge_graph", seed_top_k=1),
        readout=GraphReadout(),
        store=store,
    )

    first_packet = pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
    second_packet = pipeline.ingest(Observation(text="Alice studies graph memory systems.", source="notes"))
    pipeline.ingest(Observation(text="Bob builds retrieval tools.", source="notes"))
    readout = pipeline.recall(Query(text="Alice graph"))

    graph_records = pipeline.store.iter_records("knowledge_graph")
    linked_record = [record for record in graph_records if record.unit_id == second_packet.units[0].unit_id][0]

    assert first_packet.evolution_decisions == [False]
    assert second_packet.evolution_decisions == [True]
    assert linked_record.metadata["graph"]["links"]
    assert linked_record.metadata["graph"]["neighbor_context"]["neighbor_record_ids"]
    assert "Alice studies graph memory systems." in readout.text or "Alice likes jasmine tea." in readout.text
    assert readout.source_ids


def test_semantic_field_enrichment_and_retrieval_embedding_repair_note_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import RetrievalOrientedEmbeddingRepresentation, SemanticFieldEnrichmentRepresentation

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    packet = Packet(
        units=[
            MemoryUnit(
                text="Alice likes tea.",
                metadata={"amem": {"context": "Alice routine only", "keywords": ["alice", "tea"]}},
            )
        ]
    )

    packet, store = SemanticFieldEnrichmentRepresentation(note_namespace="amem").run(packet, MemoryStore())
    packet, _ = RetrievalOrientedEmbeddingRepresentation(note_namespace="amem").run(packet, store)

    unit = packet.units[0]
    assert unit.metadata["amem"]["note_text"].startswith("Comprehensive note:")
    assert unit.metadata["representation"]["enhanced_embedding_text"].startswith("content: Alice likes tea.")
    assert unit.embedding == _runtime._DEFAULT_RUNTIME.embed(unit.metadata["representation"]["enhanced_embedding_text"])


def test_graph_append_link_ready_organization_requires_graph_vector_layer() -> None:
    from memprimitive import IncompatibleCompositionError, MemoryPipeline
    from memprimitive.baselines import GraphAppendLinkReadyOrganization

    bad_store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory_graph", shape="Graph", indices=("graph", "keyword", "tag"))]
        )
    )

    with pytest.raises(IncompatibleCompositionError, match="vector"):
        MemoryPipeline(store=bad_store, organization=GraphAppendLinkReadyOrganization(target_layer="memory_graph"))


def test_vector_graph_seed_and_expand_retrieval_expands_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import VectorGraphSeedAndExpandRetrieval

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    store = _graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-seed",
            unit_id="unit-seed",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=_runtime._DEFAULT_RUNTIME.embed("content: Alice likes tea."),
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "context": "Alice's tea habit supports her daily routine.",
                    "enhanced_embedding_text": "content: Alice likes tea.",
                },
                "graph": {"entities": ["Alice"], "links": ["rec-neighbor"]},
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-neighbor",
            unit_id="unit-neighbor",
            layer="knowledge_graph",
            text="Tea routines improve focus.",
            timestamp="2026-03-27T00:00:01+00:00",
            embedding=_runtime._DEFAULT_RUNTIME.embed("content: Tea routines improve focus."),
            metadata={
                "amem": {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                },
                "representation": {
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "context": "Tea routines are linked to improved focus.",
                    "enhanced_embedding_text": "content: Tea routines improve focus.",
                },
                "graph": {"entities": ["Tea"], "links": []},
            },
        )
    )

    packet_out, _ = VectorGraphSeedAndExpandRetrieval(
        top_k=2,
        layer="knowledge_graph",
        candidate_k=1,
        neighbor_expansion_k=1,
        note_namespace="amem",
    ).run(Packet(query=Query(text="Alice")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-seed", "rec-neighbor"]
    assert packet_out.retrieved.trace["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_link_strengthening_and_neighbor_update_write_back_graph_and_note_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import LinkStrengtheningEvolution, NeighborContextUpdateEvolution

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    store = _graph_vector_store()
    first_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Alice likes tea.")
    second_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Tea routines improve focus.")
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=first_embedding,
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {"enhanced_embedding_text": "content: Alice likes tea."},
                "graph": {"entities": ["Alice"], "links": []},
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Tea routines improve focus.",
            timestamp="2026-03-27T00:00:01+00:00",
            embedding=second_embedding,
            metadata={
                "amem": {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                },
                "representation": {"enhanced_embedding_text": "content: Tea routines improve focus."},
                "graph": {"entities": ["Tea"], "links": []},
            },
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Tea routines improve focus.", unit_id="unit-2", embedding=second_embedding)],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        evolution_decisions=[True],
    )

    packet, store = LinkStrengtheningEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)
    packet, store = NeighborContextUpdateEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)

    current = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-2")
    neighbor = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-1")
    assert current.metadata["graph"]["links"] == ["rec-1"]
    assert neighbor.metadata["amem"]["context"] == "Alice's tea habit is now understood as a focus-supporting routine."
    assert neighbor.metadata["amem"]["tags"] == ["preference", "habit", "focus"]


def test_amem_evolution_repairs_list_shaped_llm_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import LinkStrengtheningEvolution, NeighborContextUpdateEvolution

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _WrapperShapeAMEMRuntime())
    store = _graph_vector_store()
    first_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Alice likes tea.")
    second_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Tea routines improve focus.")
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=first_embedding,
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {"enhanced_embedding_text": "content: Alice likes tea."},
                "graph": {"entities": ["Alice"], "links": []},
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Tea routines improve focus.",
            timestamp="2026-03-27T00:00:01+00:00",
            embedding=second_embedding,
            metadata={
                "amem": {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                },
                "representation": {"enhanced_embedding_text": "content: Tea routines improve focus."},
                "graph": {"entities": ["Tea"], "links": []},
            },
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Tea routines improve focus.", unit_id="unit-2", embedding=second_embedding)],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        evolution_decisions=[True],
    )

    packet, store = LinkStrengtheningEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)
    packet, store = NeighborContextUpdateEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)

    current = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-2")
    neighbor = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-1")
    assert current.metadata["graph"]["links"] == ["rec-1"]
    assert neighbor.metadata["amem"]["context"] == "Alice's tea habit is now understood as a focus-supporting routine."
    assert neighbor.metadata["amem"]["tags"] == ["preference", "habit", "focus"]


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


def test_buffer_retrieval_returns_latest_window_in_chronological_order() -> None:
    from memprimitive.baselines import BufferRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")]))
    for index in range(1, 5):
        store.append(
            MemoryRecord(
                record_id=f"rec-{index}",
                unit_id=f"unit-{index}",
                layer="reflections",
                text=f"Reflection {index}",
                timestamp=f"2026-01-01T00:00:0{index}+00:00",
            )
        )

    packet_out, _ = BufferRetrieval(top_k=2, layer="reflections").run(
        Packet(query=Query(text="Current question")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-3", "rec-4"]
    assert packet_out.retrieved.trace["candidate_count"] == 4


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


def test_prompt_context_readout_switches_between_strategies() -> None:
    from memprimitive.baselines import PromptContextReadout

    reflection_record = MemoryRecord(
        record_id="rec-reflection",
        unit_id="unit-reflection",
        layer="reflections",
        text="Reflection: handle the empty-input edge case first.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    retrieved = RetrievedSet(items=[reflection_record], scores=[])

    reflexion_packet, _ = PromptContextReadout(memory_layer="reflections", default_strategy="reflexion").run(
        Packet(
            query=Query(
                text="Parse the input stream",
                metadata={"reflexion": {"last_attempt": "Attempt missed the edge case."}},
            ),
            retrieved=retrieved,
        ),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")])),
    )
    assert "Reflection 1:" in reflexion_packet.readout.text
    assert reflexion_packet.readout.source_ids == ["rec-reflection"]

    last_attempt_packet, _ = PromptContextReadout(memory_layer="reflections", default_strategy="reflexion").run(
        Packet(
            query=Query(
                text="Parse the input stream",
                metadata={"reflexion": {"strategy": "last_trial", "last_attempt": "Attempt missed the edge case."}},
            ),
            retrieved=retrieved,
        ),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")])),
    )
    assert "Below is the last trial you attempted" in last_attempt_packet.readout.text
    assert "Reflection 1:" not in last_attempt_packet.readout.text
    assert last_attempt_packet.readout.source_ids == []


def test_json_readout_returns_json_string() -> None:
    from memprimitive.baselines import JSONReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)

    packet_out, _ = JSONReadout().run(Packet(retrieved=RetrievedSet(items=store.iter_records(), scores=[])), store)

    payload = json.loads(packet_out.readout.text)
    assert payload["items"][0]["text"] == "Alice likes tea."
