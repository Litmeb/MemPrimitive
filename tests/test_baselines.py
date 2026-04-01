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


def _represented_packet(
    text: str,
    *,
    source: str = "dialogue",
    observation_metadata: dict | None = None,
) -> tuple[Packet, MemoryStore]:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    packet = Packet(observation=Observation(text=text, source=source, metadata=observation_metadata or {}))
    packet, store = PassThroughUnitFormation().run(packet, MemoryStore())
    packet, store = BasicRepresentation().run(packet, store)
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


def test_basic_representation_rejects_legacy_triple_element() -> None:
    from memprimitive.baselines import BasicRepresentation

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "triple"))


def test_triple_representation_direct_uses_real_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice works at OpenAI in San Francisco and collaborates with Bob on graph memory systems.",
                source="notes",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = TripleRepresentation(method="direct").run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.triples
    assert unit.metadata["representation"]["triples"] == unit.triples
    assert unit.entities
    assert len(unit.entities) >= 2
    assert "triple" in unit.representation_elements
    assert "entities" in unit.representation_elements
    assert all(len(triple) == 3 for triple in unit.triples)
    flattened = " ".join(" ".join(part for part in triple) for triple in unit.triples).casefold()
    assert "alice" in flattened or "openai" in flattened or "bob" in flattened


def test_triple_representation_two_stage_uses_real_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice mentors Bob at OpenAI, and Bob researches retrieval graphs with Carol.",
                source="notes",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = TripleRepresentation(method="two_stage").run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.triples
    assert unit.entities
    assert unit.metadata["representation"]["triples"] == unit.triples
    entity_set = {entity.casefold() for entity in unit.entities}
    assert any(subject.casefold() in entity_set for subject, _, _ in unit.triples)
    assert any(obj.casefold() in entity_set for _, _, obj in unit.triples)
    assert all(subject and predicate and obj for subject, predicate, obj in unit.triples)


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
    from memprimitive.baselines import AlwaysTrigger, BasicRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)

    packet_out, _ = AlwaysTrigger().run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["module"] == "always_write_trigger"
    assert packet_out.trace["write_trigger"]["threshold"] is None
    assert packet_out.trace["write_trigger"]["constant"] == 1.0
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is True


def test_evolution_trigger_aligns_decisions_with_units() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        AppendOrganization,
        BasicRepresentation,
        NeverTrigger,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = AppendOrganization().run(packet, store)

    packet_out, _ = NeverTrigger().run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.trace["evolution_trigger"]["module"] == "never_evolution_trigger"
    assert packet_out.trace["evolution_trigger"]["decisions"] == [False]
    assert packet_out.trace["evolution_trigger"]["threshold"] is None
    assert packet_out.trace["evolution_trigger"]["constant"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is False


def test_organization_aligns_placements_with_units_and_commits_normal_write() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)

    packet_out, updated_store = AppendOrganization().run(packet, store)

    assert packet_out.placements is not None
    assert len(packet_out.placements) == len(packet_out.units)
    assert packet_out.placements[0].target_layer == "default"
    assert updated_store.count() == 1
    assert packet_out.trace["organization"]["written_record_ids"]
    assert packet_out.trace["organization"]["written_unit_ids"] == [packet_out.units[0].unit_id]
    assert packet_out.trace["organization"]["skipped_unit_count"] == 0


def test_append_only_evolution_is_noop_when_decisions_are_false() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        decisions=[False],
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
        decisions=[True],
        placements=packet.placements,
        trace=packet.trace,
    )

    packet_out, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 1
    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions"
    assert packet_out.trace["memory_evolution"]["active_unit_ids"] == [packet.units[0].unit_id]
    assert packet_out.trace["memory_evolution"]["effects"] == []


def test_append_only_evolution_requires_explicit_decisions() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        trace=packet.trace,
    )

    with pytest.raises(ValueError, match="packet.decisions"):
        AppendOnlyEvolution().run(packet, store)


def test_append_only_evolution_requires_aligned_inputs() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    with pytest.raises(ValueError, match="aligned units"):
        AppendOnlyEvolution().run(
            Packet(units=[], decisions=[True], placements=[]),
            MemoryStore(),
        )


def test_write_and_evolution_trigger_are_independent_by_default() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        AppendOrganization,
        BasicRepresentation,
        NeverTrigger,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    write_packet, store = AlwaysTrigger().run(packet, store)
    write_packet, store = AppendOrganization().run(write_packet, store)
    evolution_packet, _ = NeverTrigger().run(write_packet, store)

    assert write_packet.decisions == [True]
    assert evolution_packet.decisions == [False]
    assert write_packet.trace["write_trigger"]["module"] == "always_write_trigger"
    assert evolution_packet.trace["evolution_trigger"]["module"] == "never_evolution_trigger"


def test_threshold_write_trigger_respects_threshold_policy() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation, ThresholdTrigger

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)

    packet_out, _ = ThresholdTrigger(threshold=0.8, constant=0.7).run(packet, store)
    assert packet_out.decisions == [False]
    assert packet_out.trace["write_trigger"]["threshold"] == 0.8
    assert packet_out.trace["write_trigger"]["constant"] == 0.7
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is False

    packet_out, _ = ThresholdTrigger(threshold=0.7, constant=0.7).run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is True


def test_threshold_evolution_trigger_writes_only_decisions() -> None:
    from memprimitive.baselines import (
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
        ThresholdTrigger,
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

    packet_out, _ = ThresholdTrigger(slot="evolution_trigger", threshold=2.0, constant=1.0).run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.trace["evolution_trigger"]["threshold"] == 2.0
    assert packet_out.trace["evolution_trigger"]["constant"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is False


def test_on_input_trigger_filters_by_observation_source() -> None:
    from memprimitive.baselines import OnInputTrigger

    packet, store = _represented_packet("Alice likes tea.", source="dialogue")

    packet_out, _ = OnInputTrigger(allowed_sources=("dialogue",)).run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["module"] == "on_input_write_trigger"
    assert packet_out.trace["write_trigger"]["observation_source"] == "dialogue"

    blocked_packet, _ = OnInputTrigger(allowed_sources=("notes",)).run(packet, store)
    assert blocked_packet.decisions == [False]


def test_boundary_event_trigger_matches_structural_events_for_both_slots() -> None:
    from memprimitive.baselines import AppendOrganization, BoundaryEventTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": ["turn_end", "session_end"]}},
    )
    write_packet, store = BoundaryEventTrigger(accepted_events=("session_end",)).run(packet, store)

    assert write_packet.decisions == [True]
    assert write_packet.trace["write_trigger"]["source"] == "boundary"
    assert write_packet.trace["write_trigger"]["matched_events"] == ["session_end"]

    organized_packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )
    evolution_packet, _ = BoundaryEventTrigger(
        slot="evolution_trigger",
        accepted_events=("session_end",),
    ).run(organized_packet, store)

    assert evolution_packet.decisions == [True]
    assert evolution_packet.trace["evolution_trigger"]["source"] == "boundary"


def test_runtime_event_trigger_uses_packet_events_when_trigger_metadata_missing() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    packet, store = _represented_packet("Alice likes tea.")
    packet = replace(packet, events=["task_failed"])
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            events=packet.events,
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(accepted_events=("task_failed",)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["evolution_trigger"]["source"] == "runtime"
    assert packet_out.trace["evolution_trigger"]["observed_events"] == ["task_failed"]


def test_scalar_rule_trigger_supports_broadcast_and_per_unit_modes() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"signals": {"importance": 0.82}}},
    )
    packet_out, _ = ScalarRuleTrigger(signal_key="importance", threshold=0.8).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["signal_key"] == "importance"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["signal_value"] == 0.82

    multi_unit = replace(
        packet,
        units=[
            replace(packet.units[0], unit_id="unit-a", metadata={"importance": 0.9}),
            replace(packet.units[0], unit_id="unit-b", metadata={"importance": 0.3}),
        ],
    )
    per_unit_packet, _ = ScalarRuleTrigger(
        signal_key="importance",
        threshold=0.5,
        aggregate="per_unit",
    ).run(multi_unit, store)

    assert per_unit_packet.decisions == [True, False]
    assert per_unit_packet.trace["write_trigger"]["aggregate"] == "per_unit"


def test_model_judge_trigger_supports_injected_per_unit_and_broadcast_modes() -> None:
    from memprimitive.baselines import AppendOrganization, ModelJudgeTrigger

    packet, store = _represented_packet("Alice likes tea.")

    def per_unit_judge(payload: dict) -> dict:
        return {"decision": "alice" in payload["unit"]["text"].casefold(), "score": 0.9, "label": "write"}

    packet_out, _ = ModelJudgeTrigger(system_prompt="Judge writes.", judge_callable=per_unit_judge).run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["source"] == "model_judge"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["score"] == 0.9

    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    def broadcast_judge(payload: dict) -> dict:
        assert payload["unit"] is None
        return {"score": 0.75}

    evolution_packet, _ = ModelJudgeTrigger(
        slot="evolution_trigger",
        system_prompt="Judge evolution.",
        decision_mode="score",
        threshold=0.7,
        per_unit=False,
        judge_callable=broadcast_judge,
    ).run(packet, store)
    assert evolution_packet.decisions == [True]
    assert evolution_packet.trace["evolution_trigger"]["per_unit"][0]["score"] == 0.75


def test_periodic_and_idle_maintenance_triggers_gate_evolution_from_schedule_metadata() -> None:
    from memprimitive.baselines import AppendOrganization, IdleMaintenanceTrigger, PeriodicMaintenanceTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 12, "idle_seconds": 45.0}, "events": ["idle"]}},
    )
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(every_n=3).run(packet, store)
    assert periodic_packet.decisions == [True]
    assert periodic_packet.trace["evolution_trigger"]["tick"] == 12

    idle_packet, _ = IdleMaintenanceTrigger(min_idle_seconds=30.0).run(packet, store)
    assert idle_packet.decisions == [True]
    assert idle_packet.trace["evolution_trigger"]["idle_seconds"] == 45.0


def test_new_trigger_classes_are_registered_in_baseline_exports() -> None:
    exported = registered_baseline_class_names()

    assert {
        "OnInputTrigger",
        "BoundaryEventTrigger",
        "RuntimeEventTrigger",
        "ScalarRuleTrigger",
        "ModelJudgeTrigger",
        "PeriodicMaintenanceTrigger",
        "IdleMaintenanceTrigger",
    }.issubset(exported)


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


def test_retrieval_returns_latest_records_first_even_when_query_matches_older_records() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob prefers coffee", "Alice studies graphs"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2
    assert [record.text for record in packet_out.retrieved.items] == [
        "Alice studies graphs",
        "Bob prefers coffee",
    ]


def test_retrieval_returns_latest_records_first_regardless_of_query_text() -> None:
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
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True for _ in packet.units or []],
            trace=packet.trace,
        ),
        store,
    )

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
        packet, store = AppendOrganization(target_layer="episodic").run(
            Packet(
                observation=packet.observation,
                units=packet.units,
                decisions=[True for _ in packet.units or []],
                trace=packet.trace,
            ),
            store,
        )

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


def test_removed_trigger_family_symbols_are_not_registered() -> None:
    removed = {
        "MetadataGatedWriteTrigger",
        "KeyReadyWriteTrigger",
        "LLMJudgedWriteTrigger",
        "OutcomeConditionedEvolutionTrigger",
        "NewWriteEvolutionTrigger",
        "NeighborExistsEvolutionTrigger",
    }

    assert registered_baseline_class_names().isdisjoint(removed)


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
        decisions=[False],
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
    from memprimitive.baselines import AppendOrganization, AlwaysTrigger, BasicRepresentation, PassThroughUnitFormation

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
    packet, store = AlwaysTrigger().run(packet, store)
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




def test_conditional_layer_organization_routes_entity_rich_units_to_semantic() -> None:
    from memprimitive.baselines import AlwaysTrigger, BasicRepresentation, ConditionalLayerOrganization, PassThroughUnitFormation

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
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = ConditionalLayerOrganization(
        default_layer="working",
        rules=({"has_entity": True, "target_layer": "semantic"},),
    ).run(packet, store)

    assert packet.placements[0].target_layer == "semantic"
    assert store.count("semantic") == 1


def test_graph_append_organization_requires_graph_layer_and_writes_graph_metadata() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            return represented, {"source": "test_seed", "entities": entities, "triple_count": len(triples)}

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
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
        decisions=[True],
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
        decisions=[True],
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
        decisions=[True],
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


def test_graph_baseline_pipeline_end_to_end_supports_threshold_trigger_evolution_retrieval_and_readout() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import (
        BasicRepresentation,
        GraphAppendOrganization,
        GraphLinkEvolution,
        GraphNeighborContextTraceEvolution,
        GraphReadout,
        GraphSeedAndExpandRetrieval,
        PassThroughUnitFormation,
        ThresholdTrigger,
        TripleRepresentation,
    )

    class SeededTripleRepresentation(TripleRepresentation):
        _TRIPLES_BY_TEXT = {
            "Alice likes jasmine tea.": ([("Alice", "likes", "jasmine tea")], ["Alice", "jasmine tea"]),
            "Alice studies graph memory systems.": (
                [("Alice", "studies", "graph memory systems")],
                ["Alice", "graph memory systems"],
            ),
            "Bob builds retrieval tools.": ([("Bob", "builds", "retrieval tools")], ["Bob", "retrieval tools"]),
        }

        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples, entities = self._TRIPLES_BY_TEXT[unit.text.strip()]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            return represented, {"source": "test_seed", "entities": entities, "triple_count": len(triples)}

    store = _graph_vector_store()
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=(
            BasicRepresentation(elements=("text", "embedding")),
            SeededTripleRepresentation(),
            BasicRepresentation(elements=("tags", "keywords")),
        ),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0),
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

    assert first_packet.trace["write_trigger"]["decisions"] == [True]
    assert second_packet.trace["write_trigger"]["decisions"] == [True]
    assert first_packet.decisions == [True]
    assert second_packet.decisions == [True]
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


def test_graph_append_link_ready_organization_does_not_eagerly_validate_graph_vector_layer() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import GraphAppendLinkReadyOrganization

    bad_store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory_graph", shape="Graph", indices=("graph", "keyword", "tag"))]
        )
    )

    pipeline = MemoryPipeline(
        store=bad_store,
        organization=GraphAppendLinkReadyOrganization(target_layer="memory_graph"),
    )

    assert isinstance(pipeline.organization, GraphAppendLinkReadyOrganization)


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
        decisions=[True],
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
        decisions=[True],
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
        decisions=[True],
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
        decisions=[True],
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
    from memprimitive.baselines import AlwaysTrigger, AppendOrganization, BasicRepresentation, PassThroughUnitFormation, TagRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Alice studies graph memory", "Bob likes coffee"):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = BasicRepresentation(elements=("text", "tags")).run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
        _, store = AppendOrganization().run(packet, store)

    packet_out, _ = TagRetrieval(top_k=1).run(Packet(query=Query(text="graph")), store)

    assert packet_out.retrieved.items[0].text == "Alice studies graph memory"


def test_entity_retrieval_prefers_entity_overlap() -> None:
    from memprimitive.baselines import AlwaysTrigger, AppendOrganization, BasicRepresentation, EntityRetrieval, PassThroughUnitFormation

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob likes coffee", "Alice studies graph memory"):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = BasicRepresentation(elements=("text", "entities")).run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
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
