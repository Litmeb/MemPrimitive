from __future__ import annotations

from dataclasses import replace
import json
from typing import Any
import pytest

from memprimitive.baselines.registry import (
    registered_baseline_class_names,
)
from memprimitive.core import (
    MemoryStore,
    Observation,
    Packet,
    Placement,
    StoreLayerSpec,
    StoreTopology,
)

from baselines_test_helpers import (
    _budgeted_store,
    _represented_packet,
    _seed_layer,
    _seed_layer_with_metadata,
    _stored_pipeline_packet,
)


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


@pytest.mark.parametrize(
    ("event_name", "match_key", "match_value"),
    [
        ("session_end", "session_id", "sess-1"),
        ("turn_end", "turn_id", "turn-1"),
        ("chunk_end", "chunk_id", "chunk-1"),
        ("subgoal_end", "subgoal_id", "subgoal-1"),
        ("episode_end", "episode_id", "episode-1"),
    ],
)
def test_boundary_event_trigger_populates_decisions_store_for_matching_boundary(
    event_name: str,
    match_key: str,
    match_value: str,
) -> None:
    from memprimitive.baselines import BoundaryEventTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic", theme="episode"),
            ]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "default match", "metadata": {match_key: match_value}},
            {"text": "default miss", "metadata": {match_key: "other"}},
        ],
    )
    _seed_layer_with_metadata(
        store,
        "episodic",
        [
            {"text": "episodic match", "metadata": {match_key: match_value}},
            {"text": "episodic other", "metadata": {"session_id": "other-session"}},
        ],
    )
    packet, _ = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": [event_name], match_key: match_value}},
    )

    packet_out, _ = BoundaryEventTrigger(accepted_events=(event_name,)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"default", "episodic"}
    assert packet_out.decisions_store["default"]["record_ids"] == ["rec-1"]
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-3"]
    assert packet_out.trace["write_trigger"]["boundary_kind"] == match_key.removesuffix("_id")
    assert packet_out.trace["write_trigger"]["match_key"] == match_key
    assert packet_out.trace["write_trigger"]["match_value"] == match_value
    assert packet_out.trace["write_trigger"]["decisions_store_counts"] == {"default": 1, "episodic": 1}


def test_boundary_event_trigger_keeps_decisions_but_records_missing_match_key() -> None:
    from memprimitive.baselines import BoundaryEventTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": ["session_end"]}},
    )

    packet_out, _ = BoundaryEventTrigger(accepted_events=("session_end",)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is None
    assert packet_out.trace["write_trigger"]["missing_match_key"] is True
    assert packet_out.trace["write_trigger"]["match_key"] == "session_id"
    assert packet_out.trace["write_trigger"]["match_value"] is None
    assert packet_out.trace["write_trigger"]["decisions_store_layers"] == []


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


def test_runtime_event_trigger_computes_memory_pressure_event_from_record_budget() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=2)
    _seed_layer(store, "episodic", ["one", "two"])
    packet, _ = _represented_packet("Alice likes tea.")
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(
        accepted_events=("memory_pressure",),
        pressure_threshold=1.0,
    ).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["evolution_trigger"]["matched_events"] == ["memory_pressure"]
    assert packet_out.trace["evolution_trigger"]["computed_runtime_events"] == ["memory_pressure"]
    assert packet_out.trace["evolution_trigger"]["record_pressure"] == 1.5
    assert packet_out.trace["evolution_trigger"]["token_pressure"] is None
    assert packet_out.trace["evolution_trigger"]["target_layer"] == "episodic"
    assert packet_out.decisions_store is not None
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2", "rec-3"]
    assert packet_out.decisions_store["episodic"]["selector"]["kind"] == "layer_all"
    assert packet_out.trace["evolution_trigger"]["decisions_store_counts"] == {"episodic": 3}


def test_runtime_event_trigger_keeps_literal_memory_pressure_event_without_threshold() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=10)
    packet, _ = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": ["memory_pressure"]}},
    )
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(accepted_events=("memory_pressure",)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["evolution_trigger"]["observed_events"] == ["memory_pressure"]
    assert packet_out.trace["evolution_trigger"]["computed_runtime_events"] == []
    assert packet_out.trace["evolution_trigger"]["pressure_threshold"] is None
    assert packet_out.decisions_store is not None
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1"]


def test_runtime_event_trigger_skips_decisions_store_when_memory_pressure_not_triggered() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=10)
    packet, _ = _represented_packet("Alice likes tea.")
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(
        accepted_events=("memory_pressure",),
        pressure_threshold=2.0,
    ).run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.decisions_store is None
    assert packet_out.trace["evolution_trigger"]["decisions_store_layers"] == []


def test_runtime_event_trigger_memory_pressure_requires_single_layer_for_broadcast_resolution() -> None:
    from memprimitive.baselines import RuntimeEventTrigger

    packet, store = _represented_packet("Alice likes tea.")
    packet = replace(
        packet,
        placements=[
            Placement(unit_id="unit-a", target_layer="default"),
            Placement(unit_id="unit-b", target_layer="episodic"),
        ],
        units=[replace(packet.units[0], unit_id="unit-a"), replace(packet.units[0], unit_id="unit-b")],
    )
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default", capacity="unlimited", settings={"record_budget": 2}),
                StoreLayerSpec(name="episodic", capacity="unlimited", settings={"record_budget": 2}),
            ]
        )
    )

    with pytest.raises(ValueError, match="single target layer"):
        RuntimeEventTrigger(
            accepted_events=("memory_pressure",),
            pressure_threshold=0.5,
        ).run(packet, store)


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


def test_scalar_rule_trigger_memory_pressure_supports_record_and_token_budgets() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=4, token_budget=3)
    _seed_layer(store, "episodic", ["alpha beta", "gamma delta"])
    packet, _ = _represented_packet("Alice likes tea.")

    packet_out, _ = ScalarRuleTrigger(
        signal_key="memory_pressure",
        threshold=1.0,
        target_layer="episodic",
    ).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["target_layer_mode"] == "explicit"
    assert packet_out.trace["write_trigger"]["record_pressure"] == 0.5
    assert packet_out.trace["write_trigger"]["token_pressure"] == pytest.approx(4 / 3)
    assert packet_out.trace["write_trigger"]["memory_pressure"] == pytest.approx(4 / 3)
    assert packet_out.trace["write_trigger"]["active_budget_types"] == ["record_budget", "token_budget"]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["target_layer"] == "episodic"
    assert packet_out.decisions_store is not None
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2"]
    assert packet_out.decisions_store["episodic"]["selector"]["kind"] == "layer_all"
    assert packet_out.decisions_store["episodic"]["selector"]["source"] == "scalar_rule"
    assert packet_out.trace["write_trigger"]["decisions_store_counts"] == {"episodic": 2}


def test_scalar_rule_trigger_memory_pressure_supports_per_unit_resolution_from_placements() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working", capacity="unlimited", settings={"record_budget": 4}),
                StoreLayerSpec(name="semantic", capacity="unlimited", settings={"record_budget": 2}),
            ]
        )
    )
    _seed_layer(store, "working", ["one"])
    _seed_layer(store, "semantic", ["one", "two"])
    packet, _ = _represented_packet("Alice likes tea.")
    packet = replace(
        packet,
        units=[
            replace(packet.units[0], unit_id="unit-a"),
            replace(packet.units[0], unit_id="unit-b"),
        ],
        placements=[
            Placement(unit_id="unit-a", target_layer="working"),
            Placement(unit_id="unit-b", target_layer="semantic"),
        ],
    )

    packet_out, _ = ScalarRuleTrigger(
        slot="evolution_trigger",
        signal_key="memory_pressure",
        threshold=0.75,
        aggregate="per_unit",
    ).run(packet, store)

    assert packet_out.decisions == [False, True]
    assert packet_out.trace["evolution_trigger"]["target_layer_mode"] == "placement"
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["target_layer"] == "working"
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["record_pressure"] == 0.25
    assert packet_out.trace["evolution_trigger"]["per_unit"][1]["target_layer"] == "semantic"
    assert packet_out.trace["evolution_trigger"]["per_unit"][1]["record_pressure"] == 1.0
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"semantic"}
    assert packet_out.decisions_store["semantic"]["record_ids"] == ["rec-2", "rec-3"]
    assert packet_out.trace["evolution_trigger"]["decisions_store_counts"] == {"semantic": 2}


def test_scalar_rule_trigger_memory_pressure_requires_explicit_write_layer() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    packet, store = _represented_packet("Alice likes tea.")

    with pytest.raises(ValueError, match="explicit target_layer"):
        ScalarRuleTrigger(signal_key="memory_pressure", threshold=0.8).run(packet, store)


def test_llm_judge_trigger_supports_per_unit_and_broadcast_modes() -> None:
    from memprimitive.baselines import AppendOrganization, LLMJudgeTrigger

    packet, store = _represented_packet("Alice likes tea.")

    trigger = LLMJudgeTrigger(
        prompt="Judge whether {{ unit.text }} should be written in {{ slot }}.",
    )

    def _fake_per_unit_llm_json(*, user: str) -> Any:
        payload = json.loads(user)
        assert payload["prompt"] == "Judge whether Alice likes tea. should be written in write_trigger."
        assert payload["judge_context"]["unit"]["text"] == "Alice likes tea."
        return {"decision": True, "score": 0.9, "label": "write"}

    trigger._llm_json = _fake_per_unit_llm_json  # type: ignore[method-assign]
    packet_out, _ = trigger.run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["source"] == "llm_judge"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["score"] == 0.9
    assert packet_out.trace["write_trigger"]["per_unit"][0]["rendered_prompt"].startswith("Judge whether Alice")

    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    trigger = LLMJudgeTrigger(
        slot="evolution_trigger",
        prompt="Judge whether evolution should run for {{ store_summary.record_count }} stored records.",
        decision_mode="score",
        threshold=0.7,
        per_unit=False,
    )

    def _fake_broadcast_llm_json(*, user: str) -> Any:
        payload = json.loads(user)
        assert payload["judge_context"]["unit"] is None
        assert payload["judge_context"]["placement"] is None
        assert payload["prompt"] == "Judge whether evolution should run for 1 stored records."
        return {"score": 0.75}

    trigger._llm_json = _fake_broadcast_llm_json  # type: ignore[method-assign]
    evolution_packet, _ = trigger.run(packet, store)
    assert evolution_packet.decisions == [True]
    assert evolution_packet.trace["evolution_trigger"]["per_unit"][0]["score"] == 0.75


def test_llm_judge_trigger_prompt_template_supports_recalled_prompt() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMJudgeTrigger, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"session_id": "sess-judge"},
    )
    _seed_layer(store, "default", ["CURRENT JUDGE PROFILE"])

    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    trigger = LLMJudgeTrigger(
        prompt=text_prompt(
            "Decide for {{ unit.text }} with {{ recalled_prompt }} and {{ unit.metadata.session_id }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"profile for {context['unit']['text']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
    )

    def _fake_llm_json(*, user: str) -> Any:
        payload = json.loads(user)
        assert payload["prompt"] == "Decide for Alice likes tea. with CURRENT JUDGE PROFILE and sess-judge"
        assert payload["judge_context"]["unit"]["text"] == "Alice likes tea."
        return {"decision": True, "label": "write"}

    trigger._llm_json = _fake_llm_json  # type: ignore[method-assign]
    packet_out, _ = trigger.run(packet, store)

    prompt_trace = packet_out.trace["write_trigger"]["per_unit"][0]
    assert packet_out.decisions == [True]
    assert prompt_trace["recall_prompt"]["enabled"] is True
    assert prompt_trace["recall_prompt"]["rendered_recall_query"] == "profile for Alice likes tea."
    assert prompt_trace["recalled_prompt"] == "CURRENT JUDGE PROFILE"


def test_periodic_maintenance_trigger_runs_wrapped_trigger_when_schedule_matches() -> None:
    from memprimitive.baselines import AppendOrganization, PeriodicMaintenanceTrigger, ScalarRuleTrigger

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

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=ScalarRuleTrigger(
            slot="evolution_trigger",
            signal_key="importance",
            threshold=0.5,
        ),
    ).run(
        replace(
            packet,
            observation=replace(
                packet.observation,
                metadata={"trigger": {"schedule": {"tick": 12}, "signals": {"importance": 0.9}}},
            ),
        ),
        store,
    )

    assert periodic_packet.decisions == [True]
    assert periodic_packet.trace["evolution_trigger"]["module"] == "scalar_rule_evolution_trigger"
    assert periodic_packet.trace["evolution_trigger"]["tick"] == 12
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is True
    assert periodic_packet.trace["evolution_trigger"]["wrapped_trigger_module"] == "scalar_rule_evolution_trigger"
    assert periodic_packet.trace["evolution_trigger"]["signal_key"] == "importance"


def test_periodic_maintenance_trigger_preserves_existing_decisions_on_miss() -> None:
    from memprimitive.baselines import AppendOrganization, PeriodicMaintenanceTrigger, NeverTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 11}}},
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

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=NeverTrigger(slot="evolution_trigger"),
    ).run(packet, store)

    assert periodic_packet.decisions == [True]
    assert periodic_packet.trace["evolution_trigger"]["module"] == "periodic_maintenance_evolution_trigger"
    assert periodic_packet.trace["evolution_trigger"]["tick"] == 11
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is False
    assert periodic_packet.trace["evolution_trigger"]["wrapped_trigger_module"] == "never_evolution_trigger"


def test_periodic_maintenance_trigger_keeps_none_decisions_on_miss() -> None:
    from memprimitive.baselines import PeriodicMaintenanceTrigger, NeverTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 11}}},
    )
    packet = replace(
        packet,
        placements=[Placement(unit_id=packet.units[0].unit_id, target_layer="default")],
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=NeverTrigger(slot="evolution_trigger"),
    ).run(packet, store)

    assert periodic_packet.decisions is None
    assert periodic_packet.trace["evolution_trigger"]["decisions"] is None
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is False


def test_periodic_maintenance_trigger_uses_store_counter_when_schedule_tick_missing() -> None:
    from memprimitive.baselines import AppendOrganization, NeverTrigger, PeriodicMaintenanceTrigger

    packet, store = _represented_packet("Alice likes tea.")
    store = replace(store, metadata={**store.metadata, "ingest_count": 6})
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=NeverTrigger(slot="evolution_trigger"),
    ).run(packet, store)

    assert periodic_packet.trace["evolution_trigger"]["tick"] == 6
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is True
    assert periodic_packet.decisions == [False]


def test_periodic_maintenance_trigger_rejects_wrapped_trigger_slot_mismatch() -> None:
    from memprimitive.baselines import AlwaysTrigger, PeriodicMaintenanceTrigger

    with pytest.raises(ValueError, match="wrapped trigger slot"):
        PeriodicMaintenanceTrigger(
            every_n=3,
            trigger=AlwaysTrigger(slot="write_trigger"),
        )


def test_idle_maintenance_trigger_gates_evolution_from_schedule_metadata() -> None:
    from memprimitive.baselines import AppendOrganization, IdleMaintenanceTrigger

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

    idle_packet, _ = IdleMaintenanceTrigger(min_idle_seconds=30.0).run(packet, store)
    assert idle_packet.decisions == [True]
    assert idle_packet.trace["evolution_trigger"]["idle_seconds"] == 45.0


def test_store_all_trigger_preserves_existing_decisions_and_selects_all_layers() -> None:
    from memprimitive.baselines import AlwaysTrigger, StoreAllTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
                StoreLayerSpec(name="semantic"),
            ]
        )
    )
    _seed_layer(store, "default", ["default one"])
    _seed_layer(store, "episodic", ["episodic one", "episodic two"])
    _seed_layer(store, "semantic", ["semantic one"])

    packet, _ = _represented_packet("Alice likes tea.")
    packet, _ = AlwaysTrigger().run(packet, store)

    packet_out, _ = StoreAllTrigger().run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"default", "episodic", "semantic"}
    assert packet_out.decisions_store["default"]["record_ids"] == ["rec-1"]
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-2", "rec-3"]
    assert packet_out.decisions_store["semantic"]["record_ids"] == ["rec-4"]
    assert packet_out.decisions_store["semantic"]["selector"]["kind"] == "store_all"
    assert packet_out.decisions_store["semantic"]["selector"]["source"] == "store_all_trigger"
    assert packet_out.trace["write_trigger"]["module"] == "store_all_write_trigger"
    assert packet_out.trace["write_trigger"]["decisions"] == [True]
    assert packet_out.trace["write_trigger"]["decisions_store_counts"] == {
        "default": 1,
        "episodic": 2,
        "semantic": 1,
    }


def test_store_all_trigger_keeps_decisions_none_when_no_prior_decision_exists() -> None:
    from memprimitive.baselines import StoreAllTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    _seed_layer(store, "episodic", ["episodic one"])
    packet, _ = _represented_packet("Alice likes tea.")

    packet_out, _ = StoreAllTrigger().run(packet, store)

    assert packet_out.decisions is None
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"episodic"}
    assert packet_out.trace["write_trigger"]["decisions"] is None
    assert packet_out.trace["write_trigger"]["per_unit"] == []
    assert packet_out.trace["write_trigger"]["preserved_decisions"] is False


def test_store_all_trigger_supports_evolution_slot() -> None:
    from memprimitive.baselines import AppendOrganization, StoreAllTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    _seed_layer(store, "episodic", ["prior one"])
    packet, _ = _represented_packet("Alice likes tea.")
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = StoreAllTrigger(slot="evolution_trigger").run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"episodic"}
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2"]
    assert packet_out.trace["evolution_trigger"]["module"] == "store_all_evolution_trigger"


def test_new_trigger_classes_are_registered_in_baseline_exports() -> None:
    exported = registered_baseline_class_names()

    assert {
        "BoundaryEventTrigger",
        "RuntimeEventTrigger",
        "ScalarRuleTrigger",
        "StoreAllTrigger",
        "LLMJudgeTrigger",
        "PeriodicMaintenanceTrigger",
        "IdleMaintenanceTrigger",
    }.issubset(exported)


def test_hierarchical_classes_are_registered_in_baseline_exports() -> None:
    exported = registered_baseline_class_names()

    assert {
        "HierarchicalOrganization",
        "HierarchicalEvolution",
    }.issubset(exported)


def test_query_rewrite_retrieval_is_registered_in_baseline_exports() -> None:
    assert "QueryRewriteRetrieval" in registered_baseline_class_names()


def test_metadata_retrieval_is_registered_in_baseline_exports() -> None:
    assert "MetadataRetrieval" in registered_baseline_class_names()


def test_graph_deduplication_append_organization_is_registered_in_baseline_exports() -> None:
    assert "GraphDeduplicationAppendOrganization" in registered_baseline_class_names()


def test_graph_entity_deduplication_append_organization_is_registered_in_baseline_exports() -> None:
    assert "GraphEntityDeduplicationAppendOrganization" in registered_baseline_class_names()

