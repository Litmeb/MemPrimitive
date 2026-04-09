from __future__ import annotations

import json

import pytest

from memprimitive import MemoryStore
from memprimitive.example.classics.reflexion_memory import (
    build_reflexion_memory_system,
    ingest_failed_trial,
    recall_reflection_context,
)


class _FakeReflexionHierarchicalRuntime:
    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        latest = payload["records"][-1]
        reflexion = latest["metadata"]["reflexion"]
        return {
            "reflection": (
                "Reflection: verify the boundary condition first, then compare the final answer "
                "against the earliest valid candidate."
            ),
            "question": reflexion["question"],
            "last_attempt": reflexion["last_attempt"],
            "evaluator_feedback": reflexion["evaluator_feedback"],
            "trial_index": reflexion["trial_index"],
        }


class _IndexedReflexionRuntime:
    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        reflexion = payload["records"][-1]["metadata"]["reflexion"]
        trial_index = reflexion["trial_index"]
        return {
            "reflection": f"Reflection: plan for trial {trial_index}.",
            "question": reflexion["question"],
            "last_attempt": reflexion["last_attempt"],
            "evaluator_feedback": reflexion["evaluator_feedback"],
            "trial_index": trial_index,
        }


class _MemoryAwareReflexionRuntime:
    def __init__(self) -> None:
        self.call_count = 0

    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str):
        self.call_count += 1
        payload = json.loads(user)
        reflexion = payload["records"][-1]["metadata"]["reflexion"]
        if self.call_count == 2:
            assert "Prior retained reflections from persistent memory:" in system
            assert "Reflection: plan for trial 1." in system
        return {
            "reflection": f"Reflection: plan for trial {reflexion['trial_index']}.",
            "question": reflexion["question"],
            "last_attempt": reflexion["last_attempt"],
            "evaluator_feedback": reflexion["evaluator_feedback"],
            "trial_index": reflexion["trial_index"],
        }


def test_reflexion_memory_only_system_appends_trial_and_renders_next_prompt_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeReflexionHierarchicalRuntime())
    system = build_reflexion_memory_system(
        memory_size=2,
        default_strategy="last_trial_and_reflexion",
    )
    store = system["store"]
    assert isinstance(store, MemoryStore)

    packet = ingest_failed_trial(
        system,
        question="Find the first matching index in the stream.",
        last_attempt="I started from position 1 and skipped the first candidate.",
        evaluator_feedback="You ignored the earliest valid match.",
        trial_index=2,
    )

    assert packet.trace["memory_evolution"]["module"] == "hierarchical_evolution"
    assert packet.trace["memory_evolution"]["selection_mode"] == "latest_active_units"
    assert packet.trace["organization"]["module"] == "append_organization"
    assert store.count("trial_buffer") == 1
    assert store.count("reflections") == 1

    trial_record = store.iter_records("trial_buffer")[0]
    assert trial_record.metadata["reflexion"]["trial_index"] == 2

    reflection_record = store.iter_records("reflections")[0]
    field_payload = reflection_record.metadata["hierarchical"]["field_payload"]
    assert field_payload["trial_index"] == 2
    assert field_payload["question"] == "Find the first matching index in the stream."
    assert "verify the boundary condition first" in reflection_record.text

    readout = recall_reflection_context(
        system,
        question="Find the first matching index in the stream.",
        strategy="last_trial_and_reflexion",
        last_attempt="I started from position 1 and skipped the first candidate.",
    )

    assert "Below is the last trial you attempted" in readout.text
    assert "Reflection 1:" in readout.text
    assert "verify the boundary condition first" in readout.text
    assert readout.source_ids == [reflection_record.record_id]
    assert readout.metadata["reflection_count"] == 1


def test_reflexion_memory_only_system_prunes_reflections_to_recent_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _IndexedReflexionRuntime())
    system = build_reflexion_memory_system(memory_size=2)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    for trial_index in (1, 2, 3):
        ingest_failed_trial(
            system,
            question="Stabilize the search routine.",
            last_attempt=f"Attempt {trial_index}",
            evaluator_feedback=f"Feedback {trial_index}",
            trial_index=trial_index,
        )

    assert store.count("trial_buffer") == 3
    assert store.count("reflections") == 2
    assert [record.metadata["hierarchical"]["field_payload"]["trial_index"] for record in store.iter_records("reflections")] == [2, 3]
    assert [record.text for record in store.iter_records("reflections")] == [
        "Reflection: plan for trial 2.",
        "Reflection: plan for trial 3.",
    ]


def test_reflexion_memory_only_system_supports_reflexion_only_readout_without_last_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeReflexionHierarchicalRuntime())
    system = build_reflexion_memory_system(
        memory_size=3,
        default_strategy="reflexion",
    )

    ingest_failed_trial(
        system,
        question="What city is the capital?",
        last_attempt="I answered too quickly from a partial clue.",
        evaluator_feedback="You needed one more verification step.",
        trial_index=1,
    )

    readout = recall_reflection_context(
        system,
        question="What city is the capital?",
        strategy="reflexion",
    )

    assert "Below is the last trial you attempted" not in readout.text
    assert "Reflection 1:" in readout.text
    assert "verify the boundary condition first" in readout.text


def test_reflexion_memory_only_system_skips_reflection_write_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeReflexionHierarchicalRuntime())
    system = build_reflexion_memory_system(memory_size=2)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    packet = ingest_failed_trial(
        system,
        question="What city is the capital?",
        last_attempt="I verified the answer carefully.",
        evaluator_feedback="No issue.",
        trial_index=1,
        is_correct=True,
    )

    assert packet.trace["evolution_trigger"]["decisions"] == [False]
    assert packet.trace["memory_evolution"]["selected_record_count"] == 0
    assert store.count("trial_buffer") == 1
    assert store.count("reflections") == 0


def test_reflexion_memory_only_system_conditions_new_reflections_on_prior_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    runtime = _MemoryAwareReflexionRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", runtime)
    system = build_reflexion_memory_system(memory_size=3)

    ingest_failed_trial(
        system,
        question="Repair the parser.",
        last_attempt="Attempt 1",
        evaluator_feedback="Feedback 1",
        trial_index=1,
    )
    ingest_failed_trial(
        system,
        question="Repair the parser.",
        last_attempt="Attempt 2",
        evaluator_feedback="Feedback 2",
        trial_index=2,
    )

    assert runtime.call_count == 2


def test_reflexion_memory_only_system_prefers_full_trial_trace_over_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeReflexionHierarchicalRuntime())
    system = build_reflexion_memory_system(
        memory_size=2,
        default_strategy="last_trial_and_reflexion",
    )
    store = system["store"]
    assert isinstance(store, MemoryStore)

    packet = ingest_failed_trial(
        system,
        question="Trace the failing branch.",
        last_attempt="I guessed the return value.",
        trial_trace="Thought 1: inspect guard\nAction 1: return 5\nObservation 1: incorrect",
        evaluator_feedback="You skipped the failing branch condition.",
        trial_index=1,
    )

    trial_record = store.iter_records("trial_buffer")[0]
    assert trial_record.text == "Thought 1: inspect guard\nAction 1: return 5\nObservation 1: incorrect"
    assert trial_record.metadata["reflexion"]["last_attempt"] == trial_record.text
    assert packet.observation is not None
    assert packet.observation.text == trial_record.text

    readout = recall_reflection_context(
        system,
        question="Trace the failing branch.",
        strategy="last_trial_and_reflexion",
        trial_trace=trial_record.text,
    )

    assert "Thought 1: inspect guard" in readout.text
