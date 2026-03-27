from __future__ import annotations

import pytest

from memprimitive.example.classics.reflexion import (
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
    ReflexionStrategy,
    ReflexionWorkstream,
)


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


QUESTION = "Parse the input stream"


def _workflow(*, strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION, memory_size: int = 3) -> ReflexionWorkstream:
    return ReflexionWorkstream(strategy=strategy, memory_size=memory_size)


def _failed_trial_text(reason: str) -> str:
    return (
        "Thought: inspect the parser boundary conditions.\n"
        f"Action: Finish[wrong answer because {reason}]\n"
        "Observation: Answer is INCORRECT"
    )


def _successful_trial_text() -> str:
    return (
        "Thought: apply the lesson from the failed attempt and handle the edge case.\n"
        "Action: Finish[correct answer]\n"
        "Observation: Answer is CORRECT"
    )


def test_reflexion_generates_reflection_memory_only_for_failed_trials() -> None:
    workflow = _workflow(memory_size=2)

    success_packet = workflow.record_trial(
        question=QUESTION,
        scratchpad=_successful_trial_text(),
        is_correct=True,
        evaluator_feedback="The answer matches the expected output.",
    )
    assert success_packet.evolution_decisions == [False]
    assert workflow.store.count("reflections") == 0

    failure_packet = workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("missing edge case"),
        is_correct=False,
        evaluator_feedback="missing edge case",
    )

    assert failure_packet.evolution_decisions == [True]
    assert failure_packet.trace["evolution_trigger"]["policy"] == "trial_result"
    assert failure_packet.trace["memory_evolution"]["effects"][0]["effect_type"] == "reflection_append"
    assert "missing edge case" in failure_packet.trace["memory_evolution"]["trial_trace"]
    assert workflow.store.count("reflections") == 1

    reflection_record = workflow.store.iter_records("reflections")[0]
    assert reflection_record.metadata["reflexion"]["triggered"] is True
    assert reflection_record.metadata["reflexion"]["question"] == QUESTION


def test_reflexion_strategy_reflexion_only_uses_long_term_memory() -> None:
    workflow = _workflow(strategy=ReflexionStrategy.REFLEXION, memory_size=3)
    workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("missing edge case"),
        is_correct=False,
        evaluator_feedback="missing edge case in parser",
    )

    readout = workflow.build_memory_context(QUESTION, strategy=ReflexionStrategy.REFLEXION)

    assert readout.metadata["strategy"] == ReflexionStrategy.REFLEXION.value
    assert readout.metadata["reflection_count"] == 1
    assert REFLECTION_HEADER in readout.text
    assert LAST_TRIAL_HEADER not in readout.text
    assert readout.source_ids


def test_reflexion_strategy_last_attempt_only_uses_previous_trial_trace() -> None:
    workflow = _workflow(strategy=ReflexionStrategy.LAST_ATTEMPT, memory_size=3)
    failed_scratchpad = _failed_trial_text("wrong boundary condition")
    workflow.record_trial(
        question=QUESTION,
        scratchpad=failed_scratchpad,
        is_correct=False,
        evaluator_feedback="wrong boundary condition",
    )

    readout = workflow.build_memory_context(QUESTION, strategy=ReflexionStrategy.LAST_ATTEMPT)

    assert readout.metadata["strategy"] == ReflexionStrategy.LAST_ATTEMPT.value
    assert readout.metadata["last_attempt_present"] is True
    assert LAST_TRIAL_HEADER in readout.text
    assert "wrong boundary condition" in readout.text
    assert REFLECTION_HEADER not in readout.text
    assert readout.source_ids == []


def test_reflexion_strategy_last_attempt_and_reflexion_orders_sections_like_repo() -> None:
    workflow = _workflow(strategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION, memory_size=3)
    workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("missing edge case"),
        is_correct=False,
        evaluator_feedback="missing edge case",
    )

    readout = workflow.build_memory_context(
        QUESTION,
        strategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
    )

    assert readout.metadata["strategy"] == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION.value
    assert LAST_TRIAL_HEADER in readout.text
    assert REFLECTION_AFTER_LAST_TRIAL_HEADER in readout.text
    assert readout.text.index(LAST_TRIAL_HEADER) < readout.text.index(REFLECTION_AFTER_LAST_TRIAL_HEADER)


def test_reflexion_strategy_none_returns_empty_memory_context() -> None:
    workflow = _workflow(strategy=ReflexionStrategy.NONE, memory_size=3)
    workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("first failure"),
        is_correct=False,
        evaluator_feedback="first failure",
    )

    readout = workflow.build_memory_context(QUESTION, strategy=ReflexionStrategy.NONE)

    assert readout.metadata["strategy"] == ReflexionStrategy.NONE.value
    assert readout.text == ""
    assert readout.source_ids == []


def test_reflexion_sliding_window_prunes_old_reflections_to_memory_size() -> None:
    workflow = _workflow(memory_size=2)

    workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("first failure"),
        is_correct=False,
        evaluator_feedback="first failure",
    )
    workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("second failure"),
        is_correct=False,
        evaluator_feedback="second failure",
    )
    packet = workflow.record_trial(
        question=QUESTION,
        scratchpad=_failed_trial_text("third failure"),
        is_correct=False,
        evaluator_feedback="third failure",
    )

    records = workflow.store.iter_records("reflections")
    assert len(records) == 2
    assert packet.trace["memory_evolution"]["pruned_record_ids"]
    assert packet.trace["memory_evolution"]["retained_record_ids"] == [record.record_id for record in records]
