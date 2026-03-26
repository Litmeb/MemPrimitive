from __future__ import annotations

import pytest

from memprimitive import Observation, Query
from memprimitive.classic_modules.reflexion import ReflexionWorkstream


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _failure_observation(task: str, feedback: str, *, source: str = "failure_log") -> Observation:
    return Observation(
        text=f"Task failed: {feedback}.",
        source=source,
        metadata={
            "reflexion": {
                "event": "failure",
                "task": task,
                "feedback": feedback,
            }
        },
    )


def _success_observation(task: str) -> Observation:
    return Observation(
        text=f"Task solved: {task}.",
        source="dialogue",
        metadata={"reflexion": {"event": "success", "task": task}},
    )


def test_reflexion_generates_reflection_memory_only_for_failure_events() -> None:
    workflow = ReflexionWorkstream(reflection_window=2, reflection_top_k=2)

    success_packet = workflow.ingest(_success_observation("Parse the input stream"))
    assert success_packet.evolution_decisions == [False]
    assert workflow.store.count("reflections") == 0

    failure_packet = workflow.ingest(_failure_observation("Parse the input stream", "missing edge case"))

    assert failure_packet.evolution_decisions == [True]
    assert failure_packet.trace["evolution_trigger"]["policy"] == "failure_event"
    assert failure_packet.trace["memory_evolution"]["effects"][0]["effect_type"] == "reflection_append"
    assert workflow.store.count("reflections") == 1

    reflection_record = workflow.store.iter_records("reflections")[0]
    assert "missing edge case" in reflection_record.text
    assert reflection_record.metadata["reflexion"]["triggered"] is True


def test_reflexion_prepends_reflection_context_before_next_task() -> None:
    workflow = ReflexionWorkstream(reflection_window=3, reflection_top_k=2)

    workflow.ingest(_failure_observation("Parse the input stream", "missing edge case"))
    workflow.ingest(_failure_observation("Parse the input stream", "wrong boundary condition"))

    readout = workflow.recall(Query(text="Parse the input stream"))

    lines = readout.text.splitlines()
    assert lines[0] == "Reflection memory"
    assert lines[-1] == "Task: Parse the input stream"
    assert len(readout.source_ids) == 2
    assert readout.metadata["reflection_count"] == 2
    assert "missing edge case" in readout.text
    assert "wrong boundary condition" in readout.text


def test_reflexion_sliding_window_prunes_old_reflections() -> None:
    workflow = ReflexionWorkstream(reflection_window=2, reflection_top_k=2)

    workflow.ingest(_failure_observation("Parse the input stream", "first failure"))
    workflow.ingest(_failure_observation("Parse the input stream", "second failure"))
    packet = workflow.ingest(_failure_observation("Parse the input stream", "third failure"))

    records = workflow.store.iter_records("reflections")
    assert len(records) == 2
    assert all(record.text.startswith("Reflection on") for record in records)
    assert any("second failure" in record.text.lower() for record in records)
    assert any("third failure" in record.text.lower() for record in records)
    assert packet.trace["memory_evolution"]["pruned_record_ids"]
    assert packet.trace["memory_evolution"]["retained_record_ids"] == [record.record_id for record in records]
