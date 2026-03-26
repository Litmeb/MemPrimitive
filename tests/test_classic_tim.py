from __future__ import annotations

import pytest

from memprimitive import Observation, Query
from memprimitive.classic_modules.tim import TIM_THOUGHT_LAYER, TimWorkstream


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _reasoning_observation(text: str) -> Observation:
    return Observation(
        text=text,
        source="reasoning",
        metadata={"tim": {"reasoning_step": True}},
    )


def test_tim_forms_reasoning_step_units_and_writes_them_to_thought_memory() -> None:
    workflow = TimWorkstream(budget=10, top_k=3, readout_item_budget=3)

    packet = workflow.ingest(
        _reasoning_observation(
            "1. Gather clues.\n2. Test the hypothesis.\n3. Capture the result."
        )
    )

    assert packet.units is not None
    assert len(packet.units) == 3
    assert packet.decisions == [True, True, True]
    assert packet.placements is not None
    assert all(placement.target_layer == TIM_THOUGHT_LAYER for placement in packet.placements)
    assert packet.trace["unit_formation"]["step_count"] == 3
    assert workflow.store.count(TIM_THOUGHT_LAYER) == 3


def test_tim_budget_overflow_compacts_old_thoughts_into_a_summary_record() -> None:
    workflow = TimWorkstream(budget=2, top_k=3, readout_item_budget=3)

    workflow.ingest(_reasoning_observation("First subgoal: map the problem space."))
    workflow.ingest(_reasoning_observation("Second subgoal: check the available evidence."))
    packet = workflow.ingest(_reasoning_observation("Third subgoal: write the answer."))

    records = workflow.store.iter_records(TIM_THOUGHT_LAYER)
    assert len(records) == 2
    assert records[-1].text
    assert "First subgoal" in records[-1].text
    assert "Second subgoal" in records[-1].text
    assert packet.evolution_decisions == [True]
    assert packet.trace["memory_evolution"]["effects"][0]["effect_type"] == "summarize_and_prune"
    assert packet.trace["memory_evolution"]["effects"][0]["pruned_record_ids"]


def test_tim_retrieval_uses_evolved_thought_memory_after_compaction() -> None:
    workflow = TimWorkstream(budget=2, top_k=3, readout_item_budget=3)

    workflow.ingest(_reasoning_observation("First subgoal: map the problem space."))
    workflow.ingest(_reasoning_observation("Second subgoal: check the available evidence."))
    workflow.ingest(_reasoning_observation("Third subgoal: write the answer."))

    summary_record = workflow.store.iter_records(TIM_THOUGHT_LAYER)[-1]
    readout = workflow.recall(Query(text="First subgoal"))

    assert summary_record.record_id in readout.source_ids
    assert readout.text
    assert "First subgoal" in readout.text
