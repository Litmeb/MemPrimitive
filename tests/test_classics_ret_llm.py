from __future__ import annotations

import json
from typing import Any

from baselines_test_helpers import _invoke_runtime_tool

from memprimitive.core import MemoryRecord, Query
from memprimitive.example.classics.ret_llm_memory import build_ret_llm_memory_system


def test_ret_llm_classics_answer_pipeline_can_issue_exact_mem_read() -> None:
    system = build_ret_llm_memory_system(prefetch_top_k=2, exact_top_k=2)
    system.store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="triple_memory",
            text="Alice likes tea.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"triples": [("Alice", "likes", "tea")]}},
        )
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "RET-LLM-style explicit memory" in rendered_prompt
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "Alice >> likes >> *"}))
        assert payload["matched"] is True
        assert payload["source_ids"] == ["rec-1"]
        return f"Answer grounded in memory:\n{payload['memory_text']}"

    system.answer_readout._run_agent = _fake_run_agent.__get__(system.answer_readout, type(system.answer_readout))  # type: ignore[method-assign]
    readout = system.answer_pipeline.recall(Query(text="What does Alice like?"))

    assert "Answer grounded in memory:" in readout.text
    assert "Alice" in readout.text
    assert "likes" in readout.text
    assert "tea" in readout.text
    assert readout.source_ids == ["rec-1"]
    assert readout.metadata["memory_read_count"] == 1
