from __future__ import annotations

import json
from typing import Any

from baselines_test_helpers import _invoke_runtime_tool

from memprimitive.core import MemoryRecord
from memprimitive.example.classics.ret_llm_memory import build_ret_llm_memory_system


def test_ret_llm_classics_answer_pipeline_can_issue_exact_mem_read() -> None:
    system = build_ret_llm_memory_system(mem_read_top_k=2)
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

    def _fake_run_answer_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "RET-LLM-style explicit memory" in rendered_prompt
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "Alice >> likes >> *"}))
        assert payload["matched"] is True
        assert payload["source_ids"] == ["rec-1"]
        assert "Alice >> likes >> tea" in payload["memory_text"]
        assert "record_id=" not in payload["memory_text"]
        assert "source_text=" not in payload["memory_text"]
        return f"Answer grounded in memory:\n{payload['memory_text']}"

    system.answer_agent_runner = _fake_run_answer_agent.__get__(system, type(system))
    answer = system.answer("What does Alice like?")

    assert "Answer grounded in memory:" in answer
    assert "Alice" in answer
    assert "likes" in answer
    assert "tea" in answer


def test_ret_llm_classics_mem_read_falls_back_to_existing_memory_term() -> None:
    system = build_ret_llm_memory_system(mem_read_top_k=2, mem_read_fallback_similarity_threshold=0.7)
    system.store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="triple_memory",
            text="Washington D.C. is the capital of the United States.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"triples": [("Washington D.C.", "capital of", "United States")]}},
        )
    )

    def _fake_run_answer_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "* >> capital of >> USA"}))
        assert payload["matched"] is True
        assert payload["source_ids"] == ["rec-1"]
        assert "Washington D.C." in payload["memory_text"]
        assert "record_id=" not in payload["memory_text"]
        return payload["memory_text"]

    embed_map = {
        "usa": [1.0, 0.0],
        "united states": [1.0, 0.0],
        "capital of": [0.0, 1.0],
        "washington d.c.": [0.0, 0.5],
    }
    system.mem_read_pipeline.retrieval._embed_text = lambda text: embed_map.get(text.strip().casefold(), [0.0, 0.0])  # type: ignore[method-assign]
    system.answer_agent_runner = _fake_run_answer_agent.__get__(system, type(system))

    answer = system.answer("What is the capital of the USA?")

    assert "Washington D.C." in answer
    assert "United States" in answer


def test_ret_llm_classics_mem_read_returns_all_matching_triplets_without_top_k_truncation() -> None:
    system = build_ret_llm_memory_system(mem_read_top_k=1)
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
    system.store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="triple_memory",
            text="Bob likes coffee.",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"representation": {"triples": [("Bob", "likes", "coffee")]}},
        )
    )
    system.store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="triple_memory",
            text="Carol likes juice.",
            timestamp="2026-01-01T00:00:02+00:00",
            metadata={"representation": {"triples": [("Carol", "likes", "juice")]}},
        )
    )

    def _fake_run_answer_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "* >> likes >> *"}))
        assert payload["matched"] is True
        assert payload["source_ids"] == ["rec-1", "rec-2", "rec-3"]
        assert payload["memory_text"].splitlines() == [
            "Matched Triples",
            "Alice >> likes >> tea",
            "Bob >> likes >> coffee",
            "Carol >> likes >> juice",
        ]
        return payload["memory_text"]

    system.answer_agent_runner = _fake_run_answer_agent.__get__(system, type(system))

    answer = system.answer("Who likes what?")

    assert answer.splitlines() == [
        "Matched Triples",
        "Alice >> likes >> tea",
        "Bob >> likes >> coffee",
        "Carol >> likes >> juice",
    ]
