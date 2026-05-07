from __future__ import annotations

from typing import Any

import pytest

from baselines_test_helpers import _invoke_runtime_tool
from memprimitive.example.classics.memmachine_memory import (
    build_memmachine_memory_system,
    ingest_episode,
    recall_memmachine_context,
)
from memprimitive.utils._profile_feature_tools import PROFILE_FEATURE_METADATA_KEY


class _FakeMemMachineRuntime:
    def embed(self, text: str) -> list[float]:
        lower = text.casefold()
        return [
            1.0 if "tea" in lower or "jasmine" in lower else 0.0,
            1.0 if "agent" in lower or "memory" in lower else 0.0,
        ]

    def require_llm(self, *, capability: str) -> None:
        _ = capability

    def summarize_records(
        self,
        *,
        records: list[dict[str, Any]],
        instruction: str,
        max_sentences: int,
    ) -> str:
        _ = instruction, max_sentences
        return "summary::" + ",".join(str(record["record_id"]) for record in records)

    def rerank(self, *, query: str, candidates: list[dict[str, Any]], task: str, top_k: int) -> list[dict[str, Any]]:
        _ = query, task
        return [{"id": candidate["id"], "score": 1.0, "rationale": "fake"} for candidate in candidates[:top_k]]


def test_memmachine_classic_wires_contextual_ltm_and_profile_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval, LLMFunctionCallEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeMemMachineRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", lambda self, text: fake_runtime.embed(text))

    def _fake_profile_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        if "jasmine tea" in rendered_prompt:
            _invoke_runtime_tool(
                tools[0],
                {
                    "category": "preference",
                    "tag": "beverage",
                    "feature": "tea",
                    "value": "Alice prefers jasmine tea.",
                    "set_id": "alice",
                },
            )
        _ = self, context
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_profile_agent)

    system = build_memmachine_memory_system(stm_record_budget=1, limit=4)
    ingest_episode(
        system,
        text="Alice is comparing memory systems for long-running agents.",
        session_id="sess-1",
        user_id="alice",
        producer="Alice",
        timestamp="2026-04-28T00:01:00Z",
    )
    ingest_episode(
        system,
        text="Alice says she prefers jasmine tea during late-night coding.",
        session_id="sess-1",
        user_id="alice",
        producer="Alice",
        timestamp="2026-04-28T00:02:00Z",
    )
    ingest_episode(
        system,
        text="The assistant suggests keeping raw episodes available for audit.",
        session_id="sess-1",
        user_id="alice",
        producer="assistant",
        timestamp="2026-04-28T00:03:00Z",
    )

    store = system["store"]
    assert store.count("working") == 1
    assert store.count("episodic") == 3
    assert store.count("sentence") == 3
    assert store.count("session_summary") == 1

    profile_records = store.iter_records("profile")
    assert len(profile_records) == 1
    assert profile_records[0].metadata[PROFILE_FEATURE_METADATA_KEY]["feature"] == "tea"

    context = recall_memmachine_context(system, user_query="What tea does Alice prefer?")
    assert "<LONG TERM MEMORY EPISODES>" in context
    assert "[2026-04-28T00:02:00Z] Alice: Alice says she prefers jasmine tea during late-night coding." in context
    assert "jasmine tea" in context
    assert "[2026-04-28T00:03:00Z] assistant: The assistant suggests keeping raw episodes available for audit." in context
    assert "raw episodes available" in context
    assert "<PROFILE MEMORY>" in context
    assert "Alice prefers jasmine tea." in context


def test_memmachine_profile_evolution_max_turns_is_configurable() -> None:
    system = build_memmachine_memory_system(profile_max_turns=12)
    write_pipeline = system["write_pipeline"]
    profile_evolution = write_pipeline.memory_evolution[0]

    assert profile_evolution.max_turns == 12


def test_memmachine_indexes_ltm_before_stm_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval, LLMFunctionCallEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeMemMachineRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", lambda self, text: fake_runtime.embed(text))
    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", lambda self, **kwargs: "DONE")

    system = build_memmachine_memory_system(stm_record_budget=20, limit=4)
    ingest_episode(
        system,
        text="Caroline went to a LGBTQ support group yesterday.",
        session_id="session_1",
        user_id="conversation:conv-26",
        producer="Caroline",
        timestamp="1:56 pm on 8 May, 2023",
    )

    store = system["store"]
    assert store.count("working") == 1
    assert store.count("episodic") == 1
    assert store.count("sentence") == 1

    recall = recall_memmachine_context(system, user_query="When did Caroline mention the support group?")
    assert "[1:56 pm on 8 May, 2023] Caroline: Caroline went to a LGBTQ support group yesterday." in recall
