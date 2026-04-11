from __future__ import annotations

from typing import Any

import pytest

from baselines_test_helpers import _invoke_runtime_tool
from memprimitive.example.classics.tim_memory import (
    _build_bucket_recall_pipeline,
    _build_tim_thought_write_pipeline,
    build_tim_memory_system,
    post_think_and_update_memory,
    recall_thoughts,
)
from memprimitive.utils import _runtime


class _FakeTiMRuntime:
    def __init__(self) -> None:
        self.extractions: list[list[dict[str, str]]] = [
            [
                {
                    "thought": "Alice likes jasmine tea.",
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea",
                }
            ],
            [
                {
                    "thought": "Alice likes jasmine tea in the evening.",
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea in the evening",
                }
            ],
        ]
        self.embedding_map = {
            "Alice likes jasmine tea.": [1.0, 0.0],
            "Alice likes jasmine tea in the evening.": [1.0, 0.0],
            "What should the assistant remember about Alice's tea habit?": [1.0, 0.0],
        }

    def json(self, *, system: str, user: str) -> Any:
        _ = system
        _ = user
        if not self.extractions:
            raise AssertionError("Unexpected extra thought-extraction call.")
        return self.extractions.pop(0)

    def embed(self, text: str) -> list[float]:
        if text not in self.embedding_map:
            raise AssertionError(f"Unexpected embedding request: {text!r}")
        return list(self.embedding_map[text])


def test_tim_classics_builder_uses_existing_baselines() -> None:
    system = build_tim_memory_system(bucket_count=4)

    write_pipeline = _build_tim_thought_write_pipeline(system, hash_bucket="bucket-0")
    recall_pipeline = _build_bucket_recall_pipeline(
        system,
        hash_bucket="bucket-0",
        final_top_k=2,
        prompt="Thoughts:\n{{ retrieved.items | join_text }}",
    )

    assert write_pipeline.representation.spec.name == "basic_representation"
    assert write_pipeline.organization.spec.name == "append_organization"
    assert write_pipeline.memory_evolution.spec.name == "llm_function_call_evolution"
    assert [spec.name for spec in write_pipeline.memory_evolution.tool_specs] == ["UPDATE", "DELETE"]

    retrieval = recall_pipeline.retrieval
    assert isinstance(retrieval, tuple)
    assert [module.spec.name for module in retrieval] == [
        "metadata_retrieval",
        "embedding_similarity_retrieval",
    ]


def test_tim_classics_post_think_merge_uses_llm_function_call_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    fake_runtime = _FakeTiMRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        selected_id = context["selected_record_ids"][0]
        if selected_id == "rec-1":
            assert "rec-1" in rendered_prompt
            return "NO_ACTION"
        assert selected_id == "rec-2"
        assert "rec-1" in rendered_prompt
        assert "rec-2" in rendered_prompt
        _invoke_runtime_tool(
            tools[0],
            {
                "record_id": "rec-1",
                "text": "Alice likes jasmine tea in the evening.",
                "metadata_patch": {
                    "tail": "jasmine tea in the evening",
                },
            },
        )
        _invoke_runtime_tool(
            tools[1],
            {"record_id": "rec-2", "reason": "Merged into the canonical retained thought."},
        )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_run_agent)

    system = build_tim_memory_system(bucket_count=4, bucket_candidate_k=4, recall_top_k=2)
    store = system["store"]

    post_think_and_update_memory(
        system,
        question="What does Alice like to drink?",
        response="Alice likes jasmine tea.",
    )
    post_think_and_update_memory(
        system,
        question="When does Alice like jasmine tea?",
        response="Alice likes jasmine tea in the evening.",
    )

    records = store.iter_records("thought_memory")
    assert len(records) == 1
    assert records[0].text == "Alice likes jasmine tea in the evening."
    assert records[0].metadata["hash_bucket"].startswith("bucket-")

    recalled = recall_thoughts(
        system,
        user_query="What should the assistant remember about Alice's tea habit?",
    )
    assert recalled == [
        {
            "record_id": records[0].record_id,
            "text": "Alice likes jasmine tea in the evening.",
            "hash_bucket": records[0].metadata["hash_bucket"],
            "head": "Alice",
            "relation": "Likes",
            "tail": "jasmine tea in the evening",
        }
    ]
