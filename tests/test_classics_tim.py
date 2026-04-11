from __future__ import annotations

import json
from typing import Any

import pytest

from baselines_test_helpers import _invoke_runtime_tool
from memprimitive import MemoryRecord
from memprimitive.example.classics.tim_memory import (
    _build_bucket_recall_pipeline,
    _build_tim_bucket_evolution_module,
    _build_tim_bucket_insert_organization,
    build_tim_memory_system,
    compute_hash_bucket,
    post_think_and_update_memory,
    recall_thoughts,
)
from memprimitive.utils import _runtime


class _FakeTiMRuntime:
    def __init__(
        self,
        *,
        extractions: list[list[dict[str, str]]],
        embedding_map: dict[str, list[float]],
    ) -> None:
        self.extractions = [list(batch) for batch in extractions]
        self.embedding_map = {key: list(value) for key, value in embedding_map.items()}

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


def _append_seed_record(
    system: dict[str, object],
    *,
    text: str,
    hash_bucket: str,
    head: str,
    relation: str,
    tail: str,
) -> str:
    store = system["store"]
    assert hasattr(store, "next_sequence_id")
    record_id = f"rec-{store.next_sequence_id()}"
    store.append(
        MemoryRecord(
            record_id=record_id,
            unit_id=f"seed-{record_id}",
            layer=str(system["memory_layer"]),
            text=text,
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={
                "head": head,
                "relation": relation,
                "tail": tail,
                "hash_bucket": hash_bucket,
                "source_question": "seed",
                "source_response": "seed",
                "thought_kind": "inductive_relation_thought",
            },
        )
    )
    return record_id


def test_tim_classics_builder_uses_existing_baselines() -> None:
    system = build_tim_memory_system(bucket_count=4)

    insert_module = _build_tim_bucket_insert_organization(system, hash_bucket="bucket-0")
    evolution_module = _build_tim_bucket_evolution_module(system, hash_bucket="bucket-0")
    recall_pipeline = _build_bucket_recall_pipeline(
        system,
        hash_bucket="bucket-0",
        final_top_k=2,
        prompt="Thoughts:\n{{ retrieved.items | join_text }}",
    )

    assert insert_module.spec.name == "llm_function_call_organization"
    assert [spec.name for spec in insert_module.tool_specs] == ["ADD"]

    assert evolution_module.spec.name == "llm_function_call_evolution"
    assert [spec.name for spec in evolution_module.tool_specs] == ["UPDATE", "DELETE"]

    retrieval = recall_pipeline.retrieval
    assert isinstance(retrieval, tuple)
    assert [module.spec.name for module in retrieval] == [
        "metadata_retrieval",
        "embedding_similarity_retrieval",
    ]


def test_tim_classics_post_think_merge_uses_bucket_batch_insert_and_full_bucket_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMFunctionCallOrganization

    fake_runtime = _FakeTiMRuntime(
        extractions=[
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
        ],
        embedding_map={
            "Alice likes jasmine tea.": [1.0, 0.0],
            "Alice likes jasmine tea in the evening.": [1.0, 0.0],
            "What should the assistant remember about Alice's tea habit?": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    system = build_tim_memory_system(bucket_count=4, bucket_candidate_k=1, recall_top_k=2)
    jasmine_bucket = compute_hash_bucket(system, "Alice likes jasmine tea.")

    def _fake_org_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "Current Question-Response Pair" in rendered_prompt
        if "Alice likes jasmine tea in the evening." in rendered_prompt:
            _invoke_runtime_tool(
                tools[0],
                {
                    "text": "Alice likes jasmine tea in the evening.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "Likes",
                        "tail": "jasmine tea in the evening",
                        "hash_bucket": jasmine_bucket,
                        "source_question": "When does Alice like jasmine tea?",
                        "source_response": "Alice likes jasmine tea in the evening.",
                        "thought_kind": "inductive_relation_thought",
                    },
                },
            )
            return "DONE"

        _invoke_runtime_tool(
            tools[0],
            {
                "text": "Alice likes jasmine tea.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "Likes",
                        "tail": "jasmine tea",
                        "hash_bucket": jasmine_bucket,
                        "source_question": "What does Alice like to drink?",
                        "source_response": "Alice likes jasmine tea.",
                        "thought_kind": "inductive_relation_thought",
                    },
                },
        )
        return "DONE"

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        selected_ids = context["selected_record_ids"]
        if selected_ids == ["rec-1"]:
            assert "rec-1" in rendered_prompt
            return "NO_ACTION"

        assert selected_ids == ["rec-2"]
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

    monkeypatch.setattr(LLMFunctionCallOrganization, "_run_agent", _fake_org_run_agent)
    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    store = system["store"]

    post_think_and_update_memory(
        system,
        question="What does Alice like to drink?",
        response="Alice likes jasmine tea.",
    )
    packets = post_think_and_update_memory(
        system,
        question="When does Alice like jasmine tea?",
        response="Alice likes jasmine tea in the evening.",
    )

    assert len(packets) == 1
    records = store.iter_records("thought_memory")
    assert len(records) == 1
    assert records[0].text == "Alice likes jasmine tea in the evening."
    assert records[0].metadata["hash_bucket"] == jasmine_bucket

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


def test_tim_classics_joint_round_update_supports_multiple_adds_in_one_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMFunctionCallOrganization

    fake_runtime = _FakeTiMRuntime(
        extractions=[
            [
                {
                    "thought": "Alice likes jasmine tea in the evening.",
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea in the evening",
                },
                {
                    "thought": "Alice drinks jasmine tea after dinner.",
                    "head": "Alice",
                    "relation": "DrinksAfterDinner",
                    "tail": "jasmine tea",
                },
            ]
        ],
        embedding_map={
            "Alice likes jasmine tea in the evening.": [1.0, 0.0],
            "Alice drinks jasmine tea after dinner.": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    system = build_tim_memory_system(bucket_count=4, bucket_candidate_k=1, recall_top_k=2)
    round_bucket = compute_hash_bucket(system, "Alice likes jasmine tea in the evening.")

    def _fake_org_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "Newly Extracted Thoughts" in rendered_prompt
        _invoke_runtime_tool(
            tools[0],
            {
                "text": "Alice likes jasmine tea in the evening.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "Likes",
                        "tail": "jasmine tea in the evening",
                        "hash_bucket": round_bucket,
                        "source_question": "What should we remember about Alice's tea habits?",
                        "source_response": "Alice likes jasmine tea in the evening and drinks it after dinner.",
                        "thought_kind": "inductive_relation_thought",
                    },
            },
        )
        _invoke_runtime_tool(
            tools[0],
            {
                "text": "Alice drinks jasmine tea after dinner.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "DrinksAfterDinner",
                        "tail": "jasmine tea",
                        "hash_bucket": round_bucket,
                        "source_question": "What should we remember about Alice's tea habits?",
                        "source_response": "Alice likes jasmine tea in the evening and drinks it after dinner.",
                        "thought_kind": "inductive_relation_thought",
                    },
            },
        )
        return "DONE"

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _ = tools
        assert context["selected_record_ids"] == ["rec-1", "rec-2"]
        assert "rec-1" in rendered_prompt
        assert "rec-2" in rendered_prompt
        return "NO_ACTION"

    monkeypatch.setattr(LLMFunctionCallOrganization, "_run_agent", _fake_org_run_agent)
    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    store = system["store"]

    packets = post_think_and_update_memory(
        system,
        question="What should we remember about Alice's tea habits?",
        response="Alice likes jasmine tea in the evening and drinks it after dinner.",
    )

    assert len(packets) == 1
    assert packets[0].trace["organization"]["written_record_ids"] == ["rec-1", "rec-2"]
    assert packets[0].trace["memory_evolution"]["selected_record_ids"] == ["rec-1", "rec-2"]

    records = store.iter_records("thought_memory")
    assert [record.text for record in records] == [
        "Alice likes jasmine tea in the evening.",
        "Alice drinks jasmine tea after dinner.",
    ]


def test_tim_classics_full_bucket_evolution_visibility_ignores_old_candidate_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMFunctionCallOrganization

    fake_runtime = _FakeTiMRuntime(
        extractions=[
            [
                {
                    "thought": "Alice prefers jasmine tea.",
                    "head": "Alice",
                    "relation": "Prefers",
                    "tail": "jasmine tea",
                }
            ]
        ],
        embedding_map={
            "Historical thought 1": [1.0, 0.0],
            "Historical thought 2": [1.0, 0.0],
            "Historical thought 3": [1.0, 0.0],
            "Historical thought 4": [1.0, 0.0],
            "Alice prefers jasmine tea.": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    system = build_tim_memory_system(bucket_count=4, bucket_candidate_k=1, recall_top_k=2)
    preference_bucket = compute_hash_bucket(system, "Alice prefers jasmine tea.")

    def _fake_org_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _ = rendered_prompt
        _ = context
        _invoke_runtime_tool(
            tools[0],
            {
                "text": "Alice prefers jasmine tea.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "Prefers",
                        "tail": "jasmine tea",
                        "hash_bucket": preference_bucket,
                        "source_question": "What does Alice prefer?",
                        "source_response": "Alice prefers jasmine tea.",
                        "thought_kind": "inductive_relation_thought",
                    },
            },
        )
        return "DONE"

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert context["selected_record_ids"] == ["rec-5"]
        for record_id in ("rec-1", "rec-2", "rec-3", "rec-4", "rec-5"):
            assert record_id in rendered_prompt
        _invoke_runtime_tool(
            tools[1],
            {"record_id": "rec-4", "reason": "Full-bucket maintenance can see this older same-bucket record."},
        )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallOrganization, "_run_agent", _fake_org_run_agent)
    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    _append_seed_record(
        system,
        text="Historical thought 1",
        hash_bucket=preference_bucket,
        head="Alice",
        relation="History",
        tail="1",
    )
    _append_seed_record(
        system,
        text="Historical thought 2",
        hash_bucket=preference_bucket,
        head="Alice",
        relation="History",
        tail="2",
    )
    _append_seed_record(
        system,
        text="Historical thought 3",
        hash_bucket=preference_bucket,
        head="Alice",
        relation="History",
        tail="3",
    )
    _append_seed_record(
        system,
        text="Historical thought 4",
        hash_bucket=preference_bucket,
        head="Alice",
        relation="History",
        tail="4",
    )

    post_think_and_update_memory(
        system,
        question="What does Alice prefer?",
        response="Alice prefers jasmine tea.",
    )

    records = system["store"].iter_records("thought_memory")
    assert [record.record_id for record in records] == ["rec-1", "rec-2", "rec-3", "rec-5"]


def test_tim_classics_post_think_includes_historical_thoughts_in_extraction_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMFunctionCallOrganization

    captured_users: list[str] = []

    class _InspectingRuntime(_FakeTiMRuntime):
        def json(self, *, system: str, user: str) -> Any:
            _ = system
            captured_users.append(user)
            return super().json(system=system, user=user)

    fake_runtime = _InspectingRuntime(
        extractions=[
            [
                {
                    "thought": "Alice likes jasmine tea.",
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea",
                }
            ]
        ],
        embedding_map={
            "Alice likes green tea.": [1.0, 0.0],
            "Alice likes jasmine tea.": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    system = build_tim_memory_system(bucket_count=4, bucket_candidate_k=1, recall_top_k=2)
    history_bucket = compute_hash_bucket(system, "Alice likes jasmine tea.")

    _append_seed_record(
        system,
        text="Alice likes green tea.",
        hash_bucket=history_bucket,
        head="Alice",
        relation="Likes",
        tail="green tea",
    )

    def _fake_org_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _ = rendered_prompt
        _ = context
        _invoke_runtime_tool(
            tools[0],
            {
                "text": "Alice likes jasmine tea.",
                "metadata": {
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea",
                    "hash_bucket": history_bucket,
                    "source_question": "What tea does Alice like?",
                    "source_response": "Alice likes jasmine tea.",
                    "thought_kind": "inductive_relation_thought",
                },
            },
        )
        return "DONE"

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _ = rendered_prompt
        _ = tools
        _ = context
        return "NO_ACTION"

    monkeypatch.setattr(LLMFunctionCallOrganization, "_run_agent", _fake_org_run_agent)
    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    post_think_and_update_memory(
        system,
        question="What tea does Alice like?",
        response="Alice likes jasmine tea.",
    )

    assert len(captured_users) == 1
    extraction_payload = json.loads(captured_users[0])
    assert extraction_payload["historical_thoughts"] == [
        {
            "thought": "Alice likes green tea.",
            "head": "Alice",
            "relation": "Likes",
            "tail": "green tea",
        }
    ]
