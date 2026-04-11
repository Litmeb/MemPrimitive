from __future__ import annotations

from typing import Any

import pytest

from baselines_test_helpers import _invoke_runtime_tool
from memprimitive import MemoryRecord
from memprimitive.example.classics.tim_simple_memory import (
    _build_tim_simple_candidate_recall_pipeline,
    _build_tim_simple_evolution_module,
    build_tim_simple_memory_system,
    post_think_and_update_tim_simple_memory,
    recall_tim_simple_thoughts,
)
from memprimitive.utils import _runtime


class _FakeTiMSimpleRuntime:
    def __init__(
        self,
        *,
        extractions: list[list[dict[str, str]]],
        embedding_map: dict[str, list[float]],
    ) -> None:
        self.extractions = [list(batch) for batch in extractions]
        self.embedding_map = {key: list(value) for key, value in embedding_map.items()}

    def require_llm(self, *, capability: str) -> None:
        _ = capability

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
                "source_question": "seed",
                "source_response": "seed",
                "thought_kind": "inductive_relation_thought",
            },
        )
    )
    return record_id


def test_tim_simple_builder_uses_global_embedding_retrieval() -> None:
    system = build_tim_simple_memory_system(candidate_top_k=2, recall_top_k=3)

    evolution_module = _build_tim_simple_evolution_module(system)
    candidate_pipeline = _build_tim_simple_candidate_recall_pipeline(system)

    assert evolution_module.spec.name == "llm_function_call_evolution"
    assert [spec.name for spec in evolution_module.tool_specs] == ["ADD", "UPDATE", "DELETE"]
    assert "target_layer" in evolution_module.tool_specs[0].parameters_json_schema["properties"]
    assert "target_layer" not in evolution_module.tool_specs[1].parameters_json_schema["properties"]
    assert "target_layer" not in evolution_module.tool_specs[2].parameters_json_schema["properties"]

    retrieval = candidate_pipeline.retrieval
    assert isinstance(retrieval, tuple)
    assert [module.spec.name for module in retrieval] == ["embedding_similarity_retrieval"]


def test_tim_simple_single_thought_can_update_recalled_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMRepresentation

    fake_runtime = _FakeTiMSimpleRuntime(
        extractions=[
            [
                {
                    "thought": "Alice likes jasmine tea in the evening.",
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea in the evening",
                }
            ]
        ],
        embedding_map={
            "Alice likes jasmine tea.": [1.0, 0.0],
            "Bob likes coffee.": [0.0, 1.0],
            "Alice likes jasmine tea in the evening.": [1.0, 0.0],
            "What should the assistant remember about Alice's tea habit?": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(LLMRepresentation, "_runtime", lambda self: fake_runtime)

    system = build_tim_simple_memory_system(candidate_top_k=1, recall_top_k=2)
    _append_seed_record(
        system,
        text="Alice likes jasmine tea.",
        head="Alice",
        relation="Likes",
        tail="jasmine tea",
    )
    _append_seed_record(
        system,
        text="Bob likes coffee.",
        head="Bob",
        relation="Likes",
        tail="coffee",
    )

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "Current New Thought" in rendered_prompt
        assert "Alice likes jasmine tea." in rendered_prompt
        assert "Bob likes coffee." not in rendered_prompt
        assert context["selected_record_ids"] == []
        _invoke_runtime_tool(
            tools[1],
            {
                "record_id": "rec-1",
                "text": "Alice likes jasmine tea in the evening.",
                "metadata_patch": {
                    "tail": "jasmine tea in the evening",
                },
            },
        )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    packets = post_think_and_update_tim_simple_memory(
        system,
        question="What should the assistant remember about Alice's tea habit?",
        response="Alice likes jasmine tea in the evening.",
    )

    assert len(packets) == 1
    records = system["store"].iter_records("thought_memory")
    assert [record.record_id for record in records] == ["rec-1", "rec-2"]
    assert records[0].text == "Alice likes jasmine tea in the evening."
    assert packets[0].trace["memory_evolution"]["updated_record_ids"] == ["rec-1"]
    assert packets[0].trace["memory_evolution"]["visible_record_ids"] == ["rec-1"]


def test_tim_simple_single_thought_can_add_without_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMRepresentation

    fake_runtime = _FakeTiMSimpleRuntime(
        extractions=[
            [
                {
                    "thought": "Alice lives in Beijing.",
                    "head": "Alice",
                    "relation": "LivesIn",
                    "tail": "Beijing",
                }
            ]
        ],
        embedding_map={
            "Alice lives in Beijing.": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(LLMRepresentation, "_runtime", lambda self: fake_runtime)

    system = build_tim_simple_memory_system(candidate_top_k=2, recall_top_k=2)

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "Visible Candidate Records" not in rendered_prompt
        assert context["selected_record_ids"] == []
        _invoke_runtime_tool(
            tools[0],
            {
                "text": "Alice lives in Beijing.",
                "metadata": {
                    "head": "Alice",
                    "relation": "LivesIn",
                    "tail": "Beijing",
                    "source_question": "Where does Alice live?",
                    "source_response": "Alice lives in Beijing.",
                    "thought_kind": "inductive_relation_thought",
                },
            },
        )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    packets = post_think_and_update_tim_simple_memory(
        system,
        question="Where does Alice live?",
        response="Alice lives in Beijing.",
    )

    assert len(packets) == 1
    records = system["store"].iter_records("thought_memory")
    assert len(records) == 1
    assert records[0].text == "Alice lives in Beijing."
    assert packets[0].trace["memory_evolution"]["written_record_ids"] == ["rec-1"]


def test_tim_simple_single_thought_can_delete_recalled_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMRepresentation

    fake_runtime = _FakeTiMSimpleRuntime(
        extractions=[
            [
                {
                    "thought": "Alice no longer drinks coffee.",
                    "head": "Alice",
                    "relation": "NoLongerDrinks",
                    "tail": "coffee",
                }
            ]
        ],
        embedding_map={
            "Alice drinks coffee every evening.": [1.0, 0.0],
            "Alice no longer drinks coffee.": [1.0, 0.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(LLMRepresentation, "_runtime", lambda self: fake_runtime)

    system = build_tim_simple_memory_system(candidate_top_k=1, recall_top_k=2)
    _append_seed_record(
        system,
        text="Alice drinks coffee every evening.",
        head="Alice",
        relation="Drinks",
        tail="coffee every evening",
    )

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "Alice drinks coffee every evening." in rendered_prompt
        assert context["selected_record_ids"] == []
        _invoke_runtime_tool(
            tools[2],
            {"record_id": "rec-1", "reason": "Contradicted by the new thought."},
        )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    packets = post_think_and_update_tim_simple_memory(
        system,
        question="What changed about Alice's coffee habit?",
        response="Alice no longer drinks coffee.",
    )

    assert len(packets) == 1
    assert system["store"].iter_records("thought_memory") == []
    assert packets[0].trace["memory_evolution"]["deleted_record_ids"] == ["rec-1"]


def test_tim_simple_multi_thought_round_runs_one_recall_per_thought(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMRepresentation

    fake_runtime = _FakeTiMSimpleRuntime(
        extractions=[
            [
                {
                    "thought": "Alice likes jasmine tea.",
                    "head": "Alice",
                    "relation": "Likes",
                    "tail": "jasmine tea",
                },
                {
                    "thought": "Alice reads history books.",
                    "head": "Alice",
                    "relation": "Reads",
                    "tail": "history books",
                },
            ]
        ],
        embedding_map={
            "Alice likes jasmine tea.": [1.0, 0.0],
            "Alice reads history books.": [0.0, 1.0],
        },
    )
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(LLMRepresentation, "_runtime", lambda self: fake_runtime)

    system = build_tim_simple_memory_system(candidate_top_k=1, recall_top_k=2)
    prompts: list[str] = []

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        prompts.append(rendered_prompt)
        if "triple=(Alice, Reads, history books)" in rendered_prompt:
            _invoke_runtime_tool(
                tools[0],
                {
                    "text": "Alice reads history books.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "Reads",
                        "tail": "history books",
                        "source_question": "What should we remember about Alice?",
                        "source_response": "Alice likes jasmine tea and reads history books.",
                        "thought_kind": "inductive_relation_thought",
                    },
                },
            )
        else:
            _invoke_runtime_tool(
                tools[0],
                {
                    "text": "Alice likes jasmine tea.",
                    "metadata": {
                        "head": "Alice",
                        "relation": "Likes",
                        "tail": "jasmine tea",
                        "source_question": "What should we remember about Alice?",
                        "source_response": "Alice likes jasmine tea and reads history books.",
                        "thought_kind": "inductive_relation_thought",
                    },
                },
            )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    packets = post_think_and_update_tim_simple_memory(
        system,
        question="What should we remember about Alice?",
        response="Alice likes jasmine tea and reads history books.",
    )

    assert len(packets) == 2
    assert len(prompts) == 2
    assert [record.text for record in system["store"].iter_records("thought_memory")] == [
        "Alice likes jasmine tea.",
        "Alice reads history books.",
    ]


def test_tim_simple_recall_uses_global_embedding_similarity() -> None:
    system = build_tim_simple_memory_system(candidate_top_k=2, recall_top_k=1)
    _append_seed_record(
        system,
        text="Alice likes jasmine tea.",
        head="Alice",
        relation="Likes",
        tail="jasmine tea",
    )
    _append_seed_record(
        system,
        text="Bob likes coffee.",
        head="Bob",
        relation="Likes",
        tail="coffee",
    )

    class _RecallRuntime:
        def require_llm(self, *, capability: str) -> None:
            _ = capability

        def embed(self, text: str) -> list[float]:
            if text == "Alice likes jasmine tea.":
                return [1.0, 0.0]
            if text == "Bob likes coffee.":
                return [0.0, 1.0]
            if text == "What tea does Alice like?":
                return [1.0, 0.0]
            raise AssertionError(f"Unexpected embedding request: {text!r}")

    _runtime._DEFAULT_RUNTIME = _RecallRuntime()
    records = system["store"].iter_records("thought_memory")
    records[0].embedding = [1.0, 0.0]
    records[1].embedding = [0.0, 1.0]

    recalled = recall_tim_simple_thoughts(system, user_query="What tea does Alice like?")
    assert recalled == [
        {
            "record_id": "rec-1",
            "text": "Alice likes jasmine tea.",
            "head": "Alice",
            "relation": "Likes",
            "tail": "jasmine tea",
        }
    ]


def test_tim_simple_post_think_still_passes_historical_thoughts_into_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMRepresentation

    captured_users: list[str] = []

    class _InspectingRuntime(_FakeTiMSimpleRuntime):
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
    monkeypatch.setattr(LLMRepresentation, "_runtime", lambda self: fake_runtime)

    system = build_tim_simple_memory_system(candidate_top_k=1, recall_top_k=2)
    _append_seed_record(
        system,
        text="Alice likes green tea.",
        head="Alice",
        relation="Likes",
        tail="green tea",
    )

    def _fake_evo_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
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
                    "source_question": "What tea does Alice like?",
                    "source_response": "Alice likes jasmine tea.",
                    "thought_kind": "inductive_relation_thought",
                },
            },
        )
        return "DONE"

    monkeypatch.setattr(LLMFunctionCallEvolution, "_run_agent", _fake_evo_run_agent)

    post_think_and_update_tim_simple_memory(
        system,
        question="What tea does Alice like?",
        response="Alice likes jasmine tea.",
    )

    assert len(captured_users) == 1
    assert "Historical Thoughts" in captured_users[0]
    assert "Alice likes green tea." in captured_users[0]
    assert "hash_bucket" not in captured_users[0]
