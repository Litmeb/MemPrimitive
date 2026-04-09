from __future__ import annotations

import pytest

from memprimitive.example.classics.amem_memory import (
    build_amem_memory_system,
    ingest_note,
    recall_notes,
)
from memprimitive.utils._template import ensure_prompt_plan
from memprimitive.utils import _runtime

from baselines_test_helpers import _FakeAMEMRuntime


def test_amem_classics_builder_uses_existing_a_mem_baselines() -> None:
    system = build_amem_memory_system()

    write_pipeline = system["write_pipeline"]
    recall_pipeline = system["recall_pipeline"]

    representation = write_pipeline.representation
    memory_evolution = write_pipeline.memory_evolution

    assert isinstance(representation, tuple)
    assert [module.spec.name for module in representation[:-1]] == [
        "llm_representation",
        "llm_representation",
        "llm_representation",
        "llm_representation",
        "llm_representation",
    ]
    assert representation[-1].spec.name == "configurable_embedding_representation"
    assert "{{ unit.metadata.representation.context }}" in str(ensure_prompt_plan(representation[-1].embedding_text).template)

    assert write_pipeline.organization.spec.name == "graph_append_organization"
    assert memory_evolution.spec.name == "llm_function_call_evolution"
    assert [spec.name for spec in memory_evolution.tool_specs] == [
        "AMEM_STRENGTHEN_LINKS",
        "AMEM_UPDATE_NEIGHBOR",
    ]

    assert recall_pipeline.retrieval.spec.name == "embedding_similarity_retrieval"
    assert recall_pipeline.readout.spec.name == "note_render_readout"


def test_amem_classics_end_to_end_ingest_and_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    from memprimitive.baselines import LLMFunctionCallEvolution, LLMRepresentation

    fake_runtime = _FakeAMEMRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(LLMRepresentation, "_runtime", lambda self: fake_runtime)
    system = build_amem_memory_system(candidate_k=2, recall_top_k=2)
    store = system["store"]
    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline.memory_evolution, LLMFunctionCallEvolution)

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        selected_id = context["selected_record_ids"][0]
        if selected_id == "rec-1":
            return "NO_ACTION"
        assert selected_id == "rec-2"
        _ = rendered_prompt
        from baselines_test_helpers import _invoke_runtime_tool

        _invoke_runtime_tool(
            tools[0],
            {"record_id": "rec-2", "neighbor_record_ids": ["rec-1"]},
        )
        _invoke_runtime_tool(
            tools[1],
            {
                "record_id": "rec-1",
                "context": "Alice's tea habit is now understood as a focus-supporting routine.",
                "tags": ["preference", "habit", "focus"],
            },
        )
        return "DONE"

    monkeypatch.setattr(
        write_pipeline.memory_evolution,
        "_run_agent",
        _fake_run_agent.__get__(write_pipeline.memory_evolution, type(write_pipeline.memory_evolution)),
    )

    ingest_note(system, text="Alice likes tea.")
    ingest_note(system, text="Tea routines improve focus.")

    records = store.iter_records("knowledge_graph")
    assert len(records) == 2

    by_text = {record.text: record for record in records}
    first = by_text["Alice likes tea."]
    second = by_text["Tea routines improve focus."]

    assert first.metadata["amem"]["context"] == "Alice's tea habit is now understood as a focus-supporting routine."
    assert first.metadata["amem"]["tags"] == ["preference", "habit", "focus"]
    assert second.metadata["graph"]["links"] == ["rec-1"]

    rendered = recall_notes(system, user_query="Alice")

    assert "Query: Alice" in rendered
    assert "- Alice likes tea." in rendered
    assert "context: Alice's tea habit is now understood as a focus-supporting routine." in rendered
    assert "tags: preference, habit, focus" in rendered
