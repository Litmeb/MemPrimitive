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
    assert representation[0].spec.name == "semantic_field_enrichment_representation"
    assert representation[1].spec.name == "configurable_embedding_representation"
    assert "{{ unit.metadata.amem.content }}" in str(ensure_prompt_plan(representation[1].embedding_text).template)

    assert write_pipeline.organization.spec.name == "graph_append_organization"
    assert isinstance(memory_evolution, tuple)
    assert memory_evolution[0].spec.name == "link_strengthening_evolution"
    assert memory_evolution[1].spec.name == "neighbor_context_update_evolution"

    assert recall_pipeline.retrieval.spec.name == "embedding_similarity_retrieval"
    assert recall_pipeline.readout.spec.name == "note_render_readout"


def test_amem_classics_end_to_end_ingest_and_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    system = build_amem_memory_system(candidate_k=2, neighbor_expansion_k=1, recall_top_k=2)
    store = system["store"]

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
