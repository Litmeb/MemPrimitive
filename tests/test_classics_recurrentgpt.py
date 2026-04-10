from __future__ import annotations

import pytest

from memprimitive import MemoryStore
from memprimitive.baselines import EmbeddingSimilarityRetrieval
from memprimitive.example.classics.recurrentgpt_memory import (
    bootstrap_recurrentgpt_story,
    build_recurrentgpt_memory_system,
    current_short_memory,
    recall_related_paragraphs,
    run_recurrentgpt_iteration,
)


class _FakeRecurrentGPTRuntime:
    def require_llm(self, *, capability: str) -> None:
        return None

    def text(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        _ = (system, temperature)
        if "Name:" in user and "Paragraph 1:" in user and "Instruction 3:" in user:
            return (
                "Name: Beacon City\n"
                "Outline: An abandoned city begins to stir when a signal returns.\n"
                "Paragraph 1: Mara receives a weak beacon from the sealed city.\n"
                "Paragraph 2: She travels to the ridge and sees the silent towers flicker.\n"
                "Paragraph 3: She enters the outskirts and hears machinery waking below the streets.\n"
                "Summary: Mara follows a reawakened beacon toward an abandoned city whose hidden systems are starting to move again.\n"
                "Instruction 1: Have Mara trace the signal to the central station.\n"
                "Instruction 2: Let Mara descend into the maintenance tunnels and discover who restarted the grid.\n"
                "Instruction 3: Shift to a distant observer tracking Mara from the city wall.\n"
            )
        if "Three plans of what to write next proposed by the assistant:" in user:
            if "Continue through the powered tunnel and confront the hidden operator." in user:
                return (
                    "Selected Plan: Continue through the powered tunnel and confront the hidden operator.\n"
                    "Reason: It keeps the strongest immediate tension.\n"
                )
            return (
                "Selected Plan: Let Mara descend into the maintenance tunnels and discover who restarted the grid.\n"
                "Reason: It best develops the mystery while keeping the pacing controlled.\n"
            )
        if "Extended Paragraph:" in user and "The selected plan of what to write next:" in user:
            if "The selected plan of what to write next:\nLet Mara descend into the maintenance tunnels and discover who restarted the grid." in user:
                return (
                    "Extended Paragraph: Mara descends into the maintenance tunnels, follows the heat of the restarted grid, and finds fresh tool marks beside old control panels.\n"
                    "Selected Plan: Let Mara descend into the maintenance tunnels and discover who restarted the grid.\n"
                    "Revised Plan: Mara should follow the powered tunnel deeper, identify signs of a recent visitor, and end by opening a sealed service door.\n"
                )
            return (
                "Extended Paragraph: Mara pushes through the service door, reaches the hidden control room, and realizes another survivor has been guiding the beacon.\n"
                "Selected Plan: Continue through the powered tunnel and confront the hidden operator.\n"
                "Revised Plan: Mara should question the hidden operator, learn why the city was reactivated, and leave one major motive unresolved.\n"
            )
        if "Output Paragraph:" in user and "Input Related Paragraphs:" in user:
            return (
                "Output Paragraph: Mara follows the powered tunnel until she reaches a service vault where fresh fingerprints cover the dust.\n"
                "Output Memory: Rational: Remove small scenic details that no longer matter and keep the active mystery.\n"
                "Updated Memory: Mara is following a reawakened beacon through an abandoned city. The tunnels now show signs of a recent survivor who has restarted part of the grid.\n"
                "Output Instruction:\n"
                "Instruction 1: Have Mara inspect the service vault and decode the new logs.\n"
                "Instruction 2: Continue through the powered tunnel and confront the hidden operator.\n"
                "Instruction 3: Cut to the city surface and reveal the beacon's wider effect.\n"
            )
        raise AssertionError(f"Unexpected fake-runtime prompt:\n{user}")

    def embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            1.0 if "tunnel" in lowered else 0.0,
            1.0 if "beacon" in lowered else 0.0,
            1.0 if "operator" in lowered else 0.0,
        ]


def test_recurrentgpt_builder_sets_up_repo_style_layers() -> None:
    system = build_recurrentgpt_memory_system(related_top_k=2, new_character_prob=0.0)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    assert store.topology.layer_names == ("long_memory", "short_memory")
    assert store.topology.get_layer("long_memory").indices == ("temporal", "vector")
    assert store.topology.get_layer("short_memory").capacity == "sliding_window"
    assert store.topology.get_layer("short_memory").get_setting("record_budget") == 1

    assert system["long_memory_write_pipeline"].organization.spec.name == "append_organization"
    assert system["long_memory_recall_pipeline"].retrieval.spec.name == "embedding_similarity_retrieval"


def test_recurrentgpt_bootstrap_and_iteration_update_memory_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    fake_runtime = _FakeRecurrentGPTRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", lambda self, text: fake_runtime.embed(text))

    system = build_recurrentgpt_memory_system(related_top_k=1, new_character_prob=0.0)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    state = bootstrap_recurrentgpt_story(
        system,
        topic="an abandoned city waking to a beacon",
        story_type="science fiction",
    )

    assert state.title == "Beacon City"
    assert state.current_paragraph.startswith("Mara descends into the maintenance tunnels")
    assert state.current_instruction.startswith("Mara should follow the powered tunnel deeper")
    assert store.count("long_memory") == 2
    assert store.count("short_memory") == 1
    assert current_short_memory(system).startswith("Mara follows a reawakened beacon")

    related = recall_related_paragraphs(system, instruction=state.current_instruction)
    assert related.strip()
    assert "Mara" in related

    updated_state = run_recurrentgpt_iteration(system, state)

    assert updated_state.step_index == 1
    assert store.count("long_memory") == 3
    assert store.count("short_memory") == 1
    assert current_short_memory(system) == (
        "Mara is following a reawakened beacon through an abandoned city. "
        "The tunnels now show signs of a recent survivor who has restarted part of the grid."
    )
    assert updated_state.current_paragraph.startswith("Mara pushes through the service door")
    assert updated_state.current_instruction.startswith("Mara should question the hidden operator")
    assert updated_state.candidate_instructions[1] == "Continue through the powered tunnel and confront the hidden operator."
    assert updated_state.history[-1]["selected_plan"] == "Continue through the powered tunnel and confront the hidden operator."
