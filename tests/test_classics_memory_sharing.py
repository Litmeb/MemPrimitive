from __future__ import annotations

import json

import pytest

from memprimitive import MemoryStore
from memprimitive.example.classics.memory_sharing_memory import (
    build_memory_sharing_memory_system,
    build_memory_sharing_prompt,
    resolve_grading_category,
    store_prompt_answer_memory,
)


class _FakeMemorySharingRuntime:
    def __init__(self) -> None:
        self.judge_prompts: list[str] = []

    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str):
        _ = system
        payload = json.loads(user)
        self.judge_prompts.append(payload["prompt"])
        candidate = payload["judge_context"]["observation"]["text"].casefold()
        if "exercise" in candidate or "fitness" in candidate or "travel" in candidate:
            return {"score": 82}
        return {"score": 21}

    def embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            1.0 if ("exercise" in lowered or "fitness" in lowered or "workout" in lowered) else 0.0,
            1.0 if ("travel" in lowered or "museum" in lowered or "trip" in lowered) else 0.0,
            1.0 if ("plan" in lowered or "routine" in lowered or "schedule" in lowered) else 0.0,
        ]


def test_memory_sharing_builder_sets_up_shared_vector_pool() -> None:
    system = build_memory_sharing_memory_system(retrieval_top_k=2, score_threshold=50.0)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    assert store.topology.layer_names == ("shared_memory_pool",)
    layer = store.topology.get_layer("shared_memory_pool")
    assert layer.indices == ("temporal", "vector")
    assert layer.get_setting("embedding") == {
        "enabled": True,
        "mode": "text",
        "refresh_on_update": "semantic_text_change",
    }

    assert system["write_pipeline"].write_trigger.spec.name == "llm_judge_write_trigger"
    assert system["write_pipeline"].organization.spec.name == "append_organization"
    assert system["recall_pipeline"].retrieval.spec.name == "embedding_similarity_retrieval"
    assert system["recall_pipeline"].readout.spec.name == "template_readout"


def test_memory_sharing_accepts_high_score_examples_and_builds_augmented_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    runtime = _FakeMemorySharingRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", runtime)
    system = build_memory_sharing_memory_system(retrieval_top_k=1, score_threshold=50.0)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    fitness_packet = store_prompt_answer_memory(
        system,
        prompt_text="How can I design a weekly fitness plan for a beginner?",
        answer_text="Use three light exercise days, two recovery days, and one mobility session.",
        original_query="How can I design a weekly fitness plan for a beginner?",
        domain="plan_generation",
        agent_type="fitness",
    )
    travel_packet = store_prompt_answer_memory(
        system,
        prompt_text="What is a balanced travel plan for a three-day museum trip?",
        answer_text="Group museums by neighborhood, pre-book major tickets, and leave evenings flexible.",
        original_query="What is a balanced travel plan for a three-day museum trip?",
        domain="plan_generation",
        agent_type="travel",
    )

    assert fitness_packet.trace["write_trigger"]["decisions"] == [True]
    assert travel_packet.trace["write_trigger"]["decisions"] == [True]
    assert "Selected grading category:\nPlan" in runtime.judge_prompts[0]
    assert "Feasibility and Practicality" in runtime.judge_prompts[0]
    assert store.count("shared_memory_pool") == 2
    assert [record.record_id for record in store.iter_records("shared_memory_pool")] == ["rec-1", "rec-2"]
    assert store.metadata["pending_retriever_updates"] == [
        {
            "record_ids": ["rec-1"],
            "observation_id": fitness_packet.observation.observation_id,
            "memory_layer": "shared_memory_pool",
        },
        {
            "record_ids": ["rec-2"],
            "observation_id": travel_packet.observation.observation_id,
            "memory_layer": "shared_memory_pool",
        },
    ]

    readout = build_memory_sharing_prompt(
        system,
        query_text="How should I organize a beginner workout routine this week?",
    )

    assert "Retrieved shared memories:" in readout.text
    assert "weekly fitness plan" in readout.text
    assert "museum trip" not in readout.text
    assert "Now, based on these question and answer examples" in readout.text
    assert "How should I organize a beginner workout routine this week?" in readout.text
    assert readout.source_ids == ["rec-1"]


def test_memory_sharing_rejects_low_score_examples_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    runtime = _FakeMemorySharingRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", runtime)
    system = build_memory_sharing_memory_system(retrieval_top_k=1, score_threshold=50.0)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    packet = store_prompt_answer_memory(
        system,
        prompt_text="Hello there.",
        answer_text="Hi.",
        original_query="Hello there.",
        domain="smalltalk",
        agent_type="generic",
    )

    assert packet.trace["write_trigger"]["decisions"] == [False]
    assert "Selected grading category:\nTotal" in runtime.judge_prompts[0]
    assert packet.trace["organization"]["written_record_ids"] == []
    assert store.count("shared_memory_pool") == 0
    assert store.metadata.get("pending_retriever_updates") is None


def test_memory_sharing_resolves_repo_style_category_from_domain_and_explicit_override() -> None:
    assert resolve_grading_category(domain="plan_generation") == "Plan"
    assert resolve_grading_category(domain="logic_problem_solving") == "Logic"
    assert resolve_grading_category(domain="literal_creation") == "Literature"
    assert resolve_grading_category(domain="one_pool") == "Total"
    assert resolve_grading_category(domain="unknown_domain") == "Total"
    assert resolve_grading_category(domain="plan_generation", grading_category="Logic") == "Logic"


def test_memory_sharing_uses_explicit_grading_category_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime

    runtime = _FakeMemorySharingRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", runtime)
    system = build_memory_sharing_memory_system(retrieval_top_k=1, score_threshold=50.0)

    packet = store_prompt_answer_memory(
        system,
        prompt_text="Write a compact sonnet prompt about time and memory.",
        answer_text="Use fourteen lines, keep the volta sharp, and contrast decay with remembrance.",
        original_query="Write a compact sonnet prompt about time and memory.",
        domain="plan_generation",
        grading_category="Literature",
        agent_type="poetry",
    )

    assert packet.observation is not None
    assert packet.observation.metadata["grading_category"] == "Literature"
    assert "Selected grading category:\nLiterature" in runtime.judge_prompts[0]
    assert "Literary Quality" in runtime.judge_prompts[0]
