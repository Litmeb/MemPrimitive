"""Repo-consistent reconstruction of the Memory Sharing / INMS memory module.

This file intentionally follows the released
`GHupppp/InteractiveMemorySharingLLM` repository more closely than the paper's
cleaner prose. In particular, the implementation target here is the executable
memory behavior, not a paper-first normalization:

1. a shared memory pool stores accepted prompt-answer examples,
2. an LLM judge filters newly generated examples before storage,
3. accepted examples are appended into a vector-backed shared pool,
4. later queries retrieve the most similar prior examples, and
5. prompt construction re-injects those retrieved examples as in-context memory.

Two important scope notes:

- This file only covers the memory module. It intentionally ignores the
  surrounding multi-agent loop/orchestration.
- The upstream repo also performs online retriever training after accepted
  writes. We intentionally do not implement that training path here. Instead,
  this example leaves a small, explicit orchestration hook where such update
  logic can be inserted later without changing the primitive layout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    EmbeddingSimilarityRetrieval,
    LLMJudgeTrigger,
    TemplateReadout,
)
from memprimitive.utils._runtime import get_runtime
from memprimitive.utils._template import text_prompt


DEFAULT_MEMORY_LAYER = "shared_memory_pool"
DEFAULT_SCORE_THRESHOLD = 50.0

RUBRIC_BY_CATEGORY: dict[str, str] = {
    "Literature": (
        "General Evaluation Criteria (Total: 100 Points)\n\n"
        "Criteria: Literary Quality\n"
        "Description: Assess creativity, language quality, and emotional impact.\n"
        "Criteria: Authenticity\n"
        "Description: Reward adherence to literary form and genre conventions.\n"
        "Criteria: Clarity and Cohesion\n"
        "Description: Reward examples that are understandable and structurally coherent.\n"
        "Criteria: Innovativeness\n"
        "Description: Reward originality in theme, structure, or language use.\n"
        "Criteria: Educational Value\n"
        "Description: Prefer examples that are rich and reusable as few-shot literary demonstrations."
    ),
    "Logic": (
        "1. Clarity and Understandability\n"
        "Question and answer should be clear and understandable.\n"
        "2. Creativity and Originality\n"
        "Reward non-trivial, original logic examples.\n"
        "3. Logical Consistency and Correctness\n"
        "The answer must correctly follow from the question.\n"
        "4. Relevance and Engagement\n"
        "The example should fit logic problems, puzzles, riddles, or puns.\n"
        "5. Difficulty Level\n"
        "The difficulty should be stimulating but still usable."
    ),
    "Plan": (
        "1. Specificity and Detail\n"
        "The question should be specific and the plan should contain actionable steps.\n"
        "2. Feasibility and Practicality\n"
        "The plan should be realistic given ordinary constraints.\n"
        "3. Comprehensiveness and Scope\n"
        "The plan should cover the main components of the goal.\n"
        "4. Personalization and Relevance\n"
        "The plan should feel tailored to the user need.\n"
        "5. Clarity and Understandability\n"
        "The plan and rationale should be easy to follow."
    ),
    "Total": (
        "1. Accuracy\n"
        "2. Relevance\n"
        "3. Completeness\n"
        "4. Clarity and Coherence\n"
        "5. Creativity and Insight\n"
        "Use this general rubric for mixed-domain single-pool memories."
    ),
}

DOMAIN_TO_CATEGORY: dict[str, str] = {
    "literal_creation": "Literature",
    "literature": "Literature",
    "logic_problem_solving": "Logic",
    "logic": "Logic",
    "plan_generation": "Plan",
    "plan": "Plan",
    "one_pool": "Total",
    "single_pool": "Total",
    "total": "Total",
}


def _normalize_grading_category(value: str | None) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    if not normalized:
        return ""
    for category in RUBRIC_BY_CATEGORY:
        if normalized.casefold() == category.casefold():
            return category
    return normalized


def resolve_grading_category(*, grading_category: str | None = None, domain: str | None = None) -> str:
    explicit = _normalize_grading_category(grading_category)
    if explicit:
        return explicit
    normalized_domain = str(domain or "").strip().casefold()
    return DOMAIN_TO_CATEGORY.get(normalized_domain, "Total")


def resolve_rubric_text(*, grading_category: str | None = None, domain: str | None = None) -> str:
    category = resolve_grading_category(grading_category=grading_category, domain=domain)
    return RUBRIC_BY_CATEGORY.get(category, RUBRIC_BY_CATEGORY["Total"])


def _judge_prompt_context(packet: Packet, store: MemoryStore) -> dict[str, Any]:
    _ = store
    observation = packet.observation
    metadata = observation.metadata if observation is not None and isinstance(observation.metadata, dict) else {}
    category = resolve_grading_category(
        grading_category=metadata.get("grading_category"),
        domain=metadata.get("domain"),
    )
    return {
        "grading_category": category,
        "rubric": resolve_rubric_text(
            grading_category=metadata.get("grading_category"),
            domain=metadata.get("domain"),
        ),
    }

DEFAULT_JUDGE_PROMPT = text_prompt(
    "You are judging whether a prompt-answer example should be stored in a shared in-context memory pool.\n"
    "Score the candidate from 0 to 100.\n"
    "High scores should mean the example is clear, useful for future prompts, and helpful beyond the single current turn.\n"
    "Prefer examples that are broadly reusable by agents working on similar tasks in the same domain.\n"
    "Penalize examples that are unclear, low-quality, too trivial, too narrow, or unlikely to help later retrieval.\n\n"
    "Selected grading category:\n"
    "{{ grading_category }}\n\n"
    "Rubric:\n"
    "{{ rubric }}\n\n"
    "Candidate metadata:\n"
    "domain={{ observation.metadata.domain | default('') }}\n"
    "agent_type={{ observation.metadata.agent_type | default('') }}\n"
    "original_query={{ observation.metadata.original_query | default('') }}\n\n"
    "Candidate prompt-answer example:\n"
    "{{ observation.text }}",
    context_builder=_judge_prompt_context,
)

DEFAULT_PROMPT_TEMPLATE = text_prompt(
    "Retrieved shared memories:\n"
    "{{ retrieved.items | join_text }}\n\n"
    "Now, based on these question and answer examples, what is the answer of question:\n"
    "{{ query.text }}"
)


def format_prompt_answer_memory(*, prompt_text: str, answer_text: str) -> str:
    normalized_prompt = str(prompt_text).strip()
    normalized_answer = str(answer_text).strip()
    if not normalized_prompt:
        raise ValueError("prompt_text must be non-empty.")
    if not normalized_answer:
        raise ValueError("answer_text must be non-empty.")
    return f"Question: {normalized_prompt}\nAnswer: {normalized_answer}"


def build_memory_sharing_memory_system(
    *,
    memory_layer: str = DEFAULT_MEMORY_LAYER,
    retrieval_top_k: int = 2,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    judge_prompt=text_prompt(
        "You are judging whether a prompt-answer example should be stored in a shared in-context memory pool.\n"
        "Score the candidate from 0 to 100.\n"
        "High scores should mean the example is clear, useful for future prompts, and helpful beyond the single current turn.\n"
        "Prefer examples that are broadly reusable by agents working on similar tasks in the same domain.\n"
        "Penalize examples that are unclear, low-quality, too trivial, too narrow, or unlikely to help later retrieval.\n\n"
        "Selected grading category:\n"
        "{{ grading_category }}\n\n"
        "Rubric:\n"
        "{{ rubric }}\n\n"
        "Candidate metadata:\n"
        "domain={{ observation.metadata.domain | default('') }}\n"
        "agent_type={{ observation.metadata.agent_type | default('') }}\n"
        "original_query={{ observation.metadata.original_query | default('') }}\n\n"
        "Candidate prompt-answer example:\n"
        "{{ observation.text }}",
        context_builder=_judge_prompt_context,
    ),
    prompt_template=text_prompt(
        "Retrieved shared memories:\n"
        "{{ retrieved.items | join_text }}\n\n"
        "Now, based on these question and answer examples, what is the answer of question:\n"
        "{{ query.text }}"
    ),
) -> dict[str, object]:
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive.")

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name=memory_layer,
                theme="semantic",
                indices=("temporal", "vector"),
                settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
            )
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=LLMJudgeTrigger(
            prompt=judge_prompt,
            decision_mode="score",
            threshold=float(score_threshold),
        ),
        organization=AppendOrganization(target_layer=memory_layer),
        store=store,
    )
    recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=retrieval_top_k, layer=memory_layer),
        readout=TemplateReadout(prompt=prompt_template),
        store=store,
    )

    return {
        "store": store,
        "write_pipeline": write_pipeline,
        "recall_pipeline": recall_pipeline,
        "memory_layer": memory_layer,
        "retrieval_top_k": int(retrieval_top_k),
        "score_threshold": float(score_threshold),
    }


def build_memory_sharing_query(query_text: str) -> Query:
    normalized_query = str(query_text).strip()
    if not normalized_query:
        raise ValueError("query_text must be non-empty.")
    return Query(
        text=normalized_query,
        embedding=list(get_runtime().embed(normalized_query)),
    )


def _schedule_retriever_update_placeholder(
    system: dict[str, object],
    *,
    accepted_record_ids: list[str],
    observation: Observation,
) -> None:
    if not accepted_record_ids:
        return
    store = system["store"]
    assert isinstance(store, MemoryStore)
    placeholder_log = store.metadata.setdefault("pending_retriever_updates", [])
    if not isinstance(placeholder_log, list):
        placeholder_log = []
        store.metadata["pending_retriever_updates"] = placeholder_log
    placeholder_log.append(
        {
            "record_ids": list(accepted_record_ids),
            "observation_id": observation.observation_id,
            "memory_layer": str(system["memory_layer"]),
        }
    )
    # Intentionally left as a stub:
    # retriever online-training / encoder-refresh logic can be inserted here in
    # future example-level orchestration without changing the primitive layout.


def store_prompt_answer_memory(
    system: dict[str, object],
    *,
    prompt_text: str,
    answer_text: str,
    original_query: str | None = None,
    domain: str | None = None,
    agent_type: str | None = None,
    grading_category: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Packet:
    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline, MemoryPipeline)

    observation = Observation(
        text=format_prompt_answer_memory(prompt_text=prompt_text, answer_text=answer_text),
        source="memory_sharing_prompt_answer",
        metadata={
            "original_query": "" if original_query is None else str(original_query),
            "domain": "" if domain is None else str(domain),
            "agent_type": "" if agent_type is None else str(agent_type),
            "grading_category": resolve_grading_category(grading_category=grading_category, domain=domain),
            "prompt_text": str(prompt_text),
            "answer_text": str(answer_text),
            **({} if extra_metadata is None else dict(extra_metadata)),
        },
    )
    packet = write_pipeline.ingest(observation)
    written_record_ids = list(packet.trace.get("organization", {}).get("written_record_ids", []))
    _schedule_retriever_update_placeholder(
        system,
        accepted_record_ids=written_record_ids,
        observation=observation,
    )
    return packet


def build_memory_sharing_prompt(
    system: dict[str, object],
    *,
    query_text: str,
) -> Any:
    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)
    return recall_pipeline.recall(build_memory_sharing_query(query_text))


def recall_memory_examples(
    system: dict[str, object],
    *,
    query_text: str,
) -> Any:
    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)
    packet = recall_pipeline.retrieval.run(
        Packet(query=build_memory_sharing_query(query_text)),
        recall_pipeline.store,
    )[0]
    return packet.retrieved


def main() -> None:
    system = build_memory_sharing_memory_system(retrieval_top_k=2)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    store_prompt_answer_memory(
        system,
        prompt_text="How can I plan a realistic weekly fitness routine for a beginner?",
        answer_text="Alternate light cardio, basic strength sessions, rest days, and one mobility session.",
        original_query="How can I plan a realistic weekly fitness routine for a beginner?",
        domain="plan_generation",
        agent_type="fitness",
    )
    store_prompt_answer_memory(
        system,
        prompt_text="What is a balanced travel plan for a three-day museum trip?",
        answer_text="Reserve one neighborhood per day, pre-book major museums, and keep evenings flexible.",
        original_query="What is a balanced travel plan for a three-day museum trip?",
        domain="plan_generation",
        agent_type="travel",
    )

    readout = build_memory_sharing_prompt(
        system,
        query_text="How should I organize a beginner-friendly weekly exercise plan?",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("stored examples:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
            }
            for record in store.iter_records(str(system["memory_layer"]))
        ]
    )
    print()

    print("retrieval-conditioned prompt:")
    print(readout.text)


if __name__ == "__main__":
    main()
