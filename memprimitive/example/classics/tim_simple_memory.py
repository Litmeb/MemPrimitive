"""Simplified TiM-style memory using pure embedding-similarity retrieval.

This file keeps the same TiM-style thought extraction surface as
``tim_memory.py`` but deliberately drops the paper-oriented hash-table / LSH
locality design. It is a lighter comparison variant for the same mechanism
family:

1. extract inductive relation-bearing thoughts from the current Q-R pair,
2. for each new thought, retrieve the top-k most similar historical thoughts
   from the whole memory layer,
3. let the model decide add / update / delete using only those recalled
   candidates, and
4. recall memory later with ordinary embedding similarity over the whole layer.

Scope boundary:

- This file is an embedding-only simplification of TiM memory maintenance.
- It intentionally preserves the original `tim_memory.py` as the more
  paper-aligned hash-group reconstruction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, MemoryUnit, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import ConcatenateReadout, EmbeddingSimilarityRetrieval, LLMFunctionCallEvolution, TemplateReadout
from memprimitive.example.classics.tim_memory import extract_inductive_thoughts
from memprimitive.utils._runtime import get_runtime
from memprimitive.utils._template import structured_prompt, text_prompt


DEFAULT_MEMORY_LAYER = "thought_memory"
DEFAULT_CANDIDATE_TOP_K = 6
DEFAULT_RECALL_TOP_K = 3


def build_tim_simple_memory_system(
    *,
    memory_layer: str = DEFAULT_MEMORY_LAYER,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    recall_top_k: int = DEFAULT_RECALL_TOP_K,
) -> dict[str, object]:
    """Build a simplified TiM memory system with pure embedding retrieval."""

    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive.")
    if recall_top_k <= 0:
        raise ValueError("recall_top_k must be positive.")

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
    return {
        "store": store,
        "memory_layer": memory_layer,
        "candidate_top_k": int(candidate_top_k),
        "recall_top_k": int(recall_top_k),
    }


def build_tim_simple_query(query_text: str) -> Query:
    normalized_query = str(query_text).strip()
    if not normalized_query:
        raise ValueError("query_text must be non-empty.")
    return Query(
        text=normalized_query,
        embedding=list(get_runtime().embed(normalized_query)),
    )


def post_think_and_update_tim_simple_memory(
    system: dict[str, object],
    *,
    question: str,
    response: str,
    source: str = "tim_post_think",
    extra_metadata: dict[str, Any] | None = None,
) -> list[Packet]:
    """Extract thoughts, then maintain memory one thought at a time."""

    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    historical_thoughts = [
        {
            "thought": record.text,
            "head": str(record.metadata.get("head", "")).strip(),
            "relation": str(record.metadata.get("relation", "")).strip(),
            "tail": str(record.metadata.get("tail", "")).strip(),
        }
        for record in store.iter_records(memory_layer)
    ]
    extracted = extract_inductive_thoughts(
        question=question,
        response=response,
        historical_thoughts=historical_thoughts,
    )
    packets: list[Packet] = []
    for thought in extracted:
        packets.append(
            _run_tim_simple_thought_update(
                system,
                thought=_normalize_tim_simple_thought(
                    thought,
                    source_question=question,
                    source_response=response,
                    source=source,
                    extra_metadata=extra_metadata,
                ),
            )
        )
    return packets


def store_tim_simple_thought(
    system: dict[str, object],
    *,
    thought_text: str,
    head: str,
    relation: str,
    tail: str,
    source_question: str,
    source_response: str,
    source: str = "tim_post_think",
    extra_metadata: dict[str, Any] | None = None,
) -> Packet:
    """Store one pre-extracted thought through the simplified maintenance path."""

    thought = _normalize_tim_simple_thought(
        {
            "thought": thought_text,
            "head": head,
            "relation": relation,
            "tail": tail,
        },
        source_question=source_question,
        source_response=source_response,
        source=source,
        extra_metadata=extra_metadata,
    )
    return _run_tim_simple_thought_update(system, thought=thought)


def recall_tim_simple_thoughts(system: dict[str, object], *, user_query: str) -> list[dict[str, Any]]:
    """Recall simplified TiM thoughts with global embedding similarity."""

    recall_pipeline = build_tim_simple_recall_pipeline(system)
    packet = Packet(query=build_tim_simple_query(user_query))
    for module in recall_pipeline.retrieval:
        packet, recall_pipeline.store = module.run(packet, recall_pipeline.store)
    retrieved = packet.retrieved.items if packet.retrieved is not None else []
    return [
        {
            "record_id": record.record_id,
            "text": record.text,
            "head": record.metadata.get("head"),
            "relation": record.metadata.get("relation"),
            "tail": record.metadata.get("tail"),
        }
        for record in retrieved
    ]


def build_tim_simple_prompt(system: dict[str, object], *, query_text: str):
    """Render a retrieval-conditioned prompt from simplified TiM thoughts."""

    return build_tim_simple_recall_pipeline(system).recall(build_tim_simple_query(query_text))


def build_tim_simple_recall_pipeline(system: dict[str, object]) -> MemoryPipeline:
    """Build the global embedding-similarity recall pipeline."""

    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    return MemoryPipeline(
        retrieval=(
            EmbeddingSimilarityRetrieval(
                top_k=int(system["recall_top_k"]),
                layer=memory_layer,
            ),
        ),
        readout=TemplateReadout(
            prompt=text_prompt(
                "Recalled simplified TiM thoughts for the current query:\n"
                "{{ retrieved.items | join_text }}"
            )
        ),
        store=store,
    )


def _normalize_tim_simple_thought(
    thought: dict[str, Any],
    *,
    source_question: str,
    source_response: str,
    source: str,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_thought = str(thought.get("thought", "")).strip()
    normalized_head = str(thought.get("head", "")).strip()
    normalized_relation = str(thought.get("relation", "")).strip()
    normalized_tail = str(thought.get("tail", "")).strip()
    if not (normalized_thought and normalized_head and normalized_relation and normalized_tail):
        raise ValueError("thought_text, head, relation, and tail must all be non-empty.")
    return {
        "thought": normalized_thought,
        "head": normalized_head,
        "relation": normalized_relation,
        "tail": normalized_tail,
        "source_question": str(source_question),
        "source_response": str(source_response),
        "source": str(source),
        "thought_kind": "inductive_relation_thought",
        **({} if extra_metadata is None else dict(extra_metadata)),
    }


def _run_tim_simple_thought_update(system: dict[str, object], *, thought: dict[str, Any]) -> Packet:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    evolution = _build_tim_simple_evolution_module(system)
    packet = Packet(
        units=[_build_tim_simple_update_unit(thought=thought)],
        decisions_store={memory_layer: {"record_ids": []}},
    )
    packet, store = evolution.run(packet, store)
    return packet


def _build_tim_simple_update_unit(*, thought: dict[str, Any]) -> MemoryUnit:
    return MemoryUnit(
        text=str(thought["thought"]),
        unit_type="tim_simple_thought_update",
        metadata={
            "source": str(thought.get("source", "tim_post_think")),
            "source_question": str(thought["source_question"]),
            "source_response": str(thought["source_response"]),
            "thought": str(thought["thought"]),
            "head": str(thought["head"]),
            "relation": str(thought["relation"]),
            "tail": str(thought["tail"]),
            "thought_kind": str(thought.get("thought_kind", "inductive_relation_thought")),
        },
    )


def _build_tim_simple_evolution_module(system: dict[str, object]) -> LLMFunctionCallEvolution:
    memory_layer = str(system["memory_layer"])
    candidate_pipeline = _build_tim_simple_candidate_recall_pipeline(system)
    return LLMFunctionCallEvolution(
        target_layer=memory_layer,
        tools=["ADD", "UPDATE", "DELETE"],
        prompt=structured_prompt(
            {
                "blocks": [
                    {
                        "id": "task",
                        "title": "Task",
                        "template": (
                            "You are maintaining a simplified TiM memory store.\n"
                            "Process exactly one newly extracted thought at a time.\n"
                            "The visible records are the top-k embedding-similar historical thoughts recalled for this thought.\n"
                            "Use ADD if the thought should be stored as a new memory.\n"
                            "Use UPDATE when one visible recalled record should be rewritten into a better canonical thought.\n"
                            "Use DELETE when one visible recalled record is contradictory or redundant.\n"
                            f"The memory store writes only to the fixed '{memory_layer}' layer.\n"
                            "For ADD, do not invent any other layer name; rely on the provided default_target_layer rather than making up a target layer.\n"
                            "Only UPDATE or DELETE visible recalled records.\n"
                            "For ADD, include metadata with head, relation, tail, source_question, source_response, and thought_kind.\n"
                            "When updating a kept record, keep its head/relation/tail aligned via metadata_patch.\n"
                            "If no change is needed, make no tool call."
                        ),
                    },
                    {
                        "id": "current_thought",
                        "title": "Current New Thought",
                        "template": (
                            "text={{ current_thought.text }}\n"
                            "triple=({{ current_thought.head }}, {{ current_thought.relation }}, {{ current_thought.tail }})\n"
                            "source_question={{ current_thought.source_question }}\n"
                            "source_response={{ current_thought.source_response }}"
                        ),
                    },
                    {
                        "id": "candidate_memories",
                        "title": "Top-K Similar Historical Thoughts",
                        "template": "{{ candidate_recall }}",
                    },
                    {
                        "id": "visible_records",
                        "title": "Visible Candidate Records",
                        "condition": "visible_records | length",
                        "repeat_over": "visible_records",
                        "item_template": (
                            "- record_id={{ item.record_id }} | text={{ item.text }} | "
                            "triple=({{ item.metadata.head }}, {{ item.metadata.relation }}, {{ item.metadata.tail }})"
                        ),
                        "separator": "\n",
                    },
                    {
                        "id": "available_tools",
                        "title": "Available Tools",
                        "repeat_over": "tools",
                        "item_template": "- {{ item.name }}",
                        "separator": "\n",
                    },
                    {
                        "id": "default_target_layer",
                        "title": "Default Target Layer",
                        "template": "{{ default_target_layer }}",
                    },
                ]
            },
            context_builder=lambda packet, current_store: {
                "current_thought": _current_thought_prompt_context(packet),
            },
            labeled_recall_plans={
                "candidate_recall": text_prompt("{{ candidate_recall }}"),
            },
            labeled_sub_recall_pipelines={
                "candidate_recall": candidate_pipeline,
            },
            labeled_recall_query_builders={
                "candidate_recall": lambda packet, current_store, context: build_tim_simple_query(
                    str(context.get("current_thought", {}).get("text", ""))
                ),
            },
            visible_record_recall_labels=("candidate_recall",),
        ),
    )


def _current_thought_prompt_context(packet: Packet) -> dict[str, str]:
    unit = packet.units[0] if packet.units else None
    metadata = unit.metadata if unit is not None and isinstance(unit.metadata, dict) else {}
    return {
        "text": "" if unit is None else str(unit.text),
        "head": str(metadata.get("head", "")).strip(),
        "relation": str(metadata.get("relation", "")).strip(),
        "tail": str(metadata.get("tail", "")).strip(),
        "source_question": str(metadata.get("source_question", "")).strip(),
        "source_response": str(metadata.get("source_response", "")).strip(),
    }


def _build_tim_simple_candidate_recall_pipeline(system: dict[str, object]) -> MemoryPipeline:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    return MemoryPipeline(
        retrieval=(
            EmbeddingSimilarityRetrieval(
                top_k=int(system["candidate_top_k"]),
                layer=memory_layer,
            ),
        ),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )


def main() -> None:
    system = build_tim_simple_memory_system()
    store = system["store"]
    assert isinstance(store, MemoryStore)

    post_think_and_update_tim_simple_memory(
        system,
        question="Which city is the capital of China?",
        response="Beijing.",
    )
    post_think_and_update_tim_simple_memory(
        system,
        question="What does Alice like to drink in the evening?",
        response="Alice likes jasmine tea in the evening.",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("stored simplified TiM thoughts:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "triple": (
                    record.metadata.get("head"),
                    record.metadata.get("relation"),
                    record.metadata.get("tail"),
                ),
            }
            for record in store.iter_records(str(system["memory_layer"]))
        ]
    )
    print()

    print("retrieval-conditioned prompt:")
    print(build_tim_simple_prompt(system, query_text="What is the capital of China?").text)


if __name__ == "__main__":
    main()
