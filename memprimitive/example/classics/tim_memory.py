"""Mechanism-level reconstruction of the TiM memory module.

This file reconstructs the memory side of:

    Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory
    https://arxiv.org/pdf/2311.08719

Scope boundary:

- This file only implements the memory module.
- It intentionally excludes the surrounding agent loop.

What the paper clearly guarantees:

1. Memory stores post-thinking "thoughts" rather than raw conversation history.
2. Thoughts are inductive relation-bearing sentences that can align with
   relation triples.
3. Each thought is stored with a hash index.
4. Recall is two-stage: first retrieve a hash group, then do similarity-based
   top-k retrieval within that group.
5. Memory updating supports insert, forget, and merge.

What is not fully specified by the paper and is therefore an implementation
decision in this file:

1. The paper gives illustrative prompts for forget/merge, but does not specify
   a deterministic execution protocol, output schema, or target-selection rule.
2. The paper does not specify whether forget/merge sees the entire memory or a
   bounded local candidate set.
3. The paper does not define an explicit "merge write primitive".

Implementation decisions made here:

1. We restrict forget/merge to one local hash bucket at a time.
2. Insert happens by normal append of each newly extracted thought.
3. Forget and merge are both implemented by ``LLMFunctionCallEvolution`` over
   the visible same-bucket candidate set.
4. Merge is represented as:
   - ``UPDATE`` one keeper record into the merged canonical thought, then
   - ``DELETE`` redundant records.
5. LSH-style bucket assignment is implemented as example-level helper logic,
   not as a new reusable framework primitive. This keeps the reconstruction
   within existing MemPrimitive baseline families.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    EmbeddingSimilarityRetrieval,
    LLMFunctionCallEvolution,
    MetadataRetrieval,
    TemplateReadout,
)
from memprimitive.utils._runtime import get_runtime
from memprimitive.utils._template import PRIMARY_RECALL_LABEL, structured_prompt, text_prompt


DEFAULT_MEMORY_LAYER = "thought_memory"
DEFAULT_BUCKET_COUNT = 16
DEFAULT_BUCKET_CANDIDATE_K = 6
DEFAULT_RECALL_TOP_K = 3
DEFAULT_HASH_SEED = 7


def build_tim_memory_system(
    *,
    memory_layer: str = DEFAULT_MEMORY_LAYER,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    bucket_candidate_k: int = DEFAULT_BUCKET_CANDIDATE_K,
    recall_top_k: int = DEFAULT_RECALL_TOP_K,
    hash_seed: int = DEFAULT_HASH_SEED,
) -> dict[str, object]:
    """Build a TiM-style memory-only system from existing primitives."""

    if bucket_count <= 1 or bucket_count % 2 != 0:
        raise ValueError("bucket_count must be an even integer greater than 1.")
    if bucket_candidate_k <= 0:
        raise ValueError("bucket_candidate_k must be positive.")
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
        "bucket_count": int(bucket_count),
        "bucket_candidate_k": int(bucket_candidate_k),
        "recall_top_k": int(recall_top_k),
        "hash_seed": int(hash_seed),
        "hash_projection": None,
    }


def extract_inductive_thoughts(
    question: str,
    response: str,
) -> list[dict[str, str]]:
    """Extract TiM-style inductive thoughts from one question-response pair."""

    runtime = get_runtime()
    payload = runtime.json(
        system=(
            "Extract TiM-style inductive thoughts from a question-response pair.\n"
            "Return a strict JSON array. Each item must be an object with keys:\n"
            "thought, head, relation, tail.\n"
            "Each thought must be one grounded factual relation sentence aligned to the triple.\n"
            "Only keep thoughts supported by the response.\n"
            "Avoid duplicates and avoid vague summaries."
        ),
        user=json.dumps(
            {
                "question": question,
                "response": response,
                "paper_examples": [
                    {
                        "question": "Do you have any company recommendations for me?",
                        "response": "I recommend Google.",
                        "output": [
                            {
                                "head": "Company",
                                "relation": "Recommended",
                                "tail": "Google",
                                "thought": "Recommended company is Google.",
                            }
                        ],
                    },
                    {
                        "question": "Which City is the capital of China?",
                        "response": "Beijing.",
                        "output": [
                            {
                                "head": "China",
                                "relation": "Capital",
                                "tail": "Beijing",
                                "thought": "The capital of China is Beijing.",
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
    )
    if not isinstance(payload, list):
        raise ValueError("Thought extraction must return a JSON list.")

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        thought = str(item.get("thought", "")).strip()
        head = str(item.get("head", "")).strip()
        relation = str(item.get("relation", "")).strip()
        tail = str(item.get("tail", "")).strip()
        if not (thought and head and relation and tail):
            continue
        key = (thought, head, relation, tail)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "thought": thought,
                "head": head,
                "relation": relation,
                "tail": tail,
            }
        )
    return normalized


def post_think_and_update_memory(
    system: dict[str, object],
    *,
    question: str,
    response: str,
    source: str = "tim_post_think",
    extra_metadata: dict[str, Any] | None = None,
) -> list[Packet]:
    """Run TiM post-thinking, then insert and locally maintain extracted thoughts."""

    packets: list[Packet] = []
    for thought in extract_inductive_thoughts(question=question, response=response):
        packets.append(
            store_thought(
                system,
                thought_text=thought["thought"],
                head=thought["head"],
                relation=thought["relation"],
                tail=thought["tail"],
                source_question=question,
                source_response=response,
                source=source,
                extra_metadata=extra_metadata,
            )
        )
    return packets


def store_thought(
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
    """Store one extracted thought, then run same-bucket forget/merge maintenance."""

    normalized_thought = str(thought_text).strip()
    normalized_head = str(head).strip()
    normalized_relation = str(relation).strip()
    normalized_tail = str(tail).strip()
    if not (normalized_thought and normalized_head and normalized_relation and normalized_tail):
        raise ValueError("thought_text, head, relation, and tail must all be non-empty.")

    bucket = compute_hash_bucket(system, normalized_thought)
    write_pipeline = _build_tim_thought_write_pipeline(system, hash_bucket=bucket)
    observation = Observation(
        text=normalized_thought,
        source=source,
        metadata={
            "head": normalized_head,
            "relation": normalized_relation,
            "tail": normalized_tail,
            "hash_bucket": bucket,
            "source_question": str(source_question),
            "source_response": str(source_response),
            "thought_kind": "inductive_relation_thought",
            **({} if extra_metadata is None else dict(extra_metadata)),
        },
    )
    return write_pipeline.ingest(observation)


def recall_thoughts(system: dict[str, object], *, user_query: str) -> list[dict[str, Any]]:
    """Recall TiM thoughts with hash-bucket prefilter plus within-bucket top-k."""

    recall_pipeline = build_tim_recall_pipeline(system, query_text=user_query)
    packet = Packet(query=build_tim_query(system, query_text=user_query))
    for module in recall_pipeline.retrieval:
        packet, recall_pipeline.store = module.run(packet, recall_pipeline.store)
    retrieved = packet.retrieved.items if packet.retrieved is not None else []
    return [
        {
            "record_id": record.record_id,
            "text": record.text,
            "hash_bucket": record.metadata.get("hash_bucket"),
            "head": record.metadata.get("head"),
            "relation": record.metadata.get("relation"),
            "tail": record.metadata.get("tail"),
        }
        for record in retrieved
    ]


def build_tim_prompt(system: dict[str, object], *, query_text: str):
    """Render a retrieval-conditioned prompt from recalled TiM thoughts."""

    return build_tim_recall_pipeline(system, query_text=query_text).recall(
        build_tim_query(system, query_text=query_text)
    )


def build_tim_query(system: dict[str, object], *, query_text: str) -> Query:
    normalized_query = str(query_text).strip()
    if not normalized_query:
        raise ValueError("query_text must be non-empty.")
    return Query(
        text=normalized_query,
        embedding=list(get_runtime().embed(normalized_query)),
        metadata={"hash_bucket": compute_hash_bucket(system, normalized_query)},
    )


def build_tim_recall_pipeline(system: dict[str, object], *, query_text: str) -> MemoryPipeline:
    """Build a TiM recall pipeline for one query using the query's hash bucket."""

    bucket = compute_hash_bucket(system, query_text)
    return _build_bucket_recall_pipeline(
        system,
        hash_bucket=bucket,
        final_top_k=int(system["recall_top_k"]),
        prompt=text_prompt(
            "Recalled TiM thoughts for the current query:\n"
            "{{ retrieved.items | join_text }}"
        ),
    )


def compute_hash_bucket(system: dict[str, object], text: str) -> str:
    """Compute the example-level TiM hash bucket for one text."""

    embedding = list(get_runtime().embed(str(text).strip()))
    bucket_index = _bucket_index_for_embedding(system, embedding)
    return f"bucket-{bucket_index}"


def _build_tim_thought_write_pipeline(system: dict[str, object], *, hash_bucket: str) -> MemoryPipeline:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    bucket_candidate_k = int(system["bucket_candidate_k"])

    bucket_recall_pipeline = _build_bucket_recall_pipeline(
        system,
        hash_bucket=hash_bucket,
        final_top_k=bucket_candidate_k,
        prompt=text_prompt("{{ retrieved.items | join_text }}"),
    )

    return MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer=memory_layer),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=LLMFunctionCallEvolution(
            source_layer=memory_layer,
            target_layer=memory_layer,
            tools=["UPDATE", "DELETE"],
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "You are organizing TiM thoughts inside one local hash bucket.\n"
                                "The selected record is the newly inserted current thought.\n"
                                "Only use the provided tools.\n"
                                "Use UPDATE when a thought should be rewritten into a better merged canonical statement.\n"
                                "When you update a kept record after merge, also keep its head/relation/tail metadata aligned via metadata_patch.\n"
                                "Use DELETE when a visible thought is contradictory or redundant.\n"
                                "If multiple thoughts should merge, keep one representative record via UPDATE and delete the others.\n"
                                "Only operate on visible same-bucket records. If no change is needed, make no tool call."
                            ),
                        },
                        {
                            "id": "current_thought",
                            "title": "Current Thought",
                            "template": (
                                "record_id={{ selected_records.0.record_id }}\n"
                                "text={{ selected_records.0.text }}\n"
                                "hash_bucket={{ selected_records.0.metadata.hash_bucket }}\n"
                                "triple=({{ selected_records.0.metadata.head }}, "
                                "{{ selected_records.0.metadata.relation }}, "
                                "{{ selected_records.0.metadata.tail }})"
                            ),
                        },
                        {
                            "id": "same_bucket_candidates",
                            "title": "Same-Bucket Retrieved Thoughts",
                            "template": "{{ recalled_prompt }}",
                        },
                        {
                            "id": "visible_records",
                            "title": "Visible Records",
                            "condition": "visible_records | length",
                            "repeat_over": "visible_records",
                            "item_template": (
                                "- record_id={{ item.record_id }} | text={{ item.text }} | "
                                "hash_bucket={{ item.metadata.hash_bucket }} | "
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
                    ]
                },
                recall_plan=text_prompt("{{ recalled_prompt }}"),
                sub_recall_pipeline=bucket_recall_pipeline,
                recall_query_builder=(
                    lambda packet, current_store, context: (
                        Query(
                            text=str(context["selected_records"][0]["text"]),
                            embedding=list(context["selected_records"][0]["embedding"]),
                        )
                        if context.get("selected_records") and context["selected_records"][0].get("embedding")
                        else (
                            Query(text=str(context["selected_records"][0]["text"]))
                            if context.get("selected_records")
                            else Query(text="same-bucket thought maintenance")
                        )
                    )
                ),
                visible_record_recall_labels=(PRIMARY_RECALL_LABEL,),
            ),
        ),
        store=store,
    )


def _build_bucket_recall_pipeline(
    system: dict[str, object],
    *,
    hash_bucket: str,
    final_top_k: int,
    prompt,
) -> MemoryPipeline:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    candidate_count = max(store.count(memory_layer) + 1, final_top_k, 1)
    return MemoryPipeline(
        retrieval=(
            MetadataRetrieval(
                top_k=candidate_count,
                field="hash_bucket",
                target=hash_bucket,
                layer=memory_layer,
                source="store",
            ),
            EmbeddingSimilarityRetrieval(
                top_k=final_top_k,
                layer=memory_layer,
                source="retrieved",
            ),
        ),
        readout=TemplateReadout(prompt=prompt),
        store=store,
    )


def _bucket_index_for_embedding(system: dict[str, object], embedding: list[float]) -> int:
    if not embedding:
        raise ValueError("LSH bucket computation requires a non-empty embedding.")
    projection = _ensure_hash_projection(system, dim=len(embedding))
    scores: list[float] = []
    for column in projection:
        scores.append(sum(value * weight for value, weight in zip(embedding, column, strict=True)))
    signed_scores = [*scores, *[-score for score in scores]]
    return max(range(len(signed_scores)), key=lambda index: signed_scores[index])


def _ensure_hash_projection(system: dict[str, object], *, dim: int) -> list[list[float]]:
    cached = system.get("hash_projection")
    if isinstance(cached, list) and cached and len(cached[0]) == dim:
        return cached

    bucket_count = int(system["bucket_count"])
    rng = random.Random(int(system["hash_seed"]))
    half_bucket_count = bucket_count // 2
    projection = [
        [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        for _ in range(half_bucket_count)
    ]
    system["hash_projection"] = projection
    return projection


def main() -> None:
    system = build_tim_memory_system()
    store = system["store"]
    assert isinstance(store, MemoryStore)

    post_think_and_update_memory(
        system,
        question="Which city is the capital of China?",
        response="Beijing.",
    )
    post_think_and_update_memory(
        system,
        question="What does Alice like to drink in the evening?",
        response="Alice likes jasmine tea in the evening.",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("stored TiM thoughts:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "hash_bucket": record.metadata.get("hash_bucket"),
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
    print(build_tim_prompt(system, query_text="What is the capital of China?").text)


if __name__ == "__main__":
    main()
