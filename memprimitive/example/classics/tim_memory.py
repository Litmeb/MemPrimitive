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
2. The paper does not define an explicit "merge write primitive".
3. The paper does not specify one unique packet/module decomposition for a
   framework reconstruction.

Implementation decisions made here:

1. Update runs one bucket-level batch per affected hash bucket.
2. Insert is a bucket-level ``LLMFunctionCallOrganization`` over the current
   round's newly extracted thoughts plus historical same-bucket thoughts.
3. Forget and merge are implemented by ``LLMFunctionCallEvolution`` over the
   full visible hash group for the bucket.
4. Merge is represented as:
   - ``UPDATE`` one keeper record into the merged canonical thought, then
   - ``DELETE`` redundant records.
5. LSH-style bucket assignment is implemented as example-level helper logic,
   not as a new reusable framework primitive.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryRecord, MemoryStore, MemoryUnit, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    LLMFunctionCallEvolution,
    LLMFunctionCallOrganization,
    LLMRepresentation,
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
    *,
    historical_thoughts: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Extract TiM-style inductive thoughts via ``LLMRepresentation``."""

    normalized_history = _normalize_tim_historical_thoughts(historical_thoughts)
    extraction_packet = Packet(
        units=[
            MemoryUnit(
                text="TiM post-thinking extraction",
                unit_type="tim_thought_extraction",
                metadata={
                    "question": str(question).strip(),
                    "response": str(response).strip(),
                    "historical_thoughts": normalized_history,
                },
            )
        ]
    )
    extraction_representation = _build_tim_thought_extraction_representation()
    extraction_packet, _ = extraction_representation.run(extraction_packet, MemoryStore())
    payload = extraction_packet.units[0].metadata.get("representation", {}).get("thoughts", [])
    return payload if isinstance(payload, list) else []


def _normalize_tim_historical_thoughts(historical_thoughts: list[dict[str, str]] | None) -> list[dict[str, str]]:
    return [
        {
            "thought": str(item.get("thought", "")).strip(),
            "head": str(item.get("head", "")).strip(),
            "relation": str(item.get("relation", "")).strip(),
            "tail": str(item.get("tail", "")).strip(),
        }
        for item in (historical_thoughts or [])
        if isinstance(item, dict)
        and str(item.get("thought", "")).strip()
        and str(item.get("head", "")).strip()
        and str(item.get("relation", "")).strip()
        and str(item.get("tail", "")).strip()
    ]


def _build_tim_thought_extraction_representation() -> LLMRepresentation:
    return LLMRepresentation(
        field="thoughts",
        value_type=list[dict[str, str]],
        prompt=structured_prompt(
            {
                "blocks": [
                    {
                        "id": "task",
                        "title": "Task",
                        "template": (
                            "Extract TiM-style inductive thoughts for TiM post-thinking.\n"
                            "Return only grounded relation-bearing thoughts supported by the current response.\n"
                            "Each item must contain thought, head, relation, and tail.\n"
                            "Each thought must be one factual relation sentence aligned to its triple.\n"
                            "Use both the current question-response pair and the provided historical thoughts when useful.\n"
                            "Avoid duplicates and avoid vague summaries."
                        ),
                    },
                    {
                        "id": "current_round",
                        "title": "Current Question-Response Pair",
                        "template": (
                            "question={{ unit.metadata.question }}\n"
                            "response={{ unit.metadata.response }}"
                        ),
                    },
                    {
                        "id": "historical_thoughts",
                        "title": "Historical Thoughts",
                        "condition": "unit.metadata.historical_thoughts | length",
                        "repeat_over": "unit.metadata.historical_thoughts",
                        "item_template": (
                            "- thought={{ item.thought }} | "
                            "triple=({{ item.head }}, {{ item.relation }}, {{ item.tail }})"
                        ),
                        "separator": "\n",
                    },
                    {
                        "id": "paper_examples",
                        "title": "Paper-Style Examples",
                        "repeat_over": "examples",
                        "item_template": (
                            "question={{ item.question }}\n"
                            "response={{ item.response }}\n"
                            "output={{ item.output }}"
                        ),
                        "separator": "\n\n",
                    },
                ]
            },
            context_builder=lambda packet, current_store: {
                "examples": [
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
                ]
            },
        ),
    )


def post_think_and_update_memory(
    system: dict[str, object],
    *,
    question: str,
    response: str,
    source: str = "tim_post_think",
    extra_metadata: dict[str, Any] | None = None,
) -> list[Packet]:
    """Run one round-level TiM update over all newly extracted thoughts."""

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
    grouped = _group_thoughts_by_bucket(
        system,
        thoughts=extracted,
        source_question=question,
        source_response=response,
        source=source,
        extra_metadata=extra_metadata,
    )
    packets: list[Packet] = []
    for hash_bucket, bucket_thoughts in grouped.items():
        packets.append(_run_tim_bucket_update(system, hash_bucket=hash_bucket, thoughts=bucket_thoughts))
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
    """Store one pre-extracted thought via the same bucket-level batch path."""

    normalized_thought = str(thought_text).strip()
    normalized_head = str(head).strip()
    normalized_relation = str(relation).strip()
    normalized_tail = str(tail).strip()
    if not (normalized_thought and normalized_head and normalized_relation and normalized_tail):
        raise ValueError("thought_text, head, relation, and tail must all be non-empty.")

    hash_bucket = compute_hash_bucket(system, normalized_thought)
    thought = {
        "thought": normalized_thought,
        "head": normalized_head,
        "relation": normalized_relation,
        "tail": normalized_tail,
        "hash_bucket": hash_bucket,
        "source_question": str(source_question),
        "source_response": str(source_response),
        "source": str(source),
        "thought_kind": "inductive_relation_thought",
        **({} if extra_metadata is None else dict(extra_metadata)),
    }
    return _run_tim_bucket_update(system, hash_bucket=hash_bucket, thoughts=[thought])


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

    hash_bucket = compute_hash_bucket(system, query_text)
    return _build_bucket_recall_pipeline(
        system,
        hash_bucket=hash_bucket,
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


def _group_thoughts_by_bucket(
    system: dict[str, object],
    *,
    thoughts: list[dict[str, str]],
    source_question: str,
    source_response: str,
    source: str,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for thought in thoughts:
        normalized_thought = str(thought["thought"]).strip()
        hash_bucket = compute_hash_bucket(system, normalized_thought)
        grouped[hash_bucket].append(
            {
                "thought": normalized_thought,
                "head": str(thought["head"]).strip(),
                "relation": str(thought["relation"]).strip(),
                "tail": str(thought["tail"]).strip(),
                "hash_bucket": hash_bucket,
                "source_question": str(source_question),
                "source_response": str(source_response),
                "source": str(source),
                "thought_kind": "inductive_relation_thought",
                **({} if extra_metadata is None else dict(extra_metadata)),
            }
        )
    return dict(grouped)


def _run_tim_bucket_update(
    system: dict[str, object],
    *,
    hash_bucket: str,
    thoughts: list[dict[str, Any]],
) -> Packet:
    if not thoughts:
        raise ValueError("TiM bucket update requires at least one thought.")

    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    historical_records = _records_in_bucket(store, memory_layer=memory_layer, hash_bucket=hash_bucket)

    organization = _build_tim_bucket_insert_organization(system, hash_bucket=hash_bucket)
    organization_packet = Packet(
        units=[_build_bucket_update_unit(hash_bucket=hash_bucket, thoughts=thoughts, historical_records=historical_records)],
        decisions=[True],
    )
    organization_packet, store = organization.run(organization_packet, store)

    written_record_ids = [
        str(record_id).strip()
        for record_id in organization_packet.trace.get("organization", {}).get("written_record_ids", [])
        if str(record_id).strip()
    ]
    if not written_record_ids:
        return organization_packet

    evolution = _build_tim_bucket_evolution_module(system, hash_bucket=hash_bucket)
    evolution_packet = replace(
        organization_packet,
        decisions_store={memory_layer: {"record_ids": written_record_ids}},
        units=None,
        decisions=None,
        placements=None,
    )
    evolution_packet, store = evolution.run(evolution_packet, store)
    return evolution_packet


def _build_bucket_update_unit(
    *,
    hash_bucket: str,
    thoughts: list[dict[str, Any]],
    historical_records: list[MemoryRecord],
) -> MemoryUnit:
    source_question = str(thoughts[0].get("source_question", ""))
    source_response = str(thoughts[0].get("source_response", ""))
    source = str(thoughts[0].get("source", "tim_post_think"))
    return MemoryUnit(
        text=f"TiM bucket update for {hash_bucket}",
        unit_type="tim_bucket_update",
        metadata={
            "source": source,
            "hash_bucket": hash_bucket,
            "source_question": source_question,
            "source_response": source_response,
            "new_thoughts": [
                {
                    "text": thought["thought"],
                    "head": thought["head"],
                    "relation": thought["relation"],
                    "tail": thought["tail"],
                    "hash_bucket": thought["hash_bucket"],
                }
                for thought in thoughts
            ],
            "historical_bucket_records": [
                {
                    "record_id": record.record_id,
                    "text": record.text,
                    "head": record.metadata.get("head"),
                    "relation": record.metadata.get("relation"),
                    "tail": record.metadata.get("tail"),
                    "hash_bucket": record.metadata.get("hash_bucket"),
                }
                for record in historical_records
            ],
        },
    )


def _build_tim_bucket_insert_organization(system: dict[str, object], *, hash_bucket: str) -> LLMFunctionCallOrganization:
    memory_layer = str(system["memory_layer"])
    _ = hash_bucket
    return LLMFunctionCallOrganization(
        tools=["ADD"],
        target_layer=memory_layer,
        prompt=structured_prompt(
            {
                "blocks": [
                    {
                        "id": "task",
                        "title": "Task",
                        "template": (
                            "You are performing the TiM insert stage for one hash bucket.\n"
                            "Review the current question/response, the newly extracted thoughts, and the historical same-bucket thoughts.\n"
                            "Use ADD once for each new thought that should be inserted into memory.\n"
                            "Do not rewrite or delete records in this stage.\n"
                            "For every ADD, include metadata with head, relation, tail, hash_bucket, source_question, source_response, and thought_kind."
                        ),
                    },
                    {
                        "id": "bucket",
                        "title": "Hash Bucket",
                        "template": "{{ unit.metadata.hash_bucket }}",
                    },
                    {
                        "id": "current_round",
                        "title": "Current Question-Response Pair",
                        "template": (
                            "question={{ unit.metadata.source_question }}\n"
                            "response={{ unit.metadata.source_response }}"
                        ),
                    },
                    {
                        "id": "new_thoughts",
                        "title": "Newly Extracted Thoughts",
                        "repeat_over": "unit.metadata.new_thoughts",
                        "item_template": (
                            "- text={{ item.text }} | "
                            "triple=({{ item.head }}, {{ item.relation }}, {{ item.tail }}) | "
                            "hash_bucket={{ item.hash_bucket }}"
                        ),
                        "separator": "\n",
                    },
                    {
                        "id": "historical_thoughts",
                        "title": "Historical Same-Bucket Thoughts",
                        "condition": "unit.metadata.historical_bucket_records | length",
                        "repeat_over": "unit.metadata.historical_bucket_records",
                        "item_template": (
                            "- record_id={{ item.record_id }} | text={{ item.text }} | "
                            "triple=({{ item.head }}, {{ item.relation }}, {{ item.tail }}) | "
                            "hash_bucket={{ item.hash_bucket }}"
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
            }
        ),
    )


def _build_tim_bucket_evolution_module(system: dict[str, object], *, hash_bucket: str) -> LLMFunctionCallEvolution:
    memory_layer = str(system["memory_layer"])
    full_bucket_pipeline = _build_full_bucket_visibility_pipeline(system, hash_bucket=hash_bucket)
    return LLMFunctionCallEvolution(
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
                            "The selected records are the newly inserted thoughts from the current round.\n"
                            "The visible records are the full same-bucket hash group.\n"
                            "Only use the provided tools.\n"
                            "Use UPDATE when a thought should be rewritten into a better merged canonical statement.\n"
                            "When you update a kept record after merge, also keep its head/relation/tail metadata aligned via metadata_patch.\n"
                            "Use DELETE when a visible thought is contradictory or redundant.\n"
                            "If multiple thoughts should merge, keep one representative record via UPDATE and delete the others.\n"
                            "Only operate on visible same-bucket records. If no change is needed, make no tool call."
                        ),
                    },
                    {
                        "id": "new_records",
                        "title": "Newly Inserted Thoughts",
                        "repeat_over": "selected_records",
                        "item_template": (
                            "- record_id={{ item.record_id }} | text={{ item.text }} | "
                            "hash_bucket={{ item.metadata.hash_bucket }} | "
                            "triple=({{ item.metadata.head }}, {{ item.metadata.relation }}, {{ item.metadata.tail }})"
                        ),
                        "separator": "\n",
                    },
                    {
                        "id": "same_bucket_group",
                        "title": "Full Same-Bucket Thought Group",
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
            sub_recall_pipeline=full_bucket_pipeline,
            recall_query_builder=lambda packet, current_store, context: Query(
                text=f"same-bucket thought maintenance for {hash_bucket}",
                metadata={"hash_bucket": hash_bucket},
            ),
            visible_record_recall_labels=(PRIMARY_RECALL_LABEL,),
        ),
    )


def _build_full_bucket_visibility_pipeline(system: dict[str, object], *, hash_bucket: str) -> MemoryPipeline:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    memory_layer = str(system["memory_layer"])
    full_bucket_size = max(len(_records_in_bucket(store, memory_layer=memory_layer, hash_bucket=hash_bucket)), 1)
    return MemoryPipeline(
        retrieval=(
            MetadataRetrieval(
                top_k=full_bucket_size,
                field="hash_bucket",
                target=hash_bucket,
                layer=memory_layer,
                source="store",
            ),
        ),
        readout=ConcatenateReadout(),
        store=store,
    )


def _records_in_bucket(store: MemoryStore, *, memory_layer: str, hash_bucket: str) -> list[MemoryRecord]:
    return [
        record
        for record in store.iter_records(memory_layer)
        if str(record.metadata.get("hash_bucket", "")).strip() == hash_bucket
    ]


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
