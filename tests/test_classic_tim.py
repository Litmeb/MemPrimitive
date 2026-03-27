from __future__ import annotations

import pytest

from memprimitive import Packet, Query
from memprimitive.classic_modules.tim import (
    TIM_THOUGHT_LAYER,
)
from memprimitive.example.classics.tim_think_in_memory import (
    TimWorkstream,
    build_tim_pipeline,
    tim_postthought_observation,
)


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _thought(head: str, relation: str, tail: str, *, query: str, response: str, turn_id: str) -> dict[str, str]:
    return {
        "thought_text": f"{head} {relation} {tail}",
        "head_entity": head,
        "relation": relation,
        "tail_entity": tail,
        "source_query": query,
        "source_response": response,
        "source_turn_id": turn_id,
    }


def test_tim_materializes_structured_postthoughts_and_indexes_buckets() -> None:
    pipeline = build_tim_pipeline(top_k=3, readout_item_budget=3)
    observation = tim_postthought_observation(
        query="What drink does Alice prefer?",
        response="Alice prefers tea.",
        thoughts=[
            _thought(
                "Alice",
                "prefers",
                "tea",
                query="What drink does Alice prefer?",
                response="Alice prefers tea.",
                turn_id="turn-1",
            ),
            _thought(
                "Alice",
                "enjoys",
                "quiet work",
                query="What drink does Alice prefer?",
                response="Alice prefers tea.",
                turn_id="turn-1",
            ),
        ],
        turn_id="turn-1",
    )

    packet = pipeline.ingest(observation)

    assert packet.units is not None
    assert len(packet.units) == 2
    assert all(unit.unit_type == "tim_thought" for unit in packet.units)
    assert all(unit.metadata["tim"]["hash_index"] for unit in packet.units)
    assert pipeline.store.count(TIM_THOUGHT_LAYER) == 2
    assert pipeline.store.metadata["tim"]["buckets"]


def test_tim_retrieval_runs_bucket_then_similarity_without_global_fallback() -> None:
    pipeline = build_tim_pipeline(top_k=3, readout_item_budget=3)
    pipeline.ingest(
        tim_postthought_observation(
            query="What drink does Alice prefer?",
            response="Alice prefers tea.",
            thoughts=[
                _thought(
                    "Alice",
                    "prefers",
                    "tea",
                    query="What drink does Alice prefer?",
                    response="Alice prefers tea.",
                    turn_id="turn-1",
                )
            ],
            turn_id="turn-1",
        )
    )
    pipeline.ingest(
        tim_postthought_observation(
            query="What drink does Bob prefer?",
            response="Bob prefers coffee.",
            thoughts=[
                _thought(
                    "Bob",
                    "prefers",
                    "coffee",
                    query="What drink does Bob prefer?",
                    response="Bob prefers coffee.",
                    turn_id="turn-2",
                )
            ],
            turn_id="turn-2",
        )
    )

    packet, _ = pipeline.retrieval.run(Packet(query=Query(text="Alice tea preference")), pipeline.store)

    assert packet.retrieved is not None
    assert packet.retrieved.trace["query_bucket"]
    assert packet.retrieved.trace["candidate_bucket_ids"]
    assert packet.retrieved.trace["selected_group_size"] >= 1
    assert packet.retrieved.items
    assert "Alice" in packet.retrieved.items[0].text


def test_tim_forgets_conflicting_thoughts_with_real_llm() -> None:
    workflow = TimWorkstream(top_k=3, readout_item_budget=3)
    workflow.ingest_postthoughts(
        query="Where does Alice work?",
        response="Alice works at ACME.",
        thoughts=[
            _thought(
                "Alice",
                "works_at",
                "ACME",
                query="Where does Alice work?",
                response="Alice works at ACME.",
                turn_id="turn-1",
            )
        ],
        turn_id="turn-1",
    )
    packet = workflow.ingest_postthoughts(
        query="Where does Alice work now?",
        response="Alice works at Globex now.",
        thoughts=[
            _thought(
                "Alice",
                "works_at",
                "Globex",
                query="Where does Alice work now?",
                response="Alice works at Globex now.",
                turn_id="turn-2",
            )
        ],
        turn_id="turn-2",
    )

    effects = packet.trace["memory_evolution"]["effects"]
    assert any(effect["effect_type"] == "forget" for effect in effects)
    texts = [record.text for record in workflow.store.iter_records(TIM_THOUGHT_LAYER)]
    assert any("Globex" in text for text in texts)


def test_tim_merges_same_head_duplicate_thoughts_with_real_llm() -> None:
    workflow = TimWorkstream(top_k=3, readout_item_budget=3)
    workflow.ingest_postthoughts(
        query="What does Alice like?",
        response="Alice likes green tea.",
        thoughts=[
            _thought(
                "Alice",
                "likes",
                "green tea",
                query="What does Alice like?",
                response="Alice likes green tea.",
                turn_id="turn-1",
            )
        ],
        turn_id="turn-1",
    )
    packet = workflow.ingest_postthoughts(
        query="What else does Alice like?",
        response="Alice likes jasmine tea.",
        thoughts=[
            _thought(
                "Alice",
                "likes",
                "jasmine tea",
                query="What else does Alice like?",
                response="Alice likes jasmine tea.",
                turn_id="turn-2",
            )
        ],
        turn_id="turn-2",
    )

    assert any(effect["effect_type"] == "merge" for effect in packet.trace["memory_evolution"]["effects"])
    assert any(record.metadata["tim"].get("merged_from") for record in workflow.store.iter_records(TIM_THOUGHT_LAYER))


def test_tim_end_to_end_postthink_cycle_stores_generated_thoughts() -> None:
    workflow = TimWorkstream(top_k=3, readout_item_budget=3)

    packet = workflow.ingest_postthoughts(
        query="What beverage should we remember for Alice?",
        response="We should remember that Alice prefers tea.",
        thoughts=None,
        turn_id="turn-3",
    )

    assert packet.units is not None
    assert packet.units
    readout = workflow.recall_thoughts("Alice")
    assert readout.source_ids
