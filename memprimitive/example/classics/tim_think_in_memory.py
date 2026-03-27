"""TiM - Think-in-Memory (Liu et al., 2023) - paper-style loop sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.tim_think_in_memory

Or from this directory (script adds the repo root to ``sys.path``)::

    python tim_think_in_memory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, Readout, StoreLayerSpec, StoreTopology
from memprimitive.classic_modules._runtime import get_classic_runtime
from memprimitive.classic_modules.tim import (
    TIM_THOUGHT_LAYER,
    TimBudgetEvolutionTrigger,
    TimThoughtMemoryEvolution,
    TimThoughtMemoryOrganization,
    TimThoughtMemoryRetrieval,
    TimThoughtReadout,
    TimThoughtRepresentation,
    TimThoughtUnitFormation,
    TimThoughtWriteTrigger,
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _postthought_prompt_payload(*, query: str, response: str, recalled_thoughts: list[str], turn_id: str) -> str:
    payload = {
        "query": query,
        "response": response,
        "recalled_thoughts": recalled_thoughts,
        "source_turn_id": turn_id,
    }
    return json.dumps(payload, ensure_ascii=False)


def tim_postthought_observation(
    *,
    query: str,
    response: str,
    thoughts: list[dict[str, Any]] | None,
    turn_id: str,
    source: str = "post_think",
) -> Observation:
    normalized_thoughts = [] if thoughts is None else list(thoughts)
    preview = normalized_thoughts[0]["thought_text"] if normalized_thoughts else "TiM post-thoughts"
    return Observation(
        text=_normalize_text(preview) or "TiM post-thoughts",
        source=source,
        metadata={
            "tim": {
                "source_query": _normalize_text(query),
                "source_response": _normalize_text(response),
                "source_turn_id": _normalize_text(turn_id),
                "thoughts": normalized_thoughts,
            }
        },
    )


def generate_tim_postthoughts(
    *,
    query: str,
    response: str,
    recalled_thoughts: list[str] | None = None,
    turn_id: str,
) -> list[dict[str, Any]]:
    runtime = get_classic_runtime()
    result = runtime.json(
        system=(
            "TiM post-think extractor. "
            "Given a user query, the agent response, and recalled historical thoughts, "
            "return JSON with key 'thoughts' containing a list of inductive thoughts. "
            "Each thought must include: thought_text, head_entity, relation, tail_entity."
        ),
        user=_postthought_prompt_payload(
            query=query,
            response=response,
            recalled_thoughts=[] if recalled_thoughts is None else list(recalled_thoughts),
            turn_id=turn_id,
        ),
    )
    if not isinstance(result, dict):
        return []
    raw_thoughts = result.get("thoughts", [])
    if not isinstance(raw_thoughts, list):
        return []

    parsed: list[dict[str, Any]] = []
    for item in raw_thoughts:
        if not isinstance(item, dict):
            continue
        thought_text = _normalize_text(item.get("thought_text") or item.get("text") or item.get("thought") or "")
        head = _normalize_text(item.get("head_entity", ""))
        relation = _normalize_text(item.get("relation", ""))
        tail = _normalize_text(item.get("tail_entity", ""))
        triple = item.get("triple")
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            triple_value = [str(triple[0]), str(triple[1]), str(triple[2])]
        else:
            triple_value = [head, relation, tail]
        if thought_text:
            parsed.append(
                {
                    "thought_text": thought_text,
                    "head_entity": head,
                    "relation": relation,
                    "tail_entity": tail,
                    "triple": triple_value,
                    "source_query": _normalize_text(query),
                    "source_response": _normalize_text(response),
                    "source_turn_id": _normalize_text(turn_id),
                    "write": item.get("write", True),
                }
            )
    return parsed


def build_tim_pipeline(
    *,
    store: MemoryStore | None = None,
    thought_layer: str = TIM_THOUGHT_LAYER,
    budget: int = 4,
    top_k: int = 5,
    readout_item_budget: int = 4,
) -> MemoryPipeline:
    if store is None:
        store = MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(
                        name=thought_layer,
                        theme="working",
                        capacity="token_limited",
                        indices=("temporal", "vector", "keyword"),
                    ),
                ]
            )
        )
    elif not store.has_layer(thought_layer):
        store.ensure_layer(thought_layer, allow_create=True, theme="working")

    return MemoryPipeline(
        store=store,
        unit_formation=TimThoughtUnitFormation(),
        representation=TimThoughtRepresentation(),
        write_trigger=TimThoughtWriteTrigger(),
        organization=TimThoughtMemoryOrganization(target_layer=thought_layer),
        evolution_trigger=TimBudgetEvolutionTrigger(thought_layer=thought_layer, budget=budget),
        memory_evolution=TimThoughtMemoryEvolution(thought_layer=thought_layer, budget=budget),
        retrieval=TimThoughtMemoryRetrieval(top_k=top_k, thought_layer=thought_layer),
        readout=TimThoughtReadout(item_budget=readout_item_budget),
    )


class TimWorkstream:
    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        thought_layer: str = TIM_THOUGHT_LAYER,
        budget: int = 4,
        top_k: int = 5,
        readout_item_budget: int = 4,
    ) -> None:
        self.pipeline = build_tim_pipeline(
            store=store,
            thought_layer=thought_layer,
            budget=budget,
            top_k=top_k,
            readout_item_budget=readout_item_budget,
        )

    @property
    def store(self) -> MemoryStore:
        return self.pipeline.store

    def ingest(self, observation: Observation):
        return self.pipeline.ingest(observation)

    def recall(self, query: Query) -> Readout:
        return self.pipeline.recall(query)

    def recall_thoughts(self, query: str | Query) -> Readout:
        if isinstance(query, Query):
            return self.recall(query)
        return self.recall(Query(text=str(query)))

    def ingest_postthoughts(
        self,
        *,
        query: str,
        response: str,
        thoughts: list[dict[str, Any]] | None = None,
        turn_id: str = "turn-1",
    ):
        generated = thoughts if thoughts is not None else generate_tim_postthoughts(
            query=query,
            response=response,
            recalled_thoughts=[],
            turn_id=turn_id,
        )
        observation = tim_postthought_observation(
            query=query,
            response=response,
            thoughts=generated,
            turn_id=turn_id,
        )
        return self.ingest(observation)


def main() -> None:
    runtime = get_classic_runtime()
    workflow = TimWorkstream(budget=4, top_k=3, readout_item_budget=3)

    for turn_id, user_query in enumerate(
        [
            "What beverage should we remember for Alice?",
            "What workplace fact should we remember for Alice?",
        ],
        start=1,
    ):
        recalled = workflow.recall_thoughts(Query(text=user_query))
        response = runtime.text(
            system="Answer the user query using the recalled TiM thought context when it is useful.",
            user=(
                f"query: {user_query}\n"
                f"recalled_thoughts:\n{recalled.text or '(none)'}\n"
                "Respond briefly and factually."
            ),
        )
        packet = workflow.ingest_postthoughts(
            query=user_query,
            response=response,
            turn_id=f"turn-{turn_id}",
        )
        print(f"[turn {turn_id}] query:", user_query)
        print(f"[turn {turn_id}] response:", response)
        print(f"[turn {turn_id}] stored_thoughts:", len(packet.units or []))
        print()

    final_readout = workflow.recall_thoughts("Alice")
    print("Recalled thoughts:")
    print(final_readout.text)
    print("source record ids:", final_readout.source_ids)
    print("store count:", workflow.store.count("thought_memory"))


if __name__ == "__main__":
    main()


__all__ = [
    "TimWorkstream",
    "build_tim_pipeline",
    "generate_tim_postthoughts",
    "tim_postthought_observation",
]
