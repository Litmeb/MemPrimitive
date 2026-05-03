"""Compact MemMachine-style memory-layer reconstruction.

This example covers the memory layer only: working memory, LTM episodic
sentence indexing, contextualized episodic recall, and structured profile
features. It intentionally leaves the optional Retrieval Agent router outside
the classics example.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    AlwaysTrigger,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    EpisodeClusterRerankRetrieval,
    LLMFunctionCallEvolution,
    ParentEpisodeExpansionRetrieval,
    RecencyRetrieval,
    STMConsolidationEvolution,
    TemporalNeighborExpansionRetrieval,
)
from memprimitive.utils._profile_feature_tools import build_profile_feature_tools
from memprimitive.utils._template import structured_prompt


def build_memmachine_memory_system(
    *,
    stm_record_budget: int = 20,
    sentence_top_k: int = 30,
    episode_top_k: int = 30,
    profile_top_k: int = 10,
) -> dict[str, object]:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working", theme="working", indices=("temporal",)),
                StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
                StoreLayerSpec(
                    name="sentence",
                    theme="episodic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
                StoreLayerSpec(name="session_summary", theme="semantic", indices=("temporal",)),
                StoreLayerSpec(
                    name="profile",
                    theme="semantic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
            ]
        )
    )

    write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer="working"),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=(
            LLMFunctionCallEvolution(
                target_layer="profile",
                tools=build_profile_feature_tools(module_name="memmachine_profile"),
                prompt=_profile_prompt(),
            ),
            STMConsolidationEvolution(record_budget=stm_record_budget),
        ),
        store=store,
    )
    episodic_recall_pipeline = MemoryPipeline(
        retrieval=(
            EmbeddingSimilarityRetrieval(top_k=sentence_top_k, layer="sentence"),
            ParentEpisodeExpansionRetrieval(top_k=sentence_top_k, episode_layer="episodic"),
            TemporalNeighborExpansionRetrieval(layer="episodic", backward=1, forward=2),
            EpisodeClusterRerankRetrieval(top_k=episode_top_k),
        ),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )
    return {
        "store": store,
        "write_pipeline": write_pipeline,
        "working_recall_pipeline": MemoryPipeline(
            retrieval=RecencyRetrieval(top_k=stm_record_budget, layer="working"),
            readout=ConcatenateReadout(separator="\n"),
            store=store,
        ),
        "summary_recall_pipeline": MemoryPipeline(
            retrieval=RecencyRetrieval(top_k=1, layer="session_summary"),
            readout=ConcatenateReadout(separator="\n"),
            store=store,
        ),
        "episodic_recall_pipeline": episodic_recall_pipeline,
        "profile_recall_pipeline": MemoryPipeline(
            retrieval=EmbeddingSimilarityRetrieval(top_k=profile_top_k, layer="profile"),
            readout=ConcatenateReadout(separator="\n"),
            store=store,
        ),
    }


def ingest_episode(
    system: dict[str, object],
    *,
    text: str,
    session_id: str,
    user_id: str,
    agent_id: str = "agent",
    producer: str = "user",
    timestamp: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline, MemoryPipeline)
    observation_kwargs: dict[str, Any] = {}
    if timestamp is not None:
        observation_kwargs["timestamp"] = timestamp
    return write_pipeline.ingest(
        Observation(
            text=text,
            source=producer,
            metadata={
                **dict(metadata or {}),
                "producer": producer,
                "session_id": session_id,
                "user_id": user_id,
                "agent_id": agent_id,
            },
            **observation_kwargs,
        )
    )


def recall_memmachine_context(system: dict[str, object], *, user_query: str) -> str:
    query = Query(text=user_query)
    sections = [
        ("STM Summary", system["summary_recall_pipeline"].recall(query).text),
        ("Working Memory", system["working_recall_pipeline"].recall(query).text),
        ("Long-Term Episodes", system["episodic_recall_pipeline"].recall(query).text),
        ("Profile Memory", system["profile_recall_pipeline"].recall(query).text),
    ]
    return "\n\n".join(f"## {title}\n{text}" for title, text in sections if text.strip())


def _profile_prompt():
    return structured_prompt(
        {
            "blocks": [
                {
                    "id": "task",
                    "title": "Task",
                    "template": (
                        "Extract stable user profile features from the selected raw episode records.\n"
                        "Use UPSERT_PROFILE_FEATURE for durable preferences, facts, roles, and habits. "
                        "Use DELETE_PROFILE_FEATURE only for explicit contradictions.\n"
                        "Default target_layer={{ default_target_layer }}. "
                        "Preserve source episode ids in tool arguments."
                    ),
                },
                {
                    "id": "selected_records",
                    "title": "Selected Raw Episodes",
                    "repeat_over": "selected_records",
                    "item_template": (
                        "record_id={{ item.record_id }}\n"
                        "timestamp={{ item.timestamp }}\n"
                        "text={{ item.text }}\n"
                        "metadata={{ item.metadata }}"
                    ),
                    "separator": "\n\n",
                },
                {
                    "id": "tools",
                    "title": "Available Tools",
                    "repeat_over": "tools",
                    "item_template": "- {{ item.name }}",
                    "separator": "\n",
                },
            ]
        }
    )


def main() -> None:
    system = build_memmachine_memory_system(stm_record_budget=2)
    for index, text in enumerate(
        [
            "Alice is comparing memory systems for long-running agents.",
            "Alice says she prefers jasmine tea during late-night coding.",
            "The assistant suggests keeping raw episodes available for audit.",
        ],
        start=1,
    ):
        ingest_episode(
            system,
            text=text,
            session_id="sess-demo",
            user_id="alice",
            timestamp=f"2026-04-28T00:0{index}:00Z",
        )

    store = system["store"]
    assert isinstance(store, MemoryStore)
    pprint({layer: store.count(layer) for layer in store.topology.layer_names})
    print(recall_memmachine_context(system, user_query="What tea does Alice prefer?"))


if __name__ == "__main__":
    main()
