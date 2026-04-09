"""Session-summary pipeline using HierarchicalEvolution plus layer-aware recall.

This example writes raw turns into an ``episodic`` layer with one ingest
pipeline. A second session-close pipeline emits ``session_end`` without writing
another raw record, so ``HierarchicalEvolution`` can gather all records from the
finished session and generate one higher-level summary record in
``session_summary``.

Recall then uses ``LayerAwareRetrieval`` across both layers, with
``EmbeddingSimilarityRetrieval`` producing per-layer candidates and the layer-aware
merge keeping only the final global ``top_k``.

This script uses ``extract_mode="generate"`` for real session summaries, so the
LLM runtime must be configured, for example via:

    MEMPRIMITIVE_API_KEY
    MEMPRIMITIVE_BASE_URL
    MEMPRIMITIVE_MODEL

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.hierarchical_session_summary_pipeline

Or from this directory (script adds the repo root to ``sys.path``)::

    python hierarchical_session_summary_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    BoundaryEventTrigger,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    HierarchicalEvolution,
    LayerAwareRetrieval,
    NeverTrigger,
)

def ingest_session(
    turn_pipeline: MemoryPipeline,
    session_close_pipeline: MemoryPipeline,
    *,
    session_id: str,
    turns: list[str],
) -> None:
    for index, text in enumerate(turns):
        turn_pipeline.ingest(
            Observation(
                text=text,
                source="dialogue",
                metadata={
                    "session_id": session_id,
                    "turn_id": f"{session_id}-turn-{index + 1}",
                },
            )
        )
    session_close_pipeline.ingest(
        Observation(
            text=f"close session {session_id}",
            source="session_controller",
            metadata={
                "session_id": session_id,
                "trigger": {"events": ["session_end"]},
            },
        )
    )


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="episodic", theme="session_memory", indices=("temporal", "vector")),
            StoreLayerSpec(name="session_summary", theme="semantic", indices=("temporal", "vector")),
        ]
    )
    store = MemoryStore(topology=topology)

    turn_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "embedding")),
        organization=AppendOrganization(target_layer="episodic"),
        store=store,
    )
    session_close_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=AppendOrganization(target_layer="episodic"),
        evolution_trigger=BoundaryEventTrigger(
            slot="evolution_trigger",
            accepted_events=("session_end",),
        ),
        memory_evolution=HierarchicalEvolution(
            source_layer="episodic",
            extract_mode="generate",
            extract_fields=("summary",),
            group_by=("session_id",),
            target_layer="session_summary",
        ),
        store=store,
    )
    recall_pipeline = MemoryPipeline(
        retrieval=LayerAwareRetrieval(
            default_retriever=EmbeddingSimilarityRetrieval(top_k=4),
            retriever_by_layer={"session_summary": EmbeddingSimilarityRetrieval(top_k=4)},
            active_layers=("session_summary", "episodic"),
            top_k=4,
            top_k_by_layer={"session_summary": 4, "episodic": 4},
        ),
        readout=ConcatenateReadout(separator="\n\n"),
        store=store,
    )

    ingest_session(
        turn_pipeline,
        session_close_pipeline,
        session_id="sess-1",
        turns=[
            "Alice is debugging hierarchical graph retrieval for long conversations.",
            "She wants one session summary plus detail recall for graph-memory work.",
        ],
    )
    ingest_session(
        turn_pipeline,
        session_close_pipeline,
        session_id="sess-2",
        turns=[
            "Bob is planning a weekend trip to Hangzhou.",
            "He compares train times and hotel options before the session ends.",
        ],
    )

    query = Query(text="What happened in the graph retrieval session?")
    readout = recall_pipeline.recall(query)
    retrieval_packet, _ = recall_pipeline.retrieval.run(Packet(query=query), recall_pipeline.store)

    print("records per layer:")
    pprint({name: recall_pipeline.store.count(name) for name in recall_pipeline.store.topology.layer_names})
    print()

    print("session summaries:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "group_key": record.metadata.get("hierarchical", {}).get("group_key"),
                "source_record_ids": record.metadata.get("hierarchical", {}).get("source_record_ids"),
            }
            for record in recall_pipeline.store.iter_records("session_summary")
        ]
    )
    print()

    print("layer-aware retrieval trace:")
    pprint(retrieval_packet.trace["retrieval"])
    print()

    print("readout:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
