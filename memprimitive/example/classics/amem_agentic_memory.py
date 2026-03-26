"""A-MEM — Agentic Memory (Xu et al., 2025) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.amem_agentic_memory

Or from this directory (script adds the repo root to ``sys.path``)::

    python amem_agentic_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    GraphAppendOrganization,
    PassThroughUnitFormation,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="memory_graph",
                theme="knowledge_graph",
                shape="Graph",
                indices=("graph", "entity", "vector", "tag"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "tags", "entities", "triple")),
        write_trigger=AlwaysWriteTrigger(),
        organization=GraphAppendOrganization(target_layer="memory_graph"),
        retrieval=EmbeddingSimilarityRetrieval(top_k=10, layer="memory_graph"),
        readout=ConcatenateReadout(separator="\n"),
    )

    pipeline.ingest(Observation(text="The agent prefers graph-structured episodic notes.", source="dialogue"))
    readout = pipeline.recall(Query(text="What structure does the agent prefer?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
