"""Generative Agents (Park et al., 2023) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.generative_agents

Or from this directory (script adds the repo root to ``sys.path``)::

    python generative_agents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    PassThroughUnitFormation,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="observation_stream", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(name="reflections", theme="semantic", indices=("temporal", "keyword", "vector")),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(
            elements=("text", "embedding", "tags", "keywords", "summary", "entities", "triple"),
        ),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(target_layer="observation_stream"),
        retrieval=EmbeddingSimilarityRetrieval(top_k=50, layer="observation_stream"),
        readout=ConcatenateReadout(separator="\n"),
    )

    pipeline.ingest(Observation(text="The user prefers concise technical writing.", source="dialogue"))
    readout = pipeline.recall(Query(text="What does the user prefer?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
