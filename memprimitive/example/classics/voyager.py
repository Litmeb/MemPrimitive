"""Voyager (Wang et al., 2023) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.voyager

Or from this directory (script adds the repo root to ``sys.path``)::

    python voyager.py
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
    GraphAppendOrganization,
    PassThroughUnitFormation,
    TagRetrieval,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="skill_library",
                theme="knowledge_graph",
                shape="Graph",
                indices=("graph", "entity", "tag", "vector"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "tags", "keywords", "description")),
        write_trigger=AlwaysWriteTrigger(),
        organization=GraphAppendOrganization(target_layer="skill_library"),
        retrieval=TagRetrieval(top_k=5, layer="skill_library"),
        readout=ConcatenateReadout(separator="\n\n"),
    )

    pipeline.ingest(Observation(text="Skill: craft_planks — turns logs into planks at a bench.", source="skill"))
    readout = pipeline.recall(Query(text="craft"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
