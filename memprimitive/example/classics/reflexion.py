"""Reflexion (Shinn et al., 2023) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.reflexion

Or from this directory (script adds the repo root to ``sys.path``)::

    python reflexion.py
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
    PassThroughUnitFormation,
    RecencyRetrieval,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="reflections",
                theme="working",
                capacity="sliding_window",
                indices=("temporal",),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "summary")),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(target_layer="reflections"),
        retrieval=RecencyRetrieval(top_k=100, layer="reflections"),
        readout=ConcatenateReadout(separator="\n"),
    )

    pipeline.ingest(Observation(text="Task failed: missing edge case in parser.", source="failure_log"))
    readout = pipeline.recall(Query(text="Recent reflections"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
