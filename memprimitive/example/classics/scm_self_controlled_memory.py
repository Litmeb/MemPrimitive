"""SCM — Self-Controlled Memory (Wang et al., 2024) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.scm_self_controlled_memory

Or from this directory (script adds the repo root to ``sys.path``)::

    python scm_self_controlled_memory.py
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
    EntityRetrieval,
    PassThroughUnitFormation,
    ThresholdWriteTrigger,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="short_term", theme="working", indices=("temporal",)),
            StoreLayerSpec(name="long_term", theme="semantic", indices=("entity", "vector")),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "triple", "entities", "embedding", "kv")),
        write_trigger=ThresholdWriteTrigger(threshold=0.6),
        organization=AppendOrganization(target_layer="long_term"),
        retrieval=EntityRetrieval(top_k=10, layer="long_term"),
        readout=ConcatenateReadout(separator="\n"),
    )

    pipeline.ingest(Observation(text="(Alice, works_at, ACME)", source="structured"))
    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
