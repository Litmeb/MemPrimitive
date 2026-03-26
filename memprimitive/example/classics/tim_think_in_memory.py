"""TiM — Think-in-Memory (Liu et al., 2023) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.tim_think_in_memory

Or from this directory (script adds the repo root to ``sys.path``)::

    python tim_think_in_memory.py
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
    KeywordCountRetrieval,
    PassThroughUnitFormation,
    SummaryRewriteEvolution,
    ThresholdEvolutionTrigger,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="thought_memory", theme="working", capacity="token_limited", indices=("temporal", "vector")),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "summary")),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(target_layer="thought_memory"),
        evolution_trigger=ThresholdEvolutionTrigger(threshold=0.5),
        memory_evolution=SummaryRewriteEvolution(target_layer="thought_memory"),
        retrieval=KeywordCountRetrieval(top_k=5, layer="thought_memory"),
        readout=ConcatenateReadout(separator="\n"),
    )

    pipeline.ingest(Observation(text="Step: decompose the query into subgoals.", source="reasoning"))
    readout = pipeline.recall(Query(text="subgoals"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
