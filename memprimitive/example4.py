"""Minimal end-to-end example using embedding-based representation and retrieval.

From the repo root (recommended)::

    python -m memprimitive.example4

Or from this directory (script adds the repo root to ``sys.path``)::

    python example4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as ``python memprimitive/example4.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memprimitive import MemoryPipeline, Observation, Query
from memprimitive.baselines import (
    AlwaysEvolutionTrigger,
    AlwaysWriteTrigger,
    AppendOnlyEvolution,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    PassThroughUnitFormation,
)


def main() -> None:
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding")),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(),
        evolution_trigger=AlwaysEvolutionTrigger(),
        memory_evolution=AppendOnlyEvolution(),
        retrieval=EmbeddingSimilarityRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob prefers black coffee in the morning.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice started learning graph-based memory systems.", source="notes"))

    readout = pipeline.recall(Query(text="What do we know about Alice's interests?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
