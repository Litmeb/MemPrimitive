"""Minimal end-to-end example: wire primitives by hand, write, then read.

From the repo root (recommended)::

    python -m memprimitive.example

Or from this directory (script adds the repo root to ``sys.path``)::

    python example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as ``python memprimitive/example.py`` leaves ``__package__`` unset; repo root must be on path.
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
    PassThroughUnitFormation,
    RecencyRetrieval,
)


def main() -> None:
    # One instance per primitive slot, in pipeline order (ingest → recall).
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(),
        evolution_trigger=AlwaysEvolutionTrigger(),
        memory_evolution=AppendOnlyEvolution(),
        retrieval=RecencyRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    # Write path: each observation flows through ingest and appends to the shared store.
    pipeline.ingest(Observation(text="The user likes concise examples.", source="dialogue"))
    pipeline.ingest(Observation(text="The user works on compositional memory.", source="notes"))

    # Read path: query → retrieval → readout for the agent.
    readout = pipeline.recall(Query(text="What does the user like?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
