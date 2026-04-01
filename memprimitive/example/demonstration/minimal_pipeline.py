"""Minimal end-to-end example: wire primitives by hand, write, then read.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.minimal_pipeline

Or from this directory (script adds the repo root to ``sys.path``)::

    python minimal_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as ``python memprimitive/example/demonstration/minimal_pipeline.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, Observation, Query
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)


def main() -> None:
    # One instance per primitive slot, in pipeline order (ingest 鈫?recall).
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(),
        retrieval=RecencyRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    # Write path: each observation flows through ingest and appends to the shared store.
    pipeline.ingest(Observation(text="The user likes concise examples.", source="dialogue"))
    pipeline.ingest(Observation(text="The user works on compositional memory.", source="notes"))

    # Read path: query 鈫?retrieval 鈫?readout for the agent.
    readout = pipeline.recall(Query(text="What does the user like?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
