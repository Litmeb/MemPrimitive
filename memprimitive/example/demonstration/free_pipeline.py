"""Minimal example for the free-form ordered pipeline runner.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.free_pipeline

Or from this directory (script adds the repo root to ``sys.path``)::

    python free_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as ``python memprimitive/example/demonstration/free_pipeline.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import FreeMemoryPipeline, Observation, Query
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)


def main() -> None:
    pipeline = FreeMemoryPipeline(
        modules=(
            PassThroughUnitFormation(),
            BasicRepresentation(),
            AlwaysTrigger(),
            AppendOrganization(),
            RecencyRetrieval(top_k=2),
            ConcatenateReadout(),
        )
    )

    pipeline.ingest(Observation(text="The user likes free-form pipelines.", source="notes"))
    pipeline.ingest(Observation(text="The user also wants ordered module execution.", source="notes"))

    readout = pipeline.recall(Query(text="What does the user want?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
