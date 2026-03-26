"""A-MEM - Agentic Memory (Xu et al., 2025) - graph-memory sketch.

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

from memprimitive import Observation, Query
from memprimitive.classic_modules.amem import AMEMConfig, build_amem_pipeline


def main() -> None:
    pipeline = build_amem_pipeline(config=AMEMConfig(top_k=5, max_hops=2))
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
