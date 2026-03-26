"""SCM - Self-Controlled Memory (Wang et al., 2024) - motif sketch.

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

from memprimitive import Observation, Query
from memprimitive.classic_modules.scm import build_scm_pipeline


def main() -> None:
    pipeline = build_scm_pipeline(top_k=5, threshold=0.5)

    pipeline.ingest(Observation(text="(Alice, works_at, ACME)", source="structured"))
    pipeline.ingest(Observation(text="Alice likes tea and works on retrieval tools.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
