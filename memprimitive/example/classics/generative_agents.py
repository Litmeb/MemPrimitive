"""Generative Agents (Park et al., 2023) motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.generative_agents

Or from this directory (script adds the repo root to ``sys.path``)::

    python generative_agents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import Observation, Query
from memprimitive.classic_modules.generative_agents import build_generative_agents_pipeline


def main() -> None:
    pipeline = build_generative_agents_pipeline(
        top_k=5,
        reflection_threshold=0.75,
        reflection_batch_size=2,
    )

    pipeline.ingest(Observation(text="Alice prefers tea when writing code.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice wants concise notes for review meetings.", source="dialogue"))
    pipeline.ingest(Observation(text="The desk lamp is blue.", source="notes"))

    readout = pipeline.recall(Query(text="What does Alice care about?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
