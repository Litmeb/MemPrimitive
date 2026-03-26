"""TiM - Think-in-Memory (Liu et al., 2023) - motif sketch.

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

from memprimitive import Observation, Query
from memprimitive.classic_modules.tim import TimWorkstream


def main() -> None:
    workflow = TimWorkstream(budget=2, top_k=3, readout_item_budget=3)

    workflow.ingest(
        Observation(
            text="1. Decompose the query into subgoals.\n2. Check the available context.\n3. Draft the next step.",
            source="reasoning",
            metadata={"tim": {"reasoning_step": True}},
        )
    )
    workflow.ingest(
        Observation(
            text="1. Re-rank the candidate memories.\n2. Summarize the useful thoughts.",
            source="reasoning",
            metadata={"tim": {"reasoning_step": True}},
        )
    )

    readout = workflow.recall(Query(text="subgoals"))

    print(readout.text)
    print("source record ids:", readout.source_ids)
    print("store count:", workflow.store.count("thought_memory"))


if __name__ == "__main__":
    main()
