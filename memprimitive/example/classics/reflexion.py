"""Reflexion (Shinn et al., 2023) - motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.reflexion

Or from this directory (script adds the repo root to ``sys.path``)::

    python reflexion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import Observation, Query
from memprimitive.classic_modules.reflexion import ReflexionWorkstream


def main() -> None:
    workflow = ReflexionWorkstream(reflection_window=2, reflection_top_k=2)

    workflow.ingest(
        Observation(
            text="Task failed: missing edge case in parser.",
            source="failure_log",
            metadata={
                "reflexion": {
                    "event": "failure",
                    "task": "Parse the input stream",
                    "feedback": "missing edge case in parser",
                }
            },
        )
    )
    workflow.ingest(
        Observation(
            text="Task solved cleanly on the second attempt.",
            source="dialogue",
            metadata={"reflexion": {"event": "success", "task": "Parse the input stream"}},
        )
    )
    readout = workflow.recall(Query(text="Parse the input stream"))

    print(readout.text)
    print("source record ids:", readout.source_ids)
    print("reflection count:", workflow.store.count("reflections"))


if __name__ == "__main__":
    main()
