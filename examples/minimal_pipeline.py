"""Tiny end-to-end example for the stage-1 MemPrimitive pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memprimitive import Observation, Query, create_baseline_pipeline


def main() -> None:
    pipeline = create_baseline_pipeline(top_k=2)

    pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob prefers black coffee in the morning.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice started learning graph-based memory systems.", source="notes"))

    readout = pipeline.recall(Query(text="What do we know about Alice?"))

    print("Readout:")
    print(readout.text)
    print("Source IDs:")
    print(", ".join(readout.source_ids))


if __name__ == "__main__":
    main()
