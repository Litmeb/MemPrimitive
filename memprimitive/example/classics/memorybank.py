"""MemoryBank (Zhong et al., 2024) - motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.memorybank

Or from this directory (script adds the repo root to ``sys.path``)::

    python memorybank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import Observation, Query
from memprimitive.classic_modules.memorybank import MemoryBankConfig, build_memorybank_pipeline


def main() -> None:
    pipeline = build_memorybank_pipeline(
        config=MemoryBankConfig(short_term_window=2),
    )

    pipeline.ingest(Observation(text="Alice works at OpenAI in San Francisco.", source="dialogue"))
    pipeline.ingest(Observation(text="remember to refill the tea kettle", source="note"))
    pipeline.ingest(Observation(text="capture the next observation for later", source="note"))

    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
