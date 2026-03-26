"""MemGPT (Packer et al., 2023) - motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.memgpt

Or from this directory (script adds the repo root to ``sys.path``)::

    python memgpt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import Query
from memprimitive.classic_modules.memgpt import (
    MEMGPT_ARCHIVAL_LAYER,
    MEMGPT_MAIN_LAYER,
    MEMGPT_RECALL_LAYER,
    build_memgpt_pipeline,
    memgpt_observation,
)


def main() -> None:
    pipeline = build_memgpt_pipeline(top_k=3, main_context_budget=1, recall_budget=1, readout_item_budget=3)

    pipeline.ingest(memgpt_observation("Pinned note: review memory architecture docs.", source="dialogue"))
    pipeline.ingest(
        memgpt_observation(
            "Archive this note about the release checklist.",
            source="dialogue",
            target_layer=MEMGPT_ARCHIVAL_LAYER,
            tool="memory_save",
        )
    )
    pipeline.ingest(memgpt_observation("The user prefers concise status updates.", source="dialogue"))

    readout = pipeline.recall(Query(text="What should we remember?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)
    print(
        "store counts:",
        {
            MEMGPT_MAIN_LAYER: pipeline.store.count(MEMGPT_MAIN_LAYER),
            MEMGPT_ARCHIVAL_LAYER: pipeline.store.count(MEMGPT_ARCHIVAL_LAYER),
            MEMGPT_RECALL_LAYER: pipeline.store.count(MEMGPT_RECALL_LAYER),
        },
    )


if __name__ == "__main__":
    main()
