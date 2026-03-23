"""Canonical primitive slot names and order for MemoryPipeline (stage 1).

Single source of truth for ingest/recall sequencing and for combinatorial
baseline tests. Constructor keyword names on MemoryPipeline match these strings.
"""

from __future__ import annotations

from typing import Final

INGEST_SLOTS: Final[tuple[str, ...]] = (
    "unit_formation",
    "representation",
    "write_trigger",
    "organization",
    "evolution_trigger",
    "memory_evolution",
)

RECALL_SLOTS: Final[tuple[str, ...]] = ("retrieval", "readout")

# Full pipeline: ingest then recall (same order as MemoryPipeline parameters conceptually)
ALL_PIPELINE_SLOTS: Final[tuple[str, ...]] = INGEST_SLOTS + RECALL_SLOTS

# Ingest stages that run before memory_evolution (for tests that stop before append).
PRE_EVOLUTION_SLOTS: Final[tuple[str, ...]] = tuple(
    s for s in INGEST_SLOTS if s != "memory_evolution"
)
