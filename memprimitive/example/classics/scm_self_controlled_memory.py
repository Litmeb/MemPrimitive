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

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import AppendOnlyEvolution, BasicRepresentation, ConcatenateReadout, NeverEvolutionTrigger
from memprimitive.classic_modules.scm import (
    SCMControlledRetrieval,
    SCMEntityProfileUpsert,
    SCMJudgeGateWrite,
    SCMStructuredExtraction,
)


def build_scm_pipeline(
    *,
    top_k: int = 3,
    threshold: float = 0.55,
    semantic_layer: str = "semantic",
    profile_layer: str = "profile",
    store: MemoryStore | None = None,
) -> MemoryPipeline:
    if store is None:
        store = MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(name=semantic_layer, theme="semantic", indices=("entity", "keyword", "vector")),
                    StoreLayerSpec(name=profile_layer, theme="profile", indices=("entity", "keyword")),
                ]
            )
        )

    return MemoryPipeline(
        store=store,
        unit_formation=SCMStructuredExtraction(),
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=SCMJudgeGateWrite(threshold=threshold),
        organization=SCMEntityProfileUpsert(semantic_layer=semantic_layer, profile_layer=profile_layer),
        evolution_trigger=NeverEvolutionTrigger(),
        memory_evolution=AppendOnlyEvolution(),
        retrieval=SCMControlledRetrieval(top_k=top_k, semantic_layer=semantic_layer, profile_layer=profile_layer),
        readout=ConcatenateReadout(separator="\n\n"),
    )


def main() -> None:
    pipeline = build_scm_pipeline(top_k=5, threshold=0.5)

    pipeline.ingest(Observation(text="(Alice, works_at, ACME)", source="structured"))
    pipeline.ingest(Observation(text="Alice likes tea and works on retrieval tools.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()


__all__ = ["build_scm_pipeline"]
