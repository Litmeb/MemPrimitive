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

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query
from memprimitive.baselines.unit_formation import PassThroughUnitFormation
from memprimitive.classic_modules.amem import (
    AMEMAgenticEvolution,
    AMEMAgenticEvolutionTrigger,
    AMEMAgenticOrganization,
    AMEMAgenticReadout,
    AMEMAgenticRepresentation,
    AMEMAgenticWriteTrigger,
    AMEMConfig,
    AMEMEnhancedRetrieval,
    build_amem_store,
)


def build_amem_pipeline(
    *,
    config: AMEMConfig | None = None,
    store: MemoryStore | None = None,
    graph_layer: str = "memory_graph",
    top_k: int = 5,
    max_links_per_record: int = 4,
) -> MemoryPipeline:
    if config is None:
        config = AMEMConfig(
            graph_layer=graph_layer,
            top_k=top_k,
            max_links_per_record=max_links_per_record,
        )
    if not config.strict_llm:
        raise ValueError("A-MEM classic implementation requires strict_llm=True.")

    memory_store = store if store is not None else build_amem_store(graph_layer=config.graph_layer)
    return MemoryPipeline(
        store=memory_store,
        unit_formation=PassThroughUnitFormation(),
        representation=AMEMAgenticRepresentation(strict_llm=config.strict_llm),
        write_trigger=AMEMAgenticWriteTrigger(enabled=config.write_decision_enabled),
        organization=AMEMAgenticOrganization(target_layer=config.graph_layer),
        evolution_trigger=AMEMAgenticEvolutionTrigger(target_layer=config.graph_layer, candidate_k=config.candidate_k),
        memory_evolution=AMEMAgenticEvolution(
            target_layer=config.graph_layer,
            candidate_k=config.candidate_k,
            max_links_per_record=config.max_links_per_record,
        ),
        retrieval=AMEMEnhancedRetrieval(
            target_layer=config.graph_layer,
            candidate_k=config.candidate_k,
            top_k=config.top_k,
            neighbor_expansion_k=config.neighbor_expansion_k,
            agentic_search=config.agentic_search,
            query_expand_with_llm=config.query_expand_with_llm,
        ),
        readout=AMEMAgenticReadout(),
    )


def main() -> None:
    pipeline = build_amem_pipeline(config=AMEMConfig(top_k=5))
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()


__all__ = ["build_amem_pipeline"]
