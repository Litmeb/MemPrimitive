"""Minimal graph-dependent baseline pipeline for trigger -> evolution -> recall.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.graph_dependent_pipeline

Or from this directory (script adds the repo root to ``sys.path``)::

    python graph_dependent_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    BasicRepresentation,
    GraphAppendOrganization,
    GraphLinkEvolution,
    GraphNeighborContextTraceEvolution,
    GraphReadout,
    GraphSeedAndExpandRetrieval,
    NeighborExistsEvolutionTrigger,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default", theme="working"),
            StoreLayerSpec(
                name="knowledge_graph",
                theme="knowledge_graph",
                shape="Graph",
                indices=("graph", "entity", "vector"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "triple", "tags", "keywords")),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=2),
        memory_evolution=(
            GraphLinkEvolution(target_layer="knowledge_graph", neighbor_limit=2, rewrite_neighbor_metadata=True),
            GraphNeighborContextTraceEvolution(target_layer="knowledge_graph", rewrite_metadata=True),
        ),
        retrieval=GraphSeedAndExpandRetrieval(top_k=4, layer="knowledge_graph", seed_top_k=1),
        readout=GraphReadout(),
        store=store,
    )

    first = pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
    second = pipeline.ingest(Observation(text="Alice studies graph memory systems.", source="notes"))
    third = pipeline.ingest(Observation(text="Bob builds retrieval tools.", source="notes"))
    readout = pipeline.recall(Query(text="Alice graph"))

    print("evolution trigger traces:")
    pprint([first.trace["evolution_trigger"], second.trace["evolution_trigger"], third.trace["evolution_trigger"]])
    print()

    print("memory evolution traces:")
    pprint([first.trace["memory_evolution"], second.trace["memory_evolution"], third.trace["memory_evolution"]])
    print()

    print("graph layer records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "graph": record.metadata.get("graph"),
            }
            for record in pipeline.store.iter_records("knowledge_graph")
        ]
    )
    print()

    print("graph readout:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
