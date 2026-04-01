"""End-to-end graph baseline family example with ingest, links, neighbor recall, and readout.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.graph_baseline_pipeline

Or from this directory (script adds the repo root to ``sys.path``)::

    python graph_baseline_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    BasicRepresentation,
    GraphAppendOrganization,
    GraphNeighborAppendEvolution,
    GraphReadout,
    GraphSeedAndExpandRetrieval,
    ThresholdTrigger,
    TripleRepresentation,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default", theme="working"),
            StoreLayerSpec(
                name="knowledge_graph",
                theme="knowledge_graph",
                shape="Graph",
                indices=("graph", "entity"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    graph_pipeline = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text",)),
            TripleRepresentation(method="direct"),
            BasicRepresentation(elements=("tags", "keywords")),
        ),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0),
        memory_evolution=GraphNeighborAppendEvolution(target_layer="knowledge_graph", neighbor_limit=2),
        store=store,
    )

    packet_a = graph_pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
    packet_b = graph_pipeline.ingest(Observation(text="Alice studies graph memory systems.", source="notes"))
    packet_c = graph_pipeline.ingest(Observation(text="Bob builds graph retrieval tools.", source="notes"))

    recall_pipeline = MemoryPipeline(
        retrieval=GraphSeedAndExpandRetrieval(top_k=4, layer="knowledge_graph", seed_top_k=1),
        readout=GraphReadout(),
        store=store,
    )
    query = Query(text="Alice graph")
    readout = recall_pipeline.recall(query)
    retrieval_packet, _ = recall_pipeline.retrieval.run(Packet(query=query), store)

    print("organization traces:")
    pprint(
        [
            packet_a.trace["organization"],
            packet_b.trace["organization"],
            packet_c.trace["organization"],
        ]
    )
    print()

    print("memory evolution traces:")
    pprint(
        [
            packet_a.trace["memory_evolution"],
            packet_b.trace["memory_evolution"],
            packet_c.trace["memory_evolution"],
        ]
    )
    print()

    print("graph layer records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "graph": record.metadata.get("graph"),
            }
            for record in store.iter_records("knowledge_graph")
        ]
    )
    print()

    print("retrieval trace:")
    pprint(retrieval_packet.trace["retrieval"])
    print()

    print("graph readout:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
