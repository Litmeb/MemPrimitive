"""End-to-end example showing graph-layer storage with ``GraphAppendOrganization``.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.graph_append_entity_retrieval

Or from this directory (script adds the repo root to ``sys.path``)::

    python graph_append_entity_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/demonstration/graph_append_entity_retrieval.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    BasicRepresentation,
    ConcatenateReadout,
    EntityRetrieval,
    GraphAppendOrganization,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(
                name="knowledge_graph",
                theme="knowledge_graph",
                shape="Graph",
                indices=("graph", "entity"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    graph_writer = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "entities", "triple", "tags")),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        store=store,
    )

    graph_writer.ingest(Observation(text="Alice likes jasmine tea.", source="dialogue"))
    graph_writer.ingest(Observation(text="Bob works on memory retrieval systems.", source="notes"))
    packet = graph_writer.ingest(Observation(text="Alice studies graph memory design.", source="notes"))

    recall_pipeline = MemoryPipeline(
        retrieval=EntityRetrieval(top_k=3, layer="knowledge_graph"),
        readout=ConcatenateReadout(separator="\n\n"),
        store=store,
    )
    readout = recall_pipeline.recall(Query(text="Alice"))

    graph_records = store.iter_records("knowledge_graph")

    print("store topology:")
    pprint(
        [
            {
                "name": layer.name,
                "theme": layer.theme,
                "shape": layer.shape,
                "indices": layer.indices,
            }
            for layer in store.topology.layers
        ]
    )
    print()

    print("organization trace:")
    pprint(packet.trace["organization"])
    print()

    print("graph layer records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "graph": record.metadata.get("graph"),
                "representation": record.metadata.get("representation"),
            }
            for record in graph_records
        ]
    )
    print()

    print("entity retrieval readout:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
