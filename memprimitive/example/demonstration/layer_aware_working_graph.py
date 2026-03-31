"""End-to-end example showing layer-aware retrieval over working + knowledge_graph.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.layer_aware_working_graph

Or from this directory (script adds the repo root to ``sys.path``)::

    python layer_aware_working_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/demonstration/layer_aware_working_graph.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EntityRetrieval,
    GraphAppendOrganization,
    LayerAwareRetrieval,
    RecencyRetrieval,
    TripleRepresentation,
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

    working_writer = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "tags")),
        organization=AppendOrganization(target_layer="working"),
        store=store,
    )
    working_writer.ingest(Observation(text="Alice is debugging the retrieval merge order.", source="dialogue"))
    working_writer.ingest(Observation(text="The current task is to explain graph-backed recall.", source="dialogue"))

    graph_writer = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text",)),
            TripleRepresentation(method="direct"),
            BasicRepresentation(elements=("tags",)),
        ),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        store=store,
    )
    graph_writer.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
    graph_writer.ingest(Observation(text="Alice studies graph memory systems.", source="notes"))
    graph_writer.ingest(Observation(text="Bob works on memory retrieval systems.", source="notes"))

    recall_pipeline = MemoryPipeline(
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=2),
            retriever_by_layer={"knowledge_graph": EntityRetrieval(top_k=2)},
            top_k=4,
        ),
        readout=ConcatenateReadout(separator="\n\n"),
        store=store,
    )

    query = Query(text="Alice")
    readout = recall_pipeline.recall(query)
    packet, _ = recall_pipeline.retrieval.run(Packet(query=query), store)

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

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("layer-aware retrieval trace:")
    pprint(packet.trace["retrieval"])
    print()

    print("merged readout text:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
