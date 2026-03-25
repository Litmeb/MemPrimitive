"""End-to-end example showing layer-aware retrieval over working + knowledge_graph.

From the repo root (recommended)::

    python -m memprimitive.example.example9

Or from this directory (script adds the repo root to ``sys.path``)::

    python example9.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/example9.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memprimitive import (
    DispatchOrganization,
    MemoryPipeline,
    MemoryStore,
    Observation,
    Packet,
    Query,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EntityRetrieval,
    GraphAppendOrganization,
    LayerAwareRetrieval,
    RecencyRetrieval,
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
        representation=BasicRepresentation(elements=("text", "tags", "entities", "triple", "tags")),
        organization=DispatchOrganization(
            (
                AppendOrganization(target_layer="working"),
                GraphAppendOrganization(target_layer="knowledge_graph"),
            ),
            primary_index=0,
        ),
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=2),
            retriever_by_layer={"knowledge_graph": EntityRetrieval(top_k=2)},
            top_k=4,
        ),
        readout=ConcatenateReadout(separator="\n\n"),
        store=store,
    )
    working_writer.ingest(Observation(text="Alice is debugging the retrieval merge order.", source="dialogue"))
    working_writer.ingest(Observation(text="The current task is to explain graph-backed recall.", source="dialogue"))
    
    query = Query(text="Alice")
    readout = working_writer.recall(query)
    packet, _ = working_writer.retrieval.run(Packet(query=query), store)

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
