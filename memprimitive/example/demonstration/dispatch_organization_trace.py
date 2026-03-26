"""End-to-end example showing explicit dispatch for one slot over shared store state.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.dispatch_organization_trace

Or from this directory (script adds the repo root to ``sys.path``)::

    python dispatch_organization_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/demonstration/dispatch_organization_trace.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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

    pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "entities", "triple", "tags")),
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

    ingest_packet = pipeline.ingest(
        Observation(text="Alice is debugging graph-backed memory retrieval.", source="notes")
    )
    pipeline.ingest(Observation(text="Bob works on semantic memory routing.", source="notes"))

    query = Query(text="Alice")
    readout = pipeline.recall(query)
    retrieval_packet, _ = pipeline.retrieval.run(Packet(query=query), store)

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

    print("dispatch trace for organization:")
    pprint(ingest_packet.trace["dispatch"]["organization"])
    print()

    print("layer-aware retrieval trace:")
    pprint(retrieval_packet.trace["retrieval"])
    print()

    print("merged readout text:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
