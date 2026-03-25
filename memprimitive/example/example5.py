"""End-to-end example showing layer-aware retrieval across a topology-backed store.

From the repo root (recommended)::

    python -m memprimitive.example.example5

Or from this directory (script adds the repo root to ``sys.path``)::

    python example5.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/example5.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    LayerAwareRetrieval,
    RecencyRetrieval,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(name="semantic", theme="semantic", indices=("vector", "entity")),
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "tags")),
        organization=AppendOrganization(target_layer="working"),
        store=store,
    )
    write_pipeline.ingest(Observation(text="Alice likes short status updates.", source="dialogue"))
    write_pipeline.ingest(Observation(text="Alice is debugging a retrieval merge edge case.", source="dialogue"))

    semantic_writer = MemoryPipeline(
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "tags")),
        organization=AppendOrganization(target_layer="semantic"),
        store=store,
    )
    semantic_writer.ingest(Observation(text="Alice studies graph memory systems.", source="notes"))
    semantic_writer.ingest(Observation(text="Alice builds semantic retrieval modules.", source="notes"))

    recall_pipeline = MemoryPipeline(
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=2),
            retriever_by_layer={"semantic": EmbeddingSimilarityRetrieval(top_k=2)},
            top_k=3,
        ),
        readout=ConcatenateReadout(separator="\n\n"),
        store=store,
    )

    readout = recall_pipeline.recall(Query(text="What do we know about Alice's retrieval work?"))
    packet, _ = recall_pipeline.retrieval.run(Packet(query=Query(text="What do we know about Alice's retrieval work?")), store)

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

    print("layer-aware retrieval trace:")
    pprint(packet.trace["retrieval"])
    print()

    print("readout text:", readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
