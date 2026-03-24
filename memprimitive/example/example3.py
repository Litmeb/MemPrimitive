"""End-to-end example showing how to define and use a topology-backed store.

From the repo root (recommended)::

    python -m memprimitive.example.example3

Or from this directory (script adds the repo root to ``sys.path``)::

    python example3.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/example3.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(name="episodic", theme="session_memory", indices=("temporal", "keyword")),
            StoreLayerSpec(name="knowledge_graph", theme="knowledge_graph", shape="Graph", indices=("graph", "entity")),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(target_layer="episodic"),
        retrieval=RecencyRetrieval(top_k=2, layer="episodic"),
        readout=ConcatenateReadout(),
        store=store,
    )

    packet = pipeline.ingest(
        Observation(text="The user wants a store with an explicit topology.", source="notes")
    )
    pipeline.ingest(
        Observation(text="The episodic layer keeps recent dialogue-like memories.", source="notes")
    )

    print("store topology:")
    pprint(
        [
            {
                "name": layer.name,
                "theme": layer.theme,
                "shape": layer.shape,
                "indices": layer.indices,
                "capacity": layer.capacity,
            }
            for layer in pipeline.store.topology.layers
        ]
    )
    print()

    print("organization target layer:", packet.trace["organization"]["target_layer"])
    print("store has graph layer:", pipeline.store.has_graph_layer())
    print("episodic supports keyword:", pipeline.store.layer_supports_index("episodic", "keyword"))
    print("records per layer:", {name: pipeline.store.count(name) for name in pipeline.store.topology.layer_names})
    print()

    readout = pipeline.recall(Query(text="What kind of store does the user want?"))

    print("readout text:", readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
