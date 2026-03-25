"""End-to-end example showing automatic routing with ``ConditionalLayerOrganization``.

From the repo root (recommended)::

    python -m memprimitive.example.example8

Or from this directory (script adds the repo root to ``sys.path``)::

    python example8.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/example8.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, StoreLayerSpec, StoreTopology
from memprimitive.baselines import BasicRepresentation, ConditionalLayerOrganization


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(name="semantic", theme="semantic", indices=("entity", "keyword")),
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
        representation=BasicRepresentation(elements=("text", "entities", "triple", "tags", "keywords", "summary")),
        organization=ConditionalLayerOrganization(
            default_layer="working",
            rules=(
                {"tag_contains": "structured_triple", "target_layer": "knowledge_graph"},
                {"has_entity": True, "target_layer": "semantic"},
            ),
        ),
        store=store,
    )

    observations = [
        Observation(
            text="Remember to explain the routing demo clearly in the next reply.",
            source="dialogue",
        ),
        Observation(
            text="Alice likes jasmine tea.",
            source="notes",
        ),
        Observation(
            text="Bob works on graph memory systems.",
            source="notes",
        ),
        Observation(
            text="Need a short checklist for the benchmark run tomorrow.",
            source="dialogue",
        ),
    ]

    packets = [pipeline.ingest(observation) for observation in observations]

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

    print("routing results per observation:")
    pprint(
        [
            {
                "text": observation.text,
                "placements": [
                    {
                        "unit_id": placement.unit_id,
                        "target_layer": placement.target_layer,
                    }
                    for placement in packet.placements or []
                ],
                "trace": packet.trace["organization"],
            }
            for observation, packet in zip(observations, packets, strict=True)
        ]
    )
    print()

    print("records grouped by layer:")
    pprint(
        {
            layer_name: [
                {
                    "record_id": record.record_id,
                    "text": record.text,
                    "representation": record.metadata.get("representation"),
                }
                for record in store.iter_records(layer_name)
            ]
            for layer_name in store.topology.layer_names
        }
    )


if __name__ == "__main__":
    main()
