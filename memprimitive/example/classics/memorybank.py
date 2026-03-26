"""MemoryBank (Zhong et al., 2024) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.memorybank

Or from this directory (script adds the repo root to ``sys.path``)::

    python memorybank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    BM25Retrieval,
    ConcatenateReadout,
    ConditionalLayerOrganization,
    PassThroughUnitFormation,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="short_term", theme="working", capacity="token_limited", indices=("temporal",)),
            StoreLayerSpec(name="long_term", theme="semantic", indices=("vector", "temporal", "keyword")),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "triple", "tags")),
        write_trigger=AlwaysWriteTrigger(),
        organization=ConditionalLayerOrganization(
            default_layer="short_term",
            rules=(
                {"has_entity": True, "target_layer": "long_term"},
            ),
        ),
        retrieval=BM25Retrieval(top_k=20, layer="long_term"),
        readout=ConcatenateReadout(separator="\n"),
    )

    pipeline.ingest(Observation(text="Alice works at OpenAI in San Francisco.", source="dialogue"))
    readout = pipeline.recall(Query(text="Where does Alice work?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
