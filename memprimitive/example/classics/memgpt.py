"""MemGPT (Packer et al., 2023) — motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.memgpt

Or from this directory (script adds the repo root to ``sys.path``)::

    python memgpt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import (
    DispatchOrganization,
    MemoryPipeline,
    MemoryStore,
    Observation,
    Query,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    LayerAwareRetrieval,
    PassThroughUnitFormation,
    RecencyRetrieval,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="main_context", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(name="archival", theme="semantic", indices=("vector", "keyword", "temporal")),
            StoreLayerSpec(name="recall", theme="semantic", indices=("temporal",)),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "summary", "keywords")),
        write_trigger=AlwaysWriteTrigger(),
        organization=DispatchOrganization(
            (
                AppendOrganization(target_layer="main_context"),
                AppendOrganization(target_layer="archival"),
            ),
            primary_index=0,
        ),
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=10, layer="archival"),
            retriever_by_layer={"archival": RecencyRetrieval(top_k=10, layer="archival")},
            top_k=10,
        ),
        readout=ConcatenateReadout(separator="\n\n"),
    )

    pipeline.ingest(Observation(text="Pinned note: review memory architecture docs.", source="dialogue"))
    readout = pipeline.recall(Query(text="What should we review?"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
