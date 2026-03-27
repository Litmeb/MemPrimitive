"""Demo: shared trigger-family infrastructure for TiM-like and A-MEM-like motifs.

From the repo root::

    python -m memprimitive.example.demonstration.trigger_family_infrastructure
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryRecord, MemoryStore, MemoryUnit, Packet, Placement, StoreLayerSpec, StoreTopology
from memprimitive.baselines._trigger_family import (
    GraphLayerGate,
    HasEmbeddingGate,
    MetadataFlagSignal,
    NeighborCountSignal,
    PartitionKeyPresentSignal,
    PlacementExistsSignal,
    SchemaPresentGate,
    TopNeighborSimilaritySignal,
    TriggerContext,
    UnitTypeSignal,
    VectorIndexReadyGate,
)


def _graph_vector_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity", "vector")),
            ]
        )
    )


def tim_like_demo() -> None:
    packet = Packet(
        units=[
            MemoryUnit(
                text="Remember Alice prefers jasmine tea.",
                unit_id="unit-tim",
                unit_type="tim_thought",
                metadata={
                    "tim": {
                        "write": True,
                        "group_id": "alice-profile",
                    },
                    "note": {"summary": "Alice preference note"},
                },
            )
        ],
        placements=[Placement(unit_id="unit-tim", target_layer="default")],
    )
    context = TriggerContext(packet=packet, store=MemoryStore(), output_field="decisions", trace_key="write_trigger")

    print("TiM-like shared trigger infrastructure")
    pprint(UnitTypeSignal(expected_unit_type="tim_thought", signal_name="is_tim_thought").provide(context, 0))
    pprint(MetadataFlagSignal(path="tim.write", signal_name="write_flag").provide(context, 0))
    pprint(PlacementExistsSignal().provide(context, 0))
    pprint(PartitionKeyPresentSignal(paths=("tim.group_id", "tim.hash_index")).provide(context, 0))
    print(
        "schema_present=",
        SchemaPresentGate(paths=("note.summary",), source="unit.metadata").evaluate(
            context,
            0,
            signals={},
            score=0.0,
        ),
    )
    print()


def amem_like_demo() -> None:
    store = _graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-existing",
            layer="knowledge_graph",
            text="Alice studies graph memory.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Alice graph note", unit_id="unit-new", embedding=[0.9, 0.1])],
        placements=[Placement(unit_id="unit-new", target_layer="knowledge_graph")],
    )
    context = TriggerContext(packet=packet, store=store, output_field="evolution_decisions", trace_key="evolution_trigger")

    print("A-MEM-like shared trigger infrastructure")
    pprint(NeighborCountSignal(top_k=3).provide(context, 0))
    pprint(TopNeighborSimilaritySignal(top_k=3).provide(context, 0))
    print("has_embedding=", HasEmbeddingGate().evaluate(context, 0, signals={}, score=0.0))
    print("vector_index_ready=", VectorIndexReadyGate().evaluate(context, 0, signals={}, score=0.0))
    print("graph_layer=", GraphLayerGate().evaluate(context, 0, signals={}, score=0.0))


def main() -> None:
    tim_like_demo()
    amem_like_demo()


if __name__ == "__main__":
    main()
