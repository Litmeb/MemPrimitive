"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

from ..core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import MemoryEvolutionModule

from ._graph_family import graph_metadata_from_record, rewrite_graph_record
from ._trace import copy_trace


class AppendOnlyEvolution(MemoryEvolutionModule):
    """Run an optional extra evolution pass over already-organized memory.

    ``run`` requires ``packet.units`` and ``packet.placements``. It prefers
    ``packet.evolution_decisions`` as the extra-evolution mask. The active mask
    must align with ``units`` and ``placements``. Stage-1 baseline behavior is a
    no-op extra pass: it records which units would participate in extra evolution
    but does not modify the store.
    """

    spec = ModuleSpec(
        name="append_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("AppendOnlyEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("AppendOnlyEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError(
                "AppendOnlyEvolution requires aligned units, evolution decisions, and placements."
            )

        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": [],
        }
        return replace(packet, trace=trace), store


class TraceOnlyEvolution(MemoryEvolutionModule):
    """No-op evolution that records explicit effect placeholders for active units.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions`` aligned by index. The store is not mutated.
    """

    spec = ModuleSpec(
        name="trace_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TraceOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("TraceOnlyEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("TraceOnlyEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("TraceOnlyEvolution requires aligned units, evolution decisions, and placements.")

        effects = [
            {
                "effect_type": "trace_only",
                "unit_id": unit.unit_id,
                "target_layer": placement.target_layer,
            }
            for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True)
            if decision
        ]
        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": [effect["unit_id"] for effect in effects],
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class SummaryRewriteEvolution(MemoryEvolutionModule):
    """Append summary records for evolution-active units into a target layer.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions`` aligned by index. Active units are summarized
    into new append-only records; original records remain unchanged.
    """

    spec = ModuleSpec(
        name="summary_rewrite_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = "default") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("SummaryRewriteEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("SummaryRewriteEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("SummaryRewriteEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("SummaryRewriteEvolution requires aligned units, evolution decisions, and placements.")

        effects = []
        active_unit_ids = []
        for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True):
            if not decision:
                continue
            active_unit_ids.append(unit.unit_id)
            summary_unit = self._summary_unit(unit)
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(summary_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            effects.append(
                {
                    "effect_type": "summary_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "target_layer": self.target_layer,
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
        }
        return replace(packet, trace=trace), store

    @staticmethod
    def _summary_unit(unit: MemoryUnit) -> MemoryUnit:
        representation = unit.metadata.get("representation", {})
        summary_text = (
            representation.get("summary")
            or representation.get("description")
            or unit.description
            or unit.text
        )
        return replace(
            unit,
            text=str(summary_text).strip(),
            unit_type="summary",
            metadata={
                **unit.metadata,
                "evolution_source_unit_id": unit.unit_id,
                "evolution_style": "summary_rewrite",
            },
        )


class LayerMoveEvolution(MemoryEvolutionModule):
    """Copy-append evolution-active units into another layer.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions`` aligned by index. Active units are copied into
    ``target_layer`` as new records without deleting originals.
    """

    spec = ModuleSpec(
        name="layer_move_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = "default") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("LayerMoveEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("LayerMoveEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("LayerMoveEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("LayerMoveEvolution requires aligned units, evolution decisions, and placements.")

        effects = []
        active_unit_ids = []
        for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True):
            if not decision:
                continue
            active_unit_ids.append(unit.unit_id)
            moved_unit = replace(
                unit,
                metadata={
                    **unit.metadata,
                    "evolution_source_unit_id": unit.unit_id,
                    "move_style": "copy_append",
                },
            )
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(moved_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            effects.append(
                {
                    "effect_type": "layer_move_copy_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "target_layer": self.target_layer,
                    "move_style": "copy_append",
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
        }
        return replace(packet, trace=trace), store


def _latest_record_for_unit(store: MemoryStore, *, layer: str, unit_id: str) -> MemoryRecord | None:
    matches = [record for record in store.iter_records(layer) if record.unit_id == unit_id]
    if not matches:
        return None
    return matches[-1]


def _record_text_tokens(record: MemoryRecord) -> set[str]:
    return {token for token in record.text.casefold().split() if token}


def _graph_neighbor_score(target_record: MemoryRecord, candidate_record: MemoryRecord) -> float:
    target_graph = graph_metadata_from_record(target_record)
    candidate_graph = graph_metadata_from_record(candidate_record)
    target_entities = {entity.casefold() for entity in target_graph["entities"]}
    candidate_entities = {entity.casefold() for entity in candidate_graph["entities"]}
    entity_overlap = len(target_entities & candidate_entities)
    text_overlap = len(_record_text_tokens(target_record) & _record_text_tokens(candidate_record))
    return float((2 * entity_overlap) + text_overlap)


class GraphNeighborAppendEvolution(MemoryEvolutionModule):
    """Append graph links for evolution-active units already written to a graph layer.

    Constructor: ``target_layer`` must refer to a declared graph layer and
    ``neighbor_limit`` must be positive. The simplified baseline scores existing
    same-layer graph records by shared entities and token overlap, then appends
    links from the newly written record to the strongest neighbors. This is an
    inferred engineering decomposition of graph-link evolution.

    ``run`` requires aligned ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions``. Only units placed into ``target_layer`` are
    considered. The module mutates only records inside that graph layer.
    """

    spec = ModuleSpec(
        name="graph_neighbor_append_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )

    def __init__(self, *, target_layer: str = "knowledge_graph", neighbor_limit: int = 2, bidirectional: bool = True) -> None:
        if neighbor_limit <= 0:
            raise ValueError("GraphNeighborAppendEvolution requires neighbor_limit > 0.")
        self.target_layer = target_layer
        self.neighbor_limit = neighbor_limit
        self.bidirectional = bidirectional

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphNeighborAppendEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("GraphNeighborAppendEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("GraphNeighborAppendEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError(
                "GraphNeighborAppendEvolution requires aligned units, evolution decisions, and placements."
            )

        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True):
            if not decision or placement.target_layer != self.target_layer:
                continue
            target_record = _latest_record_for_unit(store, layer=self.target_layer, unit_id=unit.unit_id)
            if target_record is None:
                continue

            candidates = [
                record
                for record in store.iter_records(self.target_layer)
                if record.record_id != target_record.record_id
            ]
            scored_candidates = [
                (_graph_neighbor_score(target_record, candidate), candidate)
                for candidate in candidates
            ]
            scored_candidates = [item for item in scored_candidates if item[0] > 0.0]
            scored_candidates.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].record_id))
            selected_neighbors = [record for _, record in scored_candidates[: self.neighbor_limit]]
            linked_record_ids = [record.record_id for record in selected_neighbors]

            if linked_record_ids:
                active_unit_ids.append(unit.unit_id)
                store.add_graph_links(self.target_layer, target_record.record_id, linked_record_ids)
                refreshed_target = next(
                    record
                    for record in store.iter_records(self.target_layer)
                    if record.record_id == target_record.record_id
                )
                target_effect = {
                    "effect_type": "graph_neighbor_append",
                    "unit_id": unit.unit_id,
                    "record_id": target_record.record_id,
                    "target_layer": self.target_layer,
                    "linked_record_ids": linked_record_ids,
                    "bidirectional": self.bidirectional,
                }
                store.replace_record(
                    self.target_layer,
                    refreshed_target.record_id,
                    rewrite_graph_record(
                        refreshed_target,
                        linked_record_ids=linked_record_ids,
                        link_trace_entry=target_effect,
                    ),
                )

                if self.bidirectional:
                    for neighbor_record in selected_neighbors:
                        store.add_graph_links(self.target_layer, neighbor_record.record_id, [target_record.record_id])
                        refreshed_neighbor = next(
                            record
                            for record in store.iter_records(self.target_layer)
                            if record.record_id == neighbor_record.record_id
                        )
                        store.replace_record(
                            self.target_layer,
                            refreshed_neighbor.record_id,
                            rewrite_graph_record(
                                refreshed_neighbor,
                                linked_record_ids=[target_record.record_id],
                                link_trace_entry={
                                    "effect_type": "graph_neighbor_backlink",
                                    "record_id": neighbor_record.record_id,
                                    "linked_record_ids": [target_record.record_id],
                                    "source_record_id": target_record.record_id,
                                    "target_layer": self.target_layer,
                                },
                            ),
                        )

                effects.append(target_effect)

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "target_layer": self.target_layer,
        }
        return replace(packet, trace=trace), store


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (
    AppendOnlyEvolution,
    TraceOnlyEvolution,
    SummaryRewriteEvolution,
    LayerMoveEvolution,
    GraphNeighborAppendEvolution,
)
