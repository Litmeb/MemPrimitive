"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import MemoryEvolutionModule

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


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (
    AppendOnlyEvolution,
    TraceOnlyEvolution,
    SummaryRewriteEvolution,
    LayerMoveEvolution,
)
