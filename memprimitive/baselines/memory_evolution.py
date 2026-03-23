"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
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


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (AppendOnlyEvolution,)
