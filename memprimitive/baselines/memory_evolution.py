"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet
from ..interfaces import MemoryEvolutionModule

from ._trace import copy_trace


class AppendOnlyEvolution(MemoryEvolutionModule):
    """Append ``MemoryRecord``s for units whose evolution mask is true.

    ``run`` requires ``packet.units`` and ``packet.placements``. It prefers
    ``packet.evolution_decisions`` when available; otherwise it falls back to
    ``packet.decisions`` for backward compatibility. The active mask must align
    with ``units`` and ``placements``. For each triple ``(unit, decision,
    placement)``, if ``decision`` is true, appends a record to ``store`` at
    ``placement.target_layer`` using ``store.next_sequence_id()`` for stable
    record ids. Mutates ``store``; packet fields other than ``trace`` are unchanged.
    """

    spec = ModuleSpec(
        name="append_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements"),
        output_guarantees=("trace.memory_evolution.appended_record_ids",),
        side_effects=("modify_store", "append_records"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("AppendOnlyEvolution requires packet.placements.")
        active_decisions = packet.evolution_decisions
        decision_source = "evolution_decisions"
        if active_decisions is None:
            active_decisions = packet.decisions
            decision_source = "decisions"
        if active_decisions is None:
            raise ValueError("AppendOnlyEvolution requires packet.evolution_decisions or packet.decisions.")
        if not (len(packet.units) == len(active_decisions) == len(packet.placements)):
            raise ValueError(
                "AppendOnlyEvolution requires aligned units, active decisions, and placements."
            )

        appended_record_ids: list[str] = []
        for unit, decision, placement in zip(packet.units, active_decisions, packet.placements, strict=True):
            if not decision:
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
            store.append(record)
            appended_record_ids.append(record.record_id)

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": decision_source,
            "appended_record_ids": appended_record_ids,
        }
        return replace(packet, trace=trace), store


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (AppendOnlyEvolution,)
