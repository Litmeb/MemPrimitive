"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet
from ..interfaces import MemoryEvolutionModule

from ._trace import copy_trace


class AppendOnlyEvolution(MemoryEvolutionModule):
    """Append ``MemoryRecord``s for units whose decision is true (no merge/delete).

    ``run`` requires ``packet.units``, ``packet.decisions``, and ``packet.placements``
    with pairwise equal lengths. For each triple ``(unit, decision, placement)``,
    if ``decision`` is true, appends a record to ``store`` at ``placement.target_layer``
    using ``store.next_sequence_id()`` for stable record ids. Skips append when
    ``decision`` is false. Mutates ``store``; packet fields other than ``trace`` are
    unchanged.
    """

    spec = ModuleSpec(
        name="append_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "decisions", "placements"),
        output_guarantees=("trace.memory_evolution.appended_record_ids",),
        side_effects=("modify_store", "append_records"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOnlyEvolution requires packet.units.")
        if packet.decisions is None:
            raise ValueError("AppendOnlyEvolution requires packet.decisions.")
        if packet.placements is None:
            raise ValueError("AppendOnlyEvolution requires packet.placements.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError("AppendOnlyEvolution requires aligned units, decisions, and placements.")

        appended_record_ids: list[str] = []
        for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True):
            if not decision:
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
            store.append(record)
            appended_record_ids.append(record.record_id)

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "appended_record_ids": appended_record_ids,
        }
        return replace(packet, trace=trace), store


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (AppendOnlyEvolution,)
