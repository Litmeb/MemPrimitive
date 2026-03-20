"""Baseline: write trigger primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import WriteTriggerModule

from ._trace import copy_trace


class AlwaysWriteTrigger(WriteTriggerModule):
    """Mark every unit as eligible for write (always ``True``).

    ``run`` requires ``packet.units``. Emits ``decisions`` with one ``True`` per
    unit (same order as ``units``). The store is unchanged.
    """

    spec = ModuleSpec(
        name="always_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AlwaysWriteTrigger requires packet.units.")

        decisions = [True for _ in packet.units]
        trace = copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "decisions": decisions,
        }
        return replace(packet, decisions=decisions, trace=trace), store


BASELINE_SLOT: Final[str] = "write_trigger"
BASELINE_CLASSES: Final[tuple[type[WriteTriggerModule], ...]] = (AlwaysWriteTrigger,)
