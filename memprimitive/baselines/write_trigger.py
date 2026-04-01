"""Minimal baseline write-trigger implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import WriteTriggerModule
from ..utils._trace import copy_trace


def _require_units(packet: Packet, *, module_name: str) -> list:
    if packet.units is None:
        raise ValueError(f"{module_name} requires packet.units.")
    return packet.units


def _write_trace(
    packet: Packet,
    *,
    module_name: str,
    decisions: list[bool],
    constant: float,
    threshold: float | None,
) -> Packet:
    trace = copy_trace(packet)
    per_unit = []
    for unit, decision in zip(packet.units or [], decisions, strict=True):
        per_unit.append(
            {
                "unit_id": unit.unit_id,
                "constant": float(constant),
                "decision": bool(decision),
            }
        )
    trace["write_trigger"] = {
        "module": module_name,
        "decisions": list(decisions),
        "constant": float(constant),
        "threshold": None if threshold is None else float(threshold),
        "per_unit": per_unit,
    }
    return replace(packet, decisions=list(decisions), trace=trace)


class AlwaysWriteTrigger(WriteTriggerModule):
    """Mark every unit as eligible for write."""

    spec = ModuleSpec(
        name="always_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = _require_units(packet, module_name=self.spec.name)
        decisions = [True] * len(units)
        return _write_trace(packet, module_name=self.spec.name, decisions=decisions, constant=1.0, threshold=None), store


class ThresholdWriteTrigger(WriteTriggerModule):
    """Constant-threshold baseline for the write trigger slot."""

    spec = ModuleSpec(
        name="threshold_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(self, *, threshold: float = 0.5, constant: float = 1.0) -> None:
        self.threshold = float(threshold)
        self.constant = float(constant)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = _require_units(packet, module_name=self.spec.name)
        decision = self.constant >= self.threshold
        decisions = [decision] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                decisions=decisions,
                constant=self.constant,
                threshold=self.threshold,
            ),
            store,
        )


BASELINE_SLOT: Final[str] = "write_trigger"
BASELINE_CLASSES: Final[tuple[type[WriteTriggerModule], ...]] = (
    AlwaysWriteTrigger,
    ThresholdWriteTrigger,
)
