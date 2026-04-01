"""Minimal baseline evolution-trigger implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import EvolutionTriggerModule
from ..utils._trace import copy_trace


def _require_units_and_placements(packet: Packet, *, module_name: str) -> list:
    if packet.units is None:
        raise ValueError(f"{module_name} requires packet.units.")
    if packet.placements is None:
        raise ValueError(f"{module_name} requires packet.placements.")
    if len(packet.units) != len(packet.placements):
        raise ValueError(f"{module_name} requires aligned packet.units and packet.placements.")
    for unit, placement in zip(packet.units, packet.placements, strict=True):
        if placement.unit_id != unit.unit_id:
            raise ValueError(f"{module_name} requires aligned packet.units and packet.placements.")
    return packet.units


def _evolution_trace(
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
    trace["evolution_trigger"] = {
        "module": module_name,
        "evolution_decisions": list(decisions),
        "constant": float(constant),
        "threshold": None if threshold is None else float(threshold),
        "per_unit": per_unit,
    }
    return replace(packet, evolution_decisions=list(decisions), trace=trace)


class NeverEvolutionTrigger(EvolutionTriggerModule):
    """Keep extra memory evolution disabled by default."""

    spec = ModuleSpec(
        name="never_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = _require_units_and_placements(packet, module_name=self.spec.name)
        decisions = [False] * len(units)
        return _evolution_trace(packet, module_name=self.spec.name, decisions=decisions, constant=1.0, threshold=None), store


class ThresholdEvolutionTrigger(EvolutionTriggerModule):
    """Constant-threshold baseline for the evolution trigger slot."""

    spec = ModuleSpec(
        name="threshold_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(self, *, threshold: float = 0.5, constant: float = 1.0) -> None:
        self.threshold = float(threshold)
        self.constant = float(constant)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = _require_units_and_placements(packet, module_name=self.spec.name)
        decision = self.constant >= self.threshold
        decisions = [decision] * len(units)
        return (
            _evolution_trace(
                packet,
                module_name=self.spec.name,
                decisions=decisions,
                constant=self.constant,
                threshold=self.threshold,
            ),
            store,
        )


BASELINE_SLOT: Final[str] = "evolution_trigger"
BASELINE_CLASSES: Final[tuple[type[EvolutionTriggerModule], ...]] = (
    NeverEvolutionTrigger,
    ThresholdEvolutionTrigger,
)
