"""Shared baseline trigger implementations."""

from __future__ import annotations

from dataclasses import replace

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import TriggerModule
from ..utils._trace import copy_trace

_VALID_TRIGGER_SLOTS = frozenset({"write_trigger", "evolution_trigger"})


def _require_trigger_slot(slot: str) -> str:
    normalized = str(slot).strip()
    if normalized not in _VALID_TRIGGER_SLOTS:
        options = ", ".join(sorted(_VALID_TRIGGER_SLOTS))
        raise ValueError(f"trigger slot must be one of: {options}.")
    return normalized


def _module_name(prefix: str, slot: str) -> str:
    stem = "write" if slot == "write_trigger" else "evolution"
    return f"{prefix}_{stem}_trigger"


def _input_requirements(slot: str) -> tuple[str, ...]:
    if slot == "write_trigger":
        return ("units",)
    return ("units", "placements")


def _require_units(packet: Packet, *, module_name: str, slot: str) -> list:
    if packet.units is None:
        raise ValueError(f"{module_name} requires packet.units.")
    if slot == "evolution_trigger":
        if packet.placements is None:
            raise ValueError(f"{module_name} requires packet.placements.")
        if len(packet.units) != len(packet.placements):
            raise ValueError(f"{module_name} requires aligned packet.units and packet.placements.")
        for unit, placement in zip(packet.units, packet.placements, strict=True):
            if placement.unit_id != unit.unit_id:
                raise ValueError(f"{module_name} requires aligned packet.units and packet.placements.")
    return packet.units


def _write_trace(
    packet: Packet,
    *,
    module_name: str,
    trace_key: str,
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
    trace[trace_key] = {
        "module": module_name,
        "decisions": list(decisions),
        "constant": float(constant),
        "threshold": None if threshold is None else float(threshold),
        "per_unit": per_unit,
    }
    return replace(packet, decisions=list(decisions), trace=trace)


class _BaseTrigger(TriggerModule):
    _DEFAULT_SLOT: str | None = None
    _NAME_PREFIX: str = ""

    def __init__(self, *, slot: str | None = None) -> None:
        resolved_slot = _require_trigger_slot(slot or self._DEFAULT_SLOT or "")
        self.slot = resolved_slot
        self.spec = ModuleSpec(
            name=_module_name(self._NAME_PREFIX, resolved_slot),
            slot=resolved_slot,
            input_requirements=_input_requirements(resolved_slot),
            output_guarantees=("decisions",),
        )

    def _trace_key(self) -> str:
        return self.spec.slot

    def _require_units(self, packet: Packet) -> list:
        return _require_units(packet, module_name=self.spec.name, slot=self.spec.slot)


class AlwaysTrigger(_BaseTrigger):
    """Mark every unit as eligible for the current trigger slot."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "always"

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        decisions = [True] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                constant=1.0,
                threshold=None,
            ),
            store,
        )


class NeverTrigger(_BaseTrigger):
    """Mark every unit as ineligible for the current trigger slot."""

    _DEFAULT_SLOT = "evolution_trigger"
    _NAME_PREFIX = "never"

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        decisions = [False] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                constant=1.0,
                threshold=None,
            ),
            store,
        )


class ThresholdTrigger(_BaseTrigger):
    """Constant-threshold baseline for either trigger slot."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "threshold"

    def __init__(self, *, slot: str | None = None, threshold: float = 0.5, constant: float = 1.0) -> None:
        super().__init__(slot=slot)
        self.threshold = float(threshold)
        self.constant = float(constant)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        decision = self.constant >= self.threshold
        decisions = [decision] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                constant=self.constant,
                threshold=self.threshold,
            ),
            store,
        )


__all__ = [
    "AlwaysTrigger",
    "NeverTrigger",
    "ThresholdTrigger",
]
