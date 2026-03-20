"""Baseline: unit formation primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import UnitFormationModule

from ._trace import copy_trace


class PassThroughUnitFormation(UnitFormationModule):
    """Map one observation to a single memory unit without splitting or filtering.

    ``run`` requires ``packet.observation`` (validated ``Observation`` from
    ``core``). Output is ``units`` with length 1; metadata includes ``source``
    and ``provenance`` (observation id and source). The store is unchanged.
    """

    spec = ModuleSpec(
        name="pass_through_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.text", "units.metadata.provenance"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("PassThroughUnitFormation requires packet.observation.")

        unit = MemoryUnit(
            text=packet.observation.text,
            timestamp=packet.observation.timestamp,
            metadata={
                "source": packet.observation.source,
                "provenance": {
                    "observation_id": packet.observation.observation_id,
                    "source": packet.observation.source,
                },
                **packet.observation.metadata,
            },
        )
        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id],
        }
        return replace(packet, units=[unit], trace=trace), store


BASELINE_SLOT: Final[str] = "unit_formation"
BASELINE_CLASSES: Final[tuple[type[UnitFormationModule], ...]] = (PassThroughUnitFormation,)
