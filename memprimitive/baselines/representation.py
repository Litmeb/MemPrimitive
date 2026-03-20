"""Baseline: representation primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import RepresentationModule

from ._trace import copy_trace


class BasicRepresentation(RepresentationModule):
    """Normalize unit text and attach a lightweight text-based representation.

    For each unit: strips surrounding whitespace on ``text``, and sets
    ``metadata["representation"]`` with ``text`` and ``normalized_text``
    (casefold). ``run`` requires ``packet.units`` to be set (list, possibly empty).
    Unit ids are preserved. The store is unchanged.
    """

    spec = ModuleSpec(
        name="basic_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.text", "units.metadata.representation"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("BasicRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        for unit in packet.units:
            represented_units.append(
                replace(
                    unit,
                    text=unit.text.strip(),
                    metadata={
                        **unit.metadata,
                        "representation": {
                            "text": unit.text.strip(),
                            "normalized_text": unit.text.casefold().strip(),
                        },
                    },
                )
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in represented_units],
        }
        return replace(packet, units=represented_units, trace=trace), store


BASELINE_SLOT: Final[str] = "representation"
BASELINE_CLASSES: Final[tuple[type[RepresentationModule], ...]] = (BasicRepresentation,)
