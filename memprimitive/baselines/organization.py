"""Baseline: organization primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet, Placement
from ..interfaces import OrganizationModule

from ._trace import copy_trace


class AppendOrganization(OrganizationModule):
    """Assign each unit to a fixed target layer (flat placement, no graph edges).

    Constructor: ``target_layer`` must be a non-empty string (same rules as
    ``Placement.target_layer`` / ``core._require_non_empty_text``).

    ``run`` requires ``packet.units`` and ``packet.decisions`` with equal length.
    Emits one ``Placement`` per unit (even when the decision is ``False``;
    evolution decides whether to persist). The store is unchanged.
    """

    spec = ModuleSpec(
        name="append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
    )

    def __init__(self, target_layer: str = "default") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("AppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("AppendOrganization requires decisions aligned with units.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
        }
        return replace(packet, placements=placements, trace=trace), store


BASELINE_SLOT: Final[str] = "organization"
BASELINE_CLASSES: Final[tuple[type[OrganizationModule], ...]] = (AppendOrganization,)
