"""Baseline: organization primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet, Placement
from ..interfaces import OrganizationModule

from ._trace import copy_trace


class AppendOrganization(OrganizationModule):
    """Assign each unit to a fixed target layer and commit normal ingest-time writes.

    Constructor: ``target_layer`` must be a non-empty string (same rules as
    ``Placement.target_layer`` / ``core._require_non_empty_text``).

    ``run`` requires ``packet.units`` and ``packet.decisions`` with equal length.
    Emits one ``Placement`` per unit and appends ``MemoryRecord`` objects for
    units whose decision is ``True``. Mutates ``store`` as part of the normal
    write path.
    """

    spec = ModuleSpec(
        name="append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
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
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        skipped_units = 0
        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
            store.append(record)
            written_record_ids.append(record.record_id)
            written_unit_ids.append(unit.unit_id)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "skipped_unit_count": skipped_units,
        }
        return replace(packet, placements=placements, trace=trace), store


BASELINE_SLOT: Final[str] = "organization"
BASELINE_CLASSES: Final[tuple[type[OrganizationModule], ...]] = (AppendOrganization,)
