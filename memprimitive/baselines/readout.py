"""Baseline: readout primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet, Readout
from ..interfaces import ReadoutModule

from ._trace import copy_trace


class ConcatenateReadout(ReadoutModule):
    """Turn retrieval items into a single string plus source record ids.

    Constructor: ``separator`` is inserted between consecutive record texts
    (default newline).

    ``run`` requires ``packet.retrieved`` (may be empty). Sets ``readout.text``
    to the joined texts and ``readout.source_ids`` to record ids in retrieval
    order. The store is unchanged.
    """

    spec = ModuleSpec(
        name="concatenate_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, separator: str = "\n") -> None:
        self.separator = separator

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("ConcatenateReadout requires packet.retrieved.")

        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        text = self.separator.join(record.text for record in items)
        readout = Readout(
            text=text,
            source_ids=source_ids,
            metadata={"item_count": len(items)},
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
        }
        return replace(packet, readout=readout, trace=trace), store


BASELINE_SLOT: Final[str] = "readout"
BASELINE_CLASSES: Final[tuple[type[ReadoutModule], ...]] = (ConcatenateReadout,)
