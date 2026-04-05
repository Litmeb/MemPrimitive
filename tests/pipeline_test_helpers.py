from __future__ import annotations

from memprimitive.core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Packet, Readout, RetrievedSet
from memprimitive.interfaces import PrimitiveModule


class _FreePipelineProbeModule(PrimitiveModule):
    def __init__(
        self,
        *,
        name: str,
        slot: str,
        record_text: str | None = None,
        produce_retrieved: bool = False,
        produce_readout: bool = False,
    ) -> None:
        self.spec = ModuleSpec(name=name, slot=slot)
        self.record_text = record_text
        self.produce_retrieved = produce_retrieved
        self.produce_readout = produce_readout

    def run(self, packet: Packet, store: MemoryStore):
        store.metadata.setdefault("free_pipeline_log", []).append(self.spec.name)
        if self.record_text is not None:
            unit = MemoryUnit(text=self.record_text, metadata={"module": self.spec.name})
            record = MemoryRecord.from_unit(unit, layer="default", sequence_id=store.next_sequence_id())
            store.append(record)
        if self.produce_retrieved:
            packet.retrieved = packet.retrieved or RetrievedSet()
            packet.retrieved.items = store.iter_records()
        if self.produce_readout:
            packet.readout = Readout(
                text=" | ".join(record.text for record in packet.retrieved.items),
                source_ids=[record.record_id for record in packet.retrieved.items],
                metadata={"free_pipeline_log": list(store.metadata.get("free_pipeline_log", []))},
            )
        return packet, store

