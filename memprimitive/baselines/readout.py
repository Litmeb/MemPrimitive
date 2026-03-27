"""Baseline: readout primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet, Readout
from ..interfaces import ReadoutModule

from ._graph_family import graph_metadata_from_record
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


class BulletListReadout(ReadoutModule):
    """Render retrieval items as one bullet per line."""

    spec = ModuleSpec(
        name="bullet_list_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("BulletListReadout requires packet.retrieved.")
        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        text = "\n".join(f"- {record.text}" for record in items)
        readout = Readout(text=text, source_ids=source_ids, metadata={"item_count": len(items), "format": "bullet"})
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids}
        return replace(packet, readout=readout, trace=trace), store


class GroupedByLayerReadout(ReadoutModule):
    """Render retrieval items grouped by their source layer."""

    spec = ModuleSpec(
        name="grouped_by_layer_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("GroupedByLayerReadout requires packet.retrieved.")
        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        groups: dict[str, list[str]] = {}
        for record in items:
            groups.setdefault(record.layer, []).append(record.text)
        chunks = [f"[{layer}]\n" + "\n".join(texts) for layer, texts in groups.items()]
        readout = Readout(
            text="\n\n".join(chunks),
            source_ids=source_ids,
            metadata={
                "item_count": len(items),
                "group_counts": {layer: len(texts) for layer, texts in groups.items()},
                "format": "grouped_by_layer",
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids}
        return replace(packet, readout=readout, trace=trace), store


class JSONReadout(ReadoutModule):
    """Render retrieval items into a JSON string for downstream tools/agents."""

    spec = ModuleSpec(
        name="json_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("JSONReadout requires packet.retrieved.")
        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        payload = {
            "items": [
                {
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "text": record.text,
                    "timestamp": record.timestamp,
                }
                for record in items
            ],
            "source_ids": source_ids,
        }
        readout = Readout(
            text=json.dumps(payload, ensure_ascii=False),
            source_ids=source_ids,
            metadata={"item_count": len(items), "format": "json"},
        )
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids}
        return replace(packet, readout=readout, trace=trace), store


class GraphReadout(ReadoutModule):
    """Render retrieved records with graph metadata in a stable readable format.

    Constructor: ``include_links`` controls whether linked record ids are
    rendered. The module can consume mixed retrieval results, but it is designed
    for graph-layer payloads and summarizes normalized graph metadata per item.

    ``run`` requires ``packet.retrieved`` and does not mutate the store. It
    renders one graph-oriented line per record and preserves retrieval order.
    """

    spec = ModuleSpec(
        name="graph_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, *, include_links: bool = True) -> None:
        self.include_links = include_links

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("GraphReadout requires packet.retrieved.")

        lines: list[str] = []
        source_ids: list[str] = []
        graph_item_count = 0
        for record in packet.retrieved.items:
            source_ids.append(record.record_id)
            graph = graph_metadata_from_record(record)
            parts = [f"[{record.layer}] {record.text}"]
            if graph["entities"]:
                graph_item_count += 1
                parts.append(f"entities={', '.join(graph['entities'])}")
            if self.include_links and graph["links"]:
                parts.append(f"links={', '.join(graph['links'])}")
            elif self.include_links:
                parts.append("links=<none>")
            lines.append(" | ".join(parts))

        readout = Readout(
            text="\n".join(lines),
            source_ids=source_ids,
            metadata={
                "item_count": len(source_ids),
                "graph_item_count": graph_item_count,
                "format": "graph",
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "graph_item_count": graph_item_count,
        }
        return replace(packet, readout=readout, trace=trace), store


BASELINE_SLOT: Final[str] = "readout"
BASELINE_CLASSES: Final[tuple[type[ReadoutModule], ...]] = (
    ConcatenateReadout,
    BulletListReadout,
    GroupedByLayerReadout,
    JSONReadout,
    GraphReadout,
)
