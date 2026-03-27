"""Shared graph-family helpers for stage-1 baseline modules.

These helpers intentionally stay small: they normalize the baseline graph
metadata contract and provide safe record rewrite utilities reused by
organization, retrieval, evolution, and readout graph primitives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable

from ..core import MemoryRecord, MemoryUnit


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _normalize_triples(values: Iterable[Any]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            triples.append((str(value[0]), str(value[1]), str(value[2])))
    return triples


def normalize_graph_metadata(value: Any, *, layer: str | None = None) -> dict[str, Any]:
    """Return a stable graph metadata dict used by stage-1 baseline modules."""

    raw = value if isinstance(value, dict) else {}
    known_keys = {
        "layer",
        "shape",
        "entities",
        "triples",
        "links",
        "node_count",
        "link_count",
        "last_linked_at",
        "link_history",
    }
    links = _dedupe_strings(raw.get("links", []))
    entities = _dedupe_strings(raw.get("entities", []))
    triples = _normalize_triples(raw.get("triples", []))
    history = [dict(item) for item in raw.get("link_history", []) if isinstance(item, dict)]

    normalized_layer = str(raw.get("layer") or layer or "").strip()
    node_count = raw.get("node_count")
    if not isinstance(node_count, int) or node_count < 0:
        node_count = max(len(entities), 1 if triples else 0)

    return {
        "layer": normalized_layer,
        "shape": str(raw.get("shape") or "node"),
        "entities": entities,
        "triples": triples,
        "links": links,
        "node_count": node_count,
        "link_count": len(links),
        "last_linked_at": raw.get("last_linked_at"),
        "link_history": history,
        **{key: raw[key] for key in raw if key not in known_keys},
    }


def graph_metadata_for_unit(unit: MemoryUnit, *, layer: str) -> dict[str, Any]:
    """Build the normalized graph metadata payload for a newly appended unit."""

    return normalize_graph_metadata(
        {
            "layer": layer,
            "entities": unit.entities,
            "triples": unit.triples,
            "links": [],
            "link_history": [],
        },
        layer=layer,
    )


def graph_metadata_from_record(record: MemoryRecord) -> dict[str, Any]:
    """Read and normalize graph metadata from a stored record."""

    return normalize_graph_metadata(record.metadata.get("graph"), layer=record.layer)


def rewrite_graph_record(
    record: MemoryRecord,
    *,
    linked_record_ids: Iterable[str] | None = None,
    link_trace_entry: dict[str, Any] | None = None,
    extra_graph_fields: dict[str, Any] | None = None,
) -> MemoryRecord:
    """Return a new record with safely merged graph metadata updates."""

    graph = graph_metadata_from_record(record)
    if linked_record_ids is not None:
        graph["links"] = _dedupe_strings([*graph["links"], *linked_record_ids])
        graph["link_count"] = len(graph["links"])
        graph["last_linked_at"] = _utc_now_iso()
    if link_trace_entry is not None:
        graph["link_history"] = [*graph["link_history"], dict(link_trace_entry)]
    if extra_graph_fields:
        graph.update(extra_graph_fields)
    graph = normalize_graph_metadata(graph, layer=record.layer)
    return MemoryRecord(
        record_id=record.record_id,
        unit_id=record.unit_id,
        layer=record.layer,
        text=record.text,
        timestamp=record.timestamp,
        embedding=record.embedding,
        metadata={
            **record.metadata,
            "graph": graph,
        },
    )
