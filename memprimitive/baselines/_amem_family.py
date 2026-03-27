"""Shared A-MEM-style note helpers for baseline slot modules.

These helpers intentionally stay limited to the cross-slot concerns that would
otherwise be duplicated across representation, retrieval, memory evolution, and
readout modules:

- note-payload schema repair
- retrieval-oriented embedding text construction
- record<->note payload conversion
- embedding-based candidate collection for graph-note pipelines
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

from ..core import MemoryRecord, MemoryStore
from ..classic_modules._runtime import ClassicRuntime, get_classic_runtime


DEFAULT_NOTE_NAMESPACE: Final[str] = "note"
DEFAULT_EMBEDDING_VERSION: Final[str] = "content_context_keywords_tags_v2"
DEFAULT_CATEGORY: Final[str] = "Uncategorized"


def normalize_text(value: Any) -> str:
    """Return whitespace-collapsed text for note-schema fields."""

    return " ".join(str(value or "").strip().split())


def dedupe_text_list(values: Any) -> list[str]:
    """Return stable unique strings from a list-like or comma-delimited payload."""

    if isinstance(values, str):
        raw_values = [part.strip() for part in values.split(",")]
    elif isinstance(values, list):
        raw_values = [str(item).strip() for item in values]
    else:
        raw_values = []
    return list(dict.fromkeys(item for item in raw_values if item))


def coerce_llm_mapping(value: Any, *, list_key: str | None = None) -> dict[str, Any]:
    """Repair common LLM wrapper/list outputs into a plain mapping."""

    if isinstance(value, dict):
        for wrapper_key in ("result", "output", "response", "data"):
            nested = value.get(wrapper_key)
            if isinstance(nested, dict):
                return dict(nested)
            if isinstance(nested, list) and list_key is not None:
                return {list_key: nested}
        return dict(value)
    if isinstance(value, list) and list_key is not None:
        return {list_key: value}
    return {}


def coerce_index_list(value: Any) -> list[int]:
    """Repair an LLM-provided index list into validated integers."""

    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []

    repaired: list[int] = []
    for item in raw_items:
        try:
            repaired.append(int(item))
        except (TypeError, ValueError):
            continue
    return repaired


def normalize_attributes(value: Any) -> dict[str, str]:
    """Return a normalized ``dict[str, str]`` for note attributes."""

    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        normalized_key = normalize_text(key)
        normalized_value = normalize_text(raw)
        if normalized_key and normalized_value:
            out[normalized_key] = normalized_value
    return out


def repair_note_payload(
    payload: dict[str, Any] | None,
    *,
    fallback_content: str,
    default_category: str = DEFAULT_CATEGORY,
) -> dict[str, Any]:
    """Repair a note payload into the canonical enriched-note schema."""

    raw = payload if isinstance(payload, dict) else {}
    content = normalize_text(raw.get("content") or fallback_content)
    note_text = normalize_text(raw.get("note_text") or content)
    context = normalize_text(raw.get("context") or content)
    keywords = dedupe_text_list(raw.get("keywords"))
    tags = dedupe_text_list(raw.get("tags"))
    category = normalize_text(raw.get("category") or default_category)
    attributes = normalize_attributes(raw.get("attributes"))
    if not tags and keywords:
        tags = keywords[:3]
    return {
        "content": content,
        "note_text": note_text,
        "context": context,
        "keywords": keywords,
        "tags": tags,
        "category": category,
        "attributes": attributes,
    }


def build_enhanced_embedding_text(
    *,
    content: str,
    context: str,
    keywords: list[str],
    tags: list[str],
) -> str:
    """Build the retrieval-oriented composite text used for note embeddings."""

    parts = [
        f"content: {content}",
        f"context: {context}",
        f"keywords: {', '.join(keywords)}",
        f"tags: {', '.join(tags)}",
    ]
    return " | ".join(part for part in parts if normalize_text(part))


def representation_from_note_payload(
    payload: dict[str, Any],
    *,
    embedding_version: str = DEFAULT_EMBEDDING_VERSION,
) -> dict[str, Any]:
    """Project a canonical note payload into ``metadata['representation']``."""

    enhanced_embedding_text = build_enhanced_embedding_text(
        content=payload["content"],
        context=payload["context"],
        keywords=payload["keywords"],
        tags=payload["tags"],
    )
    return {
        "text": payload["content"],
        "normalized_text": payload["content"].casefold(),
        "note_text": payload["note_text"],
        "context": payload["context"],
        "keywords": list(payload["keywords"]),
        "tags": list(payload["tags"]),
        "category": payload["category"],
        "attributes": dict(payload["attributes"]),
        "enhanced_embedding_text": enhanced_embedding_text,
        "embedding_version": embedding_version,
    }


def note_payload_from_record(
    record: MemoryRecord,
    *,
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
    default_category: str = DEFAULT_CATEGORY,
) -> dict[str, Any]:
    """Read and repair the enriched-note payload stored on a record."""

    raw = record.metadata.get(note_namespace, {})
    payload = dict(raw) if isinstance(raw, dict) else {}
    payload.setdefault("content", record.text)
    payload.setdefault("note_text", record.text)
    representation = record.metadata.get("representation", {})
    if isinstance(representation, dict):
        for key in (
            "context",
            "keywords",
            "tags",
            "category",
            "attributes",
            "enhanced_embedding_text",
            "embedding_version",
        ):
            if key in representation and key not in payload:
                payload[key] = representation[key]
    return repair_note_payload(payload, fallback_content=record.text, default_category=default_category)


def rewrite_record_from_note_payload(
    store: MemoryStore,
    *,
    layer: str,
    record: MemoryRecord,
    payload: dict[str, Any],
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
    default_category: str = DEFAULT_CATEGORY,
    embedding_version: str = DEFAULT_EMBEDDING_VERSION,
    runtime: ClassicRuntime | None = None,
    preserve_graph: bool = True,
) -> MemoryRecord:
    """Rewrite a note-bearing record while preserving non-note metadata."""

    repaired = repair_note_payload(payload, fallback_content=record.text, default_category=default_category)
    representation = representation_from_note_payload(repaired, embedding_version=embedding_version)
    engine = runtime or get_classic_runtime()
    updated = replace(
        record,
        text=repaired["content"],
        embedding=engine.embed(representation["enhanced_embedding_text"]),
        metadata={
            **record.metadata,
            note_namespace: {
                **repaired,
                "enhanced_embedding_text": representation["enhanced_embedding_text"],
                "embedding_version": embedding_version,
            },
            "representation": representation,
            **({"graph": record.metadata.get("graph", {})} if preserve_graph else {}),
        },
    )
    store.replace_record(layer, record.record_id, updated)
    return updated


def stringify_note_candidates(
    records: list[MemoryRecord],
    *,
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
) -> str:
    """Render candidate note records into a trace-friendly text block."""

    lines: list[str] = []
    for index, record in enumerate(records):
        payload = note_payload_from_record(record, note_namespace=note_namespace)
        lines.append(
            "\n".join(
                [
                    f"memory index:{index}",
                    f"talk start time:{record.timestamp}",
                    f"memory content: {payload['content']}",
                    f"memory context: {payload['context']}",
                    f"memory keywords: {payload['keywords']}",
                    f"memory tags: {payload['tags']}",
                ]
            )
        )
    return "\n".join(lines)


def retrieve_candidates_by_embedding(
    *,
    store: MemoryStore,
    layer: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[float, MemoryRecord]]:
    """Return embedding-scored graph-note candidates sorted best-first."""

    scored: list[tuple[float, MemoryRecord]] = []
    for record in store.iter_records(layer):
        score = ClassicRuntime.cosine_similarity(query_embedding, record.embedding)
        scored.append((float(score), record))
    scored.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].record_id))
    return scored[:top_k]


def record_links(record: MemoryRecord) -> list[str]:
    """Return stable linked record ids from ``metadata['graph']['links']``."""

    graph = record.metadata.get("graph", {})
    if not isinstance(graph, dict):
        return []
    links = graph.get("links", [])
    if not isinstance(links, list):
        return []
    return [str(item) for item in links if str(item).strip()]


def collect_neighbor_candidates(
    *,
    store: MemoryStore,
    layer: str,
    seed_records: list[MemoryRecord],
    neighbor_expansion_k: int,
) -> list[MemoryRecord]:
    """Expand one hop from seed records through graph links."""

    if neighbor_expansion_k <= 0:
        return []
    selected: list[MemoryRecord] = []
    seen: set[str] = set()
    for seed in seed_records:
        for neighbor_id in record_links(seed):
            if neighbor_id in seen:
                continue
            neighbor = next((record for record in store.iter_records(layer) if record.record_id == neighbor_id), None)
            if neighbor is None:
                continue
            seen.add(neighbor_id)
            selected.append(neighbor)
            if len(selected) >= neighbor_expansion_k:
                return selected
    return selected


def merge_records_by_id(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Return deduplicated records while preserving first-seen order."""

    merged: list[MemoryRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        merged.append(record)
    return merged
