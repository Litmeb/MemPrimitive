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

from ..contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_NOTE_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
    TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
)
from ..core import MemoryRecord, MemoryStore
from ._llm_function_tools import WriteToolCallContext, WriteToolResult, WriteToolSpec, find_record_by_id
from ._runtime import Runtime, get_runtime


DEFAULT_NOTE_NAMESPACE: Final[str] = "note"
DEFAULT_EMBEDDING_VERSION: Final[str] = "content_context_keywords_tags_v2"
DEFAULT_CATEGORY: Final[str] = "Uncategorized"
AMEM_STRENGTHEN_LINKS_TOOL: Final[str] = "AMEM_STRENGTHEN_LINKS"
AMEM_UPDATE_NEIGHBOR_TOOL: Final[str] = "AMEM_UPDATE_NEIGHBOR"


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
    runtime: Runtime | None = None,
    preserve_graph: bool = True,
) -> MemoryRecord:
    """Rewrite a note-bearing record while preserving non-note metadata."""

    repaired = repair_note_payload(payload, fallback_content=record.text, default_category=default_category)
    representation = representation_from_note_payload(repaired, embedding_version=embedding_version)
    engine = runtime or get_runtime()
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
        score = Runtime.cosine_similarity(query_embedding, record.embedding)
        scored.append((float(score), record))
    scored.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].record_id))
    return scored[:top_k]


def build_amem_evolution_tools(
    *,
    target_layer: str,
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
    max_links_per_record: int = 4,
    default_category: str = DEFAULT_CATEGORY,
) -> list[WriteToolSpec]:
    return [
        build_amem_strengthen_links_tool(
            target_layer=target_layer,
            note_namespace=note_namespace,
            max_links_per_record=max_links_per_record,
            default_category=default_category,
        ),
        build_amem_update_neighbor_tool(
            target_layer=target_layer,
            note_namespace=note_namespace,
            default_category=default_category,
        ),
    ]


def build_amem_strengthen_links_tool(
    *,
    target_layer: str,
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
    max_links_per_record: int = 4,
    default_category: str = DEFAULT_CATEGORY,
) -> WriteToolSpec:
    if max_links_per_record <= 0:
        raise ValueError("max_links_per_record must be positive.")
    normalized_layer = str(target_layer).strip()
    if not normalized_layer:
        raise ValueError("target_layer must be a non-empty string.")

    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        current_record = _require_single_selected_record(context, tool_name=AMEM_STRENGTHEN_LINKS_TOOL)
        if current_record.layer != normalized_layer:
            raise ValueError(
                f"{AMEM_STRENGTHEN_LINKS_TOOL} requires selected record layer {normalized_layer!r}, "
                f"got {current_record.layer!r}."
            )
        if context.store.layer_shape(current_record.layer) != "Graph":
            raise ValueError(f"{AMEM_STRENGTHEN_LINKS_TOOL} requires target layer {current_record.layer!r} to be Graph.")
        record = find_record_by_id(
            context.store,
            str(arguments.get("record_id", "")).strip(),
            visible_records=context.visible_records,
            restricted=True,
        )
        if record.record_id != current_record.record_id:
            raise ValueError(f"{AMEM_STRENGTHEN_LINKS_TOOL} may only modify the current selected record.")
        requested_neighbor_ids = _coerce_visible_neighbor_ids(
            context,
            neighbor_record_ids=arguments.get("neighbor_record_ids"),
            current_record_id=current_record.record_id,
            tool_name=AMEM_STRENGTHEN_LINKS_TOOL,
        )
        graph = record.metadata.get("graph", {})
        if not isinstance(graph, dict):
            graph = {}
        previous_links = [str(value).strip() for value in graph.get("links", []) if str(value).strip()]
        merged_links = list(dict.fromkeys([*previous_links, *requested_neighbor_ids]))
        current_links = merged_links[:max_links_per_record]
        normalized_tags = _normalize_optional_tags(arguments.get("tags"))
        note_payload = note_payload_from_record(
            record,
            note_namespace=note_namespace,
            default_category=default_category,
        )
        updated_note_payload = dict(note_payload)
        if normalized_tags is not None:
            updated_note_payload["tags"] = normalized_tags
        updated = replace(
            record,
            metadata={
                **record.metadata,
                note_namespace: updated_note_payload,
                "graph": {
                    **graph,
                    "links": current_links,
                    "link_count": len(current_links),
                },
            },
        )
        context.store.replace_record(record.layer, record.record_id, updated)
        strengthened_links = [link for link in current_links if link in requested_neighbor_ids and link not in previous_links]
        return WriteToolResult(
            effects=[
                {
                    "action": "amem_strengthen_links",
                    "effect_type": "link_strengthening",
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "status": "applied",
                    "previous_links": previous_links,
                    "requested_neighbor_record_ids": list(requested_neighbor_ids),
                    "current_links": current_links,
                    "strengthened_links": strengthened_links,
                    "truncated": len(merged_links) > len(current_links),
                    "updated_tags": list(normalized_tags) if normalized_tags is not None else None,
                }
            ],
            store=context.store,
        )

    return WriteToolSpec(
        name=AMEM_STRENGTHEN_LINKS_TOOL,
        description=(
            "Strengthen outgoing links on the current A-MEM note and optionally patch the current note tags."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "neighbor_record_ids": {"type": "array"},
                "tags": {"type": "array"},
            },
            "required": ["record_id", "neighbor_record_ids"],
            "additionalProperties": False,
        },
        executor=_execute,
        requires_contracts=(
            RECORD_NOTE_PAYLOAD_CONTRACT,
            TOPOLOGY_GRAPH_LAYER_CONTRACT,
            TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
        ),
        produces_contracts=(RECORD_GRAPH_LINKS_CONTRACT, RECORD_NOTE_PAYLOAD_CONTRACT),
    )


def build_amem_update_neighbor_tool(
    *,
    target_layer: str,
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
    default_category: str = DEFAULT_CATEGORY,
) -> WriteToolSpec:
    normalized_layer = str(target_layer).strip()
    if not normalized_layer:
        raise ValueError("target_layer must be a non-empty string.")

    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        current_record = _require_single_selected_record(context, tool_name=AMEM_UPDATE_NEIGHBOR_TOOL)
        record = find_record_by_id(
            context.store,
            str(arguments.get("record_id", "")).strip(),
            visible_records=context.visible_records,
            restricted=True,
        )
        if record.record_id == current_record.record_id:
            raise ValueError(f"{AMEM_UPDATE_NEIGHBOR_TOOL} cannot modify the current selected record.")
        if record.layer != normalized_layer:
            raise ValueError(f"{AMEM_UPDATE_NEIGHBOR_TOOL} requires target layer {normalized_layer!r}.")
        if context.store.layer_shape(record.layer) != "Graph":
            raise ValueError(f"{AMEM_UPDATE_NEIGHBOR_TOOL} requires target layer {record.layer!r} to be Graph.")
        neighbor_payload = note_payload_from_record(
            record,
            note_namespace=note_namespace,
            default_category=default_category,
        )
        updated_fields: list[str] = []
        updated_note_payload = dict(neighbor_payload)
        if "context" in arguments:
            normalized_context = normalize_text(arguments.get("context"))
            if normalized_context:
                updated_note_payload["context"] = normalized_context
                updated_fields.append("context")
        normalized_tags = _normalize_optional_tags(arguments.get("tags"))
        if normalized_tags is not None:
            updated_note_payload["tags"] = normalized_tags
            updated_fields.append("tags")
        updated = replace(
            record,
            metadata={
                **record.metadata,
                note_namespace: updated_note_payload,
            },
        )
        context.store.replace_record(record.layer, record.record_id, updated)
        return WriteToolResult(
            effects=[
                {
                    "action": "amem_update_neighbor",
                    "effect_type": "neighbor_context_update",
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "status": "applied",
                    "updated_fields": updated_fields,
                }
            ],
            store=context.store,
        )

    return WriteToolSpec(
        name=AMEM_UPDATE_NEIGHBOR_TOOL,
        description=(
            "Update one visible neighbor A-MEM note by patching only its context and/or tags."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "context": {"type": "string"},
                "tags": {"type": "array"},
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
        executor=_execute,
        requires_contracts=(RECORD_NOTE_PAYLOAD_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT),
        produces_contracts=(RECORD_NOTE_PAYLOAD_CONTRACT,),
    )


def _require_single_selected_record(context: WriteToolCallContext, *, tool_name: str) -> MemoryRecord:
    if len(context.selected_records) != 1:
        raise ValueError(f"{tool_name} requires exactly one selected current record.")
    return context.selected_records[0]


def _coerce_visible_neighbor_ids(
    context: WriteToolCallContext,
    *,
    neighbor_record_ids: Any,
    current_record_id: str,
    tool_name: str,
) -> list[str]:
    if not isinstance(neighbor_record_ids, list):
        raise ValueError(f"{tool_name} neighbor_record_ids must be an array of strings.")
    visible_ids = {record.record_id for record in context.visible_records}
    normalized_ids: list[str] = []
    for value in neighbor_record_ids:
        if not isinstance(value, str):
            raise ValueError(f"{tool_name} neighbor_record_ids must be an array of strings.")
        normalized = value.strip()
        if not normalized:
            continue
        if normalized == current_record_id:
            raise ValueError(f"{tool_name} neighbor_record_ids must not include the current record_id.")
        if normalized not in visible_ids:
            raise KeyError(f"Record {normalized!r} is not in the current evolution candidate set.")
        normalized_ids.append(normalized)
    return list(dict.fromkeys(normalized_ids))


def _normalize_optional_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    return dedupe_text_list(value)


