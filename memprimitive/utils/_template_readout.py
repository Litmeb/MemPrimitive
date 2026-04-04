"""Template-oriented readout helpers for safe context binding and rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Callable

from ..core import MemoryRecord, Packet
from ._amem_family import DEFAULT_CATEGORY, DEFAULT_NOTE_NAMESPACE, note_payload_from_record
from ._graph_family import graph_metadata_from_record
from ._template import (
    ResolutionState,
    TemplateExpression,
    collect_template_references,
    parse_expression,
    project_packet_runtime_for_template,
    render_simple_template,
    resolve_expression,
    utc_now_iso,
)


@dataclass(slots=True)
class StructuredGroup:
    group_id: str
    kind: str
    label: str
    record_ids: list[str] = field(default_factory=list)
    unit_ids: list[str] = field(default_factory=list)
    layer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredRelation:
    relation: str
    from_id: str
    to_id: str
    record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredViewBundle:
    groups: list[StructuredGroup] = field(default_factory=list)
    relations: list[StructuredRelation] = field(default_factory=list)
    views: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockRenderTrace:
    block_id: str
    title: str
    rendered: bool
    matched_variables: list[str] = field(default_factory=list)
    missing_variables: list[str] = field(default_factory=list)
    used_record_ids: list[str] = field(default_factory=list)
    used_group_ids: list[str] = field(default_factory=list)
    child_count: int = 0


@dataclass(slots=True)
class ReadoutResolutionState(ResolutionState):
    block_traces: list[BlockRenderTrace] = field(default_factory=list)


def template_mode(*, simple_template: str | None, structured_template: dict[str, Any] | list[Any] | None) -> str:
    simple_present = simple_template is not None
    structured_present = structured_template is not None
    if simple_present == structured_present:
        raise ValueError("TemplateReadout requires exactly one of simple_template or structured_template.")
    return "simple" if simple_present else "structured"


def build_render_context(
    packet: Packet,
    *,
    note_namespace: str = DEFAULT_NOTE_NAMESPACE,
    default_category: str = DEFAULT_CATEGORY,
    runtime_now_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    query = packet.query
    retrieved = packet.retrieved
    score_items = list(retrieved.scores) if retrieved is not None else []
    score_by_record_id: dict[str, dict[str, Any]] = {}
    for score in score_items:
        if isinstance(score, dict):
            record_id = str(score.get("record_id", "")).strip()
            if record_id:
                score_by_record_id[record_id] = dict(score)

    projected_items: list[dict[str, Any]] = []
    if retrieved is not None:
        for item_index, record in enumerate(retrieved.items):
            projected_items.append(
                project_record(
                    record,
                    score=score_by_record_id.get(record.record_id, {}),
                    item_index=item_index,
                    note_namespace=note_namespace,
                    default_category=default_category,
                )
            )

    structured = structure_retrieved_items(projected_items)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    by_record_id: dict[str, dict[str, Any]] = {}
    by_unit_id: dict[str, list[dict[str, Any]]] = {}
    for item in projected_items:
        by_layer.setdefault(str(item.get("layer", "")), []).append(item)
        record_id = str(item.get("record_id", "")).strip()
        if record_id:
            by_record_id[record_id] = item
        unit_id = str(item.get("unit_id", "")).strip()
        if unit_id:
            by_unit_id.setdefault(unit_id, []).append(item)

    runtime = project_packet_runtime_for_template(
        packet,
        now=runtime_now_factory() if runtime_now_factory is not None else utc_now_iso(),
    )

    retrieval_trace = dict(retrieved.trace) if retrieved is not None else {}
    return {
        "query": {
            "text": query.text if query is not None else "",
            "timestamp": query.timestamp if query is not None else "",
            "metadata": dict(query.metadata) if query is not None else {},
        },
        "runtime": runtime,
        "retrieved": {
            "items": projected_items,
            "by_layer": by_layer,
            "by_record_id": by_record_id,
            "by_unit_id": by_unit_id,
            "groups": [asdict(group) for group in structured.groups],
            "relations": [asdict(relation) for relation in structured.relations],
            "views": structured.views,
            "by_group": structured.views.get("by_group", {}),
        },
        "scores": {
            "items": score_items,
            "by_record_id": score_by_record_id,
        },
        "trace": {
            "retrieval": retrieval_trace,
            "packet": dict(packet.trace),
        },
    }


def project_record(
    record: MemoryRecord,
    *,
    score: dict[str, Any],
    item_index: int,
    note_namespace: str,
    default_category: str,
) -> dict[str, Any]:
    note = note_payload_from_record(
        record,
        note_namespace=note_namespace,
        default_category=default_category,
    )
    from ._template import project_record_for_template

    return project_record_for_template(
        record,
        score=score,
        item_index=item_index,
        extra={
            "note": note,
            "graph": graph_metadata_from_record(record),
        },
    )


def structure_retrieved_items(items: list[dict[str, Any]]) -> StructuredViewBundle:
    groups: dict[str, StructuredGroup] = {}
    relations: list[StructuredRelation] = []

    def _upsert_group(
        *,
        group_id: str,
        kind: str,
        label: str,
        record_id: str,
        unit_id: str,
        layer: str | None,
        metadata: dict[str, Any],
    ) -> None:
        group = groups.get(group_id)
        if group is None:
            group = StructuredGroup(group_id=group_id, kind=kind, label=label, layer=layer, metadata=dict(metadata))
            groups[group_id] = group
        if record_id and record_id not in group.record_ids:
            group.record_ids.append(record_id)
        if unit_id and unit_id not in group.unit_ids:
            group.unit_ids.append(unit_id)

    for item in items:
        record_id = str(item.get("record_id", "")).strip()
        unit_id = str(item.get("unit_id", "")).strip()
        layer = str(item.get("layer", "")).strip() or None
        metadata = item.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        hierarchical = item.get("hierarchical", {})
        hierarchical = hierarchical if isinstance(hierarchical, dict) else {}
        group_key = hierarchical.get("group_key", {})
        if isinstance(group_key, dict) and group_key:
            token = json.dumps(group_key, ensure_ascii=False, sort_keys=True, default=str)
            group_id = f"group:hierarchical:{token}"
            label = ", ".join(f"{key}={group_key[key]}" for key in sorted(group_key))
            _upsert_group(
                group_id=group_id,
                kind="hierarchical",
                label=label or group_id,
                record_id=record_id,
                unit_id=unit_id,
                layer=layer,
                metadata={"group_key": dict(group_key), "hierarchical": dict(hierarchical)},
            )
            relations.append(
                StructuredRelation(
                    relation="belongs_to",
                    from_id=record_id,
                    to_id=group_id,
                    record_ids=[record_id],
                    metadata={"kind": "hierarchical"},
                )
            )
        for field_name, kind in (("session_id", "session"), ("subgoal_id", "subgoal")):
            raw_value = metadata.get(field_name)
            if isinstance(raw_value, str) and raw_value.strip():
                group_id = f"group:{kind}:{raw_value.strip()}"
                _upsert_group(
                    group_id=group_id,
                    kind=kind,
                    label=f"{field_name}={raw_value.strip()}",
                    record_id=record_id,
                    unit_id=unit_id,
                    layer=layer,
                    metadata={field_name: raw_value.strip()},
                )
                relations.append(
                    StructuredRelation(
                        relation="belongs_to",
                        from_id=record_id,
                        to_id=group_id,
                        record_ids=[record_id],
                        metadata={"kind": kind},
                    )
                )

        source_record_ids = hierarchical.get("source_record_ids", [])
        clean_source_ids = [str(value).strip() for value in source_record_ids if str(value).strip()] if isinstance(source_record_ids, list) else []
        if clean_source_ids:
            relation_name = "summarizes" if _looks_like_summary(item) else "derived_from"
            for source_record_id in clean_source_ids:
                relations.append(
                    StructuredRelation(
                        relation=relation_name,
                        from_id=record_id,
                        to_id=source_record_id,
                        record_ids=[record_id, source_record_id],
                        metadata={"layer": layer, "group_key": dict(group_key) if isinstance(group_key, dict) else {}},
                    )
                )

    relation_dicts = [asdict(relation) for relation in relations]
    by_group: dict[str, list[dict[str, Any]]] = {}
    item_by_record_id = {
        str(item.get("record_id", "")).strip(): item
        for item in items
        if isinstance(item, dict) and str(item.get("record_id", "")).strip()
    }
    for relation in relations:
        if relation.relation != "belongs_to":
            continue
        item = item_by_record_id.get(relation.from_id)
        if item is not None:
            by_group.setdefault(relation.to_id, []).append(item)

    summary_with_sources: list[dict[str, Any]] = []
    for item in items:
        hierarchical = item.get("hierarchical", {})
        hierarchical = hierarchical if isinstance(hierarchical, dict) else {}
        source_record_ids = hierarchical.get("source_record_ids", [])
        if not isinstance(source_record_ids, list) or not source_record_ids:
            continue
        source_items = [item_by_record_id[source_id] for source_id in source_record_ids if source_id in item_by_record_id]
        group_key = hierarchical.get("group_key", {})
        group_id = ""
        if isinstance(group_key, dict) and group_key:
            token = json.dumps(group_key, ensure_ascii=False, sort_keys=True, default=str)
            group_id = f"group:hierarchical:{token}"
        summary_with_sources.append(
            {
                "summary": item,
                "sources": source_items,
                "group_key": group_key,
                "group_id": group_id,
            }
        )

    views = {
        "by_group": by_group,
        "summary_with_sources": summary_with_sources,
        "by_relation": _group_relation_dicts(relation_dicts),
        "by_subgoal": {group_id: members for group_id, members in by_group.items() if group_id.startswith("group:subgoal:")},
    }
    return StructuredViewBundle(groups=list(groups.values()), relations=relations, views=views)


def _group_relation_dicts(relations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for relation in relations:
        grouped.setdefault(str(relation.get("relation", "")), []).append(relation)
    return grouped


def _looks_like_summary(item: dict[str, Any]) -> bool:
    layer = str(item.get("layer", "")).casefold()
    metadata = item.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    representation = item.get("representation", {})
    representation = representation if isinstance(representation, dict) else {}
    unit_type = str(metadata.get("unit_type", "")).casefold()
    return "summary" in layer or unit_type == "summary" or "summary" in representation


def render_structured_template(
    template: dict[str, Any] | list[Any],
    context: dict[str, Any],
    state: ReadoutResolutionState,
    *,
    missing_value: str = "",
) -> str:
    blocks = normalize_structured_blocks(template)
    separator = "\n\n"
    if isinstance(template, dict):
        raw_separator = template.get("separator")
        if isinstance(raw_separator, str):
            separator = raw_separator
    rendered = [render_block(block, context, state, missing_value=missing_value) for block in blocks]
    return separator.join(chunk for chunk in rendered if chunk.strip())


def normalize_structured_blocks(template: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(template, list):
        return [dict(block) for block in template if isinstance(block, dict)]
    if isinstance(template, dict):
        blocks = template.get("blocks")
        if isinstance(blocks, list):
            return [dict(block) for block in blocks if isinstance(block, dict)]
        return [dict(template)]
    raise ValueError("structured_template must be a dict or list of dict blocks.")


def render_block(
    block: dict[str, Any],
    context: dict[str, Any],
    state: ReadoutResolutionState,
    *,
    missing_value: str = "",
) -> str:
    block_id = str(block.get("id") or f"block-{len(state.block_traces) + 1}")
    title_template = str(block.get("title") or "")
    child_count = len(block.get("children", [])) if isinstance(block.get("children"), list) else 0
    start_index = len(state.resolutions)
    condition = block.get("condition")
    rendered = True
    if isinstance(condition, str) and condition.strip():
        resolved_condition = resolve_expression(parse_expression(condition), context, state)
        rendered = bool(resolved_condition.value)
    if not rendered:
        state.block_traces.append(
            _block_trace_from_state(
                block_id=block_id,
                title=title_template,
                rendered=False,
                child_count=child_count,
                state=state,
                start_index=start_index,
            )
        )
        return ""

    title_text = render_simple_template(title_template, context, state, missing_value=missing_value) if title_template else ""
    separator = str(block.get("separator", "\n"))
    body_parts: list[str] = []
    template_text = block.get("template")
    if isinstance(template_text, str) and template_text:
        body_parts.append(render_simple_template(template_text, context, state, missing_value=missing_value))

    children = [dict(child) for child in block.get("children", []) if isinstance(child, dict)]
    repeat_over = block.get("repeat_over")
    item_template = block.get("item_template")
    if isinstance(repeat_over, str) and repeat_over.strip():
        resolved_items = resolve_expression(parse_expression(repeat_over), context, state)
        iterable = list(resolved_items.value) if isinstance(resolved_items.value, list) else []
        repeated_parts: list[str] = []
        for index, item in enumerate(iterable):
            iteration_context = {
                **context,
                "item": item,
                "loop": {
                    "index": index,
                    "first": index == 0,
                    "last": index == len(iterable) - 1,
                },
            }
            iteration_parts: list[str] = []
            if isinstance(item_template, str) and item_template:
                iteration_parts.append(
                    render_simple_template(item_template, iteration_context, state, missing_value=missing_value)
                )
            for child in children:
                child_text = render_block(child, iteration_context, state, missing_value=missing_value)
                if child_text.strip():
                    iteration_parts.append(child_text)
            iteration_text = separator.join(part for part in iteration_parts if part.strip())
            if iteration_text.strip():
                repeated_parts.append(iteration_text)
        if repeated_parts:
            body_parts.append(separator.join(repeated_parts))
    else:
        for child in children:
            child_text = render_block(child, context, state, missing_value=missing_value)
            if child_text.strip():
                body_parts.append(child_text)

    body = separator.join(part for part in body_parts if part.strip())
    final_text = "\n".join(chunk for chunk in (title_text, body) if chunk.strip())
    state.block_traces.append(
        _block_trace_from_state(
            block_id=block_id,
            title=title_text or title_template,
            rendered=bool(final_text.strip()) or bool(title_text.strip()) or bool(body.strip()),
            child_count=child_count,
            state=state,
            start_index=start_index,
        )
    )
    return final_text


def _block_trace_from_state(
    *,
    block_id: str,
    title: str,
    rendered: bool,
    child_count: int,
    state: ReadoutResolutionState,
    start_index: int,
) -> BlockRenderTrace:
    resolutions = state.resolutions[start_index:]
    return BlockRenderTrace(
        block_id=block_id,
        title=title,
        rendered=rendered,
        matched_variables=[item.expression for item in resolutions if item.status == "resolved"],
        missing_variables=[item.expression for item in resolutions if item.status == "missing"],
        used_record_ids=_stable_unique(record_id for item in resolutions for record_id in item.used_record_ids),
        used_group_ids=_stable_unique(group_id for item in resolutions for group_id in item.used_group_ids),
        child_count=child_count,
    )


def metadata_from_state(
    *,
    template_mode_name: str,
    declared_variables: list[str],
    context: dict[str, Any],
    state: ResolutionState,
) -> dict[str, Any]:
    missing_variables = _stable_unique(
        resolved.expression for resolved in state.resolutions if resolved.status == "missing"
    )
    used_record_ids = _stable_unique(
        record_id for resolved in state.resolutions for record_id in resolved.used_record_ids
    )
    used_group_ids = _stable_unique(
        group_id for resolved in state.resolutions for group_id in resolved.used_group_ids
    )
    available_views = sorted(key for key in context.get("retrieved", {}).get("views", {}).keys())
    return {
        "template_mode": template_mode_name,
        "declared_variables": declared_variables,
        "resolved_variables": [asdict(item) for item in state.resolutions],
        "missing_variables": missing_variables,
        "used_group_ids": used_group_ids,
        "used_record_ids": used_record_ids,
        "block_trace": [asdict(trace) for trace in state.block_traces],
        "available_views": available_views,
        "filter_trace": list(state.filter_trace),
        "render_trace": {
            "declared_variable_count": len(declared_variables),
            "resolved_variable_count": len(state.resolutions),
            "missing_variable_count": len(missing_variables),
        },
        "structuring_summary": {
            "group_count": len(context.get("retrieved", {}).get("groups", [])),
            "relation_count": len(context.get("retrieved", {}).get("relations", [])),
        },
    }

def _stable_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
