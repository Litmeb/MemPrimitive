"""Template-oriented readout helpers for safe context binding and rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import ast
import json
import re
from typing import Any, Callable

from ..core import MemoryRecord, Packet
from ._amem_family import DEFAULT_CATEGORY, DEFAULT_NOTE_NAMESPACE, note_payload_from_record
from ._graph_family import graph_metadata_from_record

_EXPR_PATTERN = re.compile(r"{{\s*(.+?)\s*}}")
_MISSING = object()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TemplateExpression:
    raw: str
    path: str
    filters: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ResolvedVariable:
    expression: str
    value: Any
    status: str
    source_path: str
    used_record_ids: list[str] = field(default_factory=list)
    used_group_ids: list[str] = field(default_factory=list)


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
class ResolutionState:
    resolutions: list[ResolvedVariable] = field(default_factory=list)
    block_traces: list[BlockRenderTrace] = field(default_factory=list)
    filter_trace: list[dict[str, Any]] = field(default_factory=list)


def template_mode(*, simple_template: str | None, structured_template: dict[str, Any] | list[Any] | None) -> str:
    simple_present = simple_template is not None
    structured_present = structured_template is not None
    if simple_present == structured_present:
        raise ValueError("TemplateReadout requires exactly one of simple_template or structured_template.")
    return "simple" if simple_present else "structured"


def collect_template_references(template: str | dict[str, Any] | list[Any], *, structured: bool) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()

    def _remember(raw: str) -> None:
        normalized = str(raw).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            references.append(normalized)

    def _visit_string(value: str) -> None:
        for match in _EXPR_PATTERN.findall(value):
            _remember(match)

    def _visit_block(block: Any) -> None:
        if isinstance(block, str):
            _visit_string(block)
            return
        if isinstance(block, list):
            for item in block:
                _visit_block(item)
            return
        if not isinstance(block, dict):
            return
        for key in ("title", "template", "item_template"):
            value = block.get(key)
            if isinstance(value, str):
                _visit_string(value)
        for key in ("condition", "repeat_over"):
            value = block.get(key)
            if isinstance(value, str):
                _remember(value)
        children = block.get("children")
        if isinstance(children, list):
            for child in children:
                _visit_block(child)

    if structured:
        _visit_block(template)
        return references

    if isinstance(template, str):
        _visit_string(template)
    return references


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

    runtime = {
        "now": runtime_now_factory() if runtime_now_factory is not None else _utc_now_iso(),
    }
    for key in ("session_id", "user_id", "request_id", "subgoal_id", "turn_id"):
        value = _find_nested_value(query.metadata if query is not None else {}, key)
        if value in (None, _MISSING):
            value = _find_nested_value(packet.trace, key)
        if value is not _MISSING and value is not None:
            runtime[key] = value

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
    representation = record.metadata.get("representation", {})
    if not isinstance(representation, dict):
        representation = {}
    hierarchical = record.metadata.get("hierarchical", {})
    if not isinstance(hierarchical, dict):
        hierarchical = {}
    note = note_payload_from_record(
        record,
        note_namespace=note_namespace,
        default_category=default_category,
    )
    return {
        "record_id": record.record_id,
        "unit_id": record.unit_id,
        "layer": record.layer,
        "text": record.text,
        "timestamp": record.timestamp,
        "embedding": list(record.embedding) if record.embedding is not None else [],
        "metadata": dict(record.metadata),
        "representation": dict(representation),
        "note": note,
        "graph": graph_metadata_from_record(record),
        "hierarchical": dict(hierarchical),
        "score": dict(score),
        "trace": {
            "item_index": item_index,
        },
    }


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


def parse_expression(raw: str) -> TemplateExpression:
    text = str(raw).strip()
    if not text:
        return TemplateExpression(raw=text, path="", filters=[])
    parts = _split_pipeline(text)
    path = parts[0].strip()
    filters: list[dict[str, Any]] = []
    for chunk in parts[1:]:
        filters.append(_parse_filter(chunk))
    return TemplateExpression(raw=text, path=path, filters=filters)


def render_simple_template(
    template: str,
    context: dict[str, Any],
    state: ResolutionState,
    *,
    missing_value: str = "",
) -> str:
    def _replace(match: re.Match[str]) -> str:
        expression = parse_expression(match.group(1))
        resolved = resolve_expression(expression, context, state)
        if resolved.status == "missing" and resolved.value == "":
            return missing_value
        return stringify_render_value(resolved.value)

    return _EXPR_PATTERN.sub(_replace, template)


def render_structured_template(
    template: dict[str, Any] | list[Any],
    context: dict[str, Any],
    state: ResolutionState,
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
    state: ResolutionState,
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
    state: ResolutionState,
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


def resolve_expression(
    expression: TemplateExpression,
    context: dict[str, Any],
    state: ResolutionState,
) -> ResolvedVariable:
    value, used_record_ids, used_group_ids, missing = resolve_path(expression.path, context)
    current = value
    best_record_ids = list(used_record_ids)
    best_group_ids = list(used_group_ids)
    for filter_spec in expression.filters:
        before = current
        current = apply_filter(current, filter_spec)
        filter_record_ids, filter_group_ids = collect_provenance_ids(current)
        if filter_record_ids:
            best_record_ids = filter_record_ids
        if filter_group_ids:
            best_group_ids = filter_group_ids
        state.filter_trace.append(
            {
                "expression": expression.raw,
                "filter": filter_spec["name"],
                "args": list(filter_spec["args"]),
                "before_missing": before is _MISSING,
                "after_missing": current is _MISSING,
            }
        )
    if current is _MISSING:
        current = ""
    filtered_record_ids, filtered_group_ids = collect_provenance_ids(current)
    if filtered_record_ids:
        best_record_ids = filtered_record_ids
    if filtered_group_ids:
        best_group_ids = filtered_group_ids
    resolved = ResolvedVariable(
        expression=expression.raw,
        value=current,
        status="missing" if missing else "resolved",
        source_path=expression.path,
        used_record_ids=best_record_ids,
        used_group_ids=best_group_ids,
    )
    state.resolutions.append(resolved)
    return resolved


def resolve_path(path: str, context: dict[str, Any]) -> tuple[Any, list[str], list[str], bool]:
    if not path:
        return (_MISSING, [], [], True)
    current: Any = context
    used_record_ids: list[str] = []
    used_group_ids: list[str] = []
    missing = False
    for depth, token in enumerate(path.split(".")):
        token = token.strip()
        if not token:
            missing = True
            current = _MISSING
            break
        if isinstance(current, dict):
            if token not in current:
                missing = True
                current = _MISSING
                break
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError:
                missing = True
                current = _MISSING
                break
            if index < 0 or index >= len(current):
                missing = True
                current = _MISSING
                break
            current = current[index]
        else:
            missing = True
            current = _MISSING
            break
        if depth > 0 or token not in {"query", "runtime", "retrieved", "scores", "trace"}:
            record_ids, group_ids = collect_provenance_ids(current)
            used_record_ids.extend(record_ids)
            used_group_ids.extend(group_ids)
    return current, _stable_unique(used_record_ids), _stable_unique(used_group_ids), missing


def collect_provenance_ids(value: Any) -> tuple[list[str], list[str]]:
    record_ids: list[str] = []
    group_ids: list[str] = []

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            raw_record_id = node.get("record_id")
            if isinstance(raw_record_id, str) and raw_record_id.strip():
                record_ids.append(raw_record_id.strip())
            raw_group_id = node.get("group_id")
            if isinstance(raw_group_id, str) and raw_group_id.strip():
                group_ids.append(raw_group_id.strip())
            if record_ids or group_ids or any(key in node for key in ("summary", "sources", "item")):
                for nested in node.values():
                    _visit(nested)
            return
        if isinstance(node, list):
            for nested in node:
                _visit(nested)

    _visit(value)
    return _stable_unique(record_ids), _stable_unique(group_ids)


def apply_filter(value: Any, filter_spec: dict[str, Any]) -> Any:
    name = filter_spec["name"]
    args = list(filter_spec["args"])
    if name == "default":
        fallback = args[0] if args else ""
        if value is _MISSING or value in (None, "", [], {}):
            return fallback
        return value
    if name == "join_text":
        return join_text(value)
    if name == "join":
        separator = str(args[0]) if args else ", "
        return join_values(value, separator)
    if name == "length":
        if value in (_MISSING, None, ""):
            return 0
        try:
            return len(value)
        except TypeError:
            return 0
    if name == "first":
        if isinstance(value, list) and value:
            return value[0]
        return _MISSING
    if name == "topk":
        size = int(args[0]) if args else 0
        if isinstance(value, list):
            return value[: max(size, 0)]
        return []
    if name == "sort_by":
        field_name = str(args[0]) if args else ""
        reverse = bool(args[1]) if len(args) > 1 else False
        if isinstance(value, list):
            return sort_list_of_values(value, field_name, reverse=reverse)
        return value
    return value


def stringify_render_value(value: Any) -> str:
    if value in (_MISSING, None):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return join_text(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def join_text(value: Any) -> str:
    if value in (_MISSING, None):
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return stringify_render_value(value)
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
        else:
            parts.append(stringify_render_value(item))
    return "\n".join(part for part in parts if part)


def join_values(value: Any, separator: str) -> str:
    if value in (_MISSING, None):
        return ""
    if isinstance(value, list):
        parts = [stringify_render_value(item) for item in value]
        return separator.join(part for part in parts if part)
    return stringify_render_value(value)


def sort_list_of_values(values: list[Any], field_name: str, *, reverse: bool) -> list[Any]:
    def _sort_key(item: Any) -> Any:
        if not field_name:
            return stringify_render_value(item)
        if isinstance(item, dict):
            resolved, _, _, missing = resolve_path(field_name, item)
            if missing or resolved is _MISSING or resolved is None:
                return ""
            return stringify_render_value(resolved)
        return stringify_render_value(item)

    return sorted(values, key=_sort_key, reverse=reverse)


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


def _split_pipeline(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "|" and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _parse_filter(chunk: str) -> dict[str, Any]:
    text = str(chunk).strip()
    if "(" not in text or not text.endswith(")"):
        return {"name": text, "args": []}
    name, raw_args = text.split("(", 1)
    return {"name": name.strip(), "args": _parse_filter_args(raw_args[:-1])}


def _parse_filter_args(raw_args: str) -> list[Any]:
    text = str(raw_args).strip()
    if not text:
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        current.append(char)
    parts.append("".join(current).strip())
    return [_coerce_literal(part) for part in parts if part]


def _coerce_literal(text: str) -> Any:
    lowered = text.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip()


def _find_nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_nested_value(nested, key)
            if found is not _MISSING:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_nested_value(nested, key)
            if found is not _MISSING:
                return found
    return _MISSING


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
