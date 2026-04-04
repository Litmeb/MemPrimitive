"""Lightweight template helpers shared across prompt-bearing modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import ast
import json
import re
from typing import Any

from ..core import MemoryRecord, MemoryUnit, Packet, Query

_EXPR_PATTERN = re.compile(r"{{\s*(.+?)\s*}}")
_MISSING = object()

PromptTemplate = str


def looks_like_template(text: str) -> bool:
    return bool(_EXPR_PATTERN.search(str(text)))


def utc_now_iso() -> str:
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
class ResolutionState:
    resolutions: list[ResolvedVariable] = field(default_factory=list)
    filter_trace: list[dict[str, Any]] = field(default_factory=list)


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


def render_prompt_template(
    template: str,
    context: dict[str, Any],
    *,
    missing_value: str = "",
) -> tuple[str, ResolutionState]:
    state = ResolutionState()
    return render_simple_template(template, context, state, missing_value=missing_value), state


def project_unit_for_template(unit: MemoryUnit) -> dict[str, Any]:
    representation = unit.metadata.get("representation", {})
    if not isinstance(representation, dict):
        representation = {}
    return {
        "unit_id": unit.unit_id,
        "text": unit.text,
        "normalized_text": unit.normalized_text,
        "unit_type": unit.unit_type,
        "timestamp": unit.timestamp,
        "embedding": list(unit.embedding) if unit.embedding is not None else [],
        "tags": list(unit.tags),
        "entities": list(unit.entities),
        "kv": dict(unit.kv),
        "description": unit.description,
        "triples": [list(triple) for triple in unit.triples],
        "representation_elements": list(unit.representation_elements),
        "metadata": dict(unit.metadata),
        "representation": dict(representation),
    }


def project_record_for_template(
    record: MemoryRecord,
    *,
    score: dict[str, Any] | None = None,
    item_index: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    representation = record.metadata.get("representation", {})
    if not isinstance(representation, dict):
        representation = {}
    hierarchical = record.metadata.get("hierarchical", {})
    if not isinstance(hierarchical, dict):
        hierarchical = {}
    payload = {
        "record_id": record.record_id,
        "unit_id": record.unit_id,
        "layer": record.layer,
        "text": record.text,
        "timestamp": record.timestamp,
        "embedding": list(record.embedding) if record.embedding is not None else [],
        "metadata": dict(record.metadata),
        "representation": dict(representation),
        "hierarchical": dict(hierarchical),
        "score": dict(score or {}),
        "trace": {},
    }
    if item_index is not None:
        payload["trace"]["item_index"] = item_index
    if extra:
        payload.update(extra)
    return payload


def project_query_for_template(query: Query | None) -> dict[str, Any]:
    if query is None:
        return {"text": "", "timestamp": "", "embedding": [], "metadata": {}}
    return {
        "text": query.text,
        "timestamp": query.timestamp,
        "embedding": list(query.embedding) if query.embedding is not None else [],
        "metadata": dict(query.metadata),
    }


def project_packet_runtime_for_template(packet: Packet, *, now: str | None = None) -> dict[str, Any]:
    runtime = {"now": now or utc_now_iso()}
    query = packet.query
    for key in ("session_id", "user_id", "request_id", "subgoal_id", "turn_id"):
        value = _find_nested_value(query.metadata if query is not None else {}, key)
        if value in (None, _MISSING):
            value = _find_nested_value(packet.trace, key)
        if value is not _MISSING and value is not None:
            runtime[key] = value
    return runtime


def metadata_from_resolution_state(*, state: ResolutionState) -> dict[str, Any]:
    missing_variables = _stable_unique(
        resolved.expression for resolved in state.resolutions if resolved.status == "missing"
    )
    used_record_ids = _stable_unique(
        record_id for resolved in state.resolutions for record_id in resolved.used_record_ids
    )
    used_group_ids = _stable_unique(
        group_id for resolved in state.resolutions for group_id in resolved.used_group_ids
    )
    return {
        "resolved_variables": [asdict(item) for item in state.resolutions],
        "missing_variables": missing_variables,
        "used_group_ids": used_group_ids,
        "used_record_ids": used_record_ids,
        "filter_trace": list(state.filter_trace),
    }


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
    for token in path.split("."):
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
