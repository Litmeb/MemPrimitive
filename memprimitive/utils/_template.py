"""Lightweight template helpers shared across prompt-bearing modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
import ast
import json
import re
from typing import Any, Callable, Literal

from ..core import MemoryRecord, MemoryStore, MemoryUnit, Packet, Query, Readout

_EXPR_PATTERN = re.compile(r"{{\s*(.+?)\s*}}")
_MISSING = object()

PromptTemplate = str
TemplateMode = Literal["simple", "structured"]
MetadataMode = Literal["prompt", "readout"]
ContextBuilder = Callable[[Packet, MemoryStore], dict[str, Any]]
RecallQueryBuilder = Callable[[Packet, MemoryStore, dict[str, Any]], str]


def looks_like_template(text: str) -> bool:
    return bool(_EXPR_PATTERN.search(str(text)))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True, frozen=True)
class PromptPlan:
    mode: TemplateMode
    template: str | dict[str, Any] | list[Any]
    context_builder: ContextBuilder | None = None
    recall_plan: PromptPlan | None = None
    recall_query_builder: RecallQueryBuilder | None = None
    missing_value: str = ""
    metadata_mode: MetadataMode = "prompt"
    sub_recall_pipeline: Any | None = None


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


def text_prompt(
    template: str,
    *,
    context_builder: ContextBuilder | None = None,
    recall_plan: PromptPlan | None = None,
    recall_query_builder: RecallQueryBuilder | None = None,
    missing_value: str = "",
    metadata_mode: MetadataMode = "prompt",
    sub_recall_pipeline: Any | None = None,
) -> PromptPlan:
    return PromptPlan(
        mode="simple",
        template=str(template),
        context_builder=context_builder,
        recall_plan=recall_plan,
        recall_query_builder=recall_query_builder,
        missing_value=missing_value,
        metadata_mode=metadata_mode,
        sub_recall_pipeline=sub_recall_pipeline,
    )


def structured_prompt(
    template: dict[str, Any] | list[Any],
    *,
    context_builder: ContextBuilder | None = None,
    recall_plan: PromptPlan | None = None,
    recall_query_builder: RecallQueryBuilder | None = None,
    missing_value: str = "",
    metadata_mode: MetadataMode = "readout",
    sub_recall_pipeline: Any | None = None,
) -> PromptPlan:
    return PromptPlan(
        mode="structured",
        template=template,
        context_builder=context_builder,
        recall_plan=recall_plan,
        recall_query_builder=recall_query_builder,
        missing_value=missing_value,
        metadata_mode=metadata_mode,
        sub_recall_pipeline=sub_recall_pipeline,
    )


def ensure_prompt_plan(
    prompt: PromptPlan | str,
    *,
    metadata_mode: MetadataMode | None = None,
    context_builder: ContextBuilder | None = None,
) -> PromptPlan:
    if isinstance(prompt, PromptPlan):
        plan = prompt
    else:
        plan = text_prompt(str(prompt), metadata_mode=metadata_mode or "prompt")
    if metadata_mode is not None and plan.metadata_mode != metadata_mode:
        plan = replace(plan, metadata_mode=metadata_mode)
    if context_builder is not None:
        merged_builder = compose_context_builders(plan.context_builder, context_builder)
        plan = replace(plan, context_builder=merged_builder)
    return plan


def compose_context_builders(
    left: ContextBuilder | None,
    right: ContextBuilder | None,
) -> ContextBuilder | None:
    if left is None:
        return right
    if right is None:
        return left

    def _merged(packet: Packet, store: MemoryStore) -> dict[str, Any]:
        context = dict(left(packet, store))
        context.update(right(packet, store))
        return context

    return _merged


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


def build_base_template_context(
    packet: Packet,
    store: MemoryStore,
    *,
    now: str | None = None,
) -> dict[str, Any]:
    return {
        "query": project_query_for_template(packet.query),
        "runtime": project_packet_runtime_for_template(packet, now=now),
        "trace": {
            "packet": dict(packet.trace),
            "retrieval": dict(packet.retrieved.trace) if packet.retrieved is not None else {},
        },
    }


def build_retrieval_template_context(
    packet: Packet,
    store: MemoryStore,
    *,
    note_namespace: str = "note",
    default_category: str = "memory",
    runtime_now_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    from ._template_readout import build_render_context

    return build_render_context(
        packet,
        note_namespace=note_namespace,
        default_category=default_category,
        runtime_now_factory=runtime_now_factory,
    )


def render_prompt_plan(
    plan: PromptPlan | str,
    *,
    packet: Packet,
    store: MemoryStore,
    runtime_now_factory: Callable[[], str] | None = None,
) -> tuple[str, dict[str, Any], MemoryStore]:
    prompt_plan = ensure_prompt_plan(plan)
    now = runtime_now_factory() if runtime_now_factory is not None else utc_now_iso()
    context = build_base_template_context(packet, store, now=now)
    if prompt_plan.metadata_mode == "readout" or packet.retrieved is not None:
        context.update(build_retrieval_template_context(packet, store, runtime_now_factory=lambda: now))
    if prompt_plan.context_builder is not None:
        context.update(prompt_plan.context_builder(packet, store))

    recalled_prompt, recall_metadata, updated_store = resolve_recalled_prompt(
        prompt_plan,
        packet=packet,
        store=store,
        context=context,
    )
    context["recalled_prompt"] = recalled_prompt

    if prompt_plan.metadata_mode == "readout":
        return _render_readout_plan(prompt_plan, context=context, recall_metadata=recall_metadata, store=updated_store)
    return _render_prompt_mode_plan(prompt_plan, context=context, recall_metadata=recall_metadata, store=updated_store)


def _render_prompt_mode_plan(
    plan: PromptPlan,
    *,
    context: dict[str, Any],
    recall_metadata: dict[str, Any],
    store: MemoryStore,
) -> tuple[str, dict[str, Any], MemoryStore]:
    if plan.mode == "structured":
        from ._template_readout import ReadoutResolutionState, render_structured_template

        state = ReadoutResolutionState()
        rendered = render_structured_template(plan.template, context, state, missing_value=plan.missing_value)
    else:
        state = ResolutionState()
        rendered = render_simple_template(str(plan.template), context, state, missing_value=plan.missing_value)
    metadata = metadata_from_resolution_state(state=state)
    metadata.update(
        {
            "template_mode": plan.mode,
            "prompt_is_template": plan.mode == "structured" or looks_like_template(str(plan.template)),
            "rendered_prompt": rendered,
            "rendered_prompt_preview": rendered[:200],
            "recalled_prompt": context.get("recalled_prompt", ""),
            "recalled_prompt_preview": str(context.get("recalled_prompt", ""))[:200],
            "recall_prompt": recall_metadata,
            "context_summary": sorted(context.keys()),
        }
    )
    return rendered, metadata, store


def _render_readout_plan(
    plan: PromptPlan,
    *,
    context: dict[str, Any],
    recall_metadata: dict[str, Any],
    store: MemoryStore,
) -> tuple[str, dict[str, Any], MemoryStore]:
    from ._template_readout import ReadoutResolutionState, metadata_from_state, render_structured_template

    state = ReadoutResolutionState()
    if plan.mode == "structured":
        declared_variables = collect_template_references(plan.template, structured=True)
        rendered = render_structured_template(plan.template, context, state, missing_value=plan.missing_value)
    else:
        declared_variables = collect_template_references(str(plan.template), structured=False)
        rendered = render_simple_template(str(plan.template), context, state, missing_value=plan.missing_value)
    metadata = metadata_from_state(
        template_mode_name=plan.mode,
        declared_variables=declared_variables,
        context=context,
        state=state,
    )
    metadata.update(
        {
            "rendered_prompt": rendered,
            "rendered_prompt_preview": rendered[:200],
            "recalled_prompt": context.get("recalled_prompt", ""),
            "recalled_prompt_preview": str(context.get("recalled_prompt", ""))[:200],
            "recall_prompt": recall_metadata,
        }
    )
    return rendered, metadata, store


def resolve_recalled_prompt(
    plan: PromptPlan,
    *,
    packet: Packet,
    store: MemoryStore,
    context: dict[str, Any],
) -> tuple[str, dict[str, Any], MemoryStore]:
    metadata: dict[str, Any] = {
        "enabled": False,
        "rendered_recall_query": "",
        "rendered_recall_query_preview": "",
        "missing_variables": [],
        "resolved_variables": [],
        "used_record_ids": [],
        "used_group_ids": [],
        "filter_trace": [],
        "matched": False,
        "recalled_prompt": "",
        "recalled_prompt_preview": "",
        "readout_source_ids": [],
        "readout_metadata": {},
    }
    if plan.recall_plan is None or plan.recall_query_builder is None or plan.sub_recall_pipeline is None:
        metadata["disabled_reason"] = "missing_recall_plan_or_query_builder_or_pipeline"
        return "", metadata, store

    metadata["enabled"] = True
    recall_query = render_recall_query(plan.recall_query_builder, packet, store, context)
    metadata["rendered_recall_query"] = recall_query
    metadata["rendered_recall_query_preview"] = recall_query[:200]
    if not recall_query.strip():
        metadata["disabled_reason"] = "empty_rendered_recall_query"
        return "", metadata, store

    readout, updated_store = run_prompt_plan_sub_recall(
        plan.recall_plan,
        store=store,
        query_text=recall_query,
        retrieve_pipeline=plan.sub_recall_pipeline,
    )
    recalled_prompt = readout.text.strip() if readout.text else ""
    metadata["matched"] = bool(recalled_prompt)
    metadata["recalled_prompt"] = recalled_prompt
    metadata["recalled_prompt_preview"] = recalled_prompt[:200]
    metadata["readout_source_ids"] = list(readout.source_ids)
    metadata["readout_metadata"] = dict(readout.metadata)
    return recalled_prompt, metadata, updated_store


def render_recall_query(
    recall_query_builder: RecallQueryBuilder,
    packet: Packet,
    store: MemoryStore,
    context: dict[str, Any],
) -> str:
    return str(recall_query_builder(packet, store, context) or "")


def run_prompt_plan_sub_recall(
    plan: PromptPlan,
    *,
    store: MemoryStore,
    query_text: str,
    retrieve_pipeline,
) -> tuple[Readout, MemoryStore]:
    from ..pipeline import _iter_slot_modules

    packet = Packet(query=Query(text=query_text), trace={"sub_recall_started": True})
    current_store = store
    for module in _iter_slot_modules(retrieve_pipeline.retrieval):
        packet, current_store = module.run(packet, current_store)

    readout_module = retrieve_pipeline.readout
    if readout_module is not None:
        for module in _iter_slot_modules(readout_module):
            packet, current_store = module.run(packet, current_store)
    else:
        rendered_text, metadata, current_store = render_prompt_plan(
            ensure_prompt_plan(plan, metadata_mode="readout"),
            packet=packet,
            store=current_store,
        )
        packet = replace(packet, readout=Readout(text=rendered_text, source_ids=list(metadata.get("used_record_ids", [])), metadata=metadata))

    if packet.readout is None:
        raise RuntimeError("Sub recall pipeline returned no readout.")
    return packet.readout, current_store


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
