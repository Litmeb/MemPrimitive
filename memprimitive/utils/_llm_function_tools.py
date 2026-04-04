from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
from typing import Any, Callable, Literal
from uuid import uuid4

from agents.tool import FunctionTool

from ..core import MemoryRecord, MemoryStore, MemoryUnit, Packet


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _tool_generated_unit_id() -> str:
    return f"tool-unit-{uuid4().hex}"


@dataclass(slots=True, frozen=True)
class WriteToolSpec:
    name: str
    description: str
    parameters_json_schema: dict[str, Any]
    executor: Callable[["WriteToolCallContext", dict[str, Any]], "WriteToolResult"]

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("WriteToolSpec.name must be a non-empty string.")
        if not str(self.description).strip():
            raise ValueError("WriteToolSpec.description must be a non-empty string.")
        if not isinstance(self.parameters_json_schema, dict):
            raise ValueError("WriteToolSpec.parameters_json_schema must be a dict.")


@dataclass(slots=True)
class WriteToolCallContext:
    packet: Packet
    store: MemoryStore
    module_slot: Literal["organization", "memory_evolution"]
    default_target_layer: str | None
    visible_records: list[MemoryRecord]


@dataclass(slots=True)
class WriteToolResult:
    effects: list[dict[str, Any]]
    store: MemoryStore


@dataclass(slots=True)
class ToolExecutionState:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)
    written_record_ids: list[str] = field(default_factory=list)
    updated_record_ids: list[str] = field(default_factory=list)
    deleted_record_ids: list[str] = field(default_factory=list)


def project_tool_specs_for_prompt(specs: tuple[WriteToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters_json_schema": dict(spec.parameters_json_schema),
        }
        for spec in specs
    ]


def normalize_write_tool_specs(
    tools: list[str | WriteToolSpec],
    *,
    module_name: str,
) -> tuple[WriteToolSpec, ...]:
    if not tools:
        raise ValueError("tools must contain at least one entry.")
    normalized: list[WriteToolSpec] = []
    seen: set[str] = set()
    for item in tools:
        if isinstance(item, str):
            spec = builtin_write_tool_spec(item, module_name=module_name)
        elif isinstance(item, WriteToolSpec):
            spec = item
        else:
            raise TypeError("tools entries must be strings or WriteToolSpec instances.")
        if spec.name in seen:
            raise ValueError(f"Duplicate write tool name {spec.name!r}.")
        seen.add(spec.name)
        normalized.append(spec)
    return tuple(normalized)


def builtin_write_tool_spec(name: str, *, module_name: str) -> WriteToolSpec:
    normalized = str(name).strip().upper()
    if normalized == "ADD":
        return WriteToolSpec(
            name="ADD",
            description="Add one new memory record.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_layer": {"type": "string"},
                    "metadata": {"type": "object"},
                    "unit_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            executor=_build_add_executor(module_name=module_name),
        )
    if normalized == "UPDATE":
        return WriteToolSpec(
            name="UPDATE",
            description="Update one existing memory record by record_id.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "text": {"type": "string"},
                    "metadata_patch": {"type": "object"},
                },
                "required": ["record_id"],
                "additionalProperties": False,
            },
            executor=_build_update_executor(module_name=module_name),
        )
    if normalized == "DELETE":
        return WriteToolSpec(
            name="DELETE",
            description="Delete one existing memory record by record_id.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["record_id"],
                "additionalProperties": False,
            },
            executor=_build_delete_executor(module_name=module_name),
        )
    raise ValueError(f"Unknown built-in write tool {name!r}. Supported values: ADD, UPDATE, DELETE.")


def build_runtime_tools(
    specs: tuple[WriteToolSpec, ...],
    *,
    context: WriteToolCallContext,
    state: ToolExecutionState,
    strict_tools: bool,
) -> list[FunctionTool]:
    tools: list[FunctionTool] = []
    for spec in specs:

        async def _invoke_tool(_tool_context, arguments_json: str, *, spec: WriteToolSpec = spec) -> str:
            try:
                payload = json.loads(arguments_json) if arguments_json.strip() else {}
                if not isinstance(payload, dict):
                    raise ValueError(f"{spec.name} arguments must decode to a JSON object.")
                validate_tool_arguments(payload, spec.parameters_json_schema, tool_name=spec.name)
                result = spec.executor(context, payload)
                context.store = result.store
                context.visible_records = list(context.store.iter_records())
                state.effects.extend(dict(effect) for effect in result.effects)
                _accumulate_effect_ids(state, result.effects)
                state.tool_calls.append(
                    {
                        "tool_name": spec.name,
                        "arguments": dict(payload),
                        "status": "applied",
                        "result_summary": summarize_effects(result.effects),
                        "error": "",
                    }
                )
                return json.dumps({"status": "applied", "effects": result.effects}, ensure_ascii=False)
            except Exception as exc:
                state.tool_calls.append(
                    {
                        "tool_name": spec.name,
                        "arguments": dict(payload) if "payload" in locals() and isinstance(payload, dict) else {},
                        "status": "failed",
                        "result_summary": "",
                        "error": str(exc),
                    }
                )
                if strict_tools:
                    raise
                return json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)

        tools.append(
            FunctionTool(
                name=spec.name,
                description=spec.description,
                params_json_schema=dict(spec.parameters_json_schema),
                on_invoke_tool=_invoke_tool,
                strict_json_schema=False,
            )
        )
    return tools


def validate_tool_arguments(arguments: dict[str, Any], schema: dict[str, Any], *, tool_name: str) -> None:
    if schema.get("type") != "object":
        raise ValueError(f"{tool_name} only supports object parameter schemas.")
    if not isinstance(arguments, dict):
        raise ValueError(f"{tool_name} arguments must be a JSON object.")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    for key in required:
        if key not in arguments:
            raise ValueError(f"{tool_name} missing required argument {key!r}.")
    allow_extra = schema.get("additionalProperties", True) is not False
    for key, value in arguments.items():
        if key not in properties:
            if allow_extra:
                continue
            raise ValueError(f"{tool_name} does not allow extra argument {key!r}.")
        expected_type = properties[key].get("type")
        if expected_type is not None and not _matches_json_type(value, expected_type):
            raise ValueError(f"{tool_name} argument {key!r} must have JSON type {expected_type}.")


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def summarize_effects(effects: list[dict[str, Any]]) -> str:
    if not effects:
        return "no effects"
    parts = [
        f"{effect.get('action', 'unknown')}:{effect.get('record_id', '')}@{effect.get('layer', '')}"
        for effect in effects
    ]
    return ", ".join(parts)


def _accumulate_effect_ids(state: ToolExecutionState, effects: list[dict[str, Any]]) -> None:
    for effect in effects:
        action = str(effect.get("action", "")).strip().casefold()
        record_id = str(effect.get("record_id", "")).strip()
        if not record_id:
            continue
        if action == "add" and record_id not in state.written_record_ids:
            state.written_record_ids.append(record_id)
        elif action == "update" and record_id not in state.updated_record_ids:
            state.updated_record_ids.append(record_id)
        elif action == "delete" and record_id not in state.deleted_record_ids:
            state.deleted_record_ids.append(record_id)


def _build_add_executor(*, module_name: str) -> Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]:
    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        text = str(arguments.get("text", "")).strip()
        if not text:
            raise ValueError("ADD requires a non-empty text.")
        target_layer = str(arguments.get("target_layer", "")).strip() or (context.default_target_layer or "")
        if not target_layer:
            raise ValueError("ADD requires target_layer either in tool args or module constructor.")
        metadata = arguments.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("ADD metadata must be an object.")
        current_unit = _single_unit_from_packet(context.packet)
        unit_id = str(arguments.get("unit_id", "")).strip() or (current_unit.unit_id if current_unit is not None else _tool_generated_unit_id())
        timestamp = str(arguments.get("timestamp", "")).strip() or (
            current_unit.timestamp if current_unit is not None else _utc_now_iso()
        )
        record = MemoryRecord(
            record_id=f"rec-{context.store.next_sequence_id()}",
            unit_id=unit_id,
            layer=target_layer,
            text=text,
            timestamp=timestamp,
            metadata={
                **metadata,
                "llm_tool": {
                    **(metadata.get("llm_tool", {}) if isinstance(metadata.get("llm_tool"), dict) else {}),
                    "action": "ADD",
                    "module": module_name,
                    "module_slot": context.module_slot,
                },
            },
        )
        context.store.append(record)
        return WriteToolResult(
            effects=[
                {
                    "action": "add",
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "status": "applied",
                }
            ],
            store=context.store,
        )

    return _execute


def _build_update_executor(*, module_name: str) -> Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]:
    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        record = find_record_by_id(context.store, str(arguments.get("record_id", "")).strip())
        metadata_patch = arguments.get("metadata_patch", {})
        if metadata_patch is None:
            metadata_patch = {}
        if not isinstance(metadata_patch, dict):
            raise ValueError("UPDATE metadata_patch must be an object.")
        next_text = record.text
        if "text" in arguments:
            next_text = str(arguments["text"]).strip()
            if not next_text:
                raise ValueError("UPDATE text must be non-empty when provided.")
        updated = replace(
            record,
            text=next_text,
            metadata={
                **record.metadata,
                **metadata_patch,
                "llm_tool": {
                    **(record.metadata.get("llm_tool", {}) if isinstance(record.metadata.get("llm_tool"), dict) else {}),
                    "action": "UPDATE",
                    "module": module_name,
                    "module_slot": context.module_slot,
                },
            },
        )
        context.store.replace_record(record.layer, record.record_id, updated)
        return WriteToolResult(
            effects=[
                {
                    "action": "update",
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "status": "applied",
                }
            ],
            store=context.store,
        )

    return _execute


def _build_delete_executor(*, module_name: str) -> Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]:
    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        record = find_record_by_id(context.store, str(arguments.get("record_id", "")).strip())
        removed = context.store.delete_record(record.layer, record.record_id)
        return WriteToolResult(
            effects=[
                {
                    "action": "delete",
                    "record_id": removed.record_id,
                    "layer": removed.layer,
                    "status": "applied",
                    "reason": str(arguments.get("reason", "")).strip(),
                    "module": module_name,
                }
            ],
            store=context.store,
        )

    return _execute


def find_record_by_id(store: MemoryStore, record_id: str) -> MemoryRecord:
    normalized = str(record_id).strip()
    if not normalized:
        raise ValueError("record_id must be a non-empty string.")
    for record in store.iter_records():
        if record.record_id == normalized:
            return record
    raise KeyError(f"Record {normalized!r} not found.")


def _single_unit_from_packet(packet: Packet) -> MemoryUnit | None:
    if packet.units is None or len(packet.units) != 1:
        return None
    return packet.units[0]
