from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable

from agents.tool import FunctionTool

from ..core import MemoryStore, Packet
from ._template import run_child_recall_pipeline, text_prompt


@dataclass(slots=True, frozen=True)
class ReadoutToolSpec:
    name: str
    description: str
    parameters_json_schema: dict[str, Any]
    executor: Callable[["ReadoutToolCallContext", dict[str, Any]], "ReadoutToolResult"]

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("ReadoutToolSpec.name must be a non-empty string.")
        if not str(self.description).strip():
            raise ValueError("ReadoutToolSpec.description must be a non-empty string.")
        if not isinstance(self.parameters_json_schema, dict):
            raise ValueError("ReadoutToolSpec.parameters_json_schema must be a dict.")


@dataclass(slots=True)
class ReadoutToolCallContext:
    packet: Packet
    store: MemoryStore
    retrieve_pipeline: Any | None = None


@dataclass(slots=True)
class ReadoutToolResult:
    result: dict[str, Any]
    effects: list[dict[str, Any]]
    store: MemoryStore


@dataclass(slots=True)
class ToolExecutionState:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)
    memory_read_count: int = 0
    memory_read_record_ids: list[str] = field(default_factory=list)


def project_tool_specs_for_prompt(specs: tuple[ReadoutToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters_json_schema": dict(spec.parameters_json_schema),
        }
        for spec in specs
    ]


def normalize_readout_tool_specs(
    tools: list[str | ReadoutToolSpec],
    *,
    module_name: str,
    retrieve_pipeline: Any | None,
) -> tuple[ReadoutToolSpec, ...]:
    if not tools:
        raise ValueError("tools must contain at least one entry.")
    normalized: list[ReadoutToolSpec] = []
    seen: set[str] = set()
    for item in tools:
        if isinstance(item, str):
            spec = builtin_readout_tool_spec(
                item,
                module_name=module_name,
                retrieve_pipeline=retrieve_pipeline,
            )
        elif isinstance(item, ReadoutToolSpec):
            spec = item
        else:
            raise TypeError("tools entries must be strings or ReadoutToolSpec instances.")
        if spec.name in seen:
            raise ValueError(f"Duplicate readout tool name {spec.name!r}.")
        seen.add(spec.name)
        normalized.append(spec)
    return tuple(normalized)


def builtin_readout_tool_spec(
    name: str,
    *,
    module_name: str,
    retrieve_pipeline: Any | None,
) -> ReadoutToolSpec:
    normalized = str(name).strip().upper()
    if normalized == "MEM_READ":
        if retrieve_pipeline is None:
            raise ValueError("MEM_READ requires retrieve_pipeline.")
        return ReadoutToolSpec(
            name="MEM_READ",
            description="Retrieve memory by running the configured child recall pipeline.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            executor=_build_mem_read_executor(
                module_name=module_name,
                retrieve_pipeline=retrieve_pipeline,
            ),
        )
    raise ValueError(f"Unknown built-in readout tool {name!r}. Supported values: MEM_READ.")


def build_runtime_tools(
    specs: tuple[ReadoutToolSpec, ...],
    *,
    context: ReadoutToolCallContext,
    state: ToolExecutionState,
    strict_tools: bool,
) -> list[FunctionTool]:
    tools: list[FunctionTool] = []
    for spec in specs:

        async def _invoke_tool(_tool_context, arguments_json: str, *, spec: ReadoutToolSpec = spec) -> str:
            try:
                payload = json.loads(arguments_json) if arguments_json.strip() else {}
                if not isinstance(payload, dict):
                    raise ValueError(f"{spec.name} arguments must decode to a JSON object.")
                validate_tool_arguments(payload, spec.parameters_json_schema, tool_name=spec.name)
                result = spec.executor(context, payload)
                context.store = result.store
                state.effects.extend(dict(effect) for effect in result.effects)
                _accumulate_effect_state(state, result.effects)
                state.tool_calls.append(
                    {
                        "tool_name": spec.name,
                        "arguments": dict(payload),
                        "status": "applied",
                        "result_summary": summarize_effects(result.effects),
                        "error": "",
                    }
                )
                return json.dumps(result.result, ensure_ascii=False)
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


def summarize_effects(effects: list[dict[str, Any]]) -> str:
    if not effects:
        return "no effects"
    effect = effects[0]
    action = str(effect.get("action", "")).strip().casefold()
    if action == "memory_read":
        record_ids = effect.get("record_ids", [])
        count = len(record_ids) if isinstance(record_ids, list) else 0
        matched = bool(effect.get("matched", False))
        status = "matched" if matched else "empty"
        return f"memory_read:{status}:{count}"
    return ", ".join(str(effect.get("action", "unknown")) for effect in effects)


def _accumulate_effect_state(state: ToolExecutionState, effects: list[dict[str, Any]]) -> None:
    for effect in effects:
        if str(effect.get("action", "")).strip().casefold() != "memory_read":
            continue
        state.memory_read_count += 1
        for raw_record_id in effect.get("record_ids", []):
            record_id = str(raw_record_id).strip()
            if record_id and record_id not in state.memory_read_record_ids:
                state.memory_read_record_ids.append(record_id)


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


def _build_mem_read_executor(
    *,
    module_name: str,
    retrieve_pipeline,
) -> Callable[[ReadoutToolCallContext, dict[str, Any]], ReadoutToolResult]:
    def _execute(context: ReadoutToolCallContext, arguments: dict[str, Any]) -> ReadoutToolResult:
        query_text = str(arguments.get("query", "")).strip()
        if not query_text:
            raise ValueError("MEM_READ requires a non-empty query.")
        readout, updated_store = run_child_recall_pipeline(
            store=context.store,
            query_text=query_text,
            retrieve_pipeline=retrieve_pipeline,
            fallback_readout_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
        )
        record_ids = [str(record_id).strip() for record_id in readout.source_ids if str(record_id).strip()]
        result = {
            "tool_name": "MEM_READ",
            "query": query_text,
            "matched": bool(readout.text.strip() or record_ids),
            "memory_text": readout.text,
            "source_ids": record_ids,
            "match_count": len(record_ids),
        }
        return ReadoutToolResult(
            result=result,
            effects=[
                {
                    "action": "memory_read",
                    "tool_name": "MEM_READ",
                    "module": module_name,
                    "query": query_text,
                    "matched": result["matched"],
                    "record_ids": record_ids,
                }
            ],
            store=updated_store,
        )

    return _execute
