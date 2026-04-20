from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ..core import MemoryRecord, MemoryStore, ModuleSpec, Observation, Packet
from ..interfaces import TriggerModule
from ..pipeline import MemoryPipeline
from ._example_dialogue import (
    build_dialogue_pair_messages,
    recall_context_text,
    render_messages_for_prompt,
)
from ._llm_function_tools import WriteToolCallContext, WriteToolResult, WriteToolSpec
from ._trace import copy_trace


class PromptRecallSelectionTrigger(TriggerModule):
    """Use prompt-plan recall as the effective candidate selector for evolution."""

    spec = ModuleSpec(
        name="prompt_recall_selection_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("decisions", "decisions_store"),
    )

    def __init__(self, *, layer_names: tuple[str, ...]) -> None:
        normalized = tuple(str(layer).strip() for layer in layer_names if str(layer).strip())
        if not normalized:
            raise ValueError("PromptRecallSelectionTrigger requires at least one non-empty layer name.")
        self.layer_names = normalized

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = list(packet.units or [])
        decisions_store = {
            layer_name: {
                "decision": True,
                "record_ids": [],
                "selector": {"kind": "prompt_recall_visible_only", "source": self.spec.name},
            }
            for layer_name in self.layer_names
        }
        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "layer_names": list(self.layer_names),
            "decision_mode": "broadcast",
            "decisions_store_layers": list(self.layer_names),
            "decisions_store_counts": {layer_name: 0 for layer_name in self.layer_names},
        }
        return replace(packet, decisions=[True] * len(units), decisions_store=decisions_store, trace=trace), store

def build_profile_pair_context(packet, _store) -> dict[str, object]:
    unit = packet.units[0]
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    representation = metadata.get("representation", {})
    raw_messages = metadata.get("messages", [])
    messages = [dict(item) for item in raw_messages if isinstance(item, dict)]
    user_messages = [str(item.get("content", "")).strip() for item in messages if str(item.get("role", "")).strip() == "user"]
    assistant_messages = [
        str(item.get("content", "")).strip() for item in messages if str(item.get("role", "")).strip() == "assistant"
    ]
    fact_list = representation.get("fact_list", []) if isinstance(representation, dict) else []
    return {
        "unit": {
            "unit_id": unit.unit_id,
            "text": unit.text,
            "unit_type": unit.unit_type,
            "timestamp": unit.timestamp,
            "metadata": unit.metadata,
        },
        "messages": messages,
        "pair_text": str(metadata.get("pair_text", "")).strip() or render_messages_for_prompt(messages),
        "user_message": user_messages[-1] if user_messages else "",
        "assistant_message": assistant_messages[-1] if assistant_messages else "",
        "fact_list": list(fact_list) if isinstance(fact_list, list) else [],
        "recent_messages": str(metadata.get("recent_messages", "")).strip(),
        "conversation_summary": str(metadata.get("conversation_summary", "")).strip(),
    }


def build_graph_pair_context(packet, _store) -> dict[str, object]:
    unit = packet.units[0]
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    representation = metadata.get("representation", {})
    messages = build_profile_pair_context(packet, _store)
    summary = representation.get("summary", "") if isinstance(representation, dict) else ""
    messages.update(
        {
            "summary": summary,
            "entities": list(unit.entities),
            "triples": list(unit.triples),
        }
    )
    return messages


def find_visible_record(context: WriteToolCallContext, record_id: str) -> MemoryRecord:
    normalized = str(record_id).strip()
    for record in context.visible_records:
        if record.record_id == normalized:
            return record
    raise ValueError(f"Record {normalized!r} is not in the current profile candidate set.")


def _rejected_tool_result(
    context: WriteToolCallContext,
    *,
    tool: str,
    record_id: str,
    reason: str,
) -> WriteToolResult:
    return WriteToolResult(
        effects=[
            {
                "action": tool.removesuffix("_PROFILE").lower(),
                "record_id": str(record_id).strip(),
                "layer": "profile",
                "status": "rejected",
                "reason": reason,
                "tool": tool,
            }
        ],
        store=context.store,
    )


def build_fixed_profile_tools(
    *,
    embed_on_add: bool,
    embed_on_update: bool,
) -> list[WriteToolSpec]:
    del embed_on_add, embed_on_update

    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _tool_metadata(metadata: dict[str, object], *, action: str, context: WriteToolCallContext) -> dict[str, object]:
        return {
            **metadata,
            "llm_tool": {
                "action": action,
                "module_slot": context.module_slot,
            },
        }

    def _add_executor(context: WriteToolCallContext, arguments: dict[str, object]) -> WriteToolResult:
        text = str(arguments.get("text", "")).strip()
        if not text:
            raise ValueError("ADD_PROFILE requires a non-empty text.")
        metadata = arguments.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("ADD_PROFILE metadata must be an object.")
        packet_unit = context.packet.units[0] if context.packet.units else None
        unit_id = packet_unit.unit_id if packet_unit is not None else f"tool-unit-{context.store.next_sequence_id()}"
        timestamp = packet_unit.timestamp if packet_unit is not None else _now_iso()
        record = MemoryRecord(
            record_id=f"rec-{context.store.next_sequence_id()}",
            unit_id=unit_id,
            layer="profile",
            text=text,
            timestamp=timestamp,
            metadata=_tool_metadata(metadata, action="ADD_PROFILE", context=context),
        )
        context.store.append(record)
        return WriteToolResult(
            effects=[
                {
                    "action": "add",
                    "record_id": record.record_id,
                    "layer": "profile",
                    "status": "applied",
                    "tool": "ADD_PROFILE",
                }
            ],
            store=context.store,
        )

    def _update_executor(context: WriteToolCallContext, arguments: dict[str, object]) -> WriteToolResult:
        requested_record_id = str(arguments.get("record_id", ""))
        try:
            record = find_visible_record(context, requested_record_id)
        except ValueError as exc:
            return _rejected_tool_result(
                context,
                tool="UPDATE_PROFILE",
                record_id=requested_record_id,
                reason=str(exc),
            )
        text = record.text
        if "text" in arguments:
            text = str(arguments.get("text", "")).strip()
            if not text:
                raise ValueError("UPDATE_PROFILE text must be non-empty when provided.")
        metadata_patch = arguments.get("metadata_patch", {})
        if metadata_patch is None:
            metadata_patch = {}
        if not isinstance(metadata_patch, dict):
            raise ValueError("UPDATE_PROFILE metadata_patch must be an object.")
        updated = MemoryRecord(
            record_id=record.record_id,
            unit_id=record.unit_id,
            layer=record.layer,
            text=text,
            timestamp=record.timestamp,
            metadata=_tool_metadata({**record.metadata, **metadata_patch}, action="UPDATE_PROFILE", context=context),
        )
        context.store.replace_record(record.layer, record.record_id, updated)
        return WriteToolResult(
            effects=[
                {
                    "action": "update",
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "status": "applied",
                    "tool": "UPDATE_PROFILE",
                }
            ],
            store=context.store,
        )

    def _delete_executor(context: WriteToolCallContext, arguments: dict[str, object]) -> WriteToolResult:
        requested_record_id = str(arguments.get("record_id", ""))
        try:
            record = find_visible_record(context, requested_record_id)
        except ValueError as exc:
            return _rejected_tool_result(
                context,
                tool="DELETE_PROFILE",
                record_id=requested_record_id,
                reason=str(exc),
            )
        removed = context.store.delete_record(record.layer, record.record_id)
        return WriteToolResult(
            effects=[
                {
                    "action": "delete",
                    "record_id": removed.record_id,
                    "layer": removed.layer,
                    "status": "applied",
                    "reason": str(arguments.get("reason", "")).strip(),
                    "tool": "DELETE_PROFILE",
                }
            ],
            store=context.store,
        )

    return [
        WriteToolSpec(
            name="ADD_PROFILE",
            description="Add one new vector-memory record into the fixed 'profile' layer only.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            executor=_add_executor,
        ),
        WriteToolSpec(
            name="UPDATE_PROFILE",
            description="Update one existing vector-memory record in the fixed 'profile' layer by record_id.",
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
            executor=_update_executor,
        ),
        WriteToolSpec(
            name="DELETE_PROFILE",
            description="Delete one existing vector-memory record from the fixed 'profile' layer by record_id.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["record_id"],
                "additionalProperties": False,
            },
            executor=_delete_executor,
        ),
    ]


@dataclass(slots=True, frozen=True)
class DialogueTurnSnapshot:
    session_id: str
    turn_id: str
    pair_id: str
    user_text: str
    assistant_text: str
    messages: list[dict[str, str]]
    pair_text: str
    recent_messages: str
    conversation_summary: str

    def pair_metadata(self, **extra: object) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "pair_id": self.pair_id,
            "messages": self.messages,
            "previous_role": "user",
            "current_role": "assistant",
            "recent_messages": self.recent_messages,
            "conversation_summary": self.conversation_summary,
            **extra,
        }


def snapshot_dialogue_turn(
    *,
    recent_history_recall: MemoryPipeline,
    conversation_summary_recall: MemoryPipeline,
    user_text: str,
    assistant_text: str,
    session_id: str,
    turn_id: str,
) -> DialogueTurnSnapshot:
    pair_id = f"{session_id}:{turn_id}"
    messages = build_dialogue_pair_messages(user_text, assistant_text)
    pair_text = render_messages_for_prompt(messages)
    recent_messages = recall_context_text(recent_history_recall, query_text=pair_text)
    conversation_summary = recall_context_text(conversation_summary_recall, query_text=pair_text)
    return DialogueTurnSnapshot(
        session_id=session_id,
        turn_id=turn_id,
        pair_id=pair_id,
        user_text=user_text,
        assistant_text=assistant_text,
        messages=messages,
        pair_text=pair_text,
        recent_messages=recent_messages,
        conversation_summary=conversation_summary,
    )


def finalize_dialogue_turn(
    *,
    recent_dialogue_pipeline: MemoryPipeline,
    conversation_summary_update_pipeline: MemoryPipeline,
    turn: DialogueTurnSnapshot,
) -> None:
    for message_index, message in enumerate(turn.messages, start=1):
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        recent_dialogue_pipeline.ingest(
            Observation(
                text=content,
                source="dialogue_message",
                metadata={
                    "session_id": turn.session_id,
                    "turn_id": turn.turn_id,
                    "pair_id": turn.pair_id,
                    "message_index": message_index,
                    "role": message["role"],
                    "messages": turn.messages,
                },
            )
        )
    conversation_summary_update_pipeline.ingest(
        Observation(
            text=turn.pair_text,
            source="dialogue_pair_summary",
            metadata=turn.pair_metadata(),
        )
    )
