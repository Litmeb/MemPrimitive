"""Structured profile-feature write tools for MemMachine-style memory."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Callable

from ..core import MemoryRecord
from ._llm_function_tools import (
    WriteToolCallContext,
    WriteToolResult,
    WriteToolSpec,
    find_record_by_id,
)

PROFILE_FEATURE_METADATA_KEY = "profile_feature"


def build_profile_feature_tools(*, module_name: str = "profile_feature_tools") -> list[WriteToolSpec]:
    """Return tools for category/tag/feature/value profile memory updates."""

    return [
        profile_feature_tool_spec("UPSERT_PROFILE_FEATURE", module_name=module_name),
        profile_feature_tool_spec("DELETE_PROFILE_FEATURE", module_name=module_name),
    ]


def profile_feature_tool_spec(name: str, *, module_name: str) -> WriteToolSpec:
    normalized = str(name).strip().upper()
    if normalized == "UPSERT_PROFILE_FEATURE":
        return WriteToolSpec(
            name="UPSERT_PROFILE_FEATURE",
            description=(
                "Insert or update one structured profile feature. The identity key is "
                "target_layer + set_id + category + tag + feature."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "tag": {"type": "string"},
                    "feature": {"type": "string"},
                    "value": {"type": "string"},
                    "set_id": {"type": "string"},
                    "target_layer": {"type": "string"},
                    "source_episode_record_ids": {"type": "array"},
                    "citation_record_ids": {"type": "array"},
                    "metadata": {"type": "object"},
                },
                "required": ["category", "tag", "feature", "value"],
                "additionalProperties": False,
            },
            executor=_build_upsert_profile_feature_executor(module_name=module_name),
        )
    if normalized == "DELETE_PROFILE_FEATURE":
        return WriteToolSpec(
            name="DELETE_PROFILE_FEATURE",
            description=(
                "Delete one structured profile feature by record_id, or by the exact "
                "target_layer + set_id + category + tag + feature key."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "category": {"type": "string"},
                    "tag": {"type": "string"},
                    "feature": {"type": "string"},
                    "set_id": {"type": "string"},
                    "target_layer": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            },
            executor=_build_delete_profile_feature_executor(module_name=module_name),
        )
    raise ValueError(f"Unknown profile feature tool {name!r}.")


def _build_upsert_profile_feature_executor(
    *,
    module_name: str,
) -> Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]:
    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        target_layer = _target_layer(context, arguments)
        category = _required_text(arguments, "category", tool_name="UPSERT_PROFILE_FEATURE")
        tag = _required_text(arguments, "tag", tool_name="UPSERT_PROFILE_FEATURE")
        feature = _required_text(arguments, "feature", tool_name="UPSERT_PROFILE_FEATURE")
        value = _required_text(arguments, "value", tool_name="UPSERT_PROFILE_FEATURE")
        set_id = _optional_text(arguments, "set_id")
        metadata = arguments.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("UPSERT_PROFILE_FEATURE metadata must be an object.")

        source_episode_record_ids = _merged_ids(
            [record.record_id for record in context.selected_records],
            _coerce_string_list(
                arguments.get("source_episode_record_ids"),
                argument_name="source_episode_record_ids",
                tool_name="UPSERT_PROFILE_FEATURE",
            ),
        )
        citation_record_ids = _merged_ids(
            source_episode_record_ids,
            _coerce_string_list(
                arguments.get("citation_record_ids"),
                argument_name="citation_record_ids",
                tool_name="UPSERT_PROFILE_FEATURE",
            ),
        )
        existing = _find_feature_record(
            context.store.iter_records(target_layer),
            category=category,
            tag=tag,
            feature=feature,
            set_id=set_id,
        )

        now = _utc_now_iso()
        if existing is None:
            sequence_id = context.store.next_sequence_id()
            unit_id = _current_unit_id(context) or f"profile-feature-{sequence_id}"
            timestamp = _current_timestamp(context) or now
            record = MemoryRecord(
                record_id=f"rec-{sequence_id}",
                unit_id=unit_id,
                layer=target_layer,
                text=value,
                timestamp=timestamp,
                metadata=_profile_feature_metadata(
                    base_metadata=metadata,
                    category=category,
                    tag=tag,
                    feature=feature,
                    value=value,
                    set_id=set_id,
                    source_episode_record_ids=source_episode_record_ids,
                    citation_record_ids=citation_record_ids,
                    created_at=now,
                    updated_at=now,
                    tool_name="UPSERT_PROFILE_FEATURE",
                    module_name=module_name,
                    module_slot=context.module_slot,
                ),
            )
            context.store.append(record)
            return WriteToolResult(
                effects=[
                    {
                        "action": "add",
                        "effect_type": "profile_feature_upsert",
                        "record_id": record.record_id,
                        "layer": record.layer,
                        "status": "applied",
                        "tool": "UPSERT_PROFILE_FEATURE",
                        "category": category,
                        "tag": tag,
                        "feature": feature,
                        "set_id": set_id,
                        "source_episode_record_ids": source_episode_record_ids,
                        "citation_record_ids": citation_record_ids,
                    }
                ],
                store=context.store,
            )

        existing_payload = _profile_payload(existing)
        merged_source_ids = _merged_ids(
            _coerce_string_list(
                existing_payload.get("source_episode_record_ids"),
                argument_name="source_episode_record_ids",
                tool_name="UPSERT_PROFILE_FEATURE",
            ),
            source_episode_record_ids,
        )
        merged_citation_ids = _merged_ids(
            _coerce_string_list(
                existing_payload.get("citation_record_ids"),
                argument_name="citation_record_ids",
                tool_name="UPSERT_PROFILE_FEATURE",
            ),
            _coerce_string_list(
                existing_payload.get("citations"),
                argument_name="citations",
                tool_name="UPSERT_PROFILE_FEATURE",
            ),
            citation_record_ids,
            merged_source_ids,
        )
        created_at = str(existing_payload.get("created_at", "")).strip() or now
        updated = replace(
            existing,
            text=value,
            metadata=_profile_feature_metadata(
                base_metadata={**existing.metadata, **metadata},
                category=category,
                tag=tag,
                feature=feature,
                value=value,
                set_id=set_id,
                source_episode_record_ids=merged_source_ids,
                citation_record_ids=merged_citation_ids,
                created_at=created_at,
                updated_at=now,
                tool_name="UPSERT_PROFILE_FEATURE",
                module_name=module_name,
                module_slot=context.module_slot,
            ),
        )
        context.store.replace_record(existing.layer, existing.record_id, updated)
        return WriteToolResult(
            effects=[
                {
                    "action": "update",
                    "effect_type": "profile_feature_upsert",
                    "record_id": existing.record_id,
                    "layer": existing.layer,
                    "status": "applied",
                    "tool": "UPSERT_PROFILE_FEATURE",
                    "category": category,
                    "tag": tag,
                    "feature": feature,
                    "set_id": set_id,
                    "source_episode_record_ids": merged_source_ids,
                    "citation_record_ids": merged_citation_ids,
                }
            ],
            store=context.store,
        )

    return _execute


def _build_delete_profile_feature_executor(
    *,
    module_name: str,
) -> Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]:
    def _execute(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        record_id = _optional_text(arguments, "record_id")
        if record_id:
            record = find_record_by_id(
                context.store,
                record_id,
                visible_records=context.visible_records,
                restricted=context.module_slot == "memory_evolution",
            )
        else:
            target_layer = _target_layer(context, arguments)
            category = _required_text(arguments, "category", tool_name="DELETE_PROFILE_FEATURE")
            tag = _required_text(arguments, "tag", tool_name="DELETE_PROFILE_FEATURE")
            feature = _required_text(arguments, "feature", tool_name="DELETE_PROFILE_FEATURE")
            set_id = _optional_text(arguments, "set_id")
            record = _find_feature_record(
                context.store.iter_records(target_layer),
                category=category,
                tag=tag,
                feature=feature,
                set_id=set_id,
            )
            if record is None:
                return WriteToolResult(
                    effects=[
                        {
                            "action": "delete",
                            "effect_type": "profile_feature_delete",
                            "record_id": "",
                            "layer": target_layer,
                            "status": "not_found",
                            "tool": "DELETE_PROFILE_FEATURE",
                            "category": category,
                            "tag": tag,
                            "feature": feature,
                            "set_id": set_id,
                            "reason": str(arguments.get("reason", "")).strip(),
                            "module": module_name,
                        }
                    ],
                    store=context.store,
                )
        if record.metadata.get(PROFILE_FEATURE_METADATA_KEY) is None:
            raise ValueError("DELETE_PROFILE_FEATURE target is not a profile feature record.")

        removed = context.store.delete_record(record.layer, record.record_id)
        payload = _profile_payload(removed)
        return WriteToolResult(
            effects=[
                {
                    "action": "delete",
                    "effect_type": "profile_feature_delete",
                    "record_id": removed.record_id,
                    "layer": removed.layer,
                    "status": "applied",
                    "tool": "DELETE_PROFILE_FEATURE",
                    "category": str(payload.get("category", "")),
                    "tag": str(payload.get("tag", "")),
                    "feature": str(payload.get("feature", "")),
                    "set_id": str(payload.get("set_id", "")),
                    "reason": str(arguments.get("reason", "")).strip(),
                    "module": module_name,
                }
            ],
            store=context.store,
        )

    return _execute


def _profile_feature_metadata(
    *,
    base_metadata: dict[str, Any],
    category: str,
    tag: str,
    feature: str,
    value: str,
    set_id: str,
    source_episode_record_ids: list[str],
    citation_record_ids: list[str],
    created_at: str,
    updated_at: str,
    tool_name: str,
    module_name: str,
    module_slot: str,
) -> dict[str, Any]:
    profile_feature = {
        "category": category,
        "tag": tag,
        "feature": feature,
        "feature_name": feature,
        "value": value,
        "set_id": set_id,
        "source_episode_record_ids": list(source_episode_record_ids),
        "citation_record_ids": list(citation_record_ids),
        "citations": list(citation_record_ids),
        "created_at": created_at,
        "updated_at": updated_at,
    }
    metadata = {
        **base_metadata,
        PROFILE_FEATURE_METADATA_KEY: profile_feature,
        "category": category,
        "tag": tag,
        "feature": feature,
        "feature_name": feature,
        "value": value,
        "source_episode_record_ids": list(source_episode_record_ids),
        "citation_record_ids": list(citation_record_ids),
        "citations": list(citation_record_ids),
        "llm_tool": {
            **(base_metadata.get("llm_tool", {}) if isinstance(base_metadata.get("llm_tool"), dict) else {}),
            "action": tool_name,
            "module": module_name,
            "module_slot": module_slot,
            "timestamp": updated_at,
        },
    }
    if set_id:
        metadata["set_id"] = set_id
    else:
        metadata.pop("set_id", None)
    return metadata


def _find_feature_record(
    records: list[MemoryRecord],
    *,
    category: str,
    tag: str,
    feature: str,
    set_id: str,
) -> MemoryRecord | None:
    matches = [
        record
        for record in records
        if _feature_key(record) == (
            _normalize_key(set_id),
            _normalize_key(category),
            _normalize_key(tag),
            _normalize_key(feature),
        )
    ]
    return matches[-1] if matches else None


def _feature_key(record: MemoryRecord) -> tuple[str, str, str, str]:
    payload = _profile_payload(record)
    return (
        _normalize_key(payload.get("set_id", record.metadata.get("set_id", ""))),
        _normalize_key(payload.get("category", record.metadata.get("category", ""))),
        _normalize_key(payload.get("tag", record.metadata.get("tag", ""))),
        _normalize_key(
            payload.get(
                "feature",
                payload.get("feature_name", record.metadata.get("feature", record.metadata.get("feature_name", ""))),
            )
        ),
    )


def _profile_payload(record: MemoryRecord) -> dict[str, Any]:
    payload = record.metadata.get(PROFILE_FEATURE_METADATA_KEY, {})
    return dict(payload) if isinstance(payload, dict) else {}


def _target_layer(context: WriteToolCallContext, arguments: dict[str, Any]) -> str:
    target_layer = str(arguments.get("target_layer", "")).strip() or (context.default_target_layer or "")
    if not target_layer:
        raise ValueError("Profile feature tools require target_layer or module target_layer.")
    return target_layer


def _required_text(arguments: dict[str, Any], key: str, *, tool_name: str) -> str:
    value = str(arguments.get(key, "")).strip()
    if not value:
        raise ValueError(f"{tool_name} requires a non-empty {key}.")
    return value


def _optional_text(arguments: dict[str, Any], key: str) -> str:
    return str(arguments.get(key, "")).strip()


def _coerce_string_list(value: Any, *, argument_name: str, tool_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{tool_name} {argument_name} must be an array of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{tool_name} {argument_name} must be an array of strings.")
        normalized = item.strip()
        if normalized:
            result.append(normalized)
    return list(dict.fromkeys(result))


def _merged_ids(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            normalized = str(item).strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
    return merged


def _normalize_key(value: Any) -> str:
    return str(value).strip().casefold()


def _current_unit_id(context: WriteToolCallContext) -> str:
    if context.packet.units is None or len(context.packet.units) != 1:
        return ""
    return context.packet.units[0].unit_id


def _current_timestamp(context: WriteToolCallContext) -> str:
    if context.packet.units is None or len(context.packet.units) != 1:
        return ""
    return context.packet.units[0].timestamp


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
