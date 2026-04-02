from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ..core import MemoryRecord, MemoryStore, MemoryUnit, Packet

_PSEUDO_FIELDS = frozenset({"record_id", "unit_id", "layer", "text", "timestamp"})
_VALID_EXTRACT_MODES = frozenset({"copy", "generate"})


def validate_hierarchical_config(
    *,
    source_layer: str,
    target_layer: str,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    prompt: str | None,
) -> dict[str, Any]:
    normalized_source = str(source_layer).strip()
    normalized_target = str(target_layer).strip()
    normalized_mode = str(extract_mode).strip()
    normalized_fields = tuple(str(field).strip() for field in extract_fields if str(field).strip())
    normalized_group_by = tuple(str(field).strip() for field in group_by if str(field).strip())

    if not normalized_source:
        raise ValueError("source_layer must be a non-empty string.")
    if not normalized_target:
        raise ValueError("target_layer must be a non-empty string.")
    if normalized_mode not in _VALID_EXTRACT_MODES:
        raise ValueError("extract_mode must be one of: copy, generate.")
    if not normalized_fields:
        raise ValueError("extract_fields must contain at least one non-empty field name.")
    if prompt is not None and normalized_mode != "generate":
        raise ValueError("prompt is only supported when extract_mode='generate'.")
    return {
        "source_layer": normalized_source,
        "target_layer": normalized_target,
        "extract_mode": normalized_mode,
        "extract_fields": normalized_fields,
        "group_by": normalized_group_by,
        "prompt": prompt,
    }


def require_aligned_units_decisions(packet: Packet, *, include_placements: bool) -> None:
    if packet.units is None:
        raise ValueError("Hierarchical modules require packet.units.")
    if packet.decisions is None:
        raise ValueError("Hierarchical modules require packet.decisions.")
    if include_placements and packet.placements is None:
        raise ValueError("HierarchicalEvolution requires packet.placements.")
    if include_placements:
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError("HierarchicalEvolution requires aligned units, decisions, and placements.")
    else:
        if len(packet.units) != len(packet.decisions):
            raise ValueError("HierarchicalOrganization requires decisions aligned with units.")


def build_fixed_placements(packet: Packet, *, target_layer: str):
    return [replace_placement_target(unit.unit_id, target_layer) for unit in packet.units or []]


def replace_placement_target(unit_id: str, target_layer: str):
    from ..core import Placement

    return Placement(unit_id=unit_id, target_layer=target_layer)


def resolve_source_records(
    packet: Packet,
    store: MemoryStore,
    *,
    source_layer: str,
) -> tuple[list[MemoryRecord], str]:
    if packet.decisions_store is None:
        return store.iter_records(source_layer), "source_layer_scan"

    layer_entry = packet.decisions_store.get(source_layer)
    if not isinstance(layer_entry, dict):
        return [], "decisions_store"

    record_ids = [str(record_id).strip() for record_id in layer_entry.get("record_ids", []) if str(record_id).strip()]
    if not record_ids:
        return [], "decisions_store"

    records_by_id = {
        record.record_id: record
        for record in store.iter_records(source_layer)
    }
    selected = [records_by_id[record_id] for record_id in record_ids if record_id in records_by_id]
    return selected, "decisions_store"


def resolve_record_field(record: MemoryRecord, field: str) -> Any:
    normalized = str(field).strip()
    if normalized in _PSEUDO_FIELDS:
        return getattr(record, normalized)
    if normalized in record.metadata:
        return record.metadata.get(normalized)
    representation = record.metadata.get("representation")
    if isinstance(representation, dict) and normalized in representation:
        return representation.get(normalized)
    return None


def group_records(records: list[MemoryRecord], *, group_by: tuple[str, ...]) -> list[dict[str, Any]]:
    if not group_by:
        return [{"group_key": {}, "records": list(records)}] if records else []

    grouped: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for record in records:
        group_key = {field: resolve_record_field(record, field) for field in group_by}
        key_token = json.dumps(group_key, ensure_ascii=False, sort_keys=True, default=str)
        if key_token not in seen:
            seen[key_token] = len(grouped)
            grouped.append({"group_key": group_key, "records": [record]})
            continue
        grouped[seen[key_token]]["records"].append(record)
    return grouped


def aggregate_copy_payload(records: list[MemoryRecord], *, extract_fields: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in extract_fields:
        ordered_unique: list[Any] = []
        seen: set[str] = set()
        for record in records:
            value = resolve_record_field(record, field)
            token = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if token in seen:
                continue
            seen.add(token)
            ordered_unique.append(value)
        if len(ordered_unique) == 1:
            payload[field] = ordered_unique[0]
        else:
            payload[field] = ordered_unique
    return payload


def generate_payload(
    records: list[MemoryRecord],
    *,
    source_layer: str,
    target_layer: str,
    extract_fields: tuple[str, ...],
    group_key: dict[str, Any],
    prompt: str | None,
) -> dict[str, Any]:
    from ._runtime import get_runtime

    runtime = get_runtime()
    runtime.require_llm(capability="Hierarchical generate-mode abstraction")
    result = runtime.json(
        system=_generation_system_prompt(extract_fields, prompt=prompt),
        user=json.dumps(
            {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "group_key": group_key,
                "extract_fields": list(extract_fields),
                "records": [serialize_source_record(record) for record in records],
            },
            ensure_ascii=False,
        ),
    )
    if not isinstance(result, dict):
        raise ValueError("Hierarchical generate-mode must return a JSON object.")
    return {field: result.get(field) for field in extract_fields}


def _generation_system_prompt(extract_fields: tuple[str, ...], *, prompt: str | None) -> str:
    if prompt is not None:
        return str(prompt)
    fields_text = ", ".join(extract_fields)
    return (
        "You aggregate selected source memory records into a higher-level hierarchical memory record. "
        f"Return strict JSON with exactly these top-level keys: {fields_text}. "
        "Each field should summarize shared or higher-level information across the provided records."
    )


def serialize_source_record(record: MemoryRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "unit_id": record.unit_id,
        "layer": record.layer,
        "text": record.text,
        "timestamp": record.timestamp,
        "metadata": record.metadata,
    }


def render_record_text(field_payload: dict[str, Any], *, extract_fields: tuple[str, ...]) -> str:
    if len(extract_fields) == 1:
        return render_value(field_payload.get(extract_fields[0]))
    return "\n".join(f"{field}: {render_value(field_payload.get(field))}" for field in extract_fields)


def render_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_hierarchical_unit(
    *,
    source_layer: str,
    target_layer: str,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    group_key: dict[str, Any],
    records: list[MemoryRecord],
    field_payload: dict[str, Any],
) -> MemoryUnit:
    source_record_ids = [record.record_id for record in records]
    source_unit_ids = list(dict.fromkeys(record.unit_id for record in records))
    timestamp = records[-1].timestamp if records else None
    metadata = {
        "hierarchical": {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "extract_mode": extract_mode,
            "extract_fields": list(extract_fields),
            "group_by": list(group_by),
            "group_key": dict(group_key),
            "source_record_ids": source_record_ids,
            "source_unit_ids": source_unit_ids,
            "field_payload": field_payload,
            "relation": "hierarchical_source",
        }
    }
    unit = MemoryUnit(
        text=render_record_text(field_payload, extract_fields=extract_fields),
        unit_type="hierarchical",
        metadata=metadata,
    )
    if timestamp is not None:
        unit = replace(unit, timestamp=timestamp)
    return unit


def append_hierarchical_records(
    store: MemoryStore,
    *,
    source_layer: str,
    target_layer: str,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    grouped_records: list[dict[str, Any]],
    prompt: str | None,
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for group in grouped_records:
        records = list(group["records"])
        group_key = dict(group["group_key"])
        if extract_mode == "copy":
            field_payload = aggregate_copy_payload(records, extract_fields=extract_fields)
        else:
            field_payload = generate_payload(
                records,
                source_layer=source_layer,
                target_layer=target_layer,
                extract_fields=extract_fields,
                group_key=group_key,
                prompt=prompt,
            )
        unit = build_hierarchical_unit(
            source_layer=source_layer,
            target_layer=target_layer,
            extract_mode=extract_mode,
            extract_fields=extract_fields,
            group_by=group_by,
            group_key=group_key,
            records=records,
            field_payload=field_payload,
        )
        sequence_id = store.next_sequence_id()
        record = MemoryRecord.from_unit(unit=unit, layer=target_layer, sequence_id=sequence_id)
        store.append(record)
        effects.append(
            {
                "effect_type": "hierarchical_append",
                "record_id": record.record_id,
                "target_layer": target_layer,
                "group_key": group_key,
                "source_record_ids": [source.record_id for source in records],
            }
        )
    return effects
