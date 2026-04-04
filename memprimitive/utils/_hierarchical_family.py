from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..core import MemoryRecord, MemoryStore, Observation, Packet, Placement
from ..pipeline import MemoryPipeline
from ._template import (
    PromptPlan,
    ensure_prompt_plan,
    project_record_for_template,
    render_prompt_plan,
)

_PSEUDO_FIELDS = frozenset({"record_id", "unit_id", "layer", "text", "timestamp"})
_VALID_EXTRACT_MODES = frozenset({"copy", "generate"})


def validate_hierarchical_config(
    *,
    source_layer: str,
    target_layer: str | None,
    memory_pipeline: MemoryPipeline | None,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    prompt: PromptPlan | str | None,
) -> dict[str, Any]:
    normalized_source = str(source_layer).strip()
    normalized_target = None if target_layer is None else str(target_layer).strip()
    normalized_mode = str(extract_mode).strip()
    normalized_fields = tuple(str(field).strip() for field in extract_fields if str(field).strip())
    normalized_group_by = tuple(str(field).strip() for field in group_by if str(field).strip())

    if not normalized_source:
        raise ValueError("source_layer must be a non-empty string.")
    if normalized_mode not in _VALID_EXTRACT_MODES:
        raise ValueError("extract_mode must be one of: copy, generate.")
    if not normalized_fields:
        raise ValueError("extract_fields must contain at least one non-empty field name.")
    if prompt is not None and normalized_mode != "generate":
        raise ValueError("prompt is only supported when extract_mode='generate'.")

    has_target_layer = normalized_target is not None and normalized_target != ""
    has_memory_pipeline = memory_pipeline is not None
    if has_target_layer == has_memory_pipeline:
        raise ValueError("Exactly one of target_layer or memory_pipeline must be provided.")
    if has_memory_pipeline and not isinstance(memory_pipeline, MemoryPipeline):
        raise ValueError("memory_pipeline must be a MemoryPipeline instance.")
    if normalized_target == "":
        raise ValueError("target_layer must be a non-empty string when provided.")

    return {
        "source_layer": normalized_source,
        "target_layer": normalized_target,
        "memory_pipeline": memory_pipeline,
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


def build_fixed_placements(
    packet: Packet,
    *,
    target_layer: str | None,
    memory_pipeline: MemoryPipeline | None,
) -> list[Placement]:
    placement_target = inferred_target_layer(target_layer=target_layer, memory_pipeline=memory_pipeline)
    return [Placement(unit_id=unit.unit_id, target_layer=placement_target) for unit in packet.units or []]


def inferred_target_layer(*, target_layer: str | None, memory_pipeline: MemoryPipeline | None) -> str:
    if target_layer is not None:
        return target_layer
    if memory_pipeline is not None:
        organization = getattr(memory_pipeline, "organization", None)
        inferred = _extract_target_layer_from_organization(organization)
        if inferred:
            return inferred
    return "pipeline_managed"


def _extract_target_layer_from_organization(organization: Any) -> str | None:
    if organization is None:
        return None
    target_layer = getattr(organization, "target_layer", None)
    if isinstance(target_layer, str) and target_layer.strip():
        return target_layer.strip()
    if isinstance(organization, tuple):
        for child in organization:
            inferred = _extract_target_layer_from_organization(child)
            if inferred:
                return inferred
    if hasattr(organization, "iter_child_modules"):
        for child in organization.iter_child_modules():
            inferred = _extract_target_layer_from_organization(child)
            if inferred:
                return inferred
    return None


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

    records_by_id = {record.record_id: record for record in store.iter_records(source_layer)}
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
        payload[field] = ordered_unique[0] if len(ordered_unique) == 1 else ordered_unique
    return payload


def generate_payload(
    records: list[MemoryRecord],
    *,
    store: MemoryStore,
    source_layer: str,
    target_layer: str,
    extract_fields: tuple[str, ...],
    group_key: dict[str, Any],
    prompt: PromptPlan | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ._runtime import get_runtime

    runtime = get_runtime()
    runtime.require_llm(capability="Hierarchical generate-mode abstraction")
    system_prompt, prompt_trace = build_generation_system_prompt(
        store=store,
        source_layer=source_layer,
        target_layer=target_layer,
        extract_fields=extract_fields,
        group_key=group_key,
        records=records,
        prompt=prompt,
    )
    result = runtime.json(
        system=system_prompt,
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
    return {
        field: result.get(field) for field in extract_fields
    }, prompt_trace


def build_generation_system_prompt(
    *,
    store: MemoryStore,
    source_layer: str,
    target_layer: str,
    extract_fields: tuple[str, ...],
    group_key: dict[str, Any],
    records: list[MemoryRecord],
    prompt: PromptPlan | str | None,
) -> tuple[str, dict[str, Any]]:
    if prompt is not None:
        plan = ensure_prompt_plan(
            prompt,
            metadata_mode="prompt",
            context_builder=lambda packet, current_store: {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "extract_fields": list(extract_fields),
                "group_key": dict(group_key),
                "records": [project_record_for_template(record) for record in records],
                "record_count": len(records),
            },
        )
        return render_prompt_plan(plan, packet=Packet(trace={}), store=store)[:2]
    fields_text = ", ".join(extract_fields)
    return (
        "You aggregate selected source memory records into a higher-level hierarchical memory record. "
        f"Return strict JSON with exactly these top-level keys: {fields_text}. "
        "Each field should summarize shared or higher-level information across the provided records."
    ), {
        "template_mode": "simple",
        "prompt_is_template": True,
        "rendered_prompt": "",
        "rendered_prompt_preview": "",
        "missing_variables": [],
        "recalled_prompt": "",
        "recalled_prompt_preview": "",
        "recall_prompt": {
            "enabled": False,
            "disabled_reason": "default_prompt",
            "rendered_recall_query": "",
            "rendered_recall_query_preview": "",
            "matched": False,
            "recalled_prompt": "",
            "recalled_prompt_preview": "",
            "missing_variables": [],
            "resolved_variables": [],
            "used_record_ids": [],
            "used_group_ids": [],
            "filter_trace": [],
        },
        "labeled_recalled_prompts": {},
        "labeled_recalled_prompt_previews": {},
        "labeled_recall_prompts": {},
    }


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


def build_hierarchical_metadata(
    *,
    source_layer: str,
    target_layer: str,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    group_key: dict[str, Any],
    records: list[MemoryRecord],
    field_payload: dict[str, Any],
    ) -> dict[str, Any]:
    return {
        "hierarchical": {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "extract_mode": extract_mode,
            "extract_fields": list(extract_fields),
            "group_by": list(group_by),
            "group_key": dict(group_key),
            "source_record_ids": [record.record_id for record in records],
            "source_unit_ids": list(dict.fromkeys(record.unit_id for record in records)),
            "field_payload": field_payload,
            "relation": "hierarchical_source",
        }
    }


def build_extracted_triple_metadata(
    *,
    source_layer: str,
    target_layer: str,
    source_record: MemoryRecord,
    triples: list[tuple[str, str, str]],
) -> dict[str, Any]:
    return {
        "hierarchical": {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "extract_mode": "copy",
            "extract_fields": ["triples"],
            "group_by": [],
            "group_key": {},
            "source_record_ids": [source_record.record_id],
            "source_unit_ids": [source_record.unit_id],
            "field_payload": {"triples": list(triples)},
            "relation": "hierarchical_extracted_triple",
        }
    }


def build_hierarchical_observation(
    *,
    source_layer: str,
    target_layer: str,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    group_key: dict[str, Any],
    records: list[MemoryRecord],
    field_payload: dict[str, Any],
) -> Observation:
    timestamp = records[-1].timestamp if records else None
    return Observation(
        text=render_record_text(field_payload, extract_fields=extract_fields),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        source="hierarchical",
        metadata=build_hierarchical_metadata(
            source_layer=source_layer,
            target_layer=target_layer,
            extract_mode=extract_mode,
            extract_fields=extract_fields,
            group_by=group_by,
            group_key=group_key,
            records=records,
            field_payload=field_payload,
        ),
    )


def append_hierarchical_records(
    store: MemoryStore,
    *,
    source_layer: str,
    target_layer: str | None,
    memory_pipeline: MemoryPipeline | None,
    extract_mode: str,
    extract_fields: tuple[str, ...],
    group_by: tuple[str, ...],
    grouped_records: list[dict[str, Any]],
    prompt: PromptPlan | str | None,
) -> tuple[list[dict[str, Any]], str]:
    child_pipeline, writer_pipeline_mode = resolve_writer_pipeline(
        target_layer=target_layer,
        memory_pipeline=memory_pipeline,
    )
    effective_target_layer = inferred_target_layer(target_layer=target_layer, memory_pipeline=child_pipeline)
    effects: list[dict[str, Any]] = []
    for group in grouped_records:
        records = list(group["records"])
        group_key = dict(group["group_key"])
        if extract_mode == "copy":
            field_payload = aggregate_copy_payload(records, extract_fields=extract_fields)
            prompt_trace = None
        else:
            field_payload, prompt_trace = generate_payload(
                records,
                store=store,
                source_layer=source_layer,
                target_layer=effective_target_layer,
                extract_fields=extract_fields,
                group_key=group_key,
                prompt=prompt,
            )
        observation = build_hierarchical_observation(
            source_layer=source_layer,
            target_layer=effective_target_layer,
            extract_mode=extract_mode,
            extract_fields=extract_fields,
            group_by=group_by,
            group_key=group_key,
            records=records,
            field_payload=field_payload,
        )
        child_packet = child_ingest(child_pipeline, store, observation=observation)
        written_record_ids = extract_written_record_ids(child_packet)
        effects.append(
            {
                "effect_type": "hierarchical_pipeline_ingest",
                "record_id": written_record_ids[-1],
                "written_record_ids": written_record_ids,
                "target_layer": effective_target_layer,
                "group_key": group_key,
                "source_record_ids": [source.record_id for source in records],
                "sub_ingest_trace": summarize_sub_ingest_trace(child_packet),
                "prompt_trace": prompt_trace,
            }
        )
    return effects, writer_pipeline_mode


def resolve_writer_pipeline(
    *,
    target_layer: str | None,
    memory_pipeline: MemoryPipeline | None,
) -> tuple[MemoryPipeline, str]:
    if memory_pipeline is not None:
        return memory_pipeline, "provided"

    from ..pipeline import create_baseline_pipeline
    from ..baselines.organization import AppendOrganization

    if target_layer is None:
        raise ValueError("target_layer is required when memory_pipeline is not provided.")
    pipeline = create_baseline_pipeline()
    pipeline.organization = AppendOrganization(target_layer=target_layer)
    return pipeline, "default_target_layer"


def child_ingest(child_pipeline: MemoryPipeline, store: MemoryStore, *, observation: Observation) -> Packet:
    child_pipeline.store = store
    return child_pipeline.ingest(observation)


def extract_written_record_ids(packet: Packet) -> list[str]:
    organization_trace = packet.trace.get("organization", {})
    if not isinstance(organization_trace, dict):
        raise ValueError("Child pipeline ingest trace missing organization trace.")
    written_record_ids = organization_trace.get("written_record_ids")
    if not isinstance(written_record_ids, list) or not all(isinstance(item, str) and item.strip() for item in written_record_ids):
        raise ValueError("Child pipeline ingest trace missing organization.written_record_ids.")
    return [item.strip() for item in written_record_ids]


def summarize_sub_ingest_trace(packet: Packet) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for slot in ("unit_formation", "representation", "write_trigger", "organization", "evolution_trigger", "memory_evolution"):
        slot_trace = packet.trace.get(slot)
        if not isinstance(slot_trace, dict):
            continue
        slot_summary = {"module": slot_trace.get("module")}
        if slot == "organization":
            slot_summary["written_record_ids"] = list(slot_trace.get("written_record_ids", []))
            if "target_layer" in slot_trace:
                slot_summary["target_layer"] = slot_trace.get("target_layer")
        summary[slot] = slot_summary
    return summary
