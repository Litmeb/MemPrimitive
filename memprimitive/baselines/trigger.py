"""Shared baseline trigger implementations."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import TriggerModule
from ..utils._runtime import get_runtime
from ..utils._template import PromptPlan, ensure_prompt_plan, render_prompt_plan
from ..utils._trace import copy_trace

_VALID_TRIGGER_SLOTS = frozenset({"write_trigger", "evolution_trigger"})
_VALID_MATCH_MODES = frozenset({"any", "all"})
_VALID_COMPARATORS = frozenset({">", ">=", "<", "<=", "=="})
_VALID_SIGNAL_SOURCES = frozenset({"auto", "observation", "trace", "store", "unit_metadata"})
_VALID_AGGREGATES = frozenset({"broadcast", "per_unit", "any_unit", "all_units"})
_VALID_DECISION_MODES = frozenset({"bool", "score", "label"})
_VALID_MAINTENANCE_MODES = frozenset({"broadcast", "per_unit"})
_POSITIVE_LABELS = frozenset({"allow", "go", "ok", "positive", "true", "write", "evolve", "yes", "y", "trigger"})
_BOUNDARY_KIND_TO_MATCH_KEY = {
    "session": "session_id",
    "turn": "turn_id",
    "chunk": "chunk_id",
    "subgoal": "subgoal_id",
    "episode": "episode_id",
}
_BOUNDARY_EVENT_ALIASES = {
    "session": frozenset({"session", "session_boundary", "session_end"}),
    "turn": frozenset({"turn", "turn_boundary", "turn_end"}),
    "chunk": frozenset({"chunk", "chunk_boundary", "chunk_end"}),
    "subgoal": frozenset({"subgoal", "subgoal_end", "subgoal_complete"}),
    "episode": frozenset({"episode", "episode_boundary", "episode_end"}),
}
_BOUNDARY_EVENT_TO_KIND = {
    alias: kind
    for kind, aliases in _BOUNDARY_EVENT_ALIASES.items()
    for alias in aliases
}


def _require_trigger_slot(slot: str) -> str:
    normalized = str(slot).strip()
    if normalized not in _VALID_TRIGGER_SLOTS:
        options = ", ".join(sorted(_VALID_TRIGGER_SLOTS))
        raise ValueError(f"trigger slot must be one of: {options}.")
    return normalized


def _require_choice(value: str, *, field_name: str, allowed: frozenset[str]) -> str:
    normalized = str(value).strip()
    if normalized not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {options}.")
    return normalized


def _module_name(prefix: str, slot: str) -> str:
    stem = "write" if slot == "write_trigger" else "evolution"
    return f"{prefix}_{stem}_trigger"


def _input_requirements(slot: str) -> tuple[str, ...]:
    if slot == "write_trigger":
        return ("units",)
    return ("units", "placements")


def _require_units(packet: Packet, *, module_name: str, slot: str) -> list:
    if packet.units is None:
        raise ValueError(f"{module_name} requires packet.units.")
    if slot == "evolution_trigger":
        if packet.placements is None:
            raise ValueError(f"{module_name} requires packet.placements.")
        if len(packet.units) != len(packet.placements):
            raise ValueError(f"{module_name} requires aligned packet.units and packet.placements.")
        for unit, placement in zip(packet.units, packet.placements, strict=True):
            if placement.unit_id != unit.unit_id:
                raise ValueError(f"{module_name} requires aligned packet.units and packet.placements.")
    return packet.units


def _broadcast_decisions(length: int, decision: bool) -> list[bool]:
    return [bool(decision)] * length


def _trigger_payload(packet: Packet) -> dict[str, Any]:
    observation = packet.observation
    if observation is None:
        return {}
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    trigger = metadata.get("trigger", {})
    return trigger if isinstance(trigger, dict) else {}


def _resolve_events(packet: Packet) -> list[str]:
    payload = _trigger_payload(packet)
    raw_events = payload.get("events")
    if isinstance(raw_events, (list, tuple)):
        cleaned = [str(item).strip() for item in raw_events if str(item).strip()]
        if cleaned:
            return cleaned
    if packet.events:
        return [str(item).strip() for item in packet.events if str(item).strip()]
    return []


def _resolve_schedule(packet: Packet) -> dict[str, Any]:
    payload = _trigger_payload(packet)
    schedule = payload.get("schedule", {})
    return schedule if isinstance(schedule, dict) else {}


def _resolve_trace_signals(packet: Packet) -> dict[str, Any]:
    trace_signals = packet.trace.get("signals", {})
    return trace_signals if isinstance(trace_signals, dict) else {}


def _resolve_store_signals(store: MemoryStore) -> dict[str, Any]:
    trigger_signals = store.metadata.get("trigger_signals", {})
    return trigger_signals if isinstance(trigger_signals, dict) else {}


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _compare_scalar(value: float, threshold: float, comparator: str) -> bool:
    if comparator == ">":
        return value > threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "==":
        return value == threshold
    raise ValueError(f"Unsupported comparator: {comparator!r}")


def _unit_signal_values(packet: Packet, signal_key: str) -> list[float | None]:
    values: list[float | None] = []
    for unit in packet.units or []:
        values.append(_coerce_numeric(unit.metadata.get(signal_key)))
    return values


def _global_signal_value(packet: Packet, store: MemoryStore, signal_key: str, signal_source: str) -> float | None:
    payload = _trigger_payload(packet)
    sources: list[dict[str, Any]] = []
    if signal_source == "auto":
        sources = [
            payload.get("signals", {}),
            _resolve_trace_signals(packet),
            _resolve_store_signals(store),
        ]
    elif signal_source == "observation":
        sources = [payload.get("signals", {})]
    elif signal_source == "trace":
        sources = [_resolve_trace_signals(packet)]
    elif signal_source == "store":
        sources = [_resolve_store_signals(store)]
    else:
        return None

    for source in sources:
        if not isinstance(source, dict) or signal_key not in source:
            continue
        numeric = _coerce_numeric(source.get(signal_key))
        if numeric is not None:
            return numeric
    return None


def _resolve_pressure_layer(
    packet: Packet,
    *,
    slot: str,
    explicit_target_layer: str | None,
    unit_index: int | None = None,
) -> tuple[str, str]:
    if explicit_target_layer is not None:
        normalized_layer = str(explicit_target_layer).strip()
        if not normalized_layer:
            raise ValueError("target_layer must be non-empty when provided.")
        return normalized_layer, "explicit"

    if slot == "write_trigger":
        raise ValueError("memory_pressure requires explicit target_layer for write_trigger.")

    if packet.placements is None or not packet.placements:
        raise ValueError("memory_pressure requires packet.placements for evolution_trigger.")

    if unit_index is not None:
        if unit_index >= len(packet.placements):
            raise ValueError("memory_pressure requires aligned packet.placements for per-unit resolution.")
        return packet.placements[unit_index].target_layer, "placement"

    target_layers = {placement.target_layer for placement in packet.placements}
    if len(target_layers) != 1:
        raise ValueError("memory_pressure broadcast resolution requires a single target layer in packet.placements.")
    return next(iter(target_layers)), "placement"


def _layer_pressure_snapshot(store: MemoryStore, layer: str) -> dict[str, Any]:
    spec = store.layer_spec(layer)
    record_count = store.count(layer)
    token_count = store.layer_token_count(layer)

    record_budget_raw = spec.get_setting("record_budget")
    token_budget_raw = spec.get_setting("token_budget")
    record_budget = record_budget_raw if isinstance(record_budget_raw, int) and record_budget_raw > 0 else None
    token_budget = token_budget_raw if isinstance(token_budget_raw, int) and token_budget_raw > 0 else None

    record_pressure = (record_count / record_budget) if record_budget is not None else None
    token_pressure = (token_count / token_budget) if token_budget is not None else None

    pressure_values = [value for value in (record_pressure, token_pressure) if value is not None]
    active_budget_types: list[str] = []
    if record_budget is not None:
        active_budget_types.append("record_budget")
    if token_budget is not None:
        active_budget_types.append("token_budget")

    return {
        "target_layer": layer,
        "record_count": record_count,
        "token_count": token_count,
        "record_budget": record_budget,
        "token_budget": token_budget,
        "record_pressure": record_pressure,
        "token_pressure": token_pressure,
        "memory_pressure": max(pressure_values) if pressure_values else None,
        "active_budget_types": active_budget_types,
        "capacity": spec.capacity,
    }


def _pressure_trace_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_layer": snapshot["target_layer"],
        "record_count": snapshot["record_count"],
        "token_count": snapshot["token_count"],
        "record_budget": snapshot["record_budget"],
        "token_budget": snapshot["token_budget"],
        "record_pressure": snapshot["record_pressure"],
        "token_pressure": snapshot["token_pressure"],
        "memory_pressure": snapshot["memory_pressure"],
        "active_budget_types": list(snapshot["active_budget_types"]),
        "capacity": snapshot["capacity"],
    }


def _normalized_boundary_event_name(event: str) -> str:
    normalized = str(event).strip()
    if not normalized:
        return ""
    return _BOUNDARY_EVENT_TO_KIND.get(normalized, normalized)


def _observation_metadata(packet: Packet) -> dict[str, Any]:
    if packet.observation is None or not isinstance(packet.observation.metadata, dict):
        return {}
    return packet.observation.metadata


def _resolve_boundary_match_value(packet: Packet, match_key: str) -> Any:
    payload = _trigger_payload(packet)
    if match_key in payload:
        return payload.get(match_key)

    observation_metadata = _observation_metadata(packet)
    if match_key in observation_metadata:
        return observation_metadata.get(match_key)

    unit_values: list[Any] = []
    for unit in packet.units or []:
        if match_key in unit.metadata:
            unit_values.append(unit.metadata.get(match_key))
    if unit_values:
        first_value = unit_values[0]
        if all(value == first_value for value in unit_values[1:]):
            return first_value
    return None


def _build_store_decisions_summary(decisions_store: dict[str, dict[str, Any]] | None) -> tuple[list[str], dict[str, int]]:
    if not decisions_store:
        return [], {}
    layers = list(decisions_store)
    counts = {
        layer: len(entry.get("record_ids", []))
        for layer, entry in decisions_store.items()
    }
    return layers, counts


def _select_store_records_by_metadata(
    store: MemoryStore,
    *,
    event: str,
    boundary_kind: str,
    match_key: str,
    match_value: Any,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for layer in store.topology.layer_names:
        record_ids = [
            record.record_id
            for record in store.iter_records(layer)
            if record.metadata.get(match_key) == match_value
        ]
        if not record_ids:
            continue
        out[layer] = {
            "decision": True,
            "record_ids": record_ids,
            "selector": {
                "kind": "boundary_match",
                "event": event,
                "boundary_kind": boundary_kind,
                "match_key": match_key,
                "match_value": match_value,
                "count": len(record_ids),
            },
        }
    return out


def _select_store_records_for_layer(
    store: MemoryStore,
    *,
    layer: str,
    selector: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    record_ids = [record.record_id for record in store.iter_records(layer)]
    return {
        layer: {
            "decision": True,
            "record_ids": record_ids,
            "selector": {
                **selector,
                "count": len(record_ids),
            },
        }
    }


def _write_trace(
    packet: Packet,
    *,
    module_name: str,
    trace_key: str,
    decisions: list[bool] | None,
    decisions_store: dict[str, dict[str, Any]] | None = None,
    trace_payload: dict[str, Any] | None = None,
    per_unit_payload: list[dict[str, Any]] | None = None,
) -> Packet:
    trace = copy_trace(packet)
    per_unit = []
    if decisions is not None:
        for index, (unit, decision) in enumerate(zip(packet.units or [], decisions, strict=True)):
            item = {
                "unit_id": unit.unit_id,
                "decision": bool(decision),
            }
            if per_unit_payload is not None and index < len(per_unit_payload):
                item.update(per_unit_payload[index])
            per_unit.append(item)
    payload = {
        "module": module_name,
        "decisions": None if decisions is None else list(decisions),
        "per_unit": per_unit,
    }
    decisions_store_layers, decisions_store_counts = _build_store_decisions_summary(decisions_store)
    payload["decisions_store_layers"] = decisions_store_layers
    payload["decisions_store_counts"] = decisions_store_counts
    if trace_payload:
        payload.update(trace_payload)
    trace[trace_key] = payload
    next_decisions_store = decisions_store if decisions_store is not None else packet.decisions_store
    next_decisions = packet.decisions if decisions is None else list(decisions)
    return replace(packet, decisions=next_decisions, decisions_store=next_decisions_store, trace=trace)


def _select_all_store_records(store: MemoryStore, *, selector: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for layer in store.topology.layer_names:
        record_ids = [record.record_id for record in store.iter_records(layer)]
        if not record_ids:
            continue
        out[layer] = {
            "decision": True,
            "record_ids": record_ids,
            "selector": {
                **selector,
                "layer": layer,
                "count": len(record_ids),
            },
        }
    return out


def _normalize_llm_judge_output(output: Any, *, decision_mode: str, threshold: float) -> tuple[bool, dict[str, Any]]:
    raw = output
    if isinstance(output, dict):
        decision = output.get("decision")
        score = output.get("score")
        label = output.get("label")
    else:
        decision = output if decision_mode == "bool" else None
        score = output if decision_mode == "score" else None
        label = output if decision_mode == "label" else None

    normalized_score = _coerce_numeric(score)
    normalized_label = str(label).strip() if label is not None else ""

    if decision_mode == "bool":
        if isinstance(decision, bool):
            verdict = decision
        elif normalized_score is not None:
            verdict = normalized_score >= threshold
        elif normalized_label:
            verdict = normalized_label.casefold() in _POSITIVE_LABELS
        else:
            verdict = bool(decision)
    elif decision_mode == "score":
        if normalized_score is None:
            normalized_score = _coerce_numeric(decision)
        if normalized_score is None:
            raise ValueError("LLMJudgeTrigger in score mode requires a numeric score output.")
        verdict = normalized_score >= threshold
    else:
        if not normalized_label:
            normalized_label = str(decision).strip()
        verdict = normalized_label.casefold() in _POSITIVE_LABELS

    details = {
        "raw_output": raw,
        "score": normalized_score,
        "label": normalized_label or None,
        "decision_mode": decision_mode,
        "threshold": float(threshold),
    }
    return bool(verdict), details


def _judge_payload(packet: Packet, store: MemoryStore, unit_index: int | None = None) -> dict[str, Any]:
    observation = None
    if packet.observation is not None:
        observation = {
            "observation_id": packet.observation.observation_id,
            "text": packet.observation.text,
            "source": packet.observation.source,
            "timestamp": packet.observation.timestamp,
            "metadata": packet.observation.metadata,
        }

    unit = None
    placement = None
    if unit_index is not None and packet.units is not None:
        current_unit = packet.units[unit_index]
        unit = {
            "unit_id": current_unit.unit_id,
            "text": current_unit.text,
            "unit_type": current_unit.unit_type,
            "timestamp": current_unit.timestamp,
            "metadata": current_unit.metadata,
            "representation_elements": list(current_unit.representation_elements),
        }
        if packet.placements is not None and unit_index < len(packet.placements):
            current_placement = packet.placements[unit_index]
            placement = {
                "unit_id": current_placement.unit_id,
                "target_layer": current_placement.target_layer,
            }

    store_summary = {
        "layer_names": list(store.topology.layer_names),
        "record_count": store.count(),
        "metadata": store.metadata,
    }

    return {
        "observation": observation,
        "unit": unit,
        "placement": placement,
        "store_summary": store_summary,
        "trigger_context": {
            "events": _resolve_events(packet),
            "signals": _trigger_payload(packet).get("signals", {}),
            "schedule": _resolve_schedule(packet),
            "trace": packet.trace,
        },
    }


class _BaseTrigger(TriggerModule):
    _DEFAULT_SLOT: str | None = None
    _NAME_PREFIX: str = ""

    def __init__(self, *, slot: str | None = None) -> None:
        resolved_slot = _require_trigger_slot(slot or self._DEFAULT_SLOT or "")
        self.slot = resolved_slot
        self.spec = ModuleSpec(
            name=_module_name(self._NAME_PREFIX, resolved_slot),
            slot=resolved_slot,
            input_requirements=_input_requirements(resolved_slot),
            output_guarantees=("decisions",),
        )

    def _trace_key(self) -> str:
        return self.spec.slot

    def _require_units(self, packet: Packet) -> list:
        return _require_units(packet, module_name=self.spec.name, slot=self.spec.slot)


class AlwaysTrigger(_BaseTrigger):
    """Mark every unit as eligible for the current trigger slot."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "always"

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        decisions = [True] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={"constant": 1.0, "threshold": None, "source": "constant"},
                per_unit_payload=[{"constant": 1.0} for _ in decisions],
            ),
            store,
        )


class NeverTrigger(_BaseTrigger):
    """Mark every unit as ineligible for the current trigger slot."""

    _DEFAULT_SLOT = "evolution_trigger"
    _NAME_PREFIX = "never"

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        decisions = [False] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={"constant": 1.0, "threshold": None, "source": "constant"},
                per_unit_payload=[{"constant": 1.0} for _ in decisions],
            ),
            store,
        )


class StoreAllTrigger(_BaseTrigger):
    """Select every existing record across all store layers without rewriting decisions."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "store_all"

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        self._require_units(packet)
        decisions_store = _select_all_store_records(
            store,
            selector={
                "kind": "store_all",
                "source": "store_all_trigger",
            },
        )
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=packet.decisions,
                decisions_store=decisions_store,
                trace_payload={
                    "source": "store_all",
                    "preserved_decisions": packet.decisions is not None,
                },
            ),
            store,
        )


class _EventTrigger(_BaseTrigger):
    _EVENT_SOURCE = "events"

    def __init__(
        self,
        *,
        slot: str | None = None,
        accepted_events: tuple[str, ...],
        match_mode: str = "any",
        default_decision: bool = False,
        invert: bool = False,
    ) -> None:
        super().__init__(slot=slot)
        normalized_events = tuple(str(item).strip() for item in accepted_events if str(item).strip())
        if not normalized_events:
            raise ValueError(f"{type(self).__name__} requires at least one accepted event.")
        self.accepted_events = normalized_events
        self.match_mode = _require_choice(match_mode, field_name="match_mode", allowed=_VALID_MATCH_MODES)
        self.default_decision = bool(default_decision)
        self.invert = bool(invert)

    def _match(self, events: list[str]) -> bool:
        if not events:
            return self.default_decision
        event_set = set(events)
        accepted = set(self.accepted_events)
        if self.match_mode == "all":
            matched = accepted.issubset(event_set)
        else:
            matched = bool(accepted & event_set)
        return not matched if self.invert else matched

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        events = _resolve_events(packet)
        matched = self._match(events)
        decisions = _broadcast_decisions(len(units), matched)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={
                    "source": self._EVENT_SOURCE,
                    "accepted_events": list(self.accepted_events),
                    "matched_events": [event for event in events if event in self.accepted_events],
                    "observed_events": list(events),
                    "match_mode": self.match_mode,
                    "default_decision": self.default_decision,
                    "invert": self.invert,
                },
            ),
            store,
        )


class BoundaryEventTrigger(_EventTrigger):
    """Trigger from externally provided structural boundary events."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "boundary_event"
    _EVENT_SOURCE = "boundary"

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        observed_events = _resolve_events(packet)
        normalized_observed = [_normalized_boundary_event_name(event) for event in observed_events]
        normalized_accepted = [_normalized_boundary_event_name(event) for event in self.accepted_events]

        if not observed_events:
            matched = self.default_decision
            matched_events: list[str] = []
        else:
            observed_set = set(normalized_observed)
            accepted_set = set(normalized_accepted)
            if self.match_mode == "all":
                matched = accepted_set.issubset(observed_set)
            else:
                matched = bool(accepted_set & observed_set)
            if self.invert:
                matched = not matched
            matched_events = [
                event
                for event, normalized in zip(observed_events, normalized_observed, strict=True)
                if normalized in accepted_set
            ]

        decisions = _broadcast_decisions(len(units), matched)
        matched_boundary_kinds = list(
            dict.fromkeys(
                normalized
                for normalized in normalized_observed
                if normalized in set(normalized_accepted) and normalized in _BOUNDARY_KIND_TO_MATCH_KEY
            )
        )

        decisions_store: dict[str, dict[str, Any]] | None = None
        boundary_kind: str | None = matched_boundary_kinds[0] if matched_boundary_kinds else None
        match_key: str | None = None
        match_value: Any = None
        missing_match_key = False
        if matched and boundary_kind is not None:
            match_key = _BOUNDARY_KIND_TO_MATCH_KEY[boundary_kind]
            match_value = _resolve_boundary_match_value(packet, match_key)
            if match_value is None:
                missing_match_key = True
            else:
                primary_event = matched_events[0] if matched_events else boundary_kind
                decisions_store = _select_store_records_by_metadata(
                    store,
                    event=primary_event,
                    boundary_kind=boundary_kind,
                    match_key=match_key,
                    match_value=match_value,
                )

        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                decisions_store=decisions_store,
                trace_payload={
                    "source": self._EVENT_SOURCE,
                    "accepted_events": list(self.accepted_events),
                    "matched_events": matched_events,
                    "observed_events": list(observed_events),
                    "match_mode": self.match_mode,
                    "default_decision": self.default_decision,
                    "invert": self.invert,
                    "boundary_kind": boundary_kind,
                    "match_key": match_key,
                    "match_value": match_value,
                    "missing_match_key": missing_match_key,
                },
            ),
            store,
        )


class RuntimeEventTrigger(_EventTrigger):
    """Trigger from externally provided runtime callback events."""

    _DEFAULT_SLOT = "evolution_trigger"
    _NAME_PREFIX = "runtime_event"
    _EVENT_SOURCE = "runtime"

    def __init__(
        self,
        *,
        slot: str | None = None,
        accepted_events: tuple[str, ...],
        match_mode: str = "any",
        default_decision: bool = False,
        invert: bool = False,
        target_layer: str | None = None,
        pressure_threshold: float | None = None,
    ) -> None:
        super().__init__(
            slot=slot,
            accepted_events=accepted_events,
            match_mode=match_mode,
            default_decision=default_decision,
            invert=invert,
        )
        self.target_layer = None if target_layer is None else str(target_layer).strip()
        if self.target_layer == "":
            raise ValueError("target_layer must be non-empty when provided.")
        self.pressure_threshold = None if pressure_threshold is None else float(pressure_threshold)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        observed_events = _resolve_events(packet)
        effective_events = list(observed_events)
        computed_runtime_events: list[str] = []
        pressure_payload: dict[str, Any] = {}
        pressure_snapshot: dict[str, Any] | None = None
        pressure_target_layer: str | None = None

        if "memory_pressure" in self.accepted_events and self.pressure_threshold is not None:
            target_layer, target_layer_mode = _resolve_pressure_layer(
                packet,
                slot=self.slot,
                explicit_target_layer=self.target_layer,
            )
            snapshot = _layer_pressure_snapshot(store, target_layer)
            pressure_snapshot = snapshot
            pressure_target_layer = target_layer
            memory_pressure = snapshot["memory_pressure"]
            if memory_pressure is not None and memory_pressure >= self.pressure_threshold:
                computed_runtime_events.append("memory_pressure")
                effective_events.append("memory_pressure")
            pressure_payload = {
                "target_layer_mode": target_layer_mode,
                "pressure_threshold": self.pressure_threshold,
                "computed_runtime_events": list(computed_runtime_events),
                **_pressure_trace_payload(snapshot),
            }
        else:
            if "memory_pressure" in self.accepted_events:
                try:
                    pressure_target_layer, target_layer_mode = _resolve_pressure_layer(
                        packet,
                        slot=self.slot,
                        explicit_target_layer=self.target_layer,
                    )
                    pressure_snapshot = _layer_pressure_snapshot(store, pressure_target_layer)
                    pressure_payload = {
                        "target_layer_mode": target_layer_mode,
                        "pressure_threshold": self.pressure_threshold,
                        "computed_runtime_events": [],
                        **_pressure_trace_payload(pressure_snapshot),
                    }
                except ValueError:
                    pressure_payload = {
                        "pressure_threshold": self.pressure_threshold,
                        "computed_runtime_events": [],
                    }
            else:
                pressure_payload = {
                    "pressure_threshold": self.pressure_threshold,
                    "computed_runtime_events": [],
                }

        matched = self._match(effective_events)
        decisions = _broadcast_decisions(len(units), matched)
        matched_events = [event for event in self.accepted_events if event in set(effective_events)]
        decisions_store: dict[str, dict[str, Any]] | None = None
        if matched and "memory_pressure" in matched_events and pressure_target_layer is not None:
            selector = {
                "kind": "layer_all",
                "event": "memory_pressure",
                "match_key": "layer",
                "match_value": pressure_target_layer,
            }
            if pressure_snapshot is not None:
                selector.update(_pressure_trace_payload(pressure_snapshot))
            decisions_store = _select_store_records_for_layer(
                store,
                layer=pressure_target_layer,
                selector=selector,
            )
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                decisions_store=decisions_store,
                trace_payload={
                    "source": self._EVENT_SOURCE,
                    "accepted_events": list(self.accepted_events),
                    "matched_events": matched_events,
                    "observed_events": list(observed_events),
                    "match_mode": self.match_mode,
                    "default_decision": self.default_decision,
                    "invert": self.invert,
                    **pressure_payload,
                },
            ),
            store,
        )


class ScalarRuleTrigger(_BaseTrigger):
    """Compare explicit scalar signals against a threshold."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "scalar_rule"

    def __init__(
        self,
        *,
        slot: str | None = None,
        signal_key: str,
        threshold: float,
        comparator: str = ">=",
        signal_source: str = "auto",
        aggregate: str = "broadcast",
        missing_value: float | None = None,
        target_layer: str | None = None,
    ) -> None:
        super().__init__(slot=slot)
        normalized_signal_key = str(signal_key).strip()
        if not normalized_signal_key:
            raise ValueError("signal_key must be non-empty.")
        self.signal_key = normalized_signal_key
        self.threshold = float(threshold)
        self.comparator = _require_choice(comparator, field_name="comparator", allowed=_VALID_COMPARATORS)
        self.signal_source = _require_choice(signal_source, field_name="signal_source", allowed=_VALID_SIGNAL_SOURCES)
        self.aggregate = _require_choice(aggregate, field_name="aggregate", allowed=_VALID_AGGREGATES)
        self.missing_value = None if missing_value is None else float(missing_value)
        self.target_layer = None if target_layer is None else str(target_layer).strip()
        if self.target_layer == "":
            raise ValueError("target_layer must be non-empty when provided.")

    def _value_or_missing(self, value: float | None) -> tuple[float | None, bool]:
        if value is not None:
            return value, False
        if self.missing_value is not None:
            return self.missing_value, True
        return None, True

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        per_unit_payload: list[dict[str, Any]] = []
        trace_payload = {
            "source": "scalar_rule",
            "signal_key": self.signal_key,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "signal_source": self.signal_source,
            "aggregate": self.aggregate,
            "missing_value": self.missing_value,
        }

        if self.signal_key == "memory_pressure":
            snapshot_cache: dict[str, dict[str, Any]] = {}
            decisions_store: dict[str, dict[str, Any]] | None = None

            def snapshot_for(unit_index: int | None = None) -> tuple[dict[str, Any], str]:
                target_layer, target_layer_mode = _resolve_pressure_layer(
                    packet,
                    slot=self.slot,
                    explicit_target_layer=self.target_layer,
                    unit_index=unit_index,
                )
                snapshot = snapshot_cache.get(target_layer)
                if snapshot is None:
                    snapshot = _layer_pressure_snapshot(store, target_layer)
                    snapshot_cache[target_layer] = snapshot
                return snapshot, target_layer_mode

            if self.aggregate == "broadcast":
                snapshot, target_layer_mode = snapshot_for()
                value, used_missing = self._value_or_missing(snapshot["memory_pressure"])
                if value is None:
                    raise ValueError(f"{self.spec.name} could not resolve signal {self.signal_key!r}.")
                decision = _compare_scalar(value, self.threshold, self.comparator)
                decisions = _broadcast_decisions(len(units), decision)
                payload = {
                    "signal_value": value,
                    "used_missing_value": used_missing,
                    **_pressure_trace_payload(snapshot),
                }
                per_unit_payload = [dict(payload) for _ in decisions]
                trace_payload.update(
                    {
                        "target_layer_mode": target_layer_mode,
                        **_pressure_trace_payload(snapshot),
                    }
                )
                if decision:
                    decisions_store = _select_store_records_for_layer(
                        store,
                        layer=snapshot["target_layer"],
                        selector={
                            "kind": "layer_all",
                            "event": "memory_pressure",
                            "match_key": "layer",
                            "match_value": snapshot["target_layer"],
                            "source": "scalar_rule",
                            **_pressure_trace_payload(snapshot),
                        },
                    )
            else:
                normalized_values: list[float] = []
                missing_flags: list[bool] = []
                target_layer_mode: str | None = None
                unique_layers: set[str] = set()
                for index, _unit in enumerate(units):
                    snapshot, current_mode = snapshot_for(index)
                    if target_layer_mode is None:
                        target_layer_mode = current_mode
                    unique_layers.add(snapshot["target_layer"])
                    value, used_missing = self._value_or_missing(snapshot["memory_pressure"])
                    if value is None:
                        raise ValueError(
                            f"{self.spec.name} aggregate={self.aggregate!r} requires resolvable memory_pressure."
                        )
                    normalized_values.append(value)
                    missing_flags.append(used_missing)
                    per_unit_payload.append(
                        {
                            "signal_value": value,
                            "used_missing_value": used_missing,
                            **_pressure_trace_payload(snapshot),
                        }
                    )

                if self.aggregate == "per_unit":
                    decisions = [_compare_scalar(value, self.threshold, self.comparator) for value in normalized_values]
                elif self.aggregate == "any_unit":
                    decisions = _broadcast_decisions(
                        len(units),
                        any(_compare_scalar(value, self.threshold, self.comparator) for value in normalized_values),
                    )
                else:
                    decisions = _broadcast_decisions(
                        len(units),
                        all(_compare_scalar(value, self.threshold, self.comparator) for value in normalized_values),
                    )

                trace_payload["target_layer_mode"] = target_layer_mode
                if len(unique_layers) == 1 and per_unit_payload:
                    trace_payload.update(_pressure_trace_payload(per_unit_payload[0]))
                selected_layers = {
                    payload["target_layer"]
                    for payload, decision in zip(per_unit_payload, decisions, strict=True)
                    if decision
                }
                if selected_layers:
                    decisions_store = {}
                    for layer in sorted(selected_layers):
                        snapshot = snapshot_cache[layer]
                        decisions_store.update(
                            _select_store_records_for_layer(
                                store,
                                layer=layer,
                                selector={
                                    "kind": "layer_all",
                                    "event": "memory_pressure",
                                    "match_key": "layer",
                                    "match_value": layer,
                                    "source": "scalar_rule",
                                    **_pressure_trace_payload(snapshot),
                                },
                            )
                        )

            return (
                _write_trace(
                    packet,
                    module_name=self.spec.name,
                    trace_key=self._trace_key(),
                    decisions=decisions,
                    decisions_store=decisions_store,
                    trace_payload=trace_payload,
                    per_unit_payload=per_unit_payload,
                ),
                store,
            )

        if self.aggregate == "broadcast":
            value, used_missing = self._value_or_missing(_global_signal_value(packet, store, self.signal_key, self.signal_source))
            if value is None:
                raise ValueError(f"{self.spec.name} could not resolve signal {self.signal_key!r}.")
            decision = _compare_scalar(value, self.threshold, self.comparator)
            decisions = _broadcast_decisions(len(units), decision)
            per_unit_payload = [{"signal_value": value, "used_missing_value": used_missing} for _ in decisions]
        else:
            raw_values = _unit_signal_values(packet, self.signal_key)
            normalized_values: list[float] = []
            missing_flags: list[bool] = []
            for raw in raw_values:
                value, used_missing = self._value_or_missing(raw)
                if value is None:
                    raise ValueError(
                        f"{self.spec.name} aggregate={self.aggregate!r} requires per-unit signal {self.signal_key!r}."
                    )
                normalized_values.append(value)
                missing_flags.append(used_missing)

            if self.aggregate == "per_unit":
                decisions = [_compare_scalar(value, self.threshold, self.comparator) for value in normalized_values]
            elif self.aggregate == "any_unit":
                decisions = _broadcast_decisions(
                    len(units),
                    any(_compare_scalar(value, self.threshold, self.comparator) for value in normalized_values),
                )
            else:
                decisions = _broadcast_decisions(
                    len(units),
                    all(_compare_scalar(value, self.threshold, self.comparator) for value in normalized_values),
                )
            per_unit_payload = [
                {"signal_value": value, "used_missing_value": missing}
                for value, missing in zip(normalized_values, missing_flags, strict=True)
            ]

        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload=trace_payload,
                per_unit_payload=per_unit_payload,
            ),
            store,
        )


class LLMJudgeTrigger(_BaseTrigger):
    """Call a real LLM with a PromptPlan-rendered judge prompt to decide trigger activation."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "llm_judge"

    def __init__(
        self,
        *,
        slot: str | None = None,
        prompt: PromptPlan | str,
        decision_mode: str = "bool",
        threshold: float = 0.5,
        per_unit: bool = True,
    ) -> None:
        super().__init__(slot=slot)
        if isinstance(prompt, str):
            normalized_prompt = prompt.strip()
        else:
            normalized_prompt = prompt
        if isinstance(normalized_prompt, str) and not normalized_prompt:
            raise ValueError("LLMJudgeTrigger requires a non-empty prompt.")
        self.prompt = normalized_prompt
        self.decision_mode = _require_choice(decision_mode, field_name="decision_mode", allowed=_VALID_DECISION_MODES)
        self.threshold = float(threshold)
        self.per_unit = bool(per_unit)

    def _judge_once(
        self,
        packet: Packet,
        store: MemoryStore,
        *,
        unit_index: int | None,
    ) -> tuple[bool, dict[str, Any], MemoryStore]:
        payload = _judge_payload(packet, store, unit_index=unit_index)
        rendered_prompt, prompt_trace, store = self._render_prompt(packet, store, unit_index=unit_index)
        output = self._llm_json(
            user=json.dumps(
                {
                    "decision_mode": self.decision_mode,
                    "threshold": self.threshold,
                    "slot": self.slot,
                    "prompt": rendered_prompt,
                    "judge_context": payload,
                },
                ensure_ascii=False,
            )
        )
        decision, details = _normalize_llm_judge_output(
            output,
            decision_mode=self.decision_mode,
            threshold=self.threshold,
        )
        return decision, {**details, **prompt_trace}, store

    def _render_prompt(
        self,
        packet: Packet,
        store: MemoryStore,
        *,
        unit_index: int | None,
    ) -> tuple[str, dict[str, Any], MemoryStore]:
        judge_payload = _judge_payload(packet, store, unit_index=unit_index)
        plan = ensure_prompt_plan(
            self.prompt,
            metadata_mode="prompt",
            context_builder=lambda current_packet, current_store: {
                "slot": self.slot,
                "decision_mode": self.decision_mode,
                "threshold": self.threshold,
                "observation": judge_payload.get("observation"),
                "unit": judge_payload.get("unit"),
                "placement": judge_payload.get("placement"),
                "store_summary": judge_payload.get("store_summary"),
                "trigger_context": judge_payload.get("trigger_context"),
            },
        )
        return render_prompt_plan(plan, packet=packet, store=store)

    def _llm_json(self, *, user: str) -> Any:
        runtime = get_runtime()
        runtime.require_llm(capability=f"LLMJudgeTrigger slot {self.slot!r}")
        return runtime.json(
            system=(
                "You decide whether a trigger should activate. "
                "Follow the user-provided prompt exactly and evaluate the supplied judge_context only. "
                "Return strict JSON only. "
                "If decision_mode is 'bool', return an object with key 'decision' (boolean). "
                "If decision_mode is 'score', return an object with key 'score' (number). "
                "If decision_mode is 'label', return an object with key 'label' (string). "
                "You may also include optional supporting fields such as reason or confidence."
            ),
            user=user,
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        per_unit_payload: list[dict[str, Any]] = []
        prompt_plan = ensure_prompt_plan(self.prompt, metadata_mode="prompt")

        if self.per_unit:
            decisions: list[bool] = []
            for index in range(len(units)):
                decision, details, store = self._judge_once(packet, store, unit_index=index)
                decisions.append(decision)
                per_unit_payload.append(details)
        else:
            decision, details, store = self._judge_once(packet, store, unit_index=None)
            decisions = _broadcast_decisions(len(units), decision)
            per_unit_payload = [dict(details) for _ in units]

        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={
                    "source": "llm_judge",
                    "decision_mode": self.decision_mode,
                    "threshold": self.threshold,
                    "judge_per_unit": self.per_unit,
                    "prompt_is_template": prompt_plan.mode == "structured"
                    or (
                        isinstance(prompt_plan.template, str)
                        and "{{" in prompt_plan.template
                        and "}}" in prompt_plan.template
                    ),
                },
                per_unit_payload=per_unit_payload,
            ),
            store,
        )


class PeriodicMaintenanceTrigger(_BaseTrigger):
    """Invoke another trigger only when periodic schedule metadata matches."""

    _DEFAULT_SLOT = "evolution_trigger"
    _NAME_PREFIX = "periodic_maintenance"

    def __init__(
        self,
        *,
        slot: str | None = None,
        every_n: int,
        counter_key: str = "ingest_count",
        trigger: TriggerModule,
    ) -> None:
        super().__init__(slot=slot)
        if every_n <= 0:
            raise ValueError("every_n must be positive.")
        normalized_counter_key = str(counter_key).strip()
        if not normalized_counter_key:
            raise ValueError("counter_key must be non-empty.")
        if not isinstance(trigger, TriggerModule):
            raise TypeError("trigger must be a TriggerModule instance.")
        wrapped_slot = getattr(getattr(trigger, "spec", None), "slot", None)
        if wrapped_slot is None:
            wrapped_slot = getattr(trigger, "slot", None)
        if wrapped_slot is not None and wrapped_slot != self.slot:
            raise ValueError(
                f"PeriodicMaintenanceTrigger slot {self.slot!r} requires wrapped trigger slot {self.slot!r}, "
                f"got {wrapped_slot!r}."
            )
        self.every_n = int(every_n)
        self.counter_key = normalized_counter_key
        self.trigger = trigger

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        self._require_units(packet)
        schedule = _resolve_schedule(packet)
        raw_tick = schedule.get("tick")
        if raw_tick is None:
            raw_tick = store.metadata.get(self.counter_key)
        tick = _coerce_numeric(raw_tick)
        matched = bool(tick is not None and tick > 0 and int(tick) % self.every_n == 0)

        periodic_payload = {
            "source": "scheduler",
            "schedule_kind": "periodic",
            "every_n": self.every_n,
            "counter_key": self.counter_key,
            "tick": None if tick is None else int(tick),
            "periodic_matched": matched,
            "wrapped_trigger_module": getattr(getattr(self.trigger, "spec", None), "name", type(self.trigger).__name__),
        }

        if not matched:
            return (
                _write_trace(
                    packet,
                    module_name=self.spec.name,
                    trace_key=self._trace_key(),
                    decisions=None,
                    trace_payload=periodic_payload,
                ),
                store,
            )

        wrapped_packet, wrapped_store = self.trigger.run(packet, store)
        trace = copy_trace(wrapped_packet)
        slot_trace = trace.get(self._trace_key(), {})
        if not isinstance(slot_trace, dict):
            slot_trace = {}
        trace[self._trace_key()] = {
            **slot_trace,
            **periodic_payload,
        }
        return (
            replace(
                wrapped_packet,
                trace=trace,
            ),
            wrapped_store,
        )


class IdleMaintenanceTrigger(_BaseTrigger):
    """Gate evolution when the runtime reports an idle maintenance window."""

    _DEFAULT_SLOT = "evolution_trigger"
    _NAME_PREFIX = "idle_maintenance"

    def __init__(
        self,
        *,
        slot: str | None = None,
        min_idle_seconds: float | None = None,
        accepted_events: tuple[str, ...] = ("idle", "sleep", "background_idle"),
        decision_mode: str = "broadcast",
    ) -> None:
        super().__init__(slot=slot)
        normalized_events = tuple(str(item).strip() for item in accepted_events if str(item).strip())
        if not normalized_events:
            raise ValueError("accepted_events must contain at least one event.")
        self.min_idle_seconds = None if min_idle_seconds is None else float(min_idle_seconds)
        self.accepted_events = normalized_events
        self.decision_mode = _require_choice(decision_mode, field_name="decision_mode", allowed=_VALID_MAINTENANCE_MODES)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        events = _resolve_events(packet)
        schedule = _resolve_schedule(packet)
        idle_seconds = _coerce_numeric(schedule.get("idle_seconds"))
        matched_event = any(event in self.accepted_events for event in events)
        matched_idle = bool(
            self.min_idle_seconds is not None and idle_seconds is not None and idle_seconds >= self.min_idle_seconds
        )
        matched = matched_event or matched_idle
        decisions = _broadcast_decisions(len(units), matched)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={
                    "source": "scheduler",
                    "schedule_kind": "idle",
                    "accepted_events": list(self.accepted_events),
                    "observed_events": list(events),
                    "idle_seconds": idle_seconds,
                    "min_idle_seconds": self.min_idle_seconds,
                    "decision_mode": self.decision_mode,
                },
            ),
            store,
        )


__all__ = [
    "AlwaysTrigger",
    "BoundaryEventTrigger",
    "IdleMaintenanceTrigger",
    "LLMJudgeTrigger",
    "NeverTrigger",
    "PeriodicMaintenanceTrigger",
    "RuntimeEventTrigger",
    "ScalarRuleTrigger",
    "StoreAllTrigger",
]
