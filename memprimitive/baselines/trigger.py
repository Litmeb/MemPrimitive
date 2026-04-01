"""Shared baseline trigger implementations."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import TriggerModule
from ..utils._runtime import get_runtime
from ..utils._trace import copy_trace

_VALID_TRIGGER_SLOTS = frozenset({"write_trigger", "evolution_trigger"})
_VALID_MATCH_MODES = frozenset({"any", "all"})
_VALID_COMPARATORS = frozenset({">", ">=", "<", "<=", "=="})
_VALID_SIGNAL_SOURCES = frozenset({"auto", "observation", "trace", "store", "unit_metadata"})
_VALID_AGGREGATES = frozenset({"broadcast", "per_unit", "any_unit", "all_units"})
_VALID_DECISION_MODES = frozenset({"bool", "score", "label"})
_VALID_MAINTENANCE_MODES = frozenset({"broadcast", "per_unit"})
_POSITIVE_LABELS = frozenset({"allow", "go", "ok", "positive", "true", "write", "evolve", "yes", "y", "trigger"})


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


def _write_trace(
    packet: Packet,
    *,
    module_name: str,
    trace_key: str,
    decisions: list[bool],
    trace_payload: dict[str, Any] | None = None,
    per_unit_payload: list[dict[str, Any]] | None = None,
) -> Packet:
    trace = copy_trace(packet)
    per_unit = []
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
        "decisions": list(decisions),
        "per_unit": per_unit,
    }
    if trace_payload:
        payload.update(trace_payload)
    trace[trace_key] = payload
    return replace(packet, decisions=list(decisions), trace=trace)


def _normalize_model_judge_output(output: Any, *, decision_mode: str, threshold: float) -> tuple[bool, dict[str, Any]]:
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
            raise ValueError("ModelJudgeTrigger in score mode requires a numeric score output.")
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


class ThresholdTrigger(_BaseTrigger):
    """Constant-threshold baseline for either trigger slot."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "threshold"

    def __init__(self, *, slot: str | None = None, threshold: float = 0.5, constant: float = 1.0) -> None:
        super().__init__(slot=slot)
        self.threshold = float(threshold)
        self.constant = float(constant)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        decision = self.constant >= self.threshold
        decisions = [decision] * len(units)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={"constant": self.constant, "threshold": self.threshold, "source": "constant"},
                per_unit_payload=[{"constant": self.constant} for _ in decisions],
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

        if "memory_pressure" in self.accepted_events and self.pressure_threshold is not None:
            target_layer, target_layer_mode = _resolve_pressure_layer(
                packet,
                slot=self.slot,
                explicit_target_layer=self.target_layer,
            )
            snapshot = _layer_pressure_snapshot(store, target_layer)
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
            pressure_payload = {
                "pressure_threshold": self.pressure_threshold,
                "computed_runtime_events": [],
            }

        matched = self._match(effective_events)
        decisions = _broadcast_decisions(len(units), matched)
        matched_events = [event for event in self.accepted_events if event in set(effective_events)]
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
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


class ModelJudgeTrigger(_BaseTrigger):
    """Call a real model or injected judge callable to decide trigger activation."""

    _DEFAULT_SLOT = "write_trigger"
    _NAME_PREFIX = "model_judge"

    def __init__(
        self,
        *,
        slot: str | None = None,
        system_prompt: str,
        decision_mode: str = "bool",
        threshold: float = 0.5,
        per_unit: bool = True,
        strict_json: bool = True,
        judge_callable: Callable[[dict[str, Any]], dict[str, Any] | bool | float | str] | None = None,
    ) -> None:
        super().__init__(slot=slot)
        prompt = str(system_prompt).strip()
        if not prompt:
            raise ValueError("system_prompt must be non-empty.")
        self.system_prompt = prompt
        self.decision_mode = _require_choice(decision_mode, field_name="decision_mode", allowed=_VALID_DECISION_MODES)
        self.threshold = float(threshold)
        self.per_unit = bool(per_unit)
        self.strict_json = bool(strict_json)
        self.judge_callable = judge_callable

    def _judge_once(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if self.judge_callable is not None:
            output = self.judge_callable(payload)
        else:
            runtime = get_runtime()
            user = json.dumps(payload, ensure_ascii=False)
            if self.strict_json:
                output = runtime.json(system=self.system_prompt, user=user)
            else:
                output = runtime.text(system=self.system_prompt, user=user)
        return _normalize_model_judge_output(output, decision_mode=self.decision_mode, threshold=self.threshold)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        per_unit_payload: list[dict[str, Any]] = []

        if self.per_unit:
            decisions: list[bool] = []
            for index in range(len(units)):
                decision, details = self._judge_once(_judge_payload(packet, store, unit_index=index))
                decisions.append(decision)
                per_unit_payload.append(details)
        else:
            decision, details = self._judge_once(_judge_payload(packet, store, unit_index=None))
            decisions = _broadcast_decisions(len(units), decision)
            per_unit_payload = [dict(details) for _ in units]

        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={
                    "source": "model_judge",
                    "decision_mode": self.decision_mode,
                    "threshold": self.threshold,
                    "judge_per_unit": self.per_unit,
                    "strict_json": self.strict_json,
                },
                per_unit_payload=per_unit_payload,
            ),
            store,
        )


class PeriodicMaintenanceTrigger(_BaseTrigger):
    """Gate evolution on periodic schedule metadata."""

    _DEFAULT_SLOT = "evolution_trigger"
    _NAME_PREFIX = "periodic_maintenance"

    def __init__(
        self,
        *,
        slot: str | None = None,
        every_n: int,
        counter_key: str = "ingest_count",
        decision_mode: str = "broadcast",
    ) -> None:
        super().__init__(slot=slot)
        if every_n <= 0:
            raise ValueError("every_n must be positive.")
        normalized_counter_key = str(counter_key).strip()
        if not normalized_counter_key:
            raise ValueError("counter_key must be non-empty.")
        self.every_n = int(every_n)
        self.counter_key = normalized_counter_key
        self.decision_mode = _require_choice(decision_mode, field_name="decision_mode", allowed=_VALID_MAINTENANCE_MODES)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        units = self._require_units(packet)
        schedule = _resolve_schedule(packet)
        raw_tick = schedule.get("tick")
        if raw_tick is None:
            raw_tick = store.metadata.get(self.counter_key)
        tick = _coerce_numeric(raw_tick)
        matched = bool(tick is not None and tick > 0 and int(tick) % self.every_n == 0)
        decisions = _broadcast_decisions(len(units), matched)
        return (
            _write_trace(
                packet,
                module_name=self.spec.name,
                trace_key=self._trace_key(),
                decisions=decisions,
                trace_payload={
                    "source": "scheduler",
                    "schedule_kind": "periodic",
                    "every_n": self.every_n,
                    "counter_key": self.counter_key,
                    "tick": None if tick is None else int(tick),
                    "decision_mode": self.decision_mode,
                },
            ),
            store,
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
    "ModelJudgeTrigger",
    "NeverTrigger",
    "PeriodicMaintenanceTrigger",
    "RuntimeEventTrigger",
    "ScalarRuleTrigger",
    "ThresholdTrigger",
]
