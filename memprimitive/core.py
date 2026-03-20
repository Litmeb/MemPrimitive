"""Core data objects for the stage-1 memory DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _require_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _default_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


@dataclass(slots=True)
class Observation:
    """External input entering the memory pipeline."""

    text: str
    observation_id: str = field(default_factory=lambda: _default_id("obs"))
    timestamp: str = field(default_factory=_utc_now_iso)
    source: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _require_non_empty_text(self.text, "Observation.text")
        self.source = _require_non_empty_text(self.source, "Observation.source")


@dataclass(slots=True)
class MemoryUnit:
    """Intermediate memory object produced before storage."""

    text: str
    unit_id: str = field(default_factory=lambda: _default_id("unit"))
    unit_type: str = "observation"
    timestamp: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _require_non_empty_text(self.text, "MemoryUnit.text")
        self.unit_type = _require_non_empty_text(self.unit_type, "MemoryUnit.unit_type")


@dataclass(slots=True)
class MemoryRecord:
    """Stored memory record."""

    record_id: str
    unit_id: str
    layer: str
    text: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record_id = _require_non_empty_text(self.record_id, "MemoryRecord.record_id")
        self.unit_id = _require_non_empty_text(self.unit_id, "MemoryRecord.unit_id")
        self.layer = _require_non_empty_text(self.layer, "MemoryRecord.layer")
        self.text = _require_non_empty_text(self.text, "MemoryRecord.text")
        self.timestamp = _require_non_empty_text(self.timestamp, "MemoryRecord.timestamp")

    @classmethod
    def from_unit(cls, unit: MemoryUnit, layer: str, sequence_id: int) -> "MemoryRecord":
        return cls(
            record_id=f"rec-{sequence_id}",
            unit_id=unit.unit_id,
            layer=layer,
            text=unit.text,
            timestamp=unit.timestamp,
            metadata={
                **unit.metadata,
                "unit_type": unit.unit_type,
            },
        )


@dataclass(slots=True)
class Placement:
    """Storage placement plan for a unit."""

    unit_id: str
    target_layer: str

    def __post_init__(self) -> None:
        self.unit_id = _require_non_empty_text(self.unit_id, "Placement.unit_id")
        self.target_layer = _require_non_empty_text(self.target_layer, "Placement.target_layer")


@dataclass(slots=True)
class Query:
    """Query used for retrieval."""

    text: str
    query_id: str = field(default_factory=lambda: _default_id("query"))
    timestamp: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _require_non_empty_text(self.text, "Query.text")


@dataclass(slots=True)
class RetrievedSet:
    """Retrieval output before readout formatting."""

    items: list[MemoryRecord] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Readout:
    """Agent-consumable output produced after retrieval."""

    text: str
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ModuleSpec:
    """Static metadata that later DSL/search layers can inspect."""

    name: str
    slot: str
    input_requirements: tuple[str, ...] = ()
    output_guarantees: tuple[str, ...] = ()
    store_requirements: tuple[str, ...] = ()
    layer_requirements: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_text(self.name, "ModuleSpec.name")
        _require_non_empty_text(self.slot, "ModuleSpec.slot")


@dataclass(slots=True)
class Packet:
    """Shared pipeline IR. Each stage reads and fills specific fields."""

    observation: Observation | None = None
    units: list[MemoryUnit] | None = None
    decisions: list[bool] | None = None
    placements: list[Placement] | None = None
    query: Query | None = None
    retrieved: RetrievedSet | None = None
    readout: Readout | None = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryStore:
    """Minimal in-memory layered store for stage 1."""

    layers: dict[str, list[MemoryRecord]] = field(default_factory=lambda: {"default": []})
    metadata: dict[str, Any] = field(default_factory=dict)
    _next_sequence_id: int = 1

    def ensure_layer(self, layer: str) -> None:
        layer_name = _require_non_empty_text(layer, "layer")
        self.layers.setdefault(layer_name, [])

    def append(self, record: MemoryRecord) -> None:
        self.ensure_layer(record.layer)
        self.layers[record.layer].append(record)

    def next_sequence_id(self) -> int:
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        return sequence_id

    def iter_records(self, layer: str | None = None) -> list[MemoryRecord]:
        if layer is None:
            records: list[MemoryRecord] = []
            for layer_records in self.layers.values():
                records.extend(layer_records)
            return list(records)
        self.ensure_layer(layer)
        return list(self.layers[layer])

    def count(self, layer: str | None = None) -> int:
        return len(self.iter_records(layer))

    def is_empty(self) -> bool:
        return self.count() == 0
