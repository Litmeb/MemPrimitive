"""Core data objects for the stage-1 memory DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
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


_VALID_LAYER_SHAPES = frozenset({"Flat", "Graph"})
_VALID_LAYER_INDICES = frozenset({"vector", "entity", "temporal", "keyword", "graph", "tag"})
_VALID_LAYER_CAPACITIES = frozenset({"token_limited", "sliding_window", "unlimited"})


def _require_choice(value: str, field_name: str, allowed: frozenset[str]) -> str:
    normalized = _require_non_empty_text(value, field_name)
    if normalized not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {options}.")
    return normalized


def _normalize_indices(indices: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_index in indices:
        index_name = _require_choice(raw_index, "StoreLayerSpec.indices", _VALID_LAYER_INDICES)
        if index_name not in seen:
            seen.add(index_name)
            normalized.append(index_name)
    return tuple(normalized)


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
    representation_elements: tuple[str, ...] = ()
    normalized_text: str | None = None
    embedding: list[float] | None = None
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    kv: dict[str, str] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _require_non_empty_text(self.text, "MemoryUnit.text")
        self.unit_type = _require_non_empty_text(self.unit_type, "MemoryUnit.unit_type")
        self.representation_elements = tuple(dict.fromkeys(self.representation_elements))
        if self.normalized_text is not None:
            self.normalized_text = _require_non_empty_text(self.normalized_text, "MemoryUnit.normalized_text")
        if self.description is not None:
            self.description = _require_non_empty_text(self.description, "MemoryUnit.description")
        self.embedding = None if self.embedding is None else [float(value) for value in self.embedding]
        self.triples = [(str(s), str(p), str(o)) for s, p, o in self.triples]
        self.kv = {str(key): str(value) for key, value in self.kv.items()}
        self.entities = [str(value) for value in self.entities]
        self.tags = [str(value) for value in self.tags]


@dataclass(slots=True)
class MemoryRecord:
    """Stored memory record."""

    record_id: str
    unit_id: str
    layer: str
    text: str
    timestamp: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record_id = _require_non_empty_text(self.record_id, "MemoryRecord.record_id")
        self.unit_id = _require_non_empty_text(self.unit_id, "MemoryRecord.unit_id")
        self.layer = _require_non_empty_text(self.layer, "MemoryRecord.layer")
        self.text = _require_non_empty_text(self.text, "MemoryRecord.text")
        self.timestamp = _require_non_empty_text(self.timestamp, "MemoryRecord.timestamp")
        self.embedding = None if self.embedding is None else [float(value) for value in self.embedding]

    @classmethod
    def from_unit(cls, unit: MemoryUnit, layer: str, sequence_id: int) -> "MemoryRecord":
        representation_summary = _representation_summary_from_unit(unit)
        embedding = None if unit.embedding is None else [float(value) for value in unit.embedding]
        return cls(
            record_id=f"rec-{sequence_id}",
            unit_id=unit.unit_id,
            layer=layer,
            text=unit.text,
            timestamp=unit.timestamp,
            embedding=embedding,
            metadata={
                **unit.metadata,
                "unit_type": unit.unit_type,
                "representation": representation_summary,
            },
        )


def _representation_summary_from_unit(unit: MemoryUnit) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "elements": list(unit.representation_elements),
        "text": unit.text,
        "normalized_text": unit.normalized_text or unit.text.casefold().strip(),
    }
    if unit.embedding is not None:
        summary["embedding"] = {
            "dim": len(unit.embedding),
        }
    if unit.triples:
        summary["triples"] = list(unit.triples)
    if unit.kv:
        summary["kv"] = dict(unit.kv)
    if unit.entities:
        summary["entities"] = list(unit.entities)
    if unit.tags:
        summary["tags"] = list(unit.tags)
    if unit.description is not None:
        summary["description"] = unit.description
    hinted_summary = unit.metadata.get("representation")
    if isinstance(hinted_summary, dict):
        for key, value in hinted_summary.items():
            if key not in summary:
                summary[key] = value
    return summary


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
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _require_non_empty_text(self.text, "Query.text")
        self.embedding = None if self.embedding is None else [float(value) for value in self.embedding]


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


@dataclass(slots=True, frozen=True)
class StoreLayerSpec:
    """Declarative spec for one logical memory layer."""

    name: str
    theme: str = "working"
    shape: str = "Flat"
    indices: tuple[str, ...] = ()
    capacity: str = "unlimited"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_text(self.name, "StoreLayerSpec.name"))
        object.__setattr__(self, "theme", _require_non_empty_text(self.theme, "StoreLayerSpec.theme"))
        object.__setattr__(self, "shape", _require_choice(self.shape, "StoreLayerSpec.shape", _VALID_LAYER_SHAPES))
        object.__setattr__(self, "indices", _normalize_indices(self.indices))
        object.__setattr__(
            self,
            "capacity",
            _require_choice(self.capacity, "StoreLayerSpec.capacity", _VALID_LAYER_CAPACITIES),
        )

    def supports_index(self, index_name: str) -> bool:
        normalized = _require_choice(index_name, "index_name", _VALID_LAYER_INDICES)
        return normalized in self.indices


@dataclass(slots=True, frozen=True)
class StoreTopology:
    """Declarative topology for the stage-1 in-memory store."""

    layers: tuple[StoreLayerSpec, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("StoreTopology.layers must contain at least one layer.")
        normalized_layers = tuple(self.layers)
        names: set[str] = set()
        for layer in normalized_layers:
            if layer.name in names:
                raise ValueError(f"StoreTopology.layers contains duplicate layer name: {layer.name!r}.")
            names.add(layer.name)
        object.__setattr__(self, "layers", normalized_layers)

    @classmethod
    def single_flat_default(cls, layer_name: str = "default", theme: str = "working") -> "StoreTopology":
        return cls(layers=(StoreLayerSpec(name=layer_name, theme=theme),))

    @classmethod
    def from_layers(cls, layers: Iterable[StoreLayerSpec]) -> "StoreTopology":
        return cls(layers=tuple(layers))

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def has_layer(self, name: str) -> bool:
        normalized = _require_non_empty_text(name, "layer")
        return any(layer.name == normalized for layer in self.layers)

    def get_layer(self, name: str) -> StoreLayerSpec:
        normalized = _require_non_empty_text(name, "layer")
        for layer in self.layers:
            if layer.name == normalized:
                return layer
        raise KeyError(f"Layer {normalized!r} is not declared in the store topology.")

    def layer_shape(self, name: str) -> str:
        return self.get_layer(name).shape

    def layer_supports_index(self, name: str, index_name: str) -> bool:
        return self.get_layer(name).supports_index(index_name)

    def has_shape(self, shape: str) -> bool:
        normalized = _require_choice(shape, "shape", _VALID_LAYER_SHAPES)
        return any(layer.shape == normalized for layer in self.layers)

    def has_index(self, index_name: str) -> bool:
        normalized = _require_choice(index_name, "index_name", _VALID_LAYER_INDICES)
        return any(normalized in layer.indices for layer in self.layers)

    def has_graph_layer(self) -> bool:
        return self.has_shape("Graph")

    def has_vector_layer(self) -> bool:
        return self.has_index("vector")

    def has_keyword_layer(self) -> bool:
        return self.has_index("keyword")

    def with_added_layer(self, spec: StoreLayerSpec) -> "StoreTopology":
        if self.has_layer(spec.name):
            raise ValueError(f"Layer {spec.name!r} is already declared in the store topology.")
        return StoreTopology(layers=(*self.layers, spec))


@dataclass(slots=True)
class Packet:
    """Shared pipeline IR. Each stage reads and fills specific fields.

    ``decisions`` is the write-side gating mask used before organization.
    ``evolution_decisions`` is the extra-evolution gating mask used by
    ``memory_evolution`` after normal ingest-time placement/write planning
    (``organization``). :class:`~memprimitive.pipeline.MemoryPipeline` may defer
    physical ``MemoryStore.append`` until after evolution stages so ingest stays
    atomic with respect to the store when later stages fail.
    """

    observation: Observation | None = None
    units: list[MemoryUnit] | None = None
    decisions: list[bool] | None = None
    evolution_decisions: list[bool] | None = None
    placements: list[Placement] | None = None
    query: Query | None = None
    retrieved: RetrievedSet | None = None
    readout: Readout | None = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryStore:
    """Minimal in-memory layered store for stage 1."""

    topology: StoreTopology = field(default_factory=StoreTopology.single_flat_default)
    layers: dict[str, list[MemoryRecord]] = field(default_factory=dict)
    allow_topology_extend: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    _next_sequence_id: int = 1

    def __post_init__(self) -> None:
        self._synchronize_layers_with_topology()

    def _synchronize_layers_with_topology(self) -> None:
        normalized_layers: dict[str, list[MemoryRecord]] = {}
        for layer_spec in self.topology.layers:
            existing_records = self.layers.get(layer_spec.name, [])
            normalized_layers[layer_spec.name] = list(existing_records)

        extra_layer_names = [name for name in self.layers if name not in normalized_layers]
        if extra_layer_names:
            if not self.allow_topology_extend:
                extras = ", ".join(repr(name) for name in extra_layer_names)
                raise ValueError(
                    "MemoryStore.layers contains undeclared topology layers: "
                    f"{extras}. Pass allow_topology_extend=True or declare them in topology."
                )
            for layer_name in extra_layer_names:
                self.topology = self.topology.with_added_layer(StoreLayerSpec(name=layer_name))
                normalized_layers[layer_name] = list(self.layers[layer_name])

        self.layers = normalized_layers

    def ensure_layer(self, layer: str, *, allow_create: bool | None = None, theme: str = "working") -> None:
        layer_name = _require_non_empty_text(layer, "layer")
        if self.topology.has_layer(layer_name):
            self.layers.setdefault(layer_name, [])
            return

        should_create = self.allow_topology_extend if allow_create is None else allow_create
        if not should_create:
            raise ValueError(
                f"Layer {layer_name!r} is not declared in the store topology. "
                "Declare it in topology or call ensure_layer(..., allow_create=True)."
            )

        self.topology = self.topology.with_added_layer(StoreLayerSpec(name=layer_name, theme=theme))
        self.layers[layer_name] = []

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
            for layer_name in self.topology.layer_names:
                records.extend(self.layers[layer_name])
            return list(records)
        layer_name = _require_non_empty_text(layer, "layer")
        if not self.topology.has_layer(layer_name):
            raise ValueError(f"Layer {layer_name!r} is not declared in the store topology.")
        return list(self.layers[layer_name])

    def count(self, layer: str | None = None) -> int:
        return len(self.iter_records(layer))

    def is_empty(self) -> bool:
        return self.count() == 0

    def has_layer(self, layer: str) -> bool:
        return self.topology.has_layer(layer)

    def layer_shape(self, layer: str) -> str:
        return self.topology.layer_shape(layer)

    def layer_supports_index(self, layer: str, index_name: str) -> bool:
        return self.topology.layer_supports_index(layer, index_name)

    def has_graph_layer(self) -> bool:
        return self.topology.has_graph_layer()

    def has_vector_layer(self) -> bool:
        return self.topology.has_vector_layer()

    def has_keyword_layer(self) -> bool:
        return self.topology.has_keyword_layer()
