"""Core data objects for the stage-1 memory DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import uuid4

from .contracts import (
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
    TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
    TOPOLOGY_KEYWORD_INDEX_CONTRACT,
    TOPOLOGY_TAG_INDEX_CONTRACT,
    TOPOLOGY_VECTOR_INDEX_CONTRACT,
    normalize_contracts,
)
from .utils.exceptions import IncompatibleCompositionError


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
    settings: dict[str, Any] = field(default_factory=dict)

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
        object.__setattr__(
            self,
            "settings",
            {str(key): value for key, value in self.settings.items()},
        )

    def supports_index(self, index_name: str) -> bool:
        normalized = _require_choice(index_name, "index_name", _VALID_LAYER_INDICES)
        return normalized in self.indices

    def has_setting(self, key: str) -> bool:
        normalized = _require_non_empty_text(key, "key")
        return normalized in self.settings

    def get_setting(self, key: str, default: Any = None) -> Any:
        normalized = _require_non_empty_text(key, "key")
        return self.settings.get(normalized, default)


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

    def has_setting(self, key: str) -> bool:
        normalized = _require_non_empty_text(key, "key")
        return any(layer.has_setting(normalized) for layer in self.layers)

    def with_added_layer(self, spec: StoreLayerSpec) -> "StoreTopology":
        if self.has_layer(spec.name):
            raise ValueError(f"Layer {spec.name!r} is already declared in the store topology.")
        return StoreTopology(layers=(*self.layers, spec))


@dataclass(slots=True)
class Packet:
    """Shared pipeline IR. Each stage reads and fills specific fields.

    ``decisions`` is the active gating mask used by both trigger stages.
    ``write_trigger`` first fills it before organization; later
    ``evolution_trigger`` may overwrite it before ``memory_evolution`` runs.
    Earlier write-side decisions remain available in ``trace["write_trigger"]``.
    """

    observation: Observation | None = None
    units: list[MemoryUnit] | None = None
    decisions: list[bool] | None = None
    placements: list[Placement] | None = None
    query: Query | None = None
    retrieved: RetrievedSet | None = None
    readout: Readout | None = None
    events: list[str] | None = None
    tool_call: dict[str, Any] | None = None
    target_layer_hint: str | None = None
    token_budget: int | None = None
    working_set: list[str] | None = None
    retrieval_context: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryStore:
    """Minimal in-memory layered store."""

    topology: StoreTopology = field(default_factory=StoreTopology.single_flat_default)
    layers: dict[str, list[MemoryRecord]] = field(default_factory=dict)
    allow_topology_extend: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    _next_sequence_id: int = 1
    _composition_registry: list[dict[str, Any]] = field(default_factory=list)

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

    @property
    def topology_contracts(self) -> frozenset[str]:
        contracts: list[str] = []
        if self.has_graph_layer():
            contracts.append(TOPOLOGY_GRAPH_LAYER_CONTRACT)
        if self.has_vector_layer():
            contracts.append(TOPOLOGY_VECTOR_INDEX_CONTRACT)
        if self.has_keyword_layer():
            contracts.append(TOPOLOGY_KEYWORD_INDEX_CONTRACT)
        if self.topology.has_index("tag"):
            contracts.append(TOPOLOGY_TAG_INDEX_CONTRACT)
        if any(
            layer.shape == "Graph" and "vector" in layer.indices
            for layer in self.topology.layers
        ):
            contracts.append(TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT)
        return normalize_contracts(contracts)

    @property
    def required_contracts(self) -> frozenset[str]:
        contracts: list[str] = []
        for entry in self._composition_registry:
            contracts.extend(entry.get("requires_contracts", ()))
        return normalize_contracts(contracts)

    @property
    def produced_contracts(self) -> frozenset[str]:
        contracts: list[str] = list(self.topology_contracts)
        for entry in self._composition_registry:
            contracts.extend(entry.get("produces_contracts", ()))
        return normalize_contracts(contracts)

    @property
    def registered_compositions(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "slot": entry["slot"],
                "module": entry["module"],
                "requires_contracts": tuple(entry["requires_contracts"]),
                "produces_contracts": tuple(entry["produces_contracts"]),
            }
            for entry in self._composition_registry
        )

    def register_module_contracts(
        self,
        *,
        slot: str,
        module_name: str,
        requires_contracts: Iterable[str] = (),
        produces_contracts: Iterable[str] = (),
    ) -> None:
        slot_name = _require_non_empty_text(slot, "slot")
        module = _require_non_empty_text(module_name, "module_name")
        self._composition_registry.append(
            {
                "slot": slot_name,
                "module": module,
                "requires_contracts": tuple(sorted(normalize_contracts(requires_contracts))),
                "produces_contracts": tuple(sorted(normalize_contracts(produces_contracts))),
            }
        )

    def check(self) -> frozenset[str]:
        missing = normalize_contracts(self.required_contracts - self.produced_contracts)
        self.metadata["composition_contracts"] = {
            "required": sorted(self.required_contracts),
            "produced": sorted(self.produced_contracts),
            "topology": sorted(self.topology_contracts),
            "missing": sorted(missing),
            "registered_modules": list(self.registered_compositions),
        }
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise IncompatibleCompositionError(
                "MemoryStore.check() found missing composition contracts: "
                f"{missing_text}"
            )
        return missing

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
        self._enforce_layer_capacity(record.layer)

    def _enforce_layer_capacity(self, layer: str) -> None:
        spec = self.topology.get_layer(layer)
        if spec.capacity == "unlimited":
            return

        record_budget = spec.get_setting("record_budget")
        if isinstance(record_budget, int) and record_budget > 0:
            self.trim_layer_to_record_budget(layer, record_budget)

        token_budget = spec.get_setting("token_budget")
        if isinstance(token_budget, int) and token_budget > 0:
            self.trim_layer_to_token_budget(layer, token_budget)

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

    def layer_spec(self, layer: str) -> StoreLayerSpec:
        return self.topology.get_layer(layer)

    def layer_setting(self, layer: str, key: str, default: Any = None) -> Any:
        return self.layer_spec(layer).get_setting(key, default)

    def has_layer_setting(self, key: str) -> bool:
        return self.topology.has_setting(key)

    def layer_token_count(self, layer: str) -> int:
        return sum(len(record.text.split()) for record in self.iter_records(layer))

    def trim_layer_to_record_budget(self, layer: str, record_budget: int) -> list[str]:
        if record_budget <= 0:
            raise ValueError("record_budget must be positive.")
        records = self.layers[layer]
        removed_ids: list[str] = []
        while len(records) > record_budget:
            removed = records.pop(0)
            removed_ids.append(removed.record_id)
        return removed_ids

    def trim_layer_to_token_budget(self, layer: str, token_budget: int) -> list[str]:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive.")
        records = self.layers[layer]
        removed_ids: list[str] = []
        while self.layer_token_count(layer) > token_budget and records:
            removed = records.pop(0)
            removed_ids.append(removed.record_id)
        return removed_ids

    def find_records_by_metadata(self, metadata_key: str, metadata_value: Any, *, layer: str | None = None) -> list[MemoryRecord]:
        key = _require_non_empty_text(metadata_key, "metadata_key")
        matched: list[MemoryRecord] = []
        for record in self.iter_records(layer):
            if record.metadata.get(key) == metadata_value:
                matched.append(record)
        return matched

    def find_records_by_unit_type(self, unit_type: str, *, layer: str | None = None) -> list[MemoryRecord]:
        normalized = _require_non_empty_text(unit_type, "unit_type")
        return self.find_records_by_metadata("unit_type", normalized, layer=layer)

    def find_records_by_entity(self, entity: str, *, layer: str | None = None) -> list[MemoryRecord]:
        normalized = _require_non_empty_text(entity, "entity").casefold()
        matched: list[MemoryRecord] = []
        for record in self.iter_records(layer):
            representation = record.metadata.get("representation", {})
            entities = representation.get("entities", []) if isinstance(representation, dict) else []
            if any(str(candidate).casefold() == normalized for candidate in entities):
                matched.append(record)
        return matched

    def find_records_by_key(self, key_name: str, key_value: str, *, layer: str | None = None) -> list[MemoryRecord]:
        key = _require_non_empty_text(key_name, "key_name")
        value = _require_non_empty_text(key_value, "key_value")
        matched: list[MemoryRecord] = []
        for record in self.iter_records(layer):
            if str(record.metadata.get(key, "")).strip() == value:
                matched.append(record)
        return matched

    def replace_record(self, layer: str, record_id: str, new_record: MemoryRecord) -> None:
        layer_name = _require_non_empty_text(layer, "layer")
        rid = _require_non_empty_text(record_id, "record_id")
        if new_record.layer != layer_name:
            raise ValueError("new_record.layer must match replace_record layer.")
        records = self.layers[layer_name]
        for idx, record in enumerate(records):
            if record.record_id == rid:
                records[idx] = new_record
                return
        raise KeyError(f"Record {rid!r} not found in layer {layer_name!r}.")

    def upsert_record(self, record: MemoryRecord, *, key_name: str) -> tuple[str, str]:
        key = _require_non_empty_text(key_name, "key_name")
        record_key = str(record.metadata.get(key, "")).strip()
        if not record_key:
            raise ValueError(f"Record metadata missing upsert key {key!r}.")
        matches = self.find_records_by_key(key, record_key, layer=record.layer)
        if not matches:
            self.append(record)
            return ("inserted", record.record_id)

        existing = matches[-1]
        self.replace_record(record.layer, existing.record_id, record)
        return ("updated", existing.record_id)

    def add_graph_links(self, layer: str, record_id: str, linked_record_ids: Iterable[str]) -> list[str]:
        layer_name = _require_non_empty_text(layer, "layer")
        rid = _require_non_empty_text(record_id, "record_id")
        if not self.has_layer(layer_name):
            raise ValueError(f"Layer {layer_name!r} is not declared in the store topology.")
        if self.layer_shape(layer_name) != "Graph":
            raise ValueError(f"MemoryStore.add_graph_links requires graph layer {layer_name!r}.")
        additions = [str(value).strip() for value in linked_record_ids if str(value).strip()]
        if not additions:
            return []
        records = self.layers[layer_name]
        for idx, record in enumerate(records):
            if record.record_id != rid:
                continue
            graph_meta = record.metadata.get("graph", {})
            if not isinstance(graph_meta, dict):
                graph_meta = {}
            existing_links = [str(value) for value in graph_meta.get("links", [])]
            merged_links = list(dict.fromkeys(existing_links + additions))
            updated = MemoryRecord(
                record_id=record.record_id,
                unit_id=record.unit_id,
                layer=record.layer,
                text=record.text,
                timestamp=record.timestamp,
                embedding=record.embedding,
                metadata={
                    **record.metadata,
                    "graph": {
                        **graph_meta,
                        "links": merged_links,
                        "link_count": len(merged_links),
                    },
                },
            )
            records[idx] = updated
            return merged_links
        raise KeyError(f"Record {rid!r} not found in layer {layer_name!r}.")

    def iter_graph_neighbors(self, layer: str, record_id: str) -> list[MemoryRecord]:
        layer_name = _require_non_empty_text(layer, "layer")
        rid = _require_non_empty_text(record_id, "record_id")
        if not self.has_layer(layer_name):
            raise ValueError(f"Layer {layer_name!r} is not declared in the store topology.")
        if self.layer_shape(layer_name) != "Graph":
            raise ValueError(f"MemoryStore.iter_graph_neighbors requires graph layer {layer_name!r}.")
        links: list[str] = []
        for record in self.layers[layer_name]:
            if record.record_id == rid:
                graph_meta = record.metadata.get("graph", {})
                if isinstance(graph_meta, dict):
                    links = [str(value) for value in graph_meta.get("links", [])]
                break
        if not links:
            return []
        linked = []
        for record in self.layers[layer_name]:
            if record.record_id in links:
                linked.append(record)
        return linked
