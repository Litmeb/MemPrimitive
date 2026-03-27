"""Shared trigger-family building blocks for stage-1 baseline modules.

The pipeline still exposes separate ``write_trigger`` and ``evolution_trigger``
slots. This module only provides a small reusable decision framework that both
slot adapters can compose without changing packet field boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import sqrt
from typing import Any

from ..core import MemoryStore, Packet

from ._trace import copy_trace


SignalMap = dict[str, float | bool]
_MISSING = object()


@dataclass(slots=True, frozen=True)
class TriggerContext:
    """Shared per-run context used by trigger-family components."""

    packet: Packet
    store: MemoryStore
    output_field: str
    trace_key: str


@dataclass(slots=True, frozen=True)
class UnitDecision:
    """Trace-friendly decision details for a single unit."""

    unit_id: str
    signals: SignalMap
    score: float
    gate: bool
    decision: bool


class SignalProvider(ABC):
    """Produce named per-unit signals consumed by the scorer."""

    name: str

    @abstractmethod
    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        """Return one or more signal values for ``packet.units[unit_index]``."""


class ScoreAggregator(ABC):
    """Aggregate per-unit signals into a single score."""

    name: str

    @abstractmethod
    def score(self, signals: SignalMap) -> float:
        """Return the aggregate score for a unit."""


class Gate(ABC):
    """Apply hard gating conditions after scoring."""

    name: str

    @abstractmethod
    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        """Return whether the unit passes the hard gate."""


class DecisionPolicy(ABC):
    """Turn the score and gate result into the final boolean decision."""

    name: str

    @abstractmethod
    def decide(self, *, score: float, gate_open: bool) -> bool:
        """Return the final decision for the unit."""


def _trigger_controls(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Collect trigger-relevant control fields from top-level and nested metadata."""

    controls: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return controls

    nested = payload.get("reflexion")
    if isinstance(nested, dict):
        controls.update(nested)

    for key in ("is_correct", "success", "event", "feedback", "evaluator_feedback", "trial_index"):
        if key in payload and key not in controls:
            controls[key] = payload[key]
    return controls


def _coerce_bool(value: Any) -> bool | None:
    """Parse common bool-ish outcome values used by classic feedback payloads."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "success", "passed", "correct"}:
            return True
        if normalized in {"false", "no", "0", "failure", "failed", "incorrect"}:
            return False
    return None


def _outcome_correctness_from_metadata(payload: dict[str, Any] | None) -> bool | None:
    """Return parsed correctness when present; otherwise ``None``."""

    controls = _trigger_controls(payload)
    explicit = _coerce_bool(controls.get("is_correct"))
    if explicit is not None:
        return explicit
    success = _coerce_bool(controls.get("success"))
    if success is not None:
        return success
    event = str(controls.get("event", "")).strip().casefold()
    if event in {"success", "passed", "ok"}:
        return True
    if event in {"failure", "failed", "error", "incorrect"}:
        return False
    return None


def _feedback_present_in_metadata(payload: dict[str, Any] | None) -> bool:
    """Return whether feedback text is present in supported trigger metadata."""

    controls = _trigger_controls(payload)
    value = controls.get("evaluator_feedback") or controls.get("feedback") or ""
    return bool(str(value).strip())


def _resolve_unit(context: TriggerContext, unit_index: int):
    """Return the current unit with consistent index validation."""

    units = context.packet.units
    if units is None:
        raise ValueError("packet.units is required for trigger execution.")
    if unit_index < 0 or unit_index >= len(units):
        raise IndexError(f"unit_index {unit_index} is out of range for packet.units.")
    return units[unit_index]


def _resolve_placement(context: TriggerContext, unit_index: int):
    """Return the current placement and validate unit/placement alignment."""

    placements = context.packet.placements
    if placements is None:
        raise ValueError("placements is required for trigger execution.")
    if unit_index < 0 or unit_index >= len(placements):
        raise ValueError("placements must align with packet.units for trigger execution.")
    unit = _resolve_unit(context, unit_index)
    placement = placements[unit_index]
    if placement.unit_id != unit.unit_id:
        raise ValueError("placements must align with packet.units for trigger execution.")
    return placement


def _resolve_source_object(context: TriggerContext, unit_index: int, source: str) -> Any:
    """Return the configured signal/gate source object."""

    normalized = str(source).strip()
    if normalized == "unit":
        return _resolve_unit(context, unit_index)
    if normalized == "unit.metadata":
        return _resolve_unit(context, unit_index).metadata
    if normalized == "observation":
        if context.packet.observation is None:
            raise ValueError("observation is required for trigger execution.")
        return context.packet.observation
    if normalized == "observation.metadata":
        if context.packet.observation is None:
            raise ValueError("observation is required for trigger execution.")
        return context.packet.observation.metadata
    if normalized == "query":
        if context.packet.query is None:
            raise ValueError("query is required for trigger execution.")
        return context.packet.query
    if normalized == "query.metadata":
        if context.packet.query is None:
            raise ValueError("query is required for trigger execution.")
        return context.packet.query.metadata
    if normalized == "placement":
        return _resolve_placement(context, unit_index)
    raise ValueError(f"Unsupported trigger source {source!r}.")


def _lookup_dotted_path(payload: Any, path: str) -> Any:
    """Read a dotted path from a mapping/object, returning ``_MISSING`` when absent."""

    normalized = str(path).strip()
    if not normalized:
        raise ValueError("path must be a non-empty dotted field path.")
    current = payload
    for segment in normalized.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return _MISSING
            current = current[segment]
            continue
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        return _MISSING
    return current


def _coerce_signal_float(value: Any, *, label: str) -> float:
    """Convert a bool-ish or numeric value into a trace-friendly float."""

    boolean = _coerce_bool(value)
    if boolean is not None:
        return 1.0 if boolean else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{label} must resolve to a bool-ish or numeric value.")


def _has_present_value(value: Any) -> bool:
    """Return whether a field value should count as present."""

    if value is _MISSING:
        return False
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _resolve_target_layer(context: TriggerContext, unit_index: int, *, fallback_layer: str | None = None) -> str | None:
    """Resolve the target layer for a unit from explicit config or placement.

    Explicitly configured trigger-family layers take precedence over placement
    targeting so composed triggers can intentionally inspect a different layer
    than the unit's current placement when needed.
    """

    if fallback_layer is not None:
        normalized = str(fallback_layer).strip()
        if normalized:
            return normalized
    if context.packet.placements is not None:
        return _resolve_placement(context, unit_index).target_layer
    return None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity, falling back to ``0.0`` on degenerate vectors."""

    if len(left) != len(right) or not left:
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(lhs * rhs for lhs, rhs in zip(left, right))
    return dot / (left_norm * right_norm)


def _neighbor_scores_for_unit(
    context: TriggerContext,
    unit_index: int,
    *,
    layer: str | None = None,
    candidate_top_k: int = 3,
    similarity_threshold: float | None = None,
) -> list[float]:
    """Return comparable neighbor similarities for the current unit."""

    unit = _resolve_unit(context, unit_index)
    if unit.embedding is None:
        return []
    target_layer = _resolve_target_layer(context, unit_index, fallback_layer=layer)
    if target_layer is None or not context.store.has_layer(target_layer):
        return []
    if not context.store.layer_supports_index(target_layer, "vector"):
        return []

    scores: list[float] = []
    for record in context.store.iter_records(target_layer):
        if record.unit_id == unit.unit_id or record.embedding is None:
            continue
        similarity = _cosine_similarity(unit.embedding, record.embedding)
        if similarity_threshold is not None and similarity < float(similarity_threshold):
            continue
        scores.append(float(similarity))
    scores.sort(reverse=True)
    return scores[: max(0, int(candidate_top_k))]


@dataclass(slots=True, frozen=True)
class ConstantSignal(SignalProvider):
    """Emit a constant numeric signal under a fixed name."""

    signal_name: str = "constant"
    value: float = 1.0

    @property
    def name(self) -> str:
        return f"constant:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        return {self.signal_name: float(self.value)}


@dataclass(slots=True, frozen=True)
class UnitLengthSignal(SignalProvider):
    """Emit the current unit's text length under a fixed signal name."""

    signal_name: str = "unit_length"
    normalize_by: float = 100.0

    @property
    def name(self) -> str:
        return f"unit_length:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        unit = context.packet.units[unit_index]
        value = len(unit.text.strip())
        if self.normalize_by > 0:
            return {self.signal_name: value / float(self.normalize_by)}
        return {self.signal_name: float(value)}


@dataclass(slots=True, frozen=True)
class KeywordMatchSignal(SignalProvider):
    """Emit how many query tokens overlap with the unit/representation keywords."""

    signal_name: str = "keyword_match"

    @property
    def name(self) -> str:
        return f"keyword_match:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        query = context.packet.query
        if query is None:
            raise ValueError("query is required for KeywordMatchSignal.")
        unit = context.packet.units[unit_index]
        representation = unit.metadata.get("representation", {})
        query_tokens = {token for token in query.text.casefold().split() if token}
        haystack = set(str(item).casefold() for item in representation.get("keywords", []))
        haystack.update(token.casefold() for token in unit.text.split())
        matches = len(query_tokens & haystack)
        return {self.signal_name: float(matches)}


@dataclass(slots=True, frozen=True)
class HasEntitySignal(SignalProvider):
    """Emit 1.0 when the unit has at least one entity, else 0.0."""

    signal_name: str = "has_entity"

    @property
    def name(self) -> str:
        return f"has_entity:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        unit = context.packet.units[unit_index]
        return {self.signal_name: 1.0 if unit.entities else 0.0}


@dataclass(slots=True, frozen=True)
class HasTripleSignal(SignalProvider):
    """Emit 1.0 when the unit has at least one triple, else 0.0."""

    signal_name: str = "has_triple"

    @property
    def name(self) -> str:
        return f"has_triple:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        unit = context.packet.units[unit_index]
        return {self.signal_name: 1.0 if unit.triples else 0.0}


@dataclass(slots=True, frozen=True)
class HasKVSignal(SignalProvider):
    """Emit 1.0 when the unit has key-value pairs, else 0.0."""

    signal_name: str = "has_kv"

    @property
    def name(self) -> str:
        return f"has_kv:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        unit = context.packet.units[unit_index]
        return {self.signal_name: 1.0 if unit.kv else 0.0}


@dataclass(slots=True, frozen=True)
class TagMatchSignal(SignalProvider):
    """Emit the count of overlap between query tokens and unit tags."""

    signal_name: str = "tag_match"

    @property
    def name(self) -> str:
        return f"tag_match:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        query = context.packet.query
        if query is None:
            raise ValueError("query is required for TagMatchSignal.")
        unit = context.packet.units[unit_index]
        query_tokens = {token for token in query.text.casefold().split() if token}
        unit_tags = {str(tag).casefold() for tag in unit.tags}
        return {self.signal_name: float(len(query_tokens & unit_tags))}


@dataclass(slots=True, frozen=True)
class LayerTargetSignal(SignalProvider):
    """Emit 1.0 when a unit targets one of the declared layers."""

    allowed_layers: tuple[str, ...]
    signal_name: str = "layer_target"

    @property
    def name(self) -> str:
        return f"layer_target:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        placements = context.packet.placements
        if placements is None:
            raise ValueError("placements is required for LayerTargetSignal.")
        placement = placements[unit_index]
        return {self.signal_name: 1.0 if placement.target_layer in self.allowed_layers else 0.0}


@dataclass(slots=True, frozen=True)
class QueryOverlapSignal(SignalProvider):
    """Emit token overlap between query text and the current unit text."""

    signal_name: str = "query_overlap"

    @property
    def name(self) -> str:
        return f"query_overlap:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        query = context.packet.query
        if query is None:
            raise ValueError("query is required for QueryOverlapSignal.")
        unit = context.packet.units[unit_index]
        query_tokens = {token for token in query.text.casefold().split() if token}
        unit_tokens = {token for token in unit.text.casefold().split() if token}
        return {self.signal_name: float(len(query_tokens & unit_tokens))}


@dataclass(slots=True, frozen=True)
class OutcomeCorrectnessSignal(SignalProvider):
    """Emit ``1.0`` when the trial failed according to observation metadata."""

    signal_name: str = "trial_failed"

    @property
    def name(self) -> str:
        return f"outcome_correctness:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        observation = context.packet.observation
        if observation is None:
            raise ValueError("observation is required for OutcomeCorrectnessSignal.")
        is_correct = _outcome_correctness_from_metadata(observation.metadata)
        if is_correct is None:
            return {self.signal_name: 0.0}
        return {self.signal_name: 0.0 if is_correct else 1.0}


@dataclass(slots=True, frozen=True)
class FeedbackPresenceSignal(SignalProvider):
    """Emit ``1.0`` when supported feedback text is present on the observation."""

    signal_name: str = "feedback_present"

    @property
    def name(self) -> str:
        return f"feedback_presence:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        observation = context.packet.observation
        if observation is None:
            raise ValueError("observation is required for FeedbackPresenceSignal.")
        return {self.signal_name: 1.0 if _feedback_present_in_metadata(observation.metadata) else 0.0}


@dataclass(slots=True, frozen=True)
class MetadataFlagSignal(SignalProvider):
    """Read a bool-ish metadata flag from a dotted path and emit it as ``0.0``/``1.0``.

    Constructor: ``path`` must be non-empty. ``source`` selects the metadata root
    (`unit.metadata`, `observation.metadata`, or `query.metadata`). When
    ``default`` is ``None``, missing paths raise ``ValueError``; otherwise the
    default is used.
    """

    path: str
    signal_name: str = "metadata_flag"
    source: str = "unit.metadata"
    default: Any = None

    @property
    def name(self) -> str:
        return f"metadata_flag:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        payload = _resolve_source_object(context, unit_index, self.source)
        value = _lookup_dotted_path(payload, self.path)
        if value is _MISSING:
            if self.default is None:
                raise ValueError(f"MetadataFlagSignal could not find required path {self.path!r}.")
            value = self.default
        return {self.signal_name: _coerce_signal_float(value, label=f"MetadataFlagSignal({self.path})")}


@dataclass(slots=True, frozen=True)
class UnitTypeSignal(SignalProvider):
    """Emit ``1.0`` when the unit type matches ``expected_unit_type``."""

    expected_unit_type: str
    signal_name: str = "unit_type_match"

    @property
    def name(self) -> str:
        return f"unit_type:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        unit = _resolve_unit(context, unit_index)
        return {self.signal_name: 1.0 if unit.unit_type == self.expected_unit_type else 0.0}


@dataclass(slots=True, frozen=True)
class PlacementExistsSignal(SignalProvider):
    """Emit ``1.0`` when the current unit has an aligned placement."""

    signal_name: str = "has_placement"

    @property
    def name(self) -> str:
        return f"placement_exists:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        _resolve_placement(context, unit_index)
        return {self.signal_name: 1.0}


@dataclass(slots=True, frozen=True)
class PartitionKeyPresentSignal(SignalProvider):
    """Emit ``1.0`` when at least one configured partition-key path is present.

    Constructor: ``paths`` must contain at least one dotted path. When
    ``strict_missing`` is true, missing paths raise ``ValueError`` instead of
    behaving like absent-but-false readiness checks.
    """

    paths: tuple[str, ...]
    signal_name: str = "has_partition_key"
    source: str = "unit.metadata"
    strict_missing: bool = False

    @property
    def name(self) -> str:
        return f"partition_key_present:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        if not self.paths:
            raise ValueError("PartitionKeyPresentSignal requires at least one dotted path.")
        payload = _resolve_source_object(context, unit_index, self.source)
        missing_paths: list[str] = []
        for path in self.paths:
            value = _lookup_dotted_path(payload, path)
            if value is _MISSING:
                missing_paths.append(path)
                continue
            if _has_present_value(value):
                return {self.signal_name: 1.0}
        if self.strict_missing and len(missing_paths) == len(self.paths):
            joined = ", ".join(repr(path) for path in self.paths)
            raise ValueError(f"PartitionKeyPresentSignal could not find any configured paths: {joined}.")
        return {self.signal_name: 0.0}


@dataclass(slots=True, frozen=True)
class NeighborCountSignal(SignalProvider):
    """Emit how many comparable vector neighbors are available for the unit."""

    top_k: int = 3
    signal_name: str = "neighbor_count"
    layer: str | None = None
    similarity_threshold: float | None = None

    @property
    def name(self) -> str:
        return f"neighbor_count:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        scores = _neighbor_scores_for_unit(
            context,
            unit_index,
            layer=self.layer,
            candidate_top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
        )
        return {self.signal_name: float(len(scores))}


@dataclass(slots=True, frozen=True)
class TopNeighborSimilaritySignal(SignalProvider):
    """Emit the cosine similarity of the best available vector neighbor."""

    top_k: int = 3
    signal_name: str = "top_neighbor_similarity"
    layer: str | None = None
    similarity_threshold: float | None = None

    @property
    def name(self) -> str:
        return f"top_neighbor_similarity:{self.signal_name}"

    def provide(self, context: TriggerContext, unit_index: int) -> SignalMap:
        scores = _neighbor_scores_for_unit(
            context,
            unit_index,
            layer=self.layer,
            candidate_top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
        )
        return {self.signal_name: float(scores[0]) if scores else 0.0}


@dataclass(slots=True, frozen=True)
class IdentityScorer(ScoreAggregator):
    """Read a single signal as the score."""

    source: str = "constant"

    @property
    def name(self) -> str:
        return "identity"

    def score(self, signals: SignalMap) -> float:
        if self.source not in signals:
            raise ValueError(f"IdentityScorer requires signal {self.source!r}.")
        return float(signals[self.source])


@dataclass(slots=True, frozen=True)
class WeightedSumScorer(ScoreAggregator):
    """Weighted sum over one or more named signals."""

    weights: dict[str, float]

    @property
    def name(self) -> str:
        return "weighted_sum"

    def score(self, signals: SignalMap) -> float:
        total = 0.0
        for signal_name, weight in self.weights.items():
            if signal_name not in signals:
                raise ValueError(f"WeightedSumScorer requires signal {signal_name!r}.")
            total += float(signals[signal_name]) * float(weight)
        return total


@dataclass(slots=True, frozen=True)
class MaxScorer(ScoreAggregator):
    """Take the maximum over one or more named signals."""

    sources: tuple[str, ...]

    @property
    def name(self) -> str:
        return "max"

    def score(self, signals: SignalMap) -> float:
        if not self.sources:
            raise ValueError("MaxScorer requires at least one source signal.")
        values = []
        for source in self.sources:
            if source not in signals:
                raise ValueError(f"MaxScorer requires signal {source!r}.")
            values.append(float(signals[source]))
        return max(values)


@dataclass(slots=True, frozen=True)
class MinScorer(ScoreAggregator):
    """Take the minimum over one or more named signals."""

    sources: tuple[str, ...]

    @property
    def name(self) -> str:
        return "min"

    def score(self, signals: SignalMap) -> float:
        if not self.sources:
            raise ValueError("MinScorer requires at least one source signal.")
        values = []
        for source in self.sources:
            if source not in signals:
                raise ValueError(f"MinScorer requires signal {source!r}.")
            values.append(float(signals[source]))
        return min(values)


@dataclass(slots=True, frozen=True)
class AverageScorer(ScoreAggregator):
    """Average one or more named signals."""

    sources: tuple[str, ...]

    @property
    def name(self) -> str:
        return "average"

    def score(self, signals: SignalMap) -> float:
        if not self.sources:
            raise ValueError("AverageScorer requires at least one source signal.")
        values = []
        for source in self.sources:
            if source not in signals:
                raise ValueError(f"AverageScorer requires signal {source!r}.")
            values.append(float(signals[source]))
        return sum(values) / len(values)


@dataclass(slots=True, frozen=True)
class ClippedWeightedSumScorer(ScoreAggregator):
    """Weighted sum with final clipping to a bounded range."""

    weights: dict[str, float]
    min_score: float = 0.0
    max_score: float = 1.0

    @property
    def name(self) -> str:
        return "clipped_weighted_sum"

    def score(self, signals: SignalMap) -> float:
        total = WeightedSumScorer(weights=self.weights).score(signals)
        return max(float(self.min_score), min(float(self.max_score), total))


@dataclass(slots=True, frozen=True)
class AlwaysOpenGate(Gate):
    """Stage-1 baseline gate that never blocks a unit."""

    @property
    def name(self) -> str:
        return "always_open"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        return True


@dataclass(slots=True, frozen=True)
class RequireEntityGate(Gate):
    """Allow only units that contain entities."""

    @property
    def name(self) -> str:
        return "require_entity"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        return bool(context.packet.units[unit_index].entities)


@dataclass(slots=True, frozen=True)
class RequireTripleGate(Gate):
    """Allow only units that contain triples."""

    @property
    def name(self) -> str:
        return "require_triple"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        return bool(context.packet.units[unit_index].triples)


@dataclass(slots=True, frozen=True)
class RequireTagGate(Gate):
    """Allow only units that contain at least one of the required tags."""

    required_tags: tuple[str, ...]

    @property
    def name(self) -> str:
        return "require_tag"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        unit_tags = {str(tag).casefold() for tag in context.packet.units[unit_index].tags}
        required = {tag.casefold() for tag in self.required_tags}
        return bool(unit_tags & required)


@dataclass(slots=True, frozen=True)
class LayerAllowedGate(Gate):
    """Allow only placements targeting one of the declared layers."""

    allowed_layers: tuple[str, ...]

    @property
    def name(self) -> str:
        return "layer_allowed"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        if context.packet.placements is None:
            raise ValueError("placements is required for LayerAllowedGate.")
        return context.packet.placements[unit_index].target_layer in self.allowed_layers


@dataclass(slots=True, frozen=True)
class QueryPresentGate(Gate):
    """Allow only when a query is present on the packet."""

    @property
    def name(self) -> str:
        return "query_present"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        return context.packet.query is not None


@dataclass(slots=True, frozen=True)
class SchemaPresentGate(Gate):
    """Allow only when the configured schema paths are present on the source object.

    Constructor: ``paths`` must contain at least one dotted path. ``source``
    may target `unit`, `unit.metadata`, `observation`, `observation.metadata`,
    `query`, or `query.metadata`. ``require_all`` controls whether all paths or
    any path must be present.
    """

    paths: tuple[str, ...]
    source: str = "unit.metadata"
    require_all: bool = True

    @property
    def name(self) -> str:
        return "schema_present"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        if not self.paths:
            raise ValueError("SchemaPresentGate requires at least one dotted path.")
        payload = _resolve_source_object(context, unit_index, self.source)
        checks = [_has_present_value(_lookup_dotted_path(payload, path)) for path in self.paths]
        return all(checks) if self.require_all else any(checks)


@dataclass(slots=True, frozen=True)
class FeedbackSchemaGate(Gate):
    """Allow only when the observation exposes parseable outcome or feedback fields."""

    @property
    def name(self) -> str:
        return "feedback_schema"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        observation = context.packet.observation
        if observation is None:
            return False
        return (
            _outcome_correctness_from_metadata(observation.metadata) is not None
            or _feedback_present_in_metadata(observation.metadata)
        )


@dataclass(slots=True, frozen=True)
class HasEmbeddingGate(Gate):
    """Allow only when the configured source exposes a non-empty embedding vector."""

    source: str = "unit"

    @property
    def name(self) -> str:
        return "has_embedding"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        payload = _resolve_source_object(context, unit_index, self.source)
        embedding = getattr(payload, "embedding", None)
        return isinstance(embedding, list) and len(embedding) > 0


@dataclass(slots=True, frozen=True)
class VectorIndexReadyGate(Gate):
    """Allow only when the target layer exists and exposes a vector index."""

    layer: str | None = None

    @property
    def name(self) -> str:
        return "vector_index_ready"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        target_layer = _resolve_target_layer(context, unit_index, fallback_layer=self.layer)
        if target_layer is None or not context.store.has_layer(target_layer):
            return False
        return context.store.layer_supports_index(target_layer, "vector")


@dataclass(slots=True, frozen=True)
class GraphLayerGate(Gate):
    """Allow only when the target layer exists and is graph-shaped."""

    layer: str | None = None

    @property
    def name(self) -> str:
        return "graph_layer"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        target_layer = _resolve_target_layer(context, unit_index, fallback_layer=self.layer)
        if target_layer is None or not context.store.has_layer(target_layer):
            return False
        return context.store.layer_shape(target_layer) == "Graph"


@dataclass(slots=True, frozen=True)
class AllGate(Gate):
    """Open only when every child gate opens for the current unit.

    Constructor: ``gates`` must contain at least one gate. This is a reusable
    trigger-family combinator so slot adapters can stay declarative instead of
    introducing family-specific trigger classes for multi-condition readiness
    checks.
    """

    gates: tuple[Gate, ...]

    @property
    def name(self) -> str:
        return "all"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        if not self.gates:
            raise ValueError("AllGate requires at least one child gate.")
        return all(
            gate.evaluate(context, unit_index, signals=signals, score=score)
            for gate in self.gates
        )


@dataclass(slots=True, frozen=True)
class AlwaysPolicy(DecisionPolicy):
    """Always accept units that reach the policy."""

    @property
    def name(self) -> str:
        return "always"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return True


@dataclass(slots=True, frozen=True)
class NeverPolicy(DecisionPolicy):
    """Always reject units that reach the policy."""

    @property
    def name(self) -> str:
        return "never"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return False


@dataclass(slots=True, frozen=True)
class ThresholdPolicy(DecisionPolicy):
    """Accept units whose score meets or exceeds a threshold and whose gate is open."""

    threshold: float

    @property
    def name(self) -> str:
        return "threshold"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return gate_open and score >= float(self.threshold)


@dataclass(slots=True, frozen=True)
class BooleanGatePolicy(DecisionPolicy):
    """Use the gate result directly as the final decision."""

    @property
    def name(self) -> str:
        return "boolean_gate"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return gate_open


@dataclass(slots=True, frozen=True)
class BandPassThresholdPolicy(DecisionPolicy):
    """Accept when the score falls inside an inclusive band and the gate is open."""

    lower: float
    upper: float

    @property
    def name(self) -> str:
        return "band_pass_threshold"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return gate_open and float(self.lower) <= score <= float(self.upper)


@dataclass(slots=True, frozen=True)
class ThresholdOrGatePolicy(DecisionPolicy):
    """Accept when the gate is open or the score passes a threshold."""

    threshold: float

    @property
    def name(self) -> str:
        return "threshold_or_gate"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return gate_open or score >= float(self.threshold)


class TriggerFamilyRunner:
    """Execute the shared trigger-family pipeline for one trigger slot."""

    family_name = "stage1_trigger_family"

    def __init__(
        self,
        *,
        signal_providers: tuple[SignalProvider, ...],
        scorer: ScoreAggregator,
        gate: Gate,
        policy: DecisionPolicy,
    ) -> None:
        self.signal_providers = signal_providers
        self.scorer = scorer
        self.gate = gate
        self.policy = policy

    def _require_packet_fields(self, packet: Packet, *, required_fields: tuple[str, ...]) -> None:
        for field_name in required_fields:
            if getattr(packet, field_name) is None:
                raise ValueError(f"{field_name} is required for trigger execution.")

    def run(
        self,
        packet: Packet,
        store: MemoryStore,
        *,
        trace_key: str,
        output_field: str,
        module_name: str,
        required_fields: tuple[str, ...] = (),
    ) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError(f"{module_name} requires packet.units.")
        self._require_packet_fields(packet, required_fields=required_fields)

        context = TriggerContext(packet=packet, store=store, output_field=output_field, trace_key=trace_key)
        per_unit: list[dict[str, Any]] = []
        decisions: list[bool] = []
        units = packet.units
        for unit_index, unit in enumerate(units):
            signals: SignalMap = {}
            for provider in self.signal_providers:
                provided = provider.provide(context, unit_index)
                signals.update(provided)
            score = self.scorer.score(signals)
            gate_open = self.gate.evaluate(context, unit_index, signals=signals, score=score)
            decision = self.policy.decide(score=score, gate_open=gate_open)
            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "signals": dict(signals),
                    "score": score,
                    "gate": gate_open,
                    "decision": decision,
                }
            )

        trace = copy_trace(packet)
        trace[trace_key] = {
            "module": module_name,
            "family": self.family_name,
            "policy": self.policy.name,
            "scorer": self.scorer.name,
            "gate": self.gate.name,
            "output_field": output_field,
            output_field: decisions,
            "per_unit": per_unit,
        }
        return replace(packet, trace=trace, **{output_field: decisions}), store
