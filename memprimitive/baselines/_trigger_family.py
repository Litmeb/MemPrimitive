"""Shared trigger-family building blocks for stage-1 baseline modules.

The pipeline still exposes separate ``write_trigger`` and ``evolution_trigger``
slots. This module only provides a small reusable decision framework that both
slot adapters can compose without changing packet field boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from ..core import MemoryStore, Packet

from ._trace import copy_trace


SignalMap = dict[str, float | bool]


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
class AlwaysOpenGate(Gate):
    """Stage-1 baseline gate that never blocks a unit."""

    @property
    def name(self) -> str:
        return "always_open"

    def evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool:
        return True


@dataclass(slots=True, frozen=True)
class AlwaysPolicy(DecisionPolicy):
    """Always accept units that reach the policy."""

    @property
    def name(self) -> str:
        return "always"

    def decide(self, *, score: float, gate_open: bool) -> bool:
        return True


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
