"""Baseline write-trigger adapters built on the shared trigger family."""

from __future__ import annotations

from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import WriteTriggerModule

from ._trigger_family import (
    AlwaysOpenGate,
    AlwaysPolicy,
    ConstantSignal,
    DecisionPolicy,
    Gate,
    IdentityScorer,
    ScoreAggregator,
    SignalProvider,
    ThresholdPolicy,
    TriggerFamilyRunner,
    WeightedSumScorer,
)


class _TriggerFamilyWriteAdapter(WriteTriggerModule):
    """Slot adapter that writes trigger-family decisions into ``Packet.decisions``."""

    trace_key = "write_trigger"
    output_field = "decisions"
    required_fields: tuple[str, ...] = ()
    spec = ModuleSpec(
        name="trigger_family_write_adapter",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(
        self,
        *,
        runner: TriggerFamilyRunner,
        spec: ModuleSpec | None = None,
        required_fields: tuple[str, ...] | None = None,
    ) -> None:
        self._runner = runner
        if spec is not None:
            self.spec = spec
        self._required_fields = required_fields if required_fields is not None else self.required_fields

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        return self._runner.run(
            packet,
            store,
            trace_key=self.trace_key,
            output_field=self.output_field,
            module_name=self.spec.name,
            required_fields=self._required_fields,
        )


class AlwaysWriteTrigger(_TriggerFamilyWriteAdapter):
    """Mark every unit as eligible for write using the shared trigger family."""

    spec = ModuleSpec(
        name="always_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(self) -> None:
        super().__init__(
            runner=TriggerFamilyRunner(
                signal_providers=(ConstantSignal(signal_name="constant", value=1.0),),
                scorer=IdentityScorer(source="constant"),
                gate=AlwaysOpenGate(),
                policy=AlwaysPolicy(),
            )
        )


class ThresholdWriteTrigger(_TriggerFamilyWriteAdapter):
    """Constant-signal threshold baseline for the write trigger slot."""

    spec = ModuleSpec(
        name="threshold_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(self, *, threshold: float = 0.5, constant: float = 1.0) -> None:
        super().__init__(
            runner=TriggerFamilyRunner(
                signal_providers=(ConstantSignal(signal_name="constant", value=constant),),
                scorer=WeightedSumScorer(weights={"constant": 1.0}),
                gate=AlwaysOpenGate(),
                policy=ThresholdPolicy(threshold=threshold),
            )
        )


def compose_write_trigger(
    *,
    name: str,
    signal_providers: tuple[SignalProvider, ...],
    scorer: ScoreAggregator,
    gate: Gate,
    policy: DecisionPolicy,
    input_requirements: tuple[str, ...] = ("units",),
    output_guarantees: tuple[str, ...] = ("decisions",),
) -> WriteTriggerModule:
    """Assemble a write-trigger module directly from trigger-family components."""

    required_fields = tuple(field for field in input_requirements if field != "units")
    return _TriggerFamilyWriteAdapter(
        runner=TriggerFamilyRunner(
            signal_providers=tuple(signal_providers),
            scorer=scorer,
            gate=gate,
            policy=policy,
        ),
        spec=ModuleSpec(
            name=name,
            slot="write_trigger",
            input_requirements=input_requirements,
            output_guarantees=output_guarantees,
        ),
        required_fields=required_fields,
    )


BASELINE_SLOT: Final[str] = "write_trigger"
BASELINE_CLASSES: Final[tuple[type[WriteTriggerModule], ...]] = (AlwaysWriteTrigger, ThresholdWriteTrigger)
