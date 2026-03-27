"""Baseline write-trigger adapters built on the shared trigger family."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import WriteTriggerModule

from ._amem_family import DEFAULT_CATEGORY, DEFAULT_NOTE_NAMESPACE, repair_note_payload
from ._trigger_family import (
    AlwaysOpenGate,
    AlwaysPolicy,
    ConstantSignal,
    DecisionPolicy,
    Gate,
    IdentityScorer,
    MetadataFlagSignal,
    MinScorer,
    PartitionKeyPresentSignal,
    ScoreAggregator,
    SignalProvider,
    ThresholdPolicy,
    TriggerFamilyRunner,
    UnitTypeSignal,
    WeightedSumScorer,
)
from ._trace import copy_trace


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


def compose_metadata_gated_write_trigger(
    *,
    name: str = "metadata_gated_write_trigger",
    expected_unit_type: str = "tim_thought",
    metadata_path: str = "tim.write",
    metadata_source: str = "unit.metadata",
    default_write: bool = True,
) -> WriteTriggerModule:
    """Compose a TiM-style metadata-gated write trigger from shared trigger parts.

    This is an inferred decomposition of metadata-gated write motifs: upstream
    metadata decides whether a unit is writable, while the trigger itself stays
    declarative as ``unit_type signal -> metadata flag signal -> min scorer ->
    open gate -> threshold policy``.
    """

    return compose_write_trigger(
        name=name,
        signal_providers=(
            UnitTypeSignal(expected_unit_type=expected_unit_type, signal_name="unit_type_ready"),
            MetadataFlagSignal(
                path=metadata_path,
                source=metadata_source,
                default=default_write,
                signal_name="metadata_write_flag",
            ),
        ),
        scorer=MinScorer(sources=("unit_type_ready", "metadata_write_flag")),
        gate=AlwaysOpenGate(),
        policy=ThresholdPolicy(threshold=1.0),
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )


class MetadataGatedWriteTrigger(_TriggerFamilyWriteAdapter):
    """Write only units whose family identity and metadata gate are both ready.

    Constructor: ``expected_unit_type`` selects the unit family to admit,
    ``metadata_path`` points to the bool-ish write flag, and ``default_write``
    controls missing-path behavior. ``run`` requires ``packet.units`` and writes
    ``Packet.decisions`` without mutating ``store``. This class intentionally
    exposes the trigger-family decomposition instead of hiding TiM-style write
    logic behind a bespoke black box.
    """

    spec = ModuleSpec(
        name="metadata_gated_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(
        self,
        *,
        expected_unit_type: str = "tim_thought",
        metadata_path: str = "tim.write",
        metadata_source: str = "unit.metadata",
        default_write: bool = True,
    ) -> None:
        composed = compose_metadata_gated_write_trigger(
            name=self.spec.name,
            expected_unit_type=expected_unit_type,
            metadata_path=metadata_path,
            metadata_source=metadata_source,
            default_write=default_write,
        )
        super().__init__(runner=composed._runner, spec=self.spec)


def compose_key_ready_write_trigger(
    *,
    name: str = "key_ready_write_trigger",
    expected_unit_type: str | None = None,
    key_paths: tuple[str, ...] = ("memgpt_key",),
    key_source: str = "unit.metadata",
    strict_missing: bool = False,
) -> WriteTriggerModule:
    """Compose a key-presence write trigger from explicit trigger-family parts.

    The intended use is partition/key-addressable families such as MemGPT or
    other keyed-upsert flows. When ``expected_unit_type`` is provided, the unit
    must match that family *and* expose one of the configured keys before the
    final decision turns true.
    """

    signal_providers: list[SignalProvider] = [
        PartitionKeyPresentSignal(
            paths=key_paths,
            source=key_source,
            strict_missing=strict_missing,
            signal_name="key_ready",
        )
    ]
    scorer: ScoreAggregator = IdentityScorer(source="key_ready")
    if expected_unit_type is not None:
        signal_providers.insert(
            0,
            UnitTypeSignal(expected_unit_type=expected_unit_type, signal_name="unit_type_ready"),
        )
        scorer = MinScorer(sources=("unit_type_ready", "key_ready"))

    return compose_write_trigger(
        name=name,
        signal_providers=tuple(signal_providers),
        scorer=scorer,
        gate=AlwaysOpenGate(),
        policy=ThresholdPolicy(threshold=1.0),
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )


class KeyReadyWriteTrigger(_TriggerFamilyWriteAdapter):
    """Write only units that already expose the key needed by downstream storage.

    Constructor: ``key_paths`` lists one or more dotted metadata paths to check;
    ``expected_unit_type`` optionally narrows the trigger to one family, and
    ``strict_missing`` controls whether absent key paths raise. ``run`` requires
    ``packet.units`` and only writes ``Packet.decisions``. This keeps key-ready
    write motifs explicit as shared trigger-family composition rather than a
    hidden family-specific predicate.
    """

    spec = ModuleSpec(
        name="key_ready_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(
        self,
        *,
        expected_unit_type: str | None = None,
        key_paths: tuple[str, ...] = ("memgpt_key",),
        key_source: str = "unit.metadata",
        strict_missing: bool = False,
    ) -> None:
        composed = compose_key_ready_write_trigger(
            name=self.spec.name,
            expected_unit_type=expected_unit_type,
            key_paths=key_paths,
            key_source=key_source,
            strict_missing=strict_missing,
        )
        super().__init__(runner=composed._runner, spec=self.spec)


class LLMJudgedWriteTrigger(WriteTriggerModule):
    """Use an LLM to judge whether enriched note content should be written.

    Constructor: ``note_namespace`` must point to the repaired note payload to
    inspect. ``enabled=False`` turns the module into an always-write path while
    preserving explicit trace output. ``strict_llm`` is currently required to be
    ``True`` because heuristic fallback is intentionally unsupported.

    ``run`` requires ``packet.units``. It writes one boolean per unit into
    ``Packet.decisions`` and records textual reasons/confidence in trace. The
    store is unchanged.
    """

    spec = ModuleSpec(
        name="llm_judged_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(
        self,
        *,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        enabled: bool = True,
        strict_llm: bool = True,
        default_category: str = DEFAULT_CATEGORY,
    ) -> None:
        if not str(note_namespace).strip():
            raise ValueError("LLMJudgedWriteTrigger requires a non-empty note_namespace.")
        if not strict_llm:
            raise ValueError("LLMJudgedWriteTrigger requires strict_llm=True.")
        self.note_namespace = str(note_namespace).strip()
        self.enabled = enabled
        self.strict_llm = strict_llm
        self.default_category = default_category

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("LLMJudgedWriteTrigger requires packet.units.")

        decisions: list[bool] = []
        per_unit: list[dict[str, object]] = []
        for unit in packet.units:
            note_payload = repair_note_payload(
                unit.metadata.get(self.note_namespace),
                fallback_content=unit.text,
                default_category=self.default_category,
            )
            if not self.enabled:
                decision = True
                reason = "write_decision_disabled"
                confidence = 1.0
            else:
                payload = self._judge_payload(unit.text, note_payload)
                decision = str(payload.get("decision", "write")).strip().casefold() != "skip"
                reason = str(payload.get("reason", "")).strip() or "unspecified"
                raw_confidence = payload.get("confidence", 1.0)
                confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 1.0
            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "decision": "write" if decision else "skip",
                    "reason": reason,
                    "confidence": confidence,
                    "namespace": self.note_namespace,
                }
            )

        trace = copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "note_namespace": self.note_namespace,
            "per_unit": per_unit,
            "decisions": list(decisions),
        }
        return replace(packet, decisions=decisions, trace=trace), store

    def _judge_payload(self, unit_text: str, note_payload: dict[str, object]) -> dict[str, object]:
        from ..classic_modules._runtime import get_classic_runtime

        runtime = get_classic_runtime()
        runtime.require_llm(capability="LLMJudgedWriteTrigger")
        parsed = runtime.json(
            system=(
                "You are a memory write controller. Decide whether the note should be stored. "
                "Return JSON with fields: decision, reason, confidence."
            ),
            user=json.dumps(
                {
                    "content": unit_text,
                    "note_text": note_payload["note_text"],
                    "context": note_payload["context"],
                    "keywords": note_payload["keywords"],
                    "tags": note_payload["tags"],
                    "category": note_payload["category"],
                },
                ensure_ascii=False,
            ),
        )
        if not isinstance(parsed, dict):
            raise ValueError("LLMJudgedWriteTrigger must receive a JSON object response.")
        return parsed


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
BASELINE_CLASSES: Final[tuple[type[WriteTriggerModule], ...]] = (
    AlwaysWriteTrigger,
    ThresholdWriteTrigger,
    MetadataGatedWriteTrigger,
    KeyReadyWriteTrigger,
    LLMJudgedWriteTrigger,
)
