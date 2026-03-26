"""HotPotQA-style Reflexion memory support modules.

This file intentionally contains only the memory-side primitives and the memory
pipeline builder. The external trial loop / workflow wrapper lives in the
example entrypoint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    PassThroughUnitFormation,
)
from memprimitive.baselines._trace import copy_trace
from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    ModuleSpec,
    Observation,
    Packet,
    Placement,
    Query,
    Readout,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import (
    EvolutionTriggerModule,
    MemoryEvolutionModule,
    OrganizationModule,
    ReadoutModule,
    RetrievalModule,
)
from memprimitive.pipeline import MemoryPipeline
from ._runtime import get_classic_runtime

DEFAULT_REFLECTION_LAYER: Final[str] = "reflections"
DEFAULT_TRIAL_LAYER: Final[str] = "trial_buffer"
DEFAULT_MEMORY_SIZE: Final[int] = 3

_STRATEGY_NONE: Final[str] = "base"
_STRATEGY_LAST_ATTEMPT: Final[str] = "last_trial"
_STRATEGY_REFLEXION: Final[str] = "reflexion"
_STRATEGY_LAST_ATTEMPT_AND_REFLEXION: Final[str] = "last_trial_and_reflexion"
_VALID_STRATEGIES: Final[frozenset[str]] = frozenset(
    {
        _STRATEGY_NONE,
        _STRATEGY_LAST_ATTEMPT,
        _STRATEGY_REFLEXION,
        _STRATEGY_LAST_ATTEMPT_AND_REFLEXION,
    }
)

REFLECTION_HEADER: Final[str] = (
    "You have attempted to answer following question before and failed. "
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question."
)
REFLECTION_AFTER_LAST_TRIAL_HEADER: Final[str] = (
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question."
)
LAST_TRIAL_HEADER: Final[str] = (
    "You have attempted to answer the following question before and failed. "
    "Below is the last trial you attempted to answer the question."
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _reflexion_controls(payload: dict[str, Any] | None) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return controls

    nested = payload.get("reflexion")
    if isinstance(nested, dict):
        controls.update(nested)

    for key in (
        "question",
        "task",
        "scratchpad",
        "last_attempt",
        "is_correct",
        "success",
        "feedback",
        "evaluator_feedback",
        "answer",
        "trial_index",
        "strategy",
    ):
        if key in payload and key not in controls:
            controls[key] = payload[key]
    return controls


def _coerce_bool(value: Any) -> bool | None:
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


def _format_reflections(reflections: list[str], *, header: str = REFLECTION_HEADER) -> str:
    if not reflections:
        return ""
    parts = [header]
    for index, reflection in enumerate(reflections, start=1):
        parts.append(f"Reflection {index}:")
        parts.append(_normalize_text(reflection))
    return "\n".join(parts).strip()


def _format_last_attempt(question: str, scratchpad: str) -> str:
    return "\n".join(
        [
            LAST_TRIAL_HEADER,
            f"Question: {question}",
            _normalize_text(scratchpad),
        ]
    ).strip()


def _strategy_from_query(query: Query, fallback: str) -> str:
    controls = _reflexion_controls(query.metadata)
    raw = controls.get("strategy")
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized in _VALID_STRATEGIES:
            return normalized
    return fallback


def _last_attempt_from_query(query: Query) -> str:
    controls = _reflexion_controls(query.metadata)
    return _normalize_text(controls.get("last_attempt", ""))


def _question_from_payload(payload: dict[str, Any]) -> str:
    controls = _reflexion_controls(payload)
    return _normalize_text(controls.get("question") or controls.get("task") or payload.get("text") or "")


def _scratchpad_from_payload(payload: dict[str, Any]) -> str:
    controls = _reflexion_controls(payload)
    return _normalize_text(
        controls.get("scratchpad")
        or controls.get("last_attempt")
        or payload.get("text")
        or ""
    )


def _feedback_from_payload(payload: dict[str, Any]) -> str:
    controls = _reflexion_controls(payload)
    return _normalize_text(controls.get("evaluator_feedback") or controls.get("feedback") or "")


def _is_correct_payload(payload: dict[str, Any]) -> bool:
    controls = _reflexion_controls(payload)
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
    return False


def _reflection_text(
    *,
    question: str,
    scratchpad: str,
    evaluator_feedback: str,
    prior_reflections: list[str],
) -> str:
    runtime = get_classic_runtime()
    return runtime.text(
        system=(
            "You are an advanced reasoning agent that can improve based on self reflection. "
            "Given a previous reasoning trial, diagnose the likely failure and propose a concise, "
            "high-level plan for the next attempt. Return a short reflection beginning with 'Reflection'."
        ),
        user=(
            f"question: {question}\n"
            f"previous_trial: {scratchpad}\n"
            f"evaluator_feedback: {evaluator_feedback}\n"
            f"prior_reflections: {prior_reflections}"
        ),
    ).strip()


def _ensure_reflection_layer(store: MemoryStore, layer: str, *, theme: str = "semantic") -> None:
    if store.has_layer(layer):
        return
    store.ensure_layer(layer, allow_create=True, theme=theme)


def _trim_layer_to_window(store: MemoryStore, layer: str, *, memory_size: int) -> list[str]:
    records = store.layers.get(layer, [])
    if len(records) <= memory_size:
        return []
    removed = records[:-memory_size]
    store.layers[layer] = records[-memory_size:]
    return [record.record_id for record in removed]


class ReflexionTrialOrganization(OrganizationModule):
    """Emit placements for trial packets without appending trial records to the store."""

    spec = ModuleSpec(
        name="reflexion_trial_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
    )

    def __init__(self, target_layer: str = DEFAULT_TRIAL_LAYER) -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("ReflexionTrialOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("ReflexionTrialOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("ReflexionTrialOrganization requires decisions aligned with units.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "placement_count": len(placements),
            "append_trials": False,
        }
        return replace(packet, placements=placements, trace=trace), store


class TrialFailureEvolutionTrigger(EvolutionTriggerModule):
    """Trigger reflection generation only for failed trials."""

    spec = ModuleSpec(
        name="trial_failure_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "observation.metadata"),
        output_guarantees=("evolution_decisions",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("TrialFailureEvolutionTrigger requires packet.observation.")
        if packet.units is None:
            raise ValueError("TrialFailureEvolutionTrigger requires packet.units.")

        is_correct = _is_correct_payload(packet.observation.metadata)
        should_reflect = not is_correct
        controls = _reflexion_controls(packet.observation.metadata)
        decisions = [should_reflect for _ in packet.units]
        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "policy": "trial_result",
            "triggered": should_reflect,
            "trial_is_correct": is_correct,
            "question": _question_from_payload(packet.observation.metadata),
            "decision_source": "packet.observation.metadata.reflexion.is_correct",
            "per_unit": [
                {
                    "unit_id": unit.unit_id,
                    "decision": should_reflect,
                    "question": _question_from_payload(packet.observation.metadata),
                    "trial_index": controls.get("trial_index", 0),
                }
                for unit in packet.units
            ],
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class ReflectionMemoryEvolution(MemoryEvolutionModule):
    """Append trial-level reflections into the bounded long-term memory layer."""

    spec = ModuleSpec(
        name="reflection_memory_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions", "observation.metadata"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = DEFAULT_REFLECTION_LAYER, memory_size: int = DEFAULT_MEMORY_SIZE, window_size: int | None = None) -> None:
        effective_size = memory_size if window_size is None else window_size
        if effective_size <= 0:
            raise ValueError("ReflectionMemoryEvolution requires memory_size > 0.")
        self.target_layer = target_layer
        self.memory_size = effective_size

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(
                f"ReflectionMemoryEvolution requires declared layer {self.target_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("ReflectionMemoryEvolution requires packet.observation.")
        if packet.units is None:
            raise ValueError("ReflectionMemoryEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("ReflectionMemoryEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("ReflectionMemoryEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("ReflectionMemoryEvolution requires aligned units, placements, and evolution decisions.")

        _ensure_reflection_layer(store, self.target_layer)

        question = _question_from_payload(packet.observation.metadata)
        scratchpad = _scratchpad_from_payload(packet.observation.metadata)
        evaluator_feedback = _feedback_from_payload(packet.observation.metadata)
        prior_reflections = [record.text for record in store.iter_records(self.target_layer)]

        active_unit_ids: list[str] = []
        record_ids: list[str] = []
        effects: list[dict[str, Any]] = []
        for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True):
            if not decision:
                continue
            active_unit_ids.append(unit.unit_id)
            reflection_text = _reflection_text(
                question=question,
                scratchpad=scratchpad,
                evaluator_feedback=evaluator_feedback,
                prior_reflections=prior_reflections,
            )
            reflection_unit = MemoryUnit(
                text=reflection_text,
                unit_type="reflection",
                metadata={
                    **unit.metadata,
                    "reflexion": {
                        "triggered": True,
                        "question": question,
                        "trial_index": _reflexion_controls(packet.observation.metadata).get("trial_index", 0),
                        "evaluator_feedback": evaluator_feedback,
                        "source_unit_id": unit.unit_id,
                        "source_layer": placement.target_layer,
                        "target_layer": self.target_layer,
                        "memory_size": self.memory_size,
                        "last_attempt": scratchpad,
                    },
                },
            )
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(reflection_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            prior_reflections.append(record.text)
            record_ids.append(record.record_id)
            effects.append(
                {
                    "effect_type": "reflection_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "question": question,
                    "target_layer": self.target_layer,
                }
            )

        removed_record_ids = _trim_layer_to_window(store, self.target_layer, memory_size=self.memory_size)
        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "question": question,
            "active_unit_ids": active_unit_ids,
            "record_ids": record_ids,
            "effects": effects,
            "memory_size": self.memory_size,
            "target_layer": self.target_layer,
            "pruned_record_ids": removed_record_ids,
            "retained_record_ids": [record.record_id for record in store.layers[self.target_layer]],
            "trial_trace": scratchpad,
        }
        return replace(packet, trace=trace), store


class ReflexionMemoryRetrieval(RetrievalModule):
    """Read the bounded reflection buffer directly, without query search by default."""

    spec = ModuleSpec(
        name="reflexion_memory_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, reflection_layer: str = DEFAULT_REFLECTION_LAYER, memory_size: int = DEFAULT_MEMORY_SIZE) -> None:
        if memory_size <= 0:
            raise ValueError("ReflexionMemoryRetrieval requires memory_size > 0.")
        self.reflection_layer = reflection_layer
        self.memory_size = memory_size

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.reflection_layer):
            raise IncompatibleCompositionError(
                f"ReflexionMemoryRetrieval requires declared layer {self.reflection_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("ReflexionMemoryRetrieval requires packet.query.")

        reflections = list(reversed(store.iter_records(self.reflection_layer)))[: self.memory_size]
        items = list(reversed(reflections))
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "strategy": "memory_buffer",
            }
            for rank, record in enumerate(items, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "reflection_layer": self.reflection_layer,
                "memory_size": self.memory_size,
                "candidate_count": len(store.iter_records(self.reflection_layer)),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class ReflexionContextReadout(ReadoutModule):
    """Construct the next-trial memory context according to Reflexion strategy."""

    spec = ModuleSpec(
        name="reflexion_context_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        reflection_layer: str = DEFAULT_REFLECTION_LAYER,
        default_strategy: str = _STRATEGY_REFLEXION,
        memory_size: int = DEFAULT_MEMORY_SIZE,
    ) -> None:
        if memory_size <= 0:
            raise ValueError("ReflexionContextReadout requires memory_size > 0.")
        if default_strategy not in _VALID_STRATEGIES:
            raise ValueError(f"ReflexionContextReadout requires strategy in {sorted(_VALID_STRATEGIES)}.")
        self.reflection_layer = reflection_layer
        self.default_strategy = default_strategy
        self.memory_size = memory_size

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.reflection_layer):
            raise IncompatibleCompositionError(
                f"ReflexionContextReadout requires declared layer {self.reflection_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("ReflexionContextReadout requires packet.query.")

        strategy = _strategy_from_query(packet.query, self.default_strategy)
        last_attempt = _last_attempt_from_query(packet.query)
        reflections = []
        source_ids: list[str] = []
        if packet.retrieved is not None:
            reflections = [
                record for record in packet.retrieved.items if record.layer == self.reflection_layer
            ][: self.memory_size]
            if strategy in {_STRATEGY_REFLEXION, _STRATEGY_LAST_ATTEMPT_AND_REFLEXION}:
                source_ids = [record.record_id for record in reflections]

        text = self._build_context(
            strategy=strategy,
            question=packet.query.text,
            last_attempt=last_attempt,
            reflections=[record.text for record in reflections],
        )
        readout = Readout(
            text=text,
            source_ids=source_ids,
            metadata={
                "strategy": strategy,
                "reflection_count": len(reflections),
                "last_attempt_present": bool(last_attempt),
                "reflection_layer": self.reflection_layer,
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "strategy": strategy,
            "source_ids": source_ids,
        }
        return replace(packet, readout=readout, trace=trace), store

    def _build_context(
        self,
        *,
        strategy: str,
        question: str,
        last_attempt: str,
        reflections: list[str],
    ) -> str:
        sections: list[str] = []
        if strategy == _STRATEGY_NONE:
            return ""

        if strategy in {_STRATEGY_LAST_ATTEMPT, _STRATEGY_LAST_ATTEMPT_AND_REFLEXION} and last_attempt:
            sections.append(_format_last_attempt(question, last_attempt))

        if strategy in {_STRATEGY_REFLEXION, _STRATEGY_LAST_ATTEMPT_AND_REFLEXION} and reflections:
            header = REFLECTION_AFTER_LAST_TRIAL_HEADER if sections else REFLECTION_HEADER
            sections.append(_format_reflections(reflections, header=header))

        return "\n\n".join(section for section in sections if section).strip()


class ReflexionPrependedReadout(ReflexionContextReadout):
    """Backward-compatible alias for the older readout class name."""

    spec = ModuleSpec(
        name="reflexion_prepended_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        reflection_layer: str = DEFAULT_REFLECTION_LAYER,
        top_k: int = DEFAULT_MEMORY_SIZE,
        default_strategy: str = _STRATEGY_REFLEXION,
    ) -> None:
        super().__init__(
            reflection_layer=reflection_layer,
            default_strategy=default_strategy,
            memory_size=top_k,
        )


def build_reflexion_pipeline(
    *,
    store: MemoryStore | None = None,
    reflection_layer: str = DEFAULT_REFLECTION_LAYER,
    trial_layer: str = DEFAULT_TRIAL_LAYER,
    strategy: str = _STRATEGY_REFLEXION,
    memory_size: int = DEFAULT_MEMORY_SIZE,
    reflection_window: int | None = None,
    reflection_top_k: int | None = None,
) -> MemoryPipeline:
    """Build the Reflexion memory subsystem.

    Compatibility kwargs ``reflection_window`` and ``reflection_top_k`` map to
    ``memory_size`` when provided.
    """

    effective_size = memory_size
    if reflection_window is not None:
        effective_size = reflection_window
    if reflection_top_k is not None:
        effective_size = reflection_top_k
    if effective_size <= 0:
        raise ValueError("build_reflexion_pipeline requires memory_size > 0.")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"build_reflexion_pipeline requires strategy in {sorted(_VALID_STRATEGIES)}.")

    if store is None:
        store = MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(
                        name=reflection_layer,
                        theme="semantic",
                        indices=("temporal", "keyword"),
                        capacity="sliding_window",
                    ),
                ]
            )
        )
    else:
        _ensure_reflection_layer(store, reflection_layer, theme="semantic")

    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "keywords", "tags")),
        write_trigger=AlwaysWriteTrigger(),
        organization=ReflexionTrialOrganization(target_layer=trial_layer),
        evolution_trigger=TrialFailureEvolutionTrigger(),
        memory_evolution=ReflectionMemoryEvolution(target_layer=reflection_layer, memory_size=effective_size),
        retrieval=ReflexionMemoryRetrieval(reflection_layer=reflection_layer, memory_size=effective_size),
        readout=ReflexionContextReadout(
            reflection_layer=reflection_layer,
            default_strategy=strategy,
            memory_size=effective_size,
        ),
    )


__all__ = (
    "DEFAULT_MEMORY_SIZE",
    "DEFAULT_REFLECTION_LAYER",
    "DEFAULT_TRIAL_LAYER",
    "ReflectionMemoryEvolution",
    "ReflexionContextReadout",
    "ReflexionMemoryRetrieval",
    "ReflexionPrependedReadout",
    "ReflexionTrialOrganization",
    "TrialFailureEvolutionTrigger",
    "build_reflexion_pipeline",
)
