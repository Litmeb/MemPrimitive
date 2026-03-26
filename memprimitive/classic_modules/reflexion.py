"""Reflexion support for classic memory-workstream examples.

This file keeps the motif local to the forked workspace:
- failure/event-triggered reflection generation
- append-only reflection memory
- sliding-window maintenance
- reflection-prepended context readout
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

from memprimitive.baselines import AlwaysWriteTrigger, AppendOrganization, BasicRepresentation, PassThroughUnitFormation, RecencyRetrieval
from memprimitive.baselines._trace import copy_trace
from memprimitive.core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Observation, Packet, Query, Readout, StoreLayerSpec, StoreTopology
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import EvolutionTriggerModule, MemoryEvolutionModule, ReadoutModule
from memprimitive.pipeline import MemoryPipeline
from ._runtime import get_classic_runtime

DEFAULT_EPISODE_LAYER: Final[str] = "episodes"
DEFAULT_REFLECTION_LAYER: Final[str] = "reflections"


def _reflexion_controls(observation: Observation) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    raw_reflexion = observation.metadata.get("reflexion")
    if isinstance(raw_reflexion, dict):
        controls.update(raw_reflexion)

    for key in ("event", "success", "should_reflect", "task", "feedback", "failure_reason", "lesson"):
        if key in observation.metadata and key not in controls:
            controls[key] = observation.metadata[key]
    return controls


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "failure", "failed", "error", "exception"}:
            return True
        if normalized in {"false", "no", "0", "success", "passed", "ok"}:
            return False
    return None


def _heuristic_failure_reason(observation: Observation, *, failure_markers: tuple[str, ...], failure_sources: tuple[str, ...]) -> str | None:
    event = str(_reflexion_controls(observation).get("event", "")).strip().casefold()
    if event in {"failure", "failed", "error", "exception"}:
        return event

    if observation.source.casefold() in {source.casefold() for source in failure_sources}:
        return f"source:{observation.source}"

    lowered = observation.text.casefold()
    for marker in failure_markers:
        if marker.casefold() in lowered:
            return f"text:{marker}"
    return None


def _is_failure_event(
    observation: Observation,
    *,
    failure_markers: tuple[str, ...],
    failure_sources: tuple[str, ...],
) -> tuple[bool, str]:
    runtime = get_classic_runtime()
    controls = _reflexion_controls(observation)

    explicit = _coerce_bool(controls.get("should_reflect"))
    if explicit is not None:
        return explicit, "explicit should_reflect"

    success = _coerce_bool(controls.get("success"))
    if success is not None:
        return (not success), "success" if success else "failure"

    event = str(controls.get("event", "")).strip().casefold()
    if event in {"failure", "failed", "error", "exception"}:
        return True, event
    if event in {"success", "passed", "ok"}:
        return False, event

    verdict = runtime.json(
        system=(
            "You decide whether an observation should trigger Reflexion-style self-reflection. "
            "Return JSON with keys: should_reflect, reason."
        ),
        user=(
            f"text: {observation.text}\n"
            f"source: {observation.source}\n"
            f"controls: {controls}\n"
            f"failure_markers: {list(failure_markers)}\n"
            f"failure_sources: {list(failure_sources)}"
        ),
    )
    if isinstance(verdict, dict):
        should_reflect = bool(verdict.get("should_reflect", False))
        reason = str(verdict.get("reason", "")).strip() or "llm_judge"
        return should_reflect, reason

    return False, "no failure event"


def _reflection_text(observation: Observation, unit: MemoryUnit, *, reason: str) -> str:
    runtime = get_classic_runtime()
    controls = _reflexion_controls(observation)
    task = str(controls.get("task") or observation.text).strip()
    feedback = str(
        controls.get("feedback")
        or controls.get("failure_reason")
        or controls.get("lesson")
        or unit.metadata.get("failure_reason")
        or ""
    ).strip()

    return runtime.text(
        system="You write terse Reflexion memory entries after failed attempts.",
        user=(
            f"task: {task}\nfeedback: {feedback}\nreason: {reason}\n"
            "Write one short reflection beginning with 'Reflection on'."
        ),
    )


def _ensure_reflection_layer(store: MemoryStore, layer: str, *, theme: str = "semantic") -> None:
    if store.has_layer(layer):
        return
    store.ensure_layer(layer, allow_create=True, theme=theme)


def _trim_layer_to_window(store: MemoryStore, layer: str, *, window_size: int) -> list[str]:
    records = store.layers.get(layer, [])
    if len(records) <= window_size:
        return []

    removed = records[:-window_size]
    store.layers[layer] = records[-window_size:]
    return [record.record_id for record in removed]


class FailureEventEvolutionTrigger(EvolutionTriggerModule):
    """Convert failed episodes into per-unit evolution decisions.

    The trigger uses a small heuristic with explicit metadata overrides:
    - ``packet.observation.metadata["reflexion"]["should_reflect"]``
    - ``packet.observation.metadata["reflexion"]["event"]``
    - fallback failure markers in the observation text/source
    """

    spec = ModuleSpec(
        name="failure_event_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "observation.text"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(
        self,
        *,
        failure_markers: tuple[str, ...] = ("failed", "failure", "error", "exception", "wrong"),
        failure_sources: tuple[str, ...] = ("failure_log", "error_log"),
    ) -> None:
        self.failure_markers = failure_markers
        self.failure_sources = failure_sources

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("FailureEventEvolutionTrigger requires packet.observation.")
        if packet.units is None:
            raise ValueError("FailureEventEvolutionTrigger requires packet.units.")

        should_reflect, reason = _is_failure_event(
            packet.observation,
            failure_markers=self.failure_markers,
            failure_sources=self.failure_sources,
        )
        decisions = [should_reflect for _ in packet.units]
        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "policy": "failure_event",
            "triggered": should_reflect,
            "reason": reason,
            "decision_source": "packet.observation.metadata.reflexion",
            "per_unit": [
                {
                    "unit_id": unit.unit_id,
                    "decision": should_reflect,
                    "reason": reason,
                }
                for unit in packet.units
            ],
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class ReflectionMemoryEvolution(MemoryEvolutionModule):
    """Append reflection records into a dedicated reflection memory layer."""

    spec = ModuleSpec(
        name="reflection_memory_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = DEFAULT_REFLECTION_LAYER, window_size: int = 4) -> None:
        if window_size <= 0:
            raise ValueError("ReflectionMemoryEvolution requires window_size > 0.")
        self.target_layer = target_layer
        self.window_size = window_size

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

        active_unit_ids: list[str] = []
        record_ids: list[str] = []
        effects: list[dict[str, Any]] = []
        for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True):
            if not decision:
                continue

            controls = _reflexion_controls(packet.observation)
            reason = str(controls.get("event") or controls.get("feedback") or "failure").strip()
            active_unit_ids.append(unit.unit_id)
            reflection_unit = MemoryUnit(
                text=_reflection_text(packet.observation, unit, reason=reason),
                unit_type="reflection",
                metadata={
                    **unit.metadata,
                    "reflexion": {
                        "triggered": True,
                        "event": str(controls.get("event") or "failure").strip() or "failure",
                        "reason": reason,
                        "source_observation_id": packet.observation.observation_id,
                        "source_unit_id": unit.unit_id,
                        "source_layer": placement.target_layer,
                        "target_layer": self.target_layer,
                        "window_size": self.window_size,
                    },
                },
            )
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(reflection_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            record_ids.append(record.record_id)
            effects.append(
                {
                    "effect_type": "reflection_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "target_layer": self.target_layer,
                }
            )

        removed_record_ids = _trim_layer_to_window(store, self.target_layer, window_size=self.window_size)
        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "record_ids": record_ids,
            "effects": effects,
            "window_size": self.window_size,
            "target_layer": self.target_layer,
            "pruned_record_ids": removed_record_ids,
            "retained_record_ids": [record.record_id for record in store.layers[self.target_layer]],
        }
        return replace(packet, trace=trace), store


class ReflexionPrependedReadout(ReadoutModule):
    """Render reflection memory before the next task context."""

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
        top_k: int = 4,
        header: str = "Reflection memory",
        task_label: str = "Task",
    ) -> None:
        if top_k <= 0:
            raise ValueError("ReflexionPrependedReadout requires top_k > 0.")
        self.reflection_layer = reflection_layer
        self.top_k = top_k
        self.header = header
        self.task_label = task_label

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.reflection_layer):
            raise IncompatibleCompositionError(
                f"ReflexionPrependedReadout requires declared layer {self.reflection_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("ReflexionPrependedReadout requires packet.query.")

        reflections = self._select_reflections(packet, store)
        source_ids = [record.record_id for record in reflections]

        if reflections:
            lines = [self.header, *[f"- {record.text}" for record in reflections], "", f"{self.task_label}: {packet.query.text}"]
            text = "\n".join(lines)
        else:
            text = f"{self.task_label}: {packet.query.text}"

        readout = Readout(
            text=text,
            source_ids=source_ids,
            metadata={
                "reflection_count": len(reflections),
                "reflection_layer": self.reflection_layer,
                "prepend_order": source_ids,
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "prepended_reflection_count": len(reflections),
        }
        return replace(packet, readout=readout, trace=trace), store

    def _select_reflections(self, packet: Packet, store: MemoryStore) -> list[MemoryRecord]:
        if packet.retrieved is not None:
            retrieved = [record for record in packet.retrieved.items if record.layer == self.reflection_layer]
            if retrieved:
                return retrieved[: self.top_k]

        records = store.iter_records(self.reflection_layer)
        if not records:
            return []

        reranked = get_classic_runtime().rerank(
            query=packet.query.text,
            candidates=[
                {
                    "id": record.record_id,
                    "text": record.text,
                    "layer": record.layer,
                }
                for record in reversed(records)
            ],
            task="Select the most relevant Reflexion memories for the next attempt",
            top_k=self.top_k,
        )
        if not reranked:
            return list(reversed(records))[: self.top_k]
        by_id = {record.record_id: record for record in records}
        return [by_id[item["id"]] for item in reranked if item["id"] in by_id]


def build_reflexion_pipeline(
    *,
    store: MemoryStore | None = None,
    episode_layer: str = DEFAULT_EPISODE_LAYER,
    reflection_layer: str = DEFAULT_REFLECTION_LAYER,
    reflection_window: int = 4,
    reflection_top_k: int = 4,
) -> MemoryPipeline:
    """Build a small Reflexion-style pipeline around the existing stage-1 DSL."""

    if store is None:
        store = MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(name=episode_layer, theme="working", indices=("temporal", "keyword")),
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
        _ensure_reflection_layer(store, episode_layer, theme="working")
        _ensure_reflection_layer(store, reflection_layer, theme="semantic")

    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "tags", "keywords")),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(target_layer=episode_layer),
        evolution_trigger=FailureEventEvolutionTrigger(),
        memory_evolution=ReflectionMemoryEvolution(target_layer=reflection_layer, window_size=reflection_window),
        retrieval=RecencyRetrieval(top_k=reflection_top_k, layer=reflection_layer),
        readout=ReflexionPrependedReadout(reflection_layer=reflection_layer, top_k=reflection_top_k),
    )


class ReflexionWorkstream:
    """Convenience wrapper for the Reflexion example workflow."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        episode_layer: str = DEFAULT_EPISODE_LAYER,
        reflection_layer: str = DEFAULT_REFLECTION_LAYER,
        reflection_window: int = 4,
        reflection_top_k: int = 4,
    ) -> None:
        self.pipeline = build_reflexion_pipeline(
            store=store,
            episode_layer=episode_layer,
            reflection_layer=reflection_layer,
            reflection_window=reflection_window,
            reflection_top_k=reflection_top_k,
        )

    @property
    def store(self) -> MemoryStore:
        return self.pipeline.store

    def ingest(self, observation: Observation) -> Packet:
        return self.pipeline.ingest(observation)

    def recall(self, query: Query) -> Readout:
        return self.pipeline.recall(query)


__all__ = (
    "DEFAULT_EPISODE_LAYER",
    "DEFAULT_REFLECTION_LAYER",
    "FailureEventEvolutionTrigger",
    "ReflectionMemoryEvolution",
    "ReflexionPrependedReadout",
    "ReflexionWorkstream",
    "build_reflexion_pipeline",
)
