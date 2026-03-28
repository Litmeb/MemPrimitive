"""HotPotQA-style Reflexion memory support modules.

This module keeps the classic Reflexion API stable while reusing the generic
Reflexion-like baseline motifs implemented in the stage-1 slot files.
"""

from __future__ import annotations

from dataclasses import replace

from memprimitive.utils._reflexion_family import (
    DEFAULT_MEMORY_SIZE,
    DEFAULT_REFLECTION_LAYER,
    DEFAULT_TRIAL_LAYER,
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
    STRATEGY_LAST_ATTEMPT,
    STRATEGY_LAST_ATTEMPT_AND_REFLEXION,
    STRATEGY_NONE,
    STRATEGY_REFLEXION,
    question_from_payload,
    reflexion_controls,
)
from memprimitive.baselines.evolution_trigger import OutcomeConditionedEvolutionTrigger
from memprimitive.baselines.memory_evolution import ReflectionGenerationEvolution
from memprimitive.baselines.organization import PlacementWithoutAppendOrganization
from memprimitive.baselines.readout import PromptContextReadout
from memprimitive.baselines.retrieval import BufferRetrieval
from memprimitive.core import MemoryStore, ModuleSpec, Packet
from memprimitive.utils.exceptions import IncompatibleCompositionError


class ReflexionTrialOrganization(PlacementWithoutAppendOrganization):
    """Backward-compatible Reflexion organization wrapper.

    This preserves the classic class name while reusing the generic
    placement-without-append organization motif.
    """

    spec = ModuleSpec(
        name="reflexion_trial_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
    )

    def __init__(self, target_layer: str = DEFAULT_TRIAL_LAYER) -> None:
        super().__init__(target_layer=target_layer)


class TrialFailureEvolutionTrigger(OutcomeConditionedEvolutionTrigger):
    """Backward-compatible Reflexion failure-trigger wrapper.

    The underlying implementation now reuses the shared trigger-family
    decomposition via ``compose_outcome_conditioned_evolution_trigger`` instead
    of a family-specific black box.
    """

    spec = ModuleSpec(
        name="trial_failure_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements", "observation"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(self) -> None:
        super().__init__(threshold=1.0, feedback_bonus=0.1)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        packet, store = super().run(packet, store)
        if packet.observation is None or packet.units is None:
            return packet, store

        decisions = packet.evolution_decisions or []
        trace = dict(packet.trace)
        evolution_trace = dict(trace.get("evolution_trigger", {}))
        controls = reflexion_controls(packet.observation.metadata)
        evolution_trace.update(
            {
                "policy": "trial_result",
                "triggered": any(decisions),
                "trial_is_correct": not any(decisions),
                "question": question_from_payload(packet.observation.metadata),
                "decision_source": "packet.observation.metadata.reflexion.is_correct",
                "per_unit": [
                    {
                        **entry,
                        "question": question_from_payload(packet.observation.metadata),
                        "trial_index": controls.get("trial_index", 0),
                    }
                    for entry in evolution_trace.get("per_unit", [])
                ],
            }
        )
        trace["evolution_trigger"] = evolution_trace
        return replace(packet, trace=trace), store


class ReflectionMemoryEvolution(ReflectionGenerationEvolution):
    """Backward-compatible Reflexion reflection-generation wrapper.

    The generic evolution skeleton lives in the baseline slot module. The
    classic wrapper keeps the historical class name and defaults while the
    prompt wording remains a benchmark residual handled by the shared helper.
    """

    spec = ModuleSpec(
        name="reflection_memory_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions", "observation"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        target_layer: str = DEFAULT_REFLECTION_LAYER,
        memory_size: int = DEFAULT_MEMORY_SIZE,
        window_size: int | None = None,
    ) -> None:
        super().__init__(
            target_layer=target_layer,
            memory_size=memory_size,
            window_size=window_size,
        )

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(
                f"ReflectionMemoryEvolution requires declared layer {self.target_layer!r}."
            )


class ReflexionMemoryRetrieval(BufferRetrieval):
    """Backward-compatible Reflexion buffer-retrieval wrapper."""

    spec = ModuleSpec(
        name="reflexion_memory_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, reflection_layer: str = DEFAULT_REFLECTION_LAYER, memory_size: int = DEFAULT_MEMORY_SIZE) -> None:
        super().__init__(top_k=memory_size, layer=reflection_layer, chronological=True)

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.layer):
            raise IncompatibleCompositionError(
                f"ReflexionMemoryRetrieval requires declared layer {self.layer!r}."
            )


class ReflexionContextReadout(PromptContextReadout):
    """Backward-compatible Reflexion prompt-context wrapper."""

    spec = ModuleSpec(
        name="reflexion_context_readout",
        slot="readout",
        input_requirements=("query.text",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        reflection_layer: str = DEFAULT_REFLECTION_LAYER,
        default_strategy: str = STRATEGY_REFLEXION,
        memory_size: int = DEFAULT_MEMORY_SIZE,
    ) -> None:
        super().__init__(
            memory_layer=reflection_layer,
            default_strategy=default_strategy,
            top_k=memory_size,
        )

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.memory_layer):
            raise IncompatibleCompositionError(
                f"ReflexionContextReadout requires declared layer {self.memory_layer!r}."
            )


class ReflexionPrependedReadout(ReflexionContextReadout):
    """Backward-compatible alias for the older readout class name."""

    spec = ModuleSpec(
        name="reflexion_prepended_readout",
        slot="readout",
        input_requirements=("query.text",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        reflection_layer: str = DEFAULT_REFLECTION_LAYER,
        top_k: int = DEFAULT_MEMORY_SIZE,
        default_strategy: str = STRATEGY_REFLEXION,
    ) -> None:
        super().__init__(
            reflection_layer=reflection_layer,
            default_strategy=default_strategy,
            memory_size=top_k,
        )


__all__ = (
    "DEFAULT_MEMORY_SIZE",
    "DEFAULT_REFLECTION_LAYER",
    "DEFAULT_TRIAL_LAYER",
    "LAST_TRIAL_HEADER",
    "REFLECTION_AFTER_LAST_TRIAL_HEADER",
    "REFLECTION_HEADER",
    "ReflectionMemoryEvolution",
    "ReflexionContextReadout",
    "ReflexionMemoryRetrieval",
    "ReflexionPrependedReadout",
    "ReflexionTrialOrganization",
    "STRATEGY_LAST_ATTEMPT",
    "STRATEGY_LAST_ATTEMPT_AND_REFLEXION",
    "STRATEGY_NONE",
    "STRATEGY_REFLEXION",
    "TrialFailureEvolutionTrigger",
)
