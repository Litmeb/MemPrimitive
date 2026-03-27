"""Baseline evolution-trigger adapters built on the shared trigger family."""

from __future__ import annotations

from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet
from ..interfaces import EvolutionTriggerModule

from ._trigger_family import (
    AllGate,
    AlwaysOpenGate,
    ConstantSignal,
    DecisionPolicy,
    Gate,
    GraphLayerGate,
    HasEmbeddingGate,
    IdentityScorer,
    NeighborCountSignal,
    NeverPolicy,
    ScoreAggregator,
    SignalProvider,
    ThresholdPolicy,
    TopNeighborSimilaritySignal,
    TriggerFamilyRunner,
    VectorIndexReadyGate,
    WeightedSumScorer,
)


class _TriggerFamilyEvolutionAdapter(EvolutionTriggerModule):
    """Slot adapter that writes trigger-family decisions into ``Packet.evolution_decisions``."""

    trace_key = "evolution_trigger"
    output_field = "evolution_decisions"
    required_fields = ("placements",)
    spec = ModuleSpec(
        name="trigger_family_evolution_adapter",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
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


class NeverEvolutionTrigger(_TriggerFamilyEvolutionAdapter):
    """Keep extra memory evolution disabled by default after normal ingest-time write."""

    spec = ModuleSpec(
        name="never_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(self) -> None:
        super().__init__(
            runner=TriggerFamilyRunner(
                signal_providers=(ConstantSignal(signal_name="constant", value=1.0),),
                scorer=IdentityScorer(source="constant"),
                gate=AlwaysOpenGate(),
                policy=NeverPolicy(),
            )
        )


class ThresholdEvolutionTrigger(_TriggerFamilyEvolutionAdapter):
    """Constant-signal threshold baseline for the evolution trigger slot."""

    spec = ModuleSpec(
        name="threshold_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
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


def compose_graph_neighbor_evolution_trigger(
    *,
    name: str = "graph_neighbor_evolution_trigger",
    layer: str | None = None,
    candidate_top_k: int = 3,
    similarity_threshold: float | None = None,
    require_embedding: bool = True,
    require_vector_index: bool = True,
    require_graph_layer: bool = True,
    include_top_similarity_signal: bool = True,
) -> EvolutionTriggerModule:
    """Compose a graph-dependent neighbor-availability evolution trigger.

    This is an inferred decomposition of graph-triggered evolution motifs:
    candidate discovery remains a signal-provider concern, while graph/vector
    readiness stays in reusable gates. The resulting adapter writes
    ``Packet.evolution_decisions`` without changing the pipeline API.
    """

    signal_providers: list[SignalProvider] = [
        NeighborCountSignal(
            top_k=candidate_top_k,
            layer=layer,
            similarity_threshold=similarity_threshold,
            signal_name="neighbor_count",
        ),
    ]
    if include_top_similarity_signal:
        signal_providers.append(
            TopNeighborSimilaritySignal(
                top_k=candidate_top_k,
                layer=layer,
                similarity_threshold=similarity_threshold,
                signal_name="top_neighbor_similarity",
            )
        )

    gates: list[Gate] = []
    if require_embedding:
        gates.append(HasEmbeddingGate())
    if require_vector_index:
        gates.append(VectorIndexReadyGate(layer=layer))
    if require_graph_layer:
        gates.append(GraphLayerGate(layer=layer))
    gate: Gate = AllGate(tuple(gates)) if gates else AlwaysOpenGate()

    return compose_evolution_trigger(
        name=name,
        signal_providers=tuple(signal_providers),
        scorer=IdentityScorer(source="neighbor_count"),
        gate=gate,
        policy=ThresholdPolicy(threshold=1.0),
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )


class NeighborExistsEvolutionTrigger(_TriggerFamilyEvolutionAdapter):
    """Trigger graph evolution only when the current unit has graph neighbors.

    Constructor: ``target_layer`` should name a graph layer that also exposes a
    vector index. ``candidate_top_k`` must be positive. ``similarity_threshold``
    optionally filters weak neighbors before the trigger decides.

    ``run`` requires aligned ``packet.units`` and ``packet.placements``. The
    implementation is deliberately composed from shared trigger-family pieces
    rather than a bespoke black-box trigger, matching the motif guide's inferred
    decomposition of neighbor-triggered graph evolution.
    """

    spec = ModuleSpec(
        name="neighbor_exists_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
        store_requirements=("shape:Graph", "index:graph", "index:vector"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:vector"),
    )

    def __init__(
        self,
        *,
        target_layer: str = "knowledge_graph",
        candidate_top_k: int = 3,
        similarity_threshold: float | None = None,
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError("NeighborExistsEvolutionTrigger requires candidate_top_k > 0.")
        self.target_layer = target_layer
        self.candidate_top_k = candidate_top_k
        self.similarity_threshold = similarity_threshold
        composed = compose_graph_neighbor_evolution_trigger(
            name=self.spec.name,
            layer=target_layer,
            candidate_top_k=candidate_top_k,
            similarity_threshold=similarity_threshold,
            require_embedding=True,
            require_vector_index=True,
            require_graph_layer=True,
        )
        super().__init__(
            runner=composed._runner,
            spec=self.spec,
            required_fields=("placements",),
        )


def compose_evolution_trigger(
    *,
    name: str,
    signal_providers: tuple[SignalProvider, ...],
    scorer: ScoreAggregator,
    gate: Gate,
    policy: DecisionPolicy,
    input_requirements: tuple[str, ...] = ("units", "placements"),
    output_guarantees: tuple[str, ...] = ("evolution_decisions",),
) -> EvolutionTriggerModule:
    """Assemble an evolution-trigger module directly from trigger-family components."""

    required_fields = tuple(field for field in input_requirements if field != "units")
    return _TriggerFamilyEvolutionAdapter(
        runner=TriggerFamilyRunner(
            signal_providers=tuple(signal_providers),
            scorer=scorer,
            gate=gate,
            policy=policy,
        ),
        spec=ModuleSpec(
            name=name,
            slot="evolution_trigger",
            input_requirements=input_requirements,
            output_guarantees=output_guarantees,
        ),
        required_fields=required_fields,
    )


BASELINE_SLOT: Final[str] = "evolution_trigger"
BASELINE_CLASSES: Final[tuple[type[EvolutionTriggerModule], ...]] = (
    NeverEvolutionTrigger,
    ThresholdEvolutionTrigger,
    NeighborExistsEvolutionTrigger,
)
