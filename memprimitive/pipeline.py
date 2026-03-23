"""Pipeline facade for the stage-1 memory DSL."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, Final

from .core import MemoryRecord, MemoryStore, Observation, Packet, Query, Readout
from .interfaces import (
    EvolutionTriggerModule,
    MemoryEvolutionModule,
    OrganizationModule,
    ReadoutModule,
    RepresentationModule,
    RetrievalModule,
    UnitFormationModule,
    WriteTriggerModule,
)
from .pipeline_slots import INGEST_SLOTS, RECALL_SLOTS

# (constructor kwarg name, required ModuleSpec.slot, expected ABC)
_INGEST_SLOT_CHECK: Final[tuple[tuple[str, str, type], ...]] = (
    ("unit_formation", "unit_formation", UnitFormationModule),
    ("representation", "representation", RepresentationModule),
    ("write_trigger", "write_trigger", WriteTriggerModule),
    ("organization", "organization", OrganizationModule),
    ("evolution_trigger", "evolution_trigger", EvolutionTriggerModule),
    ("memory_evolution", "memory_evolution", MemoryEvolutionModule),
)
_RECALL_SLOT_CHECK: Final[tuple[tuple[str, str, type], ...]] = (
    ("retrieval", "retrieval", RetrievalModule),
    ("readout", "readout", ReadoutModule),
)


class MemoryPipeline:
    """Coordinates the baseline memory pipeline using Packet-based IO.

    Module ordering and slot names are fixed (see :mod:`memprimitive.pipeline_slots`).
    Each injected module must match the expected abstract type **and**
    :attr:`~memprimitive.core.ModuleSpec.slot` for its pipeline position; this
    rejects swapped or mismatched primitives (e.g. passing a readout where
    retrieval is required) even when types happen to share the same ``run`` shape.
    """

    INGEST_SLOTS: ClassVar[tuple[str, ...]] = INGEST_SLOTS
    RECALL_SLOTS: ClassVar[tuple[str, ...]] = RECALL_SLOTS

    def __init__(
        self,
        *,
        unit_formation: UnitFormationModule | None = None,
        representation: RepresentationModule | None = None,
        write_trigger: WriteTriggerModule | None = None,
        organization: OrganizationModule | None = None,
        evolution_trigger: EvolutionTriggerModule | None = None,
        memory_evolution: MemoryEvolutionModule | None = None,
        retrieval: RetrievalModule | None = None,
        readout: ReadoutModule | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.unit_formation = unit_formation if unit_formation is not None else _default_unit_formation()
        self.representation = representation if representation is not None else _default_representation()
        self.write_trigger = write_trigger if write_trigger is not None else _default_write_trigger()
        self.organization = organization if organization is not None else _default_organization()
        self.evolution_trigger = evolution_trigger if evolution_trigger is not None else _default_evolution_trigger()
        self.memory_evolution = memory_evolution if memory_evolution is not None else _default_memory_evolution()
        self.retrieval = retrieval if retrieval is not None else _default_retrieval()
        self.readout = readout if readout is not None else _default_readout()
        self.store = store if store is not None else MemoryStore()
        self._validate_composition()

    def _validate_composition(self) -> None:
        for kwarg, expected_slot, base in _INGEST_SLOT_CHECK + _RECALL_SLOT_CHECK:
            module = getattr(self, kwarg)
            if not isinstance(module, base):
                raise TypeError(
                    f"MemoryPipeline.{kwarg} must be an instance of {base.__name__}, "
                    f"got {type(module).__name__}."
                )
            if module.spec.slot != expected_slot:
                raise ValueError(
                    f"MemoryPipeline.{kwarg} expects ModuleSpec.slot={expected_slot!r}, "
                    f"got {module.spec.slot!r} on {type(module).__name__}."
                )

    def ingest(self, observation: Observation) -> Packet:
        packet = Packet(observation=observation, trace={"ingest_started": True})
        for module in (
            self.unit_formation,
            self.representation,
            self.write_trigger,
            self.organization,
            self.evolution_trigger,
            self.memory_evolution,
        ):
            packet, self.store = module.run(packet, self.store)
        return packet

    def recall(self, query: Query) -> Readout:
        packet = Packet(query=query, trace={"recall_started": True})
        packet, self.store = self.retrieval.run(packet, self.store)
        packet, self.store = self.readout.run(packet, self.store)
        if packet.readout is None:
            raise RuntimeError("Readout module returned no readout.")
        return packet.readout

    def run_round(self, observation: Observation, query: Query) -> Readout:
        ingest_packet = self.ingest(observation)
        readout = self.recall(query)
        return replace(
            readout,
            metadata={
                **readout.metadata,
                "ingest_trace": ingest_packet.trace,
            },
        )


def create_baseline_pipeline(*, top_k: int = 3) -> MemoryPipeline:
    """Convenience factory for the fully baseline-configured pipeline."""
    return MemoryPipeline(
        unit_formation=_default_unit_formation(),
        representation=_default_representation(),
        write_trigger=_default_write_trigger(),
        organization=_default_organization(),
        evolution_trigger=_default_evolution_trigger(),
        memory_evolution=_default_memory_evolution(),
        retrieval=_default_retrieval(top_k=top_k),
        readout=_default_readout(),
    )


def _default_unit_formation() -> UnitFormationModule:
    from .baselines import PassThroughUnitFormation

    return PassThroughUnitFormation()


def _default_representation() -> RepresentationModule:
    from .baselines import BasicRepresentation

    return BasicRepresentation()


def _default_write_trigger() -> WriteTriggerModule:
    from .baselines import AlwaysWriteTrigger

    return AlwaysWriteTrigger()


def _default_organization() -> OrganizationModule:
    from .baselines import AppendOrganization

    return AppendOrganization()


def _default_evolution_trigger() -> EvolutionTriggerModule:
    from .baselines import NeverEvolutionTrigger

    return NeverEvolutionTrigger()


def _default_memory_evolution() -> MemoryEvolutionModule:
    from .baselines import AppendOnlyEvolution

    return AppendOnlyEvolution()


def _default_retrieval(*, top_k: int = 3) -> RetrievalModule:
    from .baselines import RecencyRetrieval

    return RecencyRetrieval(top_k=top_k)


def _default_readout() -> ReadoutModule:
    from .baselines import ConcatenateReadout

    return ConcatenateReadout()
