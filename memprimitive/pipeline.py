"""Pipeline facade for the stage-1 memory DSL."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, Final

from .core import MemoryStore, Observation, Packet, Query, Readout
from .interfaces import (
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
        unit_formation: UnitFormationModule,
        representation: RepresentationModule,
        write_trigger: WriteTriggerModule,
        organization: OrganizationModule,
        memory_evolution: MemoryEvolutionModule,
        retrieval: RetrievalModule,
        readout: ReadoutModule,
        store: MemoryStore | None = None,
    ) -> None:
        self.unit_formation = unit_formation
        self.representation = representation
        self.write_trigger = write_trigger
        self.organization = organization
        self.memory_evolution = memory_evolution
        self.retrieval = retrieval
        self.readout = readout
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
    from .baselines.registry import instantiate_default_baseline_modules

    return MemoryPipeline(**instantiate_default_baseline_modules(top_k=top_k))
