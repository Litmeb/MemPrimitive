"""Pipeline facade for the stage-1 memory DSL."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import ClassVar, Final

from .core import MemoryStore, Observation, Packet, Query, Readout
from .interfaces import (
    MemoryEvolutionModule,
    OrganizationModule,
    PrimitiveModule,
    ReadoutModule,
    RepresentationModule,
    RetrievalModule,
    TriggerModule,
    UnitFormationModule,
)
from .pipeline_slots import INGEST_SLOTS, RECALL_SLOTS

# (constructor kwarg name, required ModuleSpec.slot, expected ABC)
_INGEST_SLOT_CHECK: Final[tuple[tuple[str, str, type], ...]] = (
    ("unit_formation", "unit_formation", UnitFormationModule),
    ("representation", "representation", RepresentationModule),
    ("write_trigger", "write_trigger", TriggerModule),
    ("organization", "organization", OrganizationModule),
    ("evolution_trigger", "evolution_trigger", TriggerModule),
    ("memory_evolution", "memory_evolution", MemoryEvolutionModule),
)
_RECALL_SLOT_CHECK: Final[tuple[tuple[str, str, type], ...]] = (
    ("retrieval", "retrieval", RetrievalModule),
    ("readout", "readout", ReadoutModule),
)


def _iter_slot_modules(module_or_modules) -> tuple[PrimitiveModule, ...]:
    if isinstance(module_or_modules, PrimitiveModule):
        return (module_or_modules,)
    if isinstance(module_or_modules, Iterable) and not isinstance(module_or_modules, (str, bytes)):
        materialized = tuple(module_or_modules)
        if not materialized:
            raise ValueError("MemoryPipeline slot iterables must contain at least one module.")
        return materialized
    return (module_or_modules,)


def _materialize_slot_value(module_or_modules):
    if module_or_modules is None:
        return None
    if isinstance(module_or_modules, PrimitiveModule):
        return module_or_modules
    if isinstance(module_or_modules, Iterable) and not isinstance(module_or_modules, (str, bytes)):
        return _iter_slot_modules(module_or_modules)
    return module_or_modules


def _resolve_slot_value(module_or_modules, default_module: PrimitiveModule):
    return _materialize_slot_value(
        module_or_modules if module_or_modules is not None else default_module
    )


def _iter_nested_modules(module_or_modules) -> tuple[PrimitiveModule, ...]:
    flattened: list[PrimitiveModule] = []
    for module in _iter_slot_modules(module_or_modules):
        flattened.append(module)
        if hasattr(module, "iter_child_modules"):
            flattened.extend(_iter_nested_modules(module.iter_child_modules()))
    return tuple(flattened)


def _iter_leaf_modules(module_or_modules) -> tuple[PrimitiveModule, ...]:
    leaves: list[PrimitiveModule] = []
    for module in _iter_slot_modules(module_or_modules):
        if hasattr(module, "iter_child_modules"):
            leaves.extend(_iter_leaf_modules(module.iter_child_modules()))
            continue
        leaves.append(module)
    return tuple(leaves)


def _module_slot(module: object) -> str | None:
    return getattr(getattr(module, "spec", None), "slot", None)


def _first_retrieval_index(modules: tuple[object, ...]) -> int:
    for index, module in enumerate(modules):
        if _module_slot(module) == "retrieval":
            return index
    raise ValueError("FreeMemoryPipeline requires at least one module with spec.slot='retrieval'.")


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
        unit_formation: UnitFormationModule | Iterable[UnitFormationModule] | None = None,
        representation: RepresentationModule | Iterable[RepresentationModule] | None = None,
        write_trigger: TriggerModule | Iterable[TriggerModule] | None = None,
        organization: OrganizationModule | Iterable[OrganizationModule] | None = None,
        evolution_trigger: TriggerModule | Iterable[TriggerModule] | None = None,
        memory_evolution: MemoryEvolutionModule | Iterable[MemoryEvolutionModule] | None = None,
        retrieval: RetrievalModule | Iterable[RetrievalModule] | None = None,
        readout: ReadoutModule | Iterable[ReadoutModule] | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        from .baselines import (
            AlwaysTrigger,
            AppendOnlyEvolution,
            AppendOrganization,
            BasicRepresentation,
            ConcatenateReadout,
            NeverTrigger,
            PassThroughUnitFormation,
            RecencyRetrieval,
        )

        self.unit_formation = _resolve_slot_value(unit_formation, PassThroughUnitFormation())
        self.representation = _resolve_slot_value(representation, BasicRepresentation())
        self.write_trigger = _resolve_slot_value(write_trigger, AlwaysTrigger())
        self.organization = _resolve_slot_value(organization, AppendOrganization())
        self.evolution_trigger = _resolve_slot_value(evolution_trigger, NeverTrigger())
        self.memory_evolution = _resolve_slot_value(memory_evolution, AppendOnlyEvolution())
        self.retrieval = _resolve_slot_value(retrieval, RecencyRetrieval())
        self.readout = _resolve_slot_value(readout, ConcatenateReadout())
        self.store = store if store is not None else MemoryStore()
        self._validate_composition()
        self._register_store_contracts()

    def _validate_composition(self) -> None:
        for kwarg, expected_slot, base in _INGEST_SLOT_CHECK + _RECALL_SLOT_CHECK:
            for module in _iter_slot_modules(getattr(self, kwarg)):
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
        self._run_module_validate_store_hooks()

    def _run_module_validate_store_hooks(self) -> None:
        for slot_name, _, _ in _INGEST_SLOT_CHECK + _RECALL_SLOT_CHECK:
            for module in _iter_nested_modules(getattr(self, slot_name)):
                if hasattr(module, "validate_store"):
                    module.validate_store(self.store)

    def _register_store_contracts(self) -> None:
        for slot_name, _, _ in _INGEST_SLOT_CHECK + _RECALL_SLOT_CHECK:
            for module in _iter_leaf_modules(getattr(self, slot_name)):
                self.store.register_module_contracts(
                    slot=slot_name,
                    module_name=module.spec.name,
                    requires_contracts=module.get_requires_contracts(),
                    produces_contracts=module.get_produces_contracts(),
                )

    def ingest(self, observation: Observation) -> Packet:
        packet = Packet(observation=observation, trace={"ingest_started": True})
        for slot_value in (
            self.unit_formation,
            self.representation,
            self.write_trigger,
            self.organization,
            self.evolution_trigger,
            self.memory_evolution,
        ):
            for module in _iter_slot_modules(slot_value):
                packet, self.store = module.run(packet, self.store)
        return packet

    def recall(self, query: Query) -> Readout:
        packet = Packet(query=query, trace={"recall_started": True})
        for module in _iter_slot_modules(self.retrieval):
            packet, self.store = module.run(packet, self.store)
        for module in _iter_slot_modules(self.readout):
            packet, self.store = module.run(packet, self.store)
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


class FreeMemoryPipeline:
    """Unchecked ordered pipeline split by the first retrieval module.

    Unlike :class:`MemoryPipeline`, this runner does not validate abstract
    types, slot ordering, store contracts, or nested child modules. It simply
    runs an ordered module list as-is, treating the first module whose
    ``spec.slot`` is ``"retrieval"`` as the start of the recall half.
    """

    def __init__(
        self,
        *,
        modules: Iterable[PrimitiveModule],
        store: MemoryStore | None = None,
    ) -> None:
        materialized = tuple(modules)
        if not materialized:
            raise ValueError("FreeMemoryPipeline.modules must contain at least one module.")
        self.modules = materialized
        self._retrieve_start_index = _first_retrieval_index(self.modules)
        self.store = store if store is not None else MemoryStore()

    def ingest(self, observation: Observation) -> Packet:
        packet = Packet(observation=observation, trace={"ingest_started": True})
        for module in self.modules[: self._retrieve_start_index]:
            packet, self.store = module.run(packet, self.store)
        return packet

    def recall(self, query: Query) -> Readout:
        packet = Packet(query=query, trace={"recall_started": True})
        for module in self.modules[self._retrieve_start_index :]:
            packet, self.store = module.run(packet, self.store)
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
    from .baselines import RecencyRetrieval

    return MemoryPipeline(retrieval=RecencyRetrieval(top_k=top_k))
