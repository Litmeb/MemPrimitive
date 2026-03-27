"""Pipeline facade for the stage-1 memory DSL."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import ClassVar, Final

from .core import MemoryRecord, MemoryStore, Observation, Packet, Query, Readout
from .exceptions import IncompatibleCompositionError
from .interfaces import (
    EvolutionTriggerModule,
    MemoryEvolutionModule,
    OrganizationModule,
    PrimitiveModule,
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


def _iter_nested_modules(module_or_modules) -> tuple[PrimitiveModule, ...]:
    flattened: list[PrimitiveModule] = []
    for module in _iter_slot_modules(module_or_modules):
        flattened.append(module)
        if hasattr(module, "iter_child_modules"):
            flattened.extend(_iter_nested_modules(module.iter_child_modules()))
    return tuple(flattened)


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
        write_trigger: WriteTriggerModule | Iterable[WriteTriggerModule] | None = None,
        organization: OrganizationModule | Iterable[OrganizationModule] | None = None,
        evolution_trigger: EvolutionTriggerModule | Iterable[EvolutionTriggerModule] | None = None,
        memory_evolution: MemoryEvolutionModule | Iterable[MemoryEvolutionModule] | None = None,
        retrieval: RetrievalModule | Iterable[RetrievalModule] | None = None,
        readout: ReadoutModule | Iterable[ReadoutModule] | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.unit_formation = _materialize_slot_value(
            unit_formation if unit_formation is not None else _default_unit_formation()
        )
        self.representation = _materialize_slot_value(
            representation if representation is not None else _default_representation()
        )
        self.write_trigger = _materialize_slot_value(
            write_trigger if write_trigger is not None else _default_write_trigger()
        )
        self.organization = _materialize_slot_value(
            organization if organization is not None else _default_organization()
        )
        self.evolution_trigger = _materialize_slot_value(
            evolution_trigger if evolution_trigger is not None else _default_evolution_trigger()
        )
        self.memory_evolution = _materialize_slot_value(
            memory_evolution if memory_evolution is not None else _default_memory_evolution()
        )
        self.retrieval = _materialize_slot_value(
            retrieval if retrieval is not None else _default_retrieval()
        )
        self.readout = _materialize_slot_value(readout if readout is not None else _default_readout())
        self.store = store if store is not None else MemoryStore()
        self._validate_composition()

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
        self._validate_store_compatibility()

    def _validate_store_compatibility(self) -> None:
        from .baselines import GraphAppendOrganization

        for module in (
            *(_iter_nested_modules(getattr(self, slot_name)) for slot_name, _, _ in _INGEST_SLOT_CHECK + _RECALL_SLOT_CHECK),
        ):
            for nested_module in module:
                if isinstance(nested_module, GraphAppendOrganization):
                    self._validate_graph_organization(nested_module)
                self._validate_spec_requirements(nested_module)
                if hasattr(nested_module, "validate_store"):
                    nested_module.validate_store(self.store)

    def _validate_graph_organization(self, organization) -> None:
        target_layer = organization.target_layer
        declared_layers = list(self.store.topology.layer_names)
        if not self.store.has_layer(target_layer):
            raise IncompatibleCompositionError(
                "Incompatible MemoryPipeline composition: "
                f"slot='organization', module='{organization.spec.name}' requires "
                f"declared graph layer {target_layer!r}, but store topology only declares "
                f"{declared_layers}."
            )

        layer_shape = self.store.layer_shape(target_layer)
        if layer_shape != "Graph":
            raise IncompatibleCompositionError(
                "Incompatible MemoryPipeline composition: "
                f"slot='organization', module='{organization.spec.name}' requires "
                f"layer {target_layer!r} with shape='Graph', but store topology declares "
                f"shape={layer_shape!r}."
            )

    def _validate_spec_requirements(self, module: PrimitiveModule) -> None:
        for requirement in module.spec.store_requirements:
            self._validate_store_requirement(module, requirement)
        for requirement in module.spec.layer_requirements:
            self._validate_layer_requirement(module, requirement)

    def _validate_store_requirement(self, module: PrimitiveModule, requirement: str) -> None:
        if requirement in {"record.entities", "index:entity"} and not self.store.topology.has_index("entity"):
            self._raise_incompatible(module, requirement, "store topology declares no entity index.")
        elif requirement == "index:vector" and not self.store.topology.has_index("vector"):
            self._raise_incompatible(module, requirement, "store topology declares no vector index.")
        elif requirement == "index:keyword" and not self.store.topology.has_index("keyword"):
            self._raise_incompatible(module, requirement, "store topology declares no keyword index.")
        elif requirement == "index:tag" and not self.store.topology.has_index("tag"):
            self._raise_incompatible(module, requirement, "store topology declares no tag index.")
        elif requirement == "index:graph" and not self.store.topology.has_index("graph"):
            self._raise_incompatible(module, requirement, "store topology declares no graph index.")
        elif requirement == "shape:Graph" and not self.store.topology.has_graph_layer():
            self._raise_incompatible(module, requirement, "store topology declares no graph layer.")
        elif requirement.startswith("setting:"):
            setting_name = requirement.split(":", 1)[1]
            if not self.store.has_layer_setting(setting_name):
                self._raise_incompatible(module, requirement, f"store topology declares no layer setting {setting_name!r}.")

    def _validate_layer_requirement(self, module: PrimitiveModule, requirement: str) -> None:
        if requirement == "target_layer_exists":
            layer_name = getattr(module, "target_layer", None)
            if isinstance(layer_name, str) and layer_name and not self.store.has_layer(layer_name):
                self._raise_incompatible(
                    module,
                    requirement,
                    f"target layer {layer_name!r} is not declared in the store topology.",
                )
        elif requirement.startswith("target_layer_shape:"):
            layer_name = getattr(module, "target_layer", None)
            expected_shape = requirement.split(":", 1)[1]
            if isinstance(layer_name, str) and layer_name:
                if not self.store.has_layer(layer_name):
                    self._raise_incompatible(
                        module,
                        requirement,
                        f"target layer {layer_name!r} is not declared in the store topology.",
                    )
                actual_shape = self.store.layer_shape(layer_name)
                if actual_shape != expected_shape:
                    self._raise_incompatible(
                        module,
                        requirement,
                        f"target layer {layer_name!r} has shape={actual_shape!r}, expected {expected_shape!r}.",
                    )
        elif requirement.startswith("target_layer_index:"):
            layer_name = getattr(module, "target_layer", None)
            index_name = requirement.split(":", 1)[1]
            if isinstance(layer_name, str) and layer_name:
                if not self.store.has_layer(layer_name):
                    self._raise_incompatible(
                        module,
                        requirement,
                        f"target layer {layer_name!r} is not declared in the store topology.",
                    )
                if not self.store.layer_supports_index(layer_name, index_name):
                    self._raise_incompatible(
                        module,
                        requirement,
                        f"target layer {layer_name!r} does not support index {index_name!r}.",
                    )
        elif requirement.startswith("target_layer_setting:"):
            layer_name = getattr(module, "target_layer", None)
            setting_name = requirement.split(":", 1)[1]
            if isinstance(layer_name, str) and layer_name:
                if not self.store.has_layer(layer_name):
                    self._raise_incompatible(
                        module,
                        requirement,
                        f"target layer {layer_name!r} is not declared in the store topology.",
                    )
                if self.store.layer_setting(layer_name, setting_name, None) is None:
                    self._raise_incompatible(
                        module,
                        requirement,
                        f"target layer {layer_name!r} does not declare setting {setting_name!r}.",
                    )

    def _raise_incompatible(self, module: PrimitiveModule, requirement: str, detail: str) -> None:
        raise IncompatibleCompositionError(
            "Incompatible MemoryPipeline composition: "
            f"slot='{module.spec.slot}', module='{module.spec.name}' requires {requirement!r}, but {detail}"
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
