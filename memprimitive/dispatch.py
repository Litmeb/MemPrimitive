"""Explicit slot dispatchers that fan out one packet snapshot to multiple modules."""
from __future__ import annotations

from memprimitive.interfaces import PrimitiveModule
from copy import deepcopy
from dataclasses import replace

from .contracts import normalize_contracts
from .core import MemoryStore, ModuleSpec, Packet
from .interfaces import (
    MemoryEvolutionModule,
    OrganizationModule,
    ReadoutModule,
    RepresentationModule,
    RetrievalModule,
    TriggerModule,
    UnitFormationModule,
)


class _DispatchMixin:
    """Fan out a packet snapshot to child modules while keeping one primary result.

    All child modules receive the same cloned input ``Packet`` but share the same
    evolving ``MemoryStore``. The ``primary_index`` child determines the packet
    returned to downstream slots; every child's slot-local trace is recorded under
    ``trace["dispatch"][spec.slot]``.
    """

    child_base: type[PrimitiveModule]
    spec: ModuleSpec

    def __init__(
        self,
        modules: tuple[PrimitiveModule, ...] | list[PrimitiveModule],
        *,
        primary_index: int = 0,
        name: str | None = None,
    ) -> None:
        materialized = tuple[PrimitiveModule, ...](modules)
        if not materialized:
            raise ValueError(f"{type(self).__name__} requires at least one child module.")
        if primary_index < 0 or primary_index >= len(materialized):
            raise ValueError(f"{type(self).__name__} requires primary_index within module bounds.")
        for module in materialized:
            if not isinstance(module, self.child_base):
                raise TypeError(
                    f"{type(self).__name__} children must be {self.child_base.__name__} instances, "
                    f"got {type(module).__name__}."
                )
            if module.spec.slot != self.spec.slot:
                raise ValueError(
                    f"{type(self).__name__} expects child ModuleSpec.slot={self.spec.slot!r}, "
                    f"got {module.spec.slot!r} on {type(module).__name__}."
                )
        self.modules = materialized
        self.primary_index = primary_index
        if name is not None:
            self.spec = replace(self.spec, name=name)

    def iter_child_modules(self) -> tuple[PrimitiveModule, ...]:
        return self.modules

    def get_requires_contracts(self) -> frozenset[str]:
        return normalize_contracts(
            contract
            for module in self.modules
            for contract in module.get_requires_contracts()
        )

    def get_produces_contracts(self) -> frozenset[str]:
        return normalize_contracts(
            contract
            for module in self.modules
            for contract in module.get_produces_contracts()
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        snapshot = deepcopy(packet)
        branch_results: list[tuple[PrimitiveModule, Packet]] = []
        for module in self.modules:
            branch_packet, store = module.run(deepcopy(snapshot), store)
            branch_results.append((module, branch_packet))

        _, primary_packet = branch_results[self.primary_index]
        trace = deepcopy(primary_packet.trace)
        dispatch_trace = {
            "module": self.spec.name,
            "primary_index": self.primary_index,
            "children": [
                {
                    "index": idx,
                    "module": module.spec.name,
                    "slot": module.spec.slot,
                    "slot_trace": branch_packet.trace.get(self.spec.slot),
                }
                for idx, (module, branch_packet) in enumerate(branch_results)
            ],
        }
        trace.setdefault("dispatch", {})
        trace["dispatch"][self.spec.slot] = dispatch_trace
        return replace(primary_packet, trace=trace), store


class DispatchUnitFormation(_DispatchMixin, UnitFormationModule):
    child_base = UnitFormationModule
    spec = ModuleSpec(name="dispatch_unit_formation", slot="unit_formation")


class DispatchRepresentation(_DispatchMixin, RepresentationModule):
    child_base = RepresentationModule
    spec = ModuleSpec(name="dispatch_representation", slot="representation")


class DispatchWriteTrigger(_DispatchMixin, TriggerModule):
    child_base = TriggerModule
    spec = ModuleSpec(name="dispatch_write_trigger", slot="write_trigger")


class DispatchOrganization(_DispatchMixin, OrganizationModule):
    child_base = OrganizationModule
    spec = ModuleSpec(name="dispatch_organization", slot="organization")


class DispatchEvolutionTrigger(_DispatchMixin, TriggerModule):
    child_base = TriggerModule
    spec = ModuleSpec(name="dispatch_evolution_trigger", slot="evolution_trigger")


class DispatchMemoryEvolution(_DispatchMixin, MemoryEvolutionModule):
    child_base = MemoryEvolutionModule
    spec = ModuleSpec(name="dispatch_memory_evolution", slot="memory_evolution")


class DispatchRetrieval(_DispatchMixin, RetrievalModule):
    child_base = RetrievalModule
    spec = ModuleSpec(name="dispatch_retrieval", slot="retrieval")


class DispatchReadout(_DispatchMixin, ReadoutModule):
    child_base = ReadoutModule
    spec = ModuleSpec(name="dispatch_readout", slot="readout")


__all__ = [
    "DispatchEvolutionTrigger",
    "DispatchMemoryEvolution",
    "DispatchOrganization",
    "DispatchReadout",
    "DispatchRepresentation",
    "DispatchRetrieval",
    "DispatchUnitFormation",
    "DispatchWriteTrigger",
]
