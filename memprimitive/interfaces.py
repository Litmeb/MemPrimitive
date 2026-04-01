"""Primitive interfaces for the stage-1 memory pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import normalize_contracts
from .core import MemoryStore, ModuleSpec, Packet


class PrimitiveModule(ABC):
    """Common base for all primitive modules."""

    spec: ModuleSpec
    requires_contracts: frozenset[str] = frozenset()
    produces_contracts: frozenset[str] = frozenset()

    @abstractmethod
    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        """Process a packet and optionally update the store."""

    def get_requires_contracts(self) -> frozenset[str]:
        """Return the normalized contracts this module expects from composition."""

        return normalize_contracts(getattr(self, "requires_contracts", ()))

    def get_produces_contracts(self) -> frozenset[str]:
        """Return the normalized contracts this module guarantees to contribute."""

        return normalize_contracts(getattr(self, "produces_contracts", ()))


class UnitFormationModule(PrimitiveModule):
    pass


class RepresentationModule(PrimitiveModule):
    pass


class TriggerModule(PrimitiveModule):
    pass


class OrganizationModule(PrimitiveModule):
    pass


class MemoryEvolutionModule(PrimitiveModule):
    pass


class RetrievalModule(PrimitiveModule):
    pass


class ReadoutModule(PrimitiveModule):
    pass
