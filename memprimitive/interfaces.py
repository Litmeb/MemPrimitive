"""Primitive interfaces for the stage-1 memory pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .core import MemoryStore, ModuleSpec, Packet


class PrimitiveModule(ABC):
    """Common base for all primitive modules."""

    spec: ModuleSpec

    @abstractmethod
    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        """Process a packet and optionally update the store."""


class UnitFormationModule(PrimitiveModule):
    pass


class RepresentationModule(PrimitiveModule):
    pass


class WriteTriggerModule(PrimitiveModule):
    pass


class EvolutionTriggerModule(PrimitiveModule):
    pass


class OrganizationModule(PrimitiveModule):
    pass


class MemoryEvolutionModule(PrimitiveModule):
    pass


class RetrievalModule(PrimitiveModule):
    pass


class ReadoutModule(PrimitiveModule):
    pass
