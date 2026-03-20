"""Backward-compatible aggregate of all stage-1 baselines.

Historically all baseline classes lived in this module. They are now split by
primitive slot (``unit_formation``, ``representation``, …). Import from
``memprimitive.baselines`` or from the slot module (e.g.
``memprimitive.baselines.unit_formation``) for new code.

This module re-exports the same public names so that
``from memprimitive.baselines.simple import PassThroughUnitFormation`` keeps
working.
"""

from __future__ import annotations

from .memory_evolution import AppendOnlyEvolution
from .organization import AppendOrganization
from .readout import ConcatenateReadout
from .representation import BasicRepresentation
from .retrieval import RecencyRetrieval
from .unit_formation import PassThroughUnitFormation
from .write_trigger import AlwaysWriteTrigger

__all__ = [
    "AlwaysWriteTrigger",
    "AppendOnlyEvolution",
    "AppendOrganization",
    "BasicRepresentation",
    "ConcatenateReadout",
    "PassThroughUnitFormation",
    "RecencyRetrieval",
]
