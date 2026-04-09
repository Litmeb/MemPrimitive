"""Baseline: unit formation primitive."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Final

from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import UnitFormationModule

from ..utils._trace import copy_trace

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def _require_observation(packet: Packet, module_name: str):
    if packet.observation is None:
        raise ValueError(f"{module_name} requires packet.observation.")
    return packet.observation


def _build_unit(packet: Packet, text: str, *, metadata: dict | None = None) -> MemoryUnit:
    observation = _require_observation(packet, "Unit formation")
    return MemoryUnit(
        text=text,
        timestamp=observation.timestamp,
        metadata={
            "source": observation.source,
            "provenance": {
                "observation_id": observation.observation_id,
                "source": observation.source,
            },
            **observation.metadata,
            **({} if metadata is None else metadata),
        },
    )


class PassThroughUnitFormation(UnitFormationModule):
    """Map one observation to a single memory unit without splitting or filtering.

    ``run`` requires ``packet.observation`` (validated ``Observation`` from
    ``core``). Output is ``units`` with length 1; metadata includes ``source``
    and ``provenance`` (observation id and source). The store is unchanged.
    """

    spec = ModuleSpec(
        name="pass_through_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.text", "units.metadata.provenance"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        observation = _require_observation(packet, "PassThroughUnitFormation")
        unit = _build_unit(packet, observation.text)
        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id],
        }
        return replace(packet, units=[unit], trace=trace), store


class SentenceSplitUnitFormation(UnitFormationModule):
    """Split one observation into sentence-level units using lightweight rules.

    ``run`` requires ``packet.observation``. Produces one unit per detected
    sentence, preserving source/provenance metadata. Blank fragments are skipped.
    The store is unchanged.
    """

    spec = ModuleSpec(
        name="sentence_split_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.text", "units.metadata.provenance"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        observation = _require_observation(packet, "SentenceSplitUnitFormation")
        raw_parts = _SENTENCE_BOUNDARY.split(observation.text.strip())
        texts = [part.strip() for part in raw_parts if part.strip()]
        if not texts:
            texts = [observation.text.strip()]

        units = [
            _build_unit(packet, text, metadata={"sentence_index": idx})
            for idx, text in enumerate(texts)
        ]
        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
        }
        return replace(packet, units=units, trace=trace), store


BASELINE_SLOT: Final[str] = "unit_formation"
BASELINE_CLASSES: Final[tuple[type[UnitFormationModule], ...]] = (
    PassThroughUnitFormation,
    SentenceSplitUnitFormation,
)
