"""Baseline: unit formation primitive."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Final

from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import UnitFormationModule

from ._trace import copy_trace

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


class LineSplitUnitFormation(UnitFormationModule):
    """Split a note/log observation into one unit per non-empty line.

    ``run`` requires ``packet.observation``. Empty lines are dropped, ordering is
    preserved, and provenance metadata is copied to each unit. The store is unchanged.
    """

    spec = ModuleSpec(
        name="line_split_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.text", "units.metadata.provenance"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        observation = _require_observation(packet, "LineSplitUnitFormation")
        texts = [line.strip() for line in observation.text.splitlines() if line.strip()]
        if not texts:
            texts = [observation.text.strip()]

        units = [
            _build_unit(packet, text, metadata={"line_index": idx})
            for idx, text in enumerate(texts)
        ]
        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
        }
        return replace(packet, units=units, trace=trace), store


class WindowedUnitFormation(UnitFormationModule):
    """Split long observations into fixed-size overlapping text windows.

    Constructor: ``window_size`` and ``stride`` must be positive integers.

    ``run`` requires ``packet.observation``. Produces windows over the stripped
    observation text and stores ``window_index`` in each unit metadata. The store
    is unchanged.
    """

    spec = ModuleSpec(
        name="windowed_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.text", "units.metadata.provenance"),
    )

    def __init__(self, *, window_size: int = 120, stride: int = 80) -> None:
        if window_size <= 0:
            raise ValueError("WindowedUnitFormation requires window_size > 0.")
        if stride <= 0:
            raise ValueError("WindowedUnitFormation requires stride > 0.")
        self.window_size = int(window_size)
        self.stride = int(stride)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        observation = _require_observation(packet, "WindowedUnitFormation")
        text = observation.text.strip()
        if len(text) <= self.window_size:
            units = [_build_unit(packet, text, metadata={"window_index": 0})]
        else:
            units: list[MemoryUnit] = []
            start = 0
            window_index = 0
            while start < len(text):
                chunk = text[start : start + self.window_size].strip()
                if chunk:
                    units.append(_build_unit(packet, chunk, metadata={"window_index": window_index}))
                    window_index += 1
                if start + self.window_size >= len(text):
                    break
                start += self.stride

        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
            "window_size": self.window_size,
            "stride": self.stride,
        }
        return replace(packet, units=units, trace=trace), store


class MetadataHintUnitFormation(UnitFormationModule):
    """Materialize units from ``observation.metadata['units']`` when provided.

    ``run`` requires ``packet.observation``. Supported hints are plain strings or
    dicts with at least ``text`` and optional ``unit_type`` / ``metadata``.
    Without hints, the module falls back to pass-through behavior. The store is unchanged.
    """

    spec = ModuleSpec(
        name="metadata_hint_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.text", "units.metadata.provenance"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        observation = _require_observation(packet, "MetadataHintUnitFormation")
        hints = observation.metadata.get("units")
        units: list[MemoryUnit] = []

        if isinstance(hints, list) and hints:
            for idx, hint in enumerate(hints):
                if isinstance(hint, str) and hint.strip():
                    units.append(_build_unit(packet, hint.strip(), metadata={"hint_index": idx}))
                    continue
                if isinstance(hint, dict) and str(hint.get("text", "")).strip():
                    unit = _build_unit(
                        packet,
                        str(hint["text"]).strip(),
                        metadata={
                            "hint_index": idx,
                            **(hint.get("metadata") if isinstance(hint.get("metadata"), dict) else {}),
                        },
                    )
                    if "unit_type" in hint and str(hint["unit_type"]).strip():
                        unit = replace(unit, unit_type=str(hint["unit_type"]).strip())
                    units.append(unit)
            hint_mode = "metadata"
        else:
            units = [_build_unit(packet, observation.text.strip())]
            hint_mode = "fallback"

        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "mode": hint_mode,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
        }
        return replace(packet, units=units, trace=trace), store


BASELINE_SLOT: Final[str] = "unit_formation"
BASELINE_CLASSES: Final[tuple[type[UnitFormationModule], ...]] = (
    PassThroughUnitFormation,
    SentenceSplitUnitFormation,
    LineSplitUnitFormation,
    WindowedUnitFormation,
    MetadataHintUnitFormation,
)
