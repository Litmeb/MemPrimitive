"""Baseline: organization primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet, Placement
from ..interfaces import OrganizationModule

from ..utils._graph_family import graph_metadata_for_unit
from ..utils._reflexion_family import DEFAULT_TRIAL_LAYER
from ..utils._trace import copy_trace


class AppendOrganization(OrganizationModule):
    """Assign each unit to a fixed target layer and commit normal ingest-time writes.

    Constructor: ``target_layer`` must be a non-empty string (same rules as
    ``Placement.target_layer`` / ``core._require_non_empty_text``).

    ``run`` requires ``packet.units`` and ``packet.decisions`` with equal length.
    Emits one ``Placement`` per unit and appends ``MemoryRecord`` objects for
    units whose decision is ``True``. Mutates ``store`` as part of the normal
    write path.
    """

    spec = ModuleSpec(
        name="append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, target_layer: str = "default") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("AppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("AppendOrganization requires decisions aligned with units.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        skipped_units = 0
        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
            store.append(record)
            written_record_ids.append(record.record_id)
            written_unit_ids.append(unit.unit_id)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "skipped_unit_count": skipped_units,
        }
        return replace(packet, placements=placements, trace=trace), store


def _append_records_for_placements(
    packet: Packet,
    store: MemoryStore,
    placements: list[Placement],
    *,
    metadata_builder: Any | None = None,
) -> tuple[list[str], list[str], int]:
    written_record_ids: list[str] = []
    written_unit_ids: list[str] = []
    skipped_units = 0
    for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
        if not decision:
            skipped_units += 1
            continue
        sequence_id = store.next_sequence_id()
        record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
        if metadata_builder is not None:
            record.metadata.update(metadata_builder(unit, placement))
        store.append(record)
        written_record_ids.append(record.record_id)
        written_unit_ids.append(unit.unit_id)
    return written_record_ids, written_unit_ids, skipped_units


class ConditionalLayerOrganization(OrganizationModule):
    """Route units to layers based on tags/entities/unit metadata, then append.

    Constructor: ``default_layer`` must be declared in topology or creatable by
    the store. ``rules`` is an ordered tuple of dict rules using one of
    ``has_entity``, ``unit_type``, ``tag_contains``, or ``metadata_key`` to pick
    a ``target_layer``.

    ``run`` requires ``packet.units`` and ``packet.decisions`` with equal length.
    Emits aligned placements and commits normal append-only writes to the chosen
    target layers.
    """

    spec = ModuleSpec(
        name="conditional_layer_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, default_layer: str = "default", rules: tuple[dict[str, Any], ...] = ()) -> None:
        self.default_layer = default_layer
        self.rules = rules

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("ConditionalLayerOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("ConditionalLayerOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("ConditionalLayerOrganization requires decisions aligned with units.")

        placements = [
            Placement(unit_id=unit.unit_id, target_layer=self._target_layer_for_unit(unit))
            for unit in packet.units
        ]
        written_record_ids, written_unit_ids, skipped_units = _append_records_for_placements(packet, store, placements)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "default_layer": self.default_layer,
            "placements": [
                {"unit_id": placement.unit_id, "target_layer": placement.target_layer}
                for placement in placements
            ],
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "skipped_unit_count": skipped_units,
        }
        return replace(packet, placements=placements, trace=trace), store

    def _target_layer_for_unit(self, unit) -> str:
        for rule in self.rules:
            target_layer = str(rule.get("target_layer", "")).strip()
            if not target_layer:
                continue
            if rule.get("has_entity") is True and unit.entities:
                return target_layer
            if "unit_type" in rule and str(rule["unit_type"]).strip() == unit.unit_type:
                return target_layer
            if "tag_contains" in rule and str(rule["tag_contains"]).strip():
                needle = str(rule["tag_contains"]).casefold()
                if any(needle in str(tag).casefold() for tag in unit.tags):
                    return target_layer
            if "metadata_key" in rule and str(rule["metadata_key"]).strip():
                if rule["metadata_key"] in unit.metadata:
                    return target_layer
        return self.default_layer


class GraphAppendOrganization(OrganizationModule):
    """Append units into a graph-shaped layer while annotating graph metadata.

    Constructor: ``target_layer`` must refer to a declared ``Graph`` layer.

    ``run`` requires ``packet.units`` and ``packet.decisions`` with equal length.
    The module appends standard ``MemoryRecord`` rows, but enriches metadata with
    graph-node hints derived from unit entities/triples.
    """

    spec = ModuleSpec(
        name="graph_append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        store_requirements=("shape:Graph", "index:graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = "knowledge_graph") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphAppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GraphAppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("GraphAppendOrganization requires decisions aligned with units.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(f"GraphAppendOrganization requires target layer {self.target_layer!r} to be Graph.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids, written_unit_ids, skipped_units = _append_records_for_placements(
            packet,
            store,
            placements,
            metadata_builder=self._graph_metadata,
        )
        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "skipped_unit_count": skipped_units,
            "graph_metadata_schema": (
                "graph.layer",
                "graph.shape",
                "graph.entities",
                "graph.triples",
                "graph.links",
                "graph.node_count",
                "graph.link_count",
                "graph.last_linked_at",
                "graph.link_history",
            ),
        }
        return replace(packet, placements=placements, trace=trace), store

    @staticmethod
    def _graph_metadata(unit, placement: Placement) -> dict[str, Any]:
        return {
            "graph": graph_metadata_for_unit(unit, layer=placement.target_layer),
        }


class PlacementWithoutAppendOrganization(OrganizationModule):
    """Emit placements without persisting the current units into the store.

    Constructor: ``target_layer`` must be a non-empty layer name. This module is
    useful when the packet needs a placement contract for downstream evolution,
    but the current trial/buffer contents themselves should remain ephemeral.

    ``run`` requires ``packet.units`` and ``packet.decisions`` with equal length.
    It emits aligned ``Placement`` objects and records the routing decision in
    trace, but intentionally does not append any ``MemoryRecord`` objects.
    """

    spec = ModuleSpec(
        name="placement_without_append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
    )

    def __init__(self, *, target_layer: str = DEFAULT_TRIAL_LAYER) -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("PlacementWithoutAppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("PlacementWithoutAppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("PlacementWithoutAppendOrganization requires decisions aligned with units.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "placement_count": len(placements),
            "written_record_ids": [],
            "written_unit_ids": [],
            "skipped_unit_count": 0,
            "append_trials": False,
        }
        return replace(packet, placements=placements, trace=trace), store


class GraphAppendLinkReadyOrganization(OrganizationModule):
    """Append enriched notes into a graph layer with link-ready metadata.

    Constructor: ``target_layer`` must name a declared graph layer that also
    exposes ``graph`` and ``vector`` indices. ``note_namespace`` records which
    enriched-note schema is expected to travel with each unit.

    ``run`` requires aligned ``packet.units`` and ``packet.decisions``. It
    appends regular ``MemoryRecord`` rows, preserves existing note metadata, and
    initializes graph-link fields so later graph evolution modules can safely
    write links/context without repairing the whole record shape first.
    """

    spec = ModuleSpec(
        name="graph_append_link_ready_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
        store_requirements=("index:graph", "index:vector", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph", "target_layer_index:vector"),
    )

    def __init__(self, *, target_layer: str = "knowledge_graph", note_namespace: str = "note") -> None:
        self.target_layer = target_layer
        self.note_namespace = note_namespace

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphAppendLinkReadyOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GraphAppendLinkReadyOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("GraphAppendLinkReadyOrganization requires decisions aligned with units.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(f"GraphAppendLinkReadyOrganization requires target layer {self.target_layer!r} to be Graph.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        effects: list[dict[str, Any]] = []
        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                effects.append({"unit_id": unit.unit_id, "effect_type": "skipped"})
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
            record.metadata["graph"] = {
                **graph_metadata_for_unit(unit, layer=placement.target_layer),
                "link_ready": True,
                "neighbor_context": {"neighbor_record_ids": [], "neighbor_count": 0},
            }
            store.append(record)
            effects.append(
                {
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "effect_type": "append_note",
                    "target_layer": self.target_layer,
                    "note_namespace": self.note_namespace,
                }
            )

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "note_namespace": self.note_namespace,
            "effects": effects,
            "graph_metadata_schema": (
                "graph.layer",
                "graph.shape",
                "graph.entities",
                "graph.triples",
                "graph.links",
                "graph.link_ready",
                "graph.neighbor_context",
            ),
        }
        return replace(packet, placements=placements, trace=trace), store


BASELINE_SLOT: Final[str] = "organization"
BASELINE_CLASSES: Final[tuple[type[OrganizationModule], ...]] = (
    AppendOrganization,
    ConditionalLayerOrganization,
    GraphAppendOrganization,
    PlacementWithoutAppendOrganization,
    GraphAppendLinkReadyOrganization,
)
