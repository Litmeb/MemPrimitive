"""Baseline: organization primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Final

from ..contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_NOTE_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
)
from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet, Placement
from ..interfaces import OrganizationModule

from ..utils._graph_family import graph_metadata_for_unit, graph_metadata_from_record, normalize_graph_metadata
from ..utils._hierarchical_family import (
    append_hierarchical_records,
    build_extracted_triple_metadata,
    build_fixed_placements,
    group_records,
    inferred_target_layer,
    require_aligned_units_decisions,
    resolve_source_records,
    validate_hierarchical_config,
)
from ..utils._llm_function_tools import (
    ToolExecutionState,
    WriteToolCallContext,
    WriteToolSpec,
    build_runtime_tools,
    normalize_write_tool_specs,
    project_tool_specs_for_prompt,
    write_tool_specs_require_graph_contracts,
)
from ..utils._runtime import Runtime, get_runtime
from ..utils._template import (
    PromptPlan,
    ensure_prompt_plan,
    project_record_for_template,
    project_unit_for_template,
    render_prompt_plan,
)
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


def _record_from_unit_with_text(
    unit,
    *,
    layer: str,
    sequence_id: int,
    text: str,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    normalized_text = str(text).strip()
    if not normalized_text:
        raise ValueError("text override must be a non-empty string.")
    effective_embedding = list(unit.embedding) if embedding is None and unit.embedding is not None else embedding
    representation_elements = set(unit.representation_elements)
    if effective_embedding is not None:
        representation_elements.add("embedding")
    projected_unit = replace(
        unit,
        text=normalized_text,
        normalized_text=normalized_text.casefold().strip(),
        embedding=None if effective_embedding is None else list(effective_embedding),
        representation_elements=tuple(sorted(representation_elements)),
    )
    return MemoryRecord.from_unit(unit=projected_unit, layer=layer, sequence_id=sequence_id)


def _entity_embedding_map_from_unit(unit) -> dict[str, list[float]]:
    representation = unit.metadata.get("representation", {})
    if not isinstance(representation, dict):
        return {}
    raw_entity_embeddings = representation.get("entity_embeddings", {})
    if not isinstance(raw_entity_embeddings, dict):
        return {}
    normalized: dict[str, list[float]] = {}
    for raw_entity, raw_embedding in raw_entity_embeddings.items():
        entity_text = str(raw_entity).strip()
        if not entity_text or not isinstance(raw_embedding, list):
            continue
        try:
            embedding = [float(value) for value in raw_embedding]
        except (TypeError, ValueError):
            continue
        if not embedding:
            continue
        normalized[entity_text] = embedding
    return normalized


def _representation_summary_for_graph_merge(unit, existing_record: MemoryRecord) -> dict[str, Any]:
    existing_representation = existing_record.metadata.get("representation", {})
    if not isinstance(existing_representation, dict):
        existing_representation = {}
    updated = {
        **existing_representation,
        "text": unit.text,
        "normalized_text": unit.normalized_text or unit.text.casefold().strip(),
    }
    if unit.embedding is not None:
        updated["embedding"] = {"dim": len(unit.embedding)}
    return updated


def _merge_graph_triples(
    existing_triples: list[tuple[str, str, str]],
    incoming_triples: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    merged: dict[tuple[str, str], tuple[str, str, str]] = {}
    for triple in existing_triples:
        merged[(str(triple[1]), str(triple[2]))] = tuple(str(value) for value in triple)
    for triple in incoming_triples:
        merged[(str(triple[1]), str(triple[2]))] = tuple(str(value) for value in triple)
    return list(merged.values())


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
    requires_contracts = frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, RECORD_NOTE_PAYLOAD_CONTRACT})

    def __init__(
        self,
        *,
        target_layer: str = "knowledge_graph",
        separate: bool = False,
        separate_layer: str | None = None,
    ) -> None:
        self.target_layer = target_layer
        self.separate = separate
        self.separate_layer = None if separate_layer is None else str(separate_layer).strip()

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphAppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GraphAppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("GraphAppendOrganization requires decisions aligned with units.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(f"GraphAppendOrganization requires target layer {self.target_layer!r} to be Graph.")
        if self.separate and not self.separate_layer:
            raise ValueError("GraphAppendOrganization requires separate_layer when separate=True.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        separate_source_record_ids: list[str] = []
        if self.separate:
            written_record_ids, written_unit_ids, skipped_units, separate_source_record_ids = self._append_separate_records(
                packet,
                store,
                placements,
            )
        else:
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
            "separate": self.separate,
            "separate_layer": self.separate_layer,
            "writes_embedding_from_record_field": True,
            "records_with_embedding": sum(
                1
                for record_id in written_record_ids
                for record in store.iter_records(self.target_layer)
                if record.record_id == record_id and record.embedding is not None
            ),
            "written_record_ids": written_record_ids,
            "source_written_record_ids": separate_source_record_ids,
            "triple_written_record_ids": written_record_ids,
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

    def _append_separate_records(
        self,
        packet: Packet,
        store: MemoryStore,
        placements: list[Placement],
    ) -> tuple[list[str], list[str], int, list[str]]:
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        source_written_record_ids: list[str] = []
        skipped_units = 0
        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                continue

            source_sequence_id = store.next_sequence_id()
            source_record = MemoryRecord.from_unit(unit=unit, layer=self.separate_layer, sequence_id=source_sequence_id)
            store.append(source_record)
            source_written_record_ids.append(source_record.record_id)

            triple_sequence_id = store.next_sequence_id()
            triple_record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=triple_sequence_id)
            triple_record.metadata.update(self._graph_metadata(unit, placement))
            triple_record.metadata.update(
                build_extracted_triple_metadata(
                    source_layer=self.separate_layer,
                    target_layer=placement.target_layer,
                    source_record=source_record,
                    triples=list(triple_record.metadata["graph"]["triples"]),
                )
            )
            store.append(triple_record)
            written_record_ids.append(triple_record.record_id)
            written_unit_ids.append(unit.unit_id)
        return written_record_ids, written_unit_ids, skipped_units, source_written_record_ids


class GraphEntityAppendOrganization(GraphAppendOrganization):
    """Append one graph record per extracted entity instead of one per raw unit."""

    spec = ModuleSpec(
        name="graph_entity_append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        store_requirements=("shape:Graph", "index:graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "append_records"),
    )
    requires_contracts = frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, RECORD_NOTE_PAYLOAD_CONTRACT})

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphEntityAppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GraphEntityAppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("GraphEntityAppendOrganization requires decisions aligned with units.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(f"GraphEntityAppendOrganization requires target layer {self.target_layer!r} to be Graph.")
        if self.separate and not self.separate_layer:
            raise ValueError("GraphEntityAppendOrganization requires separate_layer when separate=True.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        source_written_record_ids: list[str] = []
        if self.separate:
            written_record_ids, written_unit_ids, skipped_units, source_written_record_ids = self._append_separate_records(
                packet,
                store,
                placements,
            )
        else:
            written_record_ids, written_unit_ids, skipped_units = self._append_entity_records(packet, store, placements)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "separate": self.separate,
            "separate_layer": self.separate_layer,
            "fanout_mode": "per_entity",
            "writes_embedding_from_record_field": True,
            "records_with_embedding": sum(
                1
                for record_id in written_record_ids
                for record in store.iter_records(self.target_layer)
                if record.record_id == record_id and record.embedding is not None
            ),
            "written_record_ids": written_record_ids,
            "source_written_record_ids": source_written_record_ids,
            "entity_written_record_ids": written_record_ids,
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
    def _entity_texts(unit) -> list[str]:
        return list(dict.fromkeys(str(entity).strip() for entity in unit.entities if str(entity).strip()))

    def _append_entity_records(
        self,
        packet: Packet,
        store: MemoryStore,
        placements: list[Placement],
    ) -> tuple[list[str], list[str], int]:
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        skipped_units = 0
        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                continue
            entity_texts = self._entity_texts(unit)
            if not entity_texts:
                skipped_units += 1
                continue
            for entity_text in entity_texts:
                sequence_id = store.next_sequence_id()
                record = _record_from_unit_with_text(
                    unit,
                    layer=placement.target_layer,
                    sequence_id=sequence_id,
                    text=entity_text,
                )
                record.metadata.update(self._graph_metadata(unit, placement))
                store.append(record)
                written_record_ids.append(record.record_id)
                written_unit_ids.append(unit.unit_id)
        return written_record_ids, written_unit_ids, skipped_units

    def _append_separate_records(
        self,
        packet: Packet,
        store: MemoryStore,
        placements: list[Placement],
    ) -> tuple[list[str], list[str], int, list[str]]:
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        source_written_record_ids: list[str] = []
        skipped_units = 0
        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                continue
            entity_texts = self._entity_texts(unit)
            if not entity_texts:
                skipped_units += 1
                continue

            source_sequence_id = store.next_sequence_id()
            source_record = MemoryRecord.from_unit(unit=unit, layer=self.separate_layer, sequence_id=source_sequence_id)
            store.append(source_record)
            source_written_record_ids.append(source_record.record_id)

            for entity_text in entity_texts:
                entity_sequence_id = store.next_sequence_id()
                entity_record = _record_from_unit_with_text(
                    unit,
                    layer=placement.target_layer,
                    sequence_id=entity_sequence_id,
                    text=entity_text,
                )
                entity_record.metadata.update(self._graph_metadata(unit, placement))
                entity_record.metadata.update(
                    build_extracted_triple_metadata(
                        source_layer=self.separate_layer,
                        target_layer=placement.target_layer,
                        source_record=source_record,
                        triples=list(entity_record.metadata["graph"]["triples"]),
                    )
                )
                store.append(entity_record)
                written_record_ids.append(entity_record.record_id)
                written_unit_ids.append(unit.unit_id)
        return written_record_ids, written_unit_ids, skipped_units, source_written_record_ids


class GraphDeduplicationAppendOrganization(OrganizationModule):
    """Append graph records, but merge into the nearest existing node when similar enough."""

    spec = ModuleSpec(
        name="graph_deduplication_append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        store_requirements=("shape:Graph", "index:graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "append_records", "rewrite_records"),
    )
    requires_contracts = frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT})

    def __init__(
        self,
        *,
        target_layer: str = "knowledge_graph",
        threshold: float,
        separate: bool = False,
        separate_layer: str | None = None,
    ) -> None:
        self.target_layer = target_layer
        self.threshold = float(threshold)
        self.separate = separate
        self.separate_layer = None if separate_layer is None else str(separate_layer).strip()

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphDeduplicationAppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GraphDeduplicationAppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("GraphDeduplicationAppendOrganization requires decisions aligned with units.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(
                f"GraphDeduplicationAppendOrganization requires target layer {self.target_layer!r} to be Graph."
            )
        if self.separate and not self.separate_layer:
            raise ValueError("GraphDeduplicationAppendOrganization requires separate_layer when separate=True.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        source_written_record_ids: list[str] = []
        effects: list[dict[str, Any]] = []
        skipped_units = 0

        for raw_unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                effects.append({"unit_id": raw_unit.unit_id, "effect_type": "skipped"})
                continue

            unit = raw_unit
            source_record = None
            if self.separate:
                source_sequence_id = store.next_sequence_id()
                source_record = MemoryRecord.from_unit(unit=unit, layer=self.separate_layer, sequence_id=source_sequence_id)
                store.append(source_record)
                source_written_record_ids.append(source_record.record_id)

            matched_record, top1_similarity, embedding_source = self._find_best_match(store, unit)
            if matched_record is not None and top1_similarity > self.threshold:
                merged_record = self._merge_record(matched_record, unit)
                if source_record is not None:
                    merged_record.metadata.update(
                        build_extracted_triple_metadata(
                            source_layer=self.separate_layer,
                            target_layer=placement.target_layer,
                            source_record=source_record,
                            triples=list(merged_record.metadata["graph"]["triples"]),
                        )
                    )
                store.replace_record(self.target_layer, matched_record.record_id, merged_record)
                written_record_ids.append(matched_record.record_id)
                written_unit_ids.append(unit.unit_id)
                effects.append(
                    {
                        "unit_id": unit.unit_id,
                        "effect_type": "merge",
                        "matched_record_id": matched_record.record_id,
                        "top1_similarity": float(top1_similarity),
                        "threshold": self.threshold,
                        "embedding_source": embedding_source,
                        "record_has_embedding": merged_record.embedding is not None,
                        "source_record_id": None if source_record is None else source_record.record_id,
                    }
                )
                continue

            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=placement.target_layer, sequence_id=sequence_id)
            record.metadata.update({"graph": graph_metadata_for_unit(unit, layer=placement.target_layer)})
            if source_record is not None:
                record.metadata.update(
                    build_extracted_triple_metadata(
                        source_layer=self.separate_layer,
                        target_layer=placement.target_layer,
                        source_record=source_record,
                        triples=list(record.metadata["graph"]["triples"]),
                    )
                )
            store.append(record)
            written_record_ids.append(record.record_id)
            written_unit_ids.append(unit.unit_id)
            effects.append(
                {
                    "unit_id": unit.unit_id,
                    "effect_type": "append",
                    "record_id": record.record_id,
                    "top1_similarity": None if top1_similarity is None else float(top1_similarity),
                    "threshold": self.threshold,
                    "embedding_source": embedding_source,
                    "record_has_embedding": record.embedding is not None,
                    "source_record_id": None if source_record is None else source_record.record_id,
                }
            )

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "threshold": self.threshold,
            "separate": self.separate,
            "separate_layer": self.separate_layer,
            "writes_embedding_from_record_field": True,
            "records_with_embedding": sum(
                1
                for record_id in written_record_ids
                for record in store.iter_records(self.target_layer)
                if record.record_id == record_id and record.embedding is not None
            ),
            "written_record_ids": written_record_ids,
            "source_written_record_ids": source_written_record_ids,
            "written_unit_ids": written_unit_ids,
            "skipped_unit_count": skipped_units,
            "effects": effects,
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

    def _find_best_match(self, store: MemoryStore, unit) -> tuple[MemoryRecord | None, float | None, str]:
        embedding_source = "existing_unit_embedding"
        candidate_embedding = None if unit.embedding is None else list(unit.embedding)
        if candidate_embedding is None:
            candidate_embedding = store.embedding_for_record(self.target_layer, unit.text)
            if candidate_embedding is not None:
                embedding_source = "store_policy_fallback"
            else:
                candidate_embedding = list(get_runtime().embed(unit.text))
                embedding_source = "runtime_fallback"
        if candidate_embedding is None:
            return None, None, embedding_source
        best_record: MemoryRecord | None = None
        best_similarity: float | None = None
        for candidate in store.iter_records(self.target_layer):
            if candidate.embedding is None or len(candidate.embedding) != len(candidate_embedding):
                continue
            similarity = Runtime.cosine_similarity(candidate_embedding, candidate.embedding)
            if best_similarity is None or similarity > best_similarity:
                best_record = candidate
                best_similarity = float(similarity)
        if unit.embedding is None:
            unit.embedding = list(candidate_embedding)
        return best_record, best_similarity, embedding_source

    def _merge_record(self, existing_record: MemoryRecord, unit) -> MemoryRecord:
        existing_graph = graph_metadata_from_record(existing_record)
        merged_graph = normalize_graph_metadata(
            {
                **existing_graph,
                "entities": list(unit.entities),
                "triples": _merge_graph_triples(list(existing_graph["triples"]), list(unit.triples)),
            },
            layer=existing_record.layer,
        )
        return MemoryRecord(
            record_id=existing_record.record_id,
            unit_id=unit.unit_id,
            layer=existing_record.layer,
            text=unit.text,
            timestamp=unit.timestamp,
            embedding=None if unit.embedding is None else list(unit.embedding),
            metadata={
                **existing_record.metadata,
                "unit_type": unit.unit_type,
                "representation": _representation_summary_for_graph_merge(unit, existing_record),
                "graph": merged_graph,
            },
        )


class GraphEntityDeduplicationAppendOrganization(GraphDeduplicationAppendOrganization):
    """Deduplicate graph writes per extracted entity instead of per raw unit."""

    spec = ModuleSpec(
        name="graph_entity_deduplication_append_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        store_requirements=("shape:Graph", "index:graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "append_records", "rewrite_records"),
    )
    requires_contracts = frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT})

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphEntityDeduplicationAppendOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GraphEntityDeduplicationAppendOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("GraphEntityDeduplicationAppendOrganization requires decisions aligned with units.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(
                f"GraphEntityDeduplicationAppendOrganization requires target layer {self.target_layer!r} to be Graph."
            )
        if self.separate and not self.separate_layer:
            raise ValueError("GraphEntityDeduplicationAppendOrganization requires separate_layer when separate=True.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        source_written_record_ids: list[str] = []
        effects: list[dict[str, Any]] = []
        skipped_units = 0
        skipped_entities = 0

        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                effects.append({"unit_id": unit.unit_id, "effect_type": "skipped"})
                continue

            entity_texts = GraphEntityAppendOrganization._entity_texts(unit)
            if not entity_texts:
                skipped_units += 1
                effects.append({"unit_id": unit.unit_id, "effect_type": "skipped_no_entities"})
                continue

            source_record = None
            if self.separate:
                source_sequence_id = store.next_sequence_id()
                source_record = MemoryRecord.from_unit(unit=unit, layer=self.separate_layer, sequence_id=source_sequence_id)
                store.append(source_record)
                source_written_record_ids.append(source_record.record_id)

            entity_embeddings = _entity_embedding_map_from_unit(unit)
            wrote_any_entity = False
            for entity_text in entity_texts:
                entity_embedding = entity_embeddings.get(entity_text)
                if entity_embedding is None:
                    skipped_entities += 1
                    effects.append(
                        {
                            "unit_id": unit.unit_id,
                            "entity": entity_text,
                            "effect_type": "skipped_entity_missing_embedding",
                            "source_record_id": None if source_record is None else source_record.record_id,
                        }
                    )
                    continue

                matched_record, top1_similarity = self._find_best_match_for_entity(store, entity_embedding)
                if matched_record is not None and top1_similarity is not None and top1_similarity > self.threshold:
                    merged_record = self._merge_record_for_entity(
                        existing_record=matched_record,
                        unit=unit,
                        entity_text=entity_text,
                        entity_embedding=entity_embedding,
                    )
                    if source_record is not None:
                        merged_record.metadata.update(
                            build_extracted_triple_metadata(
                                source_layer=self.separate_layer,
                                target_layer=placement.target_layer,
                                source_record=source_record,
                                triples=list(merged_record.metadata["graph"]["triples"]),
                            )
                        )
                    store.replace_record(self.target_layer, matched_record.record_id, merged_record)
                    written_record_ids.append(matched_record.record_id)
                    written_unit_ids.append(unit.unit_id)
                    wrote_any_entity = True
                    effects.append(
                        {
                            "unit_id": unit.unit_id,
                            "entity": entity_text,
                            "effect_type": "merge",
                            "matched_record_id": matched_record.record_id,
                            "top1_similarity": float(top1_similarity),
                            "threshold": self.threshold,
                            "embedding_source": "entity_representation_embedding",
                            "record_has_embedding": merged_record.embedding is not None,
                            "source_record_id": None if source_record is None else source_record.record_id,
                        }
                    )
                    continue

                sequence_id = store.next_sequence_id()
                record = _record_from_unit_with_text(
                    unit,
                    layer=placement.target_layer,
                    sequence_id=sequence_id,
                    text=entity_text,
                    embedding=entity_embedding,
                )
                record.metadata.update({"graph": graph_metadata_for_unit(unit, layer=placement.target_layer)})
                if source_record is not None:
                    record.metadata.update(
                        build_extracted_triple_metadata(
                            source_layer=self.separate_layer,
                            target_layer=placement.target_layer,
                            source_record=source_record,
                            triples=list(record.metadata["graph"]["triples"]),
                        )
                    )
                store.append(record)
                written_record_ids.append(record.record_id)
                written_unit_ids.append(unit.unit_id)
                wrote_any_entity = True
                effects.append(
                    {
                        "unit_id": unit.unit_id,
                        "entity": entity_text,
                        "effect_type": "append",
                        "record_id": record.record_id,
                        "top1_similarity": None if top1_similarity is None else float(top1_similarity),
                        "threshold": self.threshold,
                        "embedding_source": "entity_representation_embedding",
                        "record_has_embedding": record.embedding is not None,
                        "source_record_id": None if source_record is None else source_record.record_id,
                    }
                )

            if not wrote_any_entity:
                skipped_units += 1

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "threshold": self.threshold,
            "separate": self.separate,
            "separate_layer": self.separate_layer,
            "fanout_mode": "per_entity",
            "writes_embedding_from_record_field": True,
            "records_with_embedding": sum(
                1
                for record_id in written_record_ids
                for record in store.iter_records(self.target_layer)
                if record.record_id == record_id and record.embedding is not None
            ),
            "written_record_ids": written_record_ids,
            "entity_written_record_ids": written_record_ids,
            "source_written_record_ids": source_written_record_ids,
            "written_unit_ids": written_unit_ids,
            "skipped_unit_count": skipped_units,
            "skipped_entity_count": skipped_entities,
            "effects": effects,
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

    def _find_best_match_for_entity(
        self,
        store: MemoryStore,
        entity_embedding: list[float],
    ) -> tuple[MemoryRecord | None, float | None]:
        best_record: MemoryRecord | None = None
        best_similarity: float | None = None
        for candidate in store.iter_records(self.target_layer):
            if candidate.embedding is None or len(candidate.embedding) != len(entity_embedding):
                continue
            similarity = Runtime.cosine_similarity(entity_embedding, candidate.embedding)
            if best_similarity is None or similarity > best_similarity:
                best_record = candidate
                best_similarity = float(similarity)
        return best_record, best_similarity

    def _merge_record_for_entity(
        self,
        *,
        existing_record: MemoryRecord,
        unit,
        entity_text: str,
        entity_embedding: list[float],
    ) -> MemoryRecord:
        projected_unit = replace(
            unit,
            text=entity_text,
            normalized_text=entity_text.casefold().strip(),
            embedding=list(entity_embedding),
            representation_elements=tuple(sorted(set(unit.representation_elements) | {"embedding"})),
        )
        existing_graph = graph_metadata_from_record(existing_record)
        merged_graph = normalize_graph_metadata(
            {
                **existing_graph,
                "entities": list(unit.entities),
                "triples": _merge_graph_triples(list(existing_graph["triples"]), list(unit.triples)),
            },
            layer=existing_record.layer,
        )
        return MemoryRecord(
            record_id=existing_record.record_id,
            unit_id=unit.unit_id,
            layer=existing_record.layer,
            text=entity_text,
            timestamp=unit.timestamp,
            embedding=list(entity_embedding),
            metadata={
                **existing_record.metadata,
                "unit_type": unit.unit_type,
                "representation": _representation_summary_for_graph_merge(projected_unit, existing_record),
                "graph": merged_graph,
            },
        )


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


class HierarchicalOrganization(OrganizationModule):
    """Aggregate selected source-layer records into higher-level target records."""

    spec = ModuleSpec(
        name="hierarchical_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        source_layer: str,
        extract_mode: str,
        extract_fields: tuple[str, ...],
        group_by: tuple[str, ...] = (),
        prompt: PromptPlan | str | None = None,
        target_layer: str | None = None,
        memory_pipeline=None,
    ) -> None:
        config = validate_hierarchical_config(
            source_layer=source_layer,
            target_layer=target_layer,
            memory_pipeline=memory_pipeline,
            extract_mode=extract_mode,
            extract_fields=extract_fields,
            group_by=group_by,
            prompt=prompt,
        )
        self.source_layer = config["source_layer"]
        self.target_layer = config["target_layer"]
        self.memory_pipeline = config["memory_pipeline"]
        self.extract_mode = config["extract_mode"]
        self.extract_fields = config["extract_fields"]
        self.group_by = config["group_by"]
        self.prompt = config["prompt"]

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        require_aligned_units_decisions(packet, include_placements=False)

        placements = build_fixed_placements(
            packet,
            target_layer=self.target_layer,
            memory_pipeline=self.memory_pipeline,
        )
        selected_records, selection_source = resolve_source_records(
            packet,
            store,
            source_layer=self.source_layer,
        )
        grouped = group_records(selected_records, group_by=self.group_by)
        effects, writer_pipeline_mode = append_hierarchical_records(
            store,
            source_layer=self.source_layer,
            target_layer=self.target_layer,
            memory_pipeline=self.memory_pipeline,
            extract_mode=self.extract_mode,
            extract_fields=self.extract_fields,
            group_by=self.group_by,
            grouped_records=grouped,
            prompt=self.prompt,
        )
        effective_target_layer = inferred_target_layer(
            target_layer=self.target_layer,
            memory_pipeline=self.memory_pipeline,
        )

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "source_layer": self.source_layer,
            "target_layer": effective_target_layer,
            "extract_mode": self.extract_mode,
            "extract_fields": list(self.extract_fields),
            "group_by": list(self.group_by),
            "prompt_is_template": bool(
                self.prompt is not None
                and (
                    ensure_prompt_plan(self.prompt, metadata_mode="prompt").mode == "structured"
                    or (
                        isinstance(ensure_prompt_plan(self.prompt, metadata_mode="prompt").template, str)
                        and "{{" in str(ensure_prompt_plan(self.prompt, metadata_mode="prompt").template)
                        and "}}" in str(ensure_prompt_plan(self.prompt, metadata_mode="prompt").template)
                    )
                )
            ),
            "selection_source": selection_source,
            "selected_record_count": len(selected_records),
            "group_count": len(grouped),
            "written_record_ids": [record_id for effect in effects for record_id in effect["written_record_ids"]],
            "append_current_units": False,
            "write_mode": "memory_pipeline_ingest",
            "writer_pipeline_mode": writer_pipeline_mode,
            "sub_ingest_trace": [effect["sub_ingest_trace"] for effect in effects],
            "prompt_trace": [effect["prompt_trace"] for effect in effects if effect.get("prompt_trace") is not None],
        }
        return replace(packet, placements=placements, trace=trace), store


class LLMFunctionCallOrganization(OrganizationModule):
    """Use LLM tool calls to write/update/delete records during organization."""

    spec = ModuleSpec(
        name="llm_function_call_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records", "rewrite_records", "delete_records"),
    )

    def __init__(
        self,
        *,
        prompt: PromptPlan | str,
        tools: list[str | WriteToolSpec],
        target_layer: str | None = None,
        max_turns: int = 6,
        strict_tools: bool = True,
        allow_no_tool_call: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        normalized_prompt = ensure_prompt_plan(prompt, metadata_mode="prompt")
        self.prompt = normalized_prompt
        self.tool_specs = normalize_write_tool_specs(tools, module_name=self.spec.name)
        if write_tool_specs_require_graph_contracts(self.tool_specs):
            self.requires_contracts = frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
            self.produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT})
        else:
            self.requires_contracts = frozenset()
            self.produces_contracts = frozenset()
        self.target_layer = None if target_layer is None else str(target_layer).strip() or None
        self.max_turns = int(max_turns)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        self.strict_tools = bool(strict_tools)
        self.allow_no_tool_call = bool(allow_no_tool_call)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embedding_model = embedding_model

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("LLMFunctionCallOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("LLMFunctionCallOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("LLMFunctionCallOrganization requires decisions aligned with units.")

        placement_layer = self.target_layer or store.topology.layer_names[0]
        placements = [Placement(unit_id=unit.unit_id, target_layer=placement_layer) for unit in packet.units]
        per_unit_trace: list[dict[str, Any]] = []
        aggregate_state = ToolExecutionState()

        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            unit_packet = replace(packet, units=[unit], decisions=[decision], placements=[placement])
            if not decision:
                per_unit_trace.append(
                    {
                        "unit_id": unit.unit_id,
                        "decision": False,
                        "skipped": True,
                        "tool_calls": [],
                        "effects": [],
                    }
                )
                continue
            prompt_text, prompt_trace, call_state, store = self._run_for_unit(unit_packet, store, placement)
            aggregate_state.tool_calls.extend(call_state.tool_calls)
            aggregate_state.effects.extend(call_state.effects)
            for record_id in call_state.written_record_ids:
                if record_id not in aggregate_state.written_record_ids:
                    aggregate_state.written_record_ids.append(record_id)
            for record_id in call_state.updated_record_ids:
                if record_id not in aggregate_state.updated_record_ids:
                    aggregate_state.updated_record_ids.append(record_id)
            for record_id in call_state.deleted_record_ids:
                if record_id not in aggregate_state.deleted_record_ids:
                    aggregate_state.deleted_record_ids.append(record_id)
            placement.target_layer = self._target_layer_for_unit_effects(call_state, fallback=placement.target_layer, store=store)
            per_unit_trace.append(
                {
                    "unit_id": unit.unit_id,
                    "decision": True,
                    "rendered_prompt": prompt_text,
                    **prompt_trace,
                    "tool_calls": list(call_state.tool_calls),
                    "effects": list(call_state.effects),
                }
            )

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "tool_names": [spec.name for spec in self.tool_specs],
            "prompt_is_template": self.prompt.mode == "structured"
            or (isinstance(self.prompt.template, str) and "{{" in self.prompt.template and "}}" in self.prompt.template),
            "per_unit": per_unit_trace,
            "written_record_ids": list(aggregate_state.written_record_ids),
            "updated_record_ids": list(aggregate_state.updated_record_ids),
            "deleted_record_ids": list(aggregate_state.deleted_record_ids),
            "effects": list(aggregate_state.effects),
            "tool_calls": list(aggregate_state.tool_calls),
        }
        return replace(packet, placements=placements, trace=trace), store

    def _run_for_unit(
        self,
        packet: Packet,
        store: MemoryStore,
        placement: Placement,
    ) -> tuple[str, dict[str, Any], ToolExecutionState, MemoryStore]:
        unit = packet.units[0]
        context = WriteToolCallContext(
            packet=packet,
            store=store,
            module_slot="organization",
            default_target_layer=self.target_layer or placement.target_layer,
            visible_records=list(store.iter_records()),
        )
        state = ToolExecutionState()
        runtime_tools = build_runtime_tools(
            self.tool_specs,
            context=context,
            state=state,
            strict_tools=self.strict_tools,
        )
        rendered_prompt, prompt_trace, store = render_prompt_plan(
            ensure_prompt_plan(
                self.prompt,
                metadata_mode="prompt",
                context_builder=lambda current_packet, current_store: {
                    "unit": project_unit_for_template(unit),
                    "tools": project_tool_specs_for_prompt(self.tool_specs),
                    "default_target_layer": self.target_layer or placement.target_layer,
                    "visible_records": [
                        project_record_for_template(record)
                        for record in context.visible_records
                    ],
                },
            ),
            packet=packet,
            store=store,
        )
        context.store = store
        context.visible_records = list(store.iter_records())
        self._run_agent(
            rendered_prompt=rendered_prompt,
            tools=runtime_tools,
            context={"unit_id": unit.unit_id, "slot": self.spec.slot},
        )
        if not state.tool_calls and not self.allow_no_tool_call:
            raise ValueError("LLMFunctionCallOrganization requires at least one successful or attempted tool call.")
        return rendered_prompt, prompt_trace, state, context.store

    def _run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        runtime = Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )
        runtime.require_llm(capability="LLMFunctionCallOrganization")
        return str(
            runtime.run_agent(
                name="MemPrimitiveLLMFunctionCallOrganizationAgent",
                instructions=(
                    "You manage memory writes by calling the provided tools only. "
                    "Use zero or more tool calls to apply the needed write actions. "
                    "If no change is needed, respond with NO_ACTION."
                ),
                input_text=json.dumps(
                    {
                        "prompt": rendered_prompt,
                        "context": context,
                    },
                    ensure_ascii=False,
                ),
                temperature=0.0,
                tools=tools,
                max_turns=self.max_turns,
            )
            or ""
        )

    @staticmethod
    def _target_layer_for_unit_effects(state: ToolExecutionState, *, fallback: str, store: MemoryStore) -> str:
        for effect in state.effects:
            if str(effect.get("action", "")).casefold() != "add":
                continue
            layer = str(effect.get("layer", "")).strip()
            if layer:
                return layer
        return fallback


BASELINE_SLOT: Final[str] = "organization"
BASELINE_CLASSES: Final[tuple[type[OrganizationModule], ...]] = (
    AppendOrganization,
    ConditionalLayerOrganization,
    GraphAppendOrganization,
    GraphEntityAppendOrganization,
    GraphDeduplicationAppendOrganization,
    GraphEntityDeduplicationAppendOrganization,
    PlacementWithoutAppendOrganization,
    HierarchicalOrganization,
    LLMFunctionCallOrganization,
)
