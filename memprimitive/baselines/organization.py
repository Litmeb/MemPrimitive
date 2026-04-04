"""Baseline: organization primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Final

from ..contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_NOTE_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
    TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
    UNIT_NOTE_PAYLOAD_CONTRACT,
)
from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet, Placement
from ..interfaces import OrganizationModule

from ..utils._graph_family import graph_metadata_for_unit
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
from ..utils._runtime import Runtime
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
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT})

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
    requires_contracts = frozenset({UNIT_NOTE_PAYLOAD_CONTRACT, TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, RECORD_NOTE_PAYLOAD_CONTRACT})

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
    PlacementWithoutAppendOrganization,
    GraphAppendLinkReadyOrganization,
    HierarchicalOrganization,
    LLMFunctionCallOrganization,
)
