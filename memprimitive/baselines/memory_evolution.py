"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from math import sqrt
from typing import Any, Final

from ..contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_REFLECTION_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
)
from ..core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import MemoryEvolutionModule

from ..utils._graph_family import graph_metadata_from_record, rewrite_graph_record
from ..utils._hierarchical_family import (
    append_hierarchical_records,
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
    aggregate_write_tool_contracts,
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
    render_prompt_plan,
    resolve_records_from_prompt_metadata,
)
from ..utils._reflexion_family import (
    DEFAULT_MEMORY_SIZE,
    DEFAULT_REFLECTION_LAYER,
    ReflectionGenerationPayload,
    ReflectionGenerator,
    ReflectionPromptBuilder,
    feedback_from_payload,
    question_from_payload,
    reflexion_controls,
    runtime_reflection_generator,
    scratchpad_from_payload,
)
from ..utils._trace import copy_trace


def _merge_visible_records(*record_groups: list[MemoryRecord]) -> list[MemoryRecord]:
    merged: list[MemoryRecord] = []
    seen: set[str] = set()
    for records in record_groups:
        for record in records:
            if record.record_id in seen:
                continue
            seen.add(record.record_id)
            merged.append(record)
    return merged


class AppendOnlyEvolution(MemoryEvolutionModule):
    """Run an optional extra evolution pass over already-organized memory.

    ``run`` requires ``packet.units`` and ``packet.placements``. It prefers
    ``packet.decisions`` as the extra-evolution mask. The active mask
    must align with ``units`` and ``placements``. Stage-1 baseline behavior is a
    no-op extra pass: it records which units would participate in extra evolution
    but does not modify the store.
    """

    spec = ModuleSpec(
        name="append_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("AppendOnlyEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("AppendOnlyEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError(
                "AppendOnlyEvolution requires aligned units, decisions, and placements."
            )

        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.decisions, strict=True)
            if decision
        ]

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": active_unit_ids,
            "effects": [],
        }
        return replace(packet, trace=trace), store


class TraceOnlyEvolution(MemoryEvolutionModule):
    """No-op evolution that records explicit effect placeholders for active units.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.decisions`` aligned by index. The store is not mutated.
    """

    spec = ModuleSpec(
        name="trace_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TraceOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("TraceOnlyEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("TraceOnlyEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError("TraceOnlyEvolution requires aligned units, decisions, and placements.")

        effects = [
            {
                "effect_type": "trace_only",
                "unit_id": unit.unit_id,
                "target_layer": placement.target_layer,
            }
            for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True)
            if decision
        ]
        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": [effect["unit_id"] for effect in effects],
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class SummaryRewriteEvolution(MemoryEvolutionModule):
    """Append summary records for evolution-active units into a target layer.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.decisions`` aligned by index. Active units are summarized
    into new append-only records; original records remain unchanged.
    """

    spec = ModuleSpec(
        name="summary_rewrite_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = "default") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("SummaryRewriteEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("SummaryRewriteEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("SummaryRewriteEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError("SummaryRewriteEvolution requires aligned units, decisions, and placements.")

        effects = []
        active_unit_ids = []
        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            if not decision:
                continue
            active_unit_ids.append(unit.unit_id)
            summary_unit = self._summary_unit(unit)
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(summary_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            effects.append(
                {
                    "effect_type": "summary_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "target_layer": self.target_layer,
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
        }
        return replace(packet, trace=trace), store

    @staticmethod
    def _summary_unit(unit: MemoryUnit) -> MemoryUnit:
        representation = unit.metadata.get("representation", {})
        summary_text = (
            representation.get("summary")
            or representation.get("description")
            or unit.description
            or unit.text
        )
        return replace(
            unit,
            text=str(summary_text).strip(),
            unit_type="summary",
            metadata={
                **unit.metadata,
                "evolution_source_unit_id": unit.unit_id,
                "evolution_style": "summary_rewrite",
            },
        )


class LayerMoveEvolution(MemoryEvolutionModule):
    """Copy-append evolution-active units into another layer.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.decisions`` aligned by index. Active units are copied into
    ``target_layer`` as new records without deleting originals.
    """

    spec = ModuleSpec(
        name="layer_move_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = "default") -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("LayerMoveEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("LayerMoveEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("LayerMoveEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError("LayerMoveEvolution requires aligned units, decisions, and placements.")

        effects = []
        active_unit_ids = []
        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            if not decision:
                continue
            active_unit_ids.append(unit.unit_id)
            moved_unit = replace(
                unit,
                metadata={
                    **unit.metadata,
                    "evolution_source_unit_id": unit.unit_id,
                    "move_style": "copy_append",
                },
            )
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(moved_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            effects.append(
                {
                    "effect_type": "layer_move_copy_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "target_layer": self.target_layer,
                    "move_style": "copy_append",
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
        }
        return replace(packet, trace=trace), store


def _latest_record_for_unit(store: MemoryStore, *, layer: str, unit_id: str) -> MemoryRecord | None:
    matches = [record for record in store.iter_records(layer) if record.unit_id == unit_id]
    if not matches:
        return None
    return matches[-1]


def _record_text_tokens(record: MemoryRecord) -> set[str]:
    return {token for token in record.text.casefold().split() if token}


def _graph_neighbor_score(target_record: MemoryRecord, candidate_record: MemoryRecord) -> float:
    target_graph = graph_metadata_from_record(target_record)
    candidate_graph = graph_metadata_from_record(candidate_record)
    target_entities = {entity.casefold() for entity in target_graph["entities"]}
    candidate_entities = {entity.casefold() for entity in candidate_graph["entities"]}
    entity_overlap = len(target_entities & candidate_entities)
    text_overlap = len(_record_text_tokens(target_record) & _record_text_tokens(candidate_record))
    return float((2 * entity_overlap) + text_overlap)


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None or len(left) != len(right) or not left:
        return 0.0
    numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _select_graph_neighbor_candidates(
    target_record: MemoryRecord,
    candidates: list[MemoryRecord],
    *,
    neighbor_limit: int,
    min_score: float,
) -> list[dict[str, Any]]:
    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        structural_score = _graph_neighbor_score(target_record, candidate)
        embedding_score = _cosine_similarity(target_record.embedding, candidate.embedding)
        total_score = structural_score + embedding_score
        if total_score < min_score:
            continue
        scored_candidates.append(
            {
                "record": candidate,
                "record_id": candidate.record_id,
                "unit_id": candidate.unit_id,
                "structural_score": float(structural_score),
                "embedding_score": float(embedding_score),
                "total_score": float(total_score),
            }
        )
    scored_candidates.sort(
        key=lambda item: (-item["total_score"], -item["embedding_score"], item["record"].timestamp, item["record_id"])
    )
    return scored_candidates[:neighbor_limit]


def _neighbor_context_snapshot(target_record: MemoryRecord, neighbor_records: list[MemoryRecord]) -> dict[str, Any]:
    neighbor_entities: list[str] = []
    for neighbor in neighbor_records:
        neighbor_entities.extend(graph_metadata_from_record(neighbor)["entities"])
    return {
        "source_record_id": target_record.record_id,
        "neighbor_record_ids": [record.record_id for record in neighbor_records],
        "neighbor_unit_ids": [record.unit_id for record in neighbor_records],
        "neighbor_entities": list(dict.fromkeys(neighbor_entities)),
        "neighbor_count": len(neighbor_records),
    }


class GraphLinkEvolution(MemoryEvolutionModule):
    """Link evolution for graph records based on same-layer neighbor candidates.

    Constructor: ``target_layer`` must refer to a declared graph layer.
    ``neighbor_limit`` must be positive. ``min_score`` controls the minimum
    combined structural-plus-embedding score required for a candidate neighbor.
    ``rewrite_neighbor_metadata`` enables a conservative metadata rewrite under
    ``metadata["graph"]["neighbor_context"]`` on the evolved target record only.

    ``run`` requires aligned ``packet.units``, ``packet.placements``, and
    ``packet.decisions``. Only units placed into ``target_layer`` are
    processed, and only records in that graph layer are rewritten. This is an
    inferred engineering decomposition of graph-link evolution rather than a
    paper-faithful A-MEM controller.
    """

    spec = ModuleSpec(
        name="graph_link_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )
    requires_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT})

    def __init__(
        self,
        *,
        target_layer: str = "knowledge_graph",
        neighbor_limit: int = 2,
        bidirectional: bool = True,
        min_score: float = 0.1,
        rewrite_neighbor_metadata: bool = False,
    ) -> None:
        if neighbor_limit <= 0:
            raise ValueError("GraphLinkEvolution requires neighbor_limit > 0.")
        self.target_layer = target_layer
        self.neighbor_limit = neighbor_limit
        self.bidirectional = bidirectional
        self.min_score = float(min_score)
        self.rewrite_neighbor_metadata = rewrite_neighbor_metadata

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphLinkEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("GraphLinkEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("GraphLinkEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError("GraphLinkEvolution requires aligned units, decisions, and placements.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(f"GraphLinkEvolution requires target layer {self.target_layer!r} to be Graph.")

        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True):
            if not decision or placement.target_layer != self.target_layer:
                continue
            target_record = _latest_record_for_unit(store, layer=self.target_layer, unit_id=unit.unit_id)
            if target_record is None:
                continue

            active_unit_ids.append(unit.unit_id)
            candidates = [
                record
                for record in store.iter_records(self.target_layer)
                if record.record_id != target_record.record_id
            ]
            candidate_details = _select_graph_neighbor_candidates(
                target_record,
                candidates,
                neighbor_limit=self.neighbor_limit,
                min_score=self.min_score,
            )
            linked_record_ids = [detail["record_id"] for detail in candidate_details]

            effect = {
                "effect_type": "graph_link_evolution",
                "unit_id": unit.unit_id,
                "record_id": target_record.record_id,
                "target_layer": self.target_layer,
                "candidate_count": len(candidate_details),
                "candidate_record_ids": linked_record_ids,
                "candidate_scores": [
                    {
                        "record_id": detail["record_id"],
                        "structural_score": detail["structural_score"],
                        "embedding_score": detail["embedding_score"],
                        "total_score": detail["total_score"],
                    }
                    for detail in candidate_details
                ],
                "linked_record_ids": linked_record_ids,
                "bidirectional": self.bidirectional,
                "rewrite_neighbor_metadata": self.rewrite_neighbor_metadata,
            }

            if linked_record_ids:
                store.add_graph_links(self.target_layer, target_record.record_id, linked_record_ids)
                refreshed_target = next(
                    record
                    for record in store.iter_records(self.target_layer)
                    if record.record_id == target_record.record_id
                )
                extra_graph_fields = None
                if self.rewrite_neighbor_metadata:
                    extra_graph_fields = {
                        "neighbor_context": _neighbor_context_snapshot(
                            refreshed_target,
                            [detail["record"] for detail in candidate_details],
                        )
                    }
                store.replace_record(
                    self.target_layer,
                    refreshed_target.record_id,
                    rewrite_graph_record(
                        refreshed_target,
                        linked_record_ids=linked_record_ids,
                        link_trace_entry=effect,
                        extra_graph_fields=extra_graph_fields,
                    ),
                )

                if self.bidirectional:
                    for detail in candidate_details:
                        neighbor_record = detail["record"]
                        store.add_graph_links(self.target_layer, neighbor_record.record_id, [target_record.record_id])
                        refreshed_neighbor = next(
                            record
                            for record in store.iter_records(self.target_layer)
                            if record.record_id == neighbor_record.record_id
                        )
                        store.replace_record(
                            self.target_layer,
                            refreshed_neighbor.record_id,
                            rewrite_graph_record(
                                refreshed_neighbor,
                                linked_record_ids=[target_record.record_id],
                                link_trace_entry={
                                    "effect_type": "graph_link_backlink",
                                    "record_id": neighbor_record.record_id,
                                    "linked_record_ids": [target_record.record_id],
                                    "source_record_id": target_record.record_id,
                                    "target_layer": self.target_layer,
                                },
                            ),
                        )

            effects.append(effect)

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "target_layer": self.target_layer,
        }
        return replace(packet, trace=trace), store


class GraphNeighborAppendEvolution(GraphLinkEvolution):
    """Backward-compatible graph link append baseline built on ``GraphLinkEvolution``.

    Constructor: same as the earlier baseline variant. It preserves the old
    class name and trace module id while delegating the actual graph-dependent
    candidate selection and safe rewrite logic to ``GraphLinkEvolution``.
    """

    spec = ModuleSpec(
        name="graph_neighbor_append_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )

    def __init__(self, *, target_layer: str = "knowledge_graph", neighbor_limit: int = 2, bidirectional: bool = True) -> None:
        super().__init__(
            target_layer=target_layer,
            neighbor_limit=neighbor_limit,
            bidirectional=bidirectional,
            min_score=0.1,
            rewrite_neighbor_metadata=False,
        )


class GraphNeighborContextTraceEvolution(MemoryEvolutionModule):
    """Trace linked-neighbor context and optionally write a conservative summary.

    Constructor: ``target_layer`` must refer to a graph layer.
    ``rewrite_metadata`` controls whether the target record gets a minimal
    ``graph.neighbor_context`` snapshot derived from its currently linked
    neighbors. This keeps updates namespaced and conservative instead of
    rewriting arbitrary record metadata.

    ``run`` requires aligned ``packet.units``, ``packet.placements``, and
    ``packet.decisions``. Only records in ``target_layer`` are read or
    rewritten, making this a simplified rule-based stand-in for richer
    neighbor-context evolution.
    """

    spec = ModuleSpec(
        name="graph_neighbor_context_trace_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )
    requires_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT})
    produces_contracts = frozenset({RECORD_GRAPH_LINKS_CONTRACT})

    def __init__(self, *, target_layer: str = "knowledge_graph", rewrite_metadata: bool = False) -> None:
        self.target_layer = target_layer
        self.rewrite_metadata = rewrite_metadata

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphNeighborContextTraceEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("GraphNeighborContextTraceEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("GraphNeighborContextTraceEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError(
                "GraphNeighborContextTraceEvolution requires aligned units, decisions, and placements."
            )
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(
                f"GraphNeighborContextTraceEvolution requires target layer {self.target_layer!r} to be Graph."
            )

        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True):
            if not decision or placement.target_layer != self.target_layer:
                continue
            target_record = _latest_record_for_unit(store, layer=self.target_layer, unit_id=unit.unit_id)
            if target_record is None:
                continue

            neighbor_records = store.iter_graph_neighbors(self.target_layer, target_record.record_id)
            snapshot = _neighbor_context_snapshot(target_record, neighbor_records)
            active_unit_ids.append(unit.unit_id)
            effect = {
                "effect_type": "graph_neighbor_context_trace",
                "unit_id": unit.unit_id,
                "record_id": target_record.record_id,
                "target_layer": self.target_layer,
                "neighbor_record_ids": snapshot["neighbor_record_ids"],
                "neighbor_unit_ids": snapshot["neighbor_unit_ids"],
                "neighbor_entities": snapshot["neighbor_entities"],
                "rewrite_metadata": self.rewrite_metadata,
            }
            effects.append(effect)

            if self.rewrite_metadata:
                refreshed_target = next(
                    record
                    for record in store.iter_records(self.target_layer)
                    if record.record_id == target_record.record_id
                )
                store.replace_record(
                    self.target_layer,
                    refreshed_target.record_id,
                    rewrite_graph_record(
                        refreshed_target,
                        extra_graph_fields={"neighbor_context": snapshot},
                    ),
                )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "target_layer": self.target_layer,
        }
        return replace(packet, trace=trace), store


class ReflectionGenerationEvolution(MemoryEvolutionModule):
    """Generate strategy notes from failed trials and append them to a memory layer.

    Constructor: ``target_layer`` selects where generated reflections are stored.
    ``memory_size`` is the retained sliding-window size and must be positive.
    ``reflection_generator`` may override generation for testing or custom
    controllers. ``prompt_builder`` customizes only the benchmark/prompt
    residual while preserving the generic evolution skeleton.

    ``run`` requires aligned ``packet.units``, ``packet.placements``,
    ``packet.decisions``, and ``packet.observation``. The store is
    mutated by appending generated records and pruning the target layer to the
    configured window.
    """

    spec = ModuleSpec(
        name="reflection_generation_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions", "observation"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )
    produces_contracts = frozenset({RECORD_REFLECTION_PAYLOAD_CONTRACT})

    def __init__(
        self,
        *,
        target_layer: str = DEFAULT_REFLECTION_LAYER,
        memory_size: int = DEFAULT_MEMORY_SIZE,
        window_size: int | None = None,
        reflection_generator: ReflectionGenerator | None = None,
        prompt_builder: ReflectionPromptBuilder | None = None,
    ) -> None:
        effective_size = memory_size if window_size is None else window_size
        if effective_size <= 0:
            raise ValueError("ReflectionGenerationEvolution requires memory_size > 0.")
        self.target_layer = target_layer
        self.memory_size = effective_size
        self.reflection_generator = reflection_generator
        self.prompt_builder = prompt_builder

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("ReflectionGenerationEvolution requires packet.observation.")
        if packet.units is None:
            raise ValueError("ReflectionGenerationEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("ReflectionGenerationEvolution requires packet.placements.")
        if packet.decisions is None:
            raise ValueError("ReflectionGenerationEvolution requires packet.decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.decisions)):
            raise ValueError(
                "ReflectionGenerationEvolution requires aligned units, placements, and decisions."
            )

        if not store.has_layer(self.target_layer):
            store.ensure_layer(self.target_layer, allow_create=True, theme="semantic")

        question = question_from_payload(packet.observation.metadata)
        scratchpad = scratchpad_from_payload(packet.observation.metadata)
        evaluator_feedback = feedback_from_payload(packet.observation.metadata)
        prior_reflections = tuple(record.text for record in store.iter_records(self.target_layer))
        generator = self.reflection_generator or (
            lambda payload: runtime_reflection_generator(payload, prompt_builder=self.prompt_builder)
        )
        generation_mode = "callable_override" if self.reflection_generator is not None else "runtime"

        active_unit_ids: list[str] = []
        record_ids: list[str] = []
        effects: list[dict[str, Any]] = []
        trial_index = reflexion_controls(packet.observation.metadata).get("trial_index", 0)

        for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True):
            if not decision:
                continue
            active_unit_ids.append(unit.unit_id)
            payload = ReflectionGenerationPayload(
                question=question,
                scratchpad=scratchpad,
                evaluator_feedback=evaluator_feedback,
                prior_reflections=prior_reflections,
                observation_metadata=packet.observation.metadata,
                unit_metadata=unit.metadata,
            )
            reflection_text = generator(payload).strip()
            reflection_unit = MemoryUnit(
                text=reflection_text,
                unit_type="reflection",
                metadata={
                    **unit.metadata,
                    "reflection": {
                        "question": question,
                        "trial_index": trial_index,
                        "evaluator_feedback": evaluator_feedback,
                        "source_unit_id": unit.unit_id,
                        "source_layer": placement.target_layer,
                        "target_layer": self.target_layer,
                        "memory_size": self.memory_size,
                        "last_attempt": scratchpad,
                        "generation_mode": generation_mode,
                        "inferred_decomposition": True,
                    },
                    "reflexion": {
                        "triggered": True,
                        "question": question,
                        "trial_index": trial_index,
                        "evaluator_feedback": evaluator_feedback,
                        "source_unit_id": unit.unit_id,
                        "source_layer": placement.target_layer,
                        "target_layer": self.target_layer,
                        "memory_size": self.memory_size,
                        "last_attempt": scratchpad,
                    },
                },
            )
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(reflection_unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            prior_reflections = (*prior_reflections, record.text)
            record_ids.append(record.record_id)
            effects.append(
                {
                    "effect_type": "reflection_append",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "question": question,
                    "target_layer": self.target_layer,
                }
            )

        removed_record_ids: list[str] = []
        records = store.layers.get(self.target_layer, [])
        if len(records) > self.memory_size:
            removed = records[:-self.memory_size]
            removed_record_ids = [record.record_id for record in removed]
            store.layers[self.target_layer] = records[-self.memory_size:]

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "record_ids": record_ids,
            "target_layer": self.target_layer,
            "memory_size": self.memory_size,
            "trial_trace": scratchpad,
            "question": question,
            "generation_mode": generation_mode,
            "residual_boundary": {
                "skeleton": "generic reflection generation evolution",
                "prompt_residual": "classic runtime prompt builder",
            },
            "pruned_record_ids": removed_record_ids,
            "retained_record_ids": [record.record_id for record in store.layers[self.target_layer]],
        }
        return replace(packet, trace=trace), store


class HierarchicalEvolution(MemoryEvolutionModule):
    """Aggregate selected source-layer records into higher-level target records.

    Supports both the original layer-scan / decisions-store selection path and
    an opt-in ``latest_active_units`` mode for patterns that first append raw
    trial records and then immediately abstract only those newly written
    records. Optional ``record_text_field`` lets generated payloads keep richer
    metadata while choosing a single field as the stored record text. Optional
    ``retention_size`` prunes the target layer to a bounded recency window.
    """

    spec = ModuleSpec(
        name="hierarchical_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
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
        selection_mode: str = "decisions_store_or_scan",
        record_text_field: str | None = None,
        retention_size: int | None = None,
    ) -> None:
        config = validate_hierarchical_config(
            source_layer=source_layer,
            target_layer=target_layer,
            memory_pipeline=memory_pipeline,
            extract_mode=extract_mode,
            extract_fields=extract_fields,
            group_by=group_by,
            prompt=prompt,
            selection_mode=selection_mode,
            record_text_field=record_text_field,
        )
        self.source_layer = config["source_layer"]
        self.target_layer = config["target_layer"]
        self.memory_pipeline = config["memory_pipeline"]
        self.extract_mode = config["extract_mode"]
        self.extract_fields = config["extract_fields"]
        self.group_by = config["group_by"]
        self.prompt = config["prompt"]
        self.selection_mode = config["selection_mode"]
        self.record_text_field = config["record_text_field"]
        if retention_size is not None and int(retention_size) <= 0:
            raise ValueError("retention_size must be positive when provided.")
        self.retention_size = None if retention_size is None else int(retention_size)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        require_aligned_units_decisions(packet, include_placements=True)

        selected_records, selection_source = resolve_source_records(
            packet,
            store,
            source_layer=self.source_layer,
            selection_mode=self.selection_mode,
        )
        grouped = group_records(selected_records, group_by=self.group_by)
        effects, writer_pipeline_mode = append_hierarchical_records(
            store,
            source_layer=self.source_layer,
            target_layer=self.target_layer,
            memory_pipeline=self.memory_pipeline,
            extract_mode=self.extract_mode,
            extract_fields=self.extract_fields,
            record_text_field=self.record_text_field,
            group_by=self.group_by,
            grouped_records=grouped,
            prompt=self.prompt,
        )
        effective_target_layer = inferred_target_layer(
            target_layer=self.target_layer,
            memory_pipeline=self.memory_pipeline,
        )
        pruned_record_ids: list[str] = []
        retained_record_ids: list[str] = []
        if self.retention_size is not None and store.has_layer(effective_target_layer):
            records = store.layers.get(effective_target_layer, [])
            if len(records) > self.retention_size:
                removed = records[:-self.retention_size]
                pruned_record_ids = [record.record_id for record in removed]
                store.layers[effective_target_layer] = records[-self.retention_size:]
            retained_record_ids = [record.record_id for record in store.layers.get(effective_target_layer, [])]

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": selection_source,
            "source_layer": self.source_layer,
            "target_layer": effective_target_layer,
            "extract_mode": self.extract_mode,
            "extract_fields": list(self.extract_fields),
            "selection_mode": self.selection_mode,
            "record_text_field": self.record_text_field,
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
            "selected_record_count": len(selected_records),
            "group_count": len(grouped),
            "active_group_keys": [effect["group_key"] for effect in effects],
            "effects": effects,
            "write_mode": "memory_pipeline_ingest",
            "writer_pipeline_mode": writer_pipeline_mode,
            "prompt_trace": [effect["prompt_trace"] for effect in effects if effect.get("prompt_trace") is not None],
            "retention_size": self.retention_size,
            "pruned_record_ids": pruned_record_ids,
            "retained_record_ids": retained_record_ids,
        }
        return replace(packet, trace=trace), store


class LLMFunctionCallEvolution(MemoryEvolutionModule):
    """Use LLM tool calls to mutate existing store records during evolution."""

    spec = ModuleSpec(
        name="llm_function_call_evolution",
        slot="memory_evolution",
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records", "rewrite_records", "delete_records"),
    )

    def __init__(
        self,
        *,
        prompt: PromptPlan | str,
        tools: list[str | WriteToolSpec],
        source_layer: str | None = None,
        target_layer: str | None = None,
        max_turns: int = 6,
        strict_tools: bool = True,
        allow_no_tool_call: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.prompt = ensure_prompt_plan(prompt, metadata_mode="prompt")
        self.tool_specs = normalize_write_tool_specs(tools, module_name=self.spec.name)
        required_contracts, produced_contracts = aggregate_write_tool_contracts(self.tool_specs)
        if write_tool_specs_require_graph_contracts(self.tool_specs):
            required_contracts = frozenset(set(required_contracts) | {TOPOLOGY_GRAPH_LAYER_CONTRACT})
            produced_contracts = frozenset(set(produced_contracts) | {RECORD_GRAPH_LINKS_CONTRACT})
        self.requires_contracts = required_contracts
        self.produces_contracts = produced_contracts
        self.source_layer = None if source_layer is None else str(source_layer).strip() or None
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
        selected_records, decision_source = self._select_records(packet, store)
        visible_records_holder = {"records": list(selected_records)}
        context = WriteToolCallContext(
            packet=packet,
            store=store,
            module_slot="memory_evolution",
            default_target_layer=self.target_layer,
            selected_records=list(selected_records),
            visible_records=list(visible_records_holder["records"]),
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
                    "selected_records": [project_record_for_template(record) for record in selected_records],
                    "visible_records": [project_record_for_template(record) for record in visible_records_holder["records"]],
                    "source_layer": self.source_layer or "",
                    "tools": project_tool_specs_for_prompt(self.tool_specs),
                    "default_target_layer": self.target_layer,
                },
            ),
            packet=packet,
            store=store,
            post_recall_context_builder=lambda prompt_context, prompt_metadata, current_store: (
                self._apply_prompt_visible_records(
                    prompt_metadata=prompt_metadata,
                    store=current_store,
                    selected_records=selected_records,
                    visible_records_holder=visible_records_holder,
                )
            ),
        )
        context.store = store
        context.visible_records = list(visible_records_holder["records"])
        visible_record_ids = [record.record_id for record in visible_records_holder["records"]]
        self._run_agent(
            rendered_prompt=rendered_prompt,
            tools=runtime_tools,
            context={
                "slot": self.spec.slot,
                "selected_record_ids": [record.record_id for record in selected_records],
            },
        )
        if not state.tool_calls and not self.allow_no_tool_call:
            raise ValueError("LLMFunctionCallEvolution requires at least one successful or attempted tool call.")

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "tool_names": [spec.name for spec in self.tool_specs],
            "prompt_is_template": self.prompt.mode == "structured"
            or (isinstance(self.prompt.template, str) and "{{" in self.prompt.template and "}}" in self.prompt.template),
            "decision_source": decision_source,
            "source_layer": self.source_layer,
            "selected_record_ids": [record.record_id for record in selected_records],
            "visible_record_ids": visible_record_ids,
            "visible_record_source": (
                "selected_records_plus_prompt_recall" if prompt_trace.get("visible_record_ids_by_label") else "selected_records"
            ),
            "written_record_ids": list(state.written_record_ids),
            "updated_record_ids": list(state.updated_record_ids),
            "deleted_record_ids": list(state.deleted_record_ids),
            "effects": list(state.effects),
            "tool_calls": list(state.tool_calls),
            "prompt_trace": prompt_trace,
        }
        return replace(packet, trace=trace), context.store

    def _select_records(self, packet: Packet, store: MemoryStore) -> tuple[list[MemoryRecord], str]:
        if packet.decisions_store is not None:
            if self.source_layer is not None:
                return resolve_source_records(packet, store, source_layer=self.source_layer)
            selected: list[MemoryRecord] = []
            seen: set[str] = set()
            for layer_name, entry in packet.decisions_store.items():
                if not isinstance(entry, dict):
                    continue
                record_ids = [str(value).strip() for value in entry.get("record_ids", []) if str(value).strip()]
                if not record_ids:
                    continue
                by_id = {record.record_id: record for record in store.iter_records(layer_name)}
                for record_id in record_ids:
                    if record_id in by_id and record_id not in seen:
                        selected.append(by_id[record_id])
                        seen.add(record_id)
            return selected, "decisions_store"
        if packet.units is not None or packet.placements is not None or packet.decisions is not None:
            if packet.units is None:
                raise ValueError("LLMFunctionCallEvolution requires packet.units when using decisions.")
            if packet.placements is None:
                raise ValueError("LLMFunctionCallEvolution requires packet.placements when using decisions.")
            if packet.decisions is None:
                raise ValueError("LLMFunctionCallEvolution requires packet.decisions when using decisions.")
            if not (len(packet.units) == len(packet.placements) == len(packet.decisions)):
                raise ValueError("LLMFunctionCallEvolution requires aligned units, placements, and decisions.")
            selected = []
            seen: set[str] = set()
            for unit, placement, decision in zip(packet.units, packet.placements, packet.decisions, strict=True):
                if not decision:
                    continue
                layer = self.source_layer or placement.target_layer
                if not str(layer).strip():
                    continue
                record = _latest_record_for_unit(store, layer=layer, unit_id=unit.unit_id)
                if record is None or record.record_id in seen:
                    continue
                selected.append(record)
                seen.add(record.record_id)
            return selected, "decisions"
        if self.source_layer is not None:
            return store.iter_records(self.source_layer), "source_layer_scan"
        return store.iter_records(), "store_scan"

    @staticmethod
    def _apply_prompt_visible_records(
        *,
        prompt_metadata: dict[str, Any],
        store: MemoryStore,
        selected_records: list[MemoryRecord],
        visible_records_holder: dict[str, list[MemoryRecord]],
    ) -> dict[str, Any]:
        prompt_visible_records = resolve_records_from_prompt_metadata(store, prompt_metadata)
        merged_records = _merge_visible_records(selected_records, prompt_visible_records)
        visible_records_holder["records"] = merged_records
        return {
            "visible_records": [project_record_for_template(record) for record in merged_records],
        }

    def _run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        runtime = Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )
        runtime.require_llm(capability="LLMFunctionCallEvolution")
        return str(
            runtime.run_agent(
                name="MemPrimitiveLLMFunctionCallEvolutionAgent",
                instructions=(
                    "You manage memory evolution by calling the provided tools only. "
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


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (
    AppendOnlyEvolution,
    TraceOnlyEvolution,
    SummaryRewriteEvolution,
    LayerMoveEvolution,
    GraphLinkEvolution,
    GraphNeighborContextTraceEvolution,
    GraphNeighborAppendEvolution,
    ReflectionGenerationEvolution,
    HierarchicalEvolution,
    LLMFunctionCallEvolution,
)
