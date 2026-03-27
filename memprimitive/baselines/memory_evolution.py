"""Baseline: memory evolution primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from math import sqrt
from typing import Any, Final

from ..core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Packet
from ..interfaces import MemoryEvolutionModule

from ._amem_family import (
    coerce_index_list,
    coerce_llm_mapping,
    DEFAULT_CATEGORY,
    DEFAULT_EMBEDDING_VERSION,
    DEFAULT_NOTE_NAMESPACE,
    note_payload_from_record,
    repair_note_payload,
    retrieve_candidates_by_embedding,
    rewrite_record_from_note_payload,
    stringify_note_candidates,
)
from ._graph_family import graph_metadata_from_record, rewrite_graph_record
from ._reflexion_family import (
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
from ._trace import copy_trace


class AppendOnlyEvolution(MemoryEvolutionModule):
    """Run an optional extra evolution pass over already-organized memory.

    ``run`` requires ``packet.units`` and ``packet.placements``. It prefers
    ``packet.evolution_decisions`` as the extra-evolution mask. The active mask
    must align with ``units`` and ``placements``. Stage-1 baseline behavior is a
    no-op extra pass: it records which units would participate in extra evolution
    but does not modify the store.
    """

    spec = ModuleSpec(
        name="append_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AppendOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("AppendOnlyEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("AppendOnlyEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError(
                "AppendOnlyEvolution requires aligned units, evolution decisions, and placements."
            )

        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": [],
        }
        return replace(packet, trace=trace), store


class TraceOnlyEvolution(MemoryEvolutionModule):
    """No-op evolution that records explicit effect placeholders for active units.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions`` aligned by index. The store is not mutated.
    """

    spec = ModuleSpec(
        name="trace_only_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TraceOnlyEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("TraceOnlyEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("TraceOnlyEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("TraceOnlyEvolution requires aligned units, evolution decisions, and placements.")

        effects = [
            {
                "effect_type": "trace_only",
                "unit_id": unit.unit_id,
                "target_layer": placement.target_layer,
            }
            for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True)
            if decision
        ]
        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": [effect["unit_id"] for effect in effects],
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class SummaryRewriteEvolution(MemoryEvolutionModule):
    """Append summary records for evolution-active units into a target layer.

    ``run`` requires ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions`` aligned by index. Active units are summarized
    into new append-only records; original records remain unchanged.
    """

    spec = ModuleSpec(
        name="summary_rewrite_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
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
        if packet.evolution_decisions is None:
            raise ValueError("SummaryRewriteEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("SummaryRewriteEvolution requires aligned units, evolution decisions, and placements.")

        effects = []
        active_unit_ids = []
        for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True):
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
            "decision_source": "evolution_decisions",
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
    ``packet.evolution_decisions`` aligned by index. Active units are copied into
    ``target_layer`` as new records without deleting originals.
    """

    spec = ModuleSpec(
        name="layer_move_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
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
        if packet.evolution_decisions is None:
            raise ValueError("LayerMoveEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("LayerMoveEvolution requires aligned units, evolution decisions, and placements.")

        effects = []
        active_unit_ids = []
        for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True):
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
            "decision_source": "evolution_decisions",
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
    ``packet.evolution_decisions``. Only units placed into ``target_layer`` are
    processed, and only records in that graph layer are rewritten. This is an
    inferred engineering decomposition of graph-link evolution rather than a
    paper-faithful A-MEM controller.
    """

    spec = ModuleSpec(
        name="graph_link_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )

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
        if packet.evolution_decisions is None:
            raise ValueError("GraphLinkEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError("GraphLinkEvolution requires aligned units, evolution decisions, and placements.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(f"GraphLinkEvolution requires target layer {self.target_layer!r} to be Graph.")

        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True):
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
            "decision_source": "evolution_decisions",
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
        input_requirements=("units", "placements", "evolution_decisions"),
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
    ``packet.evolution_decisions``. Only records in ``target_layer`` are read or
    rewritten, making this a simplified rule-based stand-in for richer
    neighbor-context evolution.
    """

    spec = ModuleSpec(
        name="graph_neighbor_context_trace_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )

    def __init__(self, *, target_layer: str = "knowledge_graph", rewrite_metadata: bool = False) -> None:
        self.target_layer = target_layer
        self.rewrite_metadata = rewrite_metadata

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GraphNeighborContextTraceEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("GraphNeighborContextTraceEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("GraphNeighborContextTraceEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.evolution_decisions) == len(packet.placements)):
            raise ValueError(
                "GraphNeighborContextTraceEvolution requires aligned units, evolution decisions, and placements."
            )
        if store.layer_shape(self.target_layer) != "Graph":
            raise ValueError(
                f"GraphNeighborContextTraceEvolution requires target layer {self.target_layer!r} to be Graph."
            )

        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True):
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
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "target_layer": self.target_layer,
        }
        return replace(packet, trace=trace), store


class LinkStrengtheningEvolution(MemoryEvolutionModule):
    """Strengthen graph links for enriched note records using LLM-selected neighbors.

    Constructor: ``target_layer`` must be a graph layer with a vector index.
    ``candidate_k`` and ``max_links_per_record`` must be positive. The module
    reads note payloads from ``note_namespace`` and rewrites only the current
    target record plus its graph namespace.

    ``run`` requires aligned ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions``. The store is mutated by rewriting the target
    record and appending graph links for the selected neighbor set.
    """

    spec = ModuleSpec(
        name="link_strengthening_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "index:vector", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph", "target_layer_index:vector"),
        side_effects=("modify_store", "rewrite_records"),
    )

    def __init__(
        self,
        *,
        target_layer: str = "knowledge_graph",
        candidate_k: int = 5,
        max_links_per_record: int = 4,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        default_category: str = DEFAULT_CATEGORY,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        strict_llm: bool = True,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("LinkStrengtheningEvolution requires candidate_k > 0.")
        if max_links_per_record <= 0:
            raise ValueError("LinkStrengtheningEvolution requires max_links_per_record > 0.")
        if not strict_llm:
            raise ValueError("LinkStrengtheningEvolution requires strict_llm=True.")
        self.target_layer = target_layer
        self.candidate_k = candidate_k
        self.max_links_per_record = max_links_per_record
        self.note_namespace = note_namespace
        self.default_category = default_category
        self.embedding_version = embedding_version
        self.strict_llm = strict_llm

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("LinkStrengtheningEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("LinkStrengtheningEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("LinkStrengtheningEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("LinkStrengtheningEvolution requires aligned units, placements, and evolution decisions.")

        from ..classic_modules._runtime import get_classic_runtime

        runtime = get_classic_runtime()
        runtime.require_llm(capability="LinkStrengtheningEvolution")
        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, placement, decision in zip(packet.units, packet.placements, packet.evolution_decisions, strict=True):
            if not decision or placement.target_layer != self.target_layer:
                continue
            current_record = _latest_record_for_unit(store, layer=self.target_layer, unit_id=unit.unit_id)
            if current_record is None or current_record.embedding is None:
                continue

            neighbor_candidates = [
                record
                for _, record in retrieve_candidates_by_embedding(
                    store=store,
                    layer=self.target_layer,
                    query_embedding=current_record.embedding,
                    top_k=self.candidate_k + 1,
                )
                if record.record_id != current_record.record_id
            ][: self.candidate_k]
            if not neighbor_candidates:
                continue

            current_payload = note_payload_from_record(
                current_record,
                note_namespace=self.note_namespace,
                default_category=self.default_category,
            )
            raw = runtime.json(
                system=(
                    "Given a new note and its nearest neighbors, choose which neighbors should receive "
                    "a strengthened graph link and optionally return updated tags for the current note. "
                    "Return JSON with fields: connections, tags."
                ),
                user=json.dumps(
                    {
                        "content": current_payload["content"],
                        "keywords": current_payload["keywords"],
                        "nearest_neighbors_memories": stringify_note_candidates(
                            neighbor_candidates,
                            note_namespace=self.note_namespace,
                        ),
                    },
                    ensure_ascii=False,
                ),
            )
            raw_mapping = coerce_llm_mapping(raw, list_key="connections")
            connection_indices = coerce_index_list(raw_mapping.get("connections", []))
            strengthened_neighbors = [
                neighbor_candidates[index]
                for index in connection_indices
                if isinstance(index, int) and 0 <= index < len(neighbor_candidates)
            ][: self.max_links_per_record]
            strengthened_links = [record.record_id for record in strengthened_neighbors]
            merged_links = list(dict.fromkeys(current_record.metadata.get("graph", {}).get("links", []) + strengthened_links))
            updated_payload = {
                **current_payload,
                "tags": repair_note_payload(
                    {"tags": raw_mapping.get("tags"), "content": current_payload["content"]},
                    fallback_content=current_payload["content"],
                    default_category=self.default_category,
                )["tags"]
                or current_payload["tags"],
            }
            rewritten = rewrite_record_from_note_payload(
                store,
                layer=self.target_layer,
                record=current_record,
                payload=updated_payload,
                note_namespace=self.note_namespace,
                default_category=self.default_category,
                embedding_version=self.embedding_version,
                runtime=runtime,
            )
            current_record = replace(
                rewritten,
                metadata={
                    **rewritten.metadata,
                    "graph": {
                        **(rewritten.metadata.get("graph", {}) if isinstance(rewritten.metadata.get("graph"), dict) else {}),
                        "links": merged_links,
                        "link_count": len(merged_links),
                    },
                },
            )
            store.replace_record(self.target_layer, current_record.record_id, current_record)
            active_unit_ids.append(unit.unit_id)
            effects.append(
                {
                    "effect_type": "link_strengthening",
                    "unit_id": unit.unit_id,
                    "record_id": current_record.record_id,
                    "target_layer": self.target_layer,
                    "strengthened_links": strengthened_links,
                    "rewritten_record_ids": [current_record.record_id],
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "target_layer": self.target_layer,
            "note_namespace": self.note_namespace,
        }
        return replace(packet, trace=trace), store


class NeighborContextUpdateEvolution(MemoryEvolutionModule):
    """Rewrite linked-neighbor note context/tags from the current note's perspective.

    Constructor: ``target_layer`` must be a graph layer. The module reads note
    payloads from ``note_namespace`` and updates already linked neighbors when
    available; when no links exist, it falls back to nearest-neighbor candidates.

    ``run`` requires aligned ``packet.units``, ``packet.placements``, and
    ``packet.evolution_decisions``. The store is mutated only by rewriting
    neighbor records under the configured note namespace.
    """

    spec = ModuleSpec(
        name="neighbor_context_update_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        store_requirements=("index:graph", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph"),
        side_effects=("modify_store", "rewrite_records"),
    )

    def __init__(
        self,
        *,
        target_layer: str = "knowledge_graph",
        candidate_k: int = 5,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        default_category: str = DEFAULT_CATEGORY,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        strict_llm: bool = True,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("NeighborContextUpdateEvolution requires candidate_k > 0.")
        if not strict_llm:
            raise ValueError("NeighborContextUpdateEvolution requires strict_llm=True.")
        self.target_layer = target_layer
        self.candidate_k = candidate_k
        self.note_namespace = note_namespace
        self.default_category = default_category
        self.embedding_version = embedding_version
        self.strict_llm = strict_llm

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("NeighborContextUpdateEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("NeighborContextUpdateEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("NeighborContextUpdateEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("NeighborContextUpdateEvolution requires aligned units, placements, and evolution decisions.")

        from ..classic_modules._runtime import get_classic_runtime

        runtime = get_classic_runtime()
        runtime.require_llm(capability="NeighborContextUpdateEvolution")
        effects: list[dict[str, Any]] = []
        active_unit_ids: list[str] = []

        for unit, placement, decision in zip(packet.units, packet.placements, packet.evolution_decisions, strict=True):
            if not decision or placement.target_layer != self.target_layer:
                continue
            current_record = _latest_record_for_unit(store, layer=self.target_layer, unit_id=unit.unit_id)
            if current_record is None:
                continue

            linked_neighbors = store.iter_graph_neighbors(self.target_layer, current_record.record_id)
            if not linked_neighbors and current_record.embedding is not None:
                linked_neighbors = [
                    record
                    for _, record in retrieve_candidates_by_embedding(
                        store=store,
                        layer=self.target_layer,
                        query_embedding=current_record.embedding,
                        top_k=self.candidate_k + 1,
                    )
                    if record.record_id != current_record.record_id
                ][: self.candidate_k]
            if not linked_neighbors:
                continue

            current_payload = note_payload_from_record(
                current_record,
                note_namespace=self.note_namespace,
                default_category=self.default_category,
            )
            raw = runtime.json(
                system=(
                    "Update each neighbor note's context and tags based on the current note and its linked neighbors. "
                    "Return JSON with an updates list. Each update may contain context and tags."
                ),
                user=json.dumps(
                    {
                        "content": current_payload["content"],
                        "context": current_payload["context"],
                        "nearest_neighbors_memories": stringify_note_candidates(
                            linked_neighbors,
                            note_namespace=self.note_namespace,
                        ),
                        "neighbor_count": len(linked_neighbors),
                    },
                    ensure_ascii=False,
                ),
            )
            raw_mapping = coerce_llm_mapping(raw, list_key="updates")
            updates = raw_mapping.get("updates", [])
            if not isinstance(updates, list):
                updates = []

            updated_neighbor_record_ids: list[str] = []
            rewritten_record_ids: list[str] = []
            for index, update in enumerate(updates[: len(linked_neighbors)]):
                if not isinstance(update, dict):
                    continue
                neighbor_record = linked_neighbors[index]
                neighbor_payload = note_payload_from_record(
                    neighbor_record,
                    note_namespace=self.note_namespace,
                    default_category=self.default_category,
                )
                patched_payload = {
                    **neighbor_payload,
                    "context": str(update.get("context", "")).strip() or neighbor_payload["context"],
                    "tags": repair_note_payload(
                        {"tags": update.get("tags"), "content": neighbor_payload["content"]},
                        fallback_content=neighbor_payload["content"],
                        default_category=self.default_category,
                    )["tags"]
                    or neighbor_payload["tags"],
                }
                rewrite_record_from_note_payload(
                    store,
                    layer=self.target_layer,
                    record=neighbor_record,
                    payload=patched_payload,
                    note_namespace=self.note_namespace,
                    default_category=self.default_category,
                    embedding_version=self.embedding_version,
                    runtime=runtime,
                )
                updated_neighbor_record_ids.append(neighbor_record.record_id)
                rewritten_record_ids.append(neighbor_record.record_id)

            if updated_neighbor_record_ids:
                active_unit_ids.append(unit.unit_id)
                effects.append(
                    {
                        "effect_type": "neighbor_context_update",
                        "unit_id": unit.unit_id,
                        "record_id": current_record.record_id,
                        "target_layer": self.target_layer,
                        "updated_neighbor_record_ids": updated_neighbor_record_ids,
                        "rewritten_record_ids": rewritten_record_ids,
                    }
                )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "target_layer": self.target_layer,
            "note_namespace": self.note_namespace,
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
    ``packet.evolution_decisions``, and ``packet.observation``. The store is
    mutated by appending generated records and pruning the target layer to the
    configured window.
    """

    spec = ModuleSpec(
        name="reflection_generation_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions", "observation"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

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
        if packet.evolution_decisions is None:
            raise ValueError("ReflectionGenerationEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError(
                "ReflectionGenerationEvolution requires aligned units, placements, and evolution decisions."
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
        generation_mode = "callable_override" if self.reflection_generator is not None else "classic_runtime"

        active_unit_ids: list[str] = []
        record_ids: list[str] = []
        effects: list[dict[str, Any]] = []
        trial_index = reflexion_controls(packet.observation.metadata).get("trial_index", 0)

        for unit, decision, placement in zip(packet.units, packet.evolution_decisions, packet.placements, strict=True):
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
                    # Backward-compatible namespace for existing classic tests/examples.
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
            "decision_source": "evolution_decisions",
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


BASELINE_SLOT: Final[str] = "memory_evolution"
BASELINE_CLASSES: Final[tuple[type[MemoryEvolutionModule], ...]] = (
    AppendOnlyEvolution,
    TraceOnlyEvolution,
    SummaryRewriteEvolution,
    LayerMoveEvolution,
    GraphLinkEvolution,
    GraphNeighborContextTraceEvolution,
    GraphNeighborAppendEvolution,
    LinkStrengtheningEvolution,
    NeighborContextUpdateEvolution,
    ReflectionGenerationEvolution,
)
