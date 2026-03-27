"""A-MEM classic-like modules built on the reusable graph motif baseline layer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Final

from memprimitive import (
    MemoryRecord,
    MemoryStore,
    ModuleSpec,
    Packet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.baselines._amem_family import (
    DEFAULT_CATEGORY,
    DEFAULT_EMBEDDING_VERSION,
    note_payload_from_record,
    repair_note_payload,
    retrieve_candidates_by_embedding,
    rewrite_record_from_note_payload,
    stringify_note_candidates,
)
from memprimitive.baselines._trace import copy_trace
from memprimitive.baselines.evolution_trigger import NeighborExistsEvolutionTrigger
from memprimitive.baselines.memory_evolution import LinkStrengtheningEvolution, NeighborContextUpdateEvolution
from memprimitive.baselines.organization import GraphAppendLinkReadyOrganization
from memprimitive.baselines.readout import NoteRenderReadout
from memprimitive.baselines.representation import (
    RetrievalOrientedEmbeddingRepresentation,
    SemanticFieldEnrichmentRepresentation,
)
from memprimitive.baselines.retrieval import VectorGraphSeedAndExpandRetrieval
from memprimitive.baselines.write_trigger import LLMJudgedWriteTrigger
from memprimitive.interfaces import MemoryEvolutionModule, ReadoutModule, RepresentationModule

from ._runtime import get_classic_runtime


AMEM_GRAPH_LAYER: Final[str] = "memory_graph"


@dataclass(slots=True)
class AMEMConfig:
    graph_layer: str = AMEM_GRAPH_LAYER
    candidate_k: int = 5
    top_k: int = 5
    neighbor_expansion_k: int = 3
    agentic_search: bool = False
    strict_llm: bool = True
    write_decision_enabled: bool = False
    query_expand_with_llm: bool = False
    max_links_per_record: int = 4


def build_amem_store(*, graph_layer: str = AMEM_GRAPH_LAYER) -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name=graph_layer,
                    theme="knowledge_graph",
                    shape="Graph",
                    indices=("graph", "vector", "entity", "keyword", "tag"),
                )
            ]
        )
    )


class AMEMAgenticRepresentation(RepresentationModule):
    """Paper-aligned A-MEM note representation built from two baseline motifs.

    ``run`` requires ``packet.units`` and applies:
    1. semantic field enrichment / comprehensive note construction
    2. retrieval-oriented embedding construction

    The store is unchanged. This wrapper preserves the classic public API while
    shifting the actual slot logic into reusable baseline modules.
    """

    spec = ModuleSpec(
        name="amem_agentic_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.embedding", "units.metadata.amem", "units.metadata.representation"),
    )

    def __init__(self, *, strict_llm: bool = True) -> None:
        if not strict_llm:
            raise ValueError("A-MEM classic implementation requires strict_llm=True.")
        self.strict_llm = strict_llm
        self._modules = (
            SemanticFieldEnrichmentRepresentation(note_namespace="amem", strict_llm=True),
            RetrievalOrientedEmbeddingRepresentation(note_namespace="amem", embedding_version=DEFAULT_EMBEDDING_VERSION),
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        current_packet = packet
        current_store = store
        for module in self._modules:
            current_packet, current_store = module.run(current_packet, current_store)
        trace = copy_trace(current_packet)
        trace["representation"] = {
            **trace.get("representation", {}),
            "module": self.spec.name,
            "baseline_modules": [module.spec.name for module in self._modules],
        }
        return replace(current_packet, trace=trace), current_store


class AMEMAgenticWriteTrigger(LLMJudgedWriteTrigger):
    """A-MEM-compatible wrapper over the generic LLM-judged write trigger."""

    spec = ModuleSpec(
        name="amem_agentic_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__(note_namespace="amem", enabled=enabled, strict_llm=True, default_category=DEFAULT_CATEGORY)


class AMEMAgenticOrganization(GraphAppendLinkReadyOrganization):
    """A-MEM-compatible wrapper over graph append with link-ready metadata."""

    spec = ModuleSpec(
        name="amem_agentic_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
        store_requirements=("index:graph", "index:vector", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph", "target_layer_index:vector"),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER) -> None:
        super().__init__(target_layer=target_layer, note_namespace="amem")


class AMEMAgenticEvolutionTrigger(NeighborExistsEvolutionTrigger):
    """A-MEM-compatible wrapper over the shared neighbor-exists evolution trigger."""

    spec = ModuleSpec(
        name="amem_agentic_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
        store_requirements=("shape:Graph", "index:graph", "index:vector"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:vector"),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER, candidate_k: int = 5) -> None:
        super().__init__(target_layer=target_layer, candidate_top_k=candidate_k)


class AMEMAgenticEvolution(MemoryEvolutionModule):
    """Paper-aligned A-MEM evolution controller on top of graph motif helpers.

    This wrapper keeps the paper residual at the classic layer: a high-level LLM
    decides whether a newly written note should strengthen links, update linked
    neighbors, both, or neither. The underlying note schema, embedding rewrite,
    and graph append assumptions are shared with the reusable baseline layer.
    """

    spec = ModuleSpec(
        name="amem_agentic_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "rewrite_records"),
        store_requirements=("index:graph", "index:vector", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph", "target_layer_index:vector"),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER, candidate_k: int = 5, max_links_per_record: int = 4) -> None:
        self.target_layer = target_layer
        self.candidate_k = candidate_k
        self.max_links_per_record = max_links_per_record

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None or packet.placements is None or packet.evolution_decisions is None:
            raise ValueError("AMEMAgenticEvolution requires packet.units, packet.placements, and packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("AMEMAgenticEvolution requires aligned units, placements, and evolution decisions.")

        runtime = get_classic_runtime()
        runtime.require_llm(capability="A-MEM memory evolution")
        effects: list[dict[str, Any]] = []

        for unit, placement, decision in zip(packet.units, packet.placements, packet.evolution_decisions, strict=True):
            if not decision or placement.target_layer != self.target_layer:
                continue
            current_record = next(
                (record for record in store.iter_records(self.target_layer) if record.unit_id == unit.unit_id),
                None,
            )
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

            current_payload = note_payload_from_record(current_record, note_namespace="amem", default_category=DEFAULT_CATEGORY)
            decision_payload = runtime.json(
                system=(
                    "You are an AI memory evolution agent. Decide whether the new memory should "
                    "strengthen links and/or update neighbor context/tags. Return JSON only."
                ),
                user=json.dumps(
                    {
                        "context": current_payload["context"],
                        "content": current_payload["content"],
                        "keywords": current_payload["keywords"],
                        "nearest_neighbors_memories": stringify_note_candidates(neighbor_candidates, note_namespace="amem"),
                    },
                    ensure_ascii=False,
                ),
            )
            if not isinstance(decision_payload, dict):
                raise ValueError("A-MEM evolution decision must return a JSON object.")
            evolution_decision = str(decision_payload.get("decision", "NO_EVOLUTION")).strip().upper()
            reason = str(decision_payload.get("reason", "")).strip()

            strengthened_links: list[str] = []
            updated_neighbor_record_ids: list[str] = []
            rewritten_record_ids: list[str] = []

            if evolution_decision in {"STRENGTHEN", "STRENGTHEN_AND_UPDATE"}:
                strengthen_payload = runtime.json(
                    system=(
                        "Given a new memory and its neighbors, select related neighbor indices and "
                        "return updated tags for the new memory. Return JSON only."
                    ),
                    user=json.dumps(
                        {
                            "content": current_payload["content"],
                            "keywords": current_payload["keywords"],
                            "nearest_neighbors_memories": stringify_note_candidates(neighbor_candidates, note_namespace="amem"),
                        },
                        ensure_ascii=False,
                    ),
                )
                if not isinstance(strengthen_payload, dict):
                    raise ValueError("A-MEM strengthen details must return a JSON object.")
                connection_indices = strengthen_payload.get("connections", [])
                if not isinstance(connection_indices, list):
                    connection_indices = []
                strengthened_neighbors = [
                    neighbor_candidates[index]
                    for index in connection_indices
                    if isinstance(index, int) and 0 <= index < len(neighbor_candidates)
                ][: self.max_links_per_record]
                strengthened_links = [record.record_id for record in strengthened_neighbors]
                merged_links = list(
                    dict.fromkeys(
                        (
                            current_record.metadata.get("graph", {}).get("links", [])
                            if isinstance(current_record.metadata.get("graph"), dict)
                            else []
                        )
                        + strengthened_links
                    )
                )
                updated_payload = {
                    **current_payload,
                    "tags": repair_note_payload(
                        {"tags": strengthen_payload.get("tags"), "content": current_payload["content"]},
                        fallback_content=current_payload["content"],
                        default_category=DEFAULT_CATEGORY,
                    )["tags"]
                    or current_payload["tags"],
                }
                rewritten = rewrite_record_from_note_payload(
                    store,
                    layer=self.target_layer,
                    record=current_record,
                    payload=updated_payload,
                    note_namespace="amem",
                    default_category=DEFAULT_CATEGORY,
                    embedding_version=DEFAULT_EMBEDDING_VERSION,
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
                rewritten_record_ids.append(current_record.record_id)

            if evolution_decision in {"UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"}:
                update_payload = runtime.json(
                    system=(
                        "Update each neighbor's context and tags based on holistic understanding of the "
                        "new memory and neighbors. Return JSON with updates list."
                    ),
                    user=json.dumps(
                        {
                            "content": current_payload["content"],
                            "context": current_payload["context"],
                            "nearest_neighbors_memories": stringify_note_candidates(neighbor_candidates, note_namespace="amem"),
                            "neighbor_count": len(neighbor_candidates),
                        },
                        ensure_ascii=False,
                    ),
                )
                if not isinstance(update_payload, dict):
                    raise ValueError("A-MEM neighbor updates must return a JSON object.")
                updates = update_payload.get("updates", [])
                if not isinstance(updates, list):
                    updates = []
                for index, update in enumerate(updates[: len(neighbor_candidates)]):
                    if not isinstance(update, dict):
                        continue
                    neighbor_record = neighbor_candidates[index]
                    neighbor_payload = note_payload_from_record(neighbor_record, note_namespace="amem", default_category=DEFAULT_CATEGORY)
                    patched_payload = {
                        **neighbor_payload,
                        "context": str(update.get("context", "")).strip() or neighbor_payload["context"],
                        "tags": repair_note_payload(
                            {"tags": update.get("tags"), "content": neighbor_payload["content"]},
                            fallback_content=neighbor_payload["content"],
                            default_category=DEFAULT_CATEGORY,
                        )["tags"]
                        or neighbor_payload["tags"],
                    }
                    rewrite_record_from_note_payload(
                        store,
                        layer=self.target_layer,
                        record=neighbor_record,
                        payload=patched_payload,
                        note_namespace="amem",
                        default_category=DEFAULT_CATEGORY,
                        embedding_version=DEFAULT_EMBEDDING_VERSION,
                        runtime=runtime,
                    )
                    updated_neighbor_record_ids.append(neighbor_record.record_id)
                    rewritten_record_ids.append(neighbor_record.record_id)

            effects.append(
                {
                    "unit_id": unit.unit_id,
                    "record_id": current_record.record_id,
                    "effect_type": "agentic_evolution",
                    "decision": evolution_decision,
                    "reason": reason,
                    "strengthened_links": strengthened_links,
                    "updated_neighbor_record_ids": updated_neighbor_record_ids,
                    "rewritten_record_ids": rewritten_record_ids,
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class AMEMEnhancedRetrieval(VectorGraphSeedAndExpandRetrieval):
    """A-MEM-compatible wrapper over vector seed-and-expand retrieval."""

    spec = ModuleSpec(
        name="amem_enhanced_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "index:vector", "shape:Graph"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph", "target_layer_index:graph", "target_layer_index:vector"),
    )

    def __init__(
        self,
        *,
        target_layer: str = AMEM_GRAPH_LAYER,
        candidate_k: int = 5,
        top_k: int = 5,
        neighbor_expansion_k: int = 3,
        agentic_search: bool = False,
        query_expand_with_llm: bool = False,
    ) -> None:
        super().__init__(
            top_k=top_k,
            layer=target_layer,
            candidate_k=candidate_k,
            neighbor_expansion_k=neighbor_expansion_k,
            note_namespace="amem",
            agentic_search=agentic_search,
            query_expand_with_llm=query_expand_with_llm,
        )


class AMEMAgenticReadout(NoteRenderReadout):
    """A-MEM-compatible wrapper over note render readout."""

    spec = ModuleSpec(
        name="amem_agentic_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self) -> None:
        super().__init__(note_namespace="amem", include_context=True, include_tags=True)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        packet, store = super().run(packet, store)
        readout = replace(
            packet.readout,
            metadata={
                **packet.readout.metadata,
                "format": "agentic_memory",
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            **trace.get("readout", {}),
            "module": self.spec.name,
            "format": "agentic_memory",
        }
        return replace(packet, readout=readout, trace=trace), store


AMEMGraphOrganization = AMEMAgenticOrganization
AMEMGraphEvolutionTrigger = AMEMAgenticEvolutionTrigger
AMEMGraphLinkEvolution = AMEMAgenticEvolution
AMEMGraphHopRetrieval = AMEMEnhancedRetrieval
AMEMGraphReadout = AMEMAgenticReadout


__all__ = [
    "AMEMAgenticEvolution",
    "AMEMAgenticEvolutionTrigger",
    "AMEMAgenticOrganization",
    "AMEMAgenticReadout",
    "AMEMAgenticRepresentation",
    "AMEMAgenticWriteTrigger",
    "AMEMConfig",
    "AMEMEnhancedRetrieval",
    "AMEMGraphEvolutionTrigger",
    "AMEMGraphHopRetrieval",
    "AMEMGraphLinkEvolution",
    "AMEMGraphOrganization",
    "AMEMGraphReadout",
    "AMEM_GRAPH_LAYER",
    "build_amem_store",
]
