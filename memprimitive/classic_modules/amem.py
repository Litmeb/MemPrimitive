"""A-MEM memory-side modules aligned to the agentic-memory paper motif."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Final

from memprimitive import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    ModuleSpec,
    Packet,
    Placement,
    Query,
    Readout,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.baselines._trace import copy_trace
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import (
    EvolutionTriggerModule,
    MemoryEvolutionModule,
    OrganizationModule,
    ReadoutModule,
    RepresentationModule,
    RetrievalModule,
    WriteTriggerModule,
)
from ._runtime import ClassicRuntime, get_classic_runtime

AMEM_GRAPH_LAYER: Final[str] = "memory_graph"
_EMBEDDING_VERSION: Final[str] = "content_context_keywords_tags_v2"
_DEFAULT_CATEGORY: Final[str] = "Uncategorized"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe_text_list(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [part.strip() for part in values.split(",")]
    elif isinstance(values, list):
        raw_values = [str(item).strip() for item in values]
    else:
        raw_values = []
    return list(dict.fromkeys(item for item in raw_values if item))


def _normalize_attributes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        normalized_key = _normalize_text(key)
        normalized_value = _normalize_text(raw)
        if normalized_key and normalized_value:
            out[normalized_key] = normalized_value
    return out


def _repair_note_payload(payload: dict[str, Any], *, fallback_content: str) -> dict[str, Any]:
    content = _normalize_text(payload.get("content") or fallback_content)
    note_text = _normalize_text(payload.get("note_text") or content)
    context = _normalize_text(payload.get("context") or content)
    keywords = _dedupe_text_list(payload.get("keywords"))
    tags = _dedupe_text_list(payload.get("tags"))
    category = _normalize_text(payload.get("category") or _DEFAULT_CATEGORY)
    attributes = _normalize_attributes(payload.get("attributes"))
    if not tags and keywords:
        tags = keywords[:3]
    return {
        "content": content,
        "note_text": note_text,
        "context": context,
        "keywords": keywords,
        "tags": tags,
        "category": category,
        "attributes": attributes,
    }


def _build_enhanced_embedding_text(
    *,
    content: str,
    context: str,
    keywords: list[str],
    tags: list[str],
) -> str:
    parts = [
        f"content: {content}",
        f"context: {context}",
        f"keywords: {', '.join(keywords)}",
        f"tags: {', '.join(tags)}",
    ]
    return " | ".join(part for part in parts if _normalize_text(part))


def _representation_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    enhanced_embedding_text = _build_enhanced_embedding_text(
        content=payload["content"],
        context=payload["context"],
        keywords=payload["keywords"],
        tags=payload["tags"],
    )
    representation = {
        "text": payload["content"],
        "normalized_text": payload["content"].casefold(),
        "note_text": payload["note_text"],
        "context": payload["context"],
        "keywords": list(payload["keywords"]),
        "tags": list(payload["tags"]),
        "category": payload["category"],
        "attributes": dict(payload["attributes"]),
        "enhanced_embedding_text": enhanced_embedding_text,
        "embedding_version": _EMBEDDING_VERSION,
    }
    payload["enhanced_embedding_text"] = enhanced_embedding_text
    return representation


def _amem_payload_from_record(record: MemoryRecord) -> dict[str, Any]:
    amem_meta = record.metadata.get("amem", {})
    if isinstance(amem_meta, dict):
        payload = dict(amem_meta)
    else:
        payload = {}
    payload.setdefault("content", record.text)
    payload.setdefault("note_text", record.text)
    representation = record.metadata.get("representation", {})
    if isinstance(representation, dict):
        for key in ("context", "keywords", "tags", "category", "attributes", "enhanced_embedding_text", "embedding_version"):
            if key in representation and key not in payload:
                payload[key] = representation[key]
    return _repair_note_payload(payload, fallback_content=record.text)


def _record_links(record: MemoryRecord) -> list[str]:
    graph = record.metadata.get("graph", {})
    if not isinstance(graph, dict):
        return []
    links = graph.get("links", [])
    if not isinstance(links, list):
        return []
    return [str(item) for item in links if str(item).strip()]


def _rewrite_record_from_payload(
    store: MemoryStore,
    *,
    layer: str,
    record: MemoryRecord,
    payload: dict[str, Any],
    preserve_graph: bool = True,
) -> MemoryRecord:
    repaired = _repair_note_payload(payload, fallback_content=record.text)
    representation = _representation_from_payload(repaired)
    updated = replace(
        record,
        text=repaired["note_text"],
        embedding=get_classic_runtime().embed(representation["enhanced_embedding_text"]),
        metadata={
            **record.metadata,
            "amem": {
                **repaired,
                "enhanced_embedding_text": representation["enhanced_embedding_text"],
                "embedding_version": _EMBEDDING_VERSION,
            },
            "representation": representation,
            **({"graph": record.metadata.get("graph", {})} if preserve_graph else {}),
        },
    )
    store.replace_record(layer, record.record_id, updated)
    return updated


def _stringify_candidates(records: list[MemoryRecord]) -> str:
    lines: list[str] = []
    for index, record in enumerate(records):
        payload = _amem_payload_from_record(record)
        lines.append(
            "\n".join(
                [
                    f"memory index:{index}",
                    f"talk start time:{record.timestamp}",
                    f"memory content: {payload['content']}",
                    f"memory context: {payload['context']}",
                    f"memory keywords: {payload['keywords']}",
                    f"memory tags: {payload['tags']}",
                ]
            )
        )
    return "\n".join(lines)


def _retrieve_candidates_by_embedding(
    *,
    store: MemoryStore,
    layer: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[float, MemoryRecord]]:
    scored: list[tuple[float, MemoryRecord]] = []
    for record in store.iter_records(layer):
        score = ClassicRuntime.cosine_similarity(query_embedding, record.embedding)
        scored.append((float(score), record))
    scored.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].record_id))
    return scored[:top_k]


def _collect_neighbor_candidates(
    *,
    store: MemoryStore,
    layer: str,
    seed_records: list[MemoryRecord],
    neighbor_expansion_k: int,
) -> list[MemoryRecord]:
    if neighbor_expansion_k <= 0:
        return []
    selected: list[MemoryRecord] = []
    seen: set[str] = set()
    for seed in seed_records:
        for neighbor_id in _record_links(seed):
            if neighbor_id in seen:
                continue
            neighbor = next((record for record in store.iter_records(layer) if record.record_id == neighbor_id), None)
            if neighbor is None:
                continue
            seen.add(neighbor_id)
            selected.append(neighbor)
            if len(selected) >= neighbor_expansion_k:
                return selected
    return selected


def _merge_records_by_id(records: list[MemoryRecord]) -> list[MemoryRecord]:
    merged: list[MemoryRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        merged.append(record)
    return merged


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
    """Generate comprehensive notes plus structured metadata using the classic runtime."""

    spec = ModuleSpec(
        name="amem_agentic_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.text", "units.embedding", "units.metadata.amem", "units.metadata.representation"),
    )

    def __init__(self, *, strict_llm: bool = True) -> None:
        self.strict_llm = strict_llm
        if not self.strict_llm:
            raise ValueError("A-MEM classic implementation requires strict_llm=True.")

    def _analyze_content(self, content: str) -> dict[str, Any]:
        runtime = get_classic_runtime()
        runtime.require_llm(capability="A-MEM note analysis")
        payload = runtime.json(
            system=(
                "You are the A-MEM note generator. Produce a comprehensive note with context, "
                "keywords, tags, category, and structured attributes. Return JSON only."
            ),
            user=json.dumps(
                {
                    "content": content,
                    "required_fields": [
                        "note_text",
                        "context",
                        "keywords",
                        "tags",
                        "category",
                        "attributes",
                    ],
                },
                ensure_ascii=False,
            ),
        )
        if not isinstance(payload, dict):
            raise ValueError("A-MEM note analysis must return a JSON object.")
        return _repair_note_payload(payload, fallback_content=content)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AMEMAgenticRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            note_payload = self._analyze_content(unit.text)
            representation = _representation_from_payload(note_payload)
            represented_units.append(
                replace(
                    unit,
                    text=note_payload["content"],
                    normalized_text=note_payload["content"].casefold(),
                    embedding=get_classic_runtime().embed(representation["enhanced_embedding_text"]),
                    description=note_payload["context"],
                    tags=list(note_payload["tags"]),
                    representation_elements=("text", "embedding", "description", "keywords", "tags"),
                    metadata={
                        **unit.metadata,
                        "amem": {
                            **note_payload,
                            "enhanced_embedding_text": representation["enhanced_embedding_text"],
                            "embedding_version": _EMBEDDING_VERSION,
                        },
                        "representation": representation,
                    },
                )
            )
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "content": note_payload["content"],
                    "note_text": note_payload["note_text"],
                    "context": note_payload["context"],
                    "keywords": list(note_payload["keywords"]),
                    "tags": list(note_payload["tags"]),
                    "category": note_payload["category"],
                }
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "strict_llm": True,
            "per_unit": per_unit,
        }
        return replace(packet, units=represented_units, trace=trace), store


class AMEMAgenticWriteTrigger(WriteTriggerModule):
    """Use the LLM to decide whether a generated note should be stored."""

    spec = ModuleSpec(
        name="amem_agentic_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("AMEMAgenticWriteTrigger requires packet.units.")

        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            if not self.enabled:
                decision = True
                reason = "write_decision_disabled"
                confidence = 1.0
            else:
                runtime = get_classic_runtime()
                runtime.require_llm(capability="A-MEM write decision")
                amem_meta = unit.metadata.get("amem", {})
                payload = runtime.json(
                    system=(
                        "You are the A-MEM write controller. Decide if a note should be stored. "
                        "Return JSON with fields decision, reason, confidence."
                    ),
                    user=json.dumps(
                        {
                            "content": unit.text,
                            "note_text": amem_meta.get("note_text", unit.text),
                            "context": amem_meta.get("context", ""),
                            "keywords": amem_meta.get("keywords", []),
                            "tags": amem_meta.get("tags", []),
                        },
                        ensure_ascii=False,
                    ),
                )
                if not isinstance(payload, dict):
                    raise ValueError("A-MEM write decision must return a JSON object.")
                decision = str(payload.get("decision", "write")).strip().casefold() != "skip"
                reason = _normalize_text(payload.get("reason") or "unspecified")
                raw_confidence = payload.get("confidence", 1.0)
                confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 1.0
            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "decision": "write" if decision else "skip",
                    "reason": reason,
                    "confidence": confidence,
                    "content": unit.text,
                }
            )

        trace = copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "per_unit": per_unit,
            "decisions": list(decisions),
        }
        return replace(packet, decisions=decisions, trace=trace), store


class AMEMAgenticOrganization(OrganizationModule):
    """Append note records to the graph layer while preserving A-MEM metadata."""

    spec = ModuleSpec(
        name="amem_agentic_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
        store_requirements=("index:graph", "index:vector", "index:keyword", "index:tag"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph"),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER) -> None:
        self.target_layer = target_layer

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(f"AMEMAgenticOrganization requires declared layer {self.target_layer!r}.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise IncompatibleCompositionError(f"AMEMAgenticOrganization requires layer {self.target_layer!r} with shape='Graph'.")

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None or packet.decisions is None:
            raise ValueError("AMEMAgenticOrganization requires packet.units and packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("AMEMAgenticOrganization requires aligned units and decisions.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        effects: list[dict[str, Any]] = []
        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            if not decision:
                effects.append({"unit_id": unit.unit_id, "effect_type": "skipped"})
                continue
            sequence_id = store.next_sequence_id()
            amem_payload = dict(unit.metadata.get("amem", {}))
            representation = dict(unit.metadata.get("representation", {}))
            record = MemoryRecord.from_unit(unit=unit, layer=self.target_layer, sequence_id=sequence_id)
            record = replace(
                record,
                embedding=list(unit.embedding) if unit.embedding is not None else None,
                metadata={
                    **record.metadata,
                    "amem": {
                        **amem_payload,
                        "enhanced_embedding_text": representation.get("enhanced_embedding_text", ""),
                        "embedding_version": _EMBEDDING_VERSION,
                    },
                    "representation": {
                        **representation,
                        "embedding_version": _EMBEDDING_VERSION,
                    },
                    "graph": {
                        "links": [],
                        "link_count": 0,
                    },
                },
            )
            store.append(record)
            effects.append(
                {
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "effect_type": "append_note",
                    "target_layer": self.target_layer,
                }
            )

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "effects": effects,
        }
        return replace(packet, placements=placements, trace=trace), store


class AMEMAgenticEvolutionTrigger(EvolutionTriggerModule):
    """Run A-MEM evolution when a freshly written note has nearby memories."""

    spec = ModuleSpec(
        name="amem_agentic_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
        store_requirements=("index:vector",),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER, candidate_k: int = 5) -> None:
        self.target_layer = target_layer
        self.candidate_k = candidate_k

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None or packet.placements is None:
            raise ValueError("AMEMAgenticEvolutionTrigger requires packet.units and packet.placements.")

        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            if unit.embedding is None:
                raise ValueError("A-MEM evolution trigger requires unit embeddings.")
            candidates = [
                record
                for _, record in _retrieve_candidates_by_embedding(
                    store=store,
                    layer=self.target_layer,
                    query_embedding=unit.embedding,
                    top_k=self.candidate_k + 1,
                )
                if record.unit_id != unit.unit_id
            ]
            decision = bool(candidates)
            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "candidate_record_ids": [record.record_id for record in candidates[: self.candidate_k]],
                    "decision": decision,
                }
            )

        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "per_unit": per_unit,
            "evolution_decisions": list(decisions),
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class AMEMAgenticEvolution(MemoryEvolutionModule):
    """Apply the original A-MEM strengthen/update-neighbor evolution loop."""

    spec = ModuleSpec(
        name="amem_agentic_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store",),
        store_requirements=("index:graph", "index:vector"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph"),
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
        for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True):
            if not decision:
                continue

            current_record = next(
                (record for record in store.iter_records(self.target_layer) if record.unit_id == unit.unit_id),
                None,
            )
            if current_record is None or current_record.embedding is None:
                continue

            neighbor_candidates = [
                record
                for _, record in _retrieve_candidates_by_embedding(
                    store=store,
                    layer=self.target_layer,
                    query_embedding=current_record.embedding,
                    top_k=self.candidate_k + 1,
                )
                if record.record_id != current_record.record_id
            ][: self.candidate_k]
            if not neighbor_candidates:
                continue

            current_payload = _amem_payload_from_record(current_record)
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
                        "nearest_neighbors_memories": _stringify_candidates(neighbor_candidates),
                    },
                    ensure_ascii=False,
                ),
            )
            if not isinstance(decision_payload, dict):
                raise ValueError("A-MEM evolution decision must return a JSON object.")
            evolution_decision = str(decision_payload.get("decision", "NO_EVOLUTION")).strip().upper()
            reason = _normalize_text(decision_payload.get("reason") or "")

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
                            "nearest_neighbors_memories": _stringify_candidates(neighbor_candidates),
                        },
                        ensure_ascii=False,
                    ),
                )
                if not isinstance(strengthen_payload, dict):
                    raise ValueError("A-MEM strengthen details must return a JSON object.")
                connection_indices = strengthen_payload.get("connections", [])
                if not isinstance(connection_indices, list):
                    connection_indices = []
                tags = _dedupe_text_list(strengthen_payload.get("tags"))
                strengthened_neighbors = [
                    neighbor_candidates[index]
                    for index in connection_indices
                    if isinstance(index, int) and 0 <= index < len(neighbor_candidates)
                ][: self.max_links_per_record]
                strengthened_links = [record.record_id for record in strengthened_neighbors]
                merged_links = list(dict.fromkeys(_record_links(current_record) + strengthened_links))
                updated_payload = {
                    **current_payload,
                    "tags": tags or current_payload["tags"],
                }
                rewritten_current = _rewrite_record_from_payload(store, layer=self.target_layer, record=current_record, payload=updated_payload)
                current_record = replace(
                    rewritten_current,
                    metadata={
                        **rewritten_current.metadata,
                        "graph": {
                            **(rewritten_current.metadata.get("graph", {}) if isinstance(rewritten_current.metadata.get("graph"), dict) else {}),
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
                            "nearest_neighbors_memories": _stringify_candidates(neighbor_candidates),
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
                    neighbor_payload = _amem_payload_from_record(neighbor_record)
                    patched_payload = {
                        **neighbor_payload,
                        "context": _normalize_text(update.get("context") or neighbor_payload["context"]),
                        "tags": _dedupe_text_list(update.get("tags")) or neighbor_payload["tags"],
                    }
                    _rewrite_record_from_payload(store, layer=self.target_layer, record=neighbor_record, payload=patched_payload)
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


class AMEMEnhancedRetrieval(RetrievalModule):
    """Run original-style vector retrieval with optional linked-neighbor expansion."""

    spec = ModuleSpec(
        name="amem_enhanced_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "index:vector"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph"),
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
        if candidate_k <= 0 or top_k <= 0:
            raise ValueError("AMEMEnhancedRetrieval requires candidate_k > 0 and top_k > 0.")
        self.target_layer = target_layer
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.neighbor_expansion_k = neighbor_expansion_k
        self.agentic_search = agentic_search
        self.query_expand_with_llm = query_expand_with_llm

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(f"AMEMEnhancedRetrieval requires declared layer {self.target_layer!r}.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise IncompatibleCompositionError(f"AMEMEnhancedRetrieval requires layer {self.target_layer!r} with shape='Graph'.")

    def _query_payload(self, query: Query) -> dict[str, Any]:
        if not self.query_expand_with_llm:
            payload = {
                "query_text": query.text,
                "content": query.text,
                "context": "",
                "keywords": [],
                "tags": [],
                "category": "query",
                "attributes": {},
            }
        else:
            runtime = get_classic_runtime()
            runtime.require_llm(capability="A-MEM query expansion")
            raw = runtime.json(
                system=(
                    "Expand the query for A-MEM retrieval. Return JSON with fields query_text, context, "
                    "keywords, tags, category, attributes."
                ),
                user=json.dumps({"query": query.text}, ensure_ascii=False),
            )
            if not isinstance(raw, dict):
                raise ValueError("A-MEM query expansion must return a JSON object.")
            payload = _repair_note_payload(raw, fallback_content=query.text)
        payload["query_text"] = _normalize_text(payload.get("query_text") or payload.get("note_text") or query.text)
        payload["note_text"] = payload["query_text"]
        payload["content"] = _normalize_text(payload.get("content") or payload["query_text"])
        return payload

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("AMEMEnhancedRetrieval requires packet.query.")

        query_payload = self._query_payload(packet.query)
        runtime = get_classic_runtime()
        if packet.query.embedding is not None:
            query_embedding = list(packet.query.embedding)
        elif self.query_expand_with_llm:
            enhanced_query_text = _build_enhanced_embedding_text(
                content=query_payload["content"],
                context=query_payload["context"],
                keywords=query_payload["keywords"],
                tags=query_payload["tags"],
            )
            query_embedding = runtime.embed(enhanced_query_text)
        else:
            query_embedding = runtime.embed(packet.query.text)
        query = replace(packet.query, embedding=list(query_embedding))

        primary = _retrieve_candidates_by_embedding(
            store=store,
            layer=self.target_layer,
            query_embedding=list(query_embedding),
            top_k=self.candidate_k,
        )
        primary_records = [record for _, record in primary]
        neighbor_records = _collect_neighbor_candidates(
            store=store,
            layer=self.target_layer,
            seed_records=primary_records,
            neighbor_expansion_k=self.neighbor_expansion_k,
        )
        merged_records = _merge_records_by_id(primary_records + neighbor_records)
        candidate_payload = []
        for record in merged_records:
            payload = _amem_payload_from_record(record)
            candidate_payload.append(
                {
                    "id": record.record_id,
                    "content": payload["content"],
                    "note_text": payload["note_text"],
                    "context": payload["context"],
                    "tags": payload["tags"],
                    "keywords": payload["keywords"],
                    "score": next((score for score, candidate in primary if candidate.record_id == record.record_id), 0.0),
                }
            )
        if self.agentic_search:
            reranked = runtime.rerank(
                query=query.text,
                candidates=candidate_payload,
                task="Perform A-MEM search_agentic retrieval over comprehensive memory notes.",
                top_k=self.top_k,
            )
        else:
            reranked = [
                {"id": record.record_id, "score": score, "rationale": "embedding retrieval"}
                for score, record in primary[: self.top_k]
            ]

        ordered_ids = [item["id"] for item in reranked]
        record_by_id = {record.record_id: record for record in merged_records}
        items = [record_by_id[record_id] for record_id in ordered_ids if record_id in record_by_id][: self.top_k]
        scores = [
            {
                "record_id": item["id"],
                "score": float(item.get("score", 0.0)),
                "rationale": _normalize_text(item.get("rationale") or ""),
                "strategy": "search_agentic" if self.agentic_search else "vector_plus_links",
            }
            for item in reranked
            if item["id"] in record_by_id
        ][: self.top_k]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "retrieval_mode": "search_agentic" if self.agentic_search else "vector_plus_links",
                "candidate_count": len(merged_records),
                "selected_count": len(items),
                "candidate_record_ids": [record.record_id for record in merged_records],
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, query=query, retrieved=retrieved, trace=trace), store


class AMEMAgenticReadout(ReadoutModule):
    """Render retrieved memories in A-MEM search output style."""

    spec = ModuleSpec(
        name="amem_agentic_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None or packet.retrieved is None:
            raise ValueError("AMEMAgenticReadout requires packet.query and packet.retrieved.")

        lines = [f"Query: {packet.query.text}", ""]
        source_ids: list[str] = []
        for record in packet.retrieved.items:
            payload = _amem_payload_from_record(record)
            source_ids.append(record.record_id)
            lines.append(f"- {payload['content']}")
            lines.append(f"  context: {payload['context']}")
            lines.append(f"  tags: {', '.join(payload['tags'])}")
        if not packet.retrieved.items:
            lines.append("No agentic memories retrieved.")

        readout = Readout(
            text="\n".join(lines).strip(),
            source_ids=source_ids,
            metadata={
                "item_count": len(packet.retrieved.items),
                "format": "agentic_memory",
                "retrieval_mode": packet.retrieved.trace.get("retrieval_mode", "vector_plus_links"),
                "candidate_count": packet.retrieved.trace.get("candidate_count", 0),
                "selected_count": packet.retrieved.trace.get("selected_count", len(packet.retrieved.items)),
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
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
