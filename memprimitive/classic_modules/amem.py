"""A-MEM style graph-memory support for the classic example."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Final

from memprimitive import (
    MemoryPipeline,
    MemoryRecord,
    MemoryStore,
    ModuleSpec,
    Observation,
    Packet,
    Placement,
    Query,
    Readout,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.baselines import AlwaysWriteTrigger, BasicRepresentation, PassThroughUnitFormation
from memprimitive.baselines._trace import copy_trace
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import EvolutionTriggerModule, MemoryEvolutionModule, OrganizationModule, ReadoutModule, RetrievalModule
from ._runtime import ClassicRuntime

AMEM_GRAPH_LAYER: Final[str] = "memory_graph"
_GENERIC_TAGS: Final[frozenset[str]] = frozenset({"amem", "graph_note", "entity_rich", "keyword_rich", "relation_rich"})
_GENERIC_KEYWORDS: Final[frozenset[str]] = frozenset({"observation", "entity_rich", "structured_triple", "structured_kv"})
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
_ENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\b")
_TRIPLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\(([^,()]+?),\s*([^,()]+?),\s*([^,()]+?)\)")
_RELATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+"
    r"(likes|prefers|studies|works on|works with|builds|uses|cares about)\s+"
    r"([^.;,\n]+)",
    re.I,
)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _entity_list(text: str, *, hint: Any | None = None) -> list[str]:
    if isinstance(hint, list) and hint:
        return _dedupe([str(item).strip() for item in hint if str(item).strip()])
    entities: list[str] = []
    for match in _ENTITY_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate.casefold() in _GENERIC_TAGS:
            continue
        entities.append(candidate)
    return _dedupe(entities)


def _triple_list(text: str, *, hint: Any | None = None) -> list[tuple[str, str, str]]:
    if isinstance(hint, list):
        triples: list[tuple[str, str, str]] = []
        for item in hint:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append((str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()))
        if triples:
            return triples
    triples: list[tuple[str, str, str]] = []
    for match in _TRIPLE_PATTERN.finditer(text):
        triples.append(tuple(part.strip() for part in match.groups()))
    for match in _RELATION_PATTERN.finditer(text):
        subject, relation, obj = match.groups()
        triples.append((subject.strip(), relation.lower().strip(), obj.strip()))
    return triples


def _keyword_list(text: str, *, entities: list[str], tags: list[str], hint: Any | None = None) -> list[str]:
    if isinstance(hint, list) and hint:
        return _dedupe([str(item).casefold().strip() for item in hint if str(item).strip()])
    counts: dict[str, int] = {}
    for token in _tokenize(text):
        if token in {"the", "and", "for", "with", "that", "this", "from", "into", "about", "have", "has"}:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    extras = [entity.casefold() for entity in entities] + [tag.casefold() for tag in tags if tag.casefold() not in _GENERIC_TAGS]
    return _dedupe(ranked[:8] + extras)


def _representation_profile(record_or_unit) -> dict[str, Any]:
    metadata = record_or_unit.metadata if hasattr(record_or_unit, "metadata") else {}
    representation = metadata.get("representation", {}) if isinstance(metadata.get("representation"), dict) else {}
    text = _normalize_text(getattr(record_or_unit, "text", representation.get("text", "")))
    entities = representation.get("entities") if isinstance(representation.get("entities"), list) else []
    triples = representation.get("triples") if isinstance(representation.get("triples"), list) else []
    keywords = representation.get("keywords") if isinstance(representation.get("keywords"), list) else []
    tags = representation.get("tags") if isinstance(representation.get("tags"), list) else []
    if not entities:
        entities = _entity_list(text, hint=metadata.get("entities"))
    if not triples:
        triples = _triple_list(text, hint=metadata.get("triples"))
    if not keywords:
        keywords = _keyword_list(text, entities=entities, tags=tags, hint=metadata.get("keywords"))
    if not tags:
        tags = _dedupe([getattr(record_or_unit, "unit_type", "note").casefold(), "amem", "graph_note"])
    return {
        "text": text,
        "entities": [str(item) for item in entities],
        "triples": [tuple(str(part) for part in triple) for triple in triples],
        "keywords": [str(item).casefold() for item in keywords],
        "tags": [str(item).casefold() for item in tags],
        "embedding": getattr(record_or_unit, "embedding", None),
    }


def _shared_signal_details(left, right) -> dict[str, Any]:
    left_profile = _representation_profile(left)
    right_profile = _representation_profile(right)
    left_entities = {item.casefold() for item in left_profile["entities"]}
    right_entities = {item.casefold() for item in right_profile["entities"]}
    left_keywords = {item.casefold() for item in left_profile["keywords"] if item.casefold() not in _GENERIC_KEYWORDS}
    right_keywords = {item.casefold() for item in right_profile["keywords"] if item.casefold() not in _GENERIC_KEYWORDS}
    left_tags = {item.casefold() for item in left_profile["tags"] if item.casefold() not in _GENERIC_TAGS}
    right_tags = {item.casefold() for item in right_profile["tags"] if item.casefold() not in _GENERIC_TAGS}
    left_triples = {tuple(part for part in triple) for triple in left_profile["triples"]}
    right_triples = {tuple(part for part in triple) for triple in right_profile["triples"]}
    shared_entities = sorted(left_entities & right_entities)
    shared_keywords = sorted(left_keywords & right_keywords)
    shared_tags = sorted(left_tags & right_tags)
    shared_triples = sorted(tuple(part for part in triple) for triple in left_triples & right_triples)
    embedding_similarity = ClassicRuntime.cosine_similarity(left_profile["embedding"], right_profile["embedding"])
    score = (
        (2.0 * embedding_similarity)
        + (1.8 * len(shared_entities))
        + (1.0 * len(shared_keywords))
        + (2.5 * len(shared_triples))
    )
    return {
        "score": score,
        "embedding_similarity": embedding_similarity,
        "shared_entities": shared_entities,
        "shared_keywords": shared_keywords,
        "shared_tags": shared_tags,
        "shared_triples": shared_triples,
    }


def _update_record_graph_metadata(
    store: MemoryStore,
    *,
    layer: str,
    record_id: str,
    additions: dict[str, dict[str, Any]],
) -> MemoryRecord:
    record = next(record for record in store.iter_records(layer) if record.record_id == record_id)
    graph = dict(record.metadata.get("graph", {})) if isinstance(record.metadata.get("graph"), dict) else {}
    links = [str(value) for value in graph.get("links", [])]
    strengths = dict(graph.get("link_strengths", {})) if isinstance(graph.get("link_strengths"), dict) else {}
    reasons = dict(graph.get("link_reasons", {})) if isinstance(graph.get("link_reasons"), dict) else {}
    for neighbor_id, detail in additions.items():
        if neighbor_id not in links:
            links.append(neighbor_id)
        strengths[neighbor_id] = float(detail.get("score", 0.0))
        reasons[neighbor_id] = list(detail.get("shared_keywords", [])) + list(detail.get("shared_entities", []))
    updated = replace(
        record,
        metadata={
            **record.metadata,
            "graph": {
                **graph,
                "links": links,
                "link_count": len(links),
                "link_strengths": strengths,
                "link_reasons": reasons,
            },
        },
    )
    store.replace_record(layer, record_id, updated)
    return updated


@dataclass(slots=True)
class AMEMConfig:
    graph_layer: str = AMEM_GRAPH_LAYER
    top_k: int = 5
    max_hops: int = 2
    seed_k: int = 2
    max_links_per_record: int = 4
    link_threshold: float = 1.0
    hop_decay: float = 0.72
    fallback_recent_if_isolated: bool = True


def build_amem_store(*, graph_layer: str = AMEM_GRAPH_LAYER) -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name=graph_layer,
                    theme="knowledge_graph",
                    shape="Graph",
                    indices=("graph", "entity", "keyword", "tag"),
                )
            ]
        )
    )


class AMEMGraphOrganization(OrganizationModule):
    """Append graph memories and update links to related prior memories."""

    spec = ModuleSpec(
        name="amem_graph_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
        store_requirements=("index:graph", "index:entity", "index:keyword"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph"),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER, max_links_per_record: int = 4, link_threshold: float = 1.0) -> None:
        self.target_layer = target_layer
        self.max_links_per_record = max_links_per_record
        self.link_threshold = link_threshold

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(f"AMEMGraphOrganization requires declared layer {self.target_layer!r}.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise IncompatibleCompositionError(f"AMEMGraphOrganization requires layer {self.target_layer!r} with shape='Graph'.")

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None or packet.decisions is None:
            raise ValueError("AMEMGraphOrganization requires packet.units and packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("AMEMGraphOrganization requires aligned units and decisions.")

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        effects: list[dict[str, Any]] = []
        existing_records = store.iter_records(self.target_layer)
        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            if not decision:
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=self.target_layer, sequence_id=sequence_id)
            candidate_details = []
            for candidate in existing_records:
                detail = _shared_signal_details(record, candidate)
                if detail["score"] >= self.link_threshold:
                    candidate_details.append((candidate, detail))
            candidate_details.sort(key=lambda item: (-float(item[1]["score"]), item[0].timestamp, item[0].record_id))
            selected = candidate_details[: self.max_links_per_record]
            linked_record_ids = [candidate.record_id for candidate, _ in selected]
            record = replace(
                record,
                metadata={
                    **record.metadata,
                    "graph": {
                        "layer": self.target_layer,
                        "links": linked_record_ids,
                        "link_count": len(linked_record_ids),
                        "link_strengths": {candidate.record_id: float(detail["score"]) for candidate, detail in selected},
                        "link_reasons": {
                            candidate.record_id: list(detail["shared_keywords"]) + list(detail["shared_entities"])
                            for candidate, detail in selected
                        },
                    },
                },
            )
            store.append(record)
            for candidate_id in linked_record_ids:
                store.add_graph_links(self.target_layer, candidate_id, [record.record_id])
            effects.append(
                {
                    "effect_type": "graph_link_update",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "linked_record_ids": linked_record_ids,
                }
            )
            existing_records = store.iter_records(self.target_layer)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "effects": effects,
        }
        return replace(packet, placements=placements, trace=trace), store


class AMEMGraphEvolutionTrigger(EvolutionTriggerModule):
    """Open the evolution stage when a note already has graph links."""

    spec = ModuleSpec(
        name="amem_graph_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER) -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None or packet.placements is None:
            raise ValueError("AMEMGraphEvolutionTrigger requires packet.units and packet.placements.")
        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            record = next((record for record in store.iter_records(self.target_layer) if record.unit_id == unit.unit_id), None)
            linked_record_ids: list[str] = []
            if record is not None:
                graph = record.metadata.get("graph", {})
                if isinstance(graph, dict):
                    linked_record_ids = [str(value) for value in graph.get("links", [])]
            decision = bool(linked_record_ids)
            decisions.append(decision)
            per_unit.append({"unit_id": unit.unit_id, "decision": decision, "linked_record_ids": linked_record_ids})
        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "per_unit": per_unit,
            "evolution_decisions": decisions,
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class AMEMGraphLinkEvolution(MemoryEvolutionModule):
    """Strengthen reciprocal links with graph metadata."""

    spec = ModuleSpec(
        name="amem_graph_link_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store",),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER) -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None or packet.placements is None or packet.evolution_decisions is None:
            raise ValueError("AMEMGraphLinkEvolution requires packet.units, packet.placements, and packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("AMEMGraphLinkEvolution requires aligned units, placements, and evolution decisions.")

        effects: list[dict[str, Any]] = []
        for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True):
            if not decision:
                continue
            record = next((record for record in store.iter_records(self.target_layer) if record.unit_id == unit.unit_id), None)
            if record is None:
                continue
            graph = record.metadata.get("graph", {})
            linked_record_ids = [str(value) for value in graph.get("links", [])] if isinstance(graph, dict) else []
            for linked_record_id in linked_record_ids:
                linked_record = next(
                    (candidate for candidate in store.iter_records(self.target_layer) if candidate.record_id == linked_record_id),
                    None,
                )
                if linked_record is None:
                    continue
                detail = _shared_signal_details(record, linked_record)
                _update_record_graph_metadata(
                    store,
                    layer=self.target_layer,
                    record_id=linked_record.record_id,
                    additions={
                        record.record_id: {
                            "score": detail["score"],
                            "shared_entities": detail["shared_entities"],
                            "shared_keywords": detail["shared_keywords"],
                            "shared_tags": detail["shared_tags"],
                            "shared_triples": detail["shared_triples"],
                        }
                    },
                )
            effects.append(
                {
                    "effect_type": "link_strength_update",
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "linked_record_ids": linked_record_ids,
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class AMEMGraphHopRetrieval(RetrievalModule):
    """Retrieve graph memories by query seeds and graph hops."""

    spec = ModuleSpec(
        name="amem_graph_hop_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
        store_requirements=("index:graph", "index:entity", "index:keyword"),
        layer_requirements=("target_layer_exists", "target_layer_shape:Graph"),
    )

    def __init__(
        self,
        *,
        target_layer: str = AMEM_GRAPH_LAYER,
        top_k: int = 5,
        max_hops: int = 2,
        seed_k: int = 2,
        hop_decay: float = 0.72,
        fallback_recent_if_isolated: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("AMEMGraphHopRetrieval requires top_k > 0.")
        self.target_layer = target_layer
        self.top_k = top_k
        self.max_hops = max_hops
        self.seed_k = seed_k
        self.hop_decay = hop_decay
        self.fallback_recent_if_isolated = fallback_recent_if_isolated

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(f"AMEMGraphHopRetrieval requires declared layer {self.target_layer!r}.")
        if store.layer_shape(self.target_layer) != "Graph":
            raise IncompatibleCompositionError(f"AMEMGraphHopRetrieval requires layer {self.target_layer!r} with shape='Graph'.")

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("AMEMGraphHopRetrieval requires packet.query.")

        records = store.iter_records(self.target_layer)
        if not records:
            retrieved = RetrievedSet(items=[], scores=[], trace={"module": self.spec.name, "candidate_count": 0})
            trace = copy_trace(packet)
            trace["retrieval"] = retrieved.trace
            return replace(packet, retrieved=retrieved, trace=trace), store

        query_profile = _representation_profile(packet.query)
        seed_candidates: list[tuple[float, int, MemoryRecord, dict[str, Any]]] = []
        for order_index, record in enumerate(reversed(records)):
            detail = _shared_signal_details(packet.query, record)
            seed_candidates.append((detail["score"], order_index, record, detail))
        seed_candidates.sort(key=lambda item: (-float(item[0]), item[1], item[2].record_id))
        if any(score > 0 for score, _, _, _ in seed_candidates):
            seed_candidates = [item for item in seed_candidates if item[0] > 0]
        selected_seeds = seed_candidates[: self.seed_k]
        if not selected_seeds and self.fallback_recent_if_isolated:
            selected_seeds = [
                (0.0, idx, record, {"score": 0.0, "shared_entities": [], "shared_keywords": [], "shared_tags": [], "shared_triples": []})
                for idx, record in enumerate(reversed(records[-self.seed_k :]))
            ]

        best_states: dict[str, dict[str, Any]] = {}
        frontier: list[dict[str, Any]] = []
        for score, _, record, detail in selected_seeds:
            state = {"record": record, "hop": 0, "score": float(score), "path": [record.record_id], "detail": detail}
            best_states[record.record_id] = state
            frontier.append(state)

        for hop in range(1, self.max_hops + 1):
            next_frontier: list[dict[str, Any]] = []
            for state in frontier:
                for neighbor in store.iter_graph_neighbors(self.target_layer, state["record"].record_id):
                    detail = _shared_signal_details(state["record"], neighbor)
                    hop_score = (state["score"] * self.hop_decay) + (0.5 * min(1.0, detail["score"]))
                    existing = best_states.get(neighbor.record_id)
                    if existing is not None and existing["score"] >= hop_score and existing["hop"] <= hop:
                        continue
                    new_state = {
                        "record": neighbor,
                        "hop": hop,
                        "score": float(hop_score),
                        "path": [*state["path"], neighbor.record_id],
                        "detail": detail,
                    }
                    best_states[neighbor.record_id] = new_state
                    next_frontier.append(new_state)
            frontier = next_frontier
            if not frontier:
                break

        if len(best_states) < self.top_k and self.fallback_recent_if_isolated:
            for record in reversed(records):
                if record.record_id in best_states:
                    continue
                best_states[record.record_id] = {
                    "record": record,
                    "hop": self.max_hops + 1,
                    "score": 0.0,
                    "path": [record.record_id],
                    "detail": {"score": 0.0, "shared_entities": [], "shared_keywords": [], "shared_tags": [], "shared_triples": []},
                }
                if len(best_states) >= self.top_k:
                    break

        selected_states = sorted(
            best_states.values(),
            key=lambda state: (state["hop"], -state["score"], state["record"].timestamp, state["record"].record_id),
        )[: self.top_k]
        items = [state["record"] for state in selected_states]
        scores = [
            {
                "record_id": state["record"].record_id,
                "rank": rank,
                "score": round(float(state["score"]), 4),
                "hop": state["hop"],
                "path": list(state["path"]),
                "strategy": "graph_hop",
            }
            for rank, state in enumerate(selected_states, start=1)
        ]
        hop_counts: dict[int, int] = {}
        for state in selected_states:
            hop_counts[state["hop"]] = hop_counts.get(state["hop"], 0) + 1

        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "target_layer": self.target_layer,
                "candidate_count": len(records),
                "seed_record_ids": [record.record_id for _, _, record, _ in selected_seeds],
                "visited_count": len(best_states),
                "hop_counts": hop_counts,
                "max_hops": self.max_hops,
                "seed_k": self.seed_k,
                "query_profile": {
                    "entities": query_profile["entities"],
                    "keywords": query_profile["keywords"],
                    "tags": query_profile["tags"],
                },
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


class AMEMGraphReadout(ReadoutModule):
    """Render graph retrieval results grouped by hop distance."""

    spec = ModuleSpec(
        name="amem_graph_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, *, target_layer: str = AMEM_GRAPH_LAYER) -> None:
        self.target_layer = target_layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None or packet.retrieved is None:
            raise ValueError("AMEMGraphReadout requires packet.query and packet.retrieved.")

        score_map = {score["record_id"]: score for score in packet.retrieved.scores}
        grouped: dict[int, list[MemoryRecord]] = {}
        for record in packet.retrieved.items:
            hop = int(score_map.get(record.record_id, {}).get("hop", 0))
            grouped.setdefault(hop, []).append(record)

        lines = [f"Query: {packet.query.text}", ""]
        source_ids: list[str] = []
        hop_counts: dict[int, int] = {}
        for hop in sorted(grouped):
            records = grouped[hop]
            hop_counts[hop] = len(records)
            heading = "direct matches" if hop == 0 else f"{hop}-hop neighbors"
            lines.append(f"[hop {hop}] {heading}")
            for record in records:
                graph = record.metadata.get("graph", {}) if isinstance(record.metadata.get("graph"), dict) else {}
                links = [str(value) for value in graph.get("links", [])] if isinstance(graph.get("links"), list) else []
                lines.append(f"- {record.text} (links={len(links)})")
                source_ids.append(record.record_id)
            lines.append("")

        if len(lines) == 2:
            lines.append("No graph memories retrieved.")

        readout = Readout(
            text="\n".join(lines).strip(),
            source_ids=source_ids,
            metadata={"item_count": len(packet.retrieved.items), "hop_counts": hop_counts, "format": "graph_hop"},
        )
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids, "hop_counts": hop_counts}
        return replace(packet, readout=readout, trace=trace), store


def build_amem_pipeline(
    *,
    config: AMEMConfig | None = None,
    store: MemoryStore | None = None,
    graph_layer: str = AMEM_GRAPH_LAYER,
    top_k: int = 5,
    max_hops: int = 2,
    seed_k: int = 2,
    max_links_per_record: int = 4,
    link_threshold: float = 1.0,
    hop_decay: float = 0.72,
    fallback_recent_if_isolated: bool = True,
) -> MemoryPipeline:
    if config is None:
        config = AMEMConfig(
            graph_layer=graph_layer,
            top_k=top_k,
            max_hops=max_hops,
            seed_k=seed_k,
            max_links_per_record=max_links_per_record,
            link_threshold=link_threshold,
            hop_decay=hop_decay,
            fallback_recent_if_isolated=fallback_recent_if_isolated,
        )
    memory_store = store if store is not None else build_amem_store(graph_layer=config.graph_layer)
    return MemoryPipeline(
        store=memory_store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "keywords", "tags", "triple")),
        write_trigger=AlwaysWriteTrigger(),
        organization=AMEMGraphOrganization(
            target_layer=config.graph_layer,
            max_links_per_record=config.max_links_per_record,
            link_threshold=config.link_threshold,
        ),
        evolution_trigger=AMEMGraphEvolutionTrigger(target_layer=config.graph_layer),
        memory_evolution=AMEMGraphLinkEvolution(target_layer=config.graph_layer),
        retrieval=AMEMGraphHopRetrieval(
            target_layer=config.graph_layer,
            top_k=config.top_k,
            max_hops=config.max_hops,
            seed_k=config.seed_k,
            hop_decay=config.hop_decay,
            fallback_recent_if_isolated=config.fallback_recent_if_isolated,
        ),
        readout=AMEMGraphReadout(target_layer=config.graph_layer),
    )


class AMEMWorkstream:
    """Convenience wrapper around the A-MEM pipeline."""

    def __init__(self, *, config: AMEMConfig | None = None, store: MemoryStore | None = None) -> None:
        self.pipeline = build_amem_pipeline(config=config, store=store)

    @property
    def store(self) -> MemoryStore:
        return self.pipeline.store

    def add_memory(self, content: str, *, source: str = "dialogue", metadata: dict[str, Any] | None = None) -> Packet:
        observation = Observation(text=content, source=source, metadata={} if metadata is None else dict(metadata))
        return self.pipeline.ingest(observation)

    def retrieve_memory(self, query: str) -> Readout:
        return self.pipeline.recall(Query(text=query))


__all__ = [
    "AMEMConfig",
    "AMEMGraphEvolutionTrigger",
    "AMEMGraphHopRetrieval",
    "AMEMGraphLinkEvolution",
    "AMEMGraphOrganization",
    "AMEMGraphReadout",
    "AMEM_GRAPH_LAYER",
    "AMEMWorkstream",
    "build_amem_pipeline",
    "build_amem_store",
]
