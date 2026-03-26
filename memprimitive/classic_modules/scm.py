"""Self-Controlled Memory (SCM) motif support.

The real paper uses structured extraction, a judge/gate for writes, entity-
centered profile upserts, and controlled retrieval. This module keeps that
shape while remaining fully deterministic and local to the repo.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Final

from memprimitive import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    ModuleSpec,
    Observation,
    Packet,
    Placement,
    Query,
    Readout,
    RetrievedSet,
)
from memprimitive.baselines._trace import copy_trace
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import OrganizationModule, ReadoutModule, RetrievalModule, UnitFormationModule, WriteTriggerModule
from ._runtime import get_classic_runtime

_ENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\b")
_TRIPLE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)")
_IS_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+is\s+([^.;,\n]+)", re.I)
_RELATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+"
    r"(works at|works for|works on|joined|belongs to|likes|loves|prefers|studies|manages)\s+"
    r"([^.;,\n]+)",
    re.I,
)
_KV_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Za-z][A-Za-z0-9_ ]{0,40}?)\s*[:=]\s*([^.;,\n]+)")
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
_STOPWORDS: Final[frozenset[str]] = frozenset({"the", "a", "an", "and", "or", "of", "in", "on", "to"})


def _copy_trace(packet: Packet) -> dict[str, Any]:
    return copy_trace(packet)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _scm_extract(text: str, *, metadata: dict[str, Any]) -> dict[str, Any]:
    runtime = get_classic_runtime()
    hinted_entities = metadata.get("entities")
    hinted_triples = metadata.get("triples")
    hinted_kv = metadata.get("kv")
    result = runtime.json(
        system=(
            "You extract structured memory facts from text. "
            "Return JSON with keys: entities, triples, kv, summary, tags."
        ),
        user=(
            f"text: {text}\n"
            f"hinted_entities: {hinted_entities}\n"
            f"hinted_triples: {hinted_triples}\n"
            f"hinted_kv: {hinted_kv}\n"
            "Rules: triples must be [subject, predicate, object]. kv must be an object of short strings."
        ),
    )
    if not isinstance(result, dict):
        raise ValueError("SCM extraction must return a JSON object.")
    entities = _extract_entities(text, hint=result.get("entities") or hinted_entities)
    triples = _extract_triples(text, hint=result.get("triples") or hinted_triples)
    kv = _extract_kv(text, hint=result.get("kv") or hinted_kv)
    summary = str(result.get("summary", "")).strip() or _summarize_fact(entities[0] if entities else None, triples, kv, text)
    tags = _dedupe([str(item).strip() for item in result.get("tags", [])]) if isinstance(result.get("tags"), list) else []
    return {
        "entities": entities,
        "triples": triples,
        "kv": kv,
        "summary": summary,
        "tags": tags,
    }


def _extract_entities(text: str, *, hint: Any | None = None) -> list[str]:
    if isinstance(hint, list) and hint:
        return _dedupe([str(item).strip() for item in hint if str(item).strip()])

    entities = []
    for match in _ENTITY_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate.casefold() in _STOPWORDS:
            continue
        entities.append(candidate)
    return _dedupe(entities)


def _extract_triples(text: str, *, hint: Any | None = None) -> list[tuple[str, str, str]]:
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
    for match in _IS_PATTERN.finditer(text):
        subject, obj = match.groups()
        triples.append((subject.strip(), "is", obj.strip()))
    return triples


def _extract_kv(text: str, *, hint: Any | None = None) -> dict[str, str]:
    if isinstance(hint, dict) and hint:
        return {str(key).strip(): str(value).strip() for key, value in hint.items() if str(key).strip()}

    kv: dict[str, str] = {}
    for key, value in _KV_PATTERN.findall(text):
        kv[key.strip().casefold().replace(" ", "_")] = value.strip()
    return kv


def _summarize_fact(entity: str | None, triples: list[tuple[str, str, str]], kv: dict[str, str], text: str) -> str:
    parts: list[str] = []
    if triples:
        parts.extend(f"{subject} {predicate.replace('_', ' ')} {obj}" for subject, predicate, obj in triples[:3])
    if kv:
        parts.extend(f"{key}={value}" for key, value in list(kv.items())[:3])
    if not parts:
        parts.append(_normalize_text(text))
    body = "; ".join(parts)
    if entity:
        return f"{entity}: {body}"
    return body


def _query_entities(query: Query) -> list[str]:
    hinted = query.metadata.get("entities")
    if isinstance(hinted, list) and hinted:
        return _dedupe([str(item).strip() for item in hinted if str(item).strip()])

    entities = [token for token in query.text.split() if token and token[:1].isupper()]
    return _dedupe(entities)


def _record_entities(record) -> list[str]:
    profile = record.metadata.get("profile")
    if isinstance(profile, dict):
        entity = profile.get("entity")
        aliases = profile.get("aliases")
        values = []
        if isinstance(entity, str) and entity.strip():
            values.append(entity.strip())
        if isinstance(aliases, list):
            values.extend(str(item).strip() for item in aliases if str(item).strip())
        if values:
            return _dedupe(values)

    representation = record.metadata.get("representation")
    if isinstance(representation, dict):
        entities = representation.get("entities")
        if isinstance(entities, list) and entities:
            return _dedupe([str(item).strip() for item in entities if str(item).strip()])
    return []


def _record_tokens(record) -> set[str]:
    tokens = {token.casefold() for token in _TOKEN_PATTERN.findall(record.text)}
    representation = record.metadata.get("representation")
    if isinstance(representation, dict):
        keywords = representation.get("keywords")
        if isinstance(keywords, list):
            tokens.update(str(item).casefold() for item in keywords if str(item).strip())
    profile = record.metadata.get("profile")
    if isinstance(profile, dict):
        summary = profile.get("summary")
        if isinstance(summary, str):
            tokens.update(token.casefold() for token in _TOKEN_PATTERN.findall(summary))
    return tokens


def _layer_priority(layer: str, *, profile_layer: str, semantic_layer: str) -> int:
    if layer == profile_layer:
        return 0
    if layer == semantic_layer:
        return 1
    return 2


class SCMStructuredExtraction(UnitFormationModule):
    """Extract structured SCM units from observations.

    The extractor keeps the write-path deterministic: it emits one structured
    unit per observation and fills ``entities``, ``triples``, ``kv`` and a small
    local embedding so later stages can gate and rerank on real structure.
    """

    spec = ModuleSpec(
        name="scm_structured_extraction",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.entities", "units.triples", "units.kv", "units.embedding"),
    )

    def __init__(self, *, embedding_dim: int = 16) -> None:
        self.embedding_dim = int(embedding_dim)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("SCMStructuredExtraction requires packet.observation.")

        observation = packet.observation
        text = _normalize_text(observation.text)
        extraction = _scm_extract(text, metadata=observation.metadata)
        entities = extraction["entities"]
        triples = extraction["triples"]
        kv = extraction["kv"]
        primary_entity = entities[0] if entities else None
        summary = extraction["summary"]
        embedding = get_classic_runtime().embed(text)
        tags = _dedupe(["scm", "structured", *extraction["tags"]])

        unit = MemoryUnit(
            text=text,
            unit_type="structured_fact",
            timestamp=observation.timestamp,
            representation_elements=("text", "triple", "entities", "kv", "embedding"),
            normalized_text=text.casefold(),
            embedding=embedding,
            triples=triples,
            kv=kv,
            entities=entities,
            tags=tags,
            description=summary,
            metadata={
                **observation.metadata,
                "source": observation.source,
                "provenance": {
                    "observation_id": observation.observation_id,
                    "source": observation.source,
                },
                "scm": {
                    "primary_entity": primary_entity,
                    "entity_count": len(entities),
                    "triple_count": len(triples),
                    "kv_count": len(kv),
                    "summary": summary,
                },
                "representation": {
                    "text": text,
                    "normalized_text": text.casefold(),
                    "entities": entities,
                    "triples": triples,
                    "kv": kv,
                    "embedding": {"dim": len(embedding)},
                    "description": summary,
                    "tags": tags,
                },
            },
        )

        trace = _copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id],
            "entity_count": len(entities),
            "triple_count": len(triples),
            "kv_count": len(kv),
        }
        return replace(packet, units=[unit], trace=trace), store


class SCMJudgeGateWrite(WriteTriggerModule):
    """Judge structured units and gate the normal write path.

    The score is a deterministic stand-in for the paper's LLM judge. Structured
    facts with entities/triples/kv fields receive higher scores and are more
    likely to pass the gate.
    """

    spec = ModuleSpec(
        name="scm_judge_gate_write",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(self, *, threshold: float = 0.55, gate_floor: float = 0.15) -> None:
        self.threshold = float(threshold)
        self.gate_floor = float(gate_floor)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("SCMJudgeGateWrite requires packet.units.")

        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            judge = get_classic_runtime().json(
                system=(
                    "You are an SCM memory write judge. "
                    "Return JSON with keys: score, gate_open, rationale."
                ),
                user=(
                    f"text: {unit.text}\n"
                    f"entities: {unit.entities}\n"
                    f"triples: {unit.triples}\n"
                    f"kv: {unit.kv}\n"
                    f"description: {unit.description}\n"
                    "Approve only if this is a durable factual memory worth storing."
                ),
            )
            entity_count = len(unit.entities)
            triple_count = len(unit.triples)
            kv_count = len(unit.kv)
            embedding_present = 1.0 if unit.embedding else 0.0
            raw_score = judge.get("score", 0.0) if isinstance(judge, dict) else 0.0
            structure_score = max(0.0, min(float(raw_score), 1.0)) if isinstance(raw_score, (int, float)) else 0.0
            gate_flag = judge.get("gate_open", False) if isinstance(judge, dict) else False
            gate_open = bool(gate_flag) and structure_score >= self.gate_floor
            decision = gate_open and structure_score >= self.threshold
            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "signals": {
                        "llm_value_estimate": structure_score,
                        "entity_count": float(entity_count),
                        "triple_count": float(triple_count),
                        "kv_count": float(kv_count),
                        "embedding_present": embedding_present,
                        "rationale": str(judge.get("rationale", "")).strip() if isinstance(judge, dict) else "",
                    },
                    "score": structure_score,
                    "gate": gate_open,
                    "decision": decision,
                }
            )

        trace = _copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "family": "scm_judge_gate",
            "policy": "threshold",
            "scorer": "llm_judge",
            "gate": "structured_gate",
            "output_field": "decisions",
            "decisions": decisions,
            "per_unit": per_unit,
        }
        return replace(packet, decisions=decisions, trace=trace), store


class SCMEntityProfileUpsert(OrganizationModule):
    """Append semantic records and upsert entity profiles.

    The normal write path lands in a semantic layer. In parallel, entity-backed
    profile rows are created or updated in a separate profile layer so SCM can do
    controlled retrieval against the latest entity state.
    """

    spec = ModuleSpec(
        name="scm_entity_profile_upsert",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, semantic_layer: str = "semantic", profile_layer: str = "profile") -> None:
        self.semantic_layer = semantic_layer
        self.profile_layer = profile_layer

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.semantic_layer):
            raise IncompatibleCompositionError(
                f"SCMEntityProfileUpsert requires declared semantic layer {self.semantic_layer!r}."
            )
        if not store.has_layer(self.profile_layer):
            raise IncompatibleCompositionError(
                f"SCMEntityProfileUpsert requires declared profile layer {self.profile_layer!r}."
            )
        if not store.layer_supports_index(self.profile_layer, "entity"):
            raise IncompatibleCompositionError(
                f"SCMEntityProfileUpsert requires entity index on profile layer {self.profile_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("SCMEntityProfileUpsert requires packet.units.")
        if packet.decisions is None:
            raise ValueError("SCMEntityProfileUpsert requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("SCMEntityProfileUpsert requires decisions aligned with units.")

        store.ensure_layer(self.semantic_layer)
        store.ensure_layer(self.profile_layer)

        placements = [Placement(unit_id=unit.unit_id, target_layer=self.semantic_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        profile_record_ids: list[str] = []
        accepted_unit_ids: list[str] = []
        skipped_units = 0

        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            if not decision:
                skipped_units += 1
                continue

            accepted_unit_ids.append(unit.unit_id)
            semantic_sequence_id = store.next_sequence_id()
            semantic_record = MemoryRecord.from_unit(unit=unit, layer=self.semantic_layer, sequence_id=semantic_sequence_id)
            semantic_record.metadata = {
                **semantic_record.metadata,
                "scm": {
                    "primary_entity": unit.entities[0] if unit.entities else None,
                    "entity_count": len(unit.entities),
                    "triple_count": len(unit.triples),
                    "kv_count": len(unit.kv),
                },
            }
            store.append(semantic_record)
            written_record_ids.append(semantic_record.record_id)

            primary_entity = unit.entities[0] if unit.entities else None
            if primary_entity is not None:
                profile_record = self._upsert_profile_record(
                    store,
                    entity=primary_entity,
                    unit=unit,
                    source_record_id=semantic_record.record_id,
                )
                profile_record_ids.append(profile_record.record_id)

        trace = _copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "semantic_layer": self.semantic_layer,
            "profile_layer": self.profile_layer,
            "written_record_ids": written_record_ids,
            "profile_record_ids": profile_record_ids,
            "written_unit_ids": accepted_unit_ids,
            "skipped_unit_count": skipped_units,
        }
        return replace(packet, placements=placements, trace=trace), store

    def _upsert_profile_record(
        self,
        store: MemoryStore,
        *,
        entity: str,
        unit: MemoryUnit,
        source_record_id: str,
    ) -> MemoryRecord:
        profile_records = store.layers[self.profile_layer]
        for index, existing in enumerate(profile_records):
            profile = existing.metadata.get("profile")
            if not isinstance(profile, dict):
                continue
            if str(profile.get("entity", "")).casefold() != entity.casefold():
                continue
            updated = self._merge_profile_record(existing, entity=entity, unit=unit, source_record_id=source_record_id)
            profile_records[index] = updated
            return updated

        record = MemoryRecord(
            record_id=f"rec-{store.next_sequence_id()}",
            unit_id=f"profile:{entity.casefold()}",
            layer=self.profile_layer,
            text=_summarize_fact(entity, unit.triples, unit.kv, unit.text),
            timestamp=unit.timestamp,
            metadata={
                **unit.metadata,
                "profile": self._profile_metadata(entity=entity, unit=unit, source_record_id=source_record_id),
            },
        )
        store.append(record)
        return record

    def _merge_profile_record(
        self,
        record: MemoryRecord,
        *,
        entity: str,
        unit: MemoryUnit,
        source_record_id: str,
    ) -> MemoryRecord:
        profile = record.metadata.get("profile")
        assert isinstance(profile, dict)
        source_unit_ids = _dedupe([*(str(item) for item in profile.get("source_unit_ids", [])), unit.unit_id])
        source_record_ids = _dedupe(
            [*(str(item) for item in profile.get("source_record_ids", [])), source_record_id]
        )
        aliases = _dedupe([*(str(item) for item in profile.get("aliases", [])), entity])
        triples = list(profile.get("triples", [])) if isinstance(profile.get("triples"), list) else []
        triples.extend(unit.triples)
        kv = dict(profile.get("kv", {})) if isinstance(profile.get("kv"), dict) else {}
        kv.update(unit.kv)
        facts = list(profile.get("facts", [])) if isinstance(profile.get("facts"), list) else []
        fact = _summarize_fact(entity, unit.triples, unit.kv, unit.text)
        if fact not in facts:
            facts.append(fact)
        metadata = {
            **record.metadata,
            "profile": {
                "entity": entity,
                "aliases": aliases,
                "source_unit_ids": source_unit_ids,
                "source_record_ids": source_record_ids,
                "facts": facts,
                "triples": _dedupe([str(item) for item in triples]),
                "kv": kv,
                "first_seen": profile.get("first_seen", unit.timestamp),
                "last_updated": unit.timestamp,
                "update_count": int(profile.get("update_count", 1)) + 1,
                "summary": _summarize_fact(entity, unit.triples, unit.kv, unit.text),
            },
        }
        text = metadata["profile"]["summary"]
        return replace(record, text=text, timestamp=unit.timestamp, metadata=metadata)

    def _profile_metadata(self, *, entity: str, unit: MemoryUnit, source_record_id: str) -> dict[str, Any]:
        return {
            "entity": entity,
            "aliases": [entity],
            "source_unit_ids": [unit.unit_id],
            "source_record_ids": [source_record_id],
            "facts": [_summarize_fact(entity, unit.triples, unit.kv, unit.text)],
            "triples": [tuple(triple) for triple in unit.triples],
            "kv": dict(unit.kv),
            "first_seen": unit.timestamp,
            "last_updated": unit.timestamp,
            "update_count": 1,
            "summary": _summarize_fact(entity, unit.triples, unit.kv, unit.text),
        }


class SCMControlledRetrieval(RetrievalModule):
    """Two-stage retrieval that prefers entity-backed profiles, then semantic facts."""

    spec = ModuleSpec(
        name="scm_controlled_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, top_k: int = 3, semantic_layer: str = "semantic", profile_layer: str = "profile") -> None:
        if top_k <= 0:
            raise ValueError("SCMControlledRetrieval requires top_k > 0.")
        self.top_k = int(top_k)
        self.semantic_layer = semantic_layer
        self.profile_layer = profile_layer

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.semantic_layer):
            raise IncompatibleCompositionError(
                f"SCMControlledRetrieval requires declared semantic layer {self.semantic_layer!r}."
            )
        if not store.has_layer(self.profile_layer):
            raise IncompatibleCompositionError(
                f"SCMControlledRetrieval requires declared profile layer {self.profile_layer!r}."
            )
        if not store.layer_supports_index(self.profile_layer, "entity"):
            raise IncompatibleCompositionError(
                f"SCMControlledRetrieval requires entity index on profile layer {self.profile_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("SCMControlledRetrieval requires packet.query.")

        query = packet.query
        query_entities = _query_entities(query)
        query_tokens = {token.casefold() for token in _TOKEN_PATTERN.findall(query.text)}

        profile_records = store.iter_records(self.profile_layer) if store.has_layer(self.profile_layer) else []
        semantic_records = store.iter_records(self.semantic_layer) if store.has_layer(self.semantic_layer) else []

        profile_candidates = self._score_records(
            profile_records,
            query_entities=query_entities,
            query_tokens=query_tokens,
            layer=self.profile_layer,
        )
        semantic_candidates = self._score_records(
            semantic_records,
            query_entities=query_entities,
            query_tokens=query_tokens,
            layer=self.semantic_layer,
        )

        has_entity_focus = bool(query_entities)
        candidates = profile_candidates + semantic_candidates
        if has_entity_focus and any(candidate["entity_overlap"] > 0 for candidate in candidates):
            candidates = [candidate for candidate in candidates if candidate["entity_overlap"] > 0]
            control_mode = "entity_first"
        else:
            control_mode = "semantic_rerank"

        reranked = get_classic_runtime().rerank(
            query=query.text,
            candidates=[
                {
                    "id": candidate["record"].record_id,
                    "layer": candidate["layer"],
                    "text": candidate["record"].text,
                    "entity_overlap": candidate["entity_overlap"],
                    "keyword_overlap": candidate["keyword_overlap"],
                    "entities": _record_entities(candidate["record"]),
                }
                for candidate in candidates
            ],
            task="SCM controlled retrieval with profile rows and semantic facts",
            top_k=self.top_k,
        )
        score_map = {item["id"]: item for item in reranked}
        candidates.sort(
            key=lambda item: (
                -float(score_map.get(item["record"].record_id, {}).get("score", 0.0)),
                self._sort_key(item),
            )
        )

        selected_records = []
        selected_scores = []
        seen_record_ids: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            record = candidate["record"]
            if record.record_id in seen_record_ids:
                continue
            seen_record_ids.add(record.record_id)
            selected_records.append(record)
            selected_scores.append(
                {
                    "record_id": record.record_id,
                    "rank": len(selected_records),
                    "score": float(score_map.get(record.record_id, {}).get("score", 0.0)),
                    "strategy": "entity_controlled" if candidate["entity_overlap"] > 0 else "keyword_recency",
                    "layer": candidate["layer"],
                    "entity_overlap": candidate["entity_overlap"],
                    "keyword_overlap": candidate["keyword_overlap"],
                    "rationale": str(score_map.get(record.record_id, {}).get("rationale", "")).strip(),
                }
            )
            if len(selected_records) >= self.top_k:
                break

        retrieved = RetrievedSet(
            items=selected_records,
            scores=selected_scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "query_entities": query_entities,
                "control_mode": control_mode,
                "candidate_count": len(candidates),
                "profile_candidate_count": len(profile_candidates),
                "semantic_candidate_count": len(semantic_candidates),
                "returned_count": len(selected_records),
            },
        )
        trace = _copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store

    def _score_records(
        self,
        records: list[Any],
        *,
        query_entities: list[str],
        query_tokens: set[str],
        layer: str,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        reversed_records = list(reversed(records))
        for recency_index, record in enumerate(reversed_records):
            record_entities = {entity.casefold() for entity in _record_entities(record)}
            entity_overlap = len({entity.casefold() for entity in query_entities} & record_entities)
            keyword_overlap = len(query_tokens & _record_tokens(record))
            candidates.append(
                {
                    "record": record,
                    "layer": layer,
                    "recency_index": recency_index,
                    "entity_overlap": entity_overlap,
                    "keyword_overlap": keyword_overlap,
                }
            )
        return candidates

    def _sort_key(self, candidate: dict[str, Any]) -> tuple[Any, ...]:
        return (
            -int(candidate["entity_overlap"]),
            -int(candidate["keyword_overlap"]),
            _layer_priority(candidate["layer"], profile_layer=self.profile_layer, semantic_layer=self.semantic_layer),
            int(candidate["recency_index"]),
        )


__all__ = [
    "SCMControlledRetrieval",
    "SCMEntityProfileUpsert",
    "SCMJudgeGateWrite",
    "SCMStructuredExtraction",
]
