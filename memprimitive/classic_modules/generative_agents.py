"""Generative Agents-style support primitives for the classic example."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
import re
from typing import Any, Final

from ..core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    ModuleSpec,
    Observation,
    Packet,
    Readout,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)
from ..exceptions import IncompatibleCompositionError
from ..interfaces import (
    EvolutionTriggerModule,
    MemoryEvolutionModule,
    ReadoutModule,
    RepresentationModule,
    RetrievalModule,
    UnitFormationModule,
    WriteTriggerModule,
)
from ..pipeline import MemoryPipeline
from ._runtime import ClassicRuntime, get_classic_runtime


_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
_ENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b"
)
_PREFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*|the user|user|they|she|he|we)\s+"
    r"(likes|prefers|loves|hates|wants|needs|remembers|studies|works on|cares about)\s+"
    r"([^.;,\n]+)",
    re.I,
)
_GOAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*|the user|user|they|she|he|we)\s+"
    r"(wants to|needs to|plans to|should|must)\s+([^.;,\n]+)",
    re.I,
)
_IMPORTANT_CUES: Final[frozenset[str]] = frozenset(
    {
        "important",
        "remember",
        "remembers",
        "prefer",
        "prefers",
        "like",
        "likes",
        "want",
        "wants",
        "love",
        "loves",
        "hate",
        "hates",
        "goal",
        "goals",
        "plan",
        "plans",
        "need",
        "needs",
        "must",
        "always",
        "never",
        "urgent",
        "focus",
        "focuses",
        "care",
        "cares",
        "reflect",
        "reflection",
    }
)
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "have",
        "has",
        "was",
        "were",
        "are",
        "is",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "or",
        "as",
        "it",
        "their",
        "they",
        "them",
        "we",
        "you",
        "your",
        "user",
        "theuser",
    }
)
_EXPLICIT_IDLE_TEXTS: Final[frozenset[str]] = frozenset(
    {
        "",
        "idle",
        "nothing happened",
        "nothing new",
        "no new information",
        "same as before",
    }
)


def _copy_trace(packet: Packet) -> dict[str, Any]:
    return dict(packet.trace)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOPWORDS
    ]


def _parse_iso_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)


def _sequence_number(record_id: str) -> int:
    match = re.search(r"-(\d+)$", record_id)
    return int(match.group(1)) if match else 0


def _representation_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe([str(item).strip() for item in value if str(item).strip()])


def _normalize_record_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe([str(item).strip() for item in value if str(item).strip()])


def _record_representation(record: MemoryRecord) -> dict[str, Any]:
    value = record.metadata.get("representation", {})
    return value if isinstance(value, dict) else {}


def _ga_dict_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    value = mapping.get("generative_agents", {})
    return dict(value) if isinstance(value, dict) else {}


def _ga_dict_from_record(record: MemoryRecord) -> dict[str, Any]:
    return _ga_dict_from_mapping(record.metadata)


def _update_ga_dict(mapping: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    return {**_ga_dict_from_mapping(mapping), **updates}


def _record_text(record: MemoryRecord) -> str:
    representation = _record_representation(record)
    summary = representation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return _normalize_text(summary)
    return _normalize_text(record.text)


def _extract_entities(text: str, *, hint: Any | None = None) -> list[str]:
    if isinstance(hint, list) and hint:
        return _dedupe([str(item).strip() for item in hint if str(item).strip()])
    result = get_classic_runtime().json(
        system="Extract salient named entities from autobiographical memory text. Return JSON with key entities.",
        user=f"text: {text}",
    )
    if isinstance(result, dict):
        return _dedupe([str(item).strip() for item in result.get("entities", []) if str(item).strip()])
    return []


def _extract_keywords(
    text: str,
    *,
    hint: Any | None = None,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    if isinstance(hint, list) and hint:
        return _dedupe([str(item).casefold().strip() for item in hint if str(item).strip()])
    result = get_classic_runtime().json(
        system="Extract compact retrieval keywords from a memory. Return JSON with key keywords.",
        user=f"text: {text}\nentities: {entities}\ntags: {tags}",
    )
    if isinstance(result, dict):
        return _dedupe([str(item).casefold().strip() for item in result.get("keywords", []) if str(item).strip()])
    return []


def _extract_tags(
    text: str,
    *,
    unit_type: str,
    entities: list[str],
    keywords: list[str],
    hint: Any | None = None,
) -> list[str]:
    if isinstance(hint, list) and hint:
        return _dedupe([str(item).casefold().strip() for item in hint if str(item).strip()])
    result = get_classic_runtime().json(
        system="Assign short semantic tags to a memory. Return JSON with key tags.",
        user=f"text: {text}\nunit_type: {unit_type}\nentities: {entities}\nkeywords: {keywords}",
    )
    if isinstance(result, dict):
        tags = [unit_type.casefold(), *[str(item).casefold().strip() for item in result.get("tags", [])]]
        return _dedupe(tags)
    return [unit_type.casefold()]


def _extract_summary(text: str, *, hint: Any | None = None) -> str:
    if isinstance(hint, str) and hint.strip():
        return _normalize_text(hint)
    return get_classic_runtime().text(
        system="Write a short factual summary for a generative-agent memory.",
        user=f"text: {text}",
    ).strip()


def _extract_salient_clause(text: str) -> str:
    for pattern in (_PREFERENCE_PATTERN, _GOAL_PATTERN):
        match = pattern.search(text)
        if match:
            subject, verb, obj = match.groups()
            return _normalize_text(f"{subject} {verb} {obj}").rstrip(".!?")
    collapsed = _normalize_text(text)
    if not collapsed:
        return "a recurring pattern"
    return " ".join(collapsed.split()[:14]).rstrip(",;:.")


def _extract_spo(text: str) -> tuple[str | None, str | None, str | None]:
    for pattern in (_PREFERENCE_PATTERN, _GOAL_PATTERN):
        match = pattern.search(text)
        if match:
            subject, predicate, obj = match.groups()
            return (
                _normalize_text(subject),
                _normalize_text(predicate),
                _normalize_text(obj).rstrip(".!?"),
            )
    return None, None, None


def _representation_summary_from_text(
    text: str,
    *,
    keywords: list[str],
    entities: list[str],
    tags: list[str],
    importance: float,
) -> dict[str, Any]:
    return {
        "text": text,
        "normalized_text": text.casefold(),
        "keywords": list(keywords),
        "entities": list(entities),
        "tags": list(tags),
        "importance": round(float(importance), 3),
    }


def _record_keywords(record: MemoryRecord) -> list[str]:
    representation = _record_representation(record)
    keywords = _representation_list(representation.get("keywords"))
    if keywords:
        return keywords
    ga_meta = _ga_dict_from_record(record)
    keywords = _representation_list(ga_meta.get("keywords"))
    if keywords:
        return keywords
    return _extract_keywords(record.text, entities=_record_entities(record))


def _record_entities(record: MemoryRecord) -> list[str]:
    representation = _record_representation(record)
    entities = _representation_list(representation.get("entities"))
    if entities:
        return entities
    ga_meta = _ga_dict_from_record(record)
    entities = _representation_list(ga_meta.get("entities"))
    if entities:
        return entities
    return _extract_entities(record.text)


def _record_tags(record: MemoryRecord) -> list[str]:
    representation = _record_representation(record)
    tags = _representation_list(representation.get("tags"))
    if tags:
        return tags
    ga_meta = _ga_dict_from_record(record)
    tags = _representation_list(ga_meta.get("tags"))
    if tags:
        return tags
    return _extract_tags(
        record.text,
        unit_type=str(record.metadata.get("unit_type", "event")),
        entities=_record_entities(record),
        keywords=_record_keywords(record),
    )


def _record_created_at(record: MemoryRecord) -> str:
    ga_meta = _ga_dict_from_record(record)
    created_at = ga_meta.get("created_at")
    if isinstance(created_at, str) and created_at.strip():
        return created_at
    return record.timestamp


def _record_last_accessed_at(record: MemoryRecord) -> str:
    ga_meta = _ga_dict_from_record(record)
    last_accessed_at = ga_meta.get("last_accessed_at")
    if isinstance(last_accessed_at, str) and last_accessed_at.strip():
        return last_accessed_at
    return _record_created_at(record)


def _set_record_last_accessed_at(record: MemoryRecord, timestamp: str) -> None:
    record.metadata["generative_agents"] = _update_ga_dict(
        record.metadata,
        {"last_accessed_at": timestamp},
    )


def _record_memory_type(record: MemoryRecord) -> str:
    ga_meta = _ga_dict_from_record(record)
    memory_type = str(ga_meta.get("memory_type", "")).strip()
    if memory_type:
        return memory_type
    unit_type = str(record.metadata.get("unit_type", "")).strip().casefold()
    if unit_type in {"thought", "reflection"}:
        return "thought"
    if unit_type == "chat":
        return "chat"
    return "event"


def _record_depth(record: MemoryRecord) -> int:
    ga_meta = _ga_dict_from_record(record)
    depth = ga_meta.get("depth")
    if isinstance(depth, int):
        return max(0, depth)
    if isinstance(depth, float):
        return max(0, int(depth))
    return 0 if _record_memory_type(record) == "event" else 1


def _record_evidence_ids(record: MemoryRecord) -> list[str]:
    ga_meta = _ga_dict_from_record(record)
    return _normalize_record_ids(ga_meta.get("evidence_record_ids"))


def _estimate_importance(
    *,
    text: str,
    unit_type: str,
    entities: list[str],
    keywords: list[str],
    tags: list[str],
    summary: str | None,
    existing_importance: Any | None = None,
) -> float:
    if isinstance(existing_importance, (int, float)) and not isinstance(existing_importance, bool):
        return max(0.0, min(float(existing_importance), 1.0))
    result = get_classic_runtime().json(
        system="Estimate memory importance for a generative agents system. Return JSON with key importance in [0,1].",
        user=(
            f"text: {text}\nunit_type: {unit_type}\nentities: {entities}\nkeywords: {keywords}\n"
            f"tags: {tags}\nsummary: {summary}"
        ),
    )
    raw = result.get("importance", 0.0) if isinstance(result, dict) else 0.0
    return max(0.0, min(float(raw), 1.0)) if isinstance(raw, (int, float)) else 0.0


def _record_importance(record: MemoryRecord) -> float:
    representation = _record_representation(record)
    ga_meta = _ga_dict_from_record(record)
    raw_importance = (
        ga_meta.get("importance")
        if "importance" in ga_meta
        else record.metadata.get("importance", representation.get("importance"))
    )
    if isinstance(raw_importance, (int, float)) and not isinstance(raw_importance, bool):
        return max(0.0, min(float(raw_importance), 1.0))
    return _estimate_importance(
        text=_record_text(record),
        unit_type=_record_memory_type(record),
        entities=_record_entities(record),
        keywords=_record_keywords(record),
        tags=_record_tags(record),
        summary=representation.get("summary") if isinstance(representation.get("summary"), str) else None,
    )


def _recent_context_records(
    source_records: list[MemoryRecord],
    *,
    source_record: MemoryRecord,
    window_size: int,
) -> list[MemoryRecord]:
    ordered = sorted(
        source_records,
        key=lambda record: (_parse_iso_timestamp(record.timestamp), _sequence_number(record.record_id)),
    )
    index = next((i for i, record in enumerate(ordered) if record.record_id == source_record.record_id), None)
    if index is None:
        return [source_record]
    start = max(0, index - max(1, window_size) + 1)
    return ordered[start : index + 1]


def _state(store: MemoryStore) -> dict[str, Any]:
    state = store.metadata.setdefault("generative_agents", {})
    if not isinstance(state, dict):
        state = {}
        store.metadata["generative_agents"] = state
    state.setdefault("importance_since_last_reflection", 0.0)
    state.setdefault("pending_event_record_ids", [])
    state.setdefault("last_reflection_event_seq", 0)
    state.setdefault("reflection_cycle_count", 0)
    state.setdefault("reflection_count", 0)
    return state


def _candidate_records(
    store: MemoryStore,
    *,
    layer: str | None = None,
    memory_type_filter: tuple[str, ...] | None = None,
) -> list[MemoryRecord]:
    records = store.iter_records(layer)
    if not memory_type_filter:
        return records
    allowed = {item.casefold() for item in memory_type_filter}
    return [record for record in records if _record_memory_type(record).casefold() in allowed]


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [1.0 if maximum > 0.0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _recency_signal(record: MemoryRecord) -> float:
    last_accessed = _parse_iso_timestamp(_record_last_accessed_at(record))
    created = _parse_iso_timestamp(_record_created_at(record))
    reference = last_accessed if last_accessed >= created else created
    age_seconds = max(0.0, (datetime.now(UTC) - reference).total_seconds())
    return 1.0 / (1.0 + age_seconds / 3600.0)


def _relevance_score(query_tokens: set[str], record: MemoryRecord) -> float:
    if not query_tokens:
        return 0.0
    query_embedding = get_classic_runtime().embed(" ".join(sorted(query_tokens)))
    record_embedding = record.embedding if record.embedding is not None else get_classic_runtime().embed(_record_text(record))
    return max(0.0, min(ClassicRuntime.cosine_similarity(query_embedding, record_embedding), 1.0))


def _score_records(
    query_text: str,
    records: list[MemoryRecord],
    *,
    relevance_weight: float,
    recency_weight: float,
    importance_weight: float,
) -> list[dict[str, Any]]:
    query_tokens = set(_tokenize(query_text))
    raw_rows: list[dict[str, Any]] = []
    for record in records:
        raw_rows.append(
            {
                "record": record,
                "raw_relevance": _relevance_score(query_tokens, record),
                "raw_recency": _recency_signal(record),
                "raw_importance": _record_importance(record),
            }
        )

    normalized_relevance = _normalize_scores([float(row["raw_relevance"]) for row in raw_rows])
    normalized_recency = _normalize_scores([float(row["raw_recency"]) for row in raw_rows])
    normalized_importance = _normalize_scores([float(row["raw_importance"]) for row in raw_rows])

    scored: list[dict[str, Any]] = []
    for row, relevance, recency, importance in zip(
        raw_rows,
        normalized_relevance,
        normalized_recency,
        normalized_importance,
        strict=True,
    ):
        total = (
            (relevance_weight * relevance)
            + (recency_weight * recency)
            + (importance_weight * importance)
        )
        scored.append(
            {
                "record": row["record"],
                "score": float(total),
                "relevance": float(relevance),
                "recency": float(recency),
                "importance": float(importance),
                "raw_relevance": float(row["raw_relevance"]),
                "raw_recency": float(row["raw_recency"]),
                "raw_importance": float(row["raw_importance"]),
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["relevance"]),
            -float(item["importance"]),
            -float(item["recency"]),
            -_sequence_number(item["record"].record_id),
        )
    )
    return scored


def _generate_focal_points(pending_records: list[MemoryRecord], *, focal_point_count: int) -> list[str]:
    payload = json.dumps(
        {
            "records": [
                {
                    "record_id": record.record_id,
                    "text": _record_text(record),
                    "keywords": _record_keywords(record),
                    "importance": _record_importance(record),
                }
                for record in pending_records
            ],
            "focal_point_count": focal_point_count,
        },
        ensure_ascii=False,
    )
    try:
        result = get_classic_runtime().json(
            system=(
                "Generate focal points for a Generative Agents reflection cycle. "
                "Return strict JSON with key focal_points as a list of concise strings."
            ),
            user=payload,
        )
        if isinstance(result, dict):
            focal_points = _dedupe(
                [
                    _normalize_text(str(item))
                    for item in result.get("focal_points", [])
                    if _normalize_text(str(item))
                ]
            )
            if focal_points:
                return focal_points[: max(1, focal_point_count)]
    except Exception:
        pass

    fallback = _dedupe([_extract_salient_clause(_record_text(record)) for record in pending_records])
    return fallback[: max(1, focal_point_count)]


def _fallback_insight_text(focal_point: str, evidence_records: list[MemoryRecord]) -> str:
    summary = get_classic_runtime().summarize_records(
        records=[
            {
                "record_id": record.record_id,
                "text": _record_text(record),
                "memory_type": _record_memory_type(record),
            }
            for record in evidence_records
        ],
        instruction=(
            "Infer one concise memory insight from these memories. "
            "Begin with 'Insight:' and emphasize a stable pattern when possible."
        ),
        max_sentences=2,
    ).strip()
    summary = _normalize_text(summary)
    if not summary:
        summary = f"Insight: {focal_point}"
    if not summary.lower().startswith("insight:"):
        summary = f"Insight: {summary}"
    return summary


def _generate_insights(
    focal_point: str,
    evidence_records: list[MemoryRecord],
    *,
    insights_per_cycle: int,
) -> list[dict[str, Any]]:
    by_id = {record.record_id: record for record in evidence_records}
    payload = json.dumps(
        {
            "focal_point": focal_point,
            "insights_per_cycle": insights_per_cycle,
            "records": [
                {
                    "record_id": record.record_id,
                    "text": _record_text(record),
                    "memory_type": _record_memory_type(record),
                    "importance": _record_importance(record),
                }
                for record in evidence_records
            ],
        },
        ensure_ascii=False,
    )
    try:
        result = get_classic_runtime().json(
            system=(
                "Create Generative Agents reflection insights. "
                "Return strict JSON with key insights as a list of objects. "
                "Each object must contain text and evidence_record_ids."
            ),
            user=payload,
        )
        if isinstance(result, dict):
            normalized: list[dict[str, Any]] = []
            for item in result.get("insights", []):
                if not isinstance(item, dict):
                    continue
                text = _normalize_text(str(item.get("text", "")))
                evidence_ids = [
                    record_id
                    for record_id in _normalize_record_ids(item.get("evidence_record_ids"))
                    if record_id in by_id
                ]
                if not text or not evidence_ids:
                    continue
                if not text.lower().startswith("insight:"):
                    text = f"Insight: {text}"
                normalized.append(
                    {
                        "text": text,
                        "evidence_record_ids": evidence_ids,
                    }
                )
                if len(normalized) >= max(1, insights_per_cycle):
                    break
            if normalized:
                return normalized
    except Exception:
        pass

    fallback_records = evidence_records[: max(1, min(len(evidence_records), insights_per_cycle))]
    return [
        {
            "text": _fallback_insight_text(focal_point, fallback_records),
            "evidence_record_ids": [record.record_id for record in fallback_records],
        }
    ]


class GenerativeAgentsUnitFormation(UnitFormationModule):
    """One observation becomes one memory unit in the observation stream."""

    spec = ModuleSpec(
        name="generative_agents_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("GenerativeAgentsUnitFormation requires packet.observation.")
        observation = packet.observation
        text = _normalize_text(observation.text)
        unit = MemoryUnit(
            text=text,
            unit_type="event",
            timestamp=observation.timestamp,
            normalized_text=text.casefold(),
            metadata={
                **observation.metadata,
                "source": observation.source,
                "provenance": {
                    "observation_id": observation.observation_id,
                    "source": observation.source,
                },
            },
        )
        trace = _copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id],
            "source": observation.source,
        }
        return replace(packet, units=[unit], trace=trace), store


class GenerativeAgentsRepresentation(RepresentationModule):
    """Generate event-style memory metadata for Generative Agents observations."""

    spec = ModuleSpec(
        name="generative_agents_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=(
            "units.text",
            "units.representation_elements",
            "units.normalized_text",
            "units.metadata.representation",
            "units.metadata.importance",
        ),
    )

    _valid_elements: Final[tuple[str, ...]] = ("text", "keywords", "summary", "entities", "tags")

    def __init__(
        self,
        *,
        elements: tuple[str, ...] = ("text", "keywords", "summary", "entities", "tags"),
    ) -> None:
        normalized = tuple(dict.fromkeys(elements))
        unsupported = [element for element in normalized if element not in self._valid_elements]
        if unsupported:
            raise ValueError(f"Unsupported generative-agents representation element(s): {unsupported}.")
        self.elements = normalized

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GenerativeAgentsRepresentation requires packet.units.")
        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            represented_unit = self._represent_unit(unit)
            represented_units.append(represented_unit)
            per_unit_trace.append(
                {
                    "unit_id": represented_unit.unit_id,
                    "elements": list(represented_unit.representation_elements),
                    "importance": represented_unit.metadata.get("importance"),
                    "memory_type": _ga_dict_from_mapping(represented_unit.metadata).get("memory_type"),
                }
            )
        trace = _copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in represented_units],
            "elements": list(self.elements),
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _represent_unit(self, unit: MemoryUnit) -> MemoryUnit:
        normalized_text = _normalize_text(unit.text)
        entities = _extract_entities(normalized_text, hint=unit.metadata.get("entities"))
        tags = _extract_tags(
            normalized_text,
            unit_type="event",
            entities=entities,
            keywords=[],
            hint=unit.metadata.get("tags"),
        )
        keywords = _extract_keywords(
            normalized_text,
            hint=unit.metadata.get("keywords"),
            entities=entities,
            tags=tags,
        )
        summary = _extract_summary(normalized_text, hint=unit.metadata.get("summary"))
        importance = _estimate_importance(
            text=normalized_text,
            unit_type="event",
            entities=entities,
            keywords=keywords,
            tags=tags,
            summary=summary,
            existing_importance=unit.metadata.get("importance"),
        )
        subject, predicate, obj = _extract_spo(normalized_text)
        ga_meta = {
            "memory_type": "event",
            "created_at": unit.timestamp,
            "last_accessed_at": unit.timestamp,
            "importance": importance,
            "poignancy": importance,
            "keywords": keywords,
            "entities": entities,
            "tags": tags,
            "evidence_record_ids": [],
            "depth": 0,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
        }
        representation = _representation_summary_from_text(
            normalized_text,
            keywords=keywords,
            entities=entities,
            tags=tags,
            importance=importance,
        )
        if "summary" in self.elements and summary:
            representation["summary"] = summary
        represented = replace(
            unit,
            text=normalized_text,
            unit_type="event",
            representation_elements=tuple(sorted(self.elements)),
            normalized_text=normalized_text.casefold(),
            entities=entities,
            tags=tags,
            metadata={
                **unit.metadata,
                "importance": importance,
                "generative_agents": ga_meta,
                "representation": {
                    **representation,
                    "summary": summary,
                },
            },
        )
        return represented


class GenerativeAgentsWriteTrigger(WriteTriggerModule):
    """Lightweight perceive-style write filter with duplicate suppression."""

    spec = ModuleSpec(
        name="generative_agents_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def __init__(
        self,
        *,
        source_layer: str = "observation_stream",
        duplicate_window: int = 5,
    ) -> None:
        if duplicate_window <= 0:
            raise ValueError("GenerativeAgentsWriteTrigger requires duplicate_window > 0.")
        self.source_layer = source_layer
        self.duplicate_window = int(duplicate_window)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GenerativeAgentsWriteTrigger requires packet.units.")

        recent_records: list[MemoryRecord] = []
        if store.has_layer(self.source_layer):
            recent_records = _candidate_records(
                store,
                layer=self.source_layer,
                memory_type_filter=("event",),
            )[-self.duplicate_window :]
        recent_texts = {
            (_record_representation(record).get("normalized_text") or _record_text(record).casefold()).strip()
            for record in recent_records
        }

        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            normalized_text = _normalize_text(unit.text)
            normalized_key = normalized_text.casefold()
            tokens = set(_tokenize(normalized_text))
            cue_hits = sorted(tokens & _IMPORTANT_CUES)
            has_entity = bool(unit.entities or _extract_entities(normalized_text, hint=unit.metadata.get("entities")))

            if normalized_key in _EXPLICIT_IDLE_TEXTS:
                decisions.append(False)
                per_unit.append({"unit_id": unit.unit_id, "decision": False, "reason": "idle_text"})
                continue
            if normalized_key in recent_texts:
                decisions.append(False)
                per_unit.append({"unit_id": unit.unit_id, "decision": False, "reason": "duplicate_recent_event"})
                continue
            if len(tokens) <= 1 and not cue_hits and not has_entity:
                decisions.append(False)
                per_unit.append({"unit_id": unit.unit_id, "decision": False, "reason": "low_information"})
                continue

            decisions.append(True)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "decision": True,
                    "reason": "novel_event",
                    "cue_hits": cue_hits,
                }
            )

        trace = _copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "source_layer": self.source_layer,
            "duplicate_window": self.duplicate_window,
            "per_unit": per_unit,
        }
        return replace(packet, decisions=decisions, trace=trace), store


class GenerativeAgentsReflectionTrigger(EvolutionTriggerModule):
    """Trigger reflection when cumulative event importance crosses a threshold."""

    spec = ModuleSpec(
        name="generative_agents_reflection_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "decisions", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(
        self,
        *,
        source_layer: str = "observation_stream",
        reflection_layer: str = "reflections",
        reflection_threshold: float = 0.8,
        reflection_batch_size: int = 3,
        context_window: int = 4,
    ) -> None:
        if reflection_batch_size <= 0:
            raise ValueError("GenerativeAgentsReflectionTrigger requires reflection_batch_size > 0.")
        self.source_layer = source_layer
        self.reflection_layer = reflection_layer
        self.reflection_threshold = float(reflection_threshold)
        self.reflection_batch_size = int(reflection_batch_size)
        self.context_window = int(context_window)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GenerativeAgentsReflectionTrigger requires packet.units.")
        if packet.decisions is None:
            raise ValueError("GenerativeAgentsReflectionTrigger requires packet.decisions.")
        if packet.placements is None:
            raise ValueError("GenerativeAgentsReflectionTrigger requires packet.placements.")
        if not (len(packet.units) == len(packet.decisions) == len(packet.placements)):
            raise ValueError(
                "GenerativeAgentsReflectionTrigger requires aligned units, decisions, and placements."
            )

        source_records = store.iter_records(self.source_layer) if store.has_layer(self.source_layer) else []
        source_by_unit_id = {record.unit_id: record for record in source_records}
        state = _state(store)
        pending_ids = _normalize_record_ids(state.get("pending_event_record_ids"))
        pending_id_set = set(pending_ids)
        importance_since_last_reflection = float(state.get("importance_since_last_reflection", 0.0) or 0.0)

        per_unit: list[dict[str, Any]] = []
        newly_added_record_ids: list[str] = []
        for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True):
            if not decision:
                per_unit.append(
                    {
                        "unit_id": unit.unit_id,
                        "target_layer": placement.target_layer,
                        "decision": False,
                        "reason": "not_written",
                    }
                )
                continue

            record = source_by_unit_id.get(unit.unit_id)
            if record is None or placement.target_layer != self.source_layer or _record_memory_type(record) != "event":
                per_unit.append(
                    {
                        "unit_id": unit.unit_id,
                        "target_layer": placement.target_layer,
                        "decision": False,
                        "reason": "not_source_event",
                    }
                )
                continue

            if record.record_id not in pending_id_set:
                pending_ids.append(record.record_id)
                pending_id_set.add(record.record_id)
                newly_added_record_ids.append(record.record_id)
                importance_since_last_reflection += _record_importance(record)

            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "target_layer": placement.target_layer,
                    "decision": False,
                    "reason": "queued_for_reflection",
                    "importance": _record_importance(record),
                    "record_id": record.record_id,
                }
            )

        cycle_triggered = bool(pending_ids) and importance_since_last_reflection >= self.reflection_threshold
        selected_record_ids = pending_ids[: self.reflection_batch_size] if cycle_triggered else []
        selected_unit_ids = {
            record.unit_id
            for record in source_records
            if record.record_id in set(selected_record_ids)
        }
        decisions = [
            bool(decision and unit.unit_id in selected_unit_ids)
            for unit, decision in zip(packet.units, packet.decisions, strict=True)
        ]

        state["importance_since_last_reflection"] = importance_since_last_reflection
        state["pending_event_record_ids"] = pending_ids
        if cycle_triggered:
            state["active_reflection_record_ids"] = list(selected_record_ids)

        trace = _copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "source_layer": self.source_layer,
            "reflection_layer": self.reflection_layer,
            "reflection_threshold": self.reflection_threshold,
            "reflection_batch_size": self.reflection_batch_size,
            "importance_since_last_reflection": importance_since_last_reflection,
            "newly_added_record_ids": newly_added_record_ids,
            "pending_event_record_ids": list(pending_ids),
            "selected_record_ids": list(selected_record_ids),
            "selected_unit_ids": sorted(selected_unit_ids),
            "triggered_cycle": cycle_triggered,
            "per_unit": per_unit,
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class GenerativeAgentsMemoryEvolution(MemoryEvolutionModule):
    """Generate thought memories from focal points and supporting evidence."""

    spec = ModuleSpec(
        name="generative_agents_memory_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        source_layer: str = "observation_stream",
        reflection_layer: str = "reflections",
        context_window: int = 4,
        focal_point_count: int = 3,
        insights_per_cycle: int = 2,
    ) -> None:
        if focal_point_count <= 0:
            raise ValueError("GenerativeAgentsMemoryEvolution requires focal_point_count > 0.")
        if insights_per_cycle <= 0:
            raise ValueError("GenerativeAgentsMemoryEvolution requires insights_per_cycle > 0.")
        self.source_layer = source_layer
        self.reflection_layer = reflection_layer
        self.context_window = int(context_window)
        self.focal_point_count = int(focal_point_count)
        self.insights_per_cycle = int(insights_per_cycle)

    def validate_store(self, store: MemoryStore) -> None:
        for layer_name in (self.source_layer, self.reflection_layer):
            if not store.has_layer(layer_name):
                raise IncompatibleCompositionError(
                    f"GenerativeAgentsMemoryEvolution requires declared layer {layer_name!r}."
                )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("GenerativeAgentsMemoryEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("GenerativeAgentsMemoryEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("GenerativeAgentsMemoryEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError(
                "GenerativeAgentsMemoryEvolution requires aligned units, placements, and evolution decisions."
            )

        state = _state(store)
        pending_event_ids = _normalize_record_ids(state.get("pending_event_record_ids"))
        active_event_ids = _normalize_record_ids(state.get("active_reflection_record_ids"))
        if not active_event_ids:
            active_event_ids = list(pending_event_ids)

        source_records = store.iter_records(self.source_layer)
        source_by_id = {record.record_id: record for record in source_records}
        pending_records = [source_by_id[record_id] for record_id in active_event_ids if record_id in source_by_id]

        if not pending_records:
            trace = _copy_trace(packet)
            trace["memory_evolution"] = {
                "module": self.spec.name,
                "decision_source": "evolution_decisions",
                "triggered_cycle": False,
                "focal_points": [],
                "insight_count": 0,
                "thought_record_ids": [],
                "effects": [],
            }
            return replace(packet, trace=trace), store

        focal_points = _generate_focal_points(
            pending_records,
            focal_point_count=self.focal_point_count,
        )
        all_records = _candidate_records(
            store,
            layer=None,
            memory_type_filter=("event", "thought"),
        )

        effects: list[dict[str, Any]] = []
        thought_record_ids: list[str] = []
        insight_count = 0
        used_evidence_ids: set[str] = set()

        for focal_point in focal_points:
            scored_candidates = _score_records(
                focal_point,
                all_records,
                relevance_weight=0.5,
                recency_weight=3.0,
                importance_weight=2.0,
            )
            evidence_records = [
                item["record"]
                for item in scored_candidates[: max(self.context_window, self.insights_per_cycle + 1)]
            ]
            if not evidence_records:
                continue
            insights = _generate_insights(
                focal_point,
                evidence_records,
                insights_per_cycle=self.insights_per_cycle,
            )
            for insight in insights:
                evidence_ids = [
                    record_id
                    for record_id in _normalize_record_ids(insight.get("evidence_record_ids"))
                    if record_id in {record.record_id for record in evidence_records}
                ]
                if not evidence_ids:
                    evidence_ids = [record.record_id for record in evidence_records[:1]]
                evidence_records_for_depth = [
                    record
                    for record in all_records
                    if record.record_id in set(evidence_ids)
                ]
                depth = 1 + max((_record_depth(record) for record in evidence_records_for_depth), default=0)
                insight_text = _normalize_text(str(insight.get("text", "")))
                if not insight_text:
                    continue
                if not insight_text.lower().startswith("insight:"):
                    insight_text = f"Insight: {insight_text}"
                importance = min(
                    1.0,
                    max((_record_importance(record) for record in evidence_records_for_depth), default=0.4) + 0.1,
                )
                keywords = _dedupe(
                    [keyword for record in evidence_records_for_depth for keyword in _record_keywords(record)]
                )
                entities = _dedupe(
                    [entity for record in evidence_records_for_depth for entity in _record_entities(record)]
                )
                tags = _dedupe(
                    ["thought", "reflection", "insight", *[tag for record in evidence_records_for_depth for tag in _record_tags(record)]]
                )
                subject, predicate, obj = _extract_spo(insight_text)
                now_timestamp = _utc_now_iso()
                thought_unit = MemoryUnit(
                    text=insight_text,
                    unit_type="thought",
                    representation_elements=("text", "summary", "keywords", "entities", "tags"),
                    timestamp=now_timestamp,
                    normalized_text=insight_text.casefold(),
                    entities=entities,
                    tags=tags,
                    metadata={
                        "importance": importance,
                        "source_record_ids": evidence_ids,
                        "representation": {
                            **_representation_summary_from_text(
                                insight_text,
                                keywords=keywords,
                                entities=entities,
                                tags=tags,
                                importance=importance,
                            ),
                            "summary": insight_text,
                        },
                        "generative_agents": {
                            "memory_type": "thought",
                            "created_at": now_timestamp,
                            "last_accessed_at": now_timestamp,
                            "importance": importance,
                            "poignancy": importance,
                            "keywords": keywords,
                            "entities": entities,
                            "tags": tags,
                            "evidence_record_ids": evidence_ids,
                            "depth": depth,
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "reflection_origin": "focal_point_cycle",
                            "focal_point": focal_point,
                        },
                    },
                )
                sequence_id = store.next_sequence_id()
                record = MemoryRecord.from_unit(
                    thought_unit,
                    layer=self.reflection_layer,
                    sequence_id=sequence_id,
                )
                store.append(record)
                thought_record_ids.append(record.record_id)
                used_evidence_ids.update(evidence_ids)
                insight_count += 1
                effects.append(
                    {
                        "effect_type": "thought_append",
                        "thought_record_id": record.record_id,
                        "target_layer": self.reflection_layer,
                        "focal_point": focal_point,
                        "evidence_record_ids": evidence_ids,
                        "depth": depth,
                    }
                )

        max_reflected_seq = max((_sequence_number(record.record_id) for record in pending_records), default=0)
        state["last_reflection_event_seq"] = max(
            int(state.get("last_reflection_event_seq", 0) or 0),
            max_reflected_seq,
        )
        state["reflection_cycle_count"] = int(state.get("reflection_cycle_count", 0) or 0) + 1
        state["reflection_count"] = int(state.get("reflection_count", 0) or 0) + len(thought_record_ids)
        state["importance_since_last_reflection"] = 0.0
        state["pending_event_record_ids"] = [
            record_id for record_id in pending_event_ids if record_id not in set(active_event_ids)
        ]
        state["active_reflection_record_ids"] = []

        trace = _copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "triggered_cycle": True,
            "focal_points": focal_points,
            "active_event_record_ids": active_event_ids,
            "used_evidence_record_ids": sorted(used_evidence_ids),
            "insight_count": insight_count,
            "thought_record_ids": thought_record_ids,
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class GenerativeAgentsRetrieval(RetrievalModule):
    """Rank memories with normalized relevance, recency, and importance."""

    spec = ModuleSpec(
        name="generative_agents_weighted_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(
        self,
        *,
        top_k: int = 5,
        layer: str | None = None,
        relevance_weight: float = 0.5,
        recency_weight: float = 3.0,
        importance_weight: float = 2.0,
        memory_type_filter: tuple[str, ...] | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("GenerativeAgentsRetrieval requires top_k > 0.")
        self.top_k = int(top_k)
        self.layer = layer
        self.relevance_weight = float(relevance_weight)
        self.recency_weight = float(recency_weight)
        self.importance_weight = float(importance_weight)
        self.memory_type_filter = memory_type_filter

    def validate_store(self, store: MemoryStore) -> None:
        if self.layer is not None and not store.has_layer(self.layer):
            raise IncompatibleCompositionError(
                f"GenerativeAgentsRetrieval requires declared layer {self.layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("GenerativeAgentsRetrieval requires packet.query.")

        candidates = _candidate_records(
            store,
            layer=self.layer,
            memory_type_filter=self.memory_type_filter,
        )
        if not candidates:
            retrieved = RetrievedSet(
                items=[],
                scores=[],
                trace={
                    "module": self.spec.name,
                    "top_k": self.top_k,
                    "candidate_count": 0,
                },
            )
            trace = _copy_trace(packet)
            trace["retrieval"] = retrieved.trace
            return replace(packet, retrieved=retrieved, trace=trace), store

        scored = _score_records(
            packet.query.text,
            candidates,
            relevance_weight=self.relevance_weight,
            recency_weight=self.recency_weight,
            importance_weight=self.importance_weight,
        )
        selected = scored[: self.top_k]
        access_timestamp = packet.query.timestamp or _utc_now_iso()
        for item in selected:
            _set_record_last_accessed_at(item["record"], access_timestamp)

        items = [item["record"] for item in selected]
        scores = [
            {
                "record_id": item["record"].record_id,
                "rank": rank,
                "score": float(item["score"]),
                "relevance": float(item["relevance"]),
                "recency": float(item["recency"]),
                "importance": float(item["importance"]),
                "raw_relevance": float(item["raw_relevance"]),
                "raw_recency": float(item["raw_recency"]),
                "raw_importance": float(item["raw_importance"]),
                "memory_type": _record_memory_type(item["record"]),
                "strategy": "normalized_relevance_recency_importance",
            }
            for rank, item in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "candidate_count": len(candidates),
                "layer": self.layer,
                "memory_type_filter": list(self.memory_type_filter) if self.memory_type_filter else None,
                "weights": {
                    "relevance": self.relevance_weight,
                    "recency": self.recency_weight,
                    "importance": self.importance_weight,
                },
                "updated_last_accessed_record_ids": [item["record"].record_id for item in selected],
            },
        )
        trace = _copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


def build_generative_agents_topology(
    *,
    observation_layer: str = "observation_stream",
    reflection_layer: str = "reflections",
) -> StoreTopology:
    return StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name=observation_layer,
                theme="working",
                capacity="sliding_window",
                indices=("temporal", "keyword", "vector"),
            ),
            StoreLayerSpec(
                name=reflection_layer,
                theme="semantic",
                indices=("temporal", "keyword", "vector"),
            ),
        ]
    )


class GenerativeAgentsReadout(ReadoutModule):
    """Compatibility readout that concatenates retrieval texts."""

    spec = ModuleSpec(
        name="generative_agents_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, separator: str = "\n\n") -> None:
        self.separator = separator

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("GenerativeAgentsReadout requires packet.retrieved.")

        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        readout = Readout(
            text=self.separator.join(record.text for record in items),
            source_ids=source_ids,
            metadata={"item_count": len(items), "format": "generative_agents"},
        )
        trace = _copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
        }
        return replace(packet, readout=readout, trace=trace), store


from ..baselines import AppendOrganization


def build_generative_agents_pipeline(
    *,
    store: MemoryStore | None = None,
    top_k: int = 5,
    observation_layer: str = "observation_stream",
    reflection_layer: str = "reflections",
    reflection_threshold: float = 0.8,
    reflection_batch_size: int = 3,
    context_window: int = 4,
    relevance_weight: float = 0.5,
    recency_weight: float = 3.0,
    importance_weight: float = 2.0,
    focal_point_count: int = 3,
    insights_per_cycle: int = 2,
    duplicate_window: int = 5,
) -> MemoryPipeline:
    memory_store = (
        store
        if store is not None
        else MemoryStore(
            topology=build_generative_agents_topology(
                observation_layer=observation_layer,
                reflection_layer=reflection_layer,
            )
        )
    )
    return MemoryPipeline(
        store=memory_store,
        unit_formation=GenerativeAgentsUnitFormation(),
        representation=GenerativeAgentsRepresentation(),
        write_trigger=GenerativeAgentsWriteTrigger(
            source_layer=observation_layer,
            duplicate_window=duplicate_window,
        ),
        organization=AppendOrganization(target_layer=observation_layer),
        evolution_trigger=GenerativeAgentsReflectionTrigger(
            source_layer=observation_layer,
            reflection_layer=reflection_layer,
            reflection_threshold=reflection_threshold,
            reflection_batch_size=reflection_batch_size,
            context_window=context_window,
        ),
        memory_evolution=GenerativeAgentsMemoryEvolution(
            source_layer=observation_layer,
            reflection_layer=reflection_layer,
            context_window=context_window,
            focal_point_count=focal_point_count,
            insights_per_cycle=insights_per_cycle,
        ),
        retrieval=GenerativeAgentsRetrieval(
            top_k=top_k,
            layer=None,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        ),
        readout=GenerativeAgentsReadout(separator="\n\n"),
    )


__all__ = [
    "GenerativeAgentsMemoryEvolution",
    "GenerativeAgentsReadout",
    "GenerativeAgentsReflectionTrigger",
    "GenerativeAgentsRepresentation",
    "GenerativeAgentsRetrieval",
    "GenerativeAgentsUnitFormation",
    "GenerativeAgentsWriteTrigger",
    "build_generative_agents_pipeline",
    "build_generative_agents_topology",
]
