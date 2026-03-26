"""Generative Agents-style support primitives for the classic example."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
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


def _copy_trace(packet: Packet) -> dict[str, Any]:
    return dict(packet.trace)


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


def _record_representation(record: MemoryRecord) -> dict[str, Any]:
    value = record.metadata.get("representation", {})
    return value if isinstance(value, dict) else {}


def _record_text(record: MemoryRecord) -> str:
    representation = _record_representation(record)
    summary = representation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return _normalize_text(summary)
    return _normalize_text(record.text)


def _record_keywords(record: MemoryRecord) -> list[str]:
    representation = _record_representation(record)
    keywords = _representation_list(representation.get("keywords"))
    if keywords:
        return keywords
    return _extract_keywords(record.text, entities=_record_entities(record))


def _record_entities(record: MemoryRecord) -> list[str]:
    representation = _record_representation(record)
    entities = _representation_list(representation.get("entities"))
    if entities:
        return entities
    return _extract_entities(record.text)


def _record_tags(record: MemoryRecord) -> list[str]:
    representation = _record_representation(record)
    tags = _representation_list(representation.get("tags"))
    if tags:
        return tags
    return _extract_tags(
        record.text,
        unit_type=str(record.metadata.get("unit_type", "observation")),
        entities=_record_entities(record),
        keywords=_record_keywords(record),
    )


def _record_importance(record: MemoryRecord) -> float:
    representation = _record_representation(record)
    raw_importance = record.metadata.get("importance", representation.get("importance"))
    if isinstance(raw_importance, (int, float)) and not isinstance(raw_importance, bool):
        return max(0.0, min(float(raw_importance), 1.0))
    return _estimate_importance(
        text=_record_text(record),
        unit_type=str(record.metadata.get("unit_type", "observation")),
        entities=_record_entities(record),
        keywords=_record_keywords(record),
        tags=_record_tags(record),
        summary=representation.get("summary") if isinstance(representation.get("summary"), str) else None,
    )


def _relevance_score(query_tokens: set[str], record: MemoryRecord) -> float:
    if not query_tokens:
        return 0.0
    query_embedding = get_classic_runtime().embed(" ".join(sorted(query_tokens)))
    record_embedding = record.embedding if record.embedding is not None else get_classic_runtime().embed(_record_text(record))
    return max(0.0, min(ClassicRuntime.cosine_similarity(query_embedding, record_embedding), 1.0))


def _build_reflection_text(context_records: list[MemoryRecord]) -> str:
    return get_classic_runtime().summarize_records(
        records=[
            {"record_id": record.record_id, "text": record.text, "keywords": _record_keywords(record)}
            for record in context_records
        ],
        instruction="Write one generative-agents reflection beginning with 'Reflection:'.",
        max_sentences=2,
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
    return state


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
            unit_type="observation",
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
    """Heuristic representation with summary and importance metadata."""

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
            unit_type=unit.unit_type,
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
            unit_type=unit.unit_type,
            entities=entities,
            keywords=keywords,
            tags=tags,
            summary=summary,
            existing_importance=unit.metadata.get("importance"),
        )
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
            representation_elements=tuple(sorted(self.elements)),
            normalized_text=normalized_text.casefold(),
            entities=entities,
            tags=tags,
            metadata={
                **unit.metadata,
                "importance": importance,
                "representation": {
                    **representation,
                    "summary": summary,
                },
            },
        )
        return represented


class GenerativeAgentsReflectionTrigger(EvolutionTriggerModule):
    """Select written observations that are salient enough to trigger reflection."""

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
        reflection_threshold: float = 0.55,
        reflection_batch_size: int = 2,
        context_window: int = 3,
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
        latest_reflected_seq = int(_state(store).get("last_reflected_source_seq", 0) or 0)

        candidates: list[dict[str, Any]] = []
        per_unit: list[dict[str, Any]] = []
        for unit, decision, placement in zip(packet.units, packet.decisions, packet.placements, strict=True):
            if not decision:
                per_unit.append(
                    {
                        "unit_id": unit.unit_id,
                        "target_layer": placement.target_layer,
                        "decision": False,
                        "reason": "not_written",
                        "importance": 0.0,
                        "score": 0.0,
                    }
                )
                continue
            record = source_by_unit_id.get(unit.unit_id)
            if record is None:
                per_unit.append(
                    {
                        "unit_id": unit.unit_id,
                        "target_layer": placement.target_layer,
                        "decision": False,
                        "reason": "missing_source_record",
                        "importance": 0.0,
                        "score": 0.0,
                    }
                )
                continue
            seq = _sequence_number(record.record_id)
            if seq <= latest_reflected_seq:
                per_unit.append(
                    {
                        "unit_id": unit.unit_id,
                        "target_layer": placement.target_layer,
                        "decision": False,
                        "reason": "already_reflected",
                        "importance": _record_importance(record),
                        "score": 0.0,
                    }
                )
                continue
            importance = _record_importance(record)
            cue_hits = len(set(_tokenize(record.text)) & _IMPORTANT_CUES)
            if cue_hits == 0 and importance < self.reflection_threshold:
                per_unit.append(
                    {
                        "unit_id": unit.unit_id,
                        "target_layer": placement.target_layer,
                        "decision": False,
                        "reason": "below_reflection_threshold",
                        "importance": importance,
                        "score": 0.0,
                    }
                )
                continue
            context = _recent_context_records(
                source_records,
                source_record=record,
                window_size=self.context_window,
            )
            context_keywords = _dedupe(
                [keyword for candidate in context for keyword in _record_keywords(candidate)]
            )
            context_entities = _dedupe(
                [entity for candidate in context for entity in _record_entities(candidate)]
            )
            salience = 0.0
            if context_keywords:
                salience += 0.1 * min(len(context_keywords), 4)
            if context_entities:
                salience += 0.08 * min(len(context_entities), 3)
            if _extract_salient_clause(record.text) != _normalize_text(record.text):
                salience += 0.1
            score = min(1.0, (0.7 * importance) + salience)
            candidates.append(
                {
                    "unit_id": unit.unit_id,
                    "record_id": record.record_id,
                    "sequence": seq,
                    "importance": importance,
                    "score": score,
                }
            )
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "target_layer": placement.target_layer,
                    "decision": False,
                    "reason": "candidate",
                    "importance": importance,
                    "score": score,
                }
            )

        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                -int(item["sequence"]),
                item["record_id"],
            )
        )
        selected_ids = {
            candidate["unit_id"]
            for candidate in candidates
            if float(candidate["score"]) >= self.reflection_threshold
        }
        if len(selected_ids) > self.reflection_batch_size:
            selected_ids = {
                candidate["unit_id"]
                for candidate in candidates[: self.reflection_batch_size]
                if float(candidate["score"]) >= self.reflection_threshold
            }

        decisions = [
            decision and unit.unit_id in selected_ids
            for unit, decision in zip(packet.units, packet.decisions, strict=True)
        ]
        trace = _copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "source_layer": self.source_layer,
            "reflection_layer": self.reflection_layer,
            "reflection_threshold": self.reflection_threshold,
            "reflection_batch_size": self.reflection_batch_size,
            "latest_reflected_source_seq": latest_reflected_seq,
            "selected_unit_ids": list(selected_ids),
            "candidate_count": len(candidates),
            "per_unit": per_unit,
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class GenerativeAgentsMemoryEvolution(MemoryEvolutionModule):
    """Append reflection records into the reflection layer for selected observations."""

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
        context_window: int = 3,
    ) -> None:
        self.source_layer = source_layer
        self.reflection_layer = reflection_layer
        self.context_window = int(context_window)

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

        if not store.has_layer(self.source_layer):
            trace = _copy_trace(packet)
            trace["memory_evolution"] = {
                "module": self.spec.name,
                "decision_source": "evolution_decisions",
                "active_unit_ids": [],
                "effects": [],
            }
            return replace(packet, trace=trace), store

        source_records = store.iter_records(self.source_layer)
        source_by_unit_id = {record.unit_id: record for record in source_records}
        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]
        effects: list[dict[str, Any]] = []
        max_reflected_seq = int(_state(store).get("last_reflected_source_seq", 0) or 0)

        for unit_id in active_unit_ids:
            source_record = source_by_unit_id.get(unit_id)
            if source_record is None:
                continue
            context_records = _recent_context_records(
                source_records,
                source_record=source_record,
                window_size=self.context_window,
            )
            reflection_text = _build_reflection_text(context_records)
            importance = min(1.0, _record_importance(source_record) + 0.18)
            keywords = _dedupe([keyword for record in context_records for keyword in _record_keywords(record)])
            entities = _dedupe([entity for record in context_records for entity in _record_entities(record)])
            tags = _dedupe(
                ["reflection", "pattern", *[tag for record in context_records for tag in _record_tags(record)]]
            )
            reflection_unit = MemoryUnit(
                text=reflection_text,
                unit_type="reflection",
                representation_elements=("text", "summary", "keywords", "entities", "tags"),
                timestamp=source_record.timestamp,
                normalized_text=reflection_text.casefold(),
                entities=entities,
                tags=tags,
                metadata={
                    "source_layer": self.source_layer,
                    "source_unit_id": unit_id,
                    "source_record_id": source_record.record_id,
                    "source_record_ids": [record.record_id for record in context_records],
                    "reflection_context_window": self.context_window,
                    "importance": importance,
                    "representation": _representation_summary_from_text(
                        reflection_text,
                        keywords=keywords,
                        entities=entities,
                        tags=tags,
                        importance=importance,
                    ),
                },
            )
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(
                reflection_unit,
                layer=self.reflection_layer,
                sequence_id=sequence_id,
            )
            store.append(record)
            effects.append(
                {
                    "effect_type": "reflection_append",
                    "unit_id": unit_id,
                    "source_record_id": source_record.record_id,
                    "reflection_record_id": record.record_id,
                    "target_layer": self.reflection_layer,
                    "importance": importance,
                }
            )
            max_reflected_seq = max(max_reflected_seq, _sequence_number(source_record.record_id))

        state = _state(store)
        state["last_reflected_source_seq"] = max_reflected_seq
        state["reflection_count"] = int(state.get("reflection_count", 0)) + len(effects)

        trace = _copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
        }
        return replace(packet, trace=trace), store


class GenerativeAgentsRetrieval(RetrievalModule):
    """Rank memories with relevance, recency, and importance weights."""

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
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
    ) -> None:
        if top_k <= 0:
            raise ValueError("GenerativeAgentsRetrieval requires top_k > 0.")
        self.top_k = int(top_k)
        self.layer = layer
        self.relevance_weight = float(relevance_weight)
        self.recency_weight = float(recency_weight)
        self.importance_weight = float(importance_weight)

    def validate_store(self, store: MemoryStore) -> None:
        if self.layer is not None and not store.has_layer(self.layer):
            raise IncompatibleCompositionError(
                f"GenerativeAgentsRetrieval requires declared layer {self.layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("GenerativeAgentsRetrieval requires packet.query.")

        all_records = store.iter_records(self.layer)
        if not all_records:
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

        ordered_by_recency = sorted(
            all_records,
            key=lambda record: (
                _parse_iso_timestamp(record.timestamp),
                _sequence_number(record.record_id),
            ),
            reverse=True,
        )
        recency_lookup = {
            record.record_id: (
                1.0 if len(ordered_by_recency) == 1 else 1.0 - (index / (len(ordered_by_recency) - 1))
            )
            for index, record in enumerate(ordered_by_recency)
        }
        query_tokens = set(_tokenize(packet.query.text))
        scored: list[dict[str, Any]] = []
        for record in all_records:
            relevance = _relevance_score(query_tokens, record)
            recency = recency_lookup.get(record.record_id, 0.0)
            importance = _record_importance(record)
            total = (
                (self.relevance_weight * relevance)
                + (self.recency_weight * recency)
                + (self.importance_weight * importance)
            )
            scored.append(
                {
                    "record": record,
                    "score": total,
                    "relevance": relevance,
                    "recency": recency,
                    "importance": importance,
                }
            )

        scored.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item["relevance"]),
                -float(item["importance"]),
                -float(item["recency"]),
                _sequence_number(item["record"].record_id),
            )
        )
        selected = scored[: self.top_k]
        items = [item["record"] for item in selected]
        scores = [
            {
                "record_id": item["record"].record_id,
                "rank": rank,
                "score": float(item["score"]),
                "relevance": float(item["relevance"]),
                "recency": float(item["recency"]),
                "importance": float(item["importance"]),
                "strategy": "weighted_relevance_recency_importance",
            }
            for rank, item in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "candidate_count": len(all_records),
                "layer": self.layer,
                "weights": {
                    "relevance": self.relevance_weight,
                    "recency": self.recency_weight,
                    "importance": self.importance_weight,
                },
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


def build_generative_agents_pipeline(
    *,
    store: MemoryStore | None = None,
    top_k: int = 5,
    observation_layer: str = "observation_stream",
    reflection_layer: str = "reflections",
    reflection_threshold: float = 0.55,
    reflection_batch_size: int = 2,
    context_window: int = 3,
    relevance_weight: float = 0.5,
    recency_weight: float = 0.3,
    importance_weight: float = 0.2,
) -> MemoryPipeline:
    from ..baselines import (
        AlwaysWriteTrigger,
        AppendOrganization,
        ConcatenateReadout,
        PassThroughUnitFormation,
    )

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
        unit_formation=PassThroughUnitFormation(),
        representation=GenerativeAgentsRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
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
        ),
        retrieval=GenerativeAgentsRetrieval(
            top_k=top_k,
            layer=None,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        ),
        readout=ConcatenateReadout(separator="\n\n"),
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


__all__ = [
    "GenerativeAgentsMemoryEvolution",
    "GenerativeAgentsReadout",
    "GenerativeAgentsReflectionTrigger",
    "GenerativeAgentsRepresentation",
    "GenerativeAgentsRetrieval",
    "GenerativeAgentsUnitFormation",
    "build_generative_agents_pipeline",
    "build_generative_agents_topology",
]
