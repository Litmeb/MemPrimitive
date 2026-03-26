"""TiM support for the classic memory workstream.

This local sketch keeps the TiM motif deterministic and small:

- reasoning-step unit formation
- budget-triggered compaction and summarization
- weighted thought-memory retrieval over the evolved store

The implementation stays within the repo's ``Packet`` / ``MemoryStore`` flow so
the example can be exercised like the other classic workstreams.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Final

from memprimitive import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Observation, Packet, Placement, Query, Readout, RetrievedSet, StoreLayerSpec, StoreTopology
from memprimitive.baselines._trace import copy_trace
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import EvolutionTriggerModule, MemoryEvolutionModule, OrganizationModule, ReadoutModule, RetrievalModule, RepresentationModule, UnitFormationModule, WriteTriggerModule
from memprimitive.pipeline import MemoryPipeline
from ._runtime import ClassicRuntime, get_classic_runtime

TIM_THOUGHT_LAYER: Final[str] = "thought_memory"
TIM_LAYER_ORDER: Final[tuple[str, ...]] = (TIM_THOUGHT_LAYER,)
_STEP_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?:step\s*\d+[:.\-]\s*|\d+[).:-]\s*|[-*]\s+)", re.IGNORECASE)
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[.!?]+\s+")
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "for",
        "with",
        "is",
        "are",
        "be",
        "this",
        "that",
        "it",
        "we",
        "you",
        "as",
        "at",
    }
)


def _tim_controls(metadata: dict[str, Any] | None) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return controls

    nested = metadata.get("tim")
    if isinstance(nested, dict):
        controls.update(nested)

    for key in ("reasoning_step", "reasoning", "steps", "budget", "force_compact", "compact", "source", "mode"):
        if key in metadata and key not in controls:
            controls[key] = metadata[key]
    return controls


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "step", "reasoning", "thought"}:
            return True
        if normalized in {"false", "no", "0", "dialogue", "note"}:
            return False
    return None


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _compact_text(text: str, *, limit: int = 84) -> str:
    text = _normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.casefold()) if token]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return ClassicRuntime.cosine_similarity(left, right)


def _keywords(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for token in _tokenize(text):
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _summary_from_steps(steps: list[str], *, prefix: str = "TiM summary") -> str:
    if not steps:
        return f"{prefix}: no reasoning steps"
    summary = get_classic_runtime().summarize_records(
        records=[{"step_index": index + 1, "text": step} for index, step in enumerate(steps)],
        instruction=(
            f"Summarize these TiM reasoning steps into one compact memory entry beginning with '{prefix}'."
        ),
    ).strip()
    if not summary:
        return f"{prefix}: no reasoning steps"
    return summary if summary.startswith(prefix) else f"{prefix}: {summary}"


def _step_texts_from_observation(observation: Observation) -> list[str]:
    controls = _tim_controls(observation.metadata)
    hinted_steps = controls.get("steps")
    if isinstance(hinted_steps, list) and hinted_steps:
        steps: list[str] = []
        for item in hinted_steps:
            if isinstance(item, str) and item.strip():
                steps.append(item.strip())
            elif isinstance(item, dict) and str(item.get("text", "")).strip():
                steps.append(str(item["text"]).strip())
        if steps:
            return steps

    raw_text = _normalize_text(observation.text)
    if "\n" in observation.text:
        candidates = [line.strip() for line in observation.text.splitlines() if line.strip()]
    else:
        candidates = [part.strip() for part in _SPLIT_PATTERN.split(raw_text) if part.strip()]
        if not candidates:
            candidates = [raw_text]

    steps = []
    for candidate in candidates:
        stripped = _STEP_PREFIX_PATTERN.sub("", candidate).strip()
        if stripped:
            steps.append(stripped)
    return steps or [raw_text]


def _record_tokens(record: MemoryRecord) -> set[str]:
    tokens = set(_tokenize(record.text))
    representation = record.metadata.get("representation")
    if isinstance(representation, dict):
        summary = representation.get("summary")
        if isinstance(summary, str):
            tokens.update(_tokenize(summary))
        keywords = representation.get("keywords")
        if isinstance(keywords, list):
            tokens.update(str(item).casefold() for item in keywords if str(item).strip())
    tim_meta = record.metadata.get("tim")
    if isinstance(tim_meta, dict):
        summary = tim_meta.get("summary")
        if isinstance(summary, str):
            tokens.update(_tokenize(summary))
    return {token for token in tokens if token and token not in _STOPWORDS}


def _record_embedding(record: MemoryRecord) -> list[float]:
    if record.embedding is not None:
        return list(record.embedding)
    return get_classic_runtime().embed(record.text)


class TimReasoningStepUnitFormation(UnitFormationModule):
    """Split an observation into reasoning-step units."""

    spec = ModuleSpec(
        name="tim_reasoning_step_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.metadata.tim"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("TimReasoningStepUnitFormation requires packet.observation.")

        observation = packet.observation
        controls = _tim_controls(observation.metadata)
        steps = _step_texts_from_observation(observation)
        units: list[MemoryUnit] = []
        for index, step_text in enumerate(steps):
            normalized = _normalize_text(step_text)
            unit = MemoryUnit(
                text=normalized,
                unit_type="reasoning_step",
                timestamp=observation.timestamp,
                normalized_text=normalized.casefold(),
                metadata={
                    **observation.metadata,
                    "source": observation.source,
                    "provenance": {
                        "observation_id": observation.observation_id,
                        "source": observation.source,
                    },
                    "tim": {
                        **controls,
                        "reasoning_step": True,
                        "step_index": index,
                        "step_count": len(steps),
                        "summary": _compact_text(normalized),
                    },
                    "representation": {
                        "text": normalized,
                        "summary": _compact_text(normalized),
                        "keywords": _keywords(normalized),
                    },
                },
            )
            units.append(unit)

        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
            "step_count": len(steps),
        }
        return replace(packet, units=units, trace=trace), store


class TimReasoningRepresentation(RepresentationModule):
    """Attach a compact deterministic representation to each reasoning step."""

    spec = ModuleSpec(
        name="tim_reasoning_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.embedding", "units.metadata.representation"),
    )

    def __init__(self, *, embedding_dim: int = 16) -> None:
        self.embedding_dim = int(embedding_dim)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimReasoningRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            text = _normalize_text(unit.text)
            embedding = get_classic_runtime().embed(text)
            summary = (
                unit.metadata.get("tim", {}).get("summary")
                if isinstance(unit.metadata.get("tim"), dict)
                else None
            )
            summary = _compact_text(str(summary) if summary else text)
            represented = replace(
                unit,
                text=text,
                normalized_text=text.casefold(),
                embedding=embedding,
                representation_elements=("text", "embedding", "summary", "keywords"),
                metadata={
                    **unit.metadata,
                    "tim": {
                        **(unit.metadata.get("tim") if isinstance(unit.metadata.get("tim"), dict) else {}),
                        "summary": summary,
                    },
                    "representation": {
                        "text": text,
                        "normalized_text": text.casefold(),
                        "embedding": {"dim": len(embedding)},
                        "summary": summary,
                        "keywords": _keywords(text),
                    },
                },
            )
            represented_units.append(represented)
            per_unit.append(
                {
                    "unit_id": represented.unit_id,
                    "embedding_dim": len(embedding),
                    "summary": summary,
                }
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in represented_units],
            "embedding_dim": self.embedding_dim,
            "per_unit": per_unit,
        }
        return replace(packet, units=represented_units, trace=trace), store


class TimReasoningWriteTrigger(WriteTriggerModule):
    """Accept reasoning-step units and ignore anything else."""

    spec = ModuleSpec(
        name="tim_reasoning_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimReasoningWriteTrigger requires packet.units.")

        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            tim_meta = unit.metadata.get("tim")
            is_reasoning_step = unit.unit_type == "reasoning_step"
            if isinstance(tim_meta, dict):
                explicit = _coerce_bool(tim_meta.get("reasoning_step"))
                if explicit is not None:
                    is_reasoning_step = explicit
                if _coerce_bool(tim_meta.get("write")) is False:
                    is_reasoning_step = False
            decisions.append(is_reasoning_step)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "decision": is_reasoning_step,
                    "reason": "reasoning_step" if is_reasoning_step else "non_reasoning_step",
                }
            )

        trace = copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "policy": "on_reasoning_step",
            "per_unit": per_unit,
        }
        return replace(packet, decisions=decisions, trace=trace), store


class TimBudgetEvolutionTrigger(EvolutionTriggerModule):
    """Trigger compaction when the thought-memory budget is exceeded."""

    spec = ModuleSpec(
        name="tim_budget_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(self, *, thought_layer: str = TIM_THOUGHT_LAYER, budget: int = 4) -> None:
        if budget <= 0:
            raise ValueError("TimBudgetEvolutionTrigger requires budget > 0.")
        self.thought_layer = thought_layer
        self.budget = int(budget)

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.thought_layer):
            raise IncompatibleCompositionError(
                f"TimBudgetEvolutionTrigger requires declared layer {self.thought_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimBudgetEvolutionTrigger requires packet.units.")
        if packet.placements is None:
            raise ValueError("TimBudgetEvolutionTrigger requires packet.placements.")

        controls = _tim_controls(packet.observation.metadata if packet.observation is not None else None)
        force_compact = _coerce_bool(controls.get("force_compact")) is True or _coerce_bool(controls.get("compact")) is True
        thought_count = store.count(self.thought_layer)
        should_compact = force_compact or thought_count > self.budget
        decisions = [should_compact for _ in packet.units]

        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "thought_layer": self.thought_layer,
            "budget": self.budget,
            "thought_count": thought_count,
            "force_compact": force_compact,
            "should_compact": should_compact,
            "per_unit": [
                {
                    "unit_id": unit.unit_id,
                    "decision": should_compact,
                    "reason": "budget_exceeded" if should_compact else "within_budget",
                }
                for unit in packet.units
            ],
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class TimThoughtMemoryEvolution(MemoryEvolutionModule):
    """Summarize old reasoning steps and prune the overflowing portion."""

    spec = ModuleSpec(
        name="tim_thought_memory_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, thought_layer: str = TIM_THOUGHT_LAYER, budget: int = 4) -> None:
        if budget <= 0:
            raise ValueError("TimThoughtMemoryEvolution requires budget > 0.")
        self.thought_layer = thought_layer
        self.budget = int(budget)

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.thought_layer):
            raise IncompatibleCompositionError(
                f"TimThoughtMemoryEvolution requires declared layer {self.thought_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimThoughtMemoryEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("TimThoughtMemoryEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("TimThoughtMemoryEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("TimThoughtMemoryEvolution requires aligned units, placements, and evolution decisions.")

        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]
        effects: list[dict[str, Any]] = []
        if active_unit_ids and store.count(self.thought_layer) > self.budget:
            effects.extend(self._compact_thought_memory(store))

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "budget": self.budget,
            "effects": effects,
        }
        return replace(packet, trace=trace), store

    def _compact_thought_memory(self, store: MemoryStore) -> list[dict[str, Any]]:
        records = store.iter_records(self.thought_layer)
        if len(records) <= self.budget:
            return []

        keep_count = max(0, self.budget - 1)
        if keep_count:
            kept_records = records[-keep_count:]
            pruned_records = records[:-keep_count]
        else:
            kept_records = []
            pruned_records = records

        summary_text = _summary_from_steps([record.text for record in pruned_records])
        summary_unit = MemoryUnit(
            text=summary_text,
            unit_type="thought_summary",
            timestamp=pruned_records[-1].timestamp if pruned_records else records[-1].timestamp,
            normalized_text=summary_text.casefold(),
            embedding=get_classic_runtime().embed(summary_text),
            representation_elements=("text", "embedding", "summary", "keywords"),
            tags=["tim", "summary", "compaction"],
            metadata={
                "tim": {
                    "kind": "thought_compaction",
                    "source_record_ids": [record.record_id for record in pruned_records],
                    "preserved_record_ids": [record.record_id for record in kept_records],
                    "budget": self.budget,
                    "summary": summary_text,
                },
                "representation": {
                    "text": summary_text,
                    "summary": summary_text,
                    "keywords": _keywords(summary_text),
                },
            },
        )
        store.layers[self.thought_layer] = list(kept_records)
        summary_record = MemoryRecord.from_unit(
            summary_unit,
            layer=self.thought_layer,
            sequence_id=store.next_sequence_id(),
        )
        store.append(summary_record)
        return [
            {
                "effect_type": "summarize_and_prune",
                "layer": self.thought_layer,
                "summary_record_id": summary_record.record_id,
                "pruned_record_ids": [record.record_id for record in pruned_records],
                "retained_record_ids": [record.record_id for record in kept_records] + [summary_record.record_id],
                "source_record_count": len(pruned_records),
            }
        ]


class TimThoughtMemoryRetrieval(RetrievalModule):
    """Rank thought-memory records by weighted similarity and recency."""

    spec = ModuleSpec(
        name="tim_thought_memory_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, top_k: int = 5, thought_layer: str = TIM_THOUGHT_LAYER, similarity_weight: float = 0.7) -> None:
        if top_k <= 0:
            raise ValueError("TimThoughtMemoryRetrieval requires top_k > 0.")
        self.top_k = int(top_k)
        self.thought_layer = thought_layer
        self.similarity_weight = float(similarity_weight)

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.thought_layer):
            raise IncompatibleCompositionError(
                f"TimThoughtMemoryRetrieval requires declared layer {self.thought_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("TimThoughtMemoryRetrieval requires packet.query.")

        query = packet.query
        query_embedding = list(query.embedding) if query.embedding is not None else get_classic_runtime().embed(query.text)
        if query.embedding is None:
            query = replace(query, embedding=query_embedding)
        query_tokens = set(_keywords(query.text, limit=16))
        records = store.iter_records(self.thought_layer)

        scored: list[dict[str, Any]] = []
        total = len(records)
        for recency_index, record in enumerate(reversed(records)):
            record_embedding = _record_embedding(record)
            similarity = _cosine_similarity(query_embedding, record_embedding)
            record_tokens = _record_tokens(record)
            keyword_overlap = len(query_tokens & record_tokens)
            keyword_score = keyword_overlap / max(1, len(query_tokens))
            recency_score = 1.0 / (recency_index + 1)
            score = (
                (self.similarity_weight * similarity)
                + (0.2 * keyword_score)
                + (0.1 * recency_score)
            )
            scored.append(
                {
                    "record": record,
                    "score": score,
                    "similarity": similarity,
                    "keyword_score": keyword_score,
                    "recency_index": recency_index,
                    "recency_score": recency_score,
                    "total_candidates": total,
                }
            )

        scored.sort(key=lambda item: (-float(item["score"]), int(item["recency_index"])))
        selected = scored[: self.top_k]
        items = [item["record"] for item in selected]
        scores = [
            {
                "record_id": item["record"].record_id,
                "rank": rank,
                "score": float(item["score"]),
                "strategy": "weighted_sum",
                "similarity": float(item["similarity"]),
                "keyword_score": float(item["keyword_score"]),
                "recency_score": float(item["recency_score"]),
            }
            for rank, item in enumerate(selected, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "thought_layer": self.thought_layer,
                "candidate_count": len(records),
                "returned_count": len(items),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, query=query, retrieved=retrieved, trace=trace), store


class TimThoughtReadout(ReadoutModule):
    """Render the selected thought memories as a compact prompt chunk."""

    spec = ModuleSpec(
        name="tim_thought_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, *, item_budget: int = 4) -> None:
        if item_budget <= 0:
            raise ValueError("TimThoughtReadout requires item_budget > 0.")
        self.item_budget = int(item_budget)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("TimThoughtReadout requires packet.retrieved.")

        items = packet.retrieved.items[: self.item_budget]
        omitted = max(0, len(packet.retrieved.items) - len(items))
        source_ids = [record.record_id for record in items]
        lines = [f"[{record.layer}] {record.text}" for record in items]
        text = "\n".join(lines)
        if not text:
            text = ""
        readout = Readout(
            text=text,
            source_ids=source_ids,
            metadata={
                "item_count": len(items),
                "omitted_item_count": omitted,
                "item_budget": self.item_budget,
                "layer_counts": {layer: sum(1 for record in items if record.layer == layer) for layer in TIM_LAYER_ORDER},
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "item_budget": self.item_budget,
        }
        return replace(packet, readout=readout, trace=trace), store


def build_tim_pipeline(
    *,
    store: MemoryStore | None = None,
    thought_layer: str = TIM_THOUGHT_LAYER,
    budget: int = 4,
    top_k: int = 5,
    readout_item_budget: int = 4,
) -> MemoryPipeline:
    """Build a deterministic TiM-style pipeline."""

    if store is None:
        store = MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(
                        name=thought_layer,
                        theme="working",
                        capacity="token_limited",
                        indices=("temporal", "vector", "keyword"),
                    ),
                ]
            )
        )
    elif not store.has_layer(thought_layer):
        store.ensure_layer(thought_layer, allow_create=True, theme="working")

    return MemoryPipeline(
        store=store,
        unit_formation=TimReasoningStepUnitFormation(),
        representation=TimReasoningRepresentation(),
        write_trigger=TimReasoningWriteTrigger(),
        organization=TimThoughtMemoryOrganization(target_layer=thought_layer),
        evolution_trigger=TimBudgetEvolutionTrigger(thought_layer=thought_layer, budget=budget),
        memory_evolution=TimThoughtMemoryEvolution(thought_layer=thought_layer, budget=budget),
        retrieval=TimThoughtMemoryRetrieval(top_k=top_k, thought_layer=thought_layer),
        readout=TimThoughtReadout(item_budget=readout_item_budget),
    )


class TimThoughtMemoryOrganization(OrganizationModule):
    """Append reasoning-step records into the thought-memory layer."""

    spec = ModuleSpec(
        name="tim_thought_memory_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, target_layer: str = TIM_THOUGHT_LAYER) -> None:
        self.target_layer = target_layer

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            raise IncompatibleCompositionError(
                f"TimThoughtMemoryOrganization requires declared layer {self.target_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimThoughtMemoryOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("TimThoughtMemoryOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("TimThoughtMemoryOrganization requires decisions aligned with units.")

        store.ensure_layer(self.target_layer)
        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        per_unit: list[dict[str, Any]] = []

        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "target_layer": self.target_layer,
                    "decision": decision,
                }
            )
            if not decision:
                continue
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit, layer=self.target_layer, sequence_id=sequence_id)
            store.append(record)
            written_record_ids.append(record.record_id)
            written_unit_ids.append(unit.unit_id)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "placements": [
                {"unit_id": placement.unit_id, "target_layer": placement.target_layer}
                for placement in placements
            ],
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "per_unit": per_unit,
        }
        return replace(packet, placements=placements, trace=trace), store


class TimWorkstream:
    """Convenience wrapper for the TiM example workflow."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        thought_layer: str = TIM_THOUGHT_LAYER,
        budget: int = 4,
        top_k: int = 5,
        readout_item_budget: int = 4,
    ) -> None:
        self.pipeline = build_tim_pipeline(
            store=store,
            thought_layer=thought_layer,
            budget=budget,
            top_k=top_k,
            readout_item_budget=readout_item_budget,
        )

    @property
    def store(self) -> MemoryStore:
        return self.pipeline.store

    def ingest(self, observation: Observation) -> Packet:
        return self.pipeline.ingest(observation)

    def recall(self, query: Query) -> Readout:
        return self.pipeline.recall(query)


__all__ = [
    "TIM_LAYER_ORDER",
    "TIM_THOUGHT_LAYER",
    "TimBudgetEvolutionTrigger",
    "TimReasoningRepresentation",
    "TimReasoningStepUnitFormation",
    "TimReasoningWriteTrigger",
    "TimThoughtMemoryEvolution",
    "TimThoughtMemoryOrganization",
    "TimThoughtMemoryRetrieval",
    "TimThoughtReadout",
    "TimWorkstream",
    "build_tim_pipeline",
]
