"""TiM support aligned more closely with the paper's memory loop.

This variant keeps the repo's fixed ``MemoryPipeline`` slots, but changes the
TiM internals to focus on:

- structured post-thought materialization from a ``(query, response)`` pair
- bucketed thought memory with stable LSH-style grouping
- group-local ``insert -> forget -> merge`` updates
- two-stage retrieval: nearest bucket first, then in-bucket ranking
"""

from __future__ import annotations

from dataclasses import replace
import json
import random
from typing import Any, Final

from memprimitive import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    ModuleSpec,
    Observation,
    Packet,
    Placement,
    Readout,
    RetrievedSet,
)
from memprimitive.baselines._trace import copy_trace
from memprimitive.exceptions import IncompatibleCompositionError
from memprimitive.interfaces import (
    EvolutionTriggerModule,
    MemoryEvolutionModule,
    OrganizationModule,
    ReadoutModule,
    RetrievalModule,
    RepresentationModule,
    UnitFormationModule,
    WriteTriggerModule,
)
from ._runtime import ClassicRuntime, get_classic_runtime

TIM_THOUGHT_LAYER: Final[str] = "thought_memory"
TIM_LAYER_ORDER: Final[tuple[str, ...]] = (TIM_THOUGHT_LAYER,)
TIM_HASH_BITS: Final[int] = 8
TIM_HASH_SEED: Final[int] = 17
_TOKEN_CHARS: Final[str] = "abcdefghijklmnopqrstuvwxyz0123456789_'"
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
        "from",
        "into",
        "their",
    }
)
_EXCLUSIVE_RELATIONS: Final[frozenset[str]] = frozenset(
    {
        "works_at",
        "work_at",
        "lives_in",
        "located_in",
        "born_in",
        "age",
        "birthday",
        "resides_in",
        "employed_by",
    }
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _tokenize(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for ch in str(text).casefold():
        if ch in _TOKEN_CHARS:
            current.append(ch)
        elif current:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return [chunk for chunk in chunks if chunk]


def _keywords(text: str, *, limit: int = 8) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in _tokenize(text):
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _tim_meta(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _representation_meta(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    return ClassicRuntime.cosine_similarity(left, right)


def _tim_store_state(store: MemoryStore) -> dict[str, Any]:
    state = store.metadata.get("tim")
    if not isinstance(state, dict):
        state = {"buckets": {}, "record_to_bucket": {}}
        store.metadata["tim"] = state
    state.setdefault("buckets", {})
    state.setdefault("record_to_bucket", {})
    return state


def _rebuild_tim_index(store: MemoryStore, *, thought_layer: str) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    record_to_bucket: dict[str, str] = {}
    for record in store.iter_records(thought_layer):
        tim = _tim_meta(record.metadata.get("tim"))
        bucket = str(tim.get("group_id") or tim.get("hash_index") or "").strip()
        if not bucket:
            continue
        buckets.setdefault(bucket, []).append(record.record_id)
        record_to_bucket[record.record_id] = bucket
    state = _tim_store_state(store)
    state["buckets"] = buckets
    state["record_to_bucket"] = record_to_bucket
    return buckets


def _record_tim(record: MemoryRecord) -> dict[str, Any]:
    return _tim_meta(record.metadata.get("tim"))


def _record_bucket(record: MemoryRecord) -> str:
    tim = _record_tim(record)
    return str(tim.get("group_id") or tim.get("hash_index") or "").strip()


def _record_embedding(record: MemoryRecord) -> list[float]:
    if record.embedding is not None:
        return list(record.embedding)
    representation = _representation_meta(record.metadata.get("representation"))
    return get_classic_runtime().embed(str(representation.get("text") or record.text))


def _projection_vector(dim: int, bit_index: int) -> list[float]:
    rng = random.Random(f"tim:{TIM_HASH_SEED}:{dim}:{bit_index}")
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def _hash_embedding(embedding: list[float], *, bits: int = TIM_HASH_BITS) -> str:
    if not embedding:
        return "0" * bits
    dim = len(embedding)
    bucket_bits: list[str] = []
    for bit_index in range(bits):
        projection = _projection_vector(dim, bit_index)
        score = sum(value * weight for value, weight in zip(embedding, projection, strict=True))
        bucket_bits.append("1" if score >= 0.0 else "0")
    return "".join(bucket_bits)


def _bucket_anchor_text(tim_meta: dict[str, Any], *, fallback_text: str) -> str:
    head = _normalize_text(tim_meta.get("head_entity", ""))
    relation = _normalize_text(tim_meta.get("relation", ""))
    if head and relation:
        return f"{head} {relation}"
    return fallback_text


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(1 for lch, rch in zip(left, right, strict=True) if lch != rch)


def _thought_dict_from_item(
    item: Any,
    *,
    fallback_text: str,
    source_query: str,
    source_response: str,
    source_turn_id: str,
) -> dict[str, Any]:
    data = dict(item) if isinstance(item, dict) else {"thought_text": str(item)}
    head = _normalize_text(data.get("head_entity", ""))
    relation = _normalize_text(data.get("relation", ""))
    tail = _normalize_text(data.get("tail_entity", ""))
    thought_text = _normalize_text(
        data.get("thought_text")
        or data.get("text")
        or data.get("thought")
        or fallback_text
    )
    if not thought_text and head and relation and tail:
        thought_text = f"{head} {relation} {tail}"
    triple = data.get("triple")
    if isinstance(triple, (list, tuple)) and len(triple) == 3:
        triple_value = [str(triple[0]), str(triple[1]), str(triple[2])]
    else:
        triple_value = [head, relation, tail]
    return {
        "thought_text": thought_text,
        "head_entity": head,
        "relation": relation,
        "tail_entity": tail,
        "triple": triple_value,
        "source_query": _normalize_text(data.get("source_query", source_query)),
        "source_response": _normalize_text(data.get("source_response", source_response)),
        "source_turn_id": _normalize_text(data.get("source_turn_id", source_turn_id)),
        "write": data.get("write", True),
    }


def _allows_forget(records: list[MemoryRecord], forget_ids: list[str]) -> list[str]:
    if not forget_ids:
        return []
    by_id = {record.record_id: record for record in records}
    allowed: list[str] = []
    for record_id in forget_ids:
        record = by_id.get(record_id)
        if record is None:
            continue
        tim_meta = _record_tim(record)
        relation = _normalize_text(tim_meta.get("relation", "")).casefold()
        head = _normalize_text(tim_meta.get("head_entity", "")).casefold()
        tail = _normalize_text(tim_meta.get("tail_entity", "")).casefold()
        if relation not in _EXCLUSIVE_RELATIONS:
            continue
        for other in records:
            if other.record_id == record_id:
                continue
            other_meta = _record_tim(other)
            if (
                _normalize_text(other_meta.get("head_entity", "")).casefold() == head
                and _normalize_text(other_meta.get("relation", "")).casefold() == relation
                and _normalize_text(other_meta.get("tail_entity", "")).casefold() != tail
            ):
                allowed.append(record_id)
                break
    return allowed


def _thoughts_from_observation(observation: Observation) -> list[dict[str, Any]]:
    tim = _tim_meta(observation.metadata.get("tim"))
    source_query = _normalize_text(tim.get("source_query") or observation.metadata.get("source_query", ""))
    source_response = _normalize_text(
        tim.get("source_response") or observation.metadata.get("source_response", "")
    )
    source_turn_id = _normalize_text(
        tim.get("source_turn_id") or observation.metadata.get("source_turn_id") or observation.observation_id
    )
    hinted = tim.get("thoughts")
    if isinstance(hinted, list):
        thoughts = [
            _thought_dict_from_item(
                item,
                fallback_text=observation.text,
                source_query=source_query,
                source_response=source_response,
                source_turn_id=source_turn_id,
            )
            for item in hinted
        ]
        return [thought for thought in thoughts if thought["thought_text"]]

    if any(key in tim for key in ("thought_text", "head_entity", "relation", "tail_entity", "triple")):
        thought = _thought_dict_from_item(
            tim,
            fallback_text=observation.text,
            source_query=source_query,
            source_response=source_response,
            source_turn_id=source_turn_id,
        )
        return [thought] if thought["thought_text"] else []

    fallback_text = _normalize_text(observation.text)
    if not fallback_text:
        return []
    return [
        _thought_dict_from_item(
            {"thought_text": fallback_text},
            fallback_text=fallback_text,
            source_query=source_query,
            source_response=source_response,
            source_turn_id=source_turn_id,
        )
    ]


def _forget_prompt_payload(*, bucket: str, records: list[MemoryRecord], new_record_ids: list[str]) -> str:
    payload = {
        "bucket": bucket,
        "new_record_ids": new_record_ids,
        "records": [
            {
                "record_id": record.record_id,
                "text": record.text,
                "tim": _record_tim(record),
            }
            for record in records
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _merge_prompt_payload(*, bucket: str, records: list[MemoryRecord]) -> str:
    payload = {
        "bucket": bucket,
        "records": [
            {
                "record_id": record.record_id,
                "text": record.text,
                "tim": _record_tim(record),
            }
            for record in records
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


class TimThoughtUnitFormation(UnitFormationModule):
    """Materialize post-think outputs into structured TiM thought units."""

    spec = ModuleSpec(
        name="tim_thought_unit_formation",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.metadata.tim"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("TimThoughtUnitFormation requires packet.observation.")

        observation = packet.observation
        thoughts = _thoughts_from_observation(observation)
        units: list[MemoryUnit] = []
        for index, thought in enumerate(thoughts):
            thought_text = _normalize_text(thought["thought_text"])
            if not thought_text:
                continue
            tim_meta = {
                **thought,
                "thought_index": index,
                "thought_count": len(thoughts),
                "write": _coerce_bool(thought.get("write")) is not False,
            }
            unit = MemoryUnit(
                text=thought_text,
                unit_type="tim_thought",
                timestamp=observation.timestamp,
                normalized_text=thought_text.casefold(),
                metadata={
                    **observation.metadata,
                    "source": observation.source,
                    "provenance": {
                        "observation_id": observation.observation_id,
                        "source": observation.source,
                    },
                    "tim": tim_meta,
                    "representation": {
                        "text": thought_text,
                        "summary": thought_text,
                        "keywords": _keywords(thought_text),
                        "triple": list(tim_meta["triple"]),
                    },
                },
            )
            units.append(unit)

        trace = copy_trace(packet)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
            "thought_count": len(units),
        }
        return replace(packet, units=units, trace=trace), store


class TimThoughtRepresentation(RepresentationModule):
    """Attach embedding and stable hash-group metadata to each TiM thought."""

    spec = ModuleSpec(
        name="tim_thought_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.embedding", "units.metadata.representation"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimThoughtRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            text = _normalize_text(unit.text)
            embedding = get_classic_runtime().embed(text)
            tim_meta = _tim_meta(unit.metadata.get("tim"))
            bucket_anchor = _bucket_anchor_text(tim_meta, fallback_text=text)
            hash_index = _hash_embedding(get_classic_runtime().embed(bucket_anchor))
            tim_meta.update(
                {
                    "thought_text": text,
                    "hash_index": hash_index,
                    "group_id": hash_index,
                    "summary": text,
                }
            )
            representation = _representation_meta(unit.metadata.get("representation"))
            representation.update(
                {
                    "text": text,
                    "normalized_text": text.casefold(),
                    "embedding": {"dim": len(embedding)},
                    "summary": text,
                    "keywords": _keywords(text),
                    "hash_index": hash_index,
                    "group_id": hash_index,
                    "triple": list(tim_meta.get("triple", [])),
                }
            )
            represented_units.append(
                replace(
                    unit,
                    text=text,
                    normalized_text=text.casefold(),
                    embedding=embedding,
                    representation_elements=("text", "embedding", "summary", "keywords", "triple"),
                    metadata={**unit.metadata, "tim": tim_meta, "representation": representation},
                )
            )
            per_unit.append({"unit_id": unit.unit_id, "hash_index": hash_index})

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in represented_units],
            "per_unit": per_unit,
        }
        return replace(packet, units=represented_units, trace=trace), store


class TimThoughtWriteTrigger(WriteTriggerModule):
    """Write all structured TiM thoughts unless metadata explicitly disables them."""

    spec = ModuleSpec(
        name="tim_thought_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TimThoughtWriteTrigger requires packet.units.")

        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []
        for unit in packet.units:
            tim_meta = _tim_meta(unit.metadata.get("tim"))
            decision = unit.unit_type == "tim_thought" and _coerce_bool(tim_meta.get("write")) is not False
            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "decision": decision,
                    "reason": "tim_thought" if decision else "write_disabled",
                }
            )

        trace = copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "policy": "on_tim_thought",
            "per_unit": per_unit,
        }
        return replace(packet, decisions=decisions, trace=trace), store


class TimBudgetEvolutionTrigger(EvolutionTriggerModule):
    """Trigger group-local TiM updates whenever new thoughts are written."""

    spec = ModuleSpec(
        name="tim_update_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(self, *, thought_layer: str = TIM_THOUGHT_LAYER, budget: int = 4) -> None:
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

        decisions = [unit.unit_type == "tim_thought" for unit in packet.units]
        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "thought_layer": self.thought_layer,
            "per_unit": [
                {
                    "unit_id": unit.unit_id,
                    "decision": decision,
                    "reason": "new_thought_written" if decision else "not_tim_thought",
                }
                for unit, decision in zip(packet.units, decisions, strict=True)
            ],
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class TimThoughtMemoryOrganization(OrganizationModule):
    """Insert TiM thought records and update bucket metadata."""

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
        existing_buckets = _rebuild_tim_index(store, thought_layer=self.target_layer)
        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        per_unit: list[dict[str, Any]] = []

        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            tim_meta = _tim_meta(unit.metadata.get("tim"))
            bucket = str(tim_meta.get("group_id") or tim_meta.get("hash_index") or "").strip()
            before_count = len(existing_buckets.get(bucket, []))
            if decision:
                record = MemoryRecord.from_unit(
                    unit,
                    layer=self.target_layer,
                    sequence_id=store.next_sequence_id(),
                )
                store.append(record)
                written_record_ids.append(record.record_id)
                existing_buckets.setdefault(bucket, []).append(record.record_id)
                after_count = len(existing_buckets.get(bucket, []))
            else:
                after_count = before_count
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "bucket": bucket,
                    "decision": decision,
                    "group_size_before": before_count,
                    "group_size_after": after_count,
                }
            )

        _rebuild_tim_index(store, thought_layer=self.target_layer)
        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "written_record_ids": written_record_ids,
            "per_unit": per_unit,
        }
        return replace(packet, placements=placements, trace=trace), store


class TimThoughtMemoryEvolution(MemoryEvolutionModule):
    """Apply group-local forget/merge updates inside TiM thought buckets."""

    spec = ModuleSpec(
        name="tim_thought_memory_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(self, *, thought_layer: str = TIM_THOUGHT_LAYER, budget: int = 4) -> None:
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

        active_units = [
            unit
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]
        effects: list[dict[str, Any]] = []
        if active_units:
            buckets_to_update = {
                str(_tim_meta(unit.metadata.get("tim")).get("group_id", "")).strip()
                for unit in active_units
                if str(_tim_meta(unit.metadata.get("tim")).get("group_id", "")).strip()
            }
            for bucket in sorted(buckets_to_update):
                effects.extend(self._forget_conflicts(store, bucket=bucket, new_units=active_units))
                effects.extend(self._merge_bucket(store, bucket=bucket))
            _rebuild_tim_index(store, thought_layer=self.thought_layer)

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": [unit.unit_id for unit in active_units],
            "effects": effects,
        }
        return replace(packet, trace=trace), store

    def _forget_conflicts(self, store: MemoryStore, *, bucket: str, new_units: list[MemoryUnit]) -> list[dict[str, Any]]:
        records = [
            record
            for record in store.iter_records(self.thought_layer)
            if _record_bucket(record) == bucket
        ]
        if len(records) < 2:
            return []
        new_texts = {unit.text for unit in new_units}
        new_record_ids = [record.record_id for record in records if record.text in new_texts]
        result = get_classic_runtime().json(
            system=(
                "TiM forget judge. "
                "Return JSON with key 'forget_record_ids' containing record ids that should be removed "
                "because they contradict or are made unnecessary by newly inserted thoughts in the same bucket."
            ),
            user=_forget_prompt_payload(bucket=bucket, records=records, new_record_ids=new_record_ids),
        )
        forget_ids: list[str] = []
        if isinstance(result, dict):
            raw_ids = result.get("forget_record_ids", []) or result.get("record_ids", [])
            if isinstance(raw_ids, list):
                forget_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        forget_ids = _allows_forget(records, forget_ids)
        if not forget_ids:
            return []

        forgotten_set = set(forget_ids)
        store.layers[self.thought_layer] = [
            record for record in store.iter_records(self.thought_layer) if record.record_id not in forgotten_set
        ]
        return [{"effect_type": "forget", "bucket": bucket, "forgotten_record_ids": forget_ids}]

    def _merge_bucket(self, store: MemoryStore, *, bucket: str) -> list[dict[str, Any]]:
        records = [
            record
            for record in store.iter_records(self.thought_layer)
            if _record_bucket(record) == bucket
        ]
        if len(records) < 2:
            return []
        result = get_classic_runtime().json(
            system=(
                "TiM merge judge. "
                "Return JSON with key 'merges' containing merge groups. "
                "Each merge must include record_ids and merged_thought with thought_text, head_entity, relation, tail_entity."
            ),
            user=_merge_prompt_payload(bucket=bucket, records=records),
        )
        if not isinstance(result, dict):
            result = {}
        raw_merges = result.get("merges", [])
        if not isinstance(raw_merges, list):
            raw_merges = []
        if not raw_merges:
            fallback_groups: dict[tuple[str, str], list[MemoryRecord]] = {}
            for record in records:
                tim_meta = _record_tim(record)
                head = _normalize_text(tim_meta.get("head_entity", ""))
                relation = _normalize_text(tim_meta.get("relation", ""))
                if not head or not relation:
                    continue
                fallback_groups.setdefault((head, relation), []).append(record)
            for (head, relation), grouped_records in fallback_groups.items():
                if len(grouped_records) < 2:
                    continue
                tails = [
                    _normalize_text(_record_tim(record).get("tail_entity", ""))
                    for record in grouped_records
                    if _normalize_text(_record_tim(record).get("tail_entity", ""))
                ]
                deduped_tails = list(dict.fromkeys(tails))
                merged_tail = ", ".join(deduped_tails)
                raw_merges.append(
                    {
                        "record_ids": [record.record_id for record in grouped_records],
                        "merged_thought": {
                            "thought_text": f"{head} {relation} {merged_tail}".strip(),
                            "head_entity": head,
                            "relation": relation,
                            "tail_entity": merged_tail,
                        },
                    }
                )

        effects: list[dict[str, Any]] = []
        for merge in raw_merges:
            if not isinstance(merge, dict):
                continue
            raw_ids = merge.get("record_ids", [])
            if not isinstance(raw_ids, list):
                continue
            merge_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
            if len(merge_ids) < 2:
                continue
            merge_id_set = set(merge_ids)
            candidates = [
                record for record in store.iter_records(self.thought_layer) if record.record_id in merge_id_set
            ]
            if len(candidates) < 2:
                continue
            merged_thought = _thought_dict_from_item(
                merge.get("merged_thought", {}),
                fallback_text=" ".join(record.text for record in candidates),
                source_query=_record_tim(candidates[-1]).get("source_query", ""),
                source_response=_record_tim(candidates[-1]).get("source_response", ""),
                source_turn_id=_record_tim(candidates[-1]).get("source_turn_id", ""),
            )
            merged_text = _normalize_text(merged_thought["thought_text"])
            merged_unit = MemoryUnit(
                text=merged_text,
                unit_type="tim_thought",
                normalized_text=merged_text.casefold(),
                embedding=get_classic_runtime().embed(merged_text),
                representation_elements=("text", "embedding", "summary", "keywords", "triple"),
                metadata={
                    "tim": {
                        **merged_thought,
                        "hash_index": bucket,
                        "group_id": bucket,
                        "merged_from": merge_ids,
                        "forgotten_record_ids": [],
                        "write": True,
                    },
                    "representation": {
                        "text": merged_text,
                        "normalized_text": merged_text.casefold(),
                        "summary": merged_text,
                        "keywords": _keywords(merged_text),
                        "triple": list(merged_thought["triple"]),
                        "hash_index": bucket,
                        "group_id": bucket,
                    },
                },
            )
            store.layers[self.thought_layer] = [
                record for record in store.iter_records(self.thought_layer) if record.record_id not in merge_id_set
            ]
            merged_record = MemoryRecord.from_unit(
                merged_unit,
                layer=self.thought_layer,
                sequence_id=store.next_sequence_id(),
            )
            store.append(merged_record)
            effects.append(
                {
                    "effect_type": "merge",
                    "bucket": bucket,
                    "merged_record_id": merged_record.record_id,
                    "merged_from": merge_ids,
                }
            )
        return effects


class TimThoughtMemoryRetrieval(RetrievalModule):
    """Recall from the nearest TiM bucket first, then rank within that bucket."""

    spec = ModuleSpec(
        name="tim_thought_memory_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, top_k: int = 5, thought_layer: str = TIM_THOUGHT_LAYER, similarity_weight: float = 1.0) -> None:
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
        query_bucket = _hash_embedding(query_embedding)
        bucket_map = _rebuild_tim_index(store, thought_layer=self.thought_layer)
        all_buckets = sorted(bucket_map)

        candidate_bucket_ids: list[str] = []
        selected_bucket = ""
        if all_buckets:
            distances = sorted(
                ((_hamming_distance(query_bucket, bucket), bucket) for bucket in all_buckets),
                key=lambda item: (item[0], item[1]),
            )
            if distances:
                best_distance = distances[0][0]
                candidate_bucket_ids = [bucket for distance, bucket in distances if distance == best_distance]
                selected_bucket = candidate_bucket_ids[0]

        candidate_records = [
            record
            for record in store.iter_records(self.thought_layer)
            if selected_bucket and _record_bucket(record) == selected_bucket
        ]
        query_tokens = set(_keywords(query.text, limit=16))
        scored: list[dict[str, Any]] = []
        for record in candidate_records:
            similarity = _cosine_similarity(query_embedding, _record_embedding(record))
            tie_break = len(query_tokens & set(_keywords(record.text, limit=16)))
            scored.append(
                {
                    "record": record,
                    "score": (self.similarity_weight * similarity) + (0.001 * tie_break),
                    "similarity": similarity,
                    "keyword_tie_break": tie_break,
                }
            )
        scored.sort(key=lambda item: (-float(item["score"]), -int(item["keyword_tie_break"]), item["record"].record_id))
        selected = scored[: self.top_k]
        items = [item["record"] for item in selected]
        scores = [
            {
                "record_id": item["record"].record_id,
                "rank": rank,
                "score": float(item["score"]),
                "strategy": "two_stage_bucket_then_similarity",
                "similarity": float(item["similarity"]),
                "bucket": selected_bucket,
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
                "query_bucket": query_bucket,
                "candidate_bucket_ids": candidate_bucket_ids,
                "selected_bucket": selected_bucket,
                "selected_group_size": len(candidate_records),
                "stage_1": {"available_buckets": all_buckets, "selected_bucket": selected_bucket},
                "stage_2": {"returned_record_ids": [record.record_id for record in items]},
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, query=query, retrieved=retrieved, trace=trace), store


class TimThoughtReadout(ReadoutModule):
    """Render recalled TiM thoughts as a simple list."""

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
        source_ids = [record.record_id for record in items]
        lines = [record.text for record in items]
        readout = Readout(
            text="\n".join(lines),
            source_ids=source_ids,
            metadata={
                "item_count": len(items),
                "recalled_thought_count": len(items),
                "selected_bucket": packet.retrieved.trace.get("selected_bucket"),
                "candidate_bucket_ids": packet.retrieved.trace.get("candidate_bucket_ids", []),
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "item_budget": self.item_budget,
        }
        return replace(packet, readout=readout, trace=trace), store


TimReasoningStepUnitFormation = TimThoughtUnitFormation
TimReasoningRepresentation = TimThoughtRepresentation
TimReasoningWriteTrigger = TimThoughtWriteTrigger


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
    "TimThoughtRepresentation",
    "TimThoughtUnitFormation",
    "TimThoughtWriteTrigger",
]
