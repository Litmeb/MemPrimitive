"""Baseline: memory evolution primitive."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from math import sqrt
import re
from typing import Any, Final

from ..contracts import RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT
from ..core import MemoryRecord, MemoryStore, ModuleSpec, Packet
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
    commit_store_snapshot,
    failed_tool_calls,
    format_tool_failure_summary,
    normalize_write_tool_specs,
    project_tool_specs_for_prompt,
    write_tool_specs_require_graph_contracts,
)
from ..utils._runtime import Runtime, get_runtime
from ..utils._template import (
    PromptPlan,
    ensure_prompt_plan,
    project_record_for_template,
    render_prompt_plan,
    resolve_records_from_prompt_metadata,
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


def _metadata_token(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _scope_for_record(record: MemoryRecord, scope_metadata_keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: deepcopy(record.metadata.get(key)) for key in scope_metadata_keys}


def _scope_group_key(scope: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((key, _metadata_token(value)) for key, value in scope.items())


def _record_matches_scope(record: MemoryRecord, scope: dict[str, Any]) -> bool:
    return all(record.metadata.get(key) == value for key, value in scope.items())


def _record_token_count(record: MemoryRecord) -> int:
    return len(record.text.split())


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def _split_sentences(text: str) -> list[str]:
    raw_parts = _SENTENCE_BOUNDARY.split(text.strip())
    sentences = [part.strip() for part in raw_parts if part.strip()]
    return sentences or [text.strip()]


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


class STMConsolidationEvolution(MemoryEvolutionModule):
    """Consolidate STM overflow into raw LTM episodes plus one session summary.

    This module scans ``working_layer`` directly instead of relying on store
    sliding-window trimming. For each metadata scope, it keeps the newest STM
    records within the configured budget, copy-appends older records into the
    LTM episode layer, rewrites the current session summary, and then deletes
    the consolidated STM records.
    """

    spec = ModuleSpec(
        name="stm_consolidation_evolution",
        slot="memory_evolution",
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records", "rewrite_records", "delete_records"),
    )

    def __init__(
        self,
        *,
        working_layer: str = "working",
        ltm_episode_layer: str = "episodic",
        ltm_sentence_layer: str = "sentence",
        summary_layer: str = "session_summary",
        record_budget: int | None = 20,
        token_budget: int | None = None,
        summary_max_sentences: int = 3,
        scope_metadata_keys: tuple[str, ...] = ("session_id",),
    ) -> None:
        self.working_layer = str(working_layer).strip()
        self.ltm_episode_layer = str(ltm_episode_layer).strip()
        self.ltm_sentence_layer = str(ltm_sentence_layer).strip()
        self.summary_layer = str(summary_layer).strip()
        if not self.working_layer:
            raise ValueError("working_layer must be non-empty.")
        if not self.ltm_episode_layer:
            raise ValueError("ltm_episode_layer must be non-empty.")
        if not self.ltm_sentence_layer:
            raise ValueError("ltm_sentence_layer must be non-empty.")
        if not self.summary_layer:
            raise ValueError("summary_layer must be non-empty.")
        if record_budget is None and token_budget is None:
            raise ValueError("STMConsolidationEvolution requires at least one active budget.")
        if record_budget is not None and int(record_budget) <= 0:
            raise ValueError("record_budget must be positive when provided.")
        if token_budget is not None and int(token_budget) <= 0:
            raise ValueError("token_budget must be positive when provided.")
        if int(summary_max_sentences) <= 0:
            raise ValueError("summary_max_sentences must be positive.")
        normalized_scope_keys = tuple(str(key).strip() for key in scope_metadata_keys)
        if not normalized_scope_keys or any(not key for key in normalized_scope_keys):
            raise ValueError("scope_metadata_keys must contain at least one non-empty key.")

        self.record_budget = None if record_budget is None else int(record_budget)
        self.token_budget = None if token_budget is None else int(token_budget)
        self.summary_max_sentences = int(summary_max_sentences)
        self.scope_metadata_keys = tuple(dict.fromkeys(normalized_scope_keys))

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        self._validate_store_layers(store)
        scoped_groups = self._group_working_records(store)

        effects: list[dict[str, Any]] = []
        evicted_record_ids: list[str] = []
        moved_ltm_record_ids: list[str] = []
        indexed_sentence_record_ids: list[str] = []
        deleted_working_record_ids: list[str] = []
        summary_updates: list[dict[str, Any]] = []

        for group in scoped_groups:
            scope = group["session_scope"]
            records = group["records"]
            retained_records = self._select_retained_records(records)
            retained_ids = {record.record_id for record in retained_records}
            evicted_records = [record for record in records if record.record_id not in retained_ids]
            if not evicted_records:
                continue

            old_summary_records = self._summary_records_for_scope(store, scope)
            old_summary_record = old_summary_records[-1] if old_summary_records else None
            ltm_records = [
                self._build_ltm_episode_record(store, source_record=record, session_scope=scope)
                for record in evicted_records
            ]
            sentence_records = [
                sentence_record
                for ltm_record in ltm_records
                for sentence_record in self._build_ltm_sentence_records(
                    store,
                    episode_record=ltm_record,
                    session_scope=scope,
                )
            ]
            new_summary_text = self._rewrite_summary(
                old_summary_record=old_summary_record,
                evicted_records=evicted_records,
                session_scope=scope,
            )
            new_summary_record = self._build_summary_record(
                store,
                summary_text=new_summary_text,
                session_scope=scope,
                old_summary_record=old_summary_record,
                evicted_records=evicted_records,
            )

            for ltm_record in ltm_records:
                store.append(ltm_record)
            for sentence_record in sentence_records:
                store.append(sentence_record)
            store.append(new_summary_record)
            for old_record in old_summary_records:
                try:
                    store.delete_record(self.summary_layer, old_record.record_id)
                except KeyError:
                    pass
            for evicted_record in evicted_records:
                deleted = store.delete_record(self.working_layer, evicted_record.record_id)
                deleted_working_record_ids.append(deleted.record_id)

            current_evicted_ids = [record.record_id for record in evicted_records]
            current_ltm_ids = [record.record_id for record in ltm_records]
            current_sentence_ids = [record.record_id for record in sentence_records]
            evicted_record_ids.extend(current_evicted_ids)
            moved_ltm_record_ids.extend(current_ltm_ids)
            indexed_sentence_record_ids.extend(current_sentence_ids)
            summary_update = {
                "session_scope": deepcopy(scope),
                "summary_old_record_id": old_summary_record.record_id if old_summary_record is not None else None,
                "summary_new_record_id": new_summary_record.record_id,
            }
            summary_updates.append(summary_update)
            effects.append(
                {
                    "effect_type": "stm_consolidation",
                    "session_scope": deepcopy(scope),
                    "evicted_record_ids": current_evicted_ids,
                    "retained_record_ids": [record.record_id for record in retained_records],
                    "moved_ltm_record_ids": current_ltm_ids,
                    "indexed_sentence_record_ids": current_sentence_ids,
                    "summary_old_record_id": summary_update["summary_old_record_id"],
                    "summary_new_record_id": new_summary_record.record_id,
                }
            )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "working_layer": self.working_layer,
            "ltm_episode_layer": self.ltm_episode_layer,
            "ltm_sentence_layer": self.ltm_sentence_layer,
            "summary_layer": self.summary_layer,
            "record_budget": self.record_budget,
            "token_budget": self.token_budget,
            "summary_max_sentences": self.summary_max_sentences,
            "scope_metadata_keys": list(self.scope_metadata_keys),
            "evicted_record_ids": evicted_record_ids,
            "moved_ltm_record_ids": moved_ltm_record_ids,
            "indexed_sentence_record_ids": indexed_sentence_record_ids,
            "deleted_working_record_ids": deleted_working_record_ids,
            "summary_updates": summary_updates,
            "effects": effects,
        }
        return replace(packet, trace=trace), store

    def _validate_store_layers(self, store: MemoryStore) -> None:
        for layer in (self.working_layer, self.ltm_episode_layer, self.summary_layer):
            if not store.has_layer(layer):
                raise ValueError(
                    f"STMConsolidationEvolution requires layer {layer!r} to be declared in the store topology."
                )

    def _group_working_records(self, store: MemoryStore) -> list[dict[str, Any]]:
        groups: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        for record in store.iter_records(self.working_layer):
            scope = _scope_for_record(record, self.scope_metadata_keys)
            key = _scope_group_key(scope)
            group = groups.setdefault(key, {"session_scope": scope, "records": []})
            group["records"].append(record)
        return list(groups.values())

    def _select_retained_records(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        retained_reversed: list[MemoryRecord] = []
        retained_tokens = 0
        for record in reversed(records):
            if self.record_budget is not None and len(retained_reversed) >= self.record_budget:
                break
            token_count = _record_token_count(record)
            if self.token_budget is not None and retained_reversed and retained_tokens + token_count > self.token_budget:
                break
            retained_reversed.append(record)
            retained_tokens += token_count
        return list(reversed(retained_reversed))

    def _summary_records_for_scope(self, store: MemoryStore, session_scope: dict[str, Any]) -> list[MemoryRecord]:
        return [
            record
            for record in store.iter_records(self.summary_layer)
            if _record_matches_scope(record, session_scope)
        ]

    def _build_ltm_episode_record(
        self,
        store: MemoryStore,
        *,
        source_record: MemoryRecord,
        session_scope: dict[str, Any],
    ) -> MemoryRecord:
        sequence_id = store.next_sequence_id()
        metadata = deepcopy(source_record.metadata)
        metadata["stm_consolidation_source_record_id"] = source_record.record_id
        metadata["stm_consolidation"] = {
            "source_layer": self.working_layer,
            "target_layer": self.ltm_episode_layer,
            "summary_layer": self.summary_layer,
            "source_record_id": source_record.record_id,
            "source_unit_id": source_record.unit_id,
            "session_scope": deepcopy(session_scope),
            "copy_style": "raw_episode_copy_append",
        }
        return MemoryRecord(
            record_id=f"rec-{sequence_id}",
            unit_id=source_record.unit_id,
            layer=self.ltm_episode_layer,
            text=source_record.text,
            timestamp=source_record.timestamp,
            embedding=None if source_record.embedding is None else list(source_record.embedding),
            metadata=metadata,
        )

    def _build_ltm_sentence_records(
        self,
        store: MemoryStore,
        *,
        episode_record: MemoryRecord,
        session_scope: dict[str, Any],
    ) -> list[MemoryRecord]:
        if not store.has_layer(self.ltm_sentence_layer):
            return []

        sentence_records: list[MemoryRecord] = []
        for sentence_index, sentence_text in enumerate(_split_sentences(episode_record.text)):
            sequence_id = store.next_sequence_id()
            metadata = deepcopy(episode_record.metadata)
            provenance = metadata.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
            provenance = {
                **deepcopy(provenance),
                "source": "stm_consolidation_sentence_split",
                "parent_episode_record_id": episode_record.record_id,
                "sentence_index": sentence_index,
            }
            metadata["provenance"] = provenance
            metadata["sentence_index"] = sentence_index
            metadata["parent_episode_record_id"] = episode_record.record_id
            metadata["source_episode_record_id"] = episode_record.record_id
            metadata["stm_consolidation_sentence"] = {
                "episode_layer": self.ltm_episode_layer,
                "sentence_layer": self.ltm_sentence_layer,
                "parent_episode_record_id": episode_record.record_id,
                "sentence_index": sentence_index,
                "session_scope": deepcopy(session_scope),
            }
            sentence_records.append(
                MemoryRecord(
                    record_id=f"rec-{sequence_id}",
                    unit_id=f"{episode_record.unit_id}:sentence:{sentence_index}",
                    layer=self.ltm_sentence_layer,
                    text=sentence_text,
                    timestamp=episode_record.timestamp,
                    embedding=None,
                    metadata=metadata,
                )
            )
        return sentence_records

    def _rewrite_summary(
        self,
        *,
        old_summary_record: MemoryRecord | None,
        evicted_records: list[MemoryRecord],
        session_scope: dict[str, Any],
    ) -> str:
        runtime = get_runtime()
        if hasattr(runtime, "require_llm"):
            runtime.require_llm(capability="STMConsolidationEvolution summary rewrite")
        records_payload: list[dict[str, Any]] = []
        if old_summary_record is not None:
            records_payload.append(self._serialize_summary_input_record(old_summary_record, role="previous_summary"))
        records_payload.extend(
            self._serialize_summary_input_record(record, role="evicted_episode")
            for record in evicted_records
        )
        instruction = (
            "Rewrite the current session summary for the provided STM consolidation scope. "
            "Use the previous summary when present and incorporate the evicted raw STM episodes. "
            f"Keep at most {self.summary_max_sentences} concise factual sentences. "
            f"Session scope: {json.dumps(session_scope, ensure_ascii=False, sort_keys=True, default=str)}."
        )
        summary = runtime.summarize_records(
            records=records_payload,
            instruction=instruction,
            max_sentences=self.summary_max_sentences,
        )
        return str(summary).strip() or "No summary generated."

    @staticmethod
    def _serialize_summary_input_record(record: MemoryRecord, *, role: str) -> dict[str, Any]:
        return {
            "role": role,
            "record_id": record.record_id,
            "unit_id": record.unit_id,
            "layer": record.layer,
            "text": record.text,
            "timestamp": record.timestamp,
            "metadata": deepcopy(record.metadata),
        }

    def _build_summary_record(
        self,
        store: MemoryStore,
        *,
        summary_text: str,
        session_scope: dict[str, Any],
        old_summary_record: MemoryRecord | None,
        evicted_records: list[MemoryRecord],
    ) -> MemoryRecord:
        sequence_id = store.next_sequence_id()
        metadata = {
            **deepcopy(session_scope),
            "unit_type": "summary",
            "stm_consolidation_summary": {
                "source_layer": self.working_layer,
                "ltm_episode_layer": self.ltm_episode_layer,
                "summary_layer": self.summary_layer,
                "session_scope": deepcopy(session_scope),
                "source_record_ids": [record.record_id for record in evicted_records],
                "source_unit_ids": list(dict.fromkeys(record.unit_id for record in evicted_records)),
                "old_summary_record_id": old_summary_record.record_id if old_summary_record is not None else None,
                "summary_max_sentences": self.summary_max_sentences,
                "relation": "stm_consolidation_summary",
                "current": True,
            },
        }
        return MemoryRecord(
            record_id=f"rec-{sequence_id}",
            unit_id=old_summary_record.unit_id if old_summary_record is not None else f"unit-stm-summary-{sequence_id}",
            layer=self.summary_layer,
            text=summary_text,
            timestamp=evicted_records[-1].timestamp,
            metadata=metadata,
        )


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
        strict_tools: bool | None = None,
        max_retry: int = 0,
        raise_on_tool_error: bool = False,
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
        self.max_retry = int(max_retry)
        if self.max_retry < 0:
            raise ValueError("max_retry must be non-negative.")
        self.strict_tools = bool(strict_tools) if strict_tools is not None else False
        self.raise_on_tool_error = bool(strict_tools) if strict_tools is not None else bool(raise_on_tool_error)
        self.allow_no_tool_call = bool(allow_no_tool_call)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embedding_model = embedding_model

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        selected_records, decision_source = self._select_records(packet, store)
        visible_records_holder = {"records": list(selected_records)}
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
        visible_record_ids = [record.record_id for record in visible_records_holder["records"]]
        state, committed_store, retry_trace = self._run_agent_with_retries(
            packet=packet,
            store=store,
            rendered_prompt=rendered_prompt,
            base_context={
                "slot": self.spec.slot,
                "selected_record_ids": [record.record_id for record in selected_records],
            },
            selected_record_ids=[record.record_id for record in selected_records],
            visible_record_ids=visible_record_ids,
        )
        if not state.tool_calls and not self.allow_no_tool_call:
            raise ValueError("LLMFunctionCallEvolution requires at least one successful or attempted tool call.")
        prompt_trace.update(retry_trace)

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "tool_names": [spec.name for spec in self.tool_specs],
            "max_retry": self.max_retry,
            "retry_count": retry_trace["retry_count"],
            "raise_on_tool_error": self.raise_on_tool_error,
            "failed_tool_calls": retry_trace["failed_tool_calls"],
            "attempts": retry_trace["attempts"],
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
        return replace(packet, trace=trace), committed_store

    def _run_agent_with_retries(
        self,
        *,
        packet: Packet,
        store: MemoryStore,
        rendered_prompt: str,
        base_context: dict[str, Any],
        selected_record_ids: list[str],
        visible_record_ids: list[str],
    ) -> tuple[ToolExecutionState, MemoryStore, dict[str, Any]]:
        previous_failures: list[dict[str, Any]] = []
        attempt_traces: list[dict[str, Any]] = []
        final_state = ToolExecutionState()
        final_store = deepcopy(store)

        for attempt_index in range(self.max_retry + 1):
            attempt_store = deepcopy(store)
            selected_ids = set(selected_record_ids)
            visible_ids = set(visible_record_ids)
            selected_records = [record for record in attempt_store.iter_records() if record.record_id in selected_ids]
            visible_records = [record for record in attempt_store.iter_records() if record.record_id in visible_ids]
            context = WriteToolCallContext(
                packet=packet,
                store=attempt_store,
                module_slot="memory_evolution",
                default_target_layer=self.target_layer,
                selected_records=selected_records,
                visible_records=visible_records,
            )
            state = ToolExecutionState()
            runtime_tools = build_runtime_tools(
                self.tool_specs,
                context=context,
                state=state,
                strict_tools=self.strict_tools,
            )
            agent_context = dict(base_context)
            if previous_failures:
                agent_context["retry"] = {
                    "attempt": attempt_index,
                    "max_retry": self.max_retry,
                    "previous_failed_tool_calls": previous_failures,
                    "instruction": "Retry the memory tool calls using the original prompt and these tool error details.",
                }
            self._run_agent(rendered_prompt=rendered_prompt, tools=runtime_tools, context=agent_context)

            failures = failed_tool_calls(state.tool_calls)
            attempt_traces.append(
                {
                    "attempt": attempt_index,
                    "tool_calls": list(state.tool_calls),
                    "effects": list(state.effects),
                    "failed_tool_calls": failures,
                }
            )
            final_state = state
            final_store = context.store
            if not failures:
                break
            previous_failures = failures

        final_failures = failed_tool_calls(final_state.tool_calls)
        if final_failures and self.raise_on_tool_error:
            raise ValueError(f"LLMFunctionCallEvolution tool calls failed: {format_tool_failure_summary(final_failures)}")
        return final_state, commit_store_snapshot(store, final_store), {
            "retry_count": max(0, len(attempt_traces) - 1),
            "failed_tool_calls": final_failures,
            "attempts": attempt_traces,
        }

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
    STMConsolidationEvolution,
    TraceOnlyEvolution,
    SummaryRewriteEvolution,
    LayerMoveEvolution,
    GraphLinkEvolution,
    GraphNeighborContextTraceEvolution,
    HierarchicalEvolution,
    LLMFunctionCallEvolution,
)
