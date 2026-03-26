"""MemGPT-style classic support modules.

This module keeps the motif sketch deterministic and self-contained:

- ``main_context`` stores the live working buffer.
- ``archival`` stores explicit saves and compaction summaries.
- ``recall`` stores short compacted snapshots that can be re-read on later turns.

The custom primitives below stay within the repo's ``Packet`` / ``MemoryStore``
model and can be composed into a normal :class:`~memprimitive.pipeline.MemoryPipeline`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

from ..baselines._trace import copy_trace
from ..baselines.retrieval import EmbeddingSimilarityRetrieval, LayerAwareRetrieval, RecencyRetrieval
from ..baselines.representation import BasicRepresentation
from ..baselines.unit_formation import PassThroughUnitFormation
from ..core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Observation, Packet, Placement, Readout, StoreLayerSpec, StoreTopology
from ..exceptions import IncompatibleCompositionError
from ..interfaces import EvolutionTriggerModule, MemoryEvolutionModule, OrganizationModule, ReadoutModule, WriteTriggerModule
from ..pipeline import MemoryPipeline
from ._runtime import get_classic_runtime

MEMGPT_MAIN_LAYER: Final[str] = "main_context"
MEMGPT_ARCHIVAL_LAYER: Final[str] = "archival"
MEMGPT_RECALL_LAYER: Final[str] = "recall"
MEMGPT_LAYER_ORDER: Final[tuple[str, ...]] = (
    MEMGPT_RECALL_LAYER,
    MEMGPT_MAIN_LAYER,
    MEMGPT_ARCHIVAL_LAYER,
)

_DEFAULT_TOOL_LAYER_MAP: Final[dict[str, str]] = {
    "memory_save": MEMGPT_ARCHIVAL_LAYER,
    "memory_archive": MEMGPT_RECALL_LAYER,
}


def _memgpt_hints(metadata: dict[str, Any] | None) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return hints
    nested = metadata.get("memgpt")
    if isinstance(nested, dict):
        hints.update(nested)
    for key in ("tool", "target_layer", "write", "compact", "budget", "readout_budget"):
        if key in metadata and key not in hints:
            hints[key] = metadata[key]
    return hints


def _unit_hints(unit: MemoryUnit) -> dict[str, Any]:
    return _memgpt_hints(unit.metadata)


def _normalize_target_layer(raw: Any, fallback: str) -> str:
    value = str(raw).strip() if raw is not None else ""
    return value or fallback


def _compact_text(text: str, *, limit: int = 72) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)].rstrip() + "..."


def _keywords_from_records(records: list[MemoryRecord]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for record in records:
        for token in record.text.casefold().split():
            token = token.strip(".,;:!?()[]{}<>\"'")
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= 8:
                return keywords
    return keywords


def _summarize_records(records: list[MemoryRecord], *, kind: str, limit: int = 3) -> str:
    if not records:
        return f"{kind} summary: no records"
    runtime = get_classic_runtime()
    summary = runtime.summarize_records(
        records=[
            {
                "record_id": record.record_id,
                "layer": record.layer,
                "text": record.text,
            }
            for record in records[: max(limit, len(records))]
        ],
        instruction=(
            f"Write a compact MemGPT {kind} summary. "
            "Preserve actionable details and source-record references when useful."
        ),
    ).strip()
    if not summary:
        return f"{kind} summary: no records"
    return summary if summary.casefold().startswith(f"{kind} summary".casefold()) else f"{kind} summary: {summary}"


def _summary_unit(records: list[MemoryRecord], *, kind: str, target_layer: str) -> MemoryUnit:
    summary_text = _summarize_records(records, kind=kind)
    source_record_ids = [record.record_id for record in records]
    return MemoryUnit(
        text=summary_text,
        unit_type="summary",
        metadata={
            "memgpt": {
                "summary_kind": kind,
                "source_record_ids": source_record_ids,
                "target_layer": target_layer,
            },
            "compaction": {
                "kind": kind,
                "source_record_ids": source_record_ids,
                "target_layer": target_layer,
            },
            "representation": {
                "summary": summary_text,
                "keywords": _keywords_from_records(records),
                "source_record_ids": source_record_ids,
            },
        },
    )


def build_memgpt_store() -> MemoryStore:
    """Build a three-layer MemGPT-style store topology."""

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name=MEMGPT_MAIN_LAYER,
                theme="working",
                capacity="token_limited",
                indices=("temporal", "keyword"),
            ),
            StoreLayerSpec(
                name=MEMGPT_ARCHIVAL_LAYER,
                theme="semantic",
                capacity="unlimited",
                indices=("vector", "keyword", "temporal"),
            ),
            StoreLayerSpec(
                name=MEMGPT_RECALL_LAYER,
                theme="working",
                capacity="sliding_window",
                indices=("temporal", "keyword"),
            ),
        ]
    )
    return MemoryStore(topology=topology)


def memgpt_observation(
    text: str,
    *,
    source: str = "dialogue",
    target_layer: str | None = None,
    tool: str | None = None,
    compact: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    """Create an observation with MemGPT routing hints in ``metadata``."""

    hints: dict[str, Any] = {}
    if target_layer is not None:
        hints["target_layer"] = target_layer
    if tool is not None:
        hints["tool"] = tool
    if compact:
        hints["compact"] = True
    payload = {} if metadata is None else dict(metadata)
    if hints:
        payload["memgpt"] = hints
    return Observation(text=text, source=source, metadata=payload)


class MemGPTWriteTrigger(WriteTriggerModule):
    """Gate writes with explicit MemGPT routing hints.

    Plain dialogue defaults to ``True`` so observations still enter the live
    ``main_context`` buffer. A unit can opt out by setting ``memgpt.write=False``
    or by using a read-only tool hint. Any explicit target layer or write tool
    keeps the decision open.
    """

    spec = ModuleSpec(
        name="memgpt_write_trigger",
        slot="write_trigger",
        input_requirements=("units",),
        output_guarantees=("decisions",),
    )

    _read_only_tools: Final[frozenset[str]] = frozenset({"memory_recall", "memory_retrieval", "NO_EXECUTE"})

    def __init__(self, *, default_write: bool = True) -> None:
        self.default_write = default_write

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemGPTWriteTrigger requires packet.units.")

        decisions: list[bool] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            hints = _unit_hints(unit)
            tool = _normalize_target_layer(hints.get("tool"), "")
            target_layer = _normalize_target_layer(hints.get("target_layer"), "")
            explicit_write = hints.get("write")
            decision = self.default_write
            if explicit_write is False:
                decision = False
            elif tool and tool in self._read_only_tools:
                decision = False
            elif target_layer or tool:
                decision = True

            decisions.append(decision)
            per_unit_trace.append(
                {
                    "unit_id": unit.unit_id,
                    "tool": tool or None,
                    "target_layer": target_layer or None,
                    "decision": decision,
                }
            )

        trace = copy_trace(packet)
        trace["write_trigger"] = {
            "module": self.spec.name,
            "policy": "explicit_target_or_tool",
            "default_write": self.default_write,
            "per_unit": per_unit_trace,
        }
        return replace(packet, decisions=decisions, trace=trace), store


class MemGPTOrganization(OrganizationModule):
    """Route each write to ``main_context``, ``archival``, or ``recall``.

    The default path writes into ``main_context``. Explicit hints on the unit or
    its source metadata can redirect the write to archival or recall storage.
    """

    spec = ModuleSpec(
        name="memgpt_organization",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        default_layer: str = MEMGPT_MAIN_LAYER,
        tool_target_layers: dict[str, str] | None = None,
    ) -> None:
        self.default_layer = default_layer
        self.tool_target_layers = dict(_DEFAULT_TOOL_LAYER_MAP if tool_target_layers is None else tool_target_layers)

    def validate_store(self, store: MemoryStore) -> None:
        required_layers = {self.default_layer, *self.tool_target_layers.values()}
        missing = [layer for layer in required_layers if not store.has_layer(layer)]
        if missing:
            raise IncompatibleCompositionError(
                f"MemGPTOrganization requires declared layer(s) {missing}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemGPTOrganization requires packet.units.")
        if packet.decisions is None:
            raise ValueError("MemGPTOrganization requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("MemGPTOrganization requires decisions aligned with units.")

        placements: list[Placement] = []
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        per_unit_trace: list[dict[str, Any]] = []

        for unit, decision in zip(packet.units, packet.decisions, strict=True):
            target_layer = self._target_layer_for_unit(unit)
            placements.append(Placement(unit_id=unit.unit_id, target_layer=target_layer))
            per_unit_trace.append(
                {
                    "unit_id": unit.unit_id,
                    "target_layer": target_layer,
                    "decision": decision,
                }
            )
            if not decision:
                continue

            store.ensure_layer(target_layer)
            sequence_id = store.next_sequence_id()
            record = MemoryRecord.from_unit(unit=unit, layer=target_layer, sequence_id=sequence_id)
            store.append(record)
            written_record_ids.append(record.record_id)
            written_unit_ids.append(unit.unit_id)

        trace = copy_trace(packet)
        trace["organization"] = {
            "module": self.spec.name,
            "default_layer": self.default_layer,
            "placements": [
                {"unit_id": placement.unit_id, "target_layer": placement.target_layer}
                for placement in placements
            ],
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "per_unit": per_unit_trace,
        }
        return replace(packet, placements=placements, trace=trace), store

    def _target_layer_for_unit(self, unit: MemoryUnit) -> str:
        hints = _unit_hints(unit)
        target_layer = hints.get("target_layer")
        if target_layer:
            return _normalize_target_layer(target_layer, self.default_layer)

        tool = _normalize_target_layer(hints.get("tool"), "")
        if tool in self.tool_target_layers:
            return self.tool_target_layers[tool]

        return self.default_layer


class MemGPTEvolutionTrigger(EvolutionTriggerModule):
    """Trigger compaction when the live ``main_context`` budget is exceeded."""

    spec = ModuleSpec(
        name="memgpt_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(
        self,
        *,
        main_context_layer: str = MEMGPT_MAIN_LAYER,
        main_context_budget: int = 3,
    ) -> None:
        if main_context_budget <= 0:
            raise ValueError("MemGPTEvolutionTrigger requires main_context_budget > 0.")
        self.main_context_layer = main_context_layer
        self.main_context_budget = int(main_context_budget)

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.main_context_layer):
            raise IncompatibleCompositionError(
                f"MemGPTEvolutionTrigger requires declared layer {self.main_context_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemGPTEvolutionTrigger requires packet.units.")
        if packet.placements is None:
            raise ValueError("MemGPTEvolutionTrigger requires packet.placements.")
        if len(packet.units) != len(packet.placements):
            raise ValueError("MemGPTEvolutionTrigger requires aligned units and placements.")

        hints = [_unit_hints(unit) for unit in packet.units]
        force_compact = any(bool(hint.get("compact")) for hint in hints)
        main_context_count = store.count(self.main_context_layer)
        should_compact = force_compact or main_context_count > self.main_context_budget
        decisions = [should_compact for _ in packet.units]

        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "main_context_layer": self.main_context_layer,
            "main_context_budget": self.main_context_budget,
            "main_context_count": main_context_count,
            "force_compact": force_compact,
            "should_compact": should_compact,
            "evolution_decisions": decisions,
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class MemGPTCompactionEvolution(MemoryEvolutionModule):
    """Compact overflow from ``main_context`` into archival and recall summaries."""

    spec = ModuleSpec(
        name="memgpt_compaction_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        main_context_layer: str = MEMGPT_MAIN_LAYER,
        archival_layer: str = MEMGPT_ARCHIVAL_LAYER,
        recall_layer: str = MEMGPT_RECALL_LAYER,
        main_context_budget: int = 3,
        recall_budget: int = 2,
    ) -> None:
        if main_context_budget <= 0:
            raise ValueError("MemGPTCompactionEvolution requires main_context_budget > 0.")
        if recall_budget <= 0:
            raise ValueError("MemGPTCompactionEvolution requires recall_budget > 0.")
        self.main_context_layer = main_context_layer
        self.archival_layer = archival_layer
        self.recall_layer = recall_layer
        self.main_context_budget = int(main_context_budget)
        self.recall_budget = int(recall_budget)

    def validate_store(self, store: MemoryStore) -> None:
        for layer_name in (self.main_context_layer, self.archival_layer, self.recall_layer):
            if not store.has_layer(layer_name):
                raise IncompatibleCompositionError(
                    f"MemGPTCompactionEvolution requires declared layer {layer_name!r}."
                )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemGPTCompactionEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("MemGPTCompactionEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("MemGPTCompactionEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("MemGPTCompactionEvolution requires aligned units, placements, and evolution decisions.")

        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]
        effects: list[dict[str, Any]] = []

        if active_unit_ids:
            main_records = store.iter_records(self.main_context_layer)
            overflow_count = max(0, len(main_records) - self.main_context_budget)
            if overflow_count > 0:
                overflow_records = main_records[:overflow_count]
                archival_summary = _summary_unit(
                    overflow_records,
                    kind="archival_compaction",
                    target_layer=self.archival_layer,
                )
                archival_record = MemoryRecord.from_unit(
                    archival_summary,
                    layer=self.archival_layer,
                    sequence_id=store.next_sequence_id(),
                )
                store.append(archival_record)
                effects.append(
                    {
                        "effect_type": "archive_compaction",
                        "source_record_ids": [record.record_id for record in overflow_records],
                        "record_id": archival_record.record_id,
                        "target_layer": self.archival_layer,
                    }
                )

                recall_window = main_records[-self.recall_budget :]
                recall_summary = _summary_unit(
                    recall_window,
                    kind="recall_window",
                    target_layer=self.recall_layer,
                )
                recall_record = MemoryRecord.from_unit(
                    recall_summary,
                    layer=self.recall_layer,
                    sequence_id=store.next_sequence_id(),
                )
                store.append(recall_record)
                effects.append(
                    {
                        "effect_type": "recall_compaction",
                        "source_record_ids": [record.record_id for record in recall_window],
                        "record_id": recall_record.record_id,
                        "target_layer": self.recall_layer,
                    }
                )

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "effects": effects,
            "main_context_layer": self.main_context_layer,
            "archival_layer": self.archival_layer,
            "recall_layer": self.recall_layer,
            "main_context_budget": self.main_context_budget,
            "recall_budget": self.recall_budget,
        }
        return replace(packet, trace=trace), store


class MemGPTReadout(ReadoutModule):
    """Render retrieval results as a budgeted, layer-grouped prompt chunk."""

    spec = ModuleSpec(
        name="memgpt_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, *, item_budget: int = 4, layer_order: tuple[str, ...] = MEMGPT_LAYER_ORDER) -> None:
        if item_budget <= 0:
            raise ValueError("MemGPTReadout requires item_budget > 0.")
        self.item_budget = int(item_budget)
        self.layer_order = layer_order

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("MemGPTReadout requires packet.retrieved.")

        items = packet.retrieved.items[: self.item_budget]
        omitted = max(0, len(packet.retrieved.items) - len(items))
        source_ids = [record.record_id for record in items]

        grouped: dict[str, list[str]] = {}
        for record in items:
            grouped.setdefault(record.layer, []).append(record.text)

        chunks: list[str] = []
        ordered_layers = list(self.layer_order) + [
            layer for layer in grouped if layer not in self.layer_order
        ]
        for layer in ordered_layers:
            texts = grouped.get(layer)
            if not texts:
                continue
            chunks.append(f"[{layer}]\n" + "\n".join(f"- {text}" for text in texts))
        if omitted:
            chunks.append(f"... {omitted} more item(s) omitted by readout budget")

        readout = Readout(
            text="\n\n".join(chunks),
            source_ids=source_ids,
            metadata={
                "item_count": len(items),
                "omitted_item_count": omitted,
                "item_budget": self.item_budget,
                "layer_counts": {layer: len(texts) for layer, texts in grouped.items()},
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "item_budget": self.item_budget,
        }
        return replace(packet, readout=readout, trace=trace), store


def build_memgpt_pipeline(
    *,
    top_k: int = 4,
    main_context_budget: int = 3,
    recall_budget: int = 2,
    readout_item_budget: int = 4,
) -> MemoryPipeline:
    """Assemble a deterministic MemGPT-style pipeline."""

    store = build_memgpt_store()
    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "keywords", "tags")),
        write_trigger=MemGPTWriteTrigger(),
        organization=MemGPTOrganization(),
        evolution_trigger=MemGPTEvolutionTrigger(
            main_context_budget=main_context_budget,
        ),
        memory_evolution=MemGPTCompactionEvolution(
            main_context_budget=main_context_budget,
            recall_budget=recall_budget,
        ),
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=top_k, layer=MEMGPT_MAIN_LAYER),
            retriever_by_layer={
                MEMGPT_MAIN_LAYER: RecencyRetrieval(top_k=top_k, layer=MEMGPT_MAIN_LAYER),
                MEMGPT_RECALL_LAYER: RecencyRetrieval(top_k=top_k, layer=MEMGPT_RECALL_LAYER),
                MEMGPT_ARCHIVAL_LAYER: EmbeddingSimilarityRetrieval(top_k=top_k, layer=MEMGPT_ARCHIVAL_LAYER),
            },
            active_layers=MEMGPT_LAYER_ORDER,
            top_k=top_k,
            merge_weight_by_layer={
                MEMGPT_RECALL_LAYER: 1.5,
                MEMGPT_MAIN_LAYER: 1.2,
                MEMGPT_ARCHIVAL_LAYER: 1.0,
            },
        ),
        readout=MemGPTReadout(item_budget=readout_item_budget),
    )


__all__ = [
    "MEMGPT_ARCHIVAL_LAYER",
    "MEMGPT_LAYER_ORDER",
    "MEMGPT_MAIN_LAYER",
    "MEMGPT_RECALL_LAYER",
    "MemGPTCompactionEvolution",
    "MemGPTOrganization",
    "MemGPTEvolutionTrigger",
    "MemGPTReadout",
    "MemGPTWriteTrigger",
    "build_memgpt_pipeline",
    "build_memgpt_store",
    "memgpt_observation",
]
