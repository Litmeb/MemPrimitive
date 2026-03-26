"""MemoryBank-style support primitives for the classic examples."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

from ..baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    ConditionalLayerOrganization,
    EmbeddingSimilarityRetrieval,
    GroupedByLayerReadout,
    LayerAwareRetrieval,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from ..baselines._trace import copy_trace
from ..core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    ModuleSpec,
    Packet,
    StoreLayerSpec,
    StoreTopology,
)
from ..exceptions import IncompatibleCompositionError
from ..interfaces import EvolutionTriggerModule, MemoryEvolutionModule
from ..pipeline import MemoryPipeline
from ._runtime import get_classic_runtime

_SHORT_TERM_DEFAULT_WINDOW: Final[int] = 3


@dataclass(slots=True)
class MemoryBankConfig:
    """Configuration for the classic MemoryBank motif."""

    short_term_layer: str = "short_term"
    long_term_layer: str = "long_term"
    short_term_window: int = _SHORT_TERM_DEFAULT_WINDOW
    short_term_retrieval_k: int = 3
    long_term_retrieval_k: int = 5
    combined_retrieval_k: int = 6
    merge_prefix: str = "MemoryBank merge"
    summary_prefix: str = "MemoryBank summary"


def build_memorybank_topology(config: MemoryBankConfig | None = None) -> StoreTopology:
    """Build the two-layer short-term / long-term topology used by MemoryBank."""

    config = config or MemoryBankConfig()
    return StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name=config.short_term_layer,
                theme="working",
                capacity="token_limited",
                indices=("temporal",),
            ),
            StoreLayerSpec(
                name=config.long_term_layer,
                theme="semantic",
                indices=("vector", "temporal", "keyword", "entity"),
            ),
        ]
    )


class MemoryBankEvolutionTrigger(EvolutionTriggerModule):
    """Activate extra evolution when short-term overflows or long-term entities arrive."""

    spec = ModuleSpec(
        name="memory_bank_evolution_trigger",
        slot="evolution_trigger",
        input_requirements=("units", "placements"),
        output_guarantees=("evolution_decisions",),
    )

    def __init__(
        self,
        *,
        short_term_layer: str = "short_term",
        long_term_layer: str = "long_term",
        short_term_window: int = _SHORT_TERM_DEFAULT_WINDOW,
    ) -> None:
        self.short_term_layer = short_term_layer
        self.long_term_layer = long_term_layer
        self.short_term_window = short_term_window

    def validate_store(self, store: MemoryStore) -> None:
        for layer_name in (self.short_term_layer, self.long_term_layer):
            if not store.has_layer(layer_name):
                raise IncompatibleCompositionError(
                    f"MemoryBankEvolutionTrigger requires declared layer {layer_name!r}."
                )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemoryBankEvolutionTrigger requires packet.units.")
        if packet.placements is None:
            raise ValueError("MemoryBankEvolutionTrigger requires packet.placements.")

        short_term_count = store.count(self.short_term_layer)
        overflow = short_term_count > self.short_term_window
        decisions: list[bool] = []
        per_unit: list[dict[str, Any]] = []

        for unit, placement in zip(packet.units, packet.placements, strict=True):
            decision = False
            reason = "inactive"
            if placement.target_layer == self.long_term_layer and unit.entities:
                decision = True
                reason = "entity_merge"
            elif placement.target_layer == self.short_term_layer and overflow:
                decision = True
                reason = "short_term_overflow"

            decisions.append(decision)
            per_unit.append(
                {
                    "unit_id": unit.unit_id,
                    "target_layer": placement.target_layer,
                    "decision": decision,
                    "reason": reason,
                }
            )

        trace = copy_trace(packet)
        trace["evolution_trigger"] = {
            "module": self.spec.name,
            "short_term_layer": self.short_term_layer,
            "long_term_layer": self.long_term_layer,
            "short_term_count": short_term_count,
            "short_term_window": self.short_term_window,
            "overflow": overflow,
            "per_unit": per_unit,
        }
        return replace(packet, evolution_decisions=decisions, trace=trace), store


class MemoryBankEvolution(MemoryEvolutionModule):
    """Summarize overflowing short-term memories and merge repeated entities in long-term."""

    spec = ModuleSpec(
        name="memory_bank_evolution",
        slot="memory_evolution",
        input_requirements=("units", "placements", "evolution_decisions"),
        output_guarantees=("trace.memory_evolution.effects",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        short_term_layer: str = "short_term",
        long_term_layer: str = "long_term",
        short_term_window: int = _SHORT_TERM_DEFAULT_WINDOW,
        merge_prefix: str = "MemoryBank merge",
        summary_prefix: str = "MemoryBank summary",
    ) -> None:
        self.short_term_layer = short_term_layer
        self.long_term_layer = long_term_layer
        self.short_term_window = short_term_window
        self.merge_prefix = merge_prefix
        self.summary_prefix = summary_prefix

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.short_term_layer):
            raise IncompatibleCompositionError(
                f"MemoryBankEvolution requires declared layer {self.short_term_layer!r}."
            )
        if not store.has_layer(self.long_term_layer):
            raise IncompatibleCompositionError(
                f"MemoryBankEvolution requires declared layer {self.long_term_layer!r}."
            )
        if not store.layer_supports_index(self.long_term_layer, "entity"):
            raise IncompatibleCompositionError(
                f"MemoryBankEvolution requires entity index on layer {self.long_term_layer!r}."
            )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("MemoryBankEvolution requires packet.units.")
        if packet.placements is None:
            raise ValueError("MemoryBankEvolution requires packet.placements.")
        if packet.evolution_decisions is None:
            raise ValueError("MemoryBankEvolution requires packet.evolution_decisions.")
        if not (len(packet.units) == len(packet.placements) == len(packet.evolution_decisions)):
            raise ValueError("MemoryBankEvolution requires aligned units, placements, and evolution decisions.")

        active_unit_ids = [
            unit.unit_id
            for unit, decision in zip(packet.units, packet.evolution_decisions, strict=True)
            if decision
        ]
        if not active_unit_ids:
            trace = copy_trace(packet)
            trace["memory_evolution"] = {
                "module": self.spec.name,
                "decision_source": "evolution_decisions",
                "active_unit_ids": [],
                "effects": [],
            }
            return replace(packet, trace=trace), store

        effects: list[dict[str, Any]] = []
        effects.extend(self._summarize_short_term_overflow(store))
        effects.extend(self._merge_long_term_entity_clusters(store))

        trace = copy_trace(packet)
        trace["memory_evolution"] = {
            "module": self.spec.name,
            "decision_source": "evolution_decisions",
            "active_unit_ids": active_unit_ids,
            "short_term_window": self.short_term_window,
            "effects": effects,
        }
        return replace(packet, trace=trace), store

    def _summarize_short_term_overflow(self, store: MemoryStore) -> list[dict[str, Any]]:
        short_term_records = store.iter_records(self.short_term_layer)
        overflow_count = max(0, len(short_term_records) - self.short_term_window)
        if overflow_count <= 0:
            return []

        overflow_records = short_term_records[:overflow_count]
        summary_record = self._make_summary_record(
            overflow_records,
            store=store,
            layer=self.long_term_layer,
            kind="short_term_summary",
            heading=self.summary_prefix,
        )
        store.layers[self.short_term_layer] = short_term_records[overflow_count:]
        store.append(summary_record)
        return [
            {
                "effect_type": "summarize_and_prune",
                "layer": self.short_term_layer,
                "summary_record_id": summary_record.record_id,
                "pruned_record_ids": [record.record_id for record in overflow_records],
                "preserved_record_ids": [record.record_id for record in store.iter_records(self.short_term_layer)],
            }
        ]

    def _merge_long_term_entity_clusters(self, store: MemoryStore) -> list[dict[str, Any]]:
        long_term_records = store.iter_records(self.long_term_layer)
        grouped: dict[tuple[str, ...], list[MemoryRecord]] = {}
        for record in long_term_records:
            signature = _record_entity_signature(record)
            if not signature:
                continue
            grouped.setdefault(signature, []).append(record)

        merge_effects: list[dict[str, Any]] = []
        records_to_remove: set[str] = set()
        merged_records: list[MemoryRecord] = []
        for signature, records in grouped.items():
            if len(records) < 2:
                continue
            records_to_remove.update(record.record_id for record in records)
            merged_record = self._make_summary_record(
                records,
                store=store,
                layer=self.long_term_layer,
                kind="entity_merge",
                heading=f"{self.merge_prefix}: {', '.join(_display_entities(records, signature))}",
                entity_signature=signature,
            )
            merged_records.append(merged_record)
            merge_effects.append(
                {
                    "effect_type": "entity_merge",
                    "entity_signature": list(signature),
                    "source_record_ids": [record.record_id for record in records],
                    "merged_record_id": merged_record.record_id,
                }
            )

        if not merged_records:
            return []

        kept_records = [record for record in long_term_records if record.record_id not in records_to_remove]
        store.layers[self.long_term_layer] = kept_records
        for merged_record in merged_records:
            store.append(merged_record)
        return merge_effects

    def _make_summary_record(
        self,
        records: list[MemoryRecord],
        *,
        store: MemoryStore,
        layer: str,
        kind: str,
        heading: str,
        entity_signature: tuple[str, ...] | None = None,
    ) -> MemoryRecord:
        summary_text = _compose_summary_text(records, heading=heading)
        entities = list(entity_signature) if entity_signature is not None else _record_entity_union(records)
        timestamp = max(record.timestamp for record in records)
        unit = MemoryUnit(
            text=summary_text,
            unit_type="summary",
            timestamp=timestamp,
            representation_elements=("text", "entities", "summary"),
            entities=entities,
            tags=["summary", kind],
            metadata={
                "memorybank": {
                    "kind": kind,
                    "source_record_ids": [record.record_id for record in records],
                    "source_layers": sorted({record.layer for record in records}),
                },
                "representation": {
                    "summary": summary_text,
                    "entities": entities,
                },
            },
        )
        return MemoryRecord.from_unit(unit=unit, layer=layer, sequence_id=store.next_sequence_id())


def build_memorybank_pipeline(
    *,
    config: MemoryBankConfig | None = None,
    store: MemoryStore | None = None,
) -> MemoryPipeline:
    """Build a compact MemoryBank pipeline with routing, consolidation, and layered recall."""

    config = config or MemoryBankConfig()
    topology = build_memorybank_topology(config)
    memory_store = store if store is not None else MemoryStore(topology=topology)
    return MemoryPipeline(
        store=memory_store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "tags", "keywords")),
        write_trigger=AlwaysWriteTrigger(),
        organization=ConditionalLayerOrganization(
            default_layer=config.short_term_layer,
            rules=(
                {"has_entity": True, "target_layer": config.long_term_layer},
                {"unit_type": "summary", "target_layer": config.long_term_layer},
            ),
        ),
        evolution_trigger=MemoryBankEvolutionTrigger(
            short_term_layer=config.short_term_layer,
            long_term_layer=config.long_term_layer,
            short_term_window=config.short_term_window,
        ),
        memory_evolution=MemoryBankEvolution(
            short_term_layer=config.short_term_layer,
            long_term_layer=config.long_term_layer,
            short_term_window=config.short_term_window,
            merge_prefix=config.merge_prefix,
            summary_prefix=config.summary_prefix,
        ),
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=config.short_term_retrieval_k, layer=config.short_term_layer),
            retriever_by_layer={
                config.short_term_layer: RecencyRetrieval(top_k=config.short_term_retrieval_k, layer=config.short_term_layer),
                config.long_term_layer: EmbeddingSimilarityRetrieval(
                    top_k=config.long_term_retrieval_k,
                    layer=config.long_term_layer,
                ),
            },
            active_layers=(config.short_term_layer, config.long_term_layer),
            top_k=config.combined_retrieval_k,
            top_k_by_layer={
                config.short_term_layer: config.short_term_retrieval_k,
                config.long_term_layer: config.long_term_retrieval_k,
            },
            merge_weight_by_layer={
                config.short_term_layer: 1.1,
                config.long_term_layer: 1.0,
            },
        ),
        readout=GroupedByLayerReadout(),
    )


def _record_entity_signature(record: MemoryRecord) -> tuple[str, ...]:
    representation = record.metadata.get("representation", {})
    if not isinstance(representation, dict):
        return ()
    entities = representation.get("entities", [])
    if not isinstance(entities, list):
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        text = str(entity).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return tuple(sorted(normalized))


def _record_entities(record: MemoryRecord) -> list[str]:
    representation = record.metadata.get("representation", {})
    if not isinstance(representation, dict):
        return []
    entities = representation.get("entities", [])
    if not isinstance(entities, list):
        return []
    return [str(entity).strip() for entity in entities if str(entity).strip()]


def _record_entity_union(records: list[MemoryRecord]) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for record in records:
        for entity in _record_entities(record):
            key = entity.casefold()
            if key in seen:
                continue
            seen.add(key)
            collected.append(entity)
    return collected


def _display_entities(records: list[MemoryRecord], signature: tuple[str, ...]) -> list[str]:
    display_by_key: dict[str, str] = {}
    for record in records:
        for entity in _record_entities(record):
            display_by_key.setdefault(entity.casefold(), entity)
    return [display_by_key.get(key, key) for key in signature]


def _compose_summary_text(records: list[MemoryRecord], *, heading: str) -> str:
    runtime = get_classic_runtime()
    summary = runtime.summarize_records(
        records=[
            {
                "record_id": record.record_id,
                "layer": record.layer,
                "text": record.text,
                "entities": _record_entities(record),
            }
            for record in records
        ],
        instruction=(
            f"Produce a concise MemoryBank summary with heading '{heading}'. "
            "Retain concrete facts, repeated entities, and avoid bullets."
        ),
    ).strip()
    if not summary:
        return heading
    return summary if summary.startswith(heading) else f"{heading}: {summary}"


def _compress_text(text: str, *, max_length: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


__all__ = [
    "MemoryBankConfig",
    "MemoryBankEvolution",
    "MemoryBankEvolutionTrigger",
    "build_memorybank_pipeline",
    "build_memorybank_topology",
]
