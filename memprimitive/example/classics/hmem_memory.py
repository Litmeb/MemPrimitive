"""H-MEM-style hierarchical memory reconstruction for LoCoMo benchmarking.

Four abstraction layers (domain -> category -> trace -> episode) are written per
dialogue turn with parent ``child_record_ids`` links. Recall uses
``HierarchicalTopDownRoutingRetrieval`` over the same layer order.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import (
    MemoryPipeline,
    MemoryRecord,
    MemoryStore,
    Observation,
    Query,
    Readout,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.benchmarking._types import MemoryIngestEvent, MemoryRecall, RecallContext
from memprimitive.baselines import ConcatenateReadout, HierarchicalTopDownRoutingRetrieval
from memprimitive.utils._runtime import Runtime, get_runtime

HMEM_LAYER_ORDER = ("domain", "category", "trace", "episode")

HMEM_EXTRACTION_PROMPT = (
    "You are an information analysis agent for a long-term LLM system. "
    "Given one dialogue turn, extract hierarchical memory as strict JSON with exactly "
    "these string keys: domain, category, memory_trace, episode, user_profile. "
    "domain: high-level topic domain. "
    "category: specific category or subdomain. "
    "memory_trace: short keyword summary of the turn. "
    "episode: concrete events and facts from the turn. "
    "user_profile: inferred user preferences or traits, or an empty string. "
    "Return JSON only."
)

def build_hmem_memory_system(
    *,
    top_k: int = 10,
    top_k_by_layer: dict[str, int] | None = None,
    return_layer: str = "episode",
    child_id_fields: tuple[str, ...] = ("child_record_ids", "hmem.child_record_ids"),
) -> dict[str, object]:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="domain",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
                StoreLayerSpec(
                    name="category",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
                StoreLayerSpec(
                    name="trace",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
                StoreLayerSpec(
                    name="episode",
                    theme="episodic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
            ]
        )
    )
    recall_pipeline = MemoryPipeline(
        retrieval=HierarchicalTopDownRoutingRetrieval(
            layer_order=HMEM_LAYER_ORDER,
            top_k=top_k,
            top_k_by_layer=top_k_by_layer,
            return_layer=return_layer,
            child_id_fields=child_id_fields,
        ),
        readout=ConcatenateReadout(separator="\n\n"),
        store=store,
    )
    return {
        "store": store,
        "recall_pipeline": recall_pipeline,
        "top_k": _positive_int(top_k, "top_k"),
        "return_layer": return_layer,
    }


def ingest_hmem_turn(
    system: dict[str, object],
    *,
    text: str,
    session_id: str,
    turn_id: str,
    timestamp: str | None = None,
    metadata: dict[str, Any] | None = None,
    runtime: Runtime | None = None,
) -> dict[str, Any]:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    active_runtime = runtime if runtime is not None else get_runtime()
    active_runtime.require_llm(capability="H-MEM hierarchical extraction")

    extracted = _extract_hmem_layers(text, runtime=active_runtime)
    unit_id = f"hmem-{turn_id}"
    record_timestamp = timestamp or _utc_now_iso()
    base_metadata = {
        **dict(metadata or {}),
        "session_id": session_id,
        "turn_id": turn_id,
        "hmem": {
            "domain": extracted["domain"],
            "category": extracted["category"],
            "memory_trace": extracted["memory_trace"],
            "user_profile": extracted["user_profile"],
        },
    }

    episode_record = _append_layer_record(
        store,
        layer="episode",
        unit_id=unit_id,
        text=extracted["episode"],
        timestamp=record_timestamp,
        metadata={
            **base_metadata,
            "user_profile": extracted["user_profile"],
            "memory_weight": 1.0,
        },
        runtime=active_runtime,
    )
    trace_record = _append_layer_record(
        store,
        layer="trace",
        unit_id=unit_id,
        text=extracted["memory_trace"],
        timestamp=record_timestamp,
        metadata={**base_metadata, "child_record_ids": [episode_record.record_id]},
        runtime=active_runtime,
    )
    category_record = _append_layer_record(
        store,
        layer="category",
        unit_id=unit_id,
        text=extracted["category"],
        timestamp=record_timestamp,
        metadata={**base_metadata, "child_record_ids": [trace_record.record_id]},
        runtime=active_runtime,
    )
    domain_record = _append_layer_record(
        store,
        layer="domain",
        unit_id=unit_id,
        text=extracted["domain"],
        timestamp=record_timestamp,
        metadata={**base_metadata, "child_record_ids": [category_record.record_id]},
        runtime=active_runtime,
    )
    return {
        "turn_id": turn_id,
        "record_ids": {
            "domain": domain_record.record_id,
            "category": category_record.record_id,
            "trace": trace_record.record_id,
            "episode": episode_record.record_id,
        },
    }


def recall_hmem_memory(system: dict[str, object], *, user_query: str) -> MemoryRecall:
    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)
    readout = recall_pipeline.recall(Query(text=user_query))
    assert isinstance(readout, Readout)
    return MemoryRecall(
        text=readout.text,
        source_ids=list(readout.source_ids),
        metadata={
            "top_k": system.get("top_k"),
            "return_layer": system.get("return_layer"),
            "num_episodes": len(readout.source_ids),
            **dict(readout.metadata),
        },
    )


class HMEMMemoryBinding:
    """Benchmark binding for the H-MEM hierarchical memory reconstruction."""

    name = "hmem"

    def __init__(
        self,
        *,
        top_k: int = 10,
        top_k_by_layer: dict[str, int] | None = None,
        return_layer: str = "episode",
    ) -> None:
        self.top_k = top_k
        self.top_k_by_layer = top_k_by_layer
        self.return_layer = return_layer

    def build_system(self) -> dict[str, object]:
        return build_hmem_memory_system(
            top_k=self.top_k,
            top_k_by_layer=self.top_k_by_layer,
            return_layer=self.return_layer,
        )

    def ingest_event(self, system: dict[str, object], event: MemoryIngestEvent) -> Any:
        parts = [part.strip() for part in (event.text, event.context_text) if part.strip()]
        attachment_suffix = ""
        raw_blip_caption = event.metadata.get("blip_caption")
        if raw_blip_caption:
            attachment_suffix = f" [ATTACHED: {str(raw_blip_caption).strip()}]"
        text = "\n".join(parts) if parts else ""
        if not text and not attachment_suffix:
            return {"skipped": True, "reason": "empty_turn"}
        return ingest_hmem_turn(
            system,
            text=f"{text}{attachment_suffix}".strip(),
            session_id=event.session_id,
            turn_id=event.turn_id,
            timestamp=event.timestamp,
            metadata={
                "user_id": event.user_id,
                "source_timestamp": event.timestamp or "",
                "source_speaker": event.speaker or event.role,
                **dict(event.metadata),
            },
        )

    def recall(self, system: dict[str, object], query: Query, *, context: RecallContext) -> MemoryRecall:
        del context
        return recall_hmem_memory(system, user_query=query.text)


def create_memory_binding(
    *,
    top_k: int = 10,
    top_k_by_layer: dict[str, int] | None = None,
    return_layer: str = "episode",
) -> HMEMMemoryBinding:
    return HMEMMemoryBinding(
        top_k=top_k,
        top_k_by_layer=top_k_by_layer,
        return_layer=return_layer,
    )


def _extract_hmem_layers(text: str, *, runtime: Runtime) -> dict[str, str]:
    normalized_text = str(text).strip()
    if not normalized_text:
        raise ValueError("H-MEM extraction requires non-empty dialogue text.")
    result = runtime.json(system=HMEM_EXTRACTION_PROMPT, user=normalized_text)
    if not isinstance(result, dict):
        raise ValueError("H-MEM extraction must return a JSON object.")
    return {
        "domain": _required_layer_text(result.get("domain"), field="domain"),
        "category": _required_layer_text(result.get("category"), field="category"),
        "memory_trace": _required_layer_text(result.get("memory_trace"), field="memory_trace"),
        "episode": _required_layer_text(result.get("episode"), field="episode"),
        "user_profile": str(result.get("user_profile", "")).strip(),
    }


def _append_layer_record(
    store: MemoryStore,
    *,
    layer: str,
    unit_id: str,
    text: str,
    timestamp: str,
    metadata: dict[str, Any],
    runtime: Runtime,
) -> MemoryRecord:
    normalized_text = str(text).strip()
    if not normalized_text:
        raise ValueError(f"H-MEM layer {layer!r} requires non-empty text.")
    record = MemoryRecord(
        record_id=f"rec-{store.next_sequence_id()}",
        unit_id=unit_id,
        layer=layer,
        text=normalized_text,
        timestamp=timestamp,
        embedding=list(runtime.embed(normalized_text)),
        metadata=metadata,
    )
    store.append(record)
    return record


def _required_layer_text(value: Any, *, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"H-MEM extraction field {field!r} must be a non-empty string.")
    return normalized


def _positive_int(value: int, name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def main() -> None:
    system = build_hmem_memory_system(top_k=2)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    ingest_hmem_turn(
        system,
        text="Alice asked for action movie recommendations and the assistant suggested a Jackie Chan film.",
        session_id="sess-1",
        turn_id="turn-1",
    )
    ingest_hmem_turn(
        system,
        text="Bob compared train schedules for a weekend trip to Hangzhou.",
        session_id="sess-1",
        turn_id="turn-2",
    )

    recall = recall_hmem_memory(system, user_query="What movie was recommended to Alice?")
    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()
    print("recall:")
    print(recall.text)
    print("source_ids:", recall.source_ids)


if __name__ == "__main__":
    main()
