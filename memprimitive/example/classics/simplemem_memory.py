"""Mechanism-level reconstruction of SimpleMem-style lifelong memory.

This example composes existing MemPrimitive baseline modules into SimpleMem's
three-stage pipeline (semantic structured compression, online synthesis during
write, and intent-aware hybrid retrieval) without adding new baseline modules.

Design notes aligned with the official SimpleMem repo rather than every paper
detail:
- write-time synthesis is prompt-based deduplication against recent entries;
- semantic density gating is implicit via empty LLM extraction output;
- hybrid recall uses three labeled sub-recall pipelines inside TemplateReadout;
- symbolic retrieval is orchestrated by a small example-local RetrievalModule;
- reflection and adaptive retrieval depth are intentionally disabled.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import (
    MemoryPipeline,
    MemoryRecord,
    MemoryStore,
    ModuleSpec,
    Observation,
    Packet,
    Query,
    Readout,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.benchmarking._types import MemoryIngestEvent, MemoryRecall, RecallContext
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    BM25Retrieval,
    ConfigurableEmbeddingRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    LLMRepresentation,
    NeverTrigger,
    PassThroughUnitFormation,
    QueryRewriteRetrieval,
    TemplateReadout,
)
from memprimitive.interfaces import RetrievalModule
from memprimitive.utils._template import structured_prompt, text_prompt
from memprimitive.utils._trace import copy_trace


DEFAULT_WINDOW_SIZE = 40
DEFAULT_SEMANTIC_TOP_K = 25
DEFAULT_KEYWORD_TOP_K = 5
DEFAULT_STRUCTURED_TOP_K = 5
DEFAULT_PLANNING_MAX_QUERIES = 5
MEMORY_LAYER = "memory_units"


class _SimpleMemSymbolicRetrieval(RetrievalModule):
    """Example-local symbolic retrieval orchestration for SimpleMem."""

    spec = ModuleSpec(
        name="simplemem_symbolic_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, top_k: int = DEFAULT_STRUCTURED_TOP_K, layer: str = MEMORY_LAYER) -> None:
        if top_k <= 0:
            raise ValueError("_SimpleMemSymbolicRetrieval requires top_k > 0.")
        self.top_k = int(top_k)
        self.layer = str(layer).strip() or MEMORY_LAYER

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("_SimpleMemSymbolicRetrieval requires packet.query.")

        analysis = _analyze_query_for_symbolic(packet.query.text)
        matched = _structured_symbolic_search(
            store,
            analysis,
            layer=self.layer,
            top_k=self.top_k,
        )
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "strategy": "simplemem_symbolic",
            }
            for rank, record in enumerate(matched, start=1)
        ]
        retrieved = RetrievedSet(
            items=matched,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "layer": self.layer,
                "analysis": analysis,
                "matched_count": len(matched),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


def build_simplemem_memory_system(
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
    keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
    structured_top_k: int = DEFAULT_STRUCTURED_TOP_K,
    planning_max_queries: int = DEFAULT_PLANNING_MAX_QUERIES,
) -> dict[str, object]:
    window_size = _positive_int(window_size, "window_size")
    semantic_top_k = _positive_int(semantic_top_k, "semantic_top_k")
    keyword_top_k = _positive_int(keyword_top_k, "keyword_top_k")
    structured_top_k = _positive_int(structured_top_k, "structured_top_k")
    planning_max_queries = _positive_int(planning_max_queries, "planning_max_queries")

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name=MEMORY_LAYER,
                    theme="semantic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                ),
            ]
        )
    )

    system_state: dict[str, object] = {
        "store": store,
        "window_size": window_size,
        "dialogue_buffer": [],
        "previous_entry_summaries": [],
        "session_id": "default",
    }

    memory_unit_write_pipeline = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text",)),
            ConfigurableEmbeddingRepresentation(),
        ),
        organization=AppendOrganization(target_layer=MEMORY_LAYER),
        store=store,
    )

    window_extract_pipeline = MemoryPipeline(
        representation=LLMRepresentation(
            field="memory_entries",
            value_type=list[dict[str, str]],
            prompt=_window_extraction_prompt(system_state),
        ),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=AppendOrganization(target_layer=MEMORY_LAYER),
        store=store,
    )

    semantic_recall_pipeline = MemoryPipeline(
        retrieval=QueryRewriteRetrieval(
            retriever=EmbeddingSimilarityRetrieval(top_k=semantic_top_k, layer=MEMORY_LAYER),
            strategy="llm",
            allow_multi_query=True,
            include_original=True,
            max_queries=planning_max_queries,
            prompt=_semantic_planning_prompt(planning_max_queries),
        ),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )
    lexical_recall_pipeline = MemoryPipeline(
        retrieval=BM25Retrieval(top_k=keyword_top_k, layer=MEMORY_LAYER),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )
    symbolic_recall_pipeline = MemoryPipeline(
        retrieval=_SimpleMemSymbolicRetrieval(top_k=structured_top_k, layer=MEMORY_LAYER),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )

    recall_pipeline = MemoryPipeline(
        readout=TemplateReadout(
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "header",
                            "title": "SIMPLEMEM MEMORIES",
                            "template": "<SIMPLEMEM MEMORIES>",
                        },
                        {
                            "id": "semantic",
                            "title": "Semantic",
                            "template": "{{ semantic }}",
                            "condition": "semantic",
                        },
                        {
                            "id": "lexical",
                            "title": "Lexical",
                            "template": "{{ lexical }}",
                            "condition": "lexical",
                        },
                        {
                            "id": "symbolic",
                            "title": "Symbolic",
                            "template": "{{ symbolic }}",
                            "condition": "symbolic",
                        },
                    ]
                },
                metadata_mode="readout",
                recall_query_builder=lambda packet, store, context: str(context["query"]["text"]),
                labeled_recall_plans={
                    "semantic": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
                    "lexical": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
                    "symbolic": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
                },
                labeled_sub_recall_pipelines={
                    "semantic": semantic_recall_pipeline,
                    "lexical": lexical_recall_pipeline,
                    "symbolic": symbolic_recall_pipeline,
                },
                visible_record_recall_labels=("semantic", "lexical", "symbolic"),
            )
        ),
        store=store,
    )

    system_state.update(
        {
            "memory_unit_write_pipeline": memory_unit_write_pipeline,
            "window_extract_pipeline": window_extract_pipeline,
            "recall_pipeline": recall_pipeline,
            "semantic_top_k": semantic_top_k,
            "keyword_top_k": keyword_top_k,
            "structured_top_k": structured_top_k,
            "planning_max_queries": planning_max_queries,
        }
    )
    return system_state


def ingest_dialogue_line(
    system: dict[str, object],
    *,
    speaker: str,
    content: str,
    session_id: str,
    timestamp: str | None = None,
) -> None:
    speaker_text = str(speaker).strip() or "speaker"
    content_text = str(content).strip()
    if not content_text:
        return

    system["session_id"] = str(session_id).strip() or "default"
    buffer = _dialogue_buffer(system)
    buffer.append(
        {
            "speaker": speaker_text,
            "content": content_text,
            "timestamp": timestamp or "",
        }
    )
    window_size = _positive_int(system.get("window_size", DEFAULT_WINDOW_SIZE), "window_size")
    while len(buffer) >= window_size:
        _flush_window(system, session_id=system["session_id"], window_size=window_size)


def finalize_simplemem_ingest(system: dict[str, object], *, session_id: str | None = None) -> None:
    if session_id is not None:
        system["session_id"] = str(session_id).strip() or "default"
    _flush_window(
        system,
        session_id=str(system.get("session_id", "default")),
        window_size=_positive_int(system.get("window_size", DEFAULT_WINDOW_SIZE), "window_size"),
        final=True,
    )


def recall_simplemem_memory(system: dict[str, object], *, user_query: str) -> MemoryRecall:
    finalize_simplemem_ingest(system, session_id=str(system.get("session_id", "default")))
    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)
    readout = recall_pipeline.recall(Query(text=user_query))
    assert isinstance(readout, Readout)
    source_ids = list(readout.source_ids)
    if not source_ids:
        metadata = readout.metadata if isinstance(readout.metadata, dict) else {}
        source_ids = _normalize_record_ids(metadata.get("visible_record_ids"))
    return MemoryRecall(
        text=readout.text.strip(),
        source_ids=source_ids,
        metadata={
            "semantic_top_k": system.get("semantic_top_k"),
            "keyword_top_k": system.get("keyword_top_k"),
            "structured_top_k": system.get("structured_top_k"),
            "planning_max_queries": system.get("planning_max_queries"),
            "memory_unit_count": _count_layer_records(system, MEMORY_LAYER),
            "readout_metadata": dict(readout.metadata),
        },
    )


class SimpleMemMemoryBinding:
    """Benchmark binding for the SimpleMem classics reconstruction."""

    name = "simplemem"

    def __init__(
        self,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
        structured_top_k: int = DEFAULT_STRUCTURED_TOP_K,
        planning_max_queries: int = DEFAULT_PLANNING_MAX_QUERIES,
    ) -> None:
        self.window_size = window_size
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.structured_top_k = structured_top_k
        self.planning_max_queries = planning_max_queries

    def build_system(self) -> dict[str, object]:
        return build_simplemem_memory_system(
            window_size=self.window_size,
            semantic_top_k=self.semantic_top_k,
            keyword_top_k=self.keyword_top_k,
            structured_top_k=self.structured_top_k,
            planning_max_queries=self.planning_max_queries,
        )

    def ingest_event(self, system: dict[str, object], event: MemoryIngestEvent) -> Any:
        session_id = event.session_id
        speaker = event.speaker or event.role or "speaker"
        if event.text.strip():
            ingest_dialogue_line(
                system,
                speaker=speaker,
                content=event.text,
                session_id=session_id,
                timestamp=event.timestamp,
            )
        if event.context_text.strip():
            ingest_dialogue_line(
                system,
                speaker=_other_speaker(speaker),
                content=event.context_text,
                session_id=session_id,
                timestamp=event.timestamp,
            )
        return None

    def recall(self, system: dict[str, object], query: Query, *, context: RecallContext) -> MemoryRecall:
        del context
        return recall_simplemem_memory(system, user_query=query.text)


def create_memory_binding(
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
    keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
    structured_top_k: int = DEFAULT_STRUCTURED_TOP_K,
    planning_max_queries: int = DEFAULT_PLANNING_MAX_QUERIES,
) -> SimpleMemMemoryBinding:
    return SimpleMemMemoryBinding(
        window_size=window_size,
        semantic_top_k=semantic_top_k,
        keyword_top_k=keyword_top_k,
        structured_top_k=structured_top_k,
        planning_max_queries=planning_max_queries,
    )


def _flush_window(
    system: dict[str, object],
    *,
    session_id: str,
    window_size: int,
    final: bool = False,
) -> None:
    buffer = _dialogue_buffer(system)
    if not buffer:
        return

    if final:
        window = list(buffer)
        buffer.clear()
    else:
        window = buffer[:window_size]
        del buffer[:window_size]

    dialogue_text = "\n".join(
        f"[{item.get('timestamp') or ''}] {item['speaker']}: {item['content']}" for item in window
    )
    window_extract_pipeline = system["window_extract_pipeline"]
    memory_unit_write_pipeline = system["memory_unit_write_pipeline"]
    assert isinstance(window_extract_pipeline, MemoryPipeline)
    assert isinstance(memory_unit_write_pipeline, MemoryPipeline)

    packet = window_extract_pipeline.ingest(
        Observation(
            text=dialogue_text,
            source="simplemem_window",
            metadata={"session_id": session_id, "window_dialogue_count": len(window)},
        )
    )
    entries = _extract_memory_entries(packet)
    new_summaries: list[str] = []
    for entry in entries:
        restatement = str(entry.get("lossless_restatement", "")).strip()
        if not restatement:
            continue
        memory_unit_write_pipeline.ingest(
            Observation(
                text=restatement,
                timestamp=_entry_timestamp(entry) or _window_timestamp(window),
                source="simplemem_memory_unit",
                metadata=_entry_metadata(entry, session_id=session_id),
            )
        )
        new_summaries.append(restatement)

    if new_summaries:
        previous = list(system.get("previous_entry_summaries", []))
        previous.extend(new_summaries)
        system["previous_entry_summaries"] = previous[-3:]


def _extract_memory_entries(packet: Packet) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for unit in packet.units or []:
        representation = unit.metadata.get("representation", {})
        if not isinstance(representation, dict):
            continue
        raw_entries = representation.get("memory_entries", [])
        if isinstance(raw_entries, list):
            for item in raw_entries:
                if isinstance(item, dict):
                    entries.append({str(key): str(value) for key, value in item.items() if str(value).strip()})
    return entries


def _entry_metadata(entry: dict[str, str], *, session_id: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"session_id": session_id}
    for key in ("location", "topic"):
        value = str(entry.get(key, "")).strip()
        if value and value.casefold() not in {"null", "none"}:
            metadata[key] = value
    memory_timestamp = _entry_timestamp(entry)
    if memory_timestamp:
        metadata["memory_timestamp"] = memory_timestamp
    for key in ("keywords", "persons", "entities"):
        values = _split_csv_field(entry.get(key, ""))
        if values:
            metadata[key] = values
    return metadata


def _entry_timestamp(entry: dict[str, str]) -> str:
    raw = str(entry.get("timestamp", "")).strip()
    if not raw or raw.casefold() in {"null", "none"}:
        return ""
    return raw


def _window_timestamp(window: list[dict[str, str]]) -> str:
    for item in reversed(window):
        raw = str(item.get("timestamp", "")).strip()
        if raw:
            return raw
    return datetime.now().astimezone().isoformat()


def _structured_symbolic_search(
    store: MemoryStore,
    analysis: dict[str, Any],
    *,
    layer: str,
    top_k: int,
) -> list[MemoryRecord]:
    persons = [str(item).strip() for item in analysis.get("persons", []) if str(item).strip()]
    location = str(analysis.get("location") or "").strip()
    entities = [str(item).strip() for item in analysis.get("entities", []) if str(item).strip()]
    time_start = _parse_iso_timestamp(analysis.get("time_start"))
    time_end = _parse_iso_timestamp(analysis.get("time_end"))

    if not any([persons, location, entities, time_start, time_end]):
        return []

    matched: list[MemoryRecord] = []
    for record in reversed(list(store.iter_records(layer))):
        if persons and not _record_matches_any(record, "persons", persons):
            continue
        if location and not _record_matches_text(record, "location", location):
            continue
        if entities and not _record_matches_any(record, "entities", entities):
            continue
        if (time_start or time_end) and not _record_matches_time_range(record, time_start, time_end):
            continue
        matched.append(record)
        if len(matched) >= top_k:
            break
    return matched


def _analyze_query_for_symbolic(query_text: str) -> dict[str, Any]:
    from memprimitive.utils._runtime import get_runtime

    runtime = get_runtime()
    runtime.require_llm(capability="SimpleMem symbolic query analysis")
    prompt = f"""
Analyze the following query and extract structured retrieval constraints.

Query: {query_text}

Return JSON with:
- keywords: list of important keywords
- persons: list of person names
- location: location string or null
- entities: list of organizations/products/other entities
- time_start: ISO-8601 start timestamp or null
- time_end: ISO-8601 end timestamp or null

Return ONLY JSON.
""".strip()
    try:
        payload = runtime.json(
            system="You analyze queries for structured memory retrieval. Return strict JSON only.",
            user=prompt,
            temperature=0.1,
        )
    except Exception:
        return {"keywords": [query_text], "persons": [], "location": None, "entities": [], "time_start": None, "time_end": None}

    if not isinstance(payload, dict):
        return {"keywords": [query_text], "persons": [], "location": None, "entities": [], "time_start": None, "time_end": None}
    return {
        "keywords": _normalize_string_list(payload.get("keywords")) or [query_text],
        "persons": _normalize_string_list(payload.get("persons")),
        "entities": _normalize_string_list(payload.get("entities")),
        "location": _nullable_text(payload.get("location")),
        "time_start": _nullable_text(payload.get("time_start")),
        "time_end": _nullable_text(payload.get("time_end")),
    }


def _record_matches_any(record: MemoryRecord, field: str, targets: list[str]) -> bool:
    values = _record_field_values(record, field)
    lowered_targets = {value.casefold() for value in targets}
    return any(value in lowered_targets for value in values)


def _record_matches_text(record: MemoryRecord, field: str, target: str) -> bool:
    target_text = target.casefold()
    return any(target_text in value or value in target_text for value in _record_field_values(record, field))


def _record_field_values(record: MemoryRecord, field: str) -> set[str]:
    values: set[str] = set()
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    raw = metadata.get(field)
    if isinstance(raw, str) and raw.strip():
        values.add(raw.strip().casefold())
    elif isinstance(raw, (list, tuple, set)):
        values.update(str(item).strip().casefold() for item in raw if str(item).strip())

    representation = metadata.get("representation", {})
    if isinstance(representation, dict):
        rep_value = representation.get(field)
        if isinstance(rep_value, str) and rep_value.strip():
            values.add(rep_value.strip().casefold())
        elif isinstance(rep_value, (list, tuple, set)):
            values.update(str(item).strip().casefold() for item in rep_value if str(item).strip())
    return values


def _record_matches_time_range(
    record: MemoryRecord,
    time_start: datetime | None,
    time_end: datetime | None,
) -> bool:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    candidates = [
        metadata.get("memory_timestamp"),
        record.timestamp,
    ]
    for raw in candidates:
        parsed = _parse_iso_timestamp(raw)
        if parsed is None:
            continue
        if time_start and parsed < time_start:
            continue
        if time_end and parsed > time_end:
            continue
        return True
    return False


def _window_extraction_prompt(system_state: dict[str, object]):
    return text_prompt(
        _WINDOW_EXTRACTION_TEMPLATE,
        context_builder=lambda packet, store: {
            "previous_memories": _format_previous_entries(system_state.get("previous_entry_summaries", [])),
            "window_dialogues": packet.observation.text if packet.observation is not None else "",
        },
    )


def _semantic_planning_prompt(max_queries: int):
    bounded = _positive_int(max_queries, "planning_max_queries")
    return text_prompt(
        "You plan semantic retrieval for a SimpleMem-style memory system.\n"
        "Given the user query, return JSON with key `queries` containing 1 to {{ max_queries }} focused search queries.\n"
        "Include paraphrases, entity-specific lookups, and temporal/context variants when useful.\n"
        "Prefer fewer, sharper queries over exhaustive expansion.\n"
        "Return ONLY JSON.",
        context_builder=lambda packet, store: {
            "query": packet.query.text if packet.query is not None else "",
            "max_queries": bounded,
        },
    )


def _format_previous_entries(entries: object) -> str:
    if not isinstance(entries, list) or not entries:
        return ""
    lines = [f"- {str(item).strip()}" for item in entries if str(item).strip()]
    return "\n".join(lines)


def _dialogue_buffer(system: dict[str, object]) -> list[dict[str, str]]:
    buffer = system.get("dialogue_buffer")
    if not isinstance(buffer, list):
        buffer = []
        system["dialogue_buffer"] = buffer
    return buffer


def _split_csv_field(raw: object) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.casefold() in {"null", "none", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _normalize_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return _split_csv_field(value)
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _nullable_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.casefold() in {"null", "none"}:
        return None
    return text


def _parse_iso_timestamp(value: object) -> datetime | None:
    text = _nullable_text(value)
    if text is None:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _other_speaker(current: str) -> str:
    normalized = str(current).strip().casefold()
    if normalized in {"user", "speaker_a", "a"}:
        return "assistant"
    if normalized in {"assistant", "speaker_b", "b"}:
        return "user"
    return "other"


def _normalize_record_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _positive_int(value: int, name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _count_layer_records(system: dict[str, object], layer: str) -> int:
    store = system.get("store")
    if not isinstance(store, MemoryStore):
        return 0
    return store.count(layer)


_WINDOW_EXTRACTION_TEMPLATE = """
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

[Previous Window Memory Entries (for reference to avoid duplication)]
{{ previous_memories }}

[Current Window Dialogues]
{{ window_dialogues }}

[Requirements]
1. Complete Coverage: generate enough memory entries to ensure all useful information is captured.
2. Force Disambiguation: prohibit pronouns and relative time expressions.
3. Lossless Information: each lossless_restatement must be a complete standalone sentence.
4. If the window is low-information chit-chat with no durable facts, return an empty JSON array [].
5. Use comma-separated strings for keywords, persons, and entities.

[Output Format]
Return a JSON array. Each element must use string values only:
- lossless_restatement
- keywords
- timestamp (ISO-8601 or null)
- location (or null)
- persons
- entities
- topic

Return ONLY the JSON array.
""".strip()


def main() -> None:
    system = build_simplemem_memory_system(window_size=2)
    session_id = "demo-session"

    ingest_dialogue_line(
        system,
        speaker="Alice",
        content="Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product.",
        session_id=session_id,
        timestamp="2025-11-15T14:30:00",
    )
    ingest_dialogue_line(
        system,
        speaker="Bob",
        content="Sure, I'll prepare the market analysis report.",
        session_id=session_id,
        timestamp="2025-11-15T14:31:00",
    )
    ingest_dialogue_line(
        system,
        speaker="Alice",
        content="Remember to bring the notes from our last planning session.",
        session_id=session_id,
        timestamp="2025-11-15T14:32:00",
    )

    recall = recall_simplemem_memory(system, user_query="When and where will Alice and Bob meet?")
    print(recall.text)
    print("source_ids:", recall.source_ids)


if __name__ == "__main__":
    main()
