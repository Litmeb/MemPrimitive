"""Compact MemMachine-style memory-layer reconstruction.

This example covers the memory layer only: working memory, LTM episodic
sentence indexing, contextualized episodic recall, and structured profile
features. It intentionally leaves the optional Retrieval Agent router outside
the classics example.
"""

from __future__ import annotations

import sys
from copy import deepcopy
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
    AlwaysTrigger,
    BasicRepresentation,
    EmbeddingSimilarityRetrieval,
    EpisodeClusterRerankRetrieval,
    LLMFunctionCallEvolution,
    ParentEpisodeExpansionRetrieval,
    RecencyRetrieval,
    STMConsolidationEvolution,
    TemplateReadout,
    TemporalNeighborExpansionRetrieval,
)
from memprimitive.utils._profile_feature_tools import build_profile_feature_tools
from memprimitive.utils._template import structured_prompt


def build_memmachine_memory_system(
    *,
    stm_record_budget: int = 20,
    profile_max_turns: int = 6,
    limit: int = 30,
    expand_context: int = 3,
    sentence_top_k: int | None = None,
    episode_top_k: int | None = None,
    profile_top_k: int = 10,
) -> dict[str, object]:
    episode_limit = _positive_int(episode_top_k if episode_top_k is not None else limit, "limit")
    sentence_candidate_k = _positive_int(
        sentence_top_k if sentence_top_k is not None else min(5 * episode_limit, 200),
        "sentence_top_k",
    )
    backward, forward = _neighbor_window(expand_context=expand_context, limit=episode_limit)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working", theme="working", indices=("temporal",)),
                StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
                StoreLayerSpec(
                    name="sentence",
                    theme="episodic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
                StoreLayerSpec(name="session_summary", theme="semantic", indices=("temporal",)),
                StoreLayerSpec(
                    name="profile",
                    theme="semantic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text"}},
                ),
            ]
        )
    )

    write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer="working"),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=(
            LLMFunctionCallEvolution(
                target_layer="profile",
                tools=build_profile_feature_tools(module_name="memmachine_profile"),
                prompt=_profile_prompt(),
                max_turns=_positive_int(profile_max_turns, "profile_max_turns"),
            ),
            STMConsolidationEvolution(record_budget=stm_record_budget, copy_evicted_to_ltm=False),
        ),
        store=store,
    )
    episodic_recall_pipeline = MemoryPipeline(
        retrieval=(
            EmbeddingSimilarityRetrieval(top_k=sentence_candidate_k, layer="sentence"),
            ParentEpisodeExpansionRetrieval(top_k=sentence_candidate_k, episode_layer="episodic"),
            TemporalNeighborExpansionRetrieval(layer="episodic", backward=backward, forward=forward),
            EpisodeClusterRerankRetrieval(top_k=episode_limit),
        ),
        readout=TemplateReadout(prompt=_timestamped_episode_prompt("LONG TERM MEMORY EPISODES")),
        store=store,
    )
    return {
        "store": store,
        "write_pipeline": write_pipeline,
        "limit": episode_limit,
        "expand_context": _positive_int(expand_context, "expand_context"),
        "working_recall_pipeline": MemoryPipeline(
            retrieval=RecencyRetrieval(top_k=episode_limit, layer="working"),
            readout=TemplateReadout(prompt=_timestamped_episode_prompt("SHORT TERM MEMORY EPISODES")),
            store=store,
        ),
        "summary_recall_pipeline": MemoryPipeline(
            retrieval=RecencyRetrieval(top_k=1, layer="session_summary"),
            readout=TemplateReadout(prompt=_summary_prompt()),
            store=store,
        ),
        "episodic_recall_pipeline": episodic_recall_pipeline,
        "profile_recall_pipeline": MemoryPipeline(
            retrieval=EmbeddingSimilarityRetrieval(top_k=profile_top_k, layer="profile"),
            readout=TemplateReadout(prompt=_profile_readout_prompt()),
            store=store,
        ),
    }


def ingest_episode(
    system: dict[str, object],
    *,
    text: str,
    session_id: str,
    user_id: str,
    agent_id: str = "agent",
    producer: str = "user",
    timestamp: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline, MemoryPipeline)
    observation_kwargs: dict[str, Any] = {}
    if timestamp is not None:
        observation_kwargs["timestamp"] = timestamp
    observation_metadata = dict(metadata or {})
    observation_metadata.setdefault("source_timestamp", timestamp or "")
    observation_metadata.setdefault("source_speaker", producer)
    observation_metadata.setdefault("attachment_suffix", "")
    packet = write_pipeline.ingest(
        Observation(
            text=text,
            source=producer,
            metadata={
                **observation_metadata,
                "producer": producer,
                "session_id": session_id,
                "user_id": user_id,
                "agent_id": agent_id,
            },
            **observation_kwargs,
        )
    )
    _append_direct_ltm_index(system, packet)
    return packet


def recall_memmachine_memory(
    system: dict[str, object],
    *,
    user_query: str,
    include_profile: bool = False,
) -> MemoryRecall:
    query = Query(text=user_query)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    episodic_readout = system["episodic_recall_pipeline"].recall(query)
    working_readout = system["working_recall_pipeline"].recall(query)
    summary_readout = system["summary_recall_pipeline"].recall(query)
    assert isinstance(episodic_readout, Readout)
    assert isinstance(working_readout, Readout)
    assert isinstance(summary_readout, Readout)

    working_records = _records_for_source_ids(store, working_readout.source_ids)
    working_unit_ids = {record.unit_id for record in working_records}
    episodic_records = [
        record for record in _records_for_source_ids(store, episodic_readout.source_ids)
        if record.unit_id not in working_unit_ids
    ]
    episode_readout = _render_records(
        store,
        query=query,
        records=[*episodic_records, *working_records],
        title="LONG TERM MEMORY EPISODES",
    )

    sections = [episode_readout.text]
    source_ids = [*episode_readout.source_ids, *summary_readout.source_ids]
    if summary_readout.source_ids and summary_readout.text.strip():
        sections.append(summary_readout.text)

    profile_text = ""
    if include_profile:
        profile_readout = system["profile_recall_pipeline"].recall(query)
        assert isinstance(profile_readout, Readout)
        profile_text = profile_readout.text.strip() if profile_readout.source_ids else ""
        if profile_text:
            sections.append(profile_text)
            source_ids.extend(profile_readout.source_ids)

    text = "\n".join(section for section in sections if section.strip())
    return MemoryRecall(
        text=text,
        source_ids=_stable_source_ids(source_ids),
        metadata={
            "limit": system.get("limit"),
            "expand_context": system.get("expand_context"),
            "num_long_term_episodes": len(episodic_records),
            "num_short_term_episodes": len(working_records),
            "has_working_memory_summary": bool(summary_readout.source_ids),
            "profile_memories": profile_text,
            "num_profile_memories": _count_nonempty_lines(profile_text),
        },
    )


def recall_memmachine_context(system: dict[str, object], *, user_query: str) -> str:
    return recall_memmachine_memory(system, user_query=user_query, include_profile=True).text


class MemMachineMemoryBinding:
    """Benchmark binding for the classic MemMachine memory-layer reconstruction."""

    name = "memmachine"

    def __init__(
        self,
        *,
        stm_record_budget: int = 20,
        profile_max_turns: int = 6,
        limit: int = 30,
        expand_context: int = 3,
        sentence_top_k: int | None = None,
        episode_top_k: int | None = None,
        profile_top_k: int = 10,
    ) -> None:
        self.stm_record_budget = stm_record_budget
        self.profile_max_turns = profile_max_turns
        self.limit = limit
        self.expand_context = expand_context
        self.sentence_top_k = sentence_top_k
        self.episode_top_k = episode_top_k
        self.profile_top_k = profile_top_k

    def build_system(self) -> dict[str, object]:
        return build_memmachine_memory_system(
            stm_record_budget=self.stm_record_budget,
            profile_max_turns=self.profile_max_turns,
            limit=self.limit,
            expand_context=self.expand_context,
            sentence_top_k=self.sentence_top_k,
            episode_top_k=self.episode_top_k,
            profile_top_k=self.profile_top_k,
        )

    def ingest_event(self, system: dict[str, object], event: MemoryIngestEvent) -> Any:
        text = "\n".join(part for part in (event.text, event.context_text) if part.strip())
        attachment_suffix = ""
        raw_blip_caption = event.metadata.get("blip_caption")
        if raw_blip_caption:
            attachment_suffix = f" [ATTACHED: {raw_blip_caption}]"
        return ingest_episode(
            system,
            text=text,
            session_id=event.session_id,
            user_id=event.user_id,
            producer=event.speaker or event.role,
            timestamp=event.timestamp,
            metadata={
                "turn_id": event.turn_id,
                "source_timestamp": event.timestamp or "",
                "source_speaker": event.speaker or event.role,
                "attachment_suffix": attachment_suffix,
                **dict(event.metadata),
            },
        )

    def recall(self, system: dict[str, object], query: Query, *, context: RecallContext) -> MemoryRecall:
        del context
        return recall_memmachine_memory(system, user_query=query.text, include_profile=False)


def create_memory_binding(
    *,
    stm_record_budget: int = 20,
    profile_max_turns: int = 6,
    limit: int = 30,
    expand_context: int = 3,
    sentence_top_k: int | None = None,
    episode_top_k: int | None = None,
    profile_top_k: int = 10,
) -> MemMachineMemoryBinding:
    return MemMachineMemoryBinding(
        stm_record_budget=stm_record_budget,
        profile_max_turns=profile_max_turns,
        limit=limit,
        expand_context=expand_context,
        sentence_top_k=sentence_top_k,
        episode_top_k=episode_top_k,
        profile_top_k=profile_top_k,
    )


def _positive_int(value: int, name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _neighbor_window(*, expand_context: int, limit: int) -> tuple[int, int]:
    normalized = min(max(0, int(expand_context)), max(0, int(limit) - 1))
    backward = normalized // 3
    forward = normalized - backward
    return backward, forward


def _append_direct_ltm_index(system: dict[str, object], packet: Packet) -> None:
    store = system["store"]
    assert isinstance(store, MemoryStore)
    working_ids = list(packet.trace.get("organization", {}).get("written_record_ids", []))
    for unit_index, unit in enumerate(packet.units or []):
        working_record_id = str(working_ids[unit_index]).strip() if unit_index < len(working_ids) else ""
        sequence_id = store.next_sequence_id()
        metadata = deepcopy(unit.metadata)
        source_speaker = str(metadata.get("source_speaker") or metadata.get("producer") or metadata.get("source") or "")
        source_timestamp = str(metadata.get("source_timestamp") or unit.timestamp)
        metadata.update(
            {
                "source_speaker": source_speaker,
                "source_timestamp": source_timestamp,
                "direct_ltm_index": {
                    "source": "memmachine_immediate_episode_index",
                    "source_working_record_id": working_record_id,
                    "source_unit_id": unit.unit_id,
                },
            }
        )
        episode_record = MemoryRecord(
            record_id=f"rec-{sequence_id}",
            unit_id=unit.unit_id,
            layer="episodic",
            text=unit.text,
            timestamp=unit.timestamp,
            embedding=None if unit.embedding is None else list(unit.embedding),
            metadata=metadata,
        )
        store.append(episode_record)
        for sentence_index, sentence_text in enumerate(_split_episode_sentences(episode_record.text)):
            sentence_sequence_id = store.next_sequence_id()
            sentence_metadata = deepcopy(episode_record.metadata)
            provenance = sentence_metadata.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
            sentence_metadata["provenance"] = {
                **deepcopy(provenance),
                "source": "memmachine_immediate_sentence_split",
                "parent_episode_record_id": episode_record.record_id,
                "sentence_index": sentence_index,
            }
            sentence_metadata["sentence_index"] = sentence_index
            sentence_metadata["parent_episode_record_id"] = episode_record.record_id
            sentence_metadata["source_episode_record_id"] = episode_record.record_id
            store.append(
                MemoryRecord(
                    record_id=f"rec-{sentence_sequence_id}",
                    unit_id=f"{episode_record.unit_id}:sentence:{sentence_index}",
                    layer="sentence",
                    text=sentence_text,
                    timestamp=episode_record.timestamp,
                    metadata=sentence_metadata,
                )
            )


def _split_episode_sentences(text: str) -> list[str]:
    import re

    raw_parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    sentences = [part.strip() for part in raw_parts if part.strip()]
    return sentences or [text.strip()]


def _records_for_source_ids(store: MemoryStore, source_ids: list[str]) -> list[MemoryRecord]:
    by_id = {record.record_id: record for record in store.iter_records()}
    return [by_id[record_id] for record_id in source_ids if record_id in by_id]


def _render_records(
    store: MemoryStore,
    *,
    query: Query,
    records: list[MemoryRecord],
    title: str,
) -> Readout:
    packet = Packet(
        query=query,
        retrieved=RetrievedSet(
            items=list(records),
            scores=[{"record_id": record.record_id, "rank": index} for index, record in enumerate(records, start=1)],
            trace={"module": "memmachine_combined_episode_readout", "title": title},
        ),
    )
    readout_packet, _ = TemplateReadout(prompt=_timestamped_episode_prompt(title)).run(packet, store)
    if readout_packet.readout is None:
        return Readout(text="", source_ids=[])
    return readout_packet.readout


def _stable_source_ids(source_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_source_id in source_ids:
        source_id = str(raw_source_id).strip()
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        ordered.append(source_id)
    return ordered


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in str(text).splitlines() if line.strip())


def _timestamped_episode_prompt(title: str):
    open_tag = f"<{title}>"
    close_tag = f"</{title}>"
    return structured_prompt(
        {
            "separator": "\n",
            "blocks": [
                {
                    "id": "open",
                    "template": open_tag,
                },
                {
                    "id": "episodes",
                    "repeat_over": "retrieved.items",
                    "item_template": (
                        "[{{ item.metadata.source_timestamp | default(item.timestamp) }}] "
                        "{{ item.metadata.source_speaker | default('speaker') }}: "
                        "{{ item.text }}{{ item.metadata.attachment_suffix }}"
                    ),
                    "separator": "\n",
                },
                {
                    "id": "close",
                    "template": close_tag,
                },
            ]
        }
    )


def _summary_prompt():
    return structured_prompt(
        {
            "separator": "\n",
            "blocks": [
                {"id": "open", "template": "<WORKING MEMORY SUMMARY>"},
                {"id": "summary", "repeat_over": "retrieved.items", "item_template": "{{ item.text }}", "separator": "\n"},
                {"id": "close", "template": "</WORKING MEMORY SUMMARY>"},
            ],
        }
    )


def _profile_readout_prompt():
    return structured_prompt(
        {
            "separator": "\n",
            "blocks": [
                {"id": "open", "template": "<PROFILE MEMORY>"},
                {
                    "id": "profile",
                    "repeat_over": "retrieved.items",
                    "item_template": "[{{ item.timestamp }}] {{ item.text }}",
                    "separator": "\n",
                },
                {"id": "close", "template": "</PROFILE MEMORY>"},
            ],
        }
    )


def _profile_prompt():
    return structured_prompt(
        {
            "blocks": [
                {
                    "id": "task",
                    "title": "Task",
                    "template": (
                        "Extract stable user profile features from the selected raw episode records.\n"
                        "Use UPSERT_PROFILE_FEATURE for durable preferences, facts, roles, and habits. "
                        "Use DELETE_PROFILE_FEATURE only for explicit contradictions.\n"
                        "Default target_layer={{ default_target_layer }}. "
                        "Preserve source episode ids in tool arguments."
                    ),
                },
                {
                    "id": "selected_records",
                    "title": "Selected Raw Episodes",
                    "repeat_over": "selected_records",
                    "item_template": (
                        "record_id={{ item.record_id }}\n"
                        "timestamp={{ item.timestamp }}\n"
                        "text={{ item.text }}\n"
                        "metadata={{ item.metadata }}"
                    ),
                    "separator": "\n\n",
                },
                {
                    "id": "tools",
                    "title": "Available Tools",
                    "repeat_over": "tools",
                    "item_template": "- {{ item.name }}",
                    "separator": "\n",
                },
            ]
        }
    )


def main() -> None:
    system = build_memmachine_memory_system(stm_record_budget=2)
    for index, text in enumerate(
        [
            "Alice is comparing memory systems for long-running agents.",
            "Alice says she prefers jasmine tea during late-night coding.",
            "The assistant suggests keeping raw episodes available for audit.",
        ],
        start=1,
    ):
        ingest_episode(
            system,
            text=text,
            session_id="sess-demo",
            user_id="alice",
            timestamp=f"2026-04-28T00:0{index}:00Z",
        )

    store = system["store"]
    assert isinstance(store, MemoryStore)
    pprint({layer: store.count(layer) for layer in store.topology.layer_names})
    print(recall_memmachine_context(system, user_query="What tea does Alice prefer?"))


if __name__ == "__main__":
    main()
