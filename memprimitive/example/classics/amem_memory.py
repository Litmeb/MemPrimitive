"""Mechanism-level reconstruction of A-MEM / Agentic Memory.

This example follows the same standards as the other ``classics`` files in
this repository: it aims to preserve the paper/repo memory mechanism inside the
shared MemPrimitive primitive language, not to clone the upstream storage stack
or API byte-for-byte.

For A-MEM, the core loop we want to preserve is:

1. turn a new interaction into one enriched memory note,
2. embed that note for similarity-based retrieval,
3. append it into a graph-capable note store,
4. expose the current note plus nearby candidates to one bounded function-call controller,
5. strengthen current-note links and optionally rewrite neighbor context/tags via A-MEM-specific tools, and
6. answer future queries by embedding-similarity top-k retrieval.

That loop is exactly the level of fidelity this file is trying to demonstrate.
The result is a mechanism-level A-MEM reconstruction expressed entirely with
existing baseline modules.

Important alignment note: the paper text suggests neighbor evolution may update
context, keywords, and tags, while the released repos are more concretely
implemented around context/tag updates and graph linking. This example follows
the repo-consistent write/evolution path, but keeps recall aligned to the paper
itself: plain embedding retrieval over notes rather than graph-neighbor
expansion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.benchmarking._types import MemoryIngestEvent, MemoryRecall, RecallContext
from memprimitive.baselines import (
    AlwaysTrigger,
    ConfigurableEmbeddingRepresentation,
    EmbeddingSimilarityRetrieval,
    GraphAppendOrganization,
    LLMFunctionCallEvolution,
    LLMRepresentation,
    NoteRenderReadout,
    PassThroughUnitFormation,
)
from memprimitive.utils._amem_family import build_amem_evolution_tools
from memprimitive.utils._template import PRIMARY_RECALL_LABEL, structured_prompt, text_prompt
from memprimitive.utils._runtime import get_runtime


def build_amem_memory_system(
    *,
    note_namespace: str = "amem",
    candidate_k: int = 5,
    recall_top_k: int = 5,
) -> dict[str, object]:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="knowledge_graph",
                theme="semantic",
                shape="Graph",
                indices=("graph", "vector"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=(
            LLMRepresentation(
                field="context",
                prompt=(
                    "Write one concise context sentence for this memory note. "
                    "Preserve concrete facts and make the broader relevance explicit."
                ),
            ),
            LLMRepresentation(
                field="keywords",
                value_type=list[str],
                prompt="Extract 3 to 6 short keywords for this memory note as a JSON array of strings.",
            ),
            LLMRepresentation(
                field="tags",
                value_type=list[str],
                prompt="Assign 2 to 4 compact semantic tags for this memory note as a JSON array of strings.",
            ),
            LLMRepresentation(
                field="category",
                prompt=(
                    "Assign one short category label for this memory note, such as "
                    "personal_preference, insight, task, relationship, or plan."
                ),
            ),
            LLMRepresentation(
                field="attributes",
                value_type=dict[str, str],
                prompt=(
                    "Extract a small JSON object of salient attributes for this memory note. "
                    "Use short string keys and string values only."
                ),
            ),
            ConfigurableEmbeddingRepresentation(
                embedding_text=text_prompt(
                    "{{ unit.text }} | "
                    "context: {{ unit.metadata.representation.context }} | "
                    "keywords: {{ unit.metadata.representation.keywords | join(', ') }} | "
                    "tags: {{ unit.metadata.representation.tags | join(', ') }}"
                )
            ),
        ),
        write_trigger=AlwaysTrigger(),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=LLMFunctionCallEvolution(
            source_layer="knowledge_graph",
            target_layer="knowledge_graph",
            tools=build_amem_evolution_tools(
                target_layer="knowledge_graph",
                note_namespace=note_namespace,
            ),
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "You are the A-MEM evolution controller for one newly written note.\n"
                                "Only tool calls may change memory.\n"
                                "The current note is the selected record.\n"
                                "Use AMEM_STRENGTHEN_LINKS zero or one time for the current note.\n"
                                "Use AMEM_UPDATE_NEIGHBOR zero or more times for visible neighbors that need reinterpretation.\n"
                                "For A-MEM repo consistency:\n"
                                "- current note may update links and tags\n"
                                "- neighbor notes may update only context and tags\n"
                                "- never change content or keywords on neighbors\n"
                                "- if no update is needed, make no tool call"
                            ),
                        },
                        {
                            "id": "current_note",
                            "title": "Current Note",
                            "template": (
                                "record_id={{ selected_records.0.record_id }}\n"
                                "text={{ selected_records.0.text }}\n"
                                "context={{ selected_records.0.metadata.representation.context }}\n"
                                "tags={{ selected_records.0.metadata.representation.tags }}"
                            ),
                        },
                        {
                            "id": "retrieved_candidates",
                            "title": "Retrieved Candidate Notes",
                            "template": "{{ recalled_prompt }}",
                        },
                        {
                            "id": "visible_records",
                            "title": "Visible Records",
                            "condition": "visible_records | length",
                            "repeat_over": "visible_records",
                            "item_template": (
                                "- record_id={{ item.record_id }} | text={{ item.text }} | "
                                "context={{ item.metadata.representation.context }} | "
                                "tags={{ item.metadata.representation.tags }} | links={{ item.metadata.graph.links }}"
                            ),
                            "separator": "\n",
                        },
                        {
                            "id": "available_tools",
                            "title": "Available Tools",
                            "repeat_over": "tools",
                            "item_template": "- {{ item.name }}",
                            "separator": "\n",
                        },
                    ]
                },
                recall_plan=text_prompt("{{ recalled_prompt }}"),
                sub_recall_pipeline=MemoryPipeline(
                    retrieval=EmbeddingSimilarityRetrieval(top_k=candidate_k + 1, layer="knowledge_graph"),
                    readout=NoteRenderReadout(note_namespace=note_namespace),
                    store=store,
                ),
                recall_query_builder=(
                    lambda packet, current_store, context: Query(
                        text=str(context["selected_records"][0]["text"]),
                        embedding=list(context["selected_records"][0].get("embedding", [])),
                    )
                    if context.get("selected_records")
                    else Query(text="")
                ),
                visible_record_recall_labels=(PRIMARY_RECALL_LABEL,),
            ),
        ),
        store=store,
    )

    recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(
            top_k=recall_top_k,
            layer="knowledge_graph",
        ),
        readout=NoteRenderReadout(note_namespace=note_namespace),
        store=store,
    )

    return {
        "store": store,
        "write_pipeline": write_pipeline,
        "recall_pipeline": recall_pipeline,
        "note_namespace": note_namespace,
    }


def ingest_note(
    system: dict[str, object],
    *,
    text: str,
    source: str = "dialogue",
    timestamp: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline, MemoryPipeline)
    observation_kwargs: dict[str, object] = {}
    if timestamp is not None:
        observation_kwargs["timestamp"] = timestamp
    write_pipeline.ingest(
        Observation(
            text=text,
            source=source,
            metadata={} if metadata is None else dict(metadata),
            **observation_kwargs,
        )
    )


def recall_amem_memory(system: dict[str, object], *, user_query: str) -> MemoryRecall:
    return _recall_query(
        system,
        Query(text=user_query),
    )


def recall_notes(system: dict[str, object], *, user_query: str) -> str:
    return recall_amem_memory(system, user_query=user_query).text


class AMEMMemoryBinding:
    """Benchmark binding for the classic A-MEM reconstruction."""

    name = "amem"

    def __init__(
        self,
        *,
        note_namespace: str = "amem",
        candidate_k: int = 5,
        recall_top_k: int = 5,
    ) -> None:
        self.note_namespace = note_namespace
        self.candidate_k = candidate_k
        self.recall_top_k = recall_top_k

    def build_system(self) -> dict[str, object]:
        return build_amem_memory_system(
            note_namespace=self.note_namespace,
            candidate_k=self.candidate_k,
            recall_top_k=self.recall_top_k,
        )

    def ingest_event(self, system: dict[str, object], event: MemoryIngestEvent) -> Any:
        parts = [part.strip() for part in (event.text, event.context_text) if part.strip()]
        attachment_suffix = ""
        raw_blip_caption = event.metadata.get("blip_caption")
        if raw_blip_caption:
            attachment_suffix = f" [ATTACHED: {str(raw_blip_caption).strip()}]"
        text = "\n".join(parts) if parts else ""
        return ingest_note(
            system,
            text=f"{text}{attachment_suffix}".strip(),
            source=event.speaker or event.role,
            timestamp=event.timestamp,
            metadata={
                "turn_id": event.turn_id,
                "session_id": event.session_id,
                "user_id": event.user_id,
                "source_timestamp": event.timestamp or "",
                "source_speaker": event.speaker or event.role,
                **dict(event.metadata),
            },
        )

    def recall(self, system: dict[str, object], query: Query, *, context: RecallContext) -> MemoryRecall:
        del context
        if query.embedding is None:
            return recall_amem_memory(system, user_query=query.text)
        return _recall_query(system, query)


def create_memory_binding(
    *,
    note_namespace: str = "amem",
    candidate_k: int = 5,
    recall_top_k: int = 5,
) -> AMEMMemoryBinding:
    return AMEMMemoryBinding(
        note_namespace=note_namespace,
        candidate_k=candidate_k,
        recall_top_k=recall_top_k,
    )


def _recall_query(system: dict[str, object], query: Query) -> MemoryRecall:
    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)
    normalized_query = _query_with_embedding(query)
    readout = recall_pipeline.recall(normalized_query)
    return MemoryRecall(
        text=readout.text,
        source_ids=list(readout.source_ids),
        metadata=dict(readout.metadata),
    )


def _query_with_embedding(query: Query) -> Query:
    if query.embedding is not None:
        return Query(text=query.text, metadata=dict(query.metadata), embedding=list(query.embedding))
    return Query(
        text=query.text,
        metadata=dict(query.metadata),
        embedding=list(get_runtime().embed(query.text)),
    )


def main() -> None:
    system = build_amem_memory_system()
    store = system["store"]
    assert isinstance(store, MemoryStore)

    ingest_note(
        system,
        text="Alice likes jasmine tea in the evening and treats it as part of her steady routine.",
    )
    ingest_note(
        system,
        text="Tea routines improve Alice's focus during reflective graph-memory work.",
    )
    ingest_note(
        system,
        text="Alice links her tea habit with graph memory design because it helps her think clearly.",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("knowledge graph records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "note": record.metadata.get("amem", {}),
                "graph": record.metadata.get("graph", {}),
            }
            for record in store.iter_records("knowledge_graph")
        ]
    )
    print()

    print("agentic recall:")
    print(recall_notes(system, user_query="What should the assistant remember about Alice and tea?"))


if __name__ == "__main__":
    main()
