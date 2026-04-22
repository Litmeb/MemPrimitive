"""Mechanism-level reconstruction of Mem0g-style graph memory.

Within MemPrimitive, classics examples are not judged by whether they duplicate
an upstream repository's storage stack or API surface. They are judged by
whether they recover the method's main memory mechanics in a form that can be
expressed, compared, and recomposed with shared primitives. For Mem0g, the
backbone we want to preserve is:

1. the same interaction pair updates both flat/profile memory and graph memory,
2. graph-oriented extraction turns dialogue into entities and triples,
3. graph writes try to merge with existing entity nodes instead of only
   appending disconnected records,
4. recall includes a graph-aware retrieval path alongside normal profile memory,
   and
5. graph state can influence later maintenance and downstream responses.

That is the claim this file is meant to support. It is a reconstruction of the
Mem0g motif inside the MemPrimitive design space, not a line-by-line clone of
the official Neo4j-based implementation.

Several mismatches still remain and are worth stating plainly. The graph is
stored here as graph-shaped records rather than first-class Neo4j nodes/edges,
graph seeding uses ``VectorGraphSeedAndExpandRetrieval`` (embedding seeds on
graph-vector records) rather than Neo4j's entity-index lookup, and stale relations are maintained through
direct link update/delete instead of the upstream ``valid=false`` soft
invalidation scheme. Those differences matter for exact fidelity, especially
for benchmark-grade reproduction, so this file should still be treated as a
partial alignment.

At the same time, these mismatches are less damaging for the repository's
actual research goal than they would be in a production clone. In ordinary
online use, stale-relation handling is mainly about keeping the currently
active graph consistent for future retrieval and response-time conditioning.
Under that criterion, direct link update/delete is close to the upstream
``valid=false`` scheme: if later evidence reinstates the same relation, the
system writes that relation back into the active graph either way. In the
original Mem0g repo, later maintenance and recall do in fact depend on the
current valid relation set rather than on persistent edge identity.

The main thing this approximation gives up is first-class edge history
encoding, not the active-graph update/retrieval loop itself. That loss matters
for explicit retrospective analysis over obsolete relations, but this prototype
already preserves source-turn provenance and execution traces. In other words,
the same historical edge state can still be reconstructed from the trace even
without storing it as ``valid=false`` inside the graph backend.

Freely adjustable details such as exact prompts, parameter values, model
choices, and local naming are intentionally out of scope for alignment
judgments here. We also intentionally ignore scope-mechanism mismatch
(``user_id`` / ``agent_id`` / ``run_id`` versus local session metadata),
because scope isolation is not the behavior under study in this reconstruction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import (
    MemoryPipeline,
    MemoryStore,
    Observation,
    Query,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    BM25Retrieval,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    FanoutIngestOrganization,
    GraphEntityDeduplicationAppendOrganization,
    GraphRelationReadout,
    JSONReadout,
    LLMFunctionCallEvolution,
    LLMRepresentation,
    NeverTrigger,
    QueryRewriteRetrieval,
    RecencyRetrieval,
    SummaryRewriteEvolution,
    TemplateReadout,
    TripleRepresentation,
    VectorGraphSeedAndExpandRetrieval,
)
from memprimitive.utils._mem0_family import (
    build_fixed_profile_tools,
    build_graph_pair_context,
    build_profile_pair_context,
    finalize_dialogue_turn,
    MEM0_FACT_EXTRACTION_PROMPT,
    PromptRecallSelectionTrigger,
    snapshot_dialogue_turn,
    TimestampedConcatenateReadout,
)
from memprimitive.utils._template import structured_prompt, text_prompt

_graph_pair_context = build_graph_pair_context
_profile_pair_context = build_profile_pair_context


def build_mem0g_memory_system(
    *,
    dedup_threshold: float = 0.85,
    recent_top_k: int = 6,
    similar_top_k: int = 5,
    graph_seed_top_k: int = 3,
    graph_expand_top_k: int = 8,
    rerank_top_k: int = 5,
    recall_top_k: int = 5,
) -> dict[str, object]:
    # Inner expand budget matches VectorGraphSeedAndExpandRetrieval: expand_top_k = candidate_k + neighbor_expansion_k.
    neighbor_expansion_k = max(1, graph_expand_top_k - graph_seed_top_k)
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="recent_dialogue", theme="working", indices=("temporal",)),
            StoreLayerSpec(
                name="conversation_summary",
                theme="semantic",
                indices=("temporal", "vector"),
                settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
            ),
            StoreLayerSpec(
                name="profile",
                theme="semantic",
                indices=("vector", "temporal"),
                settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
            ),
            StoreLayerSpec(
                name="graph_source_observation",
                theme="episodic",
                indices=("temporal", "vector"),
                settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
            ),
            StoreLayerSpec(
                name="knowledge_graph",
                theme="semantic",
                shape="Graph",
                indices=("graph", "vector", "entity"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    recent_history_recall = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=recent_top_k, layer="recent_dialogue"),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )
    conversation_summary_recall = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="conversation_summary"),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )
    graph_context_recall = MemoryPipeline(
        retrieval=(
            QueryRewriteRetrieval(
                retriever=VectorGraphSeedAndExpandRetrieval(
                    top_k=graph_expand_top_k,
                    layer="knowledge_graph",
                    candidate_k=graph_seed_top_k,
                    neighbor_expansion_k=neighbor_expansion_k,
                ),
                strategy="llm",
                allow_multi_query=True,
                include_original=False,
                max_queries=graph_seed_top_k,
                prompt=text_prompt(
                    "You will receive graph-memory evidence from the latest interaction.\n"
                    "Extract the most important graph entities as a JSON list of short canonical entity strings.\n"
                    "Return one entity per list item.\n"
                    "Do not return relation phrases, sentences, or explanations.\n"
                    "Omit empty or redundant entities."
                ),
            ),
            BM25Retrieval(
                top_k=rerank_top_k,
                source="retrieved",
            ),
        ),
        readout=JSONReadout(),
        store=store,
    )

    recent_dialogue_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer="recent_dialogue"),
        store=store,
    )
    conversation_summary_update_pipeline = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text",)),
            LLMRepresentation(
                field="summary",
                prompt=text_prompt(
                    "Update the running conversation summary for graph-memory extraction.\n"
                    "Write one concise but information-rich summary of the conversation so far.\n"
                    "Preserve durable user facts, current relationships, plans, preferences, identity details, and important ongoing context.\n"
                    "Prefer resolved, current information when older and newer details conflict.\n\n"
                    "Previous conversation summary:\n{{ previous_summary }}\n\n"
                    "Recent messages including the newest exchange:\n{{ recent_messages }}\n\n"
                    "Current interaction pair:\n{{ pair_text }}",
                    context_builder=build_graph_pair_context,
                    labeled_recall_plans={
                        "previous_summary": text_prompt("{{ conversation_summary }}"),
                        "recent_messages": text_prompt("{{ recent_messages }}"),
                    },
                    labeled_recall_query_builders={
                        "previous_summary": (
                            lambda packet, store, context: str(context.get("pair_text", "")),
                        ),
                        "recent_messages": (
                            lambda packet, store, context: str(context.get("pair_text", "")),
                        ),
                    },
                ),
            ),
        ),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=AppendOrganization(target_layer="conversation_summary"),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=SummaryRewriteEvolution(target_layer="conversation_summary"),
        store=store,
    )

    profile_candidate_recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=similar_top_k, layer="profile"),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )

    profile_fact_write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=AppendOrganization(target_layer="profile"),
        evolution_trigger=PromptRecallSelectionTrigger(layer_names=("profile",)),
        memory_evolution=LLMFunctionCallEvolution(
            source_layer="profile",
            target_layer="profile",
            tools=build_fixed_profile_tools(embed_on_add=False, embed_on_update=False),
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "You are updating the vector-memory branch of a Mem0g-style long-term memory system.\n"
                                "Use only the provided tools.\n"
                                "The vector-memory branch writes only to the fixed profile layer; never invent or reference any other layer name.\n"
                                "You are processing one extracted fact at a time.\n"
                                "Decide whether to ADD a new memory, UPDATE an existing one, DELETE a contradicted one, or do nothing.\n"
                                "Prefer the top-k similar memories shown below when choosing targets."
                            ),
                        },
                        {
                            "id": "current_turn",
                            "title": "Current Fact And Pair",
                            "template": (
                                "unit_id={{ unit.unit_id }}\n"
                                "fact={{ unit.text }}\n"
                                "pair_text={{ pair_text }}\n"
                                "conversation_summary={{ conversation_summary }}\n"
                                "recent_messages={{ recent_messages }}\n"
                                "user_message={{ user_message }}\n"
                                "assistant_message={{ assistant_message }}"
                            ),
                        },
                        {
                            "id": "similar_memories",
                            "title": "Top-K Similar Existing Memories",
                            "template": "{{ topk_similar }}",
                        },
                        {
                            "id": "visible_records",
                            "title": "Visible Profile Records",
                            "condition": "visible_records | length",
                            "repeat_over": "visible_records",
                            "item_template": "- record_id={{ item.record_id }} | text={{ item.text }}",
                            "separator": "\n",
                        },
                    ]
                },
                context_builder=build_profile_pair_context,
                labeled_recall_plans={
                    "topk_similar": text_prompt("{{ topk_similar }}"),
                    "conversation_summary": text_prompt("{{ conversation_summary }}"),
                    "recent_messages": text_prompt("{{ recent_messages }}"),
                },
                labeled_sub_recall_pipelines={
                    "topk_similar": profile_candidate_recall_pipeline,
                },
                labeled_recall_query_builders={
                    "topk_similar": (
                        lambda packet, store, context: str(context.get("unit", {}).get("text", "")).strip()
                        or str(context.get("user_message", ""))
                        or str(context.get("assistant_message", ""))
                        or str(context.get("pair_text", ""))
                    ),
                    "conversation_summary": (
                        lambda packet, store, context: str(context.get("pair_text", "")),
                    ),
                    "recent_messages": (
                        lambda packet, store, context: str(context.get("pair_text", "")),
                    ),
                },
                visible_record_recall_labels=("topk_similar",),
            ),
        ),
        store=store,
    )

    profile_write_pipeline = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text",)),
            LLMRepresentation(
                field="fact_list",
                value_type=list[str],
                prompt=text_prompt(
                    MEM0_FACT_EXTRACTION_PROMPT
                    + "\nConversation summary:\n{{ conversation_summary }}\n\n"
                    + "Recent messages:\n{{ recent_messages }}\n\n"
                    + "User message:\n{{ user_message }}\n\n"
                    + "Assistant reply:\n{{ assistant_message }}\n\n"
                    + "Current interaction pair:\n{{ pair_text }}\n",
                    context_builder=build_profile_pair_context,
                    labeled_recall_plans={
                        "conversation_summary": text_prompt("{{ conversation_summary }}"),
                        "recent_messages": text_prompt("{{ recent_messages }}"),
                    },
                    labeled_recall_query_builders={
                        "conversation_summary": (
                            lambda packet, store, context: str(context.get("pair_text", "")),
                        ),
                        "recent_messages": (
                            lambda packet, store, context: str(context.get("user_message", ""))
                            or str(context.get("assistant_message", ""))
                            or str(context.get("pair_text", ""))
                        ),
                    },
                ),
            ),
        ),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=FanoutIngestOrganization(field="fact_list", pipeline=profile_fact_write_pipeline),
        store=store,
    )

    mem0g_write_pipeline = MemoryPipeline(
        representation=(
            TripleRepresentation(
                method="two_stage",
                embed_extracted=True,
                embed_entities=True,
                prompt=text_prompt(
                    "Extract grounded graph entities and relation triples for Mem0g-style long-term memory.\n"
                    "Use the historical context only to resolve references and maintain continuity.\n"
                    "Prioritize durable facts stated, updated, or clarified in the current interaction pair.\n"
                    "Prefer user-grounded facts over assistant-only wording, and skip transient chit-chat.\n\n"
                    "Conversation summary:\n{{ conversation_summary }}\n\n"
                    "Recent messages:\n{{ recent_messages }}\n\n"
                    "User message:\n{{ user_message }}\n\n"
                    "Assistant reply:\n{{ assistant_message }}\n\n"
                    "Current interaction pair:\n{{ pair_text }}\n",
                    context_builder=build_graph_pair_context,
                    labeled_recall_plans={
                        "conversation_summary": text_prompt("{{ conversation_summary }}"),
                        "recent_messages": text_prompt("{{ recent_messages }}"),
                    },
                    labeled_recall_query_builders={
                        "conversation_summary": (
                            lambda packet, store, context: str(context.get("pair_text", "")),
                        ),
                        "recent_messages": (
                            lambda packet, store, context: str(context.get("user_message", ""))
                            or str(context.get("assistant_message", ""))
                            or str(context.get("pair_text", ""))
                        ),
                    },
                ),
            ),
            LLMRepresentation(
                field="summary",
                prompt=text_prompt(
                    "Summarize the following interaction pair in one short sentence for graph maintenance.\n\n"
                    "Conversation summary:\n{{ conversation_summary }}\n\n"
                    "Recent messages:\n{{ recent_messages }}\n\n"
                    "User message:\n{{ user_message }}\n\n"
                    "Assistant reply:\n{{ assistant_message }}\n\n"
                    "Current interaction pair:\n{{ pair_text }}",
                    context_builder=build_graph_pair_context,
                    labeled_recall_plans={
                        "conversation_summary": text_prompt("{{ conversation_summary }}"),
                        "recent_messages": text_prompt("{{ recent_messages }}"),
                    },
                    labeled_recall_query_builders={
                        "conversation_summary": (
                            lambda packet, store, context: str(context.get("pair_text", "")),
                        ),
                        "recent_messages": (
                            lambda packet, store, context: str(context.get("user_message", ""))
                            or str(context.get("assistant_message", ""))
                            or str(context.get("pair_text", ""))
                        ),
                    },
                ),
            ),
        ),
        write_trigger=AlwaysTrigger(),
        organization=GraphEntityDeduplicationAppendOrganization(
            target_layer="knowledge_graph",
            threshold=dedup_threshold,
            separate=True,
            separate_layer="graph_source_observation",
        ),
        evolution_trigger=PromptRecallSelectionTrigger(layer_names=("knowledge_graph",)),
        # Upstream graph memory prefers relation-level soft invalidation
        # (`valid=false`) for stale edges. Our current baseline surface does not
        # expose that exact temporal edge-state model here, so we approximate it
        # with direct link deletion/update tools instead. In the original Mem0g
        # repo, later maintenance and recall only depend on the current valid
        # relation set, so this has little effect on the online behavior here.
        memory_evolution=LLMFunctionCallEvolution(
            source_layer="knowledge_graph",
            target_layer="knowledge_graph",
            tools=["GRAPH_ADD_LINK", "GRAPH_UPDATE_LINK", "GRAPH_DELETE_LINK"],
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "You are maintaining a Mem0g-style graph memory after the latest graph write.\n"
                                "Use only the provided graph tools.\n"
                                "If the new observation makes an existing graph relation obsolete or inaccurate, "
                                "add missing outgoing links with GRAPH_ADD_LINK, patch existing outgoing links with GRAPH_UPDATE_LINK, "
                                "or remove stale links with GRAPH_DELETE_LINK.\n"
                                "When a tool needs links, pass them as a JSON array of record_id strings from the visible graph records.\n"
                                "Do not pass link objects, triples, or free-text relation descriptions in the links field.\n"
                                "If no correction is needed, make no tool call."
                            ),
                        },
                        {
                            "id": "incoming_graph_unit",
                            "title": "Incoming Graph Unit",
                            "template": (
                                "unit_id={{ unit.unit_id }}\n"
                                "pair_text={{ pair_text }}\n"
                                "conversation_summary={{ conversation_summary }}\n"
                                "recent_messages={{ recent_messages }}\n"
                                "user_message={{ user_message }}\n"
                                "assistant_message={{ assistant_message }}\n"
                                "summary={{ summary }}\n"
                                "entities={{ entities }}\n"
                                "triples={{ triples }}"
                            ),
                        },
                        {
                            "id": "graph_context",
                            "title": "Graph Recall Context",
                            "template": "{{ graph_context }}",
                        },
                        {
                            "id": "visible_records",
                            "title": "Visible Graph Records",
                            "condition": "visible_records | length",
                            "repeat_over": "visible_records",
                            "item_template": "- record_id={{ item.record_id }} | text={{ item.text }} | graph={{ item.graph }}",
                            "separator": "\n",
                        },
                    ]
                },
                context_builder=build_graph_pair_context,
                labeled_recall_plans={
                    "graph_context": text_prompt("unused"),
                    "conversation_summary": text_prompt("{{ conversation_summary }}"),
                    "recent_messages": text_prompt("{{ recent_messages }}"),
                },
                labeled_sub_recall_pipelines={
                    "graph_context": graph_context_recall,
                },
                labeled_recall_query_builders={
                    "graph_context": (
                        lambda packet, store, context: (
                            json.dumps(
                                [str(item).strip() for item in context.get("entities", []) if str(item).strip()],
                                ensure_ascii=False,
                            )
                            if any(str(item).strip() for item in context.get("entities", []))
                            else str(context.get("conversation_summary", ""))
                            or str(context.get("user_message", ""))
                            or str(context.get("pair_text", ""))
                        )
                    ),
                    "conversation_summary": (
                        lambda packet, store, context: str(context.get("pair_text", "")),
                    ),
                    "recent_messages": (
                        lambda packet, store, context: str(context.get("pair_text", "")),
                    ),
                },
                visible_record_recall_labels=("graph_context",),
            ),
        ),
        store=store,
    )

    graph_recall_pipeline = MemoryPipeline(
        retrieval=(
            QueryRewriteRetrieval(
                retriever=VectorGraphSeedAndExpandRetrieval(
                    top_k=graph_expand_top_k,
                    layer="knowledge_graph",
                    candidate_k=graph_seed_top_k,
                    neighbor_expansion_k=neighbor_expansion_k,
                ),
                strategy="llm",
                allow_multi_query=True,
                include_original=False,
                max_queries=graph_seed_top_k,
                prompt=text_prompt(
                    "Extract the most important graph entities from the query.\n"
                    "Return a JSON list of short canonical entity strings only.\n"
                    "Each item should name one important entity, actor, object, place, or event for graph lookup.\n"
                    "Do not return full sentences, relation phrases, or explanations."
                ),
            ),
            BM25Retrieval(
                top_k=rerank_top_k,
                source="retrieved",
            ),
        ),
        readout=GraphRelationReadout(),
        store=store,
    )

    profile_recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=recall_top_k, layer="profile"),
        readout=TimestampedConcatenateReadout(separator="\n"),
        store=store,
    )
    dual_recall_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="recent_dialogue"),
        readout=TemplateReadout(
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "profile_memories",
                            "title": "Memories",
                            "condition": "profile_memories | length",
                            "template": "{{ profile_memories }}",
                        },
                        {
                            "id": "graph_relations",
                            "title": "Relations",
                            "condition": "graph_relations | length",
                            "template": "{{ graph_relations }}",
                        },
                    ]
                },
                labeled_recall_plans={
                    "profile_memories": text_prompt("{{ profile_memories }}", metadata_mode="readout"),
                    "graph_relations": text_prompt("{{ graph_relations }}", metadata_mode="readout"),
                },
                labeled_sub_recall_pipelines={
                    "profile_memories": profile_recall_pipeline,
                    "graph_relations": graph_recall_pipeline,
                },
                labeled_recall_query_builders={
                    "profile_memories": (lambda packet, store, context: packet.query),
                    "graph_relations": (lambda packet, store, context: packet.query),
                },
                metadata_mode="readout",
            )
        ),
        store=store,
    )

    return {
        "store": store,
        "recent_dialogue_pipeline": recent_dialogue_pipeline,
        "recent_history_recall": recent_history_recall,
        "conversation_summary_recall": conversation_summary_recall,
        "conversation_summary_update_pipeline": conversation_summary_update_pipeline,
        "profile_write_pipeline": profile_write_pipeline,
        "profile_fact_write_pipeline": profile_fact_write_pipeline,
        "mem0g_write_pipeline": mem0g_write_pipeline,
        "graph_recall_pipeline": graph_recall_pipeline,
        "profile_recall_pipeline": profile_recall_pipeline,
        "dual_recall_pipeline": dual_recall_pipeline,
    }


def ingest_message_pair(
    system: dict[str, object],
    *,
    user_text: str,
    assistant_text: str,
    session_id: str,
    turn_id: str,
    timestamp: str | None = None,
) -> None:
    profile_write_pipeline = system["profile_write_pipeline"]
    mem0g_write_pipeline = system["mem0g_write_pipeline"]
    turn = snapshot_dialogue_turn(
        recent_history_recall=system["recent_history_recall"],
        conversation_summary_recall=system["conversation_summary_recall"],
        user_text=user_text,
        assistant_text=assistant_text,
        session_id=session_id,
        turn_id=turn_id,
        timestamp=timestamp,
    )

    profile_write_pipeline.ingest(
        Observation(
            text=turn.pair_text,
            source="dialogue_pair",
            timestamp=turn.timestamp,
            metadata=turn.pair_metadata(),
        )
    )
    mem0g_write_pipeline.ingest(
        Observation(
            text=turn.pair_text,
            source="dialogue_pair",
            timestamp=turn.timestamp,
            metadata=turn.pair_metadata(
                pair_text=turn.pair_text,
            ),
        )
    )
    finalize_dialogue_turn(
        recent_dialogue_pipeline=system["recent_dialogue_pipeline"],
        conversation_summary_update_pipeline=system["conversation_summary_update_pipeline"],
        turn=turn,
    )


def recall_graph(system: dict[str, object], *, user_query: str) -> str:
    return system["graph_recall_pipeline"].recall(Query(text=user_query)).text


def recall_all(system: dict[str, object], *, user_query: str) -> str:
    """Dual-approach recall mirroring the original repo's ``Memory.search()``."""
    return system["dual_recall_pipeline"].recall(Query(text=user_query)).text


def main() -> None:
    system = build_mem0g_memory_system()
    store = system["store"]

    ingest_message_pair(
        system,
        user_text="Alice prefers jasmine tea in the evening.",
        assistant_text="Understood. I'll remember Alice's evening jasmine tea preference.",
        session_id="sess-mem0g",
        turn_id="sess-mem0g-turn-1",
    )
    ingest_message_pair(
        system,
        user_text="Alice works on graph memory systems and links user preferences to project context.",
        assistant_text="Got it. I'll connect Alice's project work with her broader preference context.",
        session_id="sess-mem0g",
        turn_id="sess-mem0g-turn-2",
    )
    ingest_message_pair(
        system,
        user_text="Alice no longer drinks coffee at night and now associates evening work sessions with jasmine tea.",
        assistant_text="Thanks, I'll update the graph memory to reflect the night-time coffee change and the evening jasmine association.",
        session_id="sess-mem0g",
        turn_id="sess-mem0g-turn-3",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("graph records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "graph": record.metadata.get("graph", {}),
            }
            for record in store.iter_records("knowledge_graph")
        ]
    )
    print()

    print("graph source observations:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "timestamp": record.timestamp,
            }
            for record in store.iter_records("graph_source_observation")
        ]
    )
    print()

    print("profile memories:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "timestamp": record.timestamp,
            }
            for record in store.iter_records("profile")
        ]
    )
    print()

    print("dual recall result (profile + graph):")
    print(recall_all(system, user_query="What does the graph memory say about Alice and tea?"))


if __name__ == "__main__":
    main()
