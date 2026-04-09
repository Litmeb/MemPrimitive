"""Mechanism-level reconstruction of Mem0-style long-term memory.

In this repository, classics examples are meant to re-express published memory
methods in the shared primitive language used by MemPrimitive. The goal is not
to clone an upstream codebase line by line, but to preserve the causal memory
loop that matters for mechanism-level comparison:

1. take the current message pair together with prior summary and recent context,
2. extract durable profile facts,
3. retrieve similar existing memories for each fact,
4. let the model decide add/update/delete actions, and
5. use the maintained profile memory again at recall time.

That is the level of fidelity this file is aiming for. It is intended to be
persuasive as a reconstruction of the Mem0 motif inside the MemPrimitive DSL,
not as a byte-for-byte or API-for-API reproduction of the official runtime.

The main remaining mismatch is that the evolution step still exposes the whole
``profile`` layer as the visible mutable store, rather than strictly limiting
updates to the retrieved similar-memory candidate set used by upstream Mem0.
This matters for exact behavioral fidelity, so we do not claim full alignment.
At the same time, the mismatch is more structural than practically operative
here: an out-of-scope edit would require the model to invent a record id that
does not appear in the prompt and have that hallucinated id coincidentally
match a real profile record. In ordinary use, that failure mode is negligible,
so the effective update scope remains close to the retrieved candidate set.
Under that criterion, the example still preserves the more important design
claim for this project: Mem0 can be decomposed into summary-backed fact
extraction, per-fact similar-memory search, and tool-driven profile
maintenance using reusable baseline primitives instead of paper-specific glue
code.

Freely adjustable details such as exact prompts, parameter values, model
choices, and local naming are intentionally out of scope for alignment
judgments here. We also intentionally ignore scope-mechanism mismatch
(``user_id`` / ``agent_id`` / ``run_id`` versus local session metadata),
because scope isolation is not the behavior under study in this reconstruction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    LLMFunctionCallEvolution,
    LLMRepresentation,
    PlacementWithoutAppendOrganization,
    RecencyRetrieval,
    SummaryRewriteEvolution,
)
from memprimitive.utils._mem0_family import (
    build_fixed_profile_tools,
    build_profile_pair_context,
    finalize_dialogue_turn,
    PromptRecallSelectionTrigger,
    snapshot_dialogue_turn,
)
from memprimitive.utils._template import structured_prompt, text_prompt


def build_mem0_memory_system(
    *,
    recent_top_k: int = 6,
    similar_top_k: int = 5,
    recall_top_k: int = 5,
) -> dict[str, object]:
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
                    "Update the running conversation summary for long-term memory extraction.\n"
                    "Write one concise but information-rich summary of the conversation so far.\n"
                    "Preserve durable user facts, preferences, plans, identity details, and important ongoing context.\n"
                    "Prefer resolved, current information when older and newer details conflict.\n\n"
                    "Previous conversation summary:\n{{ previous_summary }}\n\n"
                    "Recent messages including the newest exchange:\n{{ recent_messages }}\n\n"
                    "Current interaction pair:\n{{ pair_text }}",
                    context_builder=build_profile_pair_context,
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
        write_trigger=AlwaysTrigger(),
        organization=PlacementWithoutAppendOrganization(target_layer="conversation_summary"),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=SummaryRewriteEvolution(target_layer="conversation_summary"),
        store=store,
    )

    profile_candidate_recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=similar_top_k, layer="profile"),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )

    mem0_write_pipeline = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text",)),
            LLMRepresentation(
                field="fact_list",
                value_type=list[str],
                prompt=text_prompt(
                    "Extract Mem0-style long-term memory candidates as a JSON list of strings.\n"
                    "This input is one interaction pair plus historical conversation context.\n"
                    "Only store stable user-related facts, preferences, plans, identity details, or durable working context.\n"
                    "Use the assistant reply only as conversational context for reference resolution.\n"
                    "Do not store facts that originate only from the assistant reply.\n"
                    "Skip transient chit-chat and wording tied too closely to the exact utterance.\n\n"
                    "Conversation summary:\n{{ conversation_summary }}\n\n"
                    "Recent messages:\n{{ recent_messages }}\n\n"
                    "User message:\n{{ user_message }}\n\n"
                    "Assistant reply:\n{{ assistant_message }}\n\n"
                    "Current interaction pair:\n{{ pair_text }}\n",
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
        write_trigger=AlwaysTrigger(),
        organization=PlacementWithoutAppendOrganization(target_layer="profile"),
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
                                "You are updating a Mem0-style long-term memory store.\n"
                                "Use only the provided tools.\n"
                                "The memory store writes only to the fixed profile layer; never invent or reference any other layer name.\n"
                                "For each fact in fact_list, decide whether to ADD a new memory, UPDATE an existing one, "
                                "DELETE a contradicted one, or do nothing.\n"
                                "Prefer the top-k similar memories shown below when choosing targets."
                            ),
                        },
                        {
                            "id": "current_turn",
                            "title": "Current Pair",
                            "template": (
                                "unit_id={{ unit.unit_id }}\n"
                                "pair_text={{ pair_text }}\n"
                                "conversation_summary={{ conversation_summary }}\n"
                                "recent_messages={{ recent_messages }}\n"
                                "user_message={{ user_message }}\n"
                                "assistant_message={{ assistant_message }}\n"
                                "fact_list={{ fact_list }}"
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
                        {
                            "id": "available_tools",
                            "title": "Available Tools",
                            "repeat_over": "tools",
                            "item_template": "- {{ item.name }}",
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
                        lambda packet, store, context: "\n".join(
                            str(item).strip() for item in context.get("fact_list", []) if str(item).strip()
                        )
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

    reply_memory_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=recall_top_k, layer="profile"),
        readout=ConcatenateReadout(),
        store=store,
    )

    return {
        "store": store,
        "recent_dialogue_pipeline": recent_dialogue_pipeline,
        "recent_history_recall": recent_history_recall,
        "conversation_summary_recall": conversation_summary_recall,
        "conversation_summary_update_pipeline": conversation_summary_update_pipeline,
        "mem0_write_pipeline": mem0_write_pipeline,
        "reply_memory_pipeline": reply_memory_pipeline,
    }


def ingest_message_pair(
    system: dict[str, object],
    *,
    user_text: str,
    assistant_text: str,
    session_id: str,
    turn_id: str,
) -> None:
    mem0_write_pipeline = system["mem0_write_pipeline"]
    turn = snapshot_dialogue_turn(
        recent_history_recall=system["recent_history_recall"],
        conversation_summary_recall=system["conversation_summary_recall"],
        user_text=user_text,
        assistant_text=assistant_text,
        session_id=session_id,
        turn_id=turn_id,
    )
    mem0_write_pipeline.ingest(
        Observation(
            text=turn.pair_text,
            source="dialogue_pair",
            metadata=turn.pair_metadata(),
        )
    )
    finalize_dialogue_turn(
        recent_dialogue_pipeline=system["recent_dialogue_pipeline"],
        conversation_summary_update_pipeline=system["conversation_summary_update_pipeline"],
        turn=turn,
    )


def recall_profile(system: dict[str, object], *, user_query: str) -> str:
    return system["reply_memory_pipeline"].recall(Query(text=user_query)).text


def main() -> None:
    system = build_mem0_memory_system()
    store = system["store"]

    ingest_message_pair(
        system,
        user_text="My name is Alice, and I usually prefer jasmine tea over coffee.",
        assistant_text="Nice to meet you, Alice. I'll remember that you usually prefer jasmine tea over coffee.",
        session_id="sess-mem0",
        turn_id="sess-mem0-turn-1",
    )
    ingest_message_pair(
        system,
        user_text="I am building a graph memory framework and want the assistant to remember that preference.",
        assistant_text="Understood. I'll keep both your project focus and your tea preference in mind.",
        session_id="sess-mem0",
        turn_id="sess-mem0-turn-2",
    )
    ingest_message_pair(
        system,
        user_text="Actually, I drink oolong most mornings now, but jasmine tea is still my favorite evening drink.",
        assistant_text="Thanks for the update. I'll distinguish your morning oolong habit from your evening jasmine preference.",
        session_id="sess-mem0",
        turn_id="sess-mem0-turn-3",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
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

    print("recall result:")
    print(recall_profile(system, user_query="What should the assistant remember about Alice?"))


if __name__ == "__main__":
    main()
