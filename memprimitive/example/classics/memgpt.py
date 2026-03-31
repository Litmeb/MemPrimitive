"""MemGPT (Packer et al., 2023) - paper-aligned agent loop sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.memgpt

Or from this directory (script adds the repo root to ``sys.path``)::

    python memgpt.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from pathlib import Path
from typing import Any

from agents import function_tool

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, Observation, Query
from memprimitive.baselines import AlwaysWriteTrigger, AppendOrganization, BasicRepresentation, PassThroughUnitFormation
from memprimitive.utils._runtime import get_classic_runtime
from memprimitive.classic_modules.memgpt import (
    MEMGPT_ARCHIVAL_LAYER,
    MEMGPT_CORE_BLOCK_HUMAN,
    MEMGPT_CORE_BLOCK_PERSONA,
    MEMGPT_CORE_LAYER,
    MEMGPT_QUEUE_LAYER,
    MEMGPT_RECALL_LAYER,
    MEMGPT_REQUIRED_CORE_BLOCKS,
    MEMGPT_WORKING_LAYER,
    MEMGPT_WORKING_SUMMARY_KEY,
    MemGPTKeyedUpsertOrganization,
    MemGPTPagedRetrieval,
    MemGPTSearchReadout,
    build_memgpt_store,
    get_core_block,
    get_working_summary,
)

DEFAULT_PERSONA = "You are a helpful assistant that manages memory deliberately."
DEFAULT_HUMAN = "The human prefers concise, high-signal answers."
DEFAULT_WORKING_SUMMARY = "No prior conversation summary."
DEFAULT_SYSTEM_PROMPT = (
    "You are MemGPT. Manage long-term memory deliberately through tools. "
    "Use conversation_search for full episodic recall, archival_memory_search "
    "for explicit long-term memory, archival_memory_insert to store durable "
    "facts, and core_memory_append/core_memory_replace to edit the persona/human "
    "blocks."
)
MAX_AGENT_STEPS = 8


def _clean_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def memgpt_observation(
    text: str,
    *,
    source: str = "dialogue",
    event_type: str = "note",
    visible: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    payload = {} if metadata is None else dict(metadata)
    nested = payload.get("memgpt")
    memgpt_meta = dict(nested) if isinstance(nested, dict) else {}
    memgpt_meta.setdefault("event_type", event_type)
    memgpt_meta.setdefault("visible", bool(visible))
    payload["memgpt"] = memgpt_meta
    payload.setdefault(
        "keywords",
        [token for token in _clean_text(text).casefold().replace("\n", " ").split() if token],
    )
    return Observation(text=text, source=source, metadata=payload)


def build_memgpt_pipeline(
    *,
    top_k: int = 4,
    main_context_budget: int = 3,
    recall_budget: int = 2,
    readout_item_budget: int = 4,
) -> MemoryPipeline:
    del main_context_budget, recall_budget, readout_item_budget
    store = build_memgpt_store()
    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "keywords", "tags", "embedding")),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(target_layer=MEMGPT_QUEUE_LAYER),
        retrieval=MemGPTPagedRetrieval(
            target_layer=MEMGPT_RECALL_LAYER,
            page_size=max(1, top_k),
            tool_name="conversation_search",
        ),
        readout=MemGPTSearchReadout(
            tool_name="conversation_search",
            target_layer=MEMGPT_RECALL_LAYER,
        ),
    )


@dataclass(slots=True)
class MemGPTAgent:
    """Reusable MemGPT-style agent loop built from shared-store pipelines."""

    queue_token_budget: int = 80
    retrieval_top_k: int = 3
    page_size: int = 3
    persona: str = DEFAULT_PERSONA
    human: str = DEFAULT_HUMAN
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    runtime: Any | None = None
    store: Any = field(init=False)
    pipelines: dict[str, MemoryPipeline] = field(init=False)
    _warning_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.queue_token_budget <= 0:
            raise ValueError("MemGPTAgent requires queue_token_budget > 0.")
        if self.retrieval_top_k <= 0:
            raise ValueError("MemGPTAgent requires retrieval_top_k > 0.")
        if self.page_size <= 0:
            raise ValueError("MemGPTAgent requires page_size > 0.")

        self.runtime = self.runtime if self.runtime is not None else get_classic_runtime()
        self.store = build_memgpt_store()
        self.pipelines = self._build_pipelines()
        self._warning_active = False

        self._upsert_core_block(MEMGPT_CORE_BLOCK_PERSONA, self.persona or DEFAULT_PERSONA)
        self._upsert_core_block(MEMGPT_CORE_BLOCK_HUMAN, self.human or DEFAULT_HUMAN)
        self._update_working_summary(DEFAULT_WORKING_SUMMARY)

    def _build_pipelines(self) -> dict[str, MemoryPipeline]:
        base_repr = BasicRepresentation(elements=("text", "keywords", "tags", "embedding"))

        def _ingest_pipeline(*, target_layer: str) -> MemoryPipeline:
            return MemoryPipeline(
                store=self.store,
                unit_formation=PassThroughUnitFormation(),
                representation=base_repr,
                write_trigger=AlwaysWriteTrigger(),
                organization=AppendOrganization(target_layer=target_layer),
            )

        return {
            "queue_ingest_pipeline": _ingest_pipeline(target_layer=MEMGPT_QUEUE_LAYER),
            "recall_history_pipeline": _ingest_pipeline(target_layer=MEMGPT_RECALL_LAYER),
            "archival_insert_pipeline": _ingest_pipeline(target_layer=MEMGPT_ARCHIVAL_LAYER),
            "core_memory_pipeline": MemoryPipeline(
                store=self.store,
                unit_formation=PassThroughUnitFormation(),
                representation=base_repr,
                write_trigger=AlwaysWriteTrigger(),
                organization=MemGPTKeyedUpsertOrganization(
                    target_layer=MEMGPT_CORE_LAYER,
                    key_name="memgpt_key",
                ),
            ),
            "working_memory_pipeline": MemoryPipeline(
                store=self.store,
                unit_formation=PassThroughUnitFormation(),
                representation=base_repr,
                write_trigger=AlwaysWriteTrigger(),
                organization=MemGPTKeyedUpsertOrganization(
                    target_layer=MEMGPT_WORKING_LAYER,
                    key_name="memgpt_key",
                ),
            ),
            "conversation_search_pipeline": MemoryPipeline(
                store=self.store,
                retrieval=MemGPTPagedRetrieval(
                    target_layer=MEMGPT_RECALL_LAYER,
                    page_size=self.page_size,
                    tool_name="conversation_search",
                ),
                readout=MemGPTSearchReadout(
                    tool_name="conversation_search",
                    target_layer=MEMGPT_RECALL_LAYER,
                ),
            ),
            "archival_search_pipeline": MemoryPipeline(
                store=self.store,
                retrieval=MemGPTPagedRetrieval(
                    target_layer=MEMGPT_ARCHIVAL_LAYER,
                    page_size=self.page_size,
                    tool_name="archival_memory_search",
                ),
                readout=MemGPTSearchReadout(
                    tool_name="archival_memory_search",
                    target_layer=MEMGPT_ARCHIVAL_LAYER,
                ),
            ),
        }

    def render_context(self) -> str:
        persona = get_core_block(self.store, MEMGPT_CORE_BLOCK_PERSONA) or DEFAULT_PERSONA
        human = get_core_block(self.store, MEMGPT_CORE_BLOCK_HUMAN) or DEFAULT_HUMAN
        summary = get_working_summary(self.store) or DEFAULT_WORKING_SUMMARY

        queue_lines = []
        for record in self.store.iter_records(MEMGPT_QUEUE_LAYER):
            event_meta = record.metadata.get("memgpt", {})
            if not isinstance(event_meta, dict):
                event_meta = {}
            event_type = str(event_meta.get("event_type", "event")).strip() or "event"
            queue_lines.append(f"[{event_type}] {record.text}")
        queue_text = "\n".join(queue_lines) if queue_lines else "(empty)"

        return (
            f"{self.system_prompt}\n\n"
            f"[core_memory.persona]\n{persona}\n\n"
            f"[core_memory.human]\n{human}\n\n"
            f"[working_memory.summary]\n{summary}\n\n"
            f"[conversation_queue]\n{queue_text}\n"
        )

    def run_turn(self, user_text: str) -> str:
        self._record_event(text=user_text, event_type="user", source="user")

        if self._queue_over_budget() and not self._warning_active:
            self._append_memory_warning()
            self._warning_active = True

        final_assistant = ""
        pending_input = _clean_text(user_text)
        for step_index in range(MAX_AGENT_STEPS):
            assistant_message = self._call_model(step_index=step_index, user_input=pending_input)
            if assistant_message:
                final_assistant = assistant_message
                self._record_event(
                    text=assistant_message,
                    event_type="assistant",
                    source="assistant",
                )

            if self._queue_over_budget():
                if self._warning_active:
                    self._flush_conversation_queue()
                    self._warning_active = False
                    pending_input = "Continue after the memory flush using the updated state."
                else:
                    self._append_memory_warning()
                    if self._queue_over_budget():
                        self._flush_conversation_queue()
                        self._warning_active = False
                        pending_input = "Continue after the memory flush using the updated state."
                    else:
                        self._warning_active = True
                        pending_input = "Continue after the memory warning using the updated state."
                continue

            self._warning_active = False
            return final_assistant

        raise RuntimeError("MemGPTAgent exceeded the maximum agent-loop steps.")

    def run_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool_name = _clean_text(name)
        args = dict(arguments)

        if tool_name == "conversation_search":
            query_text = _clean_text(args.get("query", ""))
            page = int(args.get("page", 1) or 1)
            readout = self.pipelines["conversation_search_pipeline"].recall(
                Query(text=query_text, metadata={"page": page, "page_size": self.page_size})
            )
            return json.loads(readout.text)

        if tool_name == "archival_memory_search":
            query_text = _clean_text(args.get("query", ""))
            page = int(args.get("page", 1) or 1)
            readout = self.pipelines["archival_search_pipeline"].recall(
                Query(text=query_text, metadata={"page": page, "page_size": self.page_size})
            )
            return json.loads(readout.text)

        if tool_name == "archival_memory_insert":
            memory_text = _clean_text(args.get("memory") or args.get("text") or "")
            if not memory_text:
                raise ValueError("archival_memory_insert requires a non-empty 'memory' or 'text' argument.")
            packet = self.pipelines["archival_insert_pipeline"].ingest(
                memgpt_observation(
                    memory_text,
                    source="tool",
                    event_type="archival_insert",
                    visible=False,
                    metadata={"memgpt": {"tool_name": tool_name}},
                )
            )
            return {
                "tool_name": tool_name,
                "status": "inserted",
                "record_ids": packet.trace["organization"]["written_record_ids"],
                "target_layer": MEMGPT_ARCHIVAL_LAYER,
            }

        if tool_name == "core_memory_append":
            block = _clean_text(args.get("block"))
            if block not in MEMGPT_REQUIRED_CORE_BLOCKS:
                raise ValueError(f"core_memory_append requires block in {MEMGPT_REQUIRED_CORE_BLOCKS}.")
            value = _clean_text(args.get("value") or args.get("text") or "")
            if not value:
                raise ValueError("core_memory_append requires a non-empty 'value' or 'text' argument.")
            current = get_core_block(self.store, block)
            new_text = value if not current else f"{current}\n{value}"
            self._upsert_core_block(block, new_text)
            return {
                "tool_name": tool_name,
                "status": "updated",
                "block": block,
                "value": new_text,
            }

        if tool_name == "core_memory_replace":
            block = _clean_text(args.get("block"))
            if block not in MEMGPT_REQUIRED_CORE_BLOCKS:
                raise ValueError(f"core_memory_replace requires block in {MEMGPT_REQUIRED_CORE_BLOCKS}.")
            value = _clean_text(args.get("value") or args.get("text") or "")
            if not value:
                raise ValueError("core_memory_replace requires a non-empty 'value' or 'text' argument.")
            self._upsert_core_block(block, value)
            return {
                "tool_name": tool_name,
                "status": "updated",
                "block": block,
                "value": value,
            }

        raise ValueError(f"Unsupported MemGPT tool: {tool_name!r}")

    def _build_tools(self) -> list[Any]:
        @function_tool(name_override="conversation_search")
        def conversation_search(query: str, page: int = 1) -> str:
            """Search the full episodic recall history using semantic similarity."""

            return self._run_tool_for_agent("conversation_search", {"query": query, "page": page})

        @function_tool(name_override="archival_memory_search")
        def archival_memory_search(query: str, page: int = 1) -> str:
            """Search explicit archival memory using semantic similarity."""

            return self._run_tool_for_agent("archival_memory_search", {"query": query, "page": page})

        @function_tool(name_override="archival_memory_insert")
        def archival_memory_insert(memory: str) -> str:
            """Insert a durable fact into archival memory."""

            return self._run_tool_for_agent("archival_memory_insert", {"memory": memory})

        @function_tool(name_override="core_memory_append")
        def core_memory_append(block: str, value: str) -> str:
            """Append text to one core memory block."""

            return self._run_tool_for_agent("core_memory_append", {"block": block, "value": value})

        @function_tool(name_override="core_memory_replace")
        def core_memory_replace(block: str, value: str) -> str:
            """Replace one core memory block completely."""

            return self._run_tool_for_agent("core_memory_replace", {"block": block, "value": value})

        return [
            conversation_search,
            archival_memory_search,
            archival_memory_insert,
            core_memory_append,
            core_memory_replace,
        ]

    def _run_tool_for_agent(self, tool_name: str, arguments: dict[str, Any]) -> str:
        result = self.run_tool(tool_name, arguments)
        self._record_event(
            text=json.dumps(result, ensure_ascii=False),
            event_type="tool_result",
            source="tool",
        )
        return json.dumps(result, ensure_ascii=False)

    def _call_model(self, *, step_index: int, user_input: str) -> str:
        instructions = (
            "You are the control loop for a MemGPT-style agent.\n"
            "Use tools whenever memory operations are needed.\n"
            "If no tool is needed, answer directly.\n"
            "Keep the final answer concise.\n\n"
            f"[step_index]\n{step_index}\n\n"
            f"[memory_state]\n{self.render_context()}\n"
        )
        output = self.runtime.run_agent(
            name="MemGPTController",
            instructions=instructions,
            input_text=user_input,
            tools=self._build_tools(),
            temperature=0.0,
            max_turns=MAX_AGENT_STEPS,
        )
        return _clean_text(output)

    def _record_event(
        self,
        *,
        text: str,
        event_type: str,
        source: str,
        visible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        observation = memgpt_observation(
            text,
            source=source,
            event_type=event_type,
            visible=visible,
            metadata=metadata,
        )
        self.pipelines["recall_history_pipeline"].ingest(observation)
        if visible:
            self.pipelines["queue_ingest_pipeline"].ingest(observation)

    def _append_memory_warning(self) -> None:
        warning_text = (
            "MemoryWarning: conversation queue budget exceeded. "
            "Use memory tools if you need to preserve information before flush."
        )
        self._record_event(
            text=warning_text,
            event_type="memory_warning",
            source="system",
            visible=True,
        )

    def _upsert_core_block(self, block_key: str, text: str) -> None:
        observation = memgpt_observation(
            text,
            source="system",
            event_type="core_memory",
            visible=False,
            metadata={
                "memgpt_key": block_key,
                "memgpt": {
                    "event_type": "core_memory",
                    "block_key": block_key,
                    "visible": False,
                },
            },
        )
        self.pipelines["core_memory_pipeline"].ingest(observation)

    def _update_working_summary(self, text: str) -> None:
        observation = memgpt_observation(
            text,
            source="system",
            event_type="working_summary",
            visible=False,
            metadata={
                "memgpt_key": MEMGPT_WORKING_SUMMARY_KEY,
                "memgpt": {
                    "event_type": "working_summary",
                    "visible": False,
                },
            },
        )
        self.pipelines["working_memory_pipeline"].ingest(observation)

    def _queue_over_budget(self) -> bool:
        return self._queue_token_count() > self.queue_token_budget

    def _queue_token_count(self) -> int:
        return sum(self.runtime.count_tokens(record.text) for record in self.store.iter_records(MEMGPT_QUEUE_LAYER))

    def _flush_conversation_queue(self) -> None:
        removed_records = []
        queue_records = self.store.layers[MEMGPT_QUEUE_LAYER]
        while self._queue_over_budget() and queue_records:
            removed_records.append(queue_records.pop(0))
        if not removed_records:
            return

        prior_summary = get_working_summary(self.store) or DEFAULT_WORKING_SUMMARY
        summary_records = [{"kind": "prior_summary", "text": prior_summary}]
        summary_records.extend(
            {
                "record_id": record.record_id,
                "event_type": _clean_text(record.metadata.get("memgpt", {}).get("event_type", "event")),
                "text": record.text,
            }
            for record in removed_records
        )
        new_summary = _clean_text(
            self.runtime.summarize_records(
                records=summary_records,
                instruction=(
                    "Update the recursive working-memory summary. "
                    "Preserve durable facts, open tasks, and user preferences."
                ),
                max_sentences=4,
            )
        )
        self._update_working_summary(new_summary or prior_summary)


def main() -> None:
    agent = MemGPTAgent(queue_token_budget=40, retrieval_top_k=3, page_size=2)
    final = agent.run_turn("Please remember that I prefer concise status updates and archive the release checklist.")
    print(final)
    print(
        "store counts:",
        {
            layer: agent.store.count(layer)
            for layer in (
                MEMGPT_CORE_LAYER,
                MEMGPT_WORKING_LAYER,
                MEMGPT_QUEUE_LAYER,
                MEMGPT_RECALL_LAYER,
                MEMGPT_ARCHIVAL_LAYER,
            )
        },
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_WORKING_SUMMARY",
    "MemGPTAgent",
    "build_memgpt_pipeline",
    "memgpt_observation",
    "main",
]
