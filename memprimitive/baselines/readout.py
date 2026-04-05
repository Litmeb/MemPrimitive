"""Baseline: readout primitive."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Callable, Final

from ..core import MemoryStore, ModuleSpec, Packet, Readout
from ..interfaces import ReadoutModule

from ..utils._amem_family import DEFAULT_CATEGORY, DEFAULT_NOTE_NAMESPACE, note_payload_from_record
from ..utils._graph_family import graph_metadata_from_record
from ..utils._reflexion_family import (
    DEFAULT_MEMORY_SIZE,
    DEFAULT_REFLECTION_LAYER,
    VALID_PROMPT_CONTEXT_STRATEGIES,
    build_prompt_context,
    last_attempt_from_query_metadata,
    strategy_from_query_metadata,
)
from ..utils._template import (
    PromptPlan,
    ensure_prompt_plan,
    render_prompt_plan,
)
from ..utils._mid_decoding_tools import (
    ReadoutToolCallContext,
    ReadoutToolSpec,
    ToolExecutionState,
    build_runtime_tools,
    normalize_readout_tool_specs,
    project_tool_specs_for_prompt,
)
from ..utils._runtime import Runtime
from ..utils._trace import copy_trace


class ConcatenateReadout(ReadoutModule):
    """Turn retrieval items into a single string plus source record ids.

    Constructor: ``separator`` is inserted between consecutive record texts
    (default newline).

    ``run`` requires ``packet.retrieved`` (may be empty). Sets ``readout.text``
    to the joined texts and ``readout.source_ids`` to record ids in retrieval
    order. The store is unchanged.
    """

    spec = ModuleSpec(
        name="concatenate_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, separator: str = "\n") -> None:
        self.separator = separator

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("ConcatenateReadout requires packet.retrieved.")

        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        text = self.separator.join(record.text for record in items)
        readout = Readout(
            text=text,
            source_ids=source_ids,
            metadata={"item_count": len(items)},
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
        }
        return replace(packet, readout=readout, trace=trace), store


class BulletListReadout(ReadoutModule):
    """Render retrieval items as one bullet per line."""

    spec = ModuleSpec(
        name="bullet_list_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("BulletListReadout requires packet.retrieved.")
        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        text = "\n".join(f"- {record.text}" for record in items)
        readout = Readout(text=text, source_ids=source_ids, metadata={"item_count": len(items), "format": "bullet"})
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids}
        return replace(packet, readout=readout, trace=trace), store


class GroupedByLayerReadout(ReadoutModule):
    """Render retrieval items grouped by their source layer."""

    spec = ModuleSpec(
        name="grouped_by_layer_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("GroupedByLayerReadout requires packet.retrieved.")
        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        groups: dict[str, list[str]] = {}
        for record in items:
            groups.setdefault(record.layer, []).append(record.text)
        chunks = [f"[{layer}]\n" + "\n".join(texts) for layer, texts in groups.items()]
        readout = Readout(
            text="\n\n".join(chunks),
            source_ids=source_ids,
            metadata={
                "item_count": len(items),
                "group_counts": {layer: len(texts) for layer, texts in groups.items()},
                "format": "grouped_by_layer",
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids}
        return replace(packet, readout=readout, trace=trace), store


class JSONReadout(ReadoutModule):
    """Render retrieval items into a JSON string for downstream tools/agents."""

    spec = ModuleSpec(
        name="json_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("JSONReadout requires packet.retrieved.")
        items = packet.retrieved.items
        source_ids = [record.record_id for record in items]
        payload = {
            "items": [
                {
                    "record_id": record.record_id,
                    "layer": record.layer,
                    "text": record.text,
                    "timestamp": record.timestamp,
                }
                for record in items
            ],
            "source_ids": source_ids,
        }
        readout = Readout(
            text=json.dumps(payload, ensure_ascii=False),
            source_ids=source_ids,
            metadata={"item_count": len(items), "format": "json"},
        )
        trace = copy_trace(packet)
        trace["readout"] = {"module": self.spec.name, "source_ids": source_ids}
        return replace(packet, readout=readout, trace=trace), store


class GraphReadout(ReadoutModule):
    """Render retrieved records with graph metadata in a stable readable format.

    Constructor: ``include_links`` controls whether linked record ids are
    rendered. The module can consume mixed retrieval results, but it is designed
    for graph-layer payloads and summarizes normalized graph metadata per item.

    ``run`` requires ``packet.retrieved`` and does not mutate the store. It
    renders one graph-oriented line per record and preserves retrieval order.
    """

    spec = ModuleSpec(
        name="graph_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(self, *, include_links: bool = True) -> None:
        self.include_links = include_links

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("GraphReadout requires packet.retrieved.")

        lines: list[str] = []
        source_ids: list[str] = []
        graph_item_count = 0
        for record in packet.retrieved.items:
            source_ids.append(record.record_id)
            graph = graph_metadata_from_record(record)
            parts = [f"[{record.layer}] {record.text}"]
            if graph["entities"]:
                graph_item_count += 1
                parts.append(f"entities={', '.join(graph['entities'])}")
            if self.include_links and graph["links"]:
                parts.append(f"links={', '.join(graph['links'])}")
            elif self.include_links:
                parts.append("links=<none>")
            lines.append(" | ".join(parts))

        readout = Readout(
            text="\n".join(lines),
            source_ids=source_ids,
            metadata={
                "item_count": len(source_ids),
                "graph_item_count": graph_item_count,
                "format": "graph",
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "graph_item_count": graph_item_count,
        }
        return replace(packet, readout=readout, trace=trace), store


class GraphRelationReadout(ReadoutModule):
    """Render linked graph records as relation sentences derived from triples.

    The module only emits relation sentences for retrieved graph records whose
    normalized graph metadata has non-empty ``links``. Relation text comes only
    from the same record's ``triples`` payload and is globally de-duplicated in
    retrieval order. If no relation sentence can be rendered, the module falls
    back to joining the original retrieved record texts.
    """

    spec = ModuleSpec(
        name="graph_relation_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.retrieved is None:
            raise ValueError("GraphRelationReadout requires packet.retrieved.")

        source_ids = [record.record_id for record in packet.retrieved.items]
        relation_sentences: list[str] = []
        seen_sentences: set[str] = set()
        linked_item_count = 0

        for record in packet.retrieved.items:
            graph = graph_metadata_from_record(record)
            if not graph["links"]:
                continue
            linked_item_count += 1
            for subject, relation, obj in graph["triples"]:
                sentence = f"{subject} {relation} {obj}".strip()
                if not sentence or sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                relation_sentences.append(sentence)

        fallback_used = not relation_sentences
        text = "\n".join(relation_sentences) if relation_sentences else "\n".join(
            record.text for record in packet.retrieved.items
        )
        readout = Readout(
            text=text,
            source_ids=source_ids,
            metadata={
                "item_count": len(source_ids),
                "relation_sentence_count": len(relation_sentences),
                "linked_item_count": linked_item_count,
                "format": "graph_relation",
                "fallback_used": fallback_used,
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "relation_sentence_count": len(relation_sentences),
            "linked_item_count": linked_item_count,
            "fallback_used": fallback_used,
        }
        return replace(packet, readout=readout, trace=trace), store


class PromptContextReadout(ReadoutModule):
    """Render retrieved records into next-step prompt context with switchable strategy.

    Constructor: ``default_strategy`` must be one of the supported prompt
    context modes. ``memory_layer`` filters which retrieved records count as
    prompt-memory items. ``top_k`` limits how many retrieved memory records are
    rendered into the final prompt context.

    ``run`` requires ``packet.query``. ``packet.retrieved`` is optional; when
    absent, the module still supports strategies that only use query metadata
    such as last-attempt context. The store is unchanged.
    """

    spec = ModuleSpec(
        name="prompt_context_readout",
        slot="readout",
        input_requirements=("query.text",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        memory_layer: str = DEFAULT_REFLECTION_LAYER,
        default_strategy: str = "reflexion",
        top_k: int = DEFAULT_MEMORY_SIZE,
    ) -> None:
        if top_k <= 0:
            raise ValueError("PromptContextReadout requires top_k > 0.")
        if default_strategy not in VALID_PROMPT_CONTEXT_STRATEGIES:
            raise ValueError(
                f"PromptContextReadout requires strategy in {sorted(VALID_PROMPT_CONTEXT_STRATEGIES)}."
            )
        self.memory_layer = memory_layer
        self.default_strategy = default_strategy
        self.top_k = top_k

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("PromptContextReadout requires packet.query.")

        strategy = strategy_from_query_metadata(packet.query.metadata, self.default_strategy)
        last_attempt = last_attempt_from_query_metadata(packet.query.metadata)
        memory_items = []
        if packet.retrieved is not None:
            memory_items = [
                record for record in packet.retrieved.items if record.layer == self.memory_layer
            ][: self.top_k]

        source_ids = []
        if strategy in {"reflexion", "last_trial_and_reflexion"}:
            source_ids = [record.record_id for record in memory_items]

        readout = Readout(
            text=build_prompt_context(
                strategy=strategy,
                question=packet.query.text,
                last_attempt=last_attempt,
                reflections=[record.text for record in memory_items],
            ),
            source_ids=source_ids,
            metadata={
                "strategy": strategy,
                "reflection_count": len(memory_items),
                "last_attempt_present": bool(last_attempt),
                "memory_layer": self.memory_layer,
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "strategy": strategy,
            "source_ids": source_ids,
        }
        return replace(packet, readout=readout, trace=trace), store


class NoteRenderReadout(ReadoutModule):
    """Render enriched note payloads into a readable note-centric readout.

    Constructor: ``note_namespace`` selects which repaired note payload to read.
    ``include_context`` and ``include_tags`` control optional detail lines.

    ``run`` requires ``packet.query`` and ``packet.retrieved``. The store is
    unchanged. The readout preserves retrieval order and is designed for
    enriched semantic-note payloads rather than generic graph metadata dumps.
    """

    spec = ModuleSpec(
        name="note_render_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        default_category: str = DEFAULT_CATEGORY,
        include_context: bool = True,
        include_tags: bool = True,
    ) -> None:
        self.note_namespace = note_namespace
        self.default_category = default_category
        self.include_context = include_context
        self.include_tags = include_tags

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("NoteRenderReadout requires packet.query.")
        if packet.retrieved is None:
            raise ValueError("NoteRenderReadout requires packet.retrieved.")

        lines = [f"Query: {packet.query.text}", ""]
        source_ids: list[str] = []
        for record in packet.retrieved.items:
            payload = note_payload_from_record(
                record,
                note_namespace=self.note_namespace,
                default_category=self.default_category,
            )
            source_ids.append(record.record_id)
            lines.append(f"- {payload['content']}")
            if self.include_context:
                lines.append(f"  context: {payload['context']}")
            if self.include_tags:
                lines.append(f"  tags: {', '.join(payload['tags'])}")
        if not packet.retrieved.items:
            lines.append("No enriched notes retrieved.")

        readout = Readout(
            text="\n".join(lines).strip(),
            source_ids=source_ids,
            metadata={
                "item_count": len(packet.retrieved.items),
                "format": "note_render",
                "note_namespace": self.note_namespace,
                "retrieval_mode": packet.retrieved.trace.get("retrieval_mode", "unknown"),
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "source_ids": source_ids,
            "note_namespace": self.note_namespace,
            "format": "note_render",
        }
        return replace(packet, readout=readout, trace=trace), store


class MidDecodingMemoryReadout(ReadoutModule):
    """Run a multi-turn tool-calling answer pass during readout."""

    spec = ModuleSpec(
        name="mid_decoding_memory_readout",
        slot="readout",
        input_requirements=("query.text", "retrieved.items"),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        prompt: PromptPlan | str,
        retrieve_pipeline,
        tools: list[str | ReadoutToolSpec] | None = None,
        max_turns: int = 6,
        strict_tools: bool = True,
        allow_no_tool_call: bool = True,
        runtime_now_factory: Callable[[], str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.prompt = ensure_prompt_plan(prompt, metadata_mode="prompt")
        self.retrieve_pipeline = retrieve_pipeline
        self.tool_specs = normalize_readout_tool_specs(
            ["MEM_READ"] if tools is None else tools,
            module_name=self.spec.name,
            retrieve_pipeline=retrieve_pipeline,
        )
        self.max_turns = int(max_turns)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        self.strict_tools = bool(strict_tools)
        self.allow_no_tool_call = bool(allow_no_tool_call)
        self.runtime_now_factory = runtime_now_factory
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embedding_model = embedding_model

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("MidDecodingMemoryReadout requires packet.query.")
        if packet.retrieved is None:
            raise ValueError("MidDecodingMemoryReadout requires packet.retrieved.")

        state = ToolExecutionState()
        tool_context = ReadoutToolCallContext(
            packet=packet,
            store=store,
            retrieve_pipeline=self.retrieve_pipeline,
        )
        runtime_tools = build_runtime_tools(
            self.tool_specs,
            context=tool_context,
            state=state,
            strict_tools=self.strict_tools,
        )
        rendered_prompt, prompt_trace, updated_store = render_prompt_plan(
            ensure_prompt_plan(
                self.prompt,
                metadata_mode="prompt",
                context_builder=lambda current_packet, current_store: {
                    "tools": project_tool_specs_for_prompt(self.tool_specs),
                },
            ),
            packet=packet,
            store=store,
            runtime_now_factory=self.runtime_now_factory,
        )
        tool_context.store = updated_store
        final_text = self._run_agent(
            rendered_prompt=rendered_prompt,
            tools=runtime_tools,
            context={
                "slot": self.spec.slot,
                "query_text": packet.query.text,
            },
        ).strip()
        if not state.tool_calls and not self.allow_no_tool_call:
            raise ValueError("MidDecodingMemoryReadout requires at least one successful or attempted tool call.")

        source_ids = list(state.memory_read_record_ids)
        readout = Readout(
            text=final_text,
            source_ids=source_ids,
            metadata={
                "tool_calls": list(state.tool_calls),
                "memory_read_count": state.memory_read_count,
                "memory_read_record_ids": source_ids,
                "prompt_trace": self._prompt_trace_summary(prompt_trace),
            },
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "tool_names": [spec.name for spec in self.tool_specs],
            "tool_calls": list(state.tool_calls),
            "memory_read_count": state.memory_read_count,
            "source_ids": source_ids,
            "prompt_is_template": self.prompt.mode == "structured"
            or (isinstance(self.prompt.template, str) and "{{" in self.prompt.template and "}}" in self.prompt.template),
        }
        return replace(packet, readout=readout, trace=trace), tool_context.store

    def _run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        runtime = Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )
        runtime.require_llm(capability="MidDecodingMemoryReadout")
        return str(
            runtime.run_agent(
                name="MemPrimitiveMidDecodingMemoryReadoutAgent",
                instructions=(
                    "You answer the user query and may call the provided tools mid-generation "
                    "when memory lookup is needed. Use zero or more tool calls, then continue "
                    "and finish with the final plain-text answer."
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

    @staticmethod
    def _prompt_trace_summary(prompt_trace: dict[str, Any]) -> dict[str, Any]:
        return {
            "template_mode": prompt_trace.get("template_mode"),
            "missing_variables": list(prompt_trace.get("missing_variables", [])),
            "resolved_variable_count": len(prompt_trace.get("resolved_variables", [])),
        }


class TemplateReadout(ReadoutModule):
    """Render retrieved context through a lightweight templating layer."""

    spec = ModuleSpec(
        name="template_readout",
        slot="readout",
        input_requirements=("retrieved.items",),
        output_guarantees=("readout.text", "readout.source_ids"),
    )

    def __init__(
        self,
        *,
        prompt: PromptPlan | str,
        filters: dict[str, Callable[..., Any]] | None = None,
        missing_value: str = "",
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        default_category: str = DEFAULT_CATEGORY,
        runtime_now_factory: Callable[[], str] | None = None,
    ) -> None:
        self.prompt = prompt
        self.filters = {} if filters is None else dict(filters)
        self.missing_value = missing_value
        self.note_namespace = note_namespace
        self.default_category = default_category
        self.runtime_now_factory = runtime_now_factory
        if self.filters:
            raise ValueError("TemplateReadout custom filters are not supported yet.")

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        plan = ensure_prompt_plan(
            self.prompt,
            metadata_mode="readout",
        )
        if plan.missing_value != self.missing_value:
            plan = PromptPlan(
                mode=plan.mode,
                template=plan.template,
                context_builder=plan.context_builder,
                recall_plan=plan.recall_plan,
                recall_query_builder=plan.recall_query_builder,
                missing_value=self.missing_value,
                metadata_mode="readout",
                sub_recall_pipeline=plan.sub_recall_pipeline,
            )
        text, metadata, store = render_prompt_plan(
            plan,
            packet=packet,
            store=store,
            runtime_now_factory=self.runtime_now_factory,
        )
        readout = Readout(
            text=text,
            source_ids=list(metadata["used_record_ids"]),
            metadata=metadata,
        )
        trace = copy_trace(packet)
        trace["readout"] = {
            "module": self.spec.name,
            "template_mode": metadata.get("template_mode"),
            "resolved_variable_count": len(metadata.get("resolved_variables", [])),
            "missing_variable_count": len(metadata["missing_variables"]),
            "used_record_ids": list(metadata["used_record_ids"]),
            "used_group_ids": list(metadata["used_group_ids"]),
        }
        return replace(packet, readout=readout, trace=trace), store


BASELINE_SLOT: Final[str] = "readout"
BASELINE_CLASSES: Final[tuple[type[ReadoutModule], ...]] = (
    ConcatenateReadout,
    BulletListReadout,
    GroupedByLayerReadout,
    JSONReadout,
    GraphReadout,
    GraphRelationReadout,
    PromptContextReadout,
    NoteRenderReadout,
    MidDecodingMemoryReadout,
    TemplateReadout,
)
