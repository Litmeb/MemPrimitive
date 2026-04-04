"""Demonstration: ``LLMFunctionCallOrganization`` with ``ADD`` + custom layer delete tool.

This example is intentionally tutorial-style. It shows two separate patterns:

1. ingest-time tool calling with built-in ``ADD``
2. evolution-time tool calling with a custom ``WriteToolSpec``

The important prompt-design point is that both prompts use
``structured_prompt(...)`` and explicitly render ``record_id -> text`` mappings
so the LLM can refer to exact stored memories instead of guessing.

This script uses real LLM tool calls, so configure the runtime first, for
example via:

    MEMPRIMITIVE_API_KEY
    MEMPRIMITIVE_BASE_URL
    MEMPRIMITIVE_MODEL

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.llm_function_call_tools

Or from this directory (script adds the repo root to ``sys.path``)::

    python llm_function_call_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import (
    MemoryPipeline,
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    Observation,
    Packet,
    Query,
    StoreLayerSpec,
    StoreTopology,
    WriteToolCallContext,
    WriteToolResult,
    WriteToolSpec,
)
from memprimitive.baselines import (
    BasicRepresentation,
    LLMFunctionCallEvolution,
    LLMFunctionCallOrganization,
    PassThroughUnitFormation,
)
from memprimitive.utils._template import structured_prompt


def _seed_record(store: MemoryStore, *, layer: str, unit_id: str, text: str, timestamp: str) -> None:
    unit = MemoryUnit(
        unit_id=unit_id,
        text=text,
        timestamp=timestamp,
        metadata={"seeded": True},
    )
    store.append(MemoryRecord.from_unit(unit=unit, layer=layer, sequence_id=store.next_sequence_id()))


def _delete_entire_layer_tool() -> WriteToolSpec:
    def _executor(context: WriteToolCallContext, arguments: dict[str, object]) -> WriteToolResult:
        target_layer = str(arguments.get("target_layer", "")).strip()
        if not target_layer:
            raise ValueError("DELETE_LAYER requires target_layer.")
        reason = str(arguments.get("reason", "")).strip()

        existing = list(context.store.iter_records(target_layer))
        effects: list[dict[str, object]] = []
        for record in list(existing):
            removed = context.store.delete_record(target_layer, record.record_id)
            effects.append(
                {
                    "action": "delete",
                    "record_id": removed.record_id,
                    "layer": removed.layer,
                    "status": "applied",
                    "reason": reason,
                    "tool": "DELETE_LAYER",
                }
            )
        return WriteToolResult(effects=effects, store=context.store)

    return WriteToolSpec(
        name="DELETE_LAYER",
        description="Delete every record in one target layer.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "target_layer": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["target_layer"],
            "additionalProperties": False,
        },
        executor=_executor,
    )


def build_store() -> MemoryStore:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="inbox", theme="working", indices=("temporal",)),
                StoreLayerSpec(name="profile", theme="semantic", indices=("temporal",)),
            ]
        )
    )
    _seed_record(
        store,
        layer="profile",
        unit_id="seed-profile-1",
        text="Alice prefers concise architectural explanations with explicit tradeoffs.",
        timestamp="2026-04-04T09:00:00+00:00",
    )
    _seed_record(
        store,
        layer="inbox",
        unit_id="seed-inbox-1",
        text="Temporary scratch note that should be removable as a whole layer.",
        timestamp="2026-04-04T09:05:00+00:00",
    )
    return store


def build_add_pipeline(store: MemoryStore) -> MemoryPipeline:
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        organization=LLMFunctionCallOrganization(
            tools=["ADD"],
            target_layer="profile",
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "You are deciding whether to store the incoming unit as a new long-term memory.\n"
                                "If it should be saved, call ADD exactly once.\n"
                                "When calling ADD, write to {{ default_target_layer }} unless another layer is explicitly justified."
                            ),
                        },
                        {
                            "id": "incoming_unit",
                            "title": "Incoming Unit",
                            "template": (
                                "unit_id={{ unit.unit_id }}\n"
                                "unit_type={{ unit.unit_type }}\n"
                                "text={{ unit.text }}"
                            ),
                        },
                        {
                            "id": "available_tools",
                            "title": "Available Tools",
                            "repeat_over": "tools",
                            "item_template": "- {{ item.name }}: {{ item.description }}",
                            "separator": "\n",
                        },
                        {
                            "id": "existing_records",
                            "title": "Existing Records With Exact IDs",
                            "condition": "visible_records | length",
                            "repeat_over": "visible_records",
                            "item_template": "- record_id={{ item.record_id }} | layer={{ item.layer }} | text={{ item.text }}",
                            "separator": "\n",
                        },
                    ]
                }
            ),
        ),
        store=store,
    )


def build_delete_layer_pipeline(store: MemoryStore) -> MemoryPipeline:
    return MemoryPipeline(
        memory_evolution=LLMFunctionCallEvolution(
            source_layer="inbox",
            tools=[_delete_entire_layer_tool()],
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "Review the selected records below.\n"
                                "If the whole layer is explicitly scratch / temporary and should be cleared, call DELETE_LAYER once.\n"
                                "Otherwise do nothing."
                            ),
                        },
                        {
                            "id": "selected_records",
                            "title": "Selected Records With Exact IDs",
                            "condition": "selected_records | length",
                            "repeat_over": "selected_records",
                            "item_template": "- record_id={{ item.record_id }} | layer={{ item.layer }} | text={{ item.text }}",
                            "separator": "\n",
                        },
                        {
                            "id": "tool_syntax",
                            "title": "Available Tool",
                            "repeat_over": "tools",
                            "item_template": (
                                "- {{ item.name }}\n"
                                "  schema={{ item.parameters_json_schema }}"
                            ),
                            "separator": "\n",
                        },
                    ]
                }
            ),
        ),
        store=store,
    )


def print_store_snapshot(store: MemoryStore, *, title: str) -> None:
    print(title)
    pprint(
        {
            layer_name: [
                {
                    "record_id": record.record_id,
                    "text": record.text,
                }
                for record in store.iter_records(layer_name)
            ]
            for layer_name in store.topology.layer_names
        }
    )
    print()


def main() -> None:
    store = build_store()
    print_store_snapshot(store, title="Initial store:")

    add_pipeline = build_add_pipeline(store)
    add_packet = add_pipeline.ingest(
        Observation(
            text="New stable preference: Alice also wants examples to include concrete record ids when discussing memory design.",
            source="dialogue",
            metadata={"session_id": "sess-function-tools"},
        )
    )

    print("After organization-time ADD:")
    print_store_snapshot(store, title="Store after ADD:")
    print("Organization trace:")
    pprint(add_packet.trace["organization"])
    print()

    delete_pipeline = build_delete_layer_pipeline(store)
    delete_packet = delete_pipeline.memory_evolution.run(Packet(), store)[0]

    print("After evolution-time custom DELETE_LAYER:")
    print_store_snapshot(store, title="Store after DELETE_LAYER:")
    print("Evolution trace:")
    pprint(delete_packet.trace["memory_evolution"])
    print()

    print("Prompt syntax summary:")
    print(
        "Both prompts are structured templates. The important repeated line is:\n"
        "  - record_id={{ item.record_id }} | layer={{ item.layer }} | text={{ item.text }}\n"
        "This makes the LLM see the exact mapping from record id to memory text before it issues a tool call."
    )


if __name__ == "__main__":
    main()
