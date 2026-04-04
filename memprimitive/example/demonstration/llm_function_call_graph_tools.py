"""Demonstration: LLM function-call graph tools with ``GRAPH_ADD`` / ``GRAPH_UPDATE`` / ``GRAPH_DELETE``.

This example mirrors ``llm_function_call_tools.py`` but focuses on the new
graph-aware built-in tools. It shows one compact graph-memory workflow:

1. ingest-time ``GRAPH_ADD`` writes a new graph record with normalized
   ``metadata["graph"]``
2. evolution-time ``GRAPH_UPDATE`` adds an explicit same-layer link
3. evolution-time ``GRAPH_DELETE`` removes an old graph node and automatically
   cleans dangling links from the remaining records in that layer

The prompt-design point is the same as the non-graph demo: use
``structured_prompt(...)`` and explicitly render exact ``record_id`` mappings
plus the current graph metadata so the LLM can issue precise tool calls.

This script uses real LLM tool calls, so configure the runtime first, for
example via:

    MEMPRIMITIVE_API_KEY
    MEMPRIMITIVE_BASE_URL
    MEMPRIMITIVE_MODEL

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.llm_function_call_graph_tools

Or from this directory (script adds the repo root to ``sys.path``)::

    python llm_function_call_graph_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryRecord, MemoryStore, MemoryUnit, Observation, Packet, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    LLMFunctionCallEvolution,
    LLMFunctionCallOrganization,
    PassThroughUnitFormation,
    TripleRepresentation,
)
from memprimitive.utils._graph_family import graph_metadata_for_write, graph_metadata_from_record
from memprimitive.utils._template import structured_prompt


def _seed_graph_record(
    store: MemoryStore,
    *,
    unit_id: str,
    text: str,
    timestamp: str,
    entities: list[str],
    triples: list[tuple[str, str, str]],
) -> None:
    unit = MemoryUnit(
        unit_id=unit_id,
        text=text,
        timestamp=timestamp,
        entities=entities,
        triples=triples,
        metadata={"seeded": True},
    )
    record = MemoryRecord.from_unit(unit=unit, layer="knowledge_graph", sequence_id=store.next_sequence_id())
    record.metadata["graph"] = graph_metadata_for_write(layer="knowledge_graph", unit=unit)
    store.append(record)


def build_store() -> MemoryStore:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    _seed_graph_record(
        store,
        unit_id="seed-graph-1",
        text="Alice is a tea enthusiast.",
        timestamp="2026-04-04T09:00:00+00:00",
        entities=["Alice", "tea"],
        triples=[("Alice", "likes", "tea")],
    )
    return store


def build_graph_add_pipeline(store: MemoryStore) -> MemoryPipeline:
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=TripleRepresentation(),
        organization=LLMFunctionCallOrganization(
            tools=["GRAPH_ADD"],
            target_layer="knowledge_graph",
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "You are deciding whether to write the incoming observation into the graph layer.\n"
                                "If it should become a new graph record, call GRAPH_ADD exactly once.\n"
                                "Use {{ default_target_layer }} as the target layer."
                            ),
                        },
                        {
                            "id": "incoming_unit",
                            "title": "Incoming Unit",
                            "template": (
                                "unit_id={{ unit.unit_id }}\n"
                                "text={{ unit.text }}\n"
                                "entities={{ unit.entities }}\n"
                                "triples={{ unit.triples }}"
                            ),
                        },
                        {
                            "id": "existing_records",
                            "title": "Existing Graph Records",
                            "condition": "visible_records | length",
                            "repeat_over": "visible_records",
                            "item_template": (
                                "- record_id={{ item.record_id }} | layer={{ item.layer }} | text={{ item.text }}\n"
                                "  graph={{ item.graph }}"
                            ),
                            "separator": "\n",
                        },
                        {
                            "id": "available_tools",
                            "title": "Available Tools",
                            "repeat_over": "tools",
                            "item_template": "- {{ item.name }}: {{ item.description }}",
                            "separator": "\n",
                        },
                    ]
                }
            ),
        ),
        store=store,
    )


def build_graph_update_pipeline(store: MemoryStore) -> MemoryPipeline:
    return MemoryPipeline(
        memory_evolution=LLMFunctionCallEvolution(
            source_layer="knowledge_graph",
            tools=["GRAPH_UPDATE"],
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "Review the selected graph records below.\n"
                                "If there is a newly added Alice-related record that should explicitly link to the older seeded memory,\n"
                                "call GRAPH_UPDATE once and patch only that record's graph metadata."
                            ),
                        },
                        {
                            "id": "selected_records",
                            "title": "Selected Graph Records",
                            "condition": "selected_records | length",
                            "repeat_over": "selected_records",
                            "item_template": (
                                "- record_id={{ item.record_id }} | text={{ item.text }}\n"
                                "  graph={{ item.graph }}"
                            ),
                            "separator": "\n",
                        },
                        {
                            "id": "tool_syntax",
                            "title": "Available Tool",
                            "repeat_over": "tools",
                            "item_template": "- {{ item.name }} schema={{ item.parameters_json_schema }}",
                            "separator": "\n",
                        },
                    ]
                }
            ),
        ),
        store=store,
    )


def build_graph_delete_pipeline(store: MemoryStore) -> MemoryPipeline:
    return MemoryPipeline(
        memory_evolution=LLMFunctionCallEvolution(
            source_layer="knowledge_graph",
            tools=["GRAPH_DELETE"],
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "task",
                            "title": "Task",
                            "template": (
                                "Review the selected graph records below.\n"
                                "If the older seeded node is now redundant, call GRAPH_DELETE exactly once.\n"
                                "Deleting it should also clean dangling links from the remaining graph records in the same layer."
                            ),
                        },
                        {
                            "id": "selected_records",
                            "title": "Selected Graph Records",
                            "condition": "selected_records | length",
                            "repeat_over": "selected_records",
                            "item_template": (
                                "- record_id={{ item.record_id }} | text={{ item.text }}\n"
                                "  graph={{ item.graph }}"
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
                    "graph": graph_metadata_from_record(record),
                }
                for record in store.iter_records(layer_name)
            ]
            for layer_name in store.topology.layer_names
        }
    )
    print()


def main() -> None:
    store = build_store()
    print_store_snapshot(store, title="Initial graph store:")

    add_pipeline = build_graph_add_pipeline(store)
    add_packet = add_pipeline.ingest(
        Observation(
            text="Alice recently switched from tea to jasmine tea.",
            source="dialogue",
            metadata={"session_id": "sess-graph-tools"},
        )
    )

    print("After organization-time GRAPH_ADD:")
    print_store_snapshot(store, title="Store after GRAPH_ADD:")
    print("Organization trace:")
    pprint(add_packet.trace["organization"])
    print()

    update_pipeline = build_graph_update_pipeline(store)
    update_packet = update_pipeline.memory_evolution.run(Packet(), store)[0]

    print("After evolution-time GRAPH_UPDATE:")
    print_store_snapshot(store, title="Store after GRAPH_UPDATE:")
    print("Evolution trace:")
    pprint(update_packet.trace["memory_evolution"])
    print()

    delete_pipeline = build_graph_delete_pipeline(store)
    delete_packet = delete_pipeline.memory_evolution.run(Packet(), store)[0]

    print("After evolution-time GRAPH_DELETE:")
    print_store_snapshot(store, title="Store after GRAPH_DELETE:")
    print("Evolution trace:")
    pprint(delete_packet.trace["memory_evolution"])
    print()

    print("Prompt syntax summary:")
    print(
        "This demo keeps the same core prompt pattern as the non-graph tool demo:\n"
        "  - record_id={{ item.record_id }} | text={{ item.text }}\n"
        "  graph={{ item.graph }}\n"
        "The important difference is that the available tools are graph-aware built-ins,\n"
        "so GRAPH_ADD normalizes graph metadata and GRAPH_DELETE cleans same-layer dangling links."
    )


if __name__ == "__main__":
    main()
