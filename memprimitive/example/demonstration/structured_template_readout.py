"""Structured-template readout example with summary + source-item grouping.

This example keeps the setup intentionally small and readable:

- manually seeds one ``episodic`` layer and one ``session_summary`` layer
- uses ``LayerAwareRetrieval`` to recall across both layers
- renders the result with ``TemplateReadout(prompt=build_structured_prompt_plan(...))``

The template demonstrates:

- top-level blocks
- conditional rendering
- repeating over a layer view
- repeating over ``retrieved.views.summary_with_sources``
- preserving source ids / group ids / block trace in readout metadata

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.structured_template_readout

Or from this directory (script adds the repo root to ``sys.path``)::

    python structured_template_readout.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryRecord, MemoryStore, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval, TemplateReadout
from memprimitive.utils._template import build_structured_prompt_plan


if __name__ == "__main__":
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="episodic", theme="session_memory", indices=("temporal",)),
                StoreLayerSpec(name="session_summary", theme="semantic", indices=("temporal",)),
            ]
        )
    )

    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="episodic",
            text="Alice debugged a layer-aware retrieval merge bug in the morning session.",
            timestamp="2026-04-01T09:00:00+00:00",
            metadata={
                "session_id": "sess-graph",
                "subgoal_id": "sg-retrieval",
                "representation": {
                    "keywords": ["alice", "retrieval", "merge"],
                    "summary": "Retrieval merge debugging detail.",
                },
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="episodic",
            text="She then added a structured readout template so summaries can cite source records.",
            timestamp="2026-04-01T09:10:00+00:00",
            metadata={
                "session_id": "sess-graph",
                "subgoal_id": "sg-readout",
                "representation": {
                    "keywords": ["template", "readout", "summary"],
                },
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="session_summary",
            text="Session summary: Alice improved retrieval readout and preserved source provenance.",
            timestamp="2026-04-01T09:30:00+00:00",
            metadata={
                "unit_type": "summary",
                "session_id": "sess-graph",
                "representation": {
                    "summary": "Alice improved retrieval readout and preserved source provenance.",
                },
                "hierarchical": {
                    "source_layer": "episodic",
                    "target_layer": "session_summary",
                    "group_by": ["session_id"],
                    "group_key": {"session_id": "sess-graph"},
                    "source_record_ids": ["rec-1", "rec-2"],
                    "source_unit_ids": ["unit-1", "unit-2"],
                    "field_payload": {
                        "summary": "Alice improved retrieval readout and preserved source provenance.",
                    },
                    "relation": "hierarchical_source",
                },
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-4",
            unit_id="unit-4",
            layer="episodic",
            text="Bob planned a Hangzhou trip in another session.",
            timestamp="2026-04-01T10:00:00+00:00",
            metadata={
                "session_id": "sess-trip",
                "subgoal_id": "sg-trip",
            },
        )
    )

    pipeline = MemoryPipeline(
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=3),
            retriever_by_layer={
                "session_summary": RecencyRetrieval(top_k=2),
                "episodic": RecencyRetrieval(top_k=3),
            },
            active_layers=("session_summary", "episodic"),
            top_k=4,
            top_k_by_layer={"session_summary": 2, "episodic": 3},
        ),
        readout=TemplateReadout(
            prompt=build_structured_prompt_plan({
                "blocks": [
                    {
                        "id": "header",
                        "title": "Query",
                        "template": "{{ query.text }}",
                    },
                    {
                        "id": "summary_section",
                        "title": "Session Summaries",
                        "condition": "retrieved.views.summary_with_sources | length",
                        "repeat_over": "retrieved.views.summary_with_sources",
                        "item_template": (
                            "- {{ item.summary.text }}\n"
                            "  session={{ item.group_key.session_id }}\n"
                            "  sources:\n"
                            "{{ item.sources | join_text }}"
                        ),
                        "separator": "\n\n",
                    },
                    {
                        "id": "episodic_section",
                        "title": "Recent Episodic Records",
                        "condition": "retrieved.by_layer.episodic | length",
                        "repeat_over": "retrieved.by_layer.episodic | topk(3)",
                        "item_template": "- {{ item.text }}",
                        "separator": "\n",
                    },
                    {
                        "id": "trace_section",
                        "title": "Retrieval Trace",
                        "template": (
                            "module={{ trace.retrieval.module | default('unknown') }}\n"
                            "active_layers={{ trace.retrieval.active_layers | join(', ') }}\n"
                            "candidate_count={{ trace.retrieval.total_merged_count | default('n/a') }}"
                        ),
                    },
                ]
            })
        ),
        store=store,
    )

    query = Query(text="What happened in Alice's retrieval/readout work?")
    retrieval_packet, _ = pipeline.retrieval.run(Packet(query=query), store)
    readout = pipeline.recall(query)

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("retrieval trace:")
    pprint(retrieval_packet.trace["retrieval"])
    print()

    print("structured template readout:")
    print(readout.text)
    print()

    print("source record ids:")
    pprint(readout.source_ids)
    print()

    print("readout metadata summary:")
    pprint(
        {
            "template_mode": readout.metadata.get("template_mode"),
            "used_record_ids": readout.metadata.get("used_record_ids"),
            "used_group_ids": readout.metadata.get("used_group_ids"),
            "available_views": readout.metadata.get("available_views"),
            "block_trace": readout.metadata.get("block_trace"),
        }
    )
