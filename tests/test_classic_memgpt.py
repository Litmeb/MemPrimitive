from __future__ import annotations

import pytest

from memprimitive import Query
from memprimitive.classic_modules.memgpt import (
    MEMGPT_ARCHIVAL_LAYER,
    MEMGPT_MAIN_LAYER,
    MEMGPT_RECALL_LAYER,
    build_memgpt_pipeline,
    memgpt_observation,
)


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def test_memgpt_pipeline_builds_three_layer_store() -> None:
    pipeline = build_memgpt_pipeline()

    assert pipeline.store.topology.layer_names == (
        MEMGPT_MAIN_LAYER,
        MEMGPT_ARCHIVAL_LAYER,
        MEMGPT_RECALL_LAYER,
    )
    assert pipeline.store.count(MEMGPT_MAIN_LAYER) == 0
    assert pipeline.store.count(MEMGPT_ARCHIVAL_LAYER) == 0
    assert pipeline.store.count(MEMGPT_RECALL_LAYER) == 0


def test_memgpt_default_writes_stay_in_main_context() -> None:
    pipeline = build_memgpt_pipeline(main_context_budget=10, recall_budget=2, top_k=3, readout_item_budget=3)

    packet = pipeline.ingest(memgpt_observation("Pinned memory note for the working buffer.", source="dialogue"))

    assert packet.placements[0].target_layer == MEMGPT_MAIN_LAYER
    assert packet.decisions == [True]
    assert pipeline.store.count(MEMGPT_MAIN_LAYER) == 1
    assert pipeline.store.count(MEMGPT_ARCHIVAL_LAYER) == 0
    assert pipeline.store.count(MEMGPT_RECALL_LAYER) == 0


def test_memgpt_explicit_tool_write_routes_to_archival() -> None:
    pipeline = build_memgpt_pipeline(main_context_budget=10, recall_budget=2, top_k=3, readout_item_budget=3)

    packet = pipeline.ingest(
        memgpt_observation(
            "Archive this memory note for later retrieval.",
            source="dialogue",
            tool="memory_save",
            target_layer=MEMGPT_ARCHIVAL_LAYER,
        )
    )

    assert packet.placements[0].target_layer == MEMGPT_ARCHIVAL_LAYER
    assert packet.decisions == [True]
    assert pipeline.store.count(MEMGPT_MAIN_LAYER) == 0
    assert pipeline.store.count(MEMGPT_ARCHIVAL_LAYER) == 1
    assert pipeline.store.count(MEMGPT_RECALL_LAYER) == 0


def test_memgpt_budget_compaction_populates_archival_and_recall_layers() -> None:
    pipeline = build_memgpt_pipeline(top_k=4, main_context_budget=1, recall_budget=1, readout_item_budget=2)

    pipeline.ingest(memgpt_observation("Alpha memory note for the live buffer.", source="dialogue"))
    pipeline.ingest(memgpt_observation("Archive this memory note explicitly.", source="dialogue", tool="memory_save"))
    packet = pipeline.ingest(memgpt_observation("Beta memory note that should trigger compaction.", source="dialogue"))

    assert packet.evolution_decisions == [True]
    assert packet.trace["memory_evolution"]["effects"]
    assert any(effect["effect_type"] == "archive_compaction" for effect in packet.trace["memory_evolution"]["effects"])
    assert pipeline.store.count(MEMGPT_MAIN_LAYER) == 2
    assert pipeline.store.count(MEMGPT_ARCHIVAL_LAYER) >= 2
    assert pipeline.store.count(MEMGPT_RECALL_LAYER) >= 1


def test_memgpt_recall_is_budgeted_and_groups_by_layer() -> None:
    pipeline = build_memgpt_pipeline(top_k=4, main_context_budget=1, recall_budget=1, readout_item_budget=2)

    pipeline.ingest(memgpt_observation("Alpha memory note for the live buffer.", source="dialogue"))
    pipeline.ingest(memgpt_observation("Archive this memory note explicitly.", source="dialogue", tool="memory_save"))
    pipeline.ingest(memgpt_observation("Beta memory note that should trigger compaction.", source="dialogue"))
    pipeline.ingest(memgpt_observation("Gamma memory note that stays in main_context.", source="dialogue"))

    readout = pipeline.recall(Query(text="memory"))

    assert readout.source_ids
    assert readout.metadata["item_count"] == 2
    assert readout.metadata["omitted_item_count"] >= 1
    assert "[main_context]" in readout.text or "[archival]" in readout.text or "[recall]" in readout.text
