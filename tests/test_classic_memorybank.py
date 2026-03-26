from __future__ import annotations

import pytest

from memprimitive import Observation, Query
from memprimitive.classic_modules.memorybank import MemoryBankConfig, build_memorybank_pipeline


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _ingest_texts(pipeline, texts: list[str]) -> None:
    for index, text in enumerate(texts, start=1):
        pipeline.ingest(Observation(text=text, source=f"turn-{index}"))


def test_memorybank_routes_entity_and_non_entity_memories_to_expected_layers() -> None:
    pipeline = build_memorybank_pipeline(config=MemoryBankConfig(short_term_window=4))

    pipeline.ingest(Observation(text="Alice works at OpenAI in San Francisco.", source="dialogue"))
    pipeline.ingest(Observation(text="remember to refill the tea kettle", source="note"))

    assert pipeline.store.count("long_term") == 1
    assert pipeline.store.count("short_term") == 1
    assert "Alice works at OpenAI" in pipeline.store.iter_records("long_term")[0].text
    assert "refill the tea kettle" in pipeline.store.iter_records("short_term")[0].text


def test_memorybank_merges_repeated_entities_into_one_long_term_cluster() -> None:
    pipeline = build_memorybank_pipeline(config=MemoryBankConfig(short_term_window=4))

    first = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    second = pipeline.ingest(Observation(text="Alice reads books.", source="dialogue"))

    assert pipeline.store.count("long_term") == 1
    merged_record = pipeline.store.iter_records("long_term")[0]
    assert "Alice likes tea." in merged_record.text
    assert "Alice reads books." in merged_record.text
    assert any(effect["effect_type"] == "entity_merge" for effect in second.trace["memory_evolution"]["effects"])
    assert first.trace["memory_evolution"]["effects"] == []


def test_memorybank_summarizes_and_prunes_short_term_overflow() -> None:
    pipeline = build_memorybank_pipeline(config=MemoryBankConfig(short_term_window=2))

    _ingest_texts(
        pipeline,
        [
            "remember the first note",
            "remember the second note",
            "remember the third note",
        ],
    )

    assert pipeline.store.count("short_term") == 2
    assert pipeline.store.count("long_term") == 1

    long_term_record = pipeline.store.iter_records("long_term")[0]
    assert long_term_record.text
    assert "remember the first note" in long_term_record.text


def test_memorybank_layered_recall_groups_short_and_long_term_context() -> None:
    pipeline = build_memorybank_pipeline(config=MemoryBankConfig(short_term_window=4))

    pipeline.ingest(Observation(text="Alice works at OpenAI.", source="dialogue"))
    pipeline.ingest(Observation(text="keep this short-term reminder nearby", source="note"))

    readout = pipeline.recall(Query(text="Alice"))

    assert "[long_term]" in readout.text
    assert "[short_term]" in readout.text
    assert any(record.record_id in readout.source_ids for record in pipeline.store.iter_records("long_term"))
    assert any(record.record_id in readout.source_ids for record in pipeline.store.iter_records("short_term"))
