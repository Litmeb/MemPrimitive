from __future__ import annotations

import pytest

from memprimitive import MemoryRecord, MemoryStore, Query, StoreLayerSpec, StoreTopology
from memprimitive.benchmarking import MemoryIngestEvent, RecallContext
from memprimitive.benchmarking._memory_adapters import create_hmem_memory_adapter
from memprimitive.example.classics import hmem_memory


def test_hmem_builder_uses_top_down_routing_retrieval() -> None:
    system = hmem_memory.build_hmem_memory_system(top_k=3)
    recall_pipeline = system["recall_pipeline"]

    assert recall_pipeline.retrieval.spec.name == "hierarchical_top_down_routing_retrieval"
    assert recall_pipeline.readout.spec.name == "concatenate_readout"


def test_hmem_ingest_links_four_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    system = hmem_memory.build_hmem_memory_system(top_k=1)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    monkeypatch.setattr(
        hmem_memory,
        "_extract_hmem_layers",
        lambda text, runtime=None: {
            "domain": "movies",
            "category": "action films",
            "memory_trace": "jackie chan recommendation",
            "episode": "Assistant recommended a Jackie Chan action movie.",
            "user_profile": "likes action movies",
        },
    )
    monkeypatch.setattr(
        hmem_memory.Runtime,
        "embed",
        lambda self, text: [1.0, 0.0] if "Jackie" in text or "action" in text.casefold() else [0.0, 1.0],
    )

    result = hmem_memory.ingest_hmem_turn(
        system,
        text="Alice asked for an action movie recommendation.",
        session_id="sess-1",
        turn_id="turn-1",
        runtime=hmem_memory.Runtime(),
    )

    domain = store.layers["domain"][0]
    category = store.layers["category"][0]
    trace = store.layers["trace"][0]
    episode = store.layers["episode"][0]

    assert result["record_ids"]["episode"] == episode.record_id
    assert trace.metadata["child_record_ids"] == [episode.record_id]
    assert category.metadata["child_record_ids"] == [trace.record_id]
    assert domain.metadata["child_record_ids"] == [category.record_id]


def test_hmem_recall_routes_to_matching_episode() -> None:
    system = hmem_memory.build_hmem_memory_system(top_k=1)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    records = [
        ("dom-1", "domain", [1.0, 0.0], ["cat-1"]),
        ("cat-1", "category", [1.0, 0.0], ["trace-1"]),
        ("trace-1", "trace", [1.0, 0.0], ["epi-1"]),
        ("epi-1", "episode", [1.0, 0.0], []),
        ("dom-2", "domain", [0.0, 1.0], ["cat-2"]),
        ("cat-2", "category", [0.0, 1.0], ["trace-2"]),
        ("trace-2", "trace", [0.0, 1.0], ["epi-2"]),
        ("epi-2", "episode", [0.0, 1.0], []),
    ]
    for index, (record_id, layer, embedding, children) in enumerate(records):
        store.append(
            MemoryRecord(
                record_id=record_id,
                unit_id=f"unit-{index}",
                layer=layer,
                text=f"{layer}-{record_id}",
                timestamp=f"2026-01-01T00:00:{index:02d}+00:00",
                embedding=embedding,
                metadata={"child_record_ids": children},
            )
        )

    readout = system["recall_pipeline"].recall(Query(text="action movie", embedding=[1.0, 0.0]))

    assert readout.source_ids == ["epi-1"]
    assert "epi-1" in readout.text


def test_hmem_memory_binding_factory() -> None:
    binding = hmem_memory.create_memory_binding(top_k=4)
    assert binding.name == "hmem"
    system = binding.build_system()
    assert system["top_k"] == 4


def test_create_hmem_locomo_adapter_uses_shared_conversation() -> None:
    adapter = create_hmem_memory_adapter(top_k=5)
    assert adapter.name == "hmem"
    session = adapter.create_session()
    assert session.binding.name == "hmem"
