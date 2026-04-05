from __future__ import annotations

import pytest

from memprimitive import MemoryRecord, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.example.classics import mem0_memory


class _FakeRuntime:
    def __init__(self) -> None:
        self._embeddings = {
            "alice likes jasmine tea": [1.0, 0.0],
            "alice works on graph memory": [0.0, 1.0],
        }

    def embed(self, text: str) -> list[float]:
        return list(self._embeddings[text])


def test_mem0_per_fact_profile_recall_deduplicates_across_fact_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mem0_memory, "get_runtime", lambda: _FakeRuntime())

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile", theme="semantic", indices=("vector", "temporal")),
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Alice likes jasmine tea.",
            timestamp="2026-04-05T00:00:01Z",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="profile",
            text="Alice works on graph memory.",
            timestamp="2026-04-05T00:00:02Z",
            embedding=[0.0, 1.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="profile",
            text="Alice connects tea habits with project context.",
            timestamp="2026-04-05T00:00:03Z",
            embedding=[0.8, 0.6],
        )
    )

    rendered = mem0_memory._per_fact_profile_recall(
        store,
        ["alice likes jasmine tea", "alice works on graph memory"],
        top_k=2,
        layer="profile",
    )

    assert rendered.count("record_id=rec-1") == 1
    assert rendered.count("record_id=rec-2") == 1
    assert rendered.count("record_id=rec-3") == 1
