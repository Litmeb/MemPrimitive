from __future__ import annotations

import pytest

from memprimitive import MemoryRecord, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.core import Observation, Packet
from memprimitive.example.classics import mem0_memory
from memprimitive.utils._llm_function_tools import WriteToolCallContext
from memprimitive.utils._mem0_family import build_fixed_profile_tools
from memprimitive.utils import _runtime as runtime_module


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._embeddings = {
            "alice likes jasmine tea": [1.0, 0.0],
            "alice works on graph memory": [0.0, 1.0],
            "Alice profile note": [0.5, 0.5],
            "updated profile note": [0.25, 0.75],
        }

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
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


def test_build_fixed_profile_tools_delegate_embedding_to_store_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="profile",
                    theme="semantic",
                    indices=("vector", "temporal"),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    context = WriteToolCallContext(
        packet=Packet(observation=Observation(text="tool write"), units=[]),
        store=store,
        module_slot="organization",
        default_target_layer="profile",
        visible_records=[],
    )
    add_tool, update_tool, _delete_tool = build_fixed_profile_tools(embed_on_add=False, embed_on_update=False)

    add_result = add_tool.executor(context, {"text": "Alice profile note"})
    context.store = add_result.store
    context.visible_records = list(context.store.iter_records("profile"))
    added = context.store.iter_records("profile")[0]
    assert added.embedding == [0.5, 0.5]

    update_tool.executor(context, {"record_id": added.record_id, "text": "updated profile note"})
    updated = context.store.iter_records("profile")[0]
    assert updated.embedding == [0.25, 0.75]
    assert fake_runtime.calls == ["Alice profile note", "updated profile note"]
