from __future__ import annotations

import pytest

from memprimitive import MemoryPipeline, MemoryRecord, MemoryStore, Query, RetrievedSet, StoreLayerSpec, StoreTopology
from memprimitive.core import MemoryUnit, Observation, Packet, Placement, Readout
from memprimitive.baselines import AppendOrganization, ConcatenateReadout
from memprimitive.example.classics import mem0_memory, mem0g_memory
from memprimitive.utils._llm_function_tools import WriteToolCallContext
from memprimitive.utils._mem0_family import (
    DialogueTurnSnapshot,
    MEM0_FACT_EXTRACTION_PROMPT,
    build_fixed_profile_tools,
    finalize_dialogue_turn,
)
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
        packet=Packet(
            observation=Observation(text="tool write"),
            units=[
                MemoryUnit(
                    unit_id="unit-1",
                    text="tool write",
                    timestamp="2026-04-05T00:00:00Z",
                    metadata={},
                )
            ],
        ),
        store=store,
        module_slot="organization",
        default_target_layer="profile",
        selected_records=[],
        visible_records=[],
    )
    add_tool, update_tool, _delete_tool = build_fixed_profile_tools(embed_on_add=False, embed_on_update=False)

    add_result = add_tool.executor(context, {"text": "Alice profile note"})
    context.store = add_result.store
    context.visible_records = list(context.store.iter_records("profile"))
    added = context.store.iter_records("profile")[0]
    assert added.embedding == [0.5, 0.5]
    assert added.timestamp == context.packet.units[0].timestamp

    update_tool.executor(context, {"record_id": added.record_id, "text": "updated profile note"})
    updated = context.store.iter_records("profile")[0]
    assert updated.embedding == [0.25, 0.75]
    assert fake_runtime.calls == ["Alice profile note", "updated profile note"]


def test_finalize_dialogue_turn_uses_turn_timestamp_for_recent_and_summary_records() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="recent_dialogue", theme="working", indices=("temporal",)),
                StoreLayerSpec(name="conversation_summary", theme="semantic", indices=("temporal",)),
            ]
        )
    )
    recent_dialogue_pipeline = MemoryPipeline(
        organization=AppendOrganization(target_layer="recent_dialogue"),
        store=store,
    )
    conversation_summary_update_pipeline = MemoryPipeline(
        organization=AppendOrganization(target_layer="conversation_summary"),
        store=store,
    )
    turn = DialogueTurnSnapshot(
        session_id="session-1",
        turn_id="turn-1",
        pair_id="session-1:turn-1",
        timestamp="2026-04-17T08:00:00Z",
        user_text="Alice likes tea.",
        assistant_text="Noted.",
        messages=[
            {"role": "user", "content": "Alice likes tea."},
            {"role": "assistant", "content": "Noted."},
        ],
        pair_text="user: Alice likes tea.\nassistant: Noted.",
        recent_messages="prior context",
        conversation_summary="running summary",
    )

    finalize_dialogue_turn(
        recent_dialogue_pipeline=recent_dialogue_pipeline,
        conversation_summary_update_pipeline=conversation_summary_update_pipeline,
        turn=turn,
    )

    assert [record.timestamp for record in store.iter_records("recent_dialogue")] == [
        "2026-04-17T08:00:00Z",
        "2026-04-17T08:00:00Z",
    ]
    assert [record.timestamp for record in store.iter_records("conversation_summary")] == [
        "2026-04-17T08:00:00Z",
    ]


def test_mem0_memory_system_omits_summary_update_pipeline() -> None:
    system = mem0_memory.build_mem0_memory_system()
    store = system["store"]

    assert "conversation_summary_update_pipeline" not in system

    turn = DialogueTurnSnapshot(
        session_id="session-1",
        turn_id="turn-1",
        pair_id="session-1:turn-1",
        timestamp="2026-04-17T08:00:00Z",
        user_text="Alice likes tea.",
        assistant_text="Noted.",
        messages=[
            {"role": "user", "content": "Alice likes tea."},
            {"role": "assistant", "content": "Noted."},
        ],
        pair_text="user: Alice likes tea.\nassistant: Noted.",
        recent_messages="",
        conversation_summary="",
    )
    finalize_dialogue_turn(
        recent_dialogue_pipeline=system["recent_dialogue_pipeline"],
        turn=turn,
    )

    assert store.count("recent_dialogue") == 2
    assert store.count("conversation_summary") == 0


def test_mem0_prompt_recall_visible_scope_restricts_profile_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    system = mem0_memory.build_mem0_memory_system()
    store = system["store"]
    store.append(
        MemoryRecord(
            record_id="rec-a",
            unit_id="unit-a",
            layer="profile",
            text="Alpha profile memory",
            timestamp="2026-04-05T00:00:01Z",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-b",
            unit_id="unit-b",
            layer="profile",
            text="Beta profile memory",
            timestamp="2026-04-05T00:00:02Z",
            embedding=[0.0, 1.0],
        )
    )

    evolution = system["profile_fact_write_pipeline"].memory_evolution

    def _fake_recall_run(self, packet, current_store):
        return (
            Packet(
                query=packet.query,
                retrieved=packet.retrieved,
                readout=Readout(text="recall beta", source_ids=["rec-b"], metadata={"source_ids": ["rec-b"]}),
                trace={"readout": {"source_ids": ["rec-b"]}},
            ),
            current_store,
        )

    monkeypatch.setattr(mem0_memory.ConcatenateReadout, "run", _fake_recall_run)

    unit = MemoryUnit(
        unit_id="mem0-unit",
        text="beta",
        timestamp="2026-04-05T00:01:00Z",
        metadata={
            "messages": [
                {"role": "user", "content": "I switched to beta."},
                {"role": "assistant", "content": "Noted."},
            ],
            "pair_text": "user: I switched to beta.\nassistant: Noted.",
            "recent_messages": "recent context",
            "conversation_summary": "summary",
        },
    )

    def _fake_run_agent(self, *, rendered_prompt, tools, context):
        assert "rec-b" in rendered_prompt
        assert "rec-a" not in rendered_prompt
        return "DONE"

    monkeypatch.setattr(evolution, "_run_agent", _fake_run_agent.__get__(evolution, type(evolution)))
    packet_out, _ = evolution.run(
        Packet(
            units=[unit],
            placements=[Placement(unit_id=unit.unit_id, target_layer="profile")],
            decisions_store={"profile": {"record_ids": []}},
        ),
        store,
    )

    assert packet_out.trace["memory_evolution"]["visible_record_ids"] == ["rec-b"]


def test_mem0g_prompt_recall_visible_scope_restricts_profile_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    system = mem0g_memory.build_mem0g_memory_system()
    store = system["store"]
    store.append(
        MemoryRecord(
            record_id="rec-a",
            unit_id="unit-a",
            layer="profile",
            text="Alpha profile memory",
            timestamp="2026-04-05T00:00:01Z",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-b",
            unit_id="unit-b",
            layer="profile",
            text="Beta profile memory",
            timestamp="2026-04-05T00:00:02Z",
            embedding=[0.0, 1.0],
        )
    )

    evolution = system["profile_fact_write_pipeline"].memory_evolution

    def _fake_recall_run(self, packet, current_store):
        return (
            Packet(
                query=packet.query,
                retrieved=packet.retrieved,
                readout=Readout(text="recall beta", source_ids=["rec-b"], metadata={"source_ids": ["rec-b"]}),
                trace={"readout": {"source_ids": ["rec-b"]}},
            ),
            current_store,
        )

    monkeypatch.setattr(mem0g_memory.ConcatenateReadout, "run", _fake_recall_run)

    unit = MemoryUnit(
        unit_id="mem0g-profile-unit",
        text="beta",
        timestamp="2026-04-05T00:01:00Z",
        metadata={
            "messages": [
                {"role": "user", "content": "Beta still matters."},
                {"role": "assistant", "content": "Noted."},
            ],
            "pair_text": "user: Beta still matters.\nassistant: Noted.",
            "recent_messages": "recent context",
            "conversation_summary": "summary",
        },
    )

    def _fake_run_agent(self, *, rendered_prompt, tools, context):
        assert "rec-b" in rendered_prompt
        assert "rec-a" not in rendered_prompt
        return "DONE"

    monkeypatch.setattr(evolution, "_run_agent", _fake_run_agent.__get__(evolution, type(evolution)))
    packet_out, _ = evolution.run(
        Packet(
            units=[unit],
            placements=[Placement(unit_id=unit.unit_id, target_layer="profile")],
            decisions_store={"profile": {"record_ids": []}},
        ),
        store,
    )

    assert packet_out.trace["memory_evolution"]["visible_record_ids"] == ["rec-b"]


@pytest.mark.parametrize(
    ("builder", "pipeline_key"),
    [
        (mem0_memory.build_mem0_memory_system, "mem0_write_pipeline"),
        (mem0g_memory.build_mem0g_memory_system, "profile_write_pipeline"),
    ],
)
def test_mem0_fact_extraction_prompt_matches_upstream_constraints(builder, pipeline_key: str) -> None:
    system = builder()
    prompt = system[pipeline_key].representation[1].prompt.template

    assert prompt == MEM0_FACT_EXTRACTION_PROMPT + (
        "\nConversation summary:\n{{ conversation_summary }}\n\n"
        "Recent messages:\n{{ recent_messages }}\n\n"
        "User message:\n{{ user_message }}\n\n"
        "Assistant reply:\n{{ assistant_message }}\n\n"
        "Current interaction pair:\n{{ pair_text }}\n"
    )
    for fragment in (
        "self-contained",
        "person's name",
        "emotional states",
        "ongoing journeys/future plans",
        "Specific dates",
        "assistant reply only as conversational context",
        "JSON list of strings",
    ):
        assert fragment in prompt


def test_mem0_profile_recall_includes_timestamps_and_source_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    system = mem0_memory.build_mem0_memory_system()
    store = system["store"]
    store.append(
        MemoryRecord(
            record_id="rec-a",
            unit_id="unit-a",
            layer="profile",
            text="Alice likes jasmine tea.",
            timestamp="2026-04-05T00:00:01Z",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-b",
            unit_id="unit-b",
            layer="profile",
            text="Alice is building a graph memory framework.",
            timestamp="2026-04-05T00:00:02Z",
            embedding=[0.0, 1.0],
        )
    )

    def _fake_recall_run(self, packet, current_store):
        records = list(current_store.iter_records("profile"))
        return (
            Packet(
                query=packet.query,
                retrieved=RetrievedSet(items=records),
                trace={"retrieval": {"module": "mock"}},
            ),
            current_store,
        )

    monkeypatch.setattr(mem0_memory.EmbeddingSimilarityRetrieval, "run", _fake_recall_run)

    readout = system["reply_memory_pipeline"].recall(Query(text="What should I remember about Alice?"))
    assert readout.text == (
        "2026-04-05T00:00:01Z: Alice likes jasmine tea.\n"
        "2026-04-05T00:00:02Z: Alice is building a graph memory framework."
    )
    assert readout.source_ids == ["rec-a", "rec-b"]
    assert readout.metadata["item_count"] == 2
    assert mem0_memory.recall_profile(system, user_query="What should I remember about Alice?") == readout.text


def test_mem0g_profile_recall_includes_timestamps_and_source_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    system = mem0g_memory.build_mem0g_memory_system()
    store = system["store"]
    store.append(
        MemoryRecord(
            record_id="rec-a",
            unit_id="unit-a",
            layer="profile",
            text="Alice likes jasmine tea.",
            timestamp="2026-04-05T00:00:01Z",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-b",
            unit_id="unit-b",
            layer="profile",
            text="Alice is building a graph memory framework.",
            timestamp="2026-04-05T00:00:02Z",
            embedding=[0.0, 1.0],
        )
    )

    def _fake_recall_run(self, packet, current_store):
        records = list(current_store.iter_records("profile"))
        return (
            Packet(
                query=packet.query,
                retrieved=RetrievedSet(items=records),
                trace={"retrieval": {"module": "mock"}},
            ),
            current_store,
        )

    monkeypatch.setattr(mem0g_memory.EmbeddingSimilarityRetrieval, "run", _fake_recall_run)

    readout = system["profile_recall_pipeline"].recall(Query(text="What should I remember about Alice?"))
    assert readout.text == (
        "2026-04-05T00:00:01Z: Alice likes jasmine tea.\n"
        "2026-04-05T00:00:02Z: Alice is building a graph memory framework."
    )
    assert readout.source_ids == ["rec-a", "rec-b"]
    assert readout.metadata["item_count"] == 2


def test_concatenate_readout_still_joins_text_only() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile", theme="semantic", indices=("vector",))])
    )
    packet = Packet(
        retrieved=RetrievedSet(
            items=[
                MemoryRecord(
                    record_id="rec-a",
                    unit_id="unit-a",
                    layer="profile",
                    text="Alice likes jasmine tea.",
                    timestamp="2026-04-05T00:00:01Z",
                ),
                MemoryRecord(
                    record_id="rec-b",
                    unit_id="unit-b",
                    layer="profile",
                    text="Alice is building a graph memory framework.",
                    timestamp="2026-04-05T00:00:02Z",
                ),
            ]
        )
    )

    packet_out, _ = ConcatenateReadout().run(packet, store)
    assert packet_out.readout is not None

    assert packet_out.readout.text == "Alice likes jasmine tea.\nAlice is building a graph memory framework."
    assert packet_out.readout.source_ids == ["rec-a", "rec-b"]
