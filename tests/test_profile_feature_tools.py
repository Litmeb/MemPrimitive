from __future__ import annotations

from memprimitive import (
    MemoryRecord,
    MemoryStore,
    Packet,
    StoreLayerSpec,
    StoreTopology,
    WriteToolCallContext,
)
from memprimitive.utils._llm_function_tools import normalize_write_tool_specs
from memprimitive.utils._profile_feature_tools import PROFILE_FEATURE_METADATA_KEY, build_profile_feature_tools


def _profile_feature_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
                StoreLayerSpec(name="profile", theme="semantic", indices=("temporal", "vector")),
            ]
        )
    )


def _append_source_episode(store: MemoryStore, *, record_id: str, text: str) -> MemoryRecord:
    record = MemoryRecord(
        record_id=record_id,
        unit_id=f"unit-{record_id}",
        layer="episodic",
        text=text,
        timestamp="2026-04-28T00:00:00Z",
        metadata={"session_id": "session-1"},
    )
    store.append(record)
    return record


def _context(store: MemoryStore, *, selected_records: list[MemoryRecord] | None = None) -> WriteToolCallContext:
    return WriteToolCallContext(
        packet=Packet(),
        store=store,
        module_slot="memory_evolution",
        default_target_layer="profile",
        selected_records=list(selected_records or []),
        visible_records=[],
    )


def test_profile_feature_upsert_adds_structured_record_and_citations() -> None:
    store = _profile_feature_store()
    source = _append_source_episode(store, record_id="episode-1", text="Alice said she prefers jasmine tea.")
    upsert_tool, _delete_tool = build_profile_feature_tools()
    context = _context(store, selected_records=[source])

    result = upsert_tool.executor(
        context,
        {
            "category": "preference",
            "tag": "beverage",
            "feature": "favorite tea",
            "value": "Alice prefers jasmine tea.",
            "set_id": "user-alice",
            "citation_record_ids": ["manual-citation-1"],
        },
    )

    records = result.store.iter_records("profile")
    assert len(records) == 1
    record = records[0]
    payload = record.metadata[PROFILE_FEATURE_METADATA_KEY]
    assert record.text == "Alice prefers jasmine tea."
    assert payload["category"] == "preference"
    assert payload["tag"] == "beverage"
    assert payload["feature"] == "favorite tea"
    assert payload["value"] == "Alice prefers jasmine tea."
    assert payload["set_id"] == "user-alice"
    assert payload["source_episode_record_ids"] == ["episode-1"]
    assert payload["citation_record_ids"] == ["episode-1", "manual-citation-1"]
    assert result.effects[0]["action"] == "add"
    assert result.effects[0]["effect_type"] == "profile_feature_upsert"


def test_profile_feature_upsert_updates_same_key_and_merges_citations() -> None:
    store = _profile_feature_store()
    source_1 = _append_source_episode(store, record_id="episode-1", text="Alice likes tea.")
    source_2 = _append_source_episode(store, record_id="episode-2", text="Alice now prefers oolong.")
    upsert_tool, _delete_tool = build_profile_feature_tools()

    context = _context(store, selected_records=[source_1])
    upsert_tool.executor(
        context,
        {
            "category": "preference",
            "tag": "beverage",
            "feature": "favorite tea",
            "value": "Alice prefers jasmine tea.",
            "set_id": "user-alice",
        },
    )

    context.selected_records = [source_2]
    result = upsert_tool.executor(
        context,
        {
            "category": "Preference",
            "tag": "Beverage",
            "feature": "Favorite Tea",
            "value": "Alice prefers oolong tea.",
            "set_id": "user-alice",
            "source_episode_record_ids": ["episode-extra"],
        },
    )

    records = result.store.iter_records("profile")
    assert len(records) == 1
    record = records[0]
    payload = record.metadata[PROFILE_FEATURE_METADATA_KEY]
    assert record.text == "Alice prefers oolong tea."
    assert payload["value"] == "Alice prefers oolong tea."
    assert payload["source_episode_record_ids"] == ["episode-1", "episode-2", "episode-extra"]
    assert payload["citation_record_ids"] == ["episode-1", "episode-2", "episode-extra"]
    assert result.effects[0]["action"] == "update"


def test_profile_feature_tools_are_available_as_builtin_write_tools_and_delete_by_key() -> None:
    upsert_tool, delete_tool = normalize_write_tool_specs(
        ["UPSERT_PROFILE_FEATURE", "DELETE_PROFILE_FEATURE"],
        module_name="test_module",
    )
    store = _profile_feature_store()
    context = _context(store)

    upsert_tool.executor(
        context,
        {
            "category": "profile",
            "tag": "role",
            "feature": "job",
            "value": "Alice is a product manager.",
            "set_id": "user-alice",
        },
    )
    assert len(store.iter_records("profile")) == 1

    result = delete_tool.executor(
        context,
        {
            "category": "profile",
            "tag": "role",
            "feature": "job",
            "set_id": "user-alice",
            "reason": "stale",
        },
    )

    assert store.iter_records("profile") == []
    assert result.effects[0]["action"] == "delete"
    assert result.effects[0]["status"] == "applied"
    assert result.effects[0]["tool"] == "DELETE_PROFILE_FEATURE"
