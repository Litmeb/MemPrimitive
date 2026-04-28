from __future__ import annotations

from typing import Any

import pytest

from memprimitive.core import MemoryRecord, MemoryStore, Packet, StoreLayerSpec, StoreTopology


class _FakeSTMSummarizationRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def require_llm(self, *, capability: str) -> None:
        self.calls.append({"capability": capability})

    def summarize_records(
        self,
        *,
        records: list[dict[str, Any]],
        instruction: str,
        max_sentences: int,
    ) -> str:
        self.calls.append(
            {
                "records": records,
                "instruction": instruction,
                "max_sentences": max_sentences,
            }
        )
        previous = [record["text"] for record in records if record["role"] == "previous_summary"]
        evicted_ids = [record["record_id"] for record in records if record["role"] == "evicted_episode"]
        prefix = f"{previous[-1]} | " if previous else ""
        return f"{prefix}summary::{','.join(evicted_ids)}"


def _store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working"),
                StoreLayerSpec(name="episodic"),
                StoreLayerSpec(name="session_summary"),
            ]
        )
    )


def _append_record(
    store: MemoryStore,
    *,
    layer: str,
    record_id: str,
    text: str,
    session_id: str,
    user_id: str = "user-1",
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> None:
    store.append(
        MemoryRecord(
            record_id=record_id,
            unit_id=f"unit-{record_id}",
            layer=layer,
            text=text,
            timestamp=timestamp,
            metadata={"session_id": session_id, "user_id": user_id},
        )
    )


def test_stm_consolidation_record_budget_moves_oldest_records_and_keeps_newest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import STMConsolidationEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeSTMSummarizationRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = _store()
    for index in range(1, 4):
        _append_record(
            store,
            layer="working",
            record_id=f"stm-{index}",
            text=f"raw episode {index}",
            session_id="sess-1",
            timestamp=f"2026-01-01T00:0{index}:00+00:00",
        )

    packet_out, store = STMConsolidationEvolution(record_budget=2).run(Packet(), store)

    assert [record.record_id for record in store.iter_records("working")] == ["stm-2", "stm-3"]
    ltm_records = store.iter_records("episodic")
    assert [record.text for record in ltm_records] == ["raw episode 1"]
    assert ltm_records[0].unit_id == "unit-stm-1"
    assert ltm_records[0].timestamp == "2026-01-01T00:01:00+00:00"
    assert ltm_records[0].metadata["session_id"] == "sess-1"
    assert ltm_records[0].metadata["user_id"] == "user-1"
    assert ltm_records[0].metadata["stm_consolidation_source_record_id"] == "stm-1"
    trace = packet_out.trace["memory_evolution"]
    assert trace["evicted_record_ids"] == ["stm-1"]
    assert trace["moved_ltm_record_ids"] == [ltm_records[0].record_id]
    assert trace["deleted_working_record_ids"] == ["stm-1"]
    assert trace["effects"][0]["session_scope"] == {"session_id": "sess-1"}
    assert trace["effects"][0]["retained_record_ids"] == ["stm-2", "stm-3"]


def test_stm_consolidation_summary_upsert_keeps_one_current_summary_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import STMConsolidationEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeSTMSummarizationRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = _store()
    for index in range(1, 4):
        _append_record(
            store,
            layer="working",
            record_id=f"stm-{index}",
            text=f"episode {index}",
            session_id="sess-1",
        )
    _append_record(store, layer="session_summary", record_id="summary-old-1", text="older summary", session_id="sess-1")
    _append_record(store, layer="session_summary", record_id="summary-old-2", text="newer summary", session_id="sess-1")
    _append_record(store, layer="session_summary", record_id="summary-other", text="other summary", session_id="sess-2")

    packet_out, store = STMConsolidationEvolution(record_budget=1, summary_max_sentences=2).run(Packet(), store)

    sess_1_summaries = [
        record for record in store.iter_records("session_summary") if record.metadata["session_id"] == "sess-1"
    ]
    assert len(sess_1_summaries) == 1
    assert sess_1_summaries[0].text == "newer summary | summary::stm-1,stm-2"
    assert {record.record_id for record in store.iter_records("session_summary")}.isdisjoint(
        {"summary-old-1", "summary-old-2"}
    )
    assert any(record.record_id == "summary-other" for record in store.iter_records("session_summary"))
    summary_update = packet_out.trace["memory_evolution"]["summary_updates"][0]
    assert summary_update["summary_old_record_id"] == "summary-old-2"
    assert summary_update["summary_new_record_id"] == sess_1_summaries[0].record_id
    summary_call = fake_runtime.calls[-1]
    assert summary_call["max_sentences"] == 2
    assert summary_call["records"][0]["record_id"] == "summary-old-2"


def test_stm_consolidation_scopes_sessions_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import STMConsolidationEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeSTMSummarizationRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = _store()
    for session_id, record_ids in {"sess-1": ["s1-a", "s1-b"], "sess-2": ["s2-a", "s2-b", "s2-c"]}.items():
        for record_id in record_ids:
            _append_record(
                store,
                layer="working",
                record_id=record_id,
                text=f"{session_id} {record_id}",
                session_id=session_id,
            )

    _, store = STMConsolidationEvolution(record_budget=1).run(Packet(), store)

    summaries_by_session = {
        record.metadata["session_id"]: record.text for record in store.iter_records("session_summary")
    }
    assert summaries_by_session == {
        "sess-1": "summary::s1-a",
        "sess-2": "summary::s2-a,s2-b",
    }
    payload_calls = [call for call in fake_runtime.calls if "records" in call]
    assert [[record["record_id"] for record in call["records"]] for call in payload_calls] == [
        ["s1-a"],
        ["s2-a", "s2-b"],
    ]


def test_stm_consolidation_token_budget_keeps_newest_records_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import STMConsolidationEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeSTMSummarizationRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = _store()
    _append_record(store, layer="working", record_id="stm-1", text="one two three", session_id="sess-1")
    _append_record(store, layer="working", record_id="stm-2", text="four five", session_id="sess-1")
    _append_record(store, layer="working", record_id="stm-3", text="six", session_id="sess-1")

    packet_out, store = STMConsolidationEvolution(record_budget=None, token_budget=3).run(Packet(), store)

    assert [record.record_id for record in store.iter_records("working")] == ["stm-2", "stm-3"]
    assert [record.record_id for record in store.iter_records("episodic")] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["evicted_record_ids"] == ["stm-1"]


def test_stm_consolidation_ltm_raw_episode_text_is_not_replaced_by_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import STMConsolidationEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeSTMSummarizationRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = _store()
    _append_record(store, layer="working", record_id="stm-1", text="raw first episode", session_id="sess-1")
    _append_record(store, layer="working", record_id="stm-2", text="raw second episode", session_id="sess-1")

    _, store = STMConsolidationEvolution(record_budget=1).run(Packet(), store)

    assert store.iter_records("episodic")[0].text == "raw first episode"
    assert store.iter_records("session_summary")[0].text == "summary::stm-1"


def test_stm_consolidation_is_registered_and_exported() -> None:
    import memprimitive.baselines as pkg
    from memprimitive.baselines import STMConsolidationEvolution
    from memprimitive.baselines.registry import registered_baseline_class_names

    assert STMConsolidationEvolution.__name__ == "STMConsolidationEvolution"
    assert "STMConsolidationEvolution" in registered_baseline_class_names()
    assert "STMConsolidationEvolution" in pkg.__all__


def test_stm_consolidation_validates_constructor_budgets() -> None:
    from memprimitive.baselines import STMConsolidationEvolution

    with pytest.raises(ValueError, match="at least one active budget"):
        STMConsolidationEvolution(record_budget=None, token_budget=None)
    with pytest.raises(ValueError, match="record_budget"):
        STMConsolidationEvolution(record_budget=0)
    with pytest.raises(ValueError, match="token_budget"):
        STMConsolidationEvolution(token_budget=0)
    with pytest.raises(ValueError, match="scope_metadata_keys"):
        STMConsolidationEvolution(scope_metadata_keys=())


def test_stm_consolidation_requires_declared_layers() -> None:
    from memprimitive.baselines import STMConsolidationEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="working")]))

    with pytest.raises(ValueError, match="episodic"):
        STMConsolidationEvolution().run(Packet(), store)
