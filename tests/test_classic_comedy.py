from __future__ import annotations

from memprimitive.example.classics.comedy_memory import build_comedy_memory_system, build_reply_context, ingest_session


class _FakeComedyRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str) -> dict[str, str]:
        import json

        payload = json.loads(user)
        group_key = payload.get("group_key") or {}
        group_label = group_key.get("session_id", "all_sessions")
        record_count = len(payload.get("records", []))
        text = f"{payload['target_layer']}::{group_label}::{record_count}"
        self.calls.append(
            {
                "system": system,
                "target_layer": payload["target_layer"],
                "group_key": group_key,
                "record_count": record_count,
            }
        )
        return {field: text for field in payload["extract_fields"]}


def test_comedy_memory_builds_session_and_compressive_layers(monkeypatch) -> None:
    from memprimitive.utils import _runtime

    fake_runtime = _FakeComedyRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    system = build_comedy_memory_system(compress_every_n=2)
    store = system["store"]

    ingest_session(
        system,
        session_index=1,
        session_id="sess-1",
        turns=[
            "first session turn one",
            "first session turn two",
        ],
    )
    ingest_session(
        system,
        session_index=2,
        session_id="sess-2",
        turns=[
            "second session turn one",
            "second session turn two",
        ],
    )
    ingest_session(
        system,
        session_index=3,
        session_id="sess-3",
        turns=[
            "third session turn one",
            "third session turn two",
        ],
    )

    assert store.count("episodic") == 6
    assert store.count("session_memory") == 3
    assert store.count("compressive_memory") == 1

    session_records = store.iter_records("session_memory")
    assert [record.metadata["hierarchical"]["group_key"]["session_id"] for record in session_records] == [
        "sess-1",
        "sess-2",
        "sess-3",
    ]

    compressive_record = store.iter_records("compressive_memory")[0]
    assert compressive_record.text == "compressive_memory::all_sessions::2"
    assert len(compressive_record.metadata["hierarchical"]["source_record_ids"]) == 2

    reply_context = build_reply_context(system, user_query="latest reply context")
    assert reply_context == "compressive_memory::all_sessions::2"

    assert [call["target_layer"] for call in fake_runtime.calls] == [
        "session_memory",
        "session_memory",
        "compressive_memory",
        "session_memory",
    ]
