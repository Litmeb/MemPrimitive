from __future__ import annotations

import json

import pytest

from memprimitive import Observation, Packet, Query
from memprimitive.classic_modules import _runtime
from memprimitive.classic_modules.amem import AMEMConfig, AMEM_GRAPH_LAYER
from memprimitive.example.classics.amem_agentic_memory import build_amem_pipeline


def test_runtime_coerce_json_extracts_first_valid_block() -> None:
    payload = (
        "Here is the result.\n"
        '{"decision": "write", "confidence": 0.91}\n'
        "Additional commentary that should be ignored."
    )

    assert _runtime._coerce_json(payload) == {"decision": "write", "confidence": 0.91}


class FakeClassicRuntime:
    def require_llm(self, *, capability: str) -> None:
        return None

    def embed(self, text: str) -> list[float]:
        normalized = text.casefold()
        return [
            10.0 if "alice" in normalized else 0.0,
            8.0 if "tea" in normalized else 0.0,
            6.0 if "focus" in normalized else 0.0,
            4.0 if "graph" in normalized else 0.0,
            float(len(normalized)),
        ]

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        system_lower = system.casefold()
        if "note generator" in system_lower:
            content = payload["content"]
            lowered = content.casefold()
            if "discard me" in lowered:
                return {
                    "note_text": "Comprehensive note: discard me from memory.",
                    "context": "This note is irrelevant and should be skipped.",
                    "keywords": ["discard", "irrelevant", "memory"],
                    "tags": ["irrelevant", "noise", "skip"],
                    "category": "irrelevant",
                    "attributes": {"status": "skip"},
                }
            if "alice likes tea" in lowered:
                return {
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice", "preference": "tea"},
                }
            if "tea routines improve focus" in lowered:
                return {
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus", "driver": "tea routines"},
                }
            return {
                "note_text": "Comprehensive note: Focus helps graph memory systems stay coherent.",
                "context": "Focus reinforces graph-oriented memory work.",
                "keywords": ["focus", "graph", "memory"],
                "tags": ["memory", "graph", "focus"],
                "category": "insight",
                "attributes": {"topic": "graph memory"},
            }
        if "write controller" in system_lower:
            note_text = payload["note_text"].casefold()
            return {
                "decision": "skip" if "discard me" in note_text else "write",
                "reason": "store the note" if "discard me" not in note_text else "irrelevant note",
                "confidence": 0.93,
            }
        if "ai memory evolution agent" in system_lower:
            content = payload["content"].casefold()
            if "alice" in content:
                return {"decision": "NO_EVOLUTION", "reason": "first memory"}
            if "tea routines improve focus" in content:
                return {"decision": "STRENGTHEN_AND_UPDATE", "reason": "shared tea/focus concept"}
            return {"decision": "STRENGTHEN", "reason": "extend graph relation"}
        if "select related neighbor indices" in system_lower:
            return {"connections": [0], "tags": ["memory_bridge", "focus", "tea"]}
        if "update each neighbor's context and tags" in system_lower:
            return {
                "updates": [
                    {
                        "context": "Alice's tea habit is now understood as a focus-supporting routine.",
                        "tags": ["preference", "habit", "focus"],
                    }
                ]
            }
        if "expand the query" in system_lower:
            query = payload["query"]
            lowered = query.casefold()
            return {
                "query_text": query,
                "context": "Retrieve the most relevant agentic memory note.",
                "keywords": ["alice", "tea"] if "alice" in lowered else ["focus", "graph"],
                "tags": ["query", "memory"],
                "category": "query",
                "attributes": {},
            }
        raise AssertionError(f"Unexpected system prompt: {system}")

    def rerank(self, *, query: str, candidates: list[dict[str, object]], task: str, top_k: int):
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.get("score", 0.0)),
                str(candidate.get("id", "")),
            ),
        )
        return [
            {
                "id": str(candidate["id"]),
                "score": float(candidate.get("score", 0.0)),
                "rationale": f"selected for {query}",
            }
            for candidate in sorted_candidates[:top_k]
        ]


@pytest.fixture(autouse=True)
def fake_classic_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", FakeClassicRuntime())


def _amem_pipeline() -> object:
    return build_amem_pipeline(
        config=AMEMConfig(
            top_k=3,
            candidate_k=3,
            neighbor_expansion_k=1,
            max_links_per_record=2,
        )
    )


def test_amem_ingest_generates_comprehensive_note_and_enhanced_embedding() -> None:
    pipeline = _amem_pipeline()

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    unit = packet.units[0]
    assert unit.text == "Alice likes tea."
    assert unit.metadata["amem"]["note_text"].startswith("Comprehensive note:")
    assert unit.metadata["amem"]["context"] == "Alice's tea habit supports her daily routine."
    assert unit.metadata["representation"]["enhanced_embedding_text"].startswith("content: Alice likes tea.")
    assert unit.embedding == _runtime._DEFAULT_RUNTIME.embed(unit.metadata["representation"]["enhanced_embedding_text"])
    record = pipeline.store.iter_records(AMEM_GRAPH_LAYER)[0]
    assert record.text == "Alice likes tea."
    assert record.metadata["amem"]["category"] == "personal_preference"
    assert record.metadata["representation"]["embedding_version"] == "content_context_keywords_tags_v2"


def test_amem_default_write_path_stores_all_notes() -> None:
    pipeline = _amem_pipeline()

    packet = pipeline.ingest(Observation(text="Discard me from memory.", source="dialogue"))

    assert packet.decisions == [True]
    assert pipeline.store.count(AMEM_GRAPH_LAYER) == 1
    assert packet.trace["write_trigger"]["per_unit"][0]["decision"] == "write"
    assert packet.trace["write_trigger"]["per_unit"][0]["reason"] == "write_decision_disabled"


def test_amem_evolution_rewrites_neighbor_context_tags_and_links() -> None:
    pipeline = _amem_pipeline()

    first_packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    second_packet = pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))

    records = pipeline.store.iter_records(AMEM_GRAPH_LAYER)
    first_record_id = first_packet.trace["organization"]["effects"][0]["record_id"]
    second_record_id = second_packet.trace["organization"]["effects"][0]["record_id"]
    first_record = next(record for record in records if record.record_id == first_record_id)
    second_record = next(record for record in records if record.record_id == second_record_id)

    assert second_packet.trace["memory_evolution"]["effects"][0]["decision"] == "STRENGTHEN_AND_UPDATE"
    assert second_packet.trace["memory_evolution"]["effects"][0]["strengthened_links"] == [first_record_id]
    assert second_packet.trace["memory_evolution"]["effects"][0]["updated_neighbor_record_ids"] == [first_record_id]
    assert first_record.metadata["amem"]["context"] == "Alice's tea habit is now understood as a focus-supporting routine."
    assert first_record.metadata["amem"]["tags"] == ["preference", "habit", "focus"]
    assert second_record.metadata["graph"]["links"] == [first_record_id]
    assert first_record.embedding == _runtime._DEFAULT_RUNTIME.embed(
        first_record.metadata["representation"]["enhanced_embedding_text"]
    )


def test_amem_query_without_embedding_uses_plain_query_embedding_and_link_expansion() -> None:
    pipeline = _amem_pipeline()
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))

    packet, _ = pipeline.retrieval.run(Packet(query=Query(text="Alice")), pipeline.store)

    assert packet.query is not None
    assert packet.query.embedding is not None
    assert packet.query.embedding == _runtime._DEFAULT_RUNTIME.embed("Alice")
    assert packet.retrieved is not None
    assert packet.retrieved.trace["retrieval_mode"] == "vector_plus_links"
    assert packet.retrieved.trace["candidate_count"] >= packet.retrieved.trace["selected_count"]
    assert packet.retrieved.items[0].metadata["amem"]["attributes"]["person"] == "Alice"


def test_amem_readout_uses_agentic_memory_format() -> None:
    pipeline = _amem_pipeline()
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert readout.metadata["format"] == "agentic_memory"
    assert readout.metadata["retrieval_mode"] == "vector_plus_links"
    assert readout.source_ids
    assert readout.text.startswith("Query: Alice")
    assert "context:" in readout.text
    assert "tags:" in readout.text
