from __future__ import annotations

from typing import Any

import pytest

from memprimitive.example.classics.simplemem_memory import (
    build_simplemem_memory_system,
    ingest_dialogue_line,
    recall_simplemem_memory,
)


class _FakeSimpleMemRuntime:
    def embed(self, text: str) -> list[float]:
        lower = text.casefold()
        return [
            1.0 if "starbucks" in lower or "meet" in lower else 0.0,
            1.0 if "report" in lower or "planning" in lower else 0.0,
        ]

    def require_llm(self, *, capability: str) -> None:
        _ = capability

    def json(self, *, system: str, user: str, temperature: float = 0.0) -> Any:
        _ = system, temperature
        if "structured memory retrieval" in user or "structured retrieval constraints" in user:
            return {
                "keywords": ["Alice", "Bob", "Starbucks"],
                "persons": ["Alice", "Bob"],
                "location": "Starbucks",
                "entities": [],
                "time_start": "2025-11-16T00:00:00",
                "time_end": "2025-11-16T23:59:59",
            }
        if "Return a JSON array" in user or "structured memory entries" in user:
            return [
                {
                    "lossless_restatement": "Alice suggested meeting Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
                    "keywords": "Alice,Bob,Starbucks,new product,meeting",
                    "timestamp": "2025-11-16T14:00:00",
                    "location": "Starbucks",
                    "persons": "Alice,Bob",
                    "entities": "new product",
                    "topic": "Product discussion meeting arrangement",
                }
            ]
        if "key `queries`" in user:
            return {"queries": ["When and where will Alice and Bob meet?", "Alice Bob Starbucks meeting time"]}
        return {"queries": ["fallback query"]}


def test_simplemem_classic_wires_window_compression_and_hybrid_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, EmbeddingSimilarityRetrieval, LLMRepresentation
    from memprimitive.utils import _runtime

    fake_runtime = _FakeSimpleMemRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", lambda self, text: fake_runtime.embed(text))
    monkeypatch.setattr(ConfigurableEmbeddingRepresentation, "_embed_text", lambda self, text: fake_runtime.embed(text))
    monkeypatch.setattr(LLMRepresentation, "_llm_json", lambda self, *, user: fake_runtime.json(system="", user=user))

    system = build_simplemem_memory_system(window_size=2, semantic_top_k=3, keyword_top_k=2, structured_top_k=2)
    session_id = "sess-simplemem"
    ingest_dialogue_line(
        system,
        speaker="Alice",
        content="Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product.",
        session_id=session_id,
        timestamp="2025-11-15T14:30:00",
    )
    ingest_dialogue_line(
        system,
        speaker="Bob",
        content="Sure, I'll prepare the market analysis report.",
        session_id=session_id,
        timestamp="2025-11-15T14:31:00",
    )

    store = system["store"]
    assert store.count("memory_units") == 1

    recall = recall_simplemem_memory(system, user_query="When and where will Alice and Bob meet?")
    assert "<SIMPLEMEM MEMORIES>" in recall.text
    assert "Starbucks" in recall.text
    assert recall.source_ids
