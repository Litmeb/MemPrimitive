from __future__ import annotations

from types import SimpleNamespace

import pytest

from memprimitive.utils import _runtime


def test_runtime_embed_defaults_to_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeEncoded:
        def tolist(self) -> list[float]:
            return [1, 2.5, 3]

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            calls.append({"model_name": model_name})

        def encode(self, text: str, *, normalize_embeddings: bool) -> _FakeEncoded:
            calls.append({"text": text, "normalize_embeddings": normalize_embeddings})
            return _FakeEncoded()

    monkeypatch.delenv("MEMPRIMITIVE_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("MEMPRIMITIVE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(_runtime, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(_runtime, "SentenceTransformer", _FakeSentenceTransformer)
    _runtime.Runtime._embedding_cache.clear()

    embedding = _runtime.Runtime().embed("Alice likes tea")

    assert embedding == [1.0, 2.5, 3.0]
    assert calls == [
        {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        {"text": "Alice likes tea", "normalize_embeddings": True},
    ]


def test_runtime_embed_openai_provider_calls_embeddings_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeEmbeddings:
        def create(self, *, model: str, input: str) -> SimpleNamespace:
            calls.append({"model": model, "input": input})
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0, "1.5", 2])])

    class _FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            calls.append({"api_key": api_key, "base_url": base_url})
            self.embeddings = _FakeEmbeddings()

    monkeypatch.setattr(_runtime, "OpenAI", _FakeOpenAI)

    embedding = _runtime.Runtime(
        embedding_provider="openai",
        embedding_api_key="embed-key",
        embedding_base_url="https://embeddings.example/v1",
        embedding_model="text-embedding-test",
    ).embed("hello")

    assert embedding == [0.0, 1.5, 2.0]
    assert calls == [
        {"api_key": "embed-key", "base_url": "https://embeddings.example/v1"},
        {"model": "text-embedding-test", "input": "hello"},
    ]


def test_runtime_embed_openai_provider_requires_embedding_api_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEMPRIMITIVE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("MEMPRIMITIVE_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("MEMPRIMITIVE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(_runtime, "load_dotenv", lambda *args, **kwargs: None)

    runtime = _runtime.Runtime(embedding_provider="openai")

    with pytest.raises(ValueError, match="MEMPRIMITIVE_EMBEDDING_API_KEY"):
        runtime.embed("hello")


def test_runtime_rejects_unsupported_embedding_provider() -> None:
    with pytest.raises(ValueError, match="MEMPRIMITIVE_EMBEDDING_PROVIDER"):
        _runtime.Runtime(embedding_provider="unsupported")


def test_coerce_json_accepts_markdown_fenced_json() -> None:
    assert _runtime._coerce_json('```json\n["art", "pottery"]\n```') == ["art", "pottery"]


def test_coerce_json_repairs_single_truncated_fenced_array() -> None:
    content = '```json\n[\n  "Melanie likes painting and pottery."\n```'

    assert _runtime._coerce_json(content) == ["Melanie likes painting and pottery."]
