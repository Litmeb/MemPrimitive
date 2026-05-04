from __future__ import annotations

import json
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


def test_runtime_rerank_calls_openai_compatible_rerank_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"results": ['
                b'{"index": 1, "relevance_score": 0.9},'
                b'{"index": 0, "relevance_score": 0.25}'
                b"]}"
            )

    def _fake_urlopen(request: object, *, timeout: int) -> _FakeResponse:
        body = json.loads(request.data.decode("utf-8"))
        calls.append(
            {
                "url": request.full_url,
                "authorization": request.get_header("Authorization"),
                "content_type": request.get_header("Content-type"),
                "timeout": timeout,
                "body": body,
            }
        )
        return _FakeResponse()

    monkeypatch.setattr(_runtime.urllib.request, "urlopen", _fake_urlopen)

    reranked = _runtime.Runtime(
        rerank_api_key="rerank-key",
        rerank_base_url="https://rerank.example/v1/",
        rerank_model="rerank-test",
    ).rerank(
        query="tea preference",
        candidates=[
            {"id": "a", "content": "Alice likes coffee."},
            {"id": "b", "text": "Alice likes jasmine tea."},
        ],
        task="ignored by the API-backed reranker",
        top_k=2,
    )

    assert reranked == [
        {"id": "b", "score": 0.9, "rationale": ""},
        {"id": "a", "score": 0.25, "rationale": ""},
    ]
    assert calls == [
        {
            "url": "https://rerank.example/v1/rerank",
            "authorization": "Bearer rerank-key",
            "content_type": "application/json",
            "timeout": 60,
            "body": {
                "model": "rerank-test",
                "query": "tea preference",
                "documents": ["Alice likes coffee.", "Alice likes jasmine tea."],
                "top_n": 2,
                "return_documents": False,
            },
        }
    ]


def test_runtime_rerank_requires_rerank_api_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMPRIMITIVE_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("MEMPRIMITIVE_RERANK_BASE_URL", raising=False)
    monkeypatch.delenv("MEMPRIMITIVE_RERANK_MODEL", raising=False)
    monkeypatch.setattr(_runtime, "load_dotenv", lambda *args, **kwargs: None)

    runtime = _runtime.Runtime()

    with pytest.raises(ValueError, match="MEMPRIMITIVE_RERANK_API_KEY"):
        runtime.rerank(
            query="tea",
            candidates=[{"id": "a", "content": "Alice likes tea."}],
            task="rank",
            top_k=1,
        )


def test_runtime_rerank_ignores_duplicate_and_out_of_range_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"results": ['
                b'{"index": 2, "relevance_score": 0.99},'
                b'{"index": 1, "relevance_score": 0.8},'
                b'{"index": 1, "relevance_score": 0.7},'
                b'{"index": 0, "relevance_score": "bad"}'
                b"]}"
            )

    monkeypatch.setattr(
        _runtime.urllib.request,
        "urlopen",
        lambda request, *, timeout: _FakeResponse(),
    )

    reranked = _runtime.Runtime(
        rerank_api_key="rerank-key",
        rerank_base_url="https://rerank.example/v1",
        rerank_model="rerank-test",
    ).rerank(
        query="tea",
        candidates=[
            {"id": "a", "content": "Alice likes coffee."},
            {"id": "b", "content": "Alice likes tea."},
        ],
        task="rank",
        top_k=2,
    )

    assert reranked == [{"id": "b", "score": 0.8, "rationale": ""}]


def test_runtime_rerank_rejects_invalid_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"not_results": []}'

    monkeypatch.setattr(
        _runtime.urllib.request,
        "urlopen",
        lambda request, *, timeout: _FakeResponse(),
    )

    runtime = _runtime.Runtime(
        rerank_api_key="rerank-key",
        rerank_base_url="https://rerank.example/v1",
        rerank_model="rerank-test",
    )

    with pytest.raises(ValueError, match="results list"):
        runtime.rerank(
            query="tea",
            candidates=[{"id": "a", "content": "Alice likes tea."}],
            task="rank",
            top_k=1,
        )


def test_runtime_rejects_unsupported_embedding_provider() -> None:
    with pytest.raises(ValueError, match="MEMPRIMITIVE_EMBEDDING_PROVIDER"):
        _runtime.Runtime(embedding_provider="unsupported")


def test_coerce_json_accepts_markdown_fenced_json() -> None:
    assert _runtime._coerce_json('```json\n["art", "pottery"]\n```') == ["art", "pottery"]


def test_coerce_json_repairs_single_truncated_fenced_array() -> None:
    content = '```json\n[\n  "Melanie likes painting and pottery."\n```'

    assert _runtime._coerce_json(content) == ["Melanie likes painting and pottery."]
