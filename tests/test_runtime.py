from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from agents.exceptions import MaxTurnsExceeded
from pydantic import BaseModel

from memprimitive.utils import _runtime


class _TinyStructured(BaseModel):
    field: int = 0


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
            "timeout": 180,
            "body": {
                "model": "rerank-test",
                "query": "tea preference",
                "documents": ["Alice likes coffee.", "Alice likes jasmine tea."],
                "top_n": 2,
                "return_documents": False,
            },
        }
    ]


def test_runtime_rerank_uses_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"results":[{"index":0,"relevance_score":0.5}]}'

    def _fake_urlopen(request: object, *, timeout: int) -> _FakeResponse:
        seen["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(_runtime.urllib.request, "urlopen", _fake_urlopen)

    reranked = _runtime.Runtime(
        rerank_api_key="rerank-key",
        rerank_base_url="https://rerank.example/v1",
        rerank_model="rerank-test",
        rerank_timeout_seconds=321,
    ).rerank(
        query="tea",
        candidates=[{"id": "a", "content": "Alice likes tea."}],
        task="rank",
        top_k=1,
    )

    assert reranked == [{"id": "a", "score": 0.5, "rationale": ""}]
    assert seen["timeout"] == 321


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


def test_runtime_sets_windows_selector_event_loop_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    selector_policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy_cls is None:
        pytest.skip("Windows selector event loop policy not available on this platform.")

    class _FakeNonSelectorPolicy(asyncio.DefaultEventLoopPolicy):
        pass

    original_platform = _runtime.sys.platform
    original_get = _runtime.asyncio.get_event_loop_policy
    original_set = _runtime.asyncio.set_event_loop_policy
    policy_box: dict[str, asyncio.AbstractEventLoopPolicy] = {"policy": _FakeNonSelectorPolicy()}
    set_calls: list[asyncio.AbstractEventLoopPolicy] = []

    monkeypatch.setattr(_runtime.sys, "platform", "win32")

    def _fake_get_event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
        return policy_box["policy"]

    def _fake_set_event_loop_policy(policy: asyncio.AbstractEventLoopPolicy) -> None:
        policy_box["policy"] = policy
        set_calls.append(policy)

    monkeypatch.setattr(_runtime.asyncio, "get_event_loop_policy", _fake_get_event_loop_policy)
    monkeypatch.setattr(_runtime.asyncio, "set_event_loop_policy", _fake_set_event_loop_policy)

    try:
        _runtime._ensure_windows_selector_event_loop_policy()
    finally:
        monkeypatch.setattr(_runtime.sys, "platform", original_platform)
        monkeypatch.setattr(_runtime.asyncio, "get_event_loop_policy", original_get)
        monkeypatch.setattr(_runtime.asyncio, "set_event_loop_policy", original_set)

    assert len(set_calls) == 1
    assert isinstance(policy_box["policy"], selector_policy_cls)


def test_runtime_truncate_text_to_token_limit_uses_heuristic_without_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_runtime, "tiktoken", None)
    runtime = _runtime.Runtime(model="test-model")
    body = "a" * 200
    assert runtime.count_tokens(body) == 50
    truncated = runtime.truncate_text_to_token_limit(body, 12)
    assert runtime.count_tokens(truncated) <= 12
    assert len(truncated) < len(body)


def test_runtime_text_truncates_user_for_max_input_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_runtime, "tiktoken", None)
    captured: dict[str, str] = {}

    def _recording_run_agent(self: _runtime.Runtime, **kwargs: object) -> str:
        captured["instructions"] = str(kwargs.get("instructions", ""))
        captured["input_text"] = str(kwargs.get("input_text", ""))
        return "ok"

    monkeypatch.setattr(_runtime.Runtime, "run_agent", _recording_run_agent)
    runtime = _runtime.Runtime(model="stub")
    runtime.text(system="s" * 40, user="u" * 400, max_input_tokens=20)
    assert captured["instructions"] == "s" * 40
    assert captured["input_text"] == "u" * 40


def test_coerce_json_accepts_markdown_fenced_json() -> None:
    assert _runtime._coerce_json('```json\n["art", "pottery"]\n```') == ["art", "pottery"]


def test_coerce_json_repairs_single_truncated_fenced_array() -> None:
    content = '```json\n[\n  "Melanie likes painting and pottery."\n```'

    assert _runtime._coerce_json(content) == ["Melanie likes painting and pottery."]


def test_run_agent_max_turns_exceeded_returns_empty_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_max_turns(*_args: object, **_kwargs: object) -> None:
        raise MaxTurnsExceeded("Max turns (3) exceeded")

    monkeypatch.setattr(_runtime.Runner, "run_sync", _raise_max_turns)
    runtime = _runtime.Runtime(api_key="k", base_url="http://example/v1", model="m")
    assert (
        runtime.run_agent(
            name="t",
            instructions="i",
            input_text="{}",
            max_turns=3,
        )
        == ""
    )


def test_run_agent_max_turns_exceeded_returns_default_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_max_turns(*_args: object, **_kwargs: object) -> None:
        raise MaxTurnsExceeded("Max turns (3) exceeded")

    monkeypatch.setattr(_runtime.Runner, "run_sync", _raise_max_turns)
    runtime = _runtime.Runtime(api_key="k", base_url="http://example/v1", model="m")
    out = runtime.run_agent(
        name="t",
        instructions="i",
        input_text="{}",
        max_turns=3,
        output_type=_TinyStructured,
    )
    assert isinstance(out, _TinyStructured)
    assert out.field == 0
