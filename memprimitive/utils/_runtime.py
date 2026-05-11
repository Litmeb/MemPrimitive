from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.exceptions import MaxTurnsExceeded
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency at runtime
    tiktoken = None

_JSON_START_PATTERN = re.compile(r"[\{\[]")
_MEMPRIMITIVE_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_LOCAL_EMBEDDING_PROVIDERS = frozenset({"sentence_transformers", "sentence-transformers", "local"})
_OPENAI_EMBEDDING_PROVIDER = "openai"


def _ensure_windows_selector_event_loop_policy() -> None:
    if not sys.platform.startswith("win"):
        return
    selector_policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy_cls is None:
        return
    policy = asyncio.get_event_loop_policy()
    if isinstance(policy, selector_policy_cls):
        return
    asyncio.set_event_loop_policy(selector_policy_cls())


def _coerce_json(content: str) -> Any:
    stripped = content.strip()
    if not stripped:
        raise ValueError("Model returned empty content where JSON was required.")
    first_error: json.JSONDecodeError | None = None
    decoder = json.JSONDecoder()
    for candidate in _json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc
        for match in _JSON_START_PATTERN.finditer(candidate):
            raw_candidate = candidate[match.start():].strip()
            try:
                parsed, _end = decoder.raw_decode(raw_candidate)
                return parsed
            except json.JSONDecodeError as exc:
                if first_error is None:
                    first_error = exc
            repaired = _repair_truncated_json(raw_candidate)
            if repaired is None:
                continue
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                if first_error is None:
                    first_error = exc
    if first_error is None:
        first_error = json.JSONDecodeError("Expecting value", stripped, 0)
    preview = stripped[:500].replace("\n", "\\n")
    raise ValueError(f"Model returned non-JSON content where JSON was required: {preview!r}") from first_error


def _json_candidates(stripped: str) -> list[str]:
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        unfenced = "\n".join(lines).strip()
        if unfenced:
            candidates.append(unfenced)
    return list(dict.fromkeys(candidates))


def _repair_truncated_json(candidate: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escape = False
    for char in candidate:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    return candidate + "".join(reversed(stack))


class Runtime:
    _embedding_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        rerank_api_key: str | None = None,
        rerank_base_url: str | None = None,
        rerank_model: str | None = None,
        rerank_timeout_seconds: int | None = None,
    ) -> None:
        _ensure_windows_selector_event_loop_policy()
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        self.api_key = api_key if api_key is not None else env.get("MEMPRIMITIVE_API_KEY", "")
        self.base_url = base_url if base_url is not None else env.get("MEMPRIMITIVE_BASE_URL", "")
        self.model = model if model is not None else env.get("MEMPRIMITIVE_MODEL", "")
        raw_embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else env.get("MEMPRIMITIVE_EMBEDDING_PROVIDER", "sentence_transformers")
        )
        self.embedding_provider = self._normalize_embedding_provider(raw_embedding_provider)
        raw_embedding_model = embedding_model or env.get("MEMPRIMITIVE_EMBEDDING_MODEL", "")
        self.embedding_model = (
            raw_embedding_model
            if raw_embedding_model
            else (
                _DEFAULT_SENTENCE_TRANSFORMERS_MODEL
                if self.embedding_provider == "sentence_transformers"
                else ""
            )
        )
        self.embedding_api_key = (
            embedding_api_key
            if embedding_api_key is not None
            else env.get("MEMPRIMITIVE_EMBEDDING_API_KEY", "")
        )
        self.embedding_base_url = (
            embedding_base_url
            if embedding_base_url is not None
            else env.get("MEMPRIMITIVE_EMBEDDING_BASE_URL", "")
        )
        self.rerank_api_key = (
            rerank_api_key
            if rerank_api_key is not None
            else env.get("MEMPRIMITIVE_RERANK_API_KEY", "")
        )
        self.rerank_base_url = (
            rerank_base_url
            if rerank_base_url is not None
            else env.get("MEMPRIMITIVE_RERANK_BASE_URL", "")
        )
        self.rerank_model = (
            rerank_model
            if rerank_model is not None
            else env.get("MEMPRIMITIVE_RERANK_MODEL", "")
        )
        raw_rerank_timeout = (
            rerank_timeout_seconds
            if rerank_timeout_seconds is not None
            else env.get("MEMPRIMITIVE_RERANK_TIMEOUT_SECONDS", "180")
        )
        self.rerank_timeout_seconds = max(1, int(raw_rerank_timeout))
        # Non-OpenAI-compatible providers often do not support the default tracing path.
        set_tracing_disabled(disabled=True)

    def _tiktoken_encoding(self):  # type: ignore[no-untyped-def]
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model) if self.model else tiktoken.get_encoding("cl100k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _normalize_embedding_provider(provider: str | None) -> str:
        normalized = str(provider or "sentence_transformers").strip().casefold().replace("-", "_")
        if normalized in {value.replace("-", "_") for value in _LOCAL_EMBEDDING_PROVIDERS}:
            return "sentence_transformers"
        if normalized == _OPENAI_EMBEDDING_PROVIDER:
            return _OPENAI_EMBEDDING_PROVIDER
        raise ValueError(
            "MEMPRIMITIVE_EMBEDDING_PROVIDER must be one of: sentence_transformers, openai."
        )

    def require_llm(self, *, capability: str) -> None:
        if self.api_key and self.base_url and self.model:
            return
        raise ValueError(
            f"{capability} requires an OpenAI-compatible API. "
            "Set MEMPRIMITIVE_API_KEY, MEMPRIMITIVE_BASE_URL, and MEMPRIMITIVE_MODEL. "
            "Heuristic fallback is not supported."
        )

    def _client(self) -> AsyncOpenAI:
        self.require_llm(capability="MemPrimitive runtime LLM access")
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _model(self) -> OpenAIChatCompletionsModel:
        return OpenAIChatCompletionsModel(model=self.model, openai_client=self._client())

    def run_agent(
        self,
        *,
        name: str,
        instructions: str,
        input_text: str,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        max_turns: int = 10,
        output_type: type[Any] | None = None,
    ) -> Any:
        agent = Agent(
            name=name,
            instructions=instructions,
            model=self._model(),
            model_settings=ModelSettings(temperature=temperature),
            tools=[] if tools is None else list(tools),
            output_type=output_type,
        )
        try:
            result = Runner.run_sync(agent, input=input_text, max_turns=max_turns)
        except MaxTurnsExceeded:
            # Completed tool rounds are already applied; skip only the unfinished final step.
            if output_type is None:
                return ""
            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                return output_type()
            raise
        return result.final_output

    def embed(self, text: str) -> list[float]:
        if self.embedding_provider == _OPENAI_EMBEDDING_PROVIDER:
            return self._embed_with_openai(text)
        return self._embed_with_sentence_transformers(text)

    def _embed_with_sentence_transformers(self, text: str) -> list[float]:
        model = self._embedding_cache.get(self.embedding_model)
        if model is None:
            model = SentenceTransformer(self.embedding_model)
            self._embedding_cache[self.embedding_model] = model
        return [float(value) for value in model.encode(text, normalize_embeddings=True).tolist()]

    def _embed_with_openai(self, text: str) -> list[float]:
        self.require_embedding_api()
        client = OpenAI(api_key=self.embedding_api_key, base_url=self.embedding_base_url)
        response = client.embeddings.create(model=self.embedding_model, input=text)
        if not response.data:
            raise ValueError("Embedding API returned no embedding data.")
        return [float(value) for value in response.data[0].embedding]

    def require_embedding_api(self) -> None:
        if self.embedding_api_key and self.embedding_base_url and self.embedding_model:
            return
        raise ValueError(
            "Embedding API access requires MEMPRIMITIVE_EMBEDDING_API_KEY, "
            "MEMPRIMITIVE_EMBEDDING_BASE_URL, and MEMPRIMITIVE_EMBEDDING_MODEL."
        )

    def require_rerank_api(self) -> None:
        if self.rerank_api_key and self.rerank_base_url and self.rerank_model:
            return
        raise ValueError(
            "Reranker API access requires MEMPRIMITIVE_RERANK_API_KEY, "
            "MEMPRIMITIVE_RERANK_BASE_URL, and MEMPRIMITIVE_RERANK_MODEL."
        )

    def text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_input_tokens: int | None = None,
    ) -> str:
        if max_input_tokens is not None:
            if max_input_tokens <= 0:
                raise ValueError("max_input_tokens must be positive.")
            sys_tokens = self.count_tokens(system)
            user_tokens = self.count_tokens(user)
            if sys_tokens + user_tokens > max_input_tokens:
                if sys_tokens >= max_input_tokens:
                    system = self.truncate_text_to_token_limit(system, max_input_tokens)
                    user = ""
                else:
                    user = self.truncate_text_to_token_limit(user, max_input_tokens - self.count_tokens(system))
        output = self.run_agent(
            name="MemPrimitiveTextAgent",
            instructions=system,
            input_text=user,
            temperature=temperature,
            max_turns=1,
        )
        return str(output or "").strip()

    def json(self, *, system: str, user: str) -> Any:
        content = self.text(
            system=system + "\nReturn strict JSON only.",
            user=user,
            temperature=0.0,
        )
        return _coerce_json(content)

    def count_tokens(self, text: str) -> int:
        cleaned = str(text)
        if not cleaned:
            return 0
        encoding = self._tiktoken_encoding()
        if encoding is not None:
            return len(encoding.encode(cleaned))
        return max(1, math.ceil(len(cleaned) / 4))

    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text so ``count_tokens`` is at most ``max_tokens`` (uses tiktoken when installed)."""

        if max_tokens <= 0:
            return ""
        cleaned = str(text)
        if not cleaned:
            return ""
        encoding = self._tiktoken_encoding()
        if encoding is not None:
            token_ids = encoding.encode(cleaned)
            if len(token_ids) <= max_tokens:
                return cleaned
            return encoding.decode(token_ids[:max_tokens])
        if self.count_tokens(cleaned) <= max_tokens:
            return cleaned
        lo, hi = 0, len(cleaned)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.count_tokens(cleaned[:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return cleaned[:best]

    def summarize_records(
        self,
        *,
        records: list[dict[str, Any]],
        instruction: str,
        max_sentences: int = 3,
    ) -> str:
        payload = json.dumps({"records": records, "max_sentences": max_sentences}, ensure_ascii=False)
        return self.text(
            system="You compress memory records into concise factual summaries.",
            user=f"{instruction}\n{payload}",
        )

    def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        task: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if top_k <= 0:
            return []
        self.require_rerank_api()

        ranked_candidates: list[dict[str, Any]] = []
        documents: list[str] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("id", "")).strip()
            if not candidate_id:
                continue
            ranked_candidates.append(candidate)
            documents.append(str(candidate.get("content", candidate.get("text", ""))))
        if not ranked_candidates:
            return []

        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(int(top_k), len(documents)),
            "return_documents": False,
        }
        result = self._post_rerank(payload)
        if not isinstance(result, dict):
            raise ValueError("Reranker API must return a JSON object.")
        raw_results = result.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Reranker API response must include a results list.")

        seen_indexes: set[int] = set()
        ordered: list[dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            if index < 0 or index >= len(ranked_candidates) or index in seen_indexes:
                continue
            score = item.get("relevance_score", item.get("score", 0.0))
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            candidate_id = str(ranked_candidates[index]["id"]).strip()
            if not candidate_id:
                continue
            seen_indexes.add(index)
            ordered.append(
                {
                    "id": candidate_id,
                    "score": max(0.0, min(float(score), 1.0)),
                    "rationale": "",
                }
            )
            if len(ordered) >= top_k:
                break
        return ordered

    def _post_rerank(self, payload: dict[str, Any]) -> Any:
        url = f"{self.rerank_base_url.rstrip('/')}/rerank"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.rerank_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Reranker API request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"Reranker API request failed: {exc.reason}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            preview = body[:500].replace("\n", "\\n")
            raise ValueError(f"Reranker API returned non-JSON content: {preview!r}") from exc

    @staticmethod
    def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
        if left is None or right is None or len(left) != len(right) or not left:
            return 0.0
        numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


_DEFAULT_RUNTIME: Runtime | None = None


def get_runtime() -> Runtime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = Runtime()
    return _DEFAULT_RUNTIME
