from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency at runtime
    tiktoken = None

_JSON_START_PATTERN = re.compile(r"[\{\[]")
_MEMPRIMITIVE_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _coerce_json(content: str) -> Any:
    stripped = content.strip()
    if not stripped:
        raise ValueError("Model returned empty content where JSON was required.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in _JSON_START_PATTERN.finditer(stripped):
            try:
                parsed, _end = decoder.raw_decode(stripped, match.start())
            except json.JSONDecodeError:
                continue
            return parsed
        raise


class ClassicRuntime:
    _embedding_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        self.api_key = api_key if api_key is not None else env.get("MEMPRIMITIVE_API_KEY", "")
        self.base_url = base_url if base_url is not None else env.get("MEMPRIMITIVE_BASE_URL", "")
        self.model = model if model is not None else env.get("MEMPRIMITIVE_MODEL", "")
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        # Non-OpenAI-compatible providers often do not support the default tracing path.
        set_tracing_disabled(disabled=True)

    def require_llm(self, *, capability: str) -> None:
        if self.api_key and self.base_url and self.model:
            return
        raise ValueError(
            f"{capability} requires an OpenAI-compatible API. "
            "Set MEMPRIMITIVE_API_KEY, MEMPRIMITIVE_BASE_URL, and MEMPRIMITIVE_MODEL. "
            "Heuristic fallback is not supported."
        )

    def _client(self) -> AsyncOpenAI:
        self.require_llm(capability="Classic runtime LLM access")
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
        result = Runner.run_sync(agent, input=input_text, max_turns=max_turns)
        return result.final_output

    def embed(self, text: str) -> list[float]:
        model = self._embedding_cache.get(self.embedding_model)
        if model is None:
            model = SentenceTransformer(self.embedding_model)
            self._embedding_cache[self.embedding_model] = model
        return [float(value) for value in model.encode(text, normalize_embeddings=True).tolist()]

    def text(self, *, system: str, user: str, temperature: float = 0.0) -> str:
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
        if tiktoken is not None:
            try:
                encoding = tiktoken.encoding_for_model(self.model) if self.model else tiktoken.get_encoding("cl100k_base")
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(cleaned))
        return max(1, math.ceil(len(cleaned) / 4))

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
        payload = json.dumps(
            {
                "query": query,
                "task": task,
                "top_k": top_k,
                "candidates": candidates,
            },
            ensure_ascii=False,
        )
        result = self.json(
            system=(
                "You rank memory candidates for retrieval. "
                "Output JSON as a list of objects with keys: id, score, rationale. "
                "Scores must be floats in [0, 1]. Higher is better."
            ),
            user=payload,
        )
        if not isinstance(result, list):
            raise ValueError("Reranker must return a JSON list.")
        normalized: list[dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("id", "")).strip()
            if not candidate_id:
                continue
            score = item.get("score", 0.0)
            if isinstance(score, bool):
                score = 0.0
            elif not isinstance(score, (int, float)):
                score = 0.0
            normalized.append(
                {
                    "id": candidate_id,
                    "score": max(0.0, min(float(score), 1.0)),
                    "rationale": str(item.get("rationale", "")).strip(),
                }
            )
        by_id = {str(candidate["id"]): candidate for candidate in candidates}
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for item in normalized:
            candidate_id = item["id"]
            if candidate_id not in by_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            ordered.append(item)
            if len(ordered) >= top_k:
                break
        return ordered

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


_DEFAULT_RUNTIME: ClassicRuntime | None = None


def get_classic_runtime() -> ClassicRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = ClassicRuntime()
    return _DEFAULT_RUNTIME
