"""Benchmark dataset adapters for MemPrimitive evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..core import Query
from ._types import BenchmarkSample, ConversationTurn

DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
VALID_BENCHMARKS = frozenset({"locomo", "longmemeval"})
VALID_LONGMEMEVAL_VARIANTS = frozenset({"oracle", "s_cleaned", "m_cleaned"})


class LoCoMoBenchmarkAdapter:
    """Normalize LoCoMo QA samples into the shared benchmark shape."""

    name = "locomo"

    def __init__(self, *, benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> None:
        self.benchmark_root = Path(benchmark_root)

    def iter_samples(self, *, limit: int | None = None) -> Iterator[BenchmarkSample]:
        yielded = 0
        for sample in _iter_locomo_samples(self.benchmark_root):
            yield sample
            yielded += 1
            if limit is not None and yielded >= limit:
                return


class LongMemEvalBenchmarkAdapter:
    """Normalize LongMemEval QA samples into the shared benchmark shape."""

    name = "longmemeval"

    def __init__(
        self,
        *,
        benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
        variant: str = "s_cleaned",
    ) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.variant = str(variant).strip().casefold()
        if self.variant not in VALID_LONGMEMEVAL_VARIANTS:
            raise ValueError(
                f"Unsupported LongMemEval variant {variant!r}. Choose from {sorted(VALID_LONGMEMEVAL_VARIANTS)}."
            )

    def iter_samples(self, *, limit: int | None = None) -> Iterator[BenchmarkSample]:
        yielded = 0
        for sample in _iter_longmemeval_samples(self.benchmark_root, variant=self.variant):
            yield sample
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def create_benchmark_adapter(
    name: str,
    *,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
    longmemeval_variant: str = "s_cleaned",
) -> LoCoMoBenchmarkAdapter | LongMemEvalBenchmarkAdapter:
    """Build one official benchmark adapter by name."""

    benchmark_name = str(name).strip().casefold()
    if benchmark_name == "locomo":
        return LoCoMoBenchmarkAdapter(benchmark_root=benchmark_root)
    if benchmark_name == "longmemeval":
        return LongMemEvalBenchmarkAdapter(
            benchmark_root=benchmark_root,
            variant=longmemeval_variant,
        )
    raise ValueError(f"Unsupported benchmark {name!r}. Choose from {sorted(VALID_BENCHMARKS)}.")


def _iter_locomo_samples(benchmark_root: Path) -> Iterator[BenchmarkSample]:
    path = benchmark_root / "LoCoMo" / "data" / "locomo10.json"
    conversations = json.loads(path.read_text(encoding="utf-8"))
    for conversation_payload in conversations:
        sample_prefix = str(conversation_payload["sample_id"]).strip()
        history_turns = _locomo_history_turns(conversation_payload)
        for qa_index, qa_payload in enumerate(conversation_payload.get("qa", []), start=1):
            question = str(qa_payload.get("question", "")).strip()
            answer = str(qa_payload.get("answer", "")).strip()
            if not question or not answer:
                continue
            yield BenchmarkSample(
                sample_id=f"{sample_prefix}-qa-{qa_index}",
                benchmark_name="locomo",
                history_observations=[],
                history_turns=list(history_turns),
                query=Query(text=question, metadata={"task": "question_answering"}),
                reference_answer=answer,
                metadata={
                    "locomo_sample_id": sample_prefix,
                    "qa_category": qa_payload.get("category"),
                    "evidence": list(qa_payload.get("evidence", [])),
                },
            )


def _locomo_history_turns(conversation_payload: dict[str, Any]) -> list[ConversationTurn]:
    conversation = conversation_payload.get("conversation", {})
    turns: list[ConversationTurn] = []
    session_numbers = sorted(
        int(key.split("_")[1])
        for key in conversation
        if key.startswith("session_") and key.count("_") == 1
    )
    for session_number in session_numbers:
        session_key = f"session_{session_number}"
        session_timestamp = str(conversation.get(f"{session_key}_date_time", "")).strip()
        for turn_index, turn in enumerate(conversation.get(session_key, []), start=1):
            speaker = str(turn.get("speaker", "")).strip() or "speaker"
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            turn_id = str(turn.get("dia_id", "")).strip() or f"{session_key}-turn-{turn_index}"
            turns.append(
                ConversationTurn(
                    turn_id=turn_id,
                    session_id=session_key,
                    session_timestamp=session_timestamp,
                    role=speaker,
                    speaker=speaker,
                    text=text,
                    metadata={
                        "benchmark": "locomo",
                        "turn_index": turn_index,
                        "dialogue_id": turn.get("dia_id"),
                    },
                )
            )
    return turns


def _iter_longmemeval_samples(benchmark_root: Path, *, variant: str) -> Iterator[BenchmarkSample]:
    path = benchmark_root / "LongMemEval" / _longmemeval_filename(variant)
    for row_index, payload in enumerate(_iter_json_array_file(path), start=1):
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            continue
        sample_id = str(payload.get("question_id", "")).strip() or f"longmemeval-{variant}-{row_index}"
        yield BenchmarkSample(
            sample_id=sample_id,
            benchmark_name="longmemeval",
            history_observations=[],
            history_turns=_longmemeval_history_turns(payload),
            query=Query(
                text=question,
                metadata={
                    "task": "question_answering",
                    "question_type": payload.get("question_type"),
                    "question_date": payload.get("question_date"),
                },
            ),
            reference_answer=answer,
            metadata={
                "variant": variant,
                "question_type": payload.get("question_type"),
                "question_date": payload.get("question_date"),
                "answer_session_ids": list(payload.get("answer_session_ids", [])),
                "haystack_session_ids": list(payload.get("haystack_session_ids", [])),
            },
        )


def _longmemeval_filename(variant: str) -> str:
    mapping = {
        "oracle": "longmemeval_oracle.json",
        "s_cleaned": "longmemeval_s_cleaned.json",
        "m_cleaned": "longmemeval_m_cleaned.json",
    }
    return mapping[variant]


def _longmemeval_history_turns(payload: dict[str, Any]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    session_ids = list(payload.get("haystack_session_ids", []))
    session_dates = list(payload.get("haystack_dates", []))
    sessions = list(payload.get("haystack_sessions", []))
    for session_index, session in enumerate(sessions):
        session_id = str(session_ids[session_index]).strip() if session_index < len(session_ids) else str(session_index)
        session_date = str(session_dates[session_index]).strip() if session_index < len(session_dates) else ""
        for turn_index, turn in enumerate(session, start=1):
            role = str(turn.get("role", "")).strip() or "speaker"
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            turns.append(
                ConversationTurn(
                    turn_id=f"{session_id}-turn-{turn_index}",
                    session_id=session_id,
                    session_timestamp=session_date,
                    role=role,
                    speaker=role,
                    text=content,
                    metadata={
                        "benchmark": "longmemeval",
                        "turn_index": turn_index,
                    },
                )
            )
    return turns


def _iter_json_array_file(path: Path, *, chunk_size: int = 1 << 16) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            end_of_file = chunk == ""

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if not array_started:
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array file.")
                    array_started = True
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    payload, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if end_of_file:
                        raise
                    break
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSON array item.")
                yield payload
                buffer = buffer[offset:].lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    return

            if end_of_file:
                break
    raise ValueError(f"JSON array file {path} ended unexpectedly.")
