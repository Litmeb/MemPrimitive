"""Benchmark runners and answer-generation helpers."""

from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from memprimitive.utils._runtime import Runtime
from memprimitive.utils._template import render_prompt_template

from ._types import (
    AnswerRunner,
    BenchmarkPrediction,
    BenchmarkRunResult,
    BenchmarkSample,
    MemoryRecall,
)
from .prompts import ANSWER_PROMPT, MEMMACHINE_ANSWER_PROMPT


def _call_with_supported_kwargs(func, /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    supported_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(*args, **supported_kwargs)


class SingleRecallLLMAnswerRunner:
    """Answer from one retrieved memory block using the existing OpenAI-compatible runtime."""

    name = "single_recall_llm_answer"

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.system_prompt = system_prompt or (
            "You answer only from the provided retrieved memory and the user request. "
            "If the retrieved memory is empty or insufficient, say that the memory does not contain enough information. "
            "Do not invent facts that are not grounded in the retrieved memory."
        )

    def answer(
        self,
        *,
        sample: BenchmarkSample,
        memory_recall: MemoryRecall | None = None,
        retrieved_text: str | None = None,
    ) -> str:
        if memory_recall is None:
            memory_recall = MemoryRecall(text=str(retrieved_text or ""))
        retrieved_block = memory_recall.text.strip() or "<no retrieved memory>"
        user_prompt = (
            f"Benchmark: {sample.benchmark_name}\n"
            f"Sample ID: {sample.sample_id}\n\n"
            f"User request:\n{sample.query.text}\n\n"
            f"Retrieved memory:\n{retrieved_block}\n"
        )
        return self.runtime.text(system=self.system_prompt, user=user_prompt, temperature=0.0)


class Mem0LoCoMoAnswerRunner:
    """Render the upstream Mem0 LoCoMo answer prompt and answer with the real runtime."""

    name = "mem0_locomo_answer"

    def __init__(self, *, runtime: Runtime | None = None, system_prompt: str | None = None) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.system_prompt = system_prompt or ""

    def _locomo_memory_sections(self, memory_recall: MemoryRecall) -> tuple[str, str]:
        metadata = dict(memory_recall.metadata)
        speaker_1_memories = str(metadata.get("speaker_1_memories", "")).strip()
        speaker_2_memories = str(metadata.get("speaker_2_memories", "")).strip()
        if not speaker_1_memories and not speaker_2_memories:
            speaker_1_memories = memory_recall.text.strip()
        return speaker_1_memories, speaker_2_memories

    def answer(
        self,
        *,
        sample: BenchmarkSample,
        memory_recall: MemoryRecall | None = None,
        retrieved_text: str | None = None,
    ) -> str:
        if memory_recall is None:
            memory_recall = MemoryRecall(text=str(retrieved_text or ""))
        speaker_1_name = str(memory_recall.metadata.get("speaker_1_name", "")).strip()
        speaker_2_name = str(memory_recall.metadata.get("speaker_2_name", "")).strip()
        speaker_a = speaker_1_name or str(sample.metadata.get("speaker_a", "")).strip() or "speaker_1"
        speaker_b = speaker_2_name or str(sample.metadata.get("speaker_b", "")).strip() or "speaker_2"
        speaker_1_memories, speaker_2_memories = self._locomo_memory_sections(memory_recall)
        rendered_user_prompt, _ = render_prompt_template(
            ANSWER_PROMPT,
            {
                "speaker_1_user_id": speaker_a,
                "speaker_1_memories": speaker_1_memories,
                "speaker_1_graph_memories": "",
                "speaker_2_user_id": speaker_b,
                "speaker_2_memories": speaker_2_memories,
                "speaker_2_graph_memories": "",
                "question": sample.query.text,
            },
        )
        return self.runtime.text(system=self.system_prompt, user=rendered_user_prompt, temperature=0.0)


class MemMachineLoCoMoAnswerRunner:
    """Render the MemMachine LoCoMo answer prompt against one conversation memory block."""

    name = "memmachine_locomo_answer"

    def __init__(self, *, runtime: Runtime | None = None, system_prompt: str | None = None) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.system_prompt = system_prompt or ""

    def answer(
        self,
        *,
        sample: BenchmarkSample,
        memory_recall: MemoryRecall | None = None,
        retrieved_text: str | None = None,
    ) -> str:
        if memory_recall is None:
            memory_recall = MemoryRecall(text=str(retrieved_text or ""))
        rendered_user_prompt, _ = render_prompt_template(
            MEMMACHINE_ANSWER_PROMPT,
            {
                "conversation_memories": memory_recall.text.strip(),
                "question": sample.query.text,
            },
        )
        return self.runtime.text(system=self.system_prompt, user=rendered_user_prompt, temperature=0.0)


def _answer_with_runner(runner: AnswerRunner | Any, *, sample: BenchmarkSample, memory_recall: MemoryRecall) -> str:
    return str(
        _call_with_supported_kwargs(
            runner.answer,
            sample=sample,
            memory_recall=memory_recall,
            retrieved_text=memory_recall.text,
        )
    )


def write_predictions_jsonl(predictions: Iterable[BenchmarkPrediction], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_json_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def _truncate_sample_history(sample: BenchmarkSample, *, max_history_turns: int | None) -> BenchmarkSample:
    if max_history_turns is None:
        return sample
    if max_history_turns <= 0:
        raise ValueError("max_history_turns must be positive.")
    if len(sample.history_turns) <= max_history_turns and len(sample.history_observations) <= max_history_turns:
        return sample

    history_turns = list(sample.history_turns[:max_history_turns])
    if history_turns:
        history_observations = [turn.to_observation() for turn in history_turns]
    else:
        history_observations = list(sample.history_observations[:max_history_turns])
    return BenchmarkSample(
        sample_id=sample.sample_id,
        benchmark_name=sample.benchmark_name,
        history_observations=history_observations,
        history_turns=history_turns,
        query=sample.query,
        reference_answer=sample.reference_answer,
        metadata={
            **dict(sample.metadata),
            "original_history_turn_count": len(sample.history_turns),
            "original_history_observation_count": len(sample.history_observations),
            "max_history_turns": max_history_turns,
        },
    )


def _memory_session_key(memory_adapter: Any, sample: BenchmarkSample) -> str | None:
    session_key_fn = getattr(memory_adapter, "session_key", None)
    if not callable(session_key_fn):
        return None
    key = _call_with_supported_kwargs(session_key_fn, sample=sample)
    normalized = str(key or "").strip()
    return normalized or None


def _memory_progress_turn_total(memory_adapter: Any, samples: list[BenchmarkSample]) -> int:
    seen_keys: set[str] = set()
    total = 0
    for sample in samples:
        key = _memory_session_key(memory_adapter, sample)
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        total += len(sample.history_turns)
    return total


def _load_memory_session(
    memory_adapter: Any,
    *,
    sample: BenchmarkSample,
    index: int,
    total: int,
    session_cache: dict[str, Any],
    progress_callback: Callable[..., None] | None,
) -> Any:
    session_key = _memory_session_key(memory_adapter, sample)
    if session_key is not None and session_key in session_cache:
        session = session_cache[session_key]
        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_reuse",
                index=index,
                total=total,
                sample=sample,
                session_key=session_key,
            )
        return session

    session = memory_adapter.create_session()
    _call_with_supported_kwargs(
        session.load_case,
        sample,
        progress_callback=progress_callback,
        sample_index=index,
        total_samples=total,
    )
    if session_key is not None:
        session_cache[session_key] = session
    return session


def _build_prediction(
    *,
    runner: AnswerRunner | Any,
    memory_adapter: Any,
    sample: BenchmarkSample,
    session: Any,
) -> BenchmarkPrediction:
    memory_recall = session.recall(sample.query, sample=sample)
    predicted_answer = _answer_with_runner(runner, sample=sample, memory_recall=memory_recall)
    return BenchmarkPrediction(
        sample_id=sample.sample_id,
        benchmark_name=sample.benchmark_name,
        query_text=sample.query.text,
        reference_answer=sample.reference_answer,
        predicted_answer=predicted_answer,
        retrieved_text=memory_recall.text,
        retrieved_source_ids=list(memory_recall.source_ids),
        metadata={
            **sample.metadata,
            "history_turn_count": len(sample.history_turns),
            "history_observation_count": len(sample.history_observations),
            "query_metadata": dict(sample.query.metadata),
            **dict(memory_recall.metadata),
        },
        memory_adapter_name=str(memory_adapter.name),
        memory_metadata=dict(memory_recall.metadata),
    )


def _score_prediction(benchmark_adapter: Any, prediction: BenchmarkPrediction) -> None:
    score_fn = getattr(benchmark_adapter, "score_prediction", None)
    if not callable(score_fn):
        return
    score_value = _call_with_supported_kwargs(score_fn, prediction=prediction)
    if isinstance(score_value, dict):
        prediction.scores = dict(score_value)


def run_benchmark(
    benchmark_adapter,
    memory_adapter,
    *,
    answer_runner: AnswerRunner | None = None,
    limit: int | None = None,
    max_history_turns: int | None = None,
    max_workers: int = 1,
    progress_callback: Callable[..., None] | None = None,
) -> BenchmarkRunResult:
    """Run one memory adapter against one benchmark adapter."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    runner = answer_runner if answer_runner is not None else SingleRecallLLMAnswerRunner()
    predictions: list[BenchmarkPrediction] = []
    samples = [
        _truncate_sample_history(sample, max_history_turns=max_history_turns)
        for sample in benchmark_adapter.iter_samples(limit=limit)
    ]
    total = len(samples)
    if progress_callback is not None:
        _call_with_supported_kwargs(
            progress_callback,
            phase="init",
            total=total,
            samples=samples,
            memory_turn_total=_memory_progress_turn_total(memory_adapter, samples),
        )
    session_cache: dict[str, Any] = {}
    if max_workers == 1:
        for index, sample in enumerate(samples, start=1):
            if progress_callback is not None:
                _call_with_supported_kwargs(
                    progress_callback,
                    phase="start",
                    index=index,
                    total=total,
                    sample=sample,
                )
            session = _load_memory_session(
                memory_adapter,
                sample=sample,
                index=index,
                total=total,
                session_cache=session_cache,
                progress_callback=progress_callback,
            )
            prediction = _build_prediction(
                runner=runner,
                memory_adapter=memory_adapter,
                sample=sample,
                session=session,
            )
            _score_prediction(benchmark_adapter, prediction)
            predictions.append(prediction)
            if progress_callback is not None:
                _call_with_supported_kwargs(
                    progress_callback,
                    phase="done",
                    index=index,
                    total=total,
                    sample=sample,
                    prediction=prediction,
                )
    else:
        predictions_by_index: dict[int, BenchmarkPrediction] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {}
            for index, sample in enumerate(samples, start=1):
                if progress_callback is not None:
                    _call_with_supported_kwargs(
                        progress_callback,
                        phase="start",
                        index=index,
                        total=total,
                        sample=sample,
                    )
                session = _load_memory_session(
                    memory_adapter,
                    sample=sample,
                    index=index,
                    total=total,
                    session_cache=session_cache,
                    progress_callback=progress_callback,
                )
                future = executor.submit(
                    _build_prediction,
                    runner=runner,
                    memory_adapter=memory_adapter,
                    sample=sample,
                    session=session,
                )
                future_to_item[future] = (index, sample)

            for future in as_completed(future_to_item):
                index, sample = future_to_item[future]
                prediction = future.result()
                _score_prediction(benchmark_adapter, prediction)
                predictions_by_index[index] = prediction
                if progress_callback is not None:
                    _call_with_supported_kwargs(
                        progress_callback,
                        phase="done",
                        index=index,
                        total=total,
                        sample=sample,
                        prediction=prediction,
                    )
        predictions = [predictions_by_index[index] for index in sorted(predictions_by_index)]

    aggregate_scores: dict[str, Any] = {}
    aggregate_fn = getattr(benchmark_adapter, "aggregate_scores", None)
    if callable(aggregate_fn):
        aggregate_value = _call_with_supported_kwargs(aggregate_fn, predictions=predictions)
        if isinstance(aggregate_value, dict):
            aggregate_scores = dict(aggregate_value)

    if progress_callback is not None:
        _call_with_supported_kwargs(
            progress_callback,
            phase="finish",
            total=total,
            predictions=predictions,
        )

    return BenchmarkRunResult(
        benchmark_name=str(getattr(benchmark_adapter, "name", "")),
        memory_adapter_name=str(getattr(memory_adapter, "name", "")),
        predictions=predictions,
        aggregate_scores=aggregate_scores,
    )
