"""Benchmark runners and answer-generation helpers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Iterable

from memprimitive.utils._runtime import Runtime

from ._types import (
    AnswerRunner,
    BenchmarkPrediction,
    BenchmarkRunResult,
    BenchmarkSample,
    MemoryRecall,
)


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


def run_benchmark(
    benchmark_adapter,
    memory_adapter,
    *,
    answer_runner: AnswerRunner | None = None,
    limit: int | None = None,
) -> BenchmarkRunResult:
    """Run one memory adapter against one benchmark adapter."""

    runner = answer_runner if answer_runner is not None else SingleRecallLLMAnswerRunner()
    predictions: list[BenchmarkPrediction] = []
    for sample in benchmark_adapter.iter_samples(limit=limit):
        session = memory_adapter.create_session()
        session.load_case(sample)
        memory_recall = session.recall(sample.query, sample=sample)
        predicted_answer = _answer_with_runner(runner, sample=sample, memory_recall=memory_recall)
        prediction = BenchmarkPrediction(
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
            },
            memory_adapter_name=str(memory_adapter.name),
            memory_metadata=dict(memory_recall.metadata),
        )
        score_fn = getattr(benchmark_adapter, "score_prediction", None)
        if callable(score_fn):
            score_value = _call_with_supported_kwargs(score_fn, prediction=prediction)
            if isinstance(score_value, dict):
                prediction.scores = dict(score_value)
        predictions.append(prediction)

    aggregate_scores: dict[str, Any] = {}
    aggregate_fn = getattr(benchmark_adapter, "aggregate_scores", None)
    if callable(aggregate_fn):
        aggregate_value = _call_with_supported_kwargs(aggregate_fn, predictions=predictions)
        if isinstance(aggregate_value, dict):
            aggregate_scores = dict(aggregate_value)

    return BenchmarkRunResult(
        benchmark_name=str(getattr(benchmark_adapter, "name", "")),
        memory_adapter_name=str(getattr(memory_adapter, "name", "")),
        predictions=predictions,
        aggregate_scores=aggregate_scores,
    )
