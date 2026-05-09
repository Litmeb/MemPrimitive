"""Benchmark runners and answer-generation helpers."""

from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from memprimitive.utils._runtime import Runtime
from memprimitive.utils._template import render_prompt_template

from ._benchmark_tool_errors import (
    BenchmarkToolErrorLog,
    ModelBehaviorError,
    append_benchmark_tool_error,
    push_benchmark_tool_error_log,
)
from ._types import (
    AnswerRunner,
    BenchmarkPrediction,
    BenchmarkRunResult,
    BenchmarkSample,
    MemoryRecall,
)
from .prompts import ANSWER_PROMPT, MEMMACHINE_ANSWER_PROMPT


_LOCOMO_ANSWER_QUESTION_SPLIT = "\n\n    Question:"
_SINGLE_RECALL_MEMORY_SPLIT = "\n\nRetrieved memory:\n"


def _locomo_cap_user_preserving_question(runtime: Runtime, user: str, user_token_budget: int) -> str:
    """Shrink a rendered LoCoMo answer user prompt without dropping the final Question/Answer tail."""

    if user_token_budget <= 0:
        return ""
    if runtime.count_tokens(user) <= user_token_budget:
        return user
    if _LOCOMO_ANSWER_QUESTION_SPLIT not in user:
        return runtime.truncate_text_to_token_limit(user, user_token_budget)
    prefix, suffix = user.rsplit(_LOCOMO_ANSWER_QUESTION_SPLIT, 1)
    suffix = _LOCOMO_ANSWER_QUESTION_SPLIT + suffix
    suffix_tokens = runtime.count_tokens(suffix)
    prefix_budget = user_token_budget - suffix_tokens
    if prefix_budget <= 0:
        return runtime.truncate_text_to_token_limit(suffix, user_token_budget)
    prefix_capped = runtime.truncate_text_to_token_limit(prefix, prefix_budget)
    return prefix_capped + suffix


def _locomo_cap_single_recall_user(runtime: Runtime, user: str, user_token_budget: int) -> str:
    """For minimal LoCoMo recall: keep metadata + user request, trim retrieved memory from the tail."""

    if user_token_budget <= 0:
        return ""
    if runtime.count_tokens(user) <= user_token_budget:
        return user
    if _SINGLE_RECALL_MEMORY_SPLIT not in user:
        return runtime.truncate_text_to_token_limit(user, user_token_budget)
    meta, _, mem_rest = user.partition(_SINGLE_RECALL_MEMORY_SPLIT)
    mem_block = _SINGLE_RECALL_MEMORY_SPLIT + mem_rest
    meta_tokens = runtime.count_tokens(meta)
    if meta_tokens >= user_token_budget:
        return runtime.truncate_text_to_token_limit(user, user_token_budget)
    mem_cap = runtime.truncate_text_to_token_limit(mem_block, user_token_budget - meta_tokens)
    return meta + mem_cap


def _answer_chat_with_input_cap(
    runtime: Runtime,
    *,
    system: str,
    user: str,
    temperature: float,
    max_input_tokens: int | None,
    locomo_user_cap: Callable[[Runtime, str, int], str] | None = None,
) -> str:
    if max_input_tokens is None:
        return runtime.text(system=system, user=user, temperature=temperature, max_input_tokens=None)
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive.")
    sys_tokens = runtime.count_tokens(system)
    user_tokens = runtime.count_tokens(user)
    if sys_tokens + user_tokens <= max_input_tokens:
        return runtime.text(system=system, user=user, temperature=temperature, max_input_tokens=None)
    if sys_tokens >= max_input_tokens:
        system = runtime.truncate_text_to_token_limit(system, max_input_tokens)
        return runtime.text(system=system, user="", temperature=temperature, max_input_tokens=None)
    user_budget = max_input_tokens - sys_tokens
    if locomo_user_cap is not None:
        user = locomo_user_cap(runtime, user, user_budget)
    else:
        user = runtime.truncate_text_to_token_limit(user, user_budget)
    return runtime.text(system=system, user=user, temperature=temperature, max_input_tokens=None)


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
        max_input_tokens: int | None = None,
    ) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.max_input_tokens = max_input_tokens
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
        locomo_cap = (
            _locomo_cap_single_recall_user
            if str(sample.benchmark_name).strip().casefold() == "locomo"
            else None
        )
        return _answer_chat_with_input_cap(
            self.runtime,
            system=self.system_prompt,
            user=user_prompt,
            temperature=0.0,
            max_input_tokens=self.max_input_tokens,
            locomo_user_cap=locomo_cap,
        )


class Mem0LoCoMoAnswerRunner:
    """Render the upstream Mem0 LoCoMo answer prompt and answer with the real runtime."""

    name = "mem0_locomo_answer"

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        system_prompt: str | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.system_prompt = system_prompt or ""
        self.max_input_tokens = max_input_tokens

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
        return _answer_chat_with_input_cap(
            self.runtime,
            system=self.system_prompt,
            user=rendered_user_prompt,
            temperature=0.0,
            max_input_tokens=self.max_input_tokens,
            locomo_user_cap=_locomo_cap_user_preserving_question,
        )


class MemMachineLoCoMoAnswerRunner:
    """Render the MemMachine LoCoMo answer prompt against one conversation memory block."""

    name = "memmachine_locomo_answer"

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        system_prompt: str | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.system_prompt = system_prompt or ""
        self.max_input_tokens = max_input_tokens

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
        return _answer_chat_with_input_cap(
            self.runtime,
            system=self.system_prompt,
            user=rendered_user_prompt,
            temperature=0.0,
            max_input_tokens=self.max_input_tokens,
            locomo_user_cap=_locomo_cap_user_preserving_question,
        )


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


def write_benchmark_tool_errors_jsonl(events: list[dict[str, Any]], output_path: Path) -> tuple[Path | None, int]:
    """Write per-row tool mishandling ledger next to predictions. Returns (path or None, rows written)."""

    if not events:
        return None, 0
    log_path = output_path.with_name(output_path.stem + "_tool_errors.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with log_path.open("w", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            written += 1
    return log_path, written


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


def _group_samples_by_memory_session(
    memory_adapter: Any,
    samples: list[BenchmarkSample],
) -> list[tuple[str | None, list[tuple[int, BenchmarkSample]]]]:
    groups: list[tuple[str | None, list[tuple[int, BenchmarkSample]]]] = []
    keyed_groups: dict[str, list[tuple[int, BenchmarkSample]]] = {}
    for index, sample in enumerate(samples, start=1):
        session_key = _memory_session_key(memory_adapter, sample)
        if session_key is None:
            groups.append((None, [(index, sample)]))
            continue
        group = keyed_groups.get(session_key)
        if group is None:
            group = []
            keyed_groups[session_key] = group
            groups.append((session_key, group))
        group.append((index, sample))
    return groups


def _build_prediction(
    *,
    runner: AnswerRunner | Any,
    memory_adapter: Any,
    sample: BenchmarkSample,
    session: Any,
) -> BenchmarkPrediction:
    adapter_name = str(getattr(memory_adapter, "name", "")).strip() or "memory"
    try:
        memory_recall = session.recall(sample.query, sample=sample)
    except ModelBehaviorError as exc:
        append_benchmark_tool_error(
            {
                "phase": "recall",
                "memory_adapter": adapter_name,
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark_name,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
        memory_recall = MemoryRecall(
            text="",
            source_ids=[],
            metadata={
                "benchmark_skipped_tool_error": True,
                "benchmark_skip_reason": str(exc),
            },
        )
    try:
        predicted_answer = _answer_with_runner(runner, sample=sample, memory_recall=memory_recall)
    except ModelBehaviorError as exc:
        append_benchmark_tool_error(
            {
                "phase": "answer",
                "memory_adapter": adapter_name,
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark_name,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
        predicted_answer = ""
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


def _run_benchmark_sample_group(
    *,
    benchmark_adapter: Any,
    runner: AnswerRunner | Any,
    memory_adapter: Any,
    session_key: str | None,
    indexed_samples: list[tuple[int, BenchmarkSample]],
    group_index: int,
    group_total: int,
    sample_total: int,
    progress_callback: Callable[..., None] | None,
) -> list[tuple[int, BenchmarkPrediction]]:
    first_index, first_sample = indexed_samples[0]
    if progress_callback is not None:
        _call_with_supported_kwargs(
            progress_callback,
            phase="memory_load_start",
            index=first_index,
            total=sample_total,
            sample=first_sample,
            session_key=session_key,
            group_index=group_index,
            group_total=group_total,
            group_size=len(indexed_samples),
        )
    session = memory_adapter.create_session()
    _call_with_supported_kwargs(
        session.load_case,
        first_sample,
        progress_callback=progress_callback,
        sample_index=first_index,
        total_samples=sample_total,
        session_key=session_key,
        group_index=group_index,
        group_total=group_total,
        group_size=len(indexed_samples),
    )
    if progress_callback is not None:
        _call_with_supported_kwargs(
            progress_callback,
            phase="memory_loaded",
            index=first_index,
            total=sample_total,
            sample=first_sample,
            session_key=session_key,
            group_index=group_index,
            group_total=group_total,
            group_size=len(indexed_samples),
        )

    predictions: list[tuple[int, BenchmarkPrediction]] = []
    for group_sample_index, (index, sample) in enumerate(indexed_samples, start=1):
        if group_sample_index > 1 and progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_reuse",
                index=index,
                total=sample_total,
                sample=sample,
                session_key=session_key,
                group_index=group_index,
                group_total=group_total,
                group_size=len(indexed_samples),
                group_sample_index=group_sample_index,
            )
        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="start",
                index=index,
                total=sample_total,
                sample=sample,
                session_key=session_key,
                group_index=group_index,
                group_total=group_total,
                group_size=len(indexed_samples),
                group_sample_index=group_sample_index,
            )
        prediction = _build_prediction(
            runner=runner,
            memory_adapter=memory_adapter,
            sample=sample,
            session=session,
        )
        _score_prediction(benchmark_adapter, prediction)
        predictions.append((index, prediction))
        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="done",
                index=index,
                total=sample_total,
                sample=sample,
                prediction=prediction,
                session_key=session_key,
                group_index=group_index,
                group_total=group_total,
                group_size=len(indexed_samples),
                group_sample_index=group_sample_index,
            )
    return predictions


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
    tool_error_log = BenchmarkToolErrorLog()
    prior_tool_error_log = push_benchmark_tool_error_log(tool_error_log)
    try:
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
        sample_groups = _group_samples_by_memory_session(memory_adapter, samples)
        if max_workers == 1:
            predictions_by_index: dict[int, BenchmarkPrediction] = {}
            for group_index, (session_key, indexed_samples) in enumerate(sample_groups, start=1):
                for index, prediction in _run_benchmark_sample_group(
                    benchmark_adapter=benchmark_adapter,
                    runner=runner,
                    memory_adapter=memory_adapter,
                    session_key=session_key,
                    indexed_samples=indexed_samples,
                    group_index=group_index,
                    group_total=len(sample_groups),
                    sample_total=total,
                    progress_callback=progress_callback,
                ):
                    predictions_by_index[index] = prediction
            predictions = [predictions_by_index[index] for index in sorted(predictions_by_index)]
        else:
            predictions_by_index: dict[int, BenchmarkPrediction] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for group_index, (session_key, indexed_samples) in enumerate(sample_groups, start=1):
                    future = executor.submit(
                        _run_benchmark_sample_group,
                        benchmark_adapter=benchmark_adapter,
                        runner=runner,
                        memory_adapter=memory_adapter,
                        session_key=session_key,
                        indexed_samples=indexed_samples,
                        group_index=group_index,
                        group_total=len(sample_groups),
                        sample_total=total,
                        progress_callback=progress_callback,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    for index, prediction in future.result():
                        predictions_by_index[index] = prediction
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
            tool_error_events=tool_error_log.snapshot(),
        )
    finally:
        push_benchmark_tool_error_log(prior_tool_error_log)
