"""Memory-side adapters for benchmark evaluation."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..config import load_pipeline_from_yaml
from ..core import Observation, Query, Readout
from ..pipeline import FreeMemoryPipeline, MemoryPipeline
from ..example.classics import amem_memory, mem0_memory, memmachine_memory
from ._benchmark_tool_errors import ModelBehaviorError, append_benchmark_tool_error
from ._types import (
    BenchmarkSample,
    ConversationTurn,
    MemoryIngestEvent,
    MemoryRecall,
    MemorySystemBinding,
    RecallContext,
    _locomo_caption_suffix,
    default_turn_to_observation,
)


def _log_benchmark_ingest_model_behavior_error(
    sample: BenchmarkSample,
    *,
    phase: str,
    exc: ModelBehaviorError,
    turn_id: str | None = None,
    speaker_binding: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append to the benchmark tool ledger and return a copy for embedding in session metadata."""

    row = {
        "phase": phase,
        "sample_id": str(sample.sample_id),
        "benchmark": str(sample.benchmark_name),
        "error": str(exc),
        "error_type": type(exc).__name__,
        **({} if turn_id is None or not str(turn_id).strip() else {"turn_id": str(turn_id).strip()}),
        **(
            {}
            if speaker_binding is None or not str(speaker_binding).strip()
            else {"speaker_binding": str(speaker_binding).strip()}
        ),
        **extra,
    }
    append_benchmark_tool_error(dict(row))
    return row


def _safe_binding_ingest_event(
    binding: Any,
    system: Any,
    event: MemoryIngestEvent,
    sample: BenchmarkSample,
    *,
    phase: str,
    speaker_binding: str | None = None,
) -> dict[str, Any] | None:
    try:
        binding.ingest_event(system, event)
        return None
    except ModelBehaviorError as exc:
        return _log_benchmark_ingest_model_behavior_error(
            sample,
            phase=phase,
            exc=exc,
            turn_id=str(event.turn_id).strip() if str(event.turn_id).strip() else None,
            speaker_binding=speaker_binding,
        )


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    supported_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(*args, **supported_kwargs)


def _coerce_memory_recall(value: Any, *, base_metadata: dict[str, Any] | None = None) -> MemoryRecall:
    metadata = dict(base_metadata or {})
    if isinstance(value, MemoryRecall):
        return MemoryRecall(
            text=str(value.text),
            source_ids=list(value.source_ids),
            metadata={**metadata, **dict(value.metadata)},
        )
    if isinstance(value, Readout):
        return MemoryRecall(
            text=str(value.text),
            source_ids=list(value.source_ids),
            metadata={**metadata, "readout_metadata": dict(value.metadata)},
        )
    if isinstance(value, str):
        return MemoryRecall(text=value, source_ids=[], metadata=metadata)

    text = str(getattr(value, "text", "")).strip()
    source_ids = list(getattr(value, "source_ids", []))
    extra_metadata = getattr(value, "metadata", {})
    if text:
        merged_metadata = dict(metadata)
        if isinstance(extra_metadata, dict):
            merged_metadata.update(extra_metadata)
        return MemoryRecall(text=text, source_ids=source_ids, metadata=merged_metadata)
    raise TypeError(f"Unsupported recall result type {type(value).__name__}.")


def _sample_observations(
    sample: BenchmarkSample,
    *,
    turn_to_observation: Callable[[ConversationTurn], Observation],
) -> tuple[list[Observation], dict[str, Any]]:
    if sample.history_turns:
        observations = [turn_to_observation(turn) for turn in sample.history_turns]
        return observations, {
            "load_source": "history_turns",
            "loaded_turn_count": len(sample.history_turns),
            "loaded_observation_count": len(observations),
        }
    observations = list(sample.history_observations)
    return observations, {
        "load_source": "history_observations",
        "loaded_turn_count": len(sample.history_turns),
        "loaded_observation_count": len(observations),
    }


class _PipelineMemorySession:
    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], MemoryPipeline | FreeMemoryPipeline],
        turn_to_observation: Callable[[ConversationTurn], Observation],
    ) -> None:
        self.pipeline = pipeline_factory()
        self.turn_to_observation = turn_to_observation
        self.load_metadata: dict[str, Any] = {}

    def load_case(self, sample: BenchmarkSample) -> None:
        observations, metadata = _sample_observations(sample, turn_to_observation=self.turn_to_observation)
        skipped: list[dict[str, Any]] = []
        for observation in observations:
            try:
                self.pipeline.ingest(observation)
            except ModelBehaviorError as exc:
                turn_id = str(observation.metadata.get("turn_id", "")).strip() or None
                skipped.append(
                    _log_benchmark_ingest_model_behavior_error(
                        sample,
                        phase="ingest_pipeline",
                        exc=exc,
                        turn_id=turn_id,
                    )
                )
        merged_meta = dict(metadata)
        if skipped:
            merged_meta["benchmark_skipped_ingests"] = skipped
        self.load_metadata = merged_meta

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        del sample
        readout = self.pipeline.recall(query)
        return _coerce_memory_recall(
            readout,
            base_metadata={
                **self.load_metadata,
                "session_kind": "pipeline",
            },
        )


class PipelineMemoryAdapter:
    """Wrap a MemoryPipeline or FreeMemoryPipeline factory as a benchmark adapter."""

    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], MemoryPipeline | FreeMemoryPipeline],
        name: str = "pipeline",
        turn_to_observation: Callable[[ConversationTurn], Observation] = default_turn_to_observation,
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.name = str(name).strip() or "pipeline"
        self.turn_to_observation = turn_to_observation

    def create_session(self) -> _PipelineMemorySession:
        return _PipelineMemorySession(
            pipeline_factory=self.pipeline_factory,
            turn_to_observation=self.turn_to_observation,
        )


def create_yaml_pipeline_memory_adapter(
    config_path: str,
    *,
    root: str | None = None,
    name: str | None = None,
    turn_to_observation: Callable[[ConversationTurn], Observation] = default_turn_to_observation,
) -> PipelineMemoryAdapter:
    """Create a fresh pipeline-backed adapter from one YAML config file."""

    adapter_name = str(name).strip() if name is not None else f"yaml:{config_path}"
    return PipelineMemoryAdapter(
        name=adapter_name,
        turn_to_observation=turn_to_observation,
        pipeline_factory=lambda: load_pipeline_from_yaml(config_path, root=root),
    )


class _FunctionMemorySession:
    def __init__(
        self,
        *,
        system_factory: Callable[[], Any],
        load_case: Callable[[Any, BenchmarkSample], Any],
        recall: Callable[..., Any],
    ) -> None:
        self.system = system_factory()
        self._load_case = load_case
        self._recall = recall
        self.load_metadata: dict[str, Any] = {}

    def load_case(self, sample: BenchmarkSample) -> None:
        load_result = self._load_case(self.system, sample)
        self.load_metadata = {
            "loaded_turn_count": len(sample.history_turns),
            "loaded_observation_count": len(sample.history_observations),
            "session_kind": "function",
        }
        if isinstance(load_result, dict):
            self.load_metadata.update(load_result)

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        recall_result = _call_with_supported_kwargs(self._recall, self.system, query, sample=sample)
        return _coerce_memory_recall(recall_result, base_metadata=self.load_metadata)


class FunctionMemoryAdapter:
    """Wrap a system-factory plus helper functions as a benchmark memory adapter."""

    def __init__(
        self,
        *,
        system_factory: Callable[[], Any],
        load_case: Callable[[Any, BenchmarkSample], Any],
        recall: Callable[..., Any],
        name: str = "function",
    ) -> None:
        self.system_factory = system_factory
        self.load_case_fn = load_case
        self.recall_fn = recall
        self.name = str(name).strip() or "function"

    def create_session(self) -> _FunctionMemorySession:
        return _FunctionMemorySession(
            system_factory=self.system_factory,
            load_case=self.load_case_fn,
            recall=self.recall_fn,
        )


def _iter_pairwise_turns(turns: list[ConversationTurn]) -> list[tuple[ConversationTurn, ConversationTurn | None]]:
    pairs: list[tuple[ConversationTurn, ConversationTurn | None]] = []
    pending: ConversationTurn | None = None
    pending_session = ""
    for turn in turns:
        if pending is not None and str(turn.session_id) != pending_session:
            pairs.append((pending, None))
            pending = None
            pending_session = ""
        if pending is None:
            pending = turn
            pending_session = str(turn.session_id)
            continue
        pairs.append((pending, turn))
        pending = None
        pending_session = ""
    if pending is not None:
        pairs.append((pending, None))
    return pairs


class PairwiseDialogueMemoryAdapter(FunctionMemoryAdapter):
    """Function-style adapter for systems that ingest one dialogue pair at a time."""

    def __init__(
        self,
        *,
        system_factory: Callable[[], Any],
        ingest_pair: Callable[..., Any],
        recall: Callable[..., Any],
        name: str = "pairwise_dialogue",
    ) -> None:
        self.ingest_pair = ingest_pair
        super().__init__(
            name=name,
            system_factory=system_factory,
            load_case=self._load_pairwise_case,
            recall=recall,
        )

    def _load_pairwise_case(self, system: Any, sample: BenchmarkSample) -> dict[str, Any]:
        if not sample.history_turns:
            raise ValueError("PairwiseDialogueMemoryAdapter requires sample.history_turns.")
        pair_count = 0
        skipped_pairs: list[dict[str, Any]] = []
        for first_turn, second_turn in _iter_pairwise_turns(sample.history_turns):
            pair_count += 1
            try:
                _call_with_supported_kwargs(
                    self.ingest_pair,
                    system,
                    user_text=str(first_turn.text).strip(),
                    assistant_text=str(second_turn.text).strip() if second_turn is not None else "",
                    session_id=str(first_turn.session_id).strip(),
                    turn_id=str(second_turn.turn_id).strip() if second_turn is not None else str(first_turn.turn_id).strip(),
                    timestamp=str(first_turn.session_timestamp).strip(),
                )
            except ModelBehaviorError as exc:
                skipped_pairs.append(
                    _log_benchmark_ingest_model_behavior_error(
                        sample,
                        phase="ingest_pairwise",
                        exc=exc,
                        turn_id=str(second_turn.turn_id).strip() if second_turn is not None else str(first_turn.turn_id).strip(),
                    )
                )
        meta = {
            "loaded_pair_count": pair_count,
            "load_source": "history_turn_pairs",
        }
        if skipped_pairs:
            meta["benchmark_skipped_ingests"] = skipped_pairs
        return meta


def _locomo_turn_text(turn: ConversationTurn | None) -> str:
    if turn is None:
        return ""
    speaker_label = str(turn.speaker).strip() or str(turn.role).strip() or "speaker"
    text = str(turn.text).strip()
    caption_suffix = _locomo_caption_suffix(turn.metadata)
    if text:
        return f"{speaker_label}: {text}{caption_suffix}"
    if caption_suffix:
        return f"{speaker_label}:{caption_suffix}"
    return ""


def _locomo_speaker_user_id(sample: BenchmarkSample, *, speaker_key: str, fallback_name: str) -> str:
    index = str(sample.metadata.get("locomo_user_index", "")).strip() or str(sample.sample_id).strip()
    speaker_name = str(sample.metadata.get(speaker_key, "")).strip() or fallback_name
    return f"{speaker_name}_{index}"


def _locomo_speaker_user_id_from_metadata(
    metadata: dict[str, Any],
    *,
    sample_id: str,
    speaker_key: str,
    fallback_name: str,
) -> str:
    index = str(metadata.get("locomo_user_index", "")).strip() or str(sample_id).strip()
    speaker_name = str(metadata.get(speaker_key, "")).strip() or fallback_name
    return f"{speaker_name}_{index}"


def _locomo_pair_for_speaker(
    first_turn: ConversationTurn,
    second_turn: ConversationTurn | None,
    *,
    speaker_name: str,
) -> tuple[str, str]:
    target = str(speaker_name).strip().casefold()
    first_speaker = str(first_turn.speaker).strip().casefold()
    second_speaker = str(second_turn.speaker).strip().casefold() if second_turn is not None else ""

    if first_speaker == target:
        user_turn = first_turn
        assistant_turn = second_turn if second_speaker and second_speaker != target else None
    elif second_turn is not None and second_speaker == target:
        user_turn = second_turn
        assistant_turn = first_turn if first_speaker != target else None
    else:
        return "", ""

    return _locomo_turn_text(user_turn), _locomo_turn_text(assistant_turn)


def _locomo_event_for_speaker(
    sample: BenchmarkSample,
    first_turn: ConversationTurn,
    second_turn: ConversationTurn | None,
    *,
    speaker_key: str,
    fallback_name: str,
) -> MemoryIngestEvent | None:
    speaker_name = str(sample.metadata.get(speaker_key, "")).strip() or fallback_name
    user_text, context_text = _locomo_pair_for_speaker(first_turn, second_turn, speaker_name=speaker_name)
    if not user_text:
        return None
    turn_id = str(second_turn.turn_id).strip() if second_turn is not None else str(first_turn.turn_id).strip()
    timestamp = str(first_turn.session_timestamp).strip()
    return MemoryIngestEvent(
        text=user_text,
        context_text=context_text,
        session_id=str(first_turn.session_id).strip(),
        turn_id=turn_id,
        user_id=_locomo_speaker_user_id(sample, speaker_key=speaker_key, fallback_name=fallback_name),
        speaker=speaker_name,
        role="user",
        timestamp=timestamp or None,
        metadata={
            "benchmark": "locomo",
            "speaker_key": speaker_key,
            "first_turn_id": str(first_turn.turn_id).strip(),
            "second_turn_id": str(second_turn.turn_id).strip() if second_turn is not None else "",
        },
    )


def _locomo_recall_context(
    sample: BenchmarkSample | None,
    *,
    speaker_key: str,
    fallback_name: str,
    speaker_index: int,
) -> RecallContext:
    if sample is None:
        return RecallContext(
            sample_id="",
            user_id=fallback_name,
            speaker=fallback_name,
            speaker_index=speaker_index,
        )
    metadata = dict(sample.metadata)
    speaker = str(metadata.get(speaker_key, "")).strip() or fallback_name
    return RecallContext(
        sample_id=str(sample.sample_id).strip(),
        user_id=_locomo_speaker_user_id_from_metadata(
            metadata,
            sample_id=str(sample.sample_id).strip(),
            speaker_key=speaker_key,
            fallback_name=fallback_name,
        ),
        speaker=speaker,
        speaker_index=speaker_index,
        metadata=metadata,
    )


class _DualSpeakerLoCoMoMemorySession:
    def __init__(self, *, binding_factory: Callable[[], MemorySystemBinding], speaker_workers: int) -> None:
        if speaker_workers <= 0:
            raise ValueError("speaker_workers must be positive.")
        self.speaker_1_binding = binding_factory()
        self.speaker_2_binding = binding_factory()
        self.speaker_1_system = self.speaker_1_binding.build_system()
        self.speaker_2_system = self.speaker_2_binding.build_system()
        self.speaker_workers = speaker_workers
        self.load_metadata: dict[str, Any] = {}

    def load_case(self, sample: BenchmarkSample, *, progress_callback: Callable[..., None] | None = None) -> None:
        if not sample.history_turns:
            raise ValueError("DualSpeakerLoCoMoMemoryAdapter requires sample.history_turns.")

        total_turns = len(sample.history_turns)
        loaded_turns = 0
        pair_count = 0
        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_init",
                sample=sample,
                total_turns=total_turns,
            )

        executor = ThreadPoolExecutor(max_workers=min(2, self.speaker_workers)) if self.speaker_workers > 1 else None
        skipped_ingests: list[dict[str, Any]] = []
        try:
            for first_turn, second_turn in _iter_pairwise_turns(sample.history_turns):
                pair_count += 1
                turn_id = str(second_turn.turn_id).strip() if second_turn is not None else str(first_turn.turn_id).strip()
                turns_in_pair = 2 if second_turn is not None else 1
                ingest_calls = [
                    (
                        self.speaker_1_binding,
                        self.speaker_1_system,
                        _locomo_event_for_speaker(
                            sample,
                            first_turn,
                            second_turn,
                            speaker_key="speaker_a",
                            fallback_name="speaker_1",
                        ),
                    ),
                    (
                        self.speaker_2_binding,
                        self.speaker_2_system,
                        _locomo_event_for_speaker(
                            sample,
                            first_turn,
                            second_turn,
                            speaker_key="speaker_b",
                            fallback_name="speaker_2",
                        ),
                    ),
                ]
                ingest_calls = [call for call in ingest_calls if call[2] is not None]

                if executor is None or len(ingest_calls) <= 1:
                    for binding, system, event in ingest_calls:
                        assert event is not None
                        label = "speaker_1" if binding is self.speaker_1_binding else "speaker_2"
                        row = _safe_binding_ingest_event(
                            binding,
                            system,
                            event,
                            sample,
                            phase="ingest_dual_speaker",
                            speaker_binding=label,
                        )
                        if row is not None:
                            skipped_ingests.append(row)
                else:
                    futures = [
                        executor.submit(
                            _safe_binding_ingest_event,
                            binding,
                            system,
                            event,
                            sample,
                            phase="ingest_dual_speaker",
                            speaker_binding=("speaker_1" if binding is self.speaker_1_binding else "speaker_2"),
                        )
                        for binding, system, event in ingest_calls
                        if event is not None
                    ]
                    for future in futures:
                        row = future.result()
                        if row is not None:
                            skipped_ingests.append(row)

                loaded_turns += turns_in_pair
                if progress_callback is not None:
                    _call_with_supported_kwargs(
                        progress_callback,
                        phase="memory_turn_done",
                        sample=sample,
                        turn_index=loaded_turns,
                        total_turns=total_turns,
                        turn_id=turn_id,
                        turn_increment=turns_in_pair,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_finish",
                sample=sample,
                total_turns=total_turns,
                loaded_turns=loaded_turns,
            )

        meta = {
            "loaded_pair_count": pair_count,
            "loaded_turn_count": loaded_turns,
            "load_source": "history_turn_pairs",
            "speaker_workers": self.speaker_workers,
        }
        if skipped_ingests:
            meta["benchmark_skipped_ingests"] = skipped_ingests
        self.load_metadata = meta

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        context_1 = _locomo_recall_context(sample, speaker_key="speaker_a", fallback_name="speaker_1", speaker_index=1)
        context_2 = _locomo_recall_context(sample, speaker_key="speaker_b", fallback_name="speaker_2", speaker_index=2)

        if self.speaker_workers > 1:
            with ThreadPoolExecutor(max_workers=min(2, self.speaker_workers)) as executor:
                speaker_1_future = executor.submit(
                    self.speaker_1_binding.recall,
                    self.speaker_1_system,
                    query,
                    context=context_1,
                )
                speaker_2_future = executor.submit(
                    self.speaker_2_binding.recall,
                    self.speaker_2_system,
                    query,
                    context=context_2,
                )
                speaker_1_recall = _coerce_memory_recall(speaker_1_future.result())
                speaker_2_recall = _coerce_memory_recall(speaker_2_future.result())
        else:
            speaker_1_recall = _coerce_memory_recall(
                self.speaker_1_binding.recall(self.speaker_1_system, query, context=context_1)
            )
            speaker_2_recall = _coerce_memory_recall(
                self.speaker_2_binding.recall(self.speaker_2_system, query, context=context_2)
            )

        speaker_1_memories = speaker_1_recall.text.strip()
        speaker_2_memories = speaker_2_recall.text.strip()
        recall_text = "\n\n".join(
            part
            for part in (
                f"{context_1.user_id or 'speaker_1'}\n{speaker_1_memories}" if speaker_1_memories else "",
                f"{context_2.user_id or 'speaker_2'}\n{speaker_2_memories}" if speaker_2_memories else "",
            )
            if part
        )
        binding_name = str(getattr(self.speaker_1_binding, "name", "")).strip()
        metadata = {
            **self.load_metadata,
            "recall_helper": f"{binding_name}.recall" if binding_name else "memory_binding.recall",
            "speaker_1_user_id": context_1.user_id,
            "speaker_2_user_id": context_2.user_id,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": _count_nonempty_lines(speaker_1_memories),
            "num_speaker_2_memories": _count_nonempty_lines(speaker_2_memories),
            "speaker_1_name": context_1.speaker,
            "speaker_2_name": context_2.speaker,
        }
        return MemoryRecall(
            text=recall_text,
            source_ids=list(speaker_1_recall.source_ids) + list(speaker_2_recall.source_ids),
            metadata=metadata,
        )


class DualSpeakerLoCoMoMemoryAdapter:
    """Benchmark adapter for any binding that implements the memory-system interface."""

    def __init__(
        self,
        *,
        binding_factory: Callable[[], MemorySystemBinding],
        name: str,
        speaker_workers: int = 1,
    ) -> None:
        if speaker_workers <= 0:
            raise ValueError("speaker_workers must be positive.")
        self.binding_factory = binding_factory
        self.name = str(name).strip() or "memory"
        self.speaker_workers = speaker_workers

    def create_session(self) -> _DualSpeakerLoCoMoMemorySession:
        return _DualSpeakerLoCoMoMemorySession(
            binding_factory=self.binding_factory,
            speaker_workers=self.speaker_workers,
        )

    def session_key(self, *, sample: BenchmarkSample) -> str:
        sample_metadata = dict(sample.metadata)
        locomo_sample_id = str(sample_metadata.get("locomo_sample_id", "")).strip()
        if locomo_sample_id:
            return locomo_sample_id
        user_index = str(sample_metadata.get("locomo_user_index", "")).strip()
        if user_index:
            return f"locomo-user-{user_index}"
        return str(sample.sample_id).split("-qa-", 1)[0]


def _locomo_conversation_user_id(sample: BenchmarkSample) -> str:
    metadata = dict(sample.metadata)
    locomo_sample_id = str(metadata.get("locomo_sample_id", "")).strip()
    if locomo_sample_id:
        return f"conversation:{locomo_sample_id}"
    user_index = str(metadata.get("locomo_user_index", "")).strip()
    if user_index:
        return f"conversation:locomo-user-{user_index}"
    return f"conversation:{str(sample.sample_id).split('-qa-', 1)[0]}"


def _locomo_message_event(sample: BenchmarkSample, turn: ConversationTurn) -> MemoryIngestEvent:
    metadata = dict(sample.metadata)
    speaker = str(turn.speaker).strip() or str(turn.role).strip() or "speaker"
    timestamp = str(turn.session_timestamp).strip()
    turn_metadata = dict(turn.metadata)
    blip_caption = turn_metadata.get("blip_caption")
    return MemoryIngestEvent(
        text=str(turn.text).strip(),
        context_text="",
        session_id=str(turn.session_id).strip(),
        turn_id=str(turn.turn_id).strip(),
        user_id=_locomo_conversation_user_id(sample),
        speaker=speaker,
        role=str(turn.role).strip() or speaker,
        timestamp=timestamp or None,
        metadata={
            "benchmark": "locomo",
            "locomo_sample_id": metadata.get("locomo_sample_id"),
            "locomo_user_index": metadata.get("locomo_user_index"),
            "speaker_a": metadata.get("speaker_a"),
            "speaker_b": metadata.get("speaker_b"),
            "source_timestamp": timestamp,
            "source_speaker": speaker,
            "blip_caption": blip_caption,
            "turn_index": turn_metadata.get("turn_index"),
            "dialogue_id": turn_metadata.get("dialogue_id"),
        },
    )


class _SharedConversationLoCoMoMemorySession:
    def __init__(self, *, binding_factory: Callable[[], MemorySystemBinding]) -> None:
        self.binding = binding_factory()
        self.system = self.binding.build_system()
        self.load_metadata: dict[str, Any] = {}

    def load_case(self, sample: BenchmarkSample, *, progress_callback: Callable[..., None] | None = None) -> None:
        if not sample.history_turns:
            raise ValueError("SharedConversationLoCoMoMemoryAdapter requires sample.history_turns.")

        total_turns = len(sample.history_turns)
        loaded_turns = 0
        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_init",
                sample=sample,
                total_turns=total_turns,
            )

        skipped_ingests: list[dict[str, Any]] = []
        for turn in sample.history_turns:
            event = _locomo_message_event(sample, turn)
            row = _safe_binding_ingest_event(
                self.binding,
                self.system,
                event,
                sample,
                phase="ingest_shared_conversation",
            )
            if row is not None:
                skipped_ingests.append(row)
            loaded_turns += 1
            if progress_callback is not None:
                _call_with_supported_kwargs(
                    progress_callback,
                    phase="memory_turn_done",
                    sample=sample,
                    turn_index=loaded_turns,
                    total_turns=total_turns,
                    turn_id=str(turn.turn_id).strip(),
                    turn_increment=1,
                )

        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_finish",
                sample=sample,
                total_turns=total_turns,
                loaded_turns=loaded_turns,
            )

        self.load_metadata = {
            "loaded_turn_count": loaded_turns,
            "loaded_message_count": loaded_turns,
            "load_source": "history_turn_messages",
            "conversation_user_id": _locomo_conversation_user_id(sample),
            **({"benchmark_skipped_ingests": skipped_ingests} if skipped_ingests else {}),
        }

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        metadata = dict(sample.metadata) if sample is not None else {}
        context = RecallContext(
            sample_id=str(sample.sample_id).strip() if sample is not None else "",
            user_id=_locomo_conversation_user_id(sample) if sample is not None else "",
            speaker="conversation",
            metadata=metadata,
        )
        recall = _coerce_memory_recall(self.binding.recall(self.system, query, context=context))
        binding_name = str(getattr(self.binding, "name", "")).strip()
        return MemoryRecall(
            text=recall.text,
            source_ids=list(recall.source_ids),
            metadata={
                **self.load_metadata,
                **dict(recall.metadata),
                "recall_helper": f"{binding_name}.recall" if binding_name else "memory_binding.recall",
                "speaker_1_name": str(metadata.get("speaker_a", "")).strip(),
                "speaker_2_name": str(metadata.get("speaker_b", "")).strip(),
                "conversation_memories": recall.text,
                "num_conversation_memory_lines": _count_nonempty_lines(recall.text),
            },
        )


class SharedConversationLoCoMoMemoryAdapter:
    """LoCoMo adapter for official shared-conversation memory systems."""

    def __init__(self, *, binding_factory: Callable[[], MemorySystemBinding], name: str) -> None:
        self.binding_factory = binding_factory
        self.name = str(name).strip() or "memory"

    def create_session(self) -> _SharedConversationLoCoMoMemorySession:
        return _SharedConversationLoCoMoMemorySession(binding_factory=self.binding_factory)

    def session_key(self, *, sample: BenchmarkSample) -> str:
        sample_metadata = dict(sample.metadata)
        locomo_sample_id = str(sample_metadata.get("locomo_sample_id", "")).strip()
        if locomo_sample_id:
            return locomo_sample_id
        user_index = str(sample_metadata.get("locomo_user_index", "")).strip()
        if user_index:
            return f"locomo-user-{user_index}"
        return str(sample.sample_id).split("-qa-", 1)[0]


def _generic_conversation_user_id(sample: BenchmarkSample) -> str:
    benchmark_name = str(sample.benchmark_name).strip() or "benchmark"
    sample_id = str(sample.sample_id).strip() or "sample"
    return f"{benchmark_name}:{sample_id}"


def _generic_message_event(sample: BenchmarkSample, turn: ConversationTurn) -> MemoryIngestEvent:
    sample_metadata = dict(sample.metadata)
    turn_metadata = dict(turn.metadata)
    speaker = str(turn.speaker).strip() or str(turn.role).strip() or "speaker"
    role = str(turn.role).strip() or speaker
    timestamp = str(turn.session_timestamp).strip()
    return MemoryIngestEvent(
        text=str(turn.text).strip(),
        context_text="",
        session_id=str(turn.session_id).strip(),
        turn_id=str(turn.turn_id).strip(),
        user_id=_generic_conversation_user_id(sample),
        speaker=speaker,
        role=role,
        timestamp=timestamp or None,
        metadata={
            **turn_metadata,
            "benchmark": str(sample.benchmark_name).strip(),
            "sample_id": str(sample.sample_id).strip(),
            "query_metadata": dict(sample.query.metadata),
            "session_id": str(turn.session_id).strip(),
            "turn_id": str(turn.turn_id).strip(),
            "source_timestamp": timestamp,
            "source_speaker": speaker,
            "source_role": role,
            "sample_metadata": sample_metadata,
            "turn_metadata": turn_metadata,
        },
    )


class _GenericMemoryBindingSession:
    def __init__(self, *, binding_factory: Callable[[], MemorySystemBinding]) -> None:
        self.binding = binding_factory()
        self.system = self.binding.build_system()
        self.load_metadata: dict[str, Any] = {}

    def load_case(self, sample: BenchmarkSample, *, progress_callback: Callable[..., None] | None = None) -> None:
        if not sample.history_turns:
            raise ValueError("GenericMemoryBindingAdapter requires sample.history_turns.")

        total_turns = len(sample.history_turns)
        loaded_turns = 0
        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_init",
                sample=sample,
                total_turns=total_turns,
            )

        skipped_ingests: list[dict[str, Any]] = []
        for turn in sample.history_turns:
            event = _generic_message_event(sample, turn)
            row = _safe_binding_ingest_event(
                self.binding,
                self.system,
                event,
                sample,
                phase="ingest_generic_binding",
            )
            if row is not None:
                skipped_ingests.append(row)
            loaded_turns += 1
            if progress_callback is not None:
                _call_with_supported_kwargs(
                    progress_callback,
                    phase="memory_turn_done",
                    sample=sample,
                    turn_index=loaded_turns,
                    total_turns=total_turns,
                    turn_id=str(turn.turn_id).strip(),
                    turn_increment=1,
                )

        if progress_callback is not None:
            _call_with_supported_kwargs(
                progress_callback,
                phase="memory_finish",
                sample=sample,
                total_turns=total_turns,
                loaded_turns=loaded_turns,
            )

        self.load_metadata = {
            "loaded_turn_count": loaded_turns,
            "loaded_message_count": loaded_turns,
            "load_source": "history_turn_messages",
            "conversation_user_id": _generic_conversation_user_id(sample),
            **({"benchmark_skipped_ingests": skipped_ingests} if skipped_ingests else {}),
        }

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        metadata = dict(sample.metadata) if sample is not None else {}
        context = RecallContext(
            sample_id=str(sample.sample_id).strip() if sample is not None else "",
            user_id=_generic_conversation_user_id(sample) if sample is not None else "",
            speaker="conversation",
            metadata=metadata,
        )
        recall = _coerce_memory_recall(self.binding.recall(self.system, query, context=context))
        binding_name = str(getattr(self.binding, "name", "")).strip()
        return MemoryRecall(
            text=recall.text,
            source_ids=list(recall.source_ids),
            metadata={
                **self.load_metadata,
                **dict(recall.metadata),
                "recall_helper": f"{binding_name}.recall" if binding_name else "memory_binding.recall",
                "conversation_memories": recall.text,
                "num_conversation_memory_lines": _count_nonempty_lines(recall.text),
            },
        )


class GenericMemoryBindingAdapter:
    """Benchmark-neutral adapter for one memory binding per sample."""

    def __init__(self, *, binding_factory: Callable[[], MemorySystemBinding], name: str) -> None:
        self.binding_factory = binding_factory
        self.name = str(name).strip() or "memory"

    def create_session(self) -> _GenericMemoryBindingSession:
        return _GenericMemoryBindingSession(binding_factory=self.binding_factory)


def create_dual_speaker_locomo_memory_adapter(
    binding_factory: Callable[[], MemorySystemBinding],
    *,
    name: str | None = None,
    speaker_workers: int = 1,
) -> DualSpeakerLoCoMoMemoryAdapter:
    binding_name = ""
    if name is None:
        try:
            binding_name = str(getattr(binding_factory(), "name", "")).strip()
        except Exception:
            binding_name = ""
    return DualSpeakerLoCoMoMemoryAdapter(
        binding_factory=binding_factory,
        name=name or binding_name or "memory",
        speaker_workers=speaker_workers,
    )


def create_generic_memory_binding_adapter(
    binding_factory: Callable[[], MemorySystemBinding],
    *,
    name: str | None = None,
) -> GenericMemoryBindingAdapter:
    binding_name = ""
    if name is None:
        try:
            binding_name = str(getattr(binding_factory(), "name", "")).strip()
        except Exception:
            binding_name = ""
    return GenericMemoryBindingAdapter(
        binding_factory=binding_factory,
        name=name or binding_name or "memory",
    )


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in str(text).splitlines() if str(line).strip())


def create_mem0_memory_adapter(
    *,
    name: str = "mem0",
    top_k: int | None = None,
    similar_top_k: int = 5,
    speaker_workers: int = 1,
) -> DualSpeakerLoCoMoMemoryAdapter:
    """Create a ready-to-run benchmark adapter for the classic Mem0 reconstruction."""

    recall_top_k = 30 if top_k is None else top_k
    return create_dual_speaker_locomo_memory_adapter(
        lambda: mem0_memory.create_memory_binding(
            recent_top_k=6,
            recall_top_k=recall_top_k,
            similar_top_k=similar_top_k,
        ),
        name=name,
        speaker_workers=speaker_workers,
    )


def create_memmachine_memory_adapter(
    *,
    name: str = "memmachine",
    top_k: int | None = None,
    stm_record_budget: int = 20,
    profile_max_turns: int = 6,
    speaker_workers: int = 1,
) -> SharedConversationLoCoMoMemoryAdapter:
    """Create a LoCoMo adapter for the classic MemMachine reconstruction."""

    del speaker_workers
    limit = 30 if top_k is None else top_k
    return SharedConversationLoCoMoMemoryAdapter(
        binding_factory=lambda: memmachine_memory.create_memory_binding(
            limit=limit,
            expand_context=3,
            profile_top_k=10 if top_k is None else top_k,
            stm_record_budget=stm_record_budget,
            profile_max_turns=profile_max_turns,
        ),
        name=name,
    )


def create_amem_memory_adapter(
    *,
    name: str = "amem",
    top_k: int | None = None,
    speaker_workers: int = 1,
) -> SharedConversationLoCoMoMemoryAdapter:
    """Create a LoCoMo adapter for the classic A-MEM reconstruction."""

    del speaker_workers
    recall_top_k = 30 if top_k is None else top_k
    return SharedConversationLoCoMoMemoryAdapter(
        binding_factory=lambda: amem_memory.create_memory_binding(
            note_namespace="amem",
            candidate_k=5,
            recall_top_k=recall_top_k,
        ),
        name=name,
    )
