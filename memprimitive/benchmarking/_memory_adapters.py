"""Memory-side adapters for benchmark evaluation."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..config import load_pipeline_from_yaml
from ..core import Observation, Query, Readout
from ..pipeline import FreeMemoryPipeline, MemoryPipeline
from ..example.classics import mem0_memory
from ._types import BenchmarkSample, ConversationTurn, MemoryRecall, default_turn_to_observation


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
        for observation in observations:
            self.pipeline.ingest(observation)
        self.load_metadata = metadata

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
        for first_turn, second_turn in _iter_pairwise_turns(sample.history_turns):
            pair_count += 1
            _call_with_supported_kwargs(
                self.ingest_pair,
                system,
                user_text=str(first_turn.text).strip(),
                assistant_text=str(second_turn.text).strip() if second_turn is not None else "",
                session_id=str(first_turn.session_id).strip(),
                turn_id=str(second_turn.turn_id).strip() if second_turn is not None else str(first_turn.turn_id).strip(),
                timestamp=str(first_turn.session_timestamp).strip(),
            )
        return {
            "loaded_pair_count": pair_count,
            "load_source": "history_turn_pairs",
        }


def _locomo_turn_text(turn: ConversationTurn | None) -> str:
    if turn is None:
        return ""
    speaker_label = str(turn.speaker).strip() or str(turn.role).strip() or "speaker"
    text = str(turn.text).strip()
    return f"{speaker_label}: {text}" if text else ""


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


def _build_mem0_locomo_systems(*, recent_top_k: int, recall_top_k: int, similar_top_k: int) -> dict[str, object]:
    return {
        "speaker_1_system": mem0_memory.build_mem0_memory_system(
            recent_top_k=recent_top_k,
            recall_top_k=recall_top_k,
            similar_top_k=similar_top_k,
        ),
        "speaker_2_system": mem0_memory.build_mem0_memory_system(
            recent_top_k=recent_top_k,
            recall_top_k=recall_top_k,
            similar_top_k=similar_top_k,
        ),
    }


def _load_mem0_locomo_case(
    system: dict[str, object],
    sample: BenchmarkSample,
    *,
    progress_callback: Callable[..., None] | None = None,
    speaker_workers: int = 1,
) -> dict[str, Any]:
    if not sample.history_turns:
        raise ValueError("Mem0 LoCoMo adapter requires sample.history_turns.")
    if speaker_workers <= 0:
        raise ValueError("speaker_workers must be positive.")

    speaker_a = str(sample.metadata.get("speaker_a", "")).strip() or "speaker_1"
    speaker_b = str(sample.metadata.get("speaker_b", "")).strip() or "speaker_2"
    speaker_a_user_id = _locomo_speaker_user_id(sample, speaker_key="speaker_a", fallback_name="speaker_1")
    speaker_b_user_id = _locomo_speaker_user_id(sample, speaker_key="speaker_b", fallback_name="speaker_2")
    pair_count = 0

    speaker_1_system = system["speaker_1_system"]
    speaker_2_system = system["speaker_2_system"]
    total_turns = len(sample.history_turns)
    loaded_turns = 0

    if progress_callback is not None:
        _call_with_supported_kwargs(
            progress_callback,
            phase="memory_init",
            sample=sample,
            total_turns=total_turns,
        )

    executor = (
        ThreadPoolExecutor(max_workers=min(2, speaker_workers))
        if speaker_workers > 1
        else None
    )
    try:
        for first_turn, second_turn in _iter_pairwise_turns(sample.history_turns):
            pair_count += 1
            timestamp = str(first_turn.session_timestamp).strip()
            turn_id = str(second_turn.turn_id).strip() if second_turn is not None else str(first_turn.turn_id).strip()
            turns_in_pair = 2 if second_turn is not None else 1

            speaker_1_user_text, speaker_1_assistant_text = _locomo_pair_for_speaker(
                first_turn,
                second_turn,
                speaker_name=speaker_a,
            )
            speaker_2_user_text, speaker_2_assistant_text = _locomo_pair_for_speaker(
                first_turn,
                second_turn,
                speaker_name=speaker_b,
            )

            ingest_calls = []
            if speaker_1_user_text:
                ingest_calls.append(
                    {
                        "system": speaker_1_system,
                        "user_text": speaker_1_user_text,
                        "assistant_text": speaker_1_assistant_text,
                    }
                )
            if speaker_2_user_text:
                ingest_calls.append(
                    {
                        "system": speaker_2_system,
                        "user_text": speaker_2_user_text,
                        "assistant_text": speaker_2_assistant_text,
                    }
                )

            if executor is None or len(ingest_calls) <= 1:
                for call in ingest_calls:
                    mem0_memory.ingest_message_pair(
                        call["system"],
                        user_text=call["user_text"],
                        assistant_text=call["assistant_text"],
                        session_id=str(first_turn.session_id).strip(),
                        turn_id=turn_id,
                        timestamp=timestamp,
                    )
            else:
                futures = [
                    executor.submit(
                        mem0_memory.ingest_message_pair,
                        call["system"],
                        user_text=call["user_text"],
                        assistant_text=call["assistant_text"],
                        session_id=str(first_turn.session_id).strip(),
                        turn_id=turn_id,
                        timestamp=timestamp,
                    )
                    for call in ingest_calls
                ]
                for future in futures:
                    future.result()

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

    return {
        "loaded_pair_count": pair_count,
        "loaded_turn_count": loaded_turns,
        "load_source": "history_turn_pairs",
        "speaker_1_user_id": speaker_a_user_id,
        "speaker_2_user_id": speaker_b_user_id,
        "speaker_1_name": speaker_a,
        "speaker_2_name": speaker_b,
        "speaker_workers": speaker_workers,
    }


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in str(text).splitlines() if str(line).strip())


def _recall_mem0_locomo_case(
    system: dict[str, object],
    query: Query,
    *,
    sample: BenchmarkSample | None = None,
    speaker_workers: int = 1,
) -> MemoryRecall:
    if speaker_workers <= 0:
        raise ValueError("speaker_workers must be positive.")
    sample_metadata = dict(sample.metadata) if sample is not None else {}
    speaker_a = str(sample_metadata.get("speaker_a", "")).strip()
    speaker_b = str(sample_metadata.get("speaker_b", "")).strip()
    if sample is None:
        speaker_a_user_id = "speaker_1"
        speaker_b_user_id = "speaker_2"
    else:
        speaker_a_user_id = _locomo_speaker_user_id_from_metadata(
            sample_metadata,
            sample_id=str(sample.sample_id).strip(),
            speaker_key="speaker_a",
            fallback_name="speaker_1",
        )
        speaker_b_user_id = _locomo_speaker_user_id_from_metadata(
            sample_metadata,
            sample_id=str(sample.sample_id).strip(),
            speaker_key="speaker_b",
            fallback_name="speaker_2",
        )

    speaker_1_system = system["speaker_1_system"]
    speaker_2_system = system["speaker_2_system"]

    if speaker_workers > 1:
        with ThreadPoolExecutor(max_workers=min(2, speaker_workers)) as executor:
            speaker_1_future = executor.submit(mem0_memory.recall_profile, speaker_1_system, user_query=query.text)
            speaker_2_future = executor.submit(mem0_memory.recall_profile, speaker_2_system, user_query=query.text)
            speaker_1_memories = speaker_1_future.result()
            speaker_2_memories = speaker_2_future.result()
    else:
        speaker_1_memories = mem0_memory.recall_profile(speaker_1_system, user_query=query.text)
        speaker_2_memories = mem0_memory.recall_profile(speaker_2_system, user_query=query.text)

    speaker_1_memories_text = str(speaker_1_memories).strip()
    speaker_2_memories_text = str(speaker_2_memories).strip()
    recall_text = "\n\n".join(
        part
        for part in (
            f"{speaker_a_user_id or 'speaker_1'}\n{speaker_1_memories_text}" if speaker_1_memories_text else "",
            f"{speaker_b_user_id or 'speaker_2'}\n{speaker_2_memories_text}" if speaker_2_memories_text else "",
        )
        if part
    )

    metadata = {
        "recall_helper": "recall_profile",
        "speaker_1_user_id": speaker_a_user_id,
        "speaker_2_user_id": speaker_b_user_id,
        "speaker_1_memories": speaker_1_memories_text,
        "speaker_2_memories": speaker_2_memories_text,
        "num_speaker_1_memories": _count_nonempty_lines(speaker_1_memories_text),
        "num_speaker_2_memories": _count_nonempty_lines(speaker_2_memories_text),
        "speaker_1_name": speaker_a,
        "speaker_2_name": speaker_b,
    }
    return MemoryRecall(text=recall_text, source_ids=[], metadata=metadata)


class _Mem0LoCoMoMemorySession:
    def __init__(self, *, system_factory: Callable[[], dict[str, object]], speaker_workers: int) -> None:
        if speaker_workers <= 0:
            raise ValueError("speaker_workers must be positive.")
        self.system = system_factory()
        self.speaker_workers = speaker_workers
        self.load_metadata: dict[str, Any] = {}

    def load_case(self, sample: BenchmarkSample, *, progress_callback: Callable[..., None] | None = None) -> None:
        self.load_metadata = _load_mem0_locomo_case(
            self.system,
            sample,
            progress_callback=progress_callback,
            speaker_workers=self.speaker_workers,
        )

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        recall_result = _recall_mem0_locomo_case(
            self.system,
            query,
            sample=sample,
            speaker_workers=self.speaker_workers,
        )
        return _coerce_memory_recall(recall_result, base_metadata=self.load_metadata)


class Mem0LoCoMoMemoryAdapter:
    name = "mem0"

    def __init__(
        self,
        *,
        recent_top_k: int,
        recall_top_k: int,
        similar_top_k: int,
        speaker_workers: int = 1,
        name: str = "mem0",
    ) -> None:
        if speaker_workers <= 0:
            raise ValueError("speaker_workers must be positive.")
        self.recent_top_k = recent_top_k
        self.recall_top_k = recall_top_k
        self.similar_top_k = similar_top_k
        self.speaker_workers = speaker_workers
        self.name = str(name).strip() or "mem0"

    def create_session(self) -> _Mem0LoCoMoMemorySession:
        return _Mem0LoCoMoMemorySession(
            system_factory=lambda: _build_mem0_locomo_systems(
                recent_top_k=self.recent_top_k,
                recall_top_k=self.recall_top_k,
                similar_top_k=self.similar_top_k,
            ),
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


def create_mem0_memory_adapter(
    *,
    name: str = "mem0",
    top_k: int | None = None,
    similar_top_k: int = 5,
    speaker_workers: int = 1,
) -> Mem0LoCoMoMemoryAdapter:
    """Create a ready-to-run benchmark adapter for the classic Mem0 reconstruction."""

    recall_top_k = 30 if top_k is None else top_k
    return Mem0LoCoMoMemoryAdapter(
        name=name,
        recent_top_k=6,
        recall_top_k=recall_top_k,
        similar_top_k=similar_top_k,
        speaker_workers=speaker_workers,
    )
