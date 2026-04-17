"""Memory-side adapters for benchmark evaluation."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from ..config import load_pipeline_from_yaml
from ..core import Observation, Query, Readout
from ..pipeline import FreeMemoryPipeline, MemoryPipeline
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
            )
        return {
            "loaded_pair_count": pair_count,
            "load_source": "history_turn_pairs",
        }


def create_mem0_memory_adapter(*, name: str = "mem0") -> PairwiseDialogueMemoryAdapter:
    """Create a ready-to-run benchmark adapter for the classic Mem0 reconstruction."""

    from ..example.classics.mem0_memory import build_mem0_memory_system, ingest_message_pair, recall_profile

    def _recall(system: dict[str, object], query: Query) -> MemoryRecall:
        return MemoryRecall(
            text=recall_profile(system, user_query=query.text),
            source_ids=[],
            metadata={"recall_helper": "recall_profile"},
        )

    return PairwiseDialogueMemoryAdapter(
        name=name,
        system_factory=build_mem0_memory_system,
        ingest_pair=ingest_message_pair,
        recall=_recall,
    )
