"""Shared types and runtime protocols for MemPrimitive benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core import Observation, Query


@dataclass(slots=True)
class ConversationTurn:
    turn_id: str
    session_id: str
    session_timestamp: str
    role: str
    speaker: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> Observation:
        speaker_label = str(self.speaker).strip() or str(self.role).strip() or "speaker"
        timestamp = str(self.session_timestamp).strip()
        observation_metadata = {
            **dict(self.metadata),
            "turn_id": str(self.turn_id).strip(),
            "session_id": str(self.session_id).strip(),
            "session_timestamp": timestamp,
            "role": str(self.role).strip(),
            "speaker": str(self.speaker).strip(),
        }
        observation_kwargs: dict[str, Any] = {
            "text": f"{speaker_label}: {str(self.text).strip()}",
            "source": "dialogue",
            "metadata": observation_metadata,
        }
        if timestamp:
            observation_kwargs["timestamp"] = timestamp
        return Observation(**observation_kwargs)


def default_turn_to_observation(turn: ConversationTurn) -> Observation:
    """Project one normalized conversation turn into a standard Observation."""

    return turn.to_observation()


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    benchmark_name: str
    history_observations: list[Observation]
    query: Query
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    history_turns: list[ConversationTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history_observations = list(self.history_observations)
        self.history_turns = list(self.history_turns)
        self.metadata = dict(self.metadata)
        if not self.history_observations and self.history_turns:
            self.history_observations = [default_turn_to_observation(turn) for turn in self.history_turns]


@dataclass(slots=True)
class MemoryRecall:
    text: str
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkPrediction:
    sample_id: str
    benchmark_name: str
    query_text: str
    reference_answer: str
    predicted_answer: str
    retrieved_text: str
    retrieved_source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_adapter_name: str = ""
    memory_metadata: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRunResult:
    benchmark_name: str
    memory_adapter_name: str
    predictions: list[BenchmarkPrediction]
    aggregate_scores: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AnswerRunner(Protocol):
    name: str

    def answer(self, *, sample: BenchmarkSample, memory_recall: MemoryRecall) -> str:
        """Generate the final benchmark answer from one normalized memory recall."""


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str

    def iter_samples(self, *, limit: int | None = None) -> list[BenchmarkSample] | tuple[BenchmarkSample, ...] | Any:
        """Yield normalized benchmark samples."""


@runtime_checkable
class MemorySession(Protocol):
    def load_case(self, sample: BenchmarkSample) -> None:
        """Load one benchmark case into a fresh memory runtime."""

    def recall(self, query: Query, *, sample: BenchmarkSample | None = None) -> MemoryRecall:
        """Recall memory for one benchmark query."""


@runtime_checkable
class MemoryAdapter(Protocol):
    name: str

    def create_session(self) -> MemorySession:
        """Create a fresh per-sample memory session."""
