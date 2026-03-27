"""Reflexion (Shinn et al., 2023) - HotPotQA-style reasoning loop sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.reflexion

Or from this directory (script adds the repo root to ``sys.path``)::

    python reflexion.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, Readout, StoreLayerSpec, StoreTopology
from memprimitive.baselines import AlwaysWriteTrigger, BasicRepresentation, PassThroughUnitFormation
from memprimitive.classic_modules.reflexion import (
    DEFAULT_MEMORY_SIZE,
    DEFAULT_REFLECTION_LAYER,
    DEFAULT_TRIAL_LAYER,
    ReflectionMemoryEvolution,
    ReflexionContextReadout,
    ReflexionMemoryRetrieval,
    ReflexionTrialOrganization,
    TrialFailureEvolutionTrigger,
)

REFLECTION_HEADER = (
    "You have attempted to answer following question before and failed. "
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question."
)
REFLECTION_AFTER_LAST_TRIAL_HEADER = (
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question."
)
LAST_TRIAL_HEADER = (
    "You have attempted to answer the following question before and failed. "
    "Below is the last trial you attempted to answer the question."
)


class ReflexionStrategy(Enum):
    """Reasoning-task memory injection strategies from the original repo."""

    NONE = "base"
    LAST_ATTEMPT = "last_trial"
    REFLEXION = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION = "last_trial_and_reflexion"


@dataclass(slots=True)
class ReflexionTrial:
    """One reasoning trial recorded by the example workflow."""

    question: str
    scratchpad: str
    is_correct: bool
    evaluator_feedback: str = ""
    answer: str = ""
    trial_index: int = 0

    def as_metadata(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "scratchpad": self.scratchpad,
            "is_correct": self.is_correct,
            "evaluator_feedback": self.evaluator_feedback,
            "answer": self.answer,
            "trial_index": self.trial_index,
        }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def build_reflexion_pipeline(
    *,
    store: MemoryStore | None = None,
    reflection_layer: str = DEFAULT_REFLECTION_LAYER,
    trial_layer: str = DEFAULT_TRIAL_LAYER,
    strategy: str = ReflexionStrategy.REFLEXION.value,
    memory_size: int = DEFAULT_MEMORY_SIZE,
    reflection_window: int | None = None,
    reflection_top_k: int | None = None,
) -> MemoryPipeline:
    effective_size = memory_size
    if reflection_window is not None:
        effective_size = reflection_window
    if reflection_top_k is not None:
        effective_size = reflection_top_k
    if effective_size <= 0:
        raise ValueError("build_reflexion_pipeline requires memory_size > 0.")
    valid_strategies = {item.value for item in ReflexionStrategy}
    if strategy not in valid_strategies:
        raise ValueError(f"build_reflexion_pipeline requires strategy in {sorted(valid_strategies)}.")

    if store is None:
        store = MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(
                        name=reflection_layer,
                        theme="semantic",
                        indices=("temporal", "keyword"),
                        capacity="sliding_window",
                    ),
                ]
            )
        )
    elif not store.has_layer(reflection_layer):
        store.ensure_layer(reflection_layer, allow_create=True, theme="semantic")

    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "keywords", "tags")),
        write_trigger=AlwaysWriteTrigger(),
        organization=ReflexionTrialOrganization(target_layer=trial_layer),
        evolution_trigger=TrialFailureEvolutionTrigger(),
        memory_evolution=ReflectionMemoryEvolution(target_layer=reflection_layer, memory_size=effective_size),
        retrieval=ReflexionMemoryRetrieval(reflection_layer=reflection_layer, memory_size=effective_size),
        readout=ReflexionContextReadout(
            reflection_layer=reflection_layer,
            default_strategy=strategy,
            memory_size=effective_size,
        ),
    )


class ReflexionWorkstream:
    """Workflow wrapper for HotPotQA-style Reflexion reasoning trials."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        reflection_layer: str = DEFAULT_REFLECTION_LAYER,
        strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION,
        memory_size: int = DEFAULT_MEMORY_SIZE,
        reflection_window: int | None = None,
        reflection_top_k: int | None = None,
    ) -> None:
        effective_size = memory_size
        if reflection_window is not None:
            effective_size = reflection_window
        if reflection_top_k is not None:
            effective_size = reflection_top_k
        self.strategy = strategy
        self.memory_size = effective_size
        self.reflection_layer = reflection_layer
        self.pipeline = build_reflexion_pipeline(
            store=store,
            reflection_layer=reflection_layer,
            strategy=strategy.value,
            memory_size=effective_size,
        )
        self.current_question: str | None = None
        self.last_attempt: str = ""
        self.last_correct: bool | None = None
        self.last_reflection: str = ""
        self.trial_count = 0

    @property
    def store(self) -> MemoryStore:
        return self.pipeline.store

    def _sync_state_from_store(self) -> None:
        records = self.store.iter_records(self.reflection_layer)
        self.last_reflection = records[-1].text if records else ""

    def record_trial(
        self,
        *,
        question: str,
        scratchpad: str,
        is_correct: bool,
        evaluator_feedback: str = "",
        answer: str = "",
    ):
        self.current_question = question
        self.trial_count += 1
        observation = Observation(
            text=_normalize_text(scratchpad) or question,
            source="reasoning_trial",
            metadata={
                "reflexion": {
                    "question": question,
                    "task": question,
                    "scratchpad": scratchpad,
                    "is_correct": is_correct,
                    "evaluator_feedback": evaluator_feedback,
                    "feedback": evaluator_feedback,
                    "answer": answer,
                    "trial_index": self.trial_count,
                }
            },
        )
        packet = self.pipeline.ingest(observation)
        self.last_attempt = _normalize_text(scratchpad)
        self.last_correct = is_correct
        self._sync_state_from_store()
        return packet

    def build_memory_context(
        self,
        question: str | None = None,
        *,
        strategy: ReflexionStrategy | None = None,
        last_attempt: str | None = None,
    ) -> Readout:
        resolved_question = question or self.current_question or ""
        query = Query(
            text=resolved_question or "Current question",
            metadata={
                "reflexion": {
                    "strategy": (strategy or self.strategy).value,
                    "last_attempt": self.last_attempt if last_attempt is None else last_attempt,
                }
            },
        )
        return self.pipeline.recall(query)

    def run_trial(
        self,
        *,
        question: str,
        scratchpad: str,
        is_correct: bool,
        evaluator_feedback: str = "",
        answer: str = "",
        next_strategy: ReflexionStrategy | None = None,
    ) -> tuple[Any, Readout]:
        packet = self.record_trial(
            question=question,
            scratchpad=scratchpad,
            is_correct=is_correct,
            evaluator_feedback=evaluator_feedback,
            answer=answer,
        )
        context = self.build_memory_context(question, strategy=next_strategy)
        return packet, context

    def ingest(self, observation: Observation):
        """Backward-compatible ingest wrapper for tests/examples."""
        metadata = dict(observation.metadata)
        controls = metadata.get("reflexion", {}) if isinstance(metadata.get("reflexion"), dict) else {}
        question = _normalize_text(controls.get("question") or controls.get("task") or observation.text)
        scratchpad = _normalize_text(controls.get("scratchpad") or controls.get("last_attempt") or observation.text)
        raw_correct = controls.get("is_correct", controls.get("success", False))
        is_correct = bool(raw_correct) if isinstance(raw_correct, (bool, int, float)) else str(raw_correct).strip().casefold() in {"true", "yes", "1", "success", "passed", "correct"}
        evaluator_feedback = _normalize_text(controls.get("evaluator_feedback") or controls.get("feedback") or "")
        return self.record_trial(
            question=question,
            scratchpad=scratchpad,
            is_correct=is_correct,
            evaluator_feedback=evaluator_feedback,
        )

    def recall(self, query: Query) -> Readout:
        metadata = dict(query.metadata)
        metadata.setdefault("reflexion", {})
        if isinstance(metadata["reflexion"], dict):
            metadata["reflexion"].setdefault("strategy", self.strategy.value)
            metadata["reflexion"].setdefault("last_attempt", self.last_attempt)
        return self.pipeline.recall(Query(text=query.text, query_id=query.query_id, timestamp=query.timestamp, embedding=query.embedding, metadata=metadata))


def main() -> None:
    question = "Who wrote Pride and Prejudice?"
    workflow = ReflexionWorkstream(
        strategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        memory_size=3,
    )

    scripted_trials = [
        {
            "scratchpad": (
                "Thought: I think Pride and Prejudice might have been written by Charlotte Bronte.\n"
                "Action: Finish[Charlotte Bronte]\n"
                "Observation: Answer is INCORRECT"
            ),
            "is_correct": False,
            "feedback": "The answer confused Jane Austen with the Bronte sisters.",
        },
        {
            "scratchpad": (
                "Thought: The previous answer confused Austen and Bronte. Pride and Prejudice is by Jane Austen.\n"
                "Action: Finish[Jane Austen]\n"
                "Observation: Answer is CORRECT"
            ),
            "is_correct": True,
            "feedback": "The answer matches the gold solution.",
        },
    ]

    for trial_index, trial in enumerate(scripted_trials, start=1):
        context = workflow.build_memory_context(question, strategy=workflow.strategy)
        print(f"=== Trial {trial_index} memory context ===")
        print(context.text or "(no memory)")
        print()

        packet, next_context = workflow.run_trial(
            question=question,
            scratchpad=trial["scratchpad"],
            is_correct=trial["is_correct"],
            evaluator_feedback=trial["feedback"],
        )

        print(f"=== Trial {trial_index} result ===")
        print("triggered reflection:", packet.trace["evolution_trigger"]["triggered"])
        print("reflection count:", workflow.store.count("reflections"))
        print()

        if trial["is_correct"]:
            print("=== Final memory context ===")
            print(next_context.text or "(no memory)")
            print("source record ids:", next_context.source_ids)
            break


if __name__ == "__main__":
    main()


__all__ = (
    "LAST_TRIAL_HEADER",
    "REFLECTION_AFTER_LAST_TRIAL_HEADER",
    "REFLECTION_HEADER",
    "ReflexionStrategy",
    "ReflexionTrial",
    "ReflexionWorkstream",
    "build_reflexion_pipeline",
)
