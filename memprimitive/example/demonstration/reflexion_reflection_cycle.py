"""Demo: failed trial -> reflection append -> next recall gets prompt context.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.reflexion_reflection_cycle

Or from this directory (script adds the repo root to ``sys.path``)::

    python reflexion_reflection_cycle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import Query
from memprimitive.example.classics.reflexion import ReflexionStrategy, ReflexionWorkstream


def main() -> None:
    workflow = ReflexionWorkstream(
        strategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        memory_size=3,
    )
    question = "Parse the input stream"

    packet = workflow.record_trial(
        question=question,
        scratchpad=(
            "Thought: parse the normal case first.\n"
            "Action: Finish[wrong answer]\n"
            "Observation: Answer is INCORRECT"
        ),
        is_correct=False,
        evaluator_feedback="The answer missed the empty-input edge case.",
    )
    next_context = workflow.recall(
        Query(
            text=question,
            metadata={
                "reflexion": {
                    "strategy": ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION.value,
                    "last_attempt": workflow.last_attempt,
                }
            },
        )
    )

    print("triggered reflection:", packet.evolution_decisions)
    print("stored reflections:", workflow.store.count("reflections"))
    print()
    print("=== Next-trial prompt context ===")
    print(next_context.text or "(no memory)")
    print("source record ids:", next_context.source_ids)


if __name__ == "__main__":
    main()
