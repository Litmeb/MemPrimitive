"""Demo: failed-trial feedback opens the Reflexion-style evolution trigger.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.reflexion_trigger_failed_trial

Or from this directory (script adds the repo root to ``sys.path``)::

    python reflexion_trigger_failed_trial.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example/demonstration/reflexion_trigger_failed_trial.py``
# leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, Observation
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOnlyEvolution,
    AppendOrganization,
    BasicRepresentation,
    OutcomeConditionedEvolutionTrigger,
    PassThroughUnitFormation,
)


def build_pipeline() -> MemoryPipeline:
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(),
        evolution_trigger=OutcomeConditionedEvolutionTrigger(),
        memory_evolution=AppendOnlyEvolution(),
    )


def main() -> None:
    pipeline = build_pipeline()
    packet = pipeline.ingest(
        Observation(
            text="Trial scratchpad: parser answer misses the edge case.",
            source="dialogue",
            metadata={
                "reflexion": {
                    "is_correct": False,
                    "evaluator_feedback": "The answer missed the empty-input edge case.",
                    "trial_index": 2,
                }
            },
        )
    )

    print("evolution_decisions:", packet.evolution_decisions)
    print("evolution_trigger trace:")
    pprint(packet.trace["evolution_trigger"])


if __name__ == "__main__":
    main()
