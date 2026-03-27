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
    PassThroughUnitFormation,
)
from memprimitive.baselines._trigger_family import (
    FeedbackPresenceSignal,
    FeedbackSchemaGate,
    OutcomeCorrectnessSignal,
    ThresholdPolicy,
    WeightedSumScorer,
)
from memprimitive.baselines.evolution_trigger import compose_evolution_trigger


def build_pipeline() -> MemoryPipeline:
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=AlwaysWriteTrigger(),
        organization=AppendOrganization(),
        evolution_trigger=compose_evolution_trigger(
            name="demo_reflexion_failed_trial_trigger",
            signal_providers=(OutcomeCorrectnessSignal(), FeedbackPresenceSignal()),
            scorer=WeightedSumScorer(weights={"trial_failed": 1.0, "feedback_present": 0.1}),
            gate=FeedbackSchemaGate(),
            policy=ThresholdPolicy(threshold=1.0),
            input_requirements=("units", "observation"),
        ),
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
