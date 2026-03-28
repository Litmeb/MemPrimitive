"""End-to-end example showing Reflexion-like back-half composition with baselines.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.reflexion_reflection_cycle

Or from this directory (script adds the repo root to ``sys.path``)::

    python reflexion_reflection_cycle.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    BasicRepresentation,
    BufferRetrieval,
    OutcomeConditionedEvolutionTrigger,
    PassThroughUnitFormation,
    PlacementWithoutAppendOrganization,
    PromptContextReadout,
    ReflectionGenerationEvolution,
)
from memprimitive.utils._runtime import get_classic_runtime


def main() -> None:
    get_classic_runtime().require_llm(capability="Reflexion reflection demonstration")

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="reflections", theme="semantic", indices=("temporal", "keyword")),
        ]
    )
    store = MemoryStore(topology=topology)

    ingest_pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        organization=PlacementWithoutAppendOrganization(target_layer="trial_buffer"),
        evolution_trigger=OutcomeConditionedEvolutionTrigger(),
        memory_evolution=ReflectionGenerationEvolution(target_layer="reflections"),
        store=store,
    )

    question = "Parse the input stream"
    packet = ingest_pipeline.ingest(
        Observation(
            text="Thought: parse the normal case first. Action: Finish[wrong answer].",
            source="dialogue",
            metadata={
                "reflexion": {
                    "question": question,
                    "scratchpad": (
                        "Thought: parse the normal case first.\n"
                        "Action: Finish[wrong answer]\n"
                        "Observation: Answer is INCORRECT"
                    ),
                    "is_correct": False,
                    "evaluator_feedback": "The answer missed the empty-input edge case.",
                    "trial_index": 1,
                }
            },
        )
    )

    recall_pipeline = MemoryPipeline(
        retrieval=BufferRetrieval(top_k=3, layer="reflections"),
        readout=PromptContextReadout(
            memory_layer="reflections",
            default_strategy="last_trial_and_reflexion",
            top_k=3,
        ),
        store=store,
    )
    readout = recall_pipeline.recall(
        Query(
            text=question,
            metadata={
                "reflexion": {
                    "strategy": "last_trial_and_reflexion",
                    "last_attempt": (
                        "Thought: parse the normal case first.\n"
                        "Action: Finish[wrong answer]\n"
                        "Observation: Answer is INCORRECT"
                    ),
                }
            },
        )
    )

    print("store topology:")
    pprint(
        [
            {
                "name": layer.name,
                "theme": layer.theme,
                "shape": layer.shape,
                "indices": layer.indices,
            }
            for layer in store.topology.layers
        ]
    )
    print()

    print("evolution trace:")
    pprint(packet.trace["memory_evolution"])
    print()

    print("reflection layer records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "metadata": record.metadata.get("reflection"),
            }
            for record in store.iter_records("reflections")
        ]
    )
    print()

    print("next-trial prompt context:")
    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
