"""Mechanism-level reconstruction of Reflexion memory without the agent loop.

This version intentionally avoids the dedicated Reflexion helper modules and
instead reuses a more general pipeline pattern:

1. append each failed trial trace into a raw trial buffer,
2. trigger hierarchical generate-mode abstraction on that newly written trial,
3. write the generated reflection into a bounded reflections layer, and
4. render the retained reflections back into next-trial prompt context.

The design aligns with the released Reflexion repository's simpler memory path:
append-only textual reflections plus a small sliding window used as prompt
context for the next attempt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, Readout, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    BufferRetrieval,
    HierarchicalEvolution,
    PromptContextReadout,
    ScalarRuleTrigger,
)
from memprimitive.utils._reflexion_family import (
    DEFAULT_MEMORY_SIZE,
    DEFAULT_REFLECTION_LAYER,
    DEFAULT_TRIAL_LAYER,
)


DEFAULT_REFLEXION_HIERARCHICAL_PROMPT = (
    "You are generating Reflexion-style verbal reinforcement memory from failed trials.\n"
    "Return strict JSON with exactly these top-level keys: "
    "reflection, question, last_attempt, evaluator_feedback, trial_index.\n"
    "Use the most recent failed trial in the provided records as the canonical source of question, "
    "last_attempt, evaluator_feedback, and trial_index.\n"
    "The field 'reflection' must be a concise high-level plan that starts with 'Reflection'.\n"
    "Do not include any keys beyond the requested schema."
)


def build_reflexion_memory_system(
    *,
    memory_size: int = DEFAULT_MEMORY_SIZE,
    reflection_layer: str = DEFAULT_REFLECTION_LAYER,
    trial_layer: str = DEFAULT_TRIAL_LAYER,
    default_strategy: str = "reflexion",
    reflection_prompt: str = DEFAULT_REFLEXION_HIERARCHICAL_PROMPT,
) -> dict[str, object]:
    """Build a Reflexion-style memory-only system from general primitives."""

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name=trial_layer, theme="working", indices=("temporal",)),
            StoreLayerSpec(name=reflection_layer, theme="semantic", indices=("temporal",)),
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer=trial_layer),
        evolution_trigger=ScalarRuleTrigger(
            slot="evolution_trigger",
            signal_key="is_correct",
            threshold=0.5,
            comparator="<",
            signal_source="observation",
            aggregate="broadcast",
        ),
        memory_evolution=HierarchicalEvolution(
            source_layer=trial_layer,
            extract_mode="generate",
            extract_fields=("reflection", "question", "last_attempt", "evaluator_feedback", "trial_index"),
            record_text_field="reflection",
            target_layer=reflection_layer,
            selection_mode="latest_active_units",
            retention_size=memory_size,
            prompt=reflection_prompt,
        ),
        store=store,
    )

    recall_pipeline = MemoryPipeline(
        retrieval=BufferRetrieval(top_k=memory_size, layer=reflection_layer),
        readout=PromptContextReadout(
            memory_layer=reflection_layer,
            default_strategy=default_strategy,
            top_k=memory_size,
        ),
        store=store,
    )

    return {
        "store": store,
        "write_pipeline": write_pipeline,
        "recall_pipeline": recall_pipeline,
        "memory_size": memory_size,
        "reflection_layer": reflection_layer,
        "trial_layer": trial_layer,
        "default_strategy": default_strategy,
        "reflection_prompt": reflection_prompt,
    }


def ingest_failed_trial(
    system: dict[str, object],
    *,
    question: str,
    last_attempt: str,
    evaluator_feedback: str,
    trial_index: int = 1,
    strategy: str | None = None,
    source: str = "reasoning_trial",
    observation_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    is_correct: bool = False,
) -> Any:
    """Append one trial and, if failed, abstract it into a reflection record."""

    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline, MemoryPipeline)

    payload = dict(metadata or {})
    reflexion_metadata = dict(payload.get("reflexion", {}))
    reflexion_metadata.update(
        {
            "question": question,
            "last_attempt": last_attempt,
            "scratchpad": last_attempt,
            "evaluator_feedback": evaluator_feedback,
            "trial_index": trial_index,
        }
    )
    if strategy is not None:
        reflexion_metadata["strategy"] = strategy
    payload["reflexion"] = reflexion_metadata

    trigger_payload = dict(payload.get("trigger", {}))
    trigger_signals = dict(trigger_payload.get("signals", {}))
    trigger_signals["is_correct"] = 1.0 if is_correct else 0.0
    trigger_payload["signals"] = trigger_signals
    payload["trigger"] = trigger_payload

    return write_pipeline.ingest(
        Observation(
            text=observation_text or last_attempt,
            source=source,
            metadata=payload,
        )
    )


def recall_reflection_context(
    system: dict[str, object],
    *,
    question: str,
    strategy: str | None = None,
    last_attempt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Readout:
    """Render Reflexion-style prompt context for the next trial."""

    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)

    query_metadata = dict(metadata or {})
    reflexion_metadata = dict(query_metadata.get("reflexion", {}))
    if strategy is not None:
        reflexion_metadata["strategy"] = strategy
    if last_attempt is not None:
        reflexion_metadata["last_attempt"] = last_attempt
    if reflexion_metadata:
        query_metadata["reflexion"] = reflexion_metadata

    return recall_pipeline.recall(Query(text=question, metadata=query_metadata))


def recall_reflections(
    system: dict[str, object],
    *,
    question: str,
    strategy: str = "reflexion",
    last_attempt: str | None = None,
) -> str:
    """Convenience wrapper that returns only the rendered prompt text."""

    return recall_reflection_context(
        system,
        question=question,
        strategy=strategy,
        last_attempt=last_attempt,
    ).text


def main() -> None:
    system = build_reflexion_memory_system(
        memory_size=3,
        default_strategy="last_trial_and_reflexion",
    )
    store = system["store"]
    assert isinstance(store, MemoryStore)

    ingest_failed_trial(
        system,
        question="Find the first matching index in the stream.",
        last_attempt="I started from index 1 and returned 3 without checking the first valid match.",
        evaluator_feedback="You skipped the earliest valid match.",
        trial_index=1,
    )
    ingest_failed_trial(
        system,
        question="Find the first matching index in the stream.",
        last_attempt="I checked the earlier item but still finalized the off-by-one result.",
        evaluator_feedback="You still returned the wrong boundary index.",
        trial_index=2,
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("stored trials:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "trial_index": record.metadata.get("reflexion", {}).get("trial_index"),
            }
            for record in store.iter_records("trial_buffer")
        ]
    )
    print()

    print("stored reflections:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "field_payload": record.metadata.get("hierarchical", {}).get("field_payload", {}),
            }
            for record in store.iter_records("reflections")
        ]
    )
    print()

    print("next-trial prompt context:")
    print(
        recall_reflection_context(
            system,
            question="Find the first matching index in the stream.",
            strategy="last_trial_and_reflexion",
            last_attempt="I checked the earlier item but still finalized the off-by-one result.",
        ).text
    )


if __name__ == "__main__":
    main()
