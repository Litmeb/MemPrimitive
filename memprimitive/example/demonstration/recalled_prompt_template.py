"""Demonstration: inject a sub-recall result into a prompt template.

This example shows one practical pattern for the new ``{{ recalled_prompt }}``
support:

- seed the current store with an earlier profile memory
- configure an ``LLMRepresentation`` with ``retrieve_pipeline`` plus
  ``recall_query_template``
- let the representation prompt read ``{{ recalled_prompt }}`` while extracting
  a new summary for the incoming unit

The sub recall runs against the *current* store used by the active module,
not the ``store`` originally attached to the provided retrieve pipeline.

Usage:

    python -m memprimitive.example.demonstration.recalled_prompt_template
"""

from __future__ import annotations

from pprint import pprint

from memprimitive.baselines import (
    AppendOrganization,
    ConcatenateReadout,
    LLMRepresentation,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.core import MemoryRecord, MemoryStore, MemoryUnit, Observation, Query
from memprimitive.pipeline import MemoryPipeline


def _seed_profile_memory(store: MemoryStore, text: str) -> None:
    unit = MemoryUnit(
        unit_id="seed-profile",
        text=text,
        timestamp="2026-04-04T00:00:00+00:00",
        metadata={"session_id": "seed-session"},
    )
    store.append(MemoryRecord.from_unit(unit=unit, layer="default", sequence_id=store.next_sequence_id()))


def build_pipeline() -> MemoryPipeline:
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
    )
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=LLMRepresentation(
            field="summary",
            prompt=(
                "Summarize the incoming memory for later retrieval.\n"
                "Previously recalled context:\n{{ recalled_prompt }}\n\n"
                "Current memory:\n{{ unit.text }}"
            ),
            retrieve_pipeline=retrieve_pipeline,
            recall_query_template="profile context for {{ unit.text }}",
        ),
        organization=AppendOrganization(target_layer="default"),
        store=MemoryStore(),
    )


def main() -> None:
    pipeline = build_pipeline()
    _seed_profile_memory(
        pipeline.store,
        "Alice prefers concise technical answers and usually wants retrieval details preserved.",
    )

    packet = pipeline.ingest(
        Observation(
            text="Alice is adding template-driven recalled prompt support to her memory system.",
            source="dialogue",
            metadata={"session_id": "sess-demo"},
        )
    )

    recall = pipeline.recall(Query(text="What do we know about Alice?"))

    print("Latest stored summary field:")
    latest_record = pipeline.store.iter_records("default")[-1]
    pprint(latest_record.metadata.get("representation", {}))
    print()
    print("Representation prompt trace:")
    pprint(packet.trace.get("representation", {}))
    print()
    print("Recall after ingest:")
    print(recall.text)


if __name__ == "__main__":
    main()
