"""Demonstration: inject a sub-recall result into a prompt template.

This example shows one practical pattern for the new ``{{ recalled_prompt }}``
support:

- seed the current store with an earlier profile memory
- configure an ``LLMRepresentation`` with ``text_prompt(...)`` that sets
  ``recall_plan``, ``recall_query_builder``, and ``sub_recall_pipeline``
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
from memprimitive.utils._template import text_prompt


def _seed_profile_memory(store: MemoryStore, text: str) -> None:
    unit = MemoryUnit(
        unit_id="seed-profile",
        text=text,
        timestamp="2026-04-04T00:00:00+00:00",
        metadata={"session_id": "seed-session"},
    )
    store.append(MemoryRecord.from_unit(unit=unit, layer="default", sequence_id=store.next_sequence_id()))


def build_recalled_prompt_query(packet, current_store, context) -> str:
    """Named helper so YAML config can import the same recall-query builder."""

    unit = context.get("unit", {})
    unit_text = str(unit.get("text", "")).strip()
    return f"profile context for {unit_text}"


def build_recalled_readout_query(packet, current_store, context) -> str:
    """Return the active query text for template-side sub recall."""

    if packet.query is not None:
        return packet.query.text
    query = context.get("query", {})
    return str(query.get("text", "")).strip()


def build_pipeline() -> MemoryPipeline:
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
    )
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=LLMRepresentation(
            field="summary",
            prompt=text_prompt(
                (
                    "Summarize the incoming memory for later retrieval.\n"
                    "Previously recalled context:\n{{ recalled_prompt }}\n\n"
                    "Current memory:\n{{ unit.text }}"
                ),
                recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
                recall_query_builder=build_recalled_prompt_query,
                sub_recall_pipeline=retrieve_pipeline,
            ),
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
