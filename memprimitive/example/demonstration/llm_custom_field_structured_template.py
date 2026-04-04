"""End-to-end example: custom ``LLMRepresentation`` fields + structured ``TemplateReadout``.

This demonstration builds the full pipeline in two stages:

- ingest pipeline:
  - writes records into a ``profile`` layer
  - uses ``LLMRepresentation`` to extract custom fields
- recall pipeline:
  - retrieves relevant profile records with embedding similarity
  - renders them through ``TemplateReadout(structured_template=...)``

The template directly reads custom representation fields such as
``item.representation.user_profile`` and ``item.representation.response_hint``.

This script uses real ``LLMRepresentation`` calls, so the LLM runtime should be
configured, for example via:

    MEMPRIMITIVE_API_KEY
    MEMPRIMITIVE_BASE_URL
    MEMPRIMITIVE_MODEL

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.llm_custom_field_structured_template

Or from this directory (script adds the repo root to ``sys.path``)::

    python llm_custom_field_structured_template.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    EmbeddingSimilarityRetrieval,
    LLMRepresentation,
    TemplateReadout,
)


def main() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile", theme="semantic", indices=("temporal", "vector")),
            ]
        )
    )

    writer = MemoryPipeline(
        representation=(
            BasicRepresentation(elements=("text", "embedding")),
            LLMRepresentation(
                field="user_profile",
                prompt=(
                    "Extract a compact {{ field }} note from this {{ unit.unit_type | default('memory') }} unit. "
                    "Source text: {{ unit.text }}\n"
                    "Focus on stable preferences, working style, and enduring interests. "
                    "Return one short paragraph."
                ),
            ),
            LLMRepresentation(
                field="response_hint",
                prompt=(
                    "Extract one concise response hint for a future assistant reply. "
                    "Focus on how to speak, what to emphasize, or what to avoid. "
                    "Return one short sentence."
                ),
            ),
            LLMRepresentation(
                field="keywords",
                prompt="Extract short retrieval keywords for this memory unit.",
            ),
        ),
        organization=AppendOrganization(target_layer="profile"),
        store=store,
    )

    writer.ingest(
        Observation(
            text="Alice prefers concise technical explanations with concrete examples and dislikes vague summaries.",
            source="dialogue",
            metadata={"session_id": "sess-profile"},
        )
    )
    writer.ingest(
        Observation(
            text="Alice is actively building retrieval and readout modules for long-term memory systems.",
            source="notes",
            metadata={"session_id": "sess-profile"},
        )
    )
    writer_packet = writer.ingest(
        Observation(
            text="When discussing design tradeoffs, Alice wants the answer to clearly separate stable architecture from optional extensions.",
            source="dialogue",
            metadata={"session_id": "sess-profile"},
        )
    )

    recall = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=3, layer="profile"),
        readout=TemplateReadout(
            structured_template={
                "blocks": [
                    {
                        "id": "query",
                        "title": "Query",
                        "template": "{{ query.text }}",
                    },
                    {
                        "id": "profile_cards",
                        "title": "Profile Cards",
                        "condition": "retrieved.items | length",
                        "repeat_over": "retrieved.items | sort_by('score.rank')",
                        "item_template": (
                            "- source={{ item.metadata.source | default('unknown') }} | rank={{ item.score.rank | default('n/a') }}\n"
                            "  text={{ item.text }}\n"
                            "  user_profile={{ item.representation.user_profile | default('') }}\n"
                            "  response_hint={{ item.representation.response_hint | default('') }}\n"
                            "  keywords={{ item.representation.keywords | join(', ') }}"
                        ),
                        "separator": "\n\n",
                    },
                    {
                        "id": "rollup",
                        "title": "Readout Summary",
                        "template": (
                            "matched_records={{ retrieved.items | length }}\n"
                            "top_profile={{ retrieved.items | first | default('') }}\n"
                            "retrieval_module={{ trace.retrieval.module }}"
                        ),
                    },
                ]
            }
        ),
        store=store,
    )

    query = Query(text="How should we respond to Alice about retrieval architecture design?")
    retrieval_packet, _ = recall.retrieval.run(Packet(query=query), store)
    readout = recall.recall(query)

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

    print("last ingest representation trace:")
    pprint(writer_packet.trace["representation"])
    print()

    print("stored profile records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "representation": record.metadata.get("representation"),
            }
            for record in store.iter_records("profile")
        ]
    )
    print()

    print("retrieval trace:")
    pprint(retrieval_packet.trace["retrieval"])
    print()

    print("structured template readout:")
    print(readout.text)
    print()

    print("source record ids:")
    pprint(readout.source_ids)
    print()

    print("readout metadata summary:")
    pprint(
        {
            "template_mode": readout.metadata.get("template_mode"),
            "used_record_ids": readout.metadata.get("used_record_ids"),
            "used_group_ids": readout.metadata.get("used_group_ids"),
            "block_trace": readout.metadata.get("block_trace"),
        }
    )


if __name__ == "__main__":
    main()
