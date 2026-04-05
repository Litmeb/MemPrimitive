"""Demonstration: mid-decoding memory read during readout.

This example keeps the pipeline shape unchanged: normal retrieval runs first,
then ``MidDecodingMemoryReadout`` uses ``MEM_READ`` during answer generation.

Usage:

    python -m memprimitive.example.demonstration.mid_decoding_memory_readout
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    KeywordCountRetrieval,
    MidDecodingMemoryReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.core import MemoryStore
from memprimitive.utils._template import text_prompt


def build_pipeline() -> tuple[MemoryPipeline, MidDecodingMemoryReadout]:
    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=2, layer="profile"),
        readout=ConcatenateReadout(),
    )
    readout = MidDecodingMemoryReadout(
        prompt=text_prompt(
            (
                "Answer the user's question.\n"
                "Question: {{ query.text }}\n"
                "You may call MEM_READ if you need more memory before finishing."
            )
        ),
        retrieve_pipeline=retrieve_pipeline,
    )
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="profile"),
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=readout,
        store=MemoryStore(
            topology=StoreTopology.from_layers(
                [
                    StoreLayerSpec(name="profile"),
                ]
            )
        ),
    )
    return pipeline, readout


def main() -> None:
    pipeline, readout_module = build_pipeline()

    pipeline.ingest(Observation(text="Alice likes concise technical answers.", source="dialogue"))
    pipeline.ingest(Observation(text="Alice often asks for retrieval details and source provenance.", source="dialogue"))

    def _fake_run_agent(self, *, rendered_prompt: str, tools, context: dict[str, object]) -> str:
        tool_output = asyncio.run(
            tools[0].on_invoke_tool(
                None,
                json.dumps({"query": "Alice concise provenance"}, ensure_ascii=False),
            )
        )
        payload = json.loads(tool_output)
        return (
            "Final answer:\n"
            f"{payload['memory_text']}\n"
            f"(memory sources: {payload['source_ids']})"
        )

    readout_module._run_agent = _fake_run_agent.__get__(readout_module, type(readout_module))  # type: ignore[method-assign]

    readout = pipeline.recall(Query(text="What should I remember about Alice when replying?"))

    print(readout.text)
    print()
    print("metadata:")
    pprint(readout.metadata)


if __name__ == "__main__":
    main()
