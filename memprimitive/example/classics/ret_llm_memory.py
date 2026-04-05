"""Mechanism-level reconstruction of RET-LLM / MemLLM-style explicit memory.

This file assembles the current MemPrimitive baseline surface into a practical
RET-LLM-style system:

1. informative text is split into sentences,
2. each sentence is converted into relation triples,
3. the triples are stored in an explicit external memory layer, and
4. answer generation can issue mid-decoding ``MEM_READ`` calls that run an
   exact structured triple lookup before continuing the final response.

This is a mechanism-level reconstruction, not a training-faithful clone of the
paper's later MemLLM setup. In particular, the memory here is still backed by
ordinary `MemoryRecord`s rather than separate entity / relation tables with
paper-style thresholded candidate retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BM25Retrieval,
    MidDecodingMemoryReadout,
    SentenceSplitUnitFormation,
    TemplateReadout,
    TripleExactMatchRetrieval,
    TripleRepresentation,
)
from memprimitive.utils._template import structured_prompt, text_prompt


@dataclass(slots=True)
class RETLLMSystem:
    store: MemoryStore
    write_pipeline: MemoryPipeline
    mem_read_pipeline: MemoryPipeline
    answer_pipeline: MemoryPipeline
    answer_readout: MidDecodingMemoryReadout

    def memorize(self, text: str, *, source: str = "document") -> None:
        self.write_pipeline.ingest(Observation(text=text, source=source))

    def mem_read(self, structured_query: str) -> str:
        return self.mem_read_pipeline.recall(Query(text=structured_query)).text

    def answer(self, question: str) -> str:
        return self.answer_pipeline.recall(Query(text=question)).text


def build_ret_llm_memory_system(
    *,
    prefetch_top_k: int = 4,
    exact_top_k: int = 6,
    max_turns: int = 6,
) -> RETLLMSystem:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="triple_memory",
                theme="semantic",
                shape="Flat",
                indices=("keyword", "entity", "vector"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        unit_formation=SentenceSplitUnitFormation(),
        representation=TripleRepresentation(
            method="two_stage",
            prompt=text_prompt(
                "Extract grounded knowledge triples from the sentence.\n"
                "Keep only facts explicitly supported by the sentence.\n"
                "Use canonical entity names where possible.\n"
                "Use short relation phrases.\n"
                "Do not infer missing facts.\n\n"
                "Sentence:\n{{ unit.text }}"
            ),
        ),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="triple_memory"),
        store=store,
    )

    mem_read_pipeline = MemoryPipeline(
        retrieval=TripleExactMatchRetrieval(top_k=exact_top_k, layer="triple_memory"),
        readout=TemplateReadout(
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "matches",
                            "title": "Matched Memory Facts",
                            "condition": "retrieved.items | length",
                            "repeat_over": "retrieved.items",
                            "item_template": (
                                "- record_id={{ item.record_id }} | "
                                "matched_triples={{ item.score.matched_triples }} | "
                                "source_text={{ item.text }}"
                            ),
                            "separator": "\n",
                        },
                    ]
                }
            )
        ),
        store=store,
    )

    answer_readout = MidDecodingMemoryReadout(
        prompt=structured_prompt(
            {
                "blocks": [
                    {
                        "id": "task",
                        "title": "Task",
                        "template": (
                            "You are answering with a RET-LLM-style explicit memory.\n"
                            "You may call MEM_READ one or more times during reasoning.\n"
                            "Each MEM_READ query must use the exact format "
                            "'subject >> relation >> object'.\n"
                            "Use '*' for one wildcard side when needed, but keep relation explicit.\n"
                            "If memory returns no match, say the memory does not contain the needed fact "
                            "instead of inventing one.\n"
                            "When memory does return a match, ground the final answer in those facts."
                        ),
                    },
                    {
                        "id": "question",
                        "title": "Question",
                        "template": "{{ query.text }}",
                    },
                    {
                        "id": "prefetched_candidates",
                        "title": "Prefetched Candidate Memories",
                        "condition": "retrieved.items | length",
                        "repeat_over": "retrieved.items",
                        "item_template": (
                            "- record_id={{ item.record_id }} | "
                            "text={{ item.text }} | "
                            "triples={{ item.representation.triples }}"
                        ),
                        "separator": "\n",
                    },
                    {
                        "id": "tools",
                        "title": "Available Tools",
                        "repeat_over": "tools",
                        "item_template": "- {{ item.name }}: {{ item.description }}",
                        "separator": "\n",
                    },
                ]
            }
        ),
        retrieve_pipeline=mem_read_pipeline,
        max_turns=max_turns,
        allow_no_tool_call=True,
    )
    answer_pipeline = MemoryPipeline(
        retrieval=BM25Retrieval(top_k=prefetch_top_k, layer="triple_memory"),
        readout=answer_readout,
        store=store,
    )

    return RETLLMSystem(
        store=store,
        write_pipeline=write_pipeline,
        mem_read_pipeline=mem_read_pipeline,
        answer_pipeline=answer_pipeline,
        answer_readout=answer_readout,
    )


def main() -> None:
    system = build_ret_llm_memory_system()

    system.memorize(
        "Washington D.C. is the capital of the United States. "
        "Marie Curie discovered polonium with Pierre Curie. "
        "The album Alla Mia Eta contains the song Il Regalo Piu Grande."
    )

    print("records per layer:")
    pprint({name: system.store.count(name) for name in system.store.topology.layer_names})
    print()

    print("stored memory records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "triples": record.metadata.get("representation", {}).get("triples", []),
            }
            for record in system.store.iter_records("triple_memory")
        ]
    )
    print()

    print("exact memory read:")
    print(system.mem_read("Washington D.C. >> capital of >> *"))
    print()

    print("mid-decoding answer:")
    print(system.answer("What is the capital of the United States?"))


if __name__ == "__main__":
    main()
