"""Runnable A-MEM-like graph cycle using baseline slot composition.

This demonstration intentionally stays at the ``MemoryPipeline + module
composition`` level instead of hiding the flow behind the classic wrapper. It
shows how the graph-pipeline base, enriched-note representation, LLM-judged
write, neighbor-triggered evolution, seed-and-expand retrieval, and note
readout fit together as explicit slot choices.

The script uses the shared classic runtime, so it requires a real
OpenAI-compatible API configuration via ``MEMPRIMITIVE_API_KEY``,
``MEMPRIMITIVE_BASE_URL``, and ``MEMPRIMITIVE_MODEL``.
"""

from __future__ import annotations

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    GraphAppendLinkReadyOrganization,
    LLMJudgedWriteTrigger,
    LinkStrengtheningEvolution,
    NeighborContextUpdateEvolution,
    NeighborExistsEvolutionTrigger,
    NoteRenderReadout,
    PassThroughUnitFormation,
    RetrievalOrientedEmbeddingRepresentation,
    SemanticFieldEnrichmentRepresentation,
    VectorGraphSeedAndExpandRetrieval,
)
from memprimitive.classic_modules._runtime import get_classic_runtime


def build_pipeline() -> MemoryPipeline:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="memory_graph",
                    theme="knowledge_graph",
                    shape="Graph",
                    indices=("graph", "vector", "entity", "keyword", "tag"),
                )
            ]
        )
    )
    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=(
            SemanticFieldEnrichmentRepresentation(note_namespace="amem"),
            RetrievalOrientedEmbeddingRepresentation(note_namespace="amem"),
        ),
        write_trigger=LLMJudgedWriteTrigger(note_namespace="amem", enabled=True),
        organization=GraphAppendLinkReadyOrganization(target_layer="memory_graph", note_namespace="amem"),
        evolution_trigger=NeighborExistsEvolutionTrigger(target_layer="memory_graph", candidate_top_k=3),
        memory_evolution=(
            LinkStrengtheningEvolution(target_layer="memory_graph", note_namespace="amem"),
            NeighborContextUpdateEvolution(target_layer="memory_graph", note_namespace="amem"),
        ),
        retrieval=VectorGraphSeedAndExpandRetrieval(
            top_k=3,
            layer="memory_graph",
            candidate_k=3,
            neighbor_expansion_k=1,
            note_namespace="amem",
        ),
        readout=NoteRenderReadout(note_namespace="amem"),
    )


def main() -> None:
    get_classic_runtime().require_llm(capability="A-MEM-like graph demonstration")
    pipeline = build_pipeline()
    first = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    second = pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))
    readout = pipeline.recall(Query(text="Alice"))

    print("first evolution decisions:", first.evolution_decisions)
    print("second evolution decisions:", second.evolution_decisions)
    print()
    print(readout.text)
    print()
    print("retrieved ids:", readout.source_ids)


if __name__ == "__main__":
    main()
