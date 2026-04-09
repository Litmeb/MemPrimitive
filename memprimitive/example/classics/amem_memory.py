"""Mechanism-level reconstruction of A-MEM / Agentic Memory.

This example follows the same standards as the other ``classics`` files in
this repository: it aims to preserve the paper/repo memory mechanism inside the
shared MemPrimitive primitive language, not to clone the upstream storage stack
or API byte-for-byte.

For A-MEM, the core loop we want to preserve is:

1. turn a new interaction into one enriched memory note,
2. embed that note for similarity-based retrieval,
3. append it into a graph-capable note store,
4. use nearby notes to strengthen semantic links,
5. optionally rewrite neighbor note context/tags, and
6. answer future queries by embedding-similarity top-k retrieval.

That loop is exactly the level of fidelity this file is trying to demonstrate.
The result is a mechanism-level A-MEM reconstruction expressed entirely with
existing baseline modules.

Important alignment note: the paper text suggests neighbor evolution may update
context, keywords, and tags, while the released repos are more concretely
implemented around context/tag updates and graph linking. This example follows
the repo-consistent write/evolution path, but keeps recall aligned to the paper
itself: plain embedding retrieval over notes rather than graph-neighbor
expansion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    ConfigurableEmbeddingRepresentation,
    EmbeddingSimilarityRetrieval,
    GraphAppendOrganization,
    LinkStrengtheningEvolution,
    NeighborContextUpdateEvolution,
    NoteRenderReadout,
    PassThroughUnitFormation,
    SemanticFieldEnrichmentRepresentation,
)
from memprimitive.utils._template import text_prompt
from memprimitive.utils._runtime import get_runtime


def build_amem_memory_system(
    *,
    note_namespace: str = "amem",
    candidate_k: int = 5,
    neighbor_expansion_k: int = 3,
    recall_top_k: int = 5,
) -> dict[str, object]:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="knowledge_graph",
                theme="semantic",
                shape="Graph",
                indices=("graph", "vector"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=(
            SemanticFieldEnrichmentRepresentation(note_namespace=note_namespace),
            ConfigurableEmbeddingRepresentation(
                embedding_text=text_prompt(
                    f"{{{{ unit.metadata.{note_namespace}.content }}}} | "
                    f"context: {{{{ unit.metadata.{note_namespace}.context }}}} | "
                    f"keywords: {{{{ unit.metadata.{note_namespace}.keywords | join(', ') }}}} | "
                    f"tags: {{{{ unit.metadata.{note_namespace}.tags | join(', ') }}}}"
                )
            ),
        ),
        write_trigger=AlwaysTrigger(),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=(
            LinkStrengtheningEvolution(
                target_layer="knowledge_graph",
                candidate_k=candidate_k,
                note_namespace=note_namespace,
            ),
            NeighborContextUpdateEvolution(
                target_layer="knowledge_graph",
                candidate_k=candidate_k,
                note_namespace=note_namespace,
            ),
        ),
        store=store,
    )

    recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(
            top_k=recall_top_k,
            layer="knowledge_graph",
        ),
        readout=NoteRenderReadout(note_namespace=note_namespace),
        store=store,
    )

    return {
        "store": store,
        "write_pipeline": write_pipeline,
        "recall_pipeline": recall_pipeline,
        "note_namespace": note_namespace,
    }


def ingest_note(
    system: dict[str, object],
    *,
    text: str,
    source: str = "dialogue",
    metadata: dict[str, object] | None = None,
) -> None:
    write_pipeline = system["write_pipeline"]
    assert isinstance(write_pipeline, MemoryPipeline)
    write_pipeline.ingest(
        Observation(
            text=text,
            source=source,
            metadata={} if metadata is None else dict(metadata),
        )
    )


def recall_notes(system: dict[str, object], *, user_query: str) -> str:
    recall_pipeline = system["recall_pipeline"]
    assert isinstance(recall_pipeline, MemoryPipeline)
    return recall_pipeline.recall(
        Query(
            text=user_query,
            embedding=list(get_runtime().embed(user_query)),
        )
    ).text


def main() -> None:
    system = build_amem_memory_system()
    store = system["store"]
    assert isinstance(store, MemoryStore)

    ingest_note(
        system,
        text="Alice likes jasmine tea in the evening and treats it as part of her steady routine.",
    )
    ingest_note(
        system,
        text="Tea routines improve Alice's focus during reflective graph-memory work.",
    )
    ingest_note(
        system,
        text="Alice links her tea habit with graph memory design because it helps her think clearly.",
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("knowledge graph records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "note": record.metadata.get("amem", {}),
                "graph": record.metadata.get("graph", {}),
            }
            for record in store.iter_records("knowledge_graph")
        ]
    )
    print()

    print("agentic recall:")
    print(recall_notes(system, user_query="What should the assistant remember about Alice and tea?"))


if __name__ == "__main__":
    main()
