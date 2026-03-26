"""MemoryBank (Zhong et al., 2024) - motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.memorybank

Or from this directory (script adds the repo root to ``sys.path``)::

    python memorybank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    ConditionalLayerOrganization,
    EmbeddingSimilarityRetrieval,
    GroupedByLayerReadout,
    LayerAwareRetrieval,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.classic_modules.memorybank import (
    MemoryBankConfig,
    MemoryBankEvolution,
    MemoryBankEvolutionTrigger,
    build_memorybank_topology,
)


def build_memorybank_pipeline(
    *,
    config: MemoryBankConfig | None = None,
    store: MemoryStore | None = None,
) -> MemoryPipeline:
    config = config or MemoryBankConfig()
    topology = build_memorybank_topology(config)
    memory_store = store if store is not None else MemoryStore(topology=topology)
    return MemoryPipeline(
        store=memory_store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding", "entities", "tags", "keywords")),
        write_trigger=AlwaysWriteTrigger(),
        organization=ConditionalLayerOrganization(
            default_layer=config.short_term_layer,
            rules=(
                {"has_entity": True, "target_layer": config.long_term_layer},
                {"unit_type": "summary", "target_layer": config.long_term_layer},
            ),
        ),
        evolution_trigger=MemoryBankEvolutionTrigger(
            short_term_layer=config.short_term_layer,
            long_term_layer=config.long_term_layer,
            short_term_window=config.short_term_window,
        ),
        memory_evolution=MemoryBankEvolution(
            short_term_layer=config.short_term_layer,
            long_term_layer=config.long_term_layer,
            short_term_window=config.short_term_window,
            merge_prefix=config.merge_prefix,
            summary_prefix=config.summary_prefix,
        ),
        retrieval=LayerAwareRetrieval(
            default_retriever=RecencyRetrieval(top_k=config.short_term_retrieval_k, layer=config.short_term_layer),
            retriever_by_layer={
                config.short_term_layer: RecencyRetrieval(top_k=config.short_term_retrieval_k, layer=config.short_term_layer),
                config.long_term_layer: EmbeddingSimilarityRetrieval(
                    top_k=config.long_term_retrieval_k,
                    layer=config.long_term_layer,
                ),
            },
            active_layers=(config.short_term_layer, config.long_term_layer),
            top_k=config.combined_retrieval_k,
            top_k_by_layer={
                config.short_term_layer: config.short_term_retrieval_k,
                config.long_term_layer: config.long_term_retrieval_k,
            },
            merge_weight_by_layer={
                config.short_term_layer: 1.1,
                config.long_term_layer: 1.0,
            },
        ),
        readout=GroupedByLayerReadout(),
    )


def main() -> None:
    pipeline = build_memorybank_pipeline(
        config=MemoryBankConfig(short_term_window=2),
    )

    pipeline.ingest(Observation(text="Alice works at OpenAI in San Francisco.", source="dialogue"))
    pipeline.ingest(Observation(text="remember to refill the tea kettle", source="note"))
    pipeline.ingest(Observation(text="capture the next observation for later", source="note"))

    readout = pipeline.recall(Query(text="Alice"))

    print(readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()


__all__ = ["build_memorybank_pipeline"]
