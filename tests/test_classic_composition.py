from __future__ import annotations

import pytest

from memprimitive import IncompatibleCompositionError, MemoryPipeline
from memprimitive.classic_modules.amem import AMEMGraphHopRetrieval, AMEMGraphOrganization
from memprimitive.classic_modules.memgpt import MEMGPT_CORE_LAYER, MEMGPT_RECALL_LAYER, MemGPTKeyedUpsertOrganization, MemGPTPagedRetrieval
from memprimitive.classic_modules.reflexion import ReflectionMemoryEvolution, ReflexionPrependedReadout
from memprimitive.classic_modules.tim import TimBudgetEvolutionTrigger, TimThoughtMemoryEvolution, TimThoughtMemoryOrganization, TimThoughtMemoryRetrieval
from memprimitive.core import MemoryStore, StoreLayerSpec, StoreTopology

def test_reflexion_requires_declared_reflection_layer() -> None:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="episodes")]))

    with pytest.raises(IncompatibleCompositionError, match="reflections"):
        MemoryPipeline(store=store, memory_evolution=ReflectionMemoryEvolution(target_layer="reflections"))

    with pytest.raises(IncompatibleCompositionError, match="reflections"):
        MemoryPipeline(store=store, readout=ReflexionPrependedReadout(reflection_layer="reflections"))


def test_memgpt_requires_declared_budget_layers() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working_memory", indices=("temporal", "keyword")),
                StoreLayerSpec(name="conversation_queue", indices=("temporal", "keyword")),
            ]
        )
    )

    with pytest.raises(IncompatibleCompositionError, match=MEMGPT_CORE_LAYER):
        MemoryPipeline(
            store=store,
            organization=MemGPTKeyedUpsertOrganization(
                target_layer=MEMGPT_CORE_LAYER,
                key_name="memgpt_key",
            ),
        )

    with pytest.raises(IncompatibleCompositionError, match=MEMGPT_RECALL_LAYER):
        MemoryPipeline(store=store, retrieval=MemGPTPagedRetrieval(target_layer=MEMGPT_RECALL_LAYER, tool_name="conversation_search"))


def test_tim_requires_declared_thought_layer() -> None:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="other")]))

    with pytest.raises(IncompatibleCompositionError, match="thought_memory"):
        MemoryPipeline(store=store, organization=TimThoughtMemoryOrganization())

    with pytest.raises(IncompatibleCompositionError, match="thought_memory"):
        MemoryPipeline(
            store=store,
            evolution_trigger=TimBudgetEvolutionTrigger(),
            memory_evolution=TimThoughtMemoryEvolution(),
        )

    with pytest.raises(IncompatibleCompositionError, match="thought_memory"):
        MemoryPipeline(store=store, retrieval=TimThoughtMemoryRetrieval())


def test_amem_requires_graph_shape_and_index() -> None:
    bad_shape_store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory_graph", shape="Flat", indices=("graph", "vector", "keyword", "tag"))]
        )
    )
    with pytest.raises(IncompatibleCompositionError, match="Graph"):
        MemoryPipeline(store=bad_shape_store, organization=AMEMGraphOrganization())

    bad_index_store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory_graph", shape="Graph", indices=("vector", "keyword", "tag"))]
        )
    )
    with pytest.raises(IncompatibleCompositionError, match="graph"):
        MemoryPipeline(store=bad_index_store, retrieval=AMEMGraphHopRetrieval())
