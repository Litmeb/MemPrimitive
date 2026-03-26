from __future__ import annotations

import pytest

from memprimitive import IncompatibleCompositionError, MemoryPipeline
from memprimitive.classic_modules.amem import AMEMGraphHopRetrieval, AMEMGraphOrganization
from memprimitive.classic_modules.generative_agents import GenerativeAgentsMemoryEvolution, GenerativeAgentsRetrieval
from memprimitive.classic_modules.memgpt import MEMGPT_CORE_LAYER, MEMGPT_RECALL_LAYER, MemGPTKeyedUpsertOrganization, MemGPTPagedRetrieval
from memprimitive.classic_modules.memorybank import MemoryBankEvolution, MemoryBankEvolutionTrigger
from memprimitive.classic_modules.reflexion import ReflectionMemoryEvolution, ReflexionPrependedReadout
from memprimitive.classic_modules.scm import SCMControlledRetrieval, SCMEntityProfileUpsert
from memprimitive.classic_modules.tim import TimBudgetEvolutionTrigger, TimThoughtMemoryEvolution, TimThoughtMemoryOrganization, TimThoughtMemoryRetrieval
from memprimitive.classic_modules.voyager import MixedSkillRetrieval, UpsertByKeySkillLibrary
from memprimitive.core import MemoryStore, StoreLayerSpec, StoreTopology


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def test_reflexion_requires_declared_reflection_layer() -> None:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="episodes")]))

    with pytest.raises(IncompatibleCompositionError, match="reflections"):
        MemoryPipeline(store=store, memory_evolution=ReflectionMemoryEvolution(target_layer="reflections"))

    with pytest.raises(IncompatibleCompositionError, match="reflections"):
        MemoryPipeline(store=store, readout=ReflexionPrependedReadout(reflection_layer="reflections"))


def test_memorybank_requires_declared_layers_and_entity_index() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="short_term", indices=("temporal",)),
                StoreLayerSpec(name="long_term", indices=("keyword",)),
            ]
        )
    )

    with pytest.raises(IncompatibleCompositionError, match="entity index"):
        MemoryPipeline(
            store=store,
            evolution_trigger=MemoryBankEvolutionTrigger(short_term_layer="short_term", long_term_layer="long_term"),
            memory_evolution=MemoryBankEvolution(short_term_layer="short_term", long_term_layer="long_term"),
        )


def test_scm_requires_profile_entity_index() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="semantic", indices=("entity", "keyword")),
                StoreLayerSpec(name="profile", indices=("keyword",)),
            ]
        )
    )

    with pytest.raises(IncompatibleCompositionError, match="profile"):
        MemoryPipeline(store=store, organization=SCMEntityProfileUpsert())

    with pytest.raises(IncompatibleCompositionError, match="profile"):
        MemoryPipeline(store=store, retrieval=SCMControlledRetrieval())


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


def test_generative_agents_requires_declared_reflection_layer() -> None:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="observation_stream", indices=("keyword",))]))

    with pytest.raises(IncompatibleCompositionError, match="reflections"):
        MemoryPipeline(store=store, memory_evolution=GenerativeAgentsMemoryEvolution())

    with pytest.raises(IncompatibleCompositionError, match="reflections"):
        MemoryPipeline(store=store, retrieval=GenerativeAgentsRetrieval(layer="reflections"))


def test_voyager_requires_skill_indices_when_layer_is_declared() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="skill_library", indices=("keyword",))])
    )

    with pytest.raises(IncompatibleCompositionError, match="tag"):
        MemoryPipeline(store=store, organization=UpsertByKeySkillLibrary(allow_create_layer=False))

    with pytest.raises(IncompatibleCompositionError, match="tag"):
        MemoryPipeline(store=store, retrieval=MixedSkillRetrieval())


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
            [StoreLayerSpec(name="memory_graph", shape="Flat", indices=("graph", "entity", "keyword"))]
        )
    )
    with pytest.raises(IncompatibleCompositionError, match="Graph"):
        MemoryPipeline(store=bad_shape_store, organization=AMEMGraphOrganization())

    bad_index_store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory_graph", shape="Graph", indices=("entity", "keyword"))]
        )
    )
    with pytest.raises(IncompatibleCompositionError, match="graph"):
        MemoryPipeline(store=bad_index_store, retrieval=AMEMGraphHopRetrieval())
