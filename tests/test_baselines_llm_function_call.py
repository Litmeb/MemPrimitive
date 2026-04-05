from __future__ import annotations

from dataclasses import replace
from typing import Any
import pytest

from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    Observation,
    Packet,
    StoreLayerSpec,
    StoreTopology,
)

from baselines_test_helpers import (
    _graph_store,
    _invoke_runtime_tool,
    _seed_layer,
)


def test_llm_function_call_organization_adds_record_and_renders_prompt_template() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")])),
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Store {{ unit.text }} in {{ default_target_layer }}",
        tools=["ADD"],
        target_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Store Alice likes tea. in profile"
        _invoke_runtime_tool(tools[0], {"text": "Alice profile note"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    profile_records = store.iter_records("profile")
    assert len(profile_records) == 1
    assert profile_records[0].text == "Alice profile note"
    assert profile_records[0].metadata["llm_tool"]["action"] == "ADD"
    assert packet_out.placements[0].target_layer == "profile"
    assert packet_out.trace["organization"]["written_record_ids"] == [profile_records[0].record_id]
    assert packet_out.trace["organization"]["per_unit"][0]["rendered_prompt"] == "Store Alice likes tea. in profile"


def test_llm_function_call_organization_updates_existing_record() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    store = MemoryStore()
    _seed_layer(store, "default", ["old text"])
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="trigger update", source="notes")),
        store,
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Update memory for {{ unit.text }}",
        tools=["UPDATE"],
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "trigger update" in rendered_prompt
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "text": "new text"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    assert store.iter_records("default")[0].text == "new text"
    assert packet_out.trace["organization"]["updated_record_ids"] == ["rec-1"]
    assert packet_out.trace["organization"]["effects"][0]["action"] == "update"


def test_llm_function_call_organization_deletes_existing_record() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    store = MemoryStore()
    _seed_layer(store, "default", ["delete me"])
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="trigger delete", source="notes")),
        store,
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Delete stale memory for {{ unit.text }}",
        tools=["DELETE"],
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "reason": "stale"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    assert store.iter_records("default") == []
    assert packet_out.trace["organization"]["deleted_record_ids"] == ["rec-1"]
    assert packet_out.trace["organization"]["tool_calls"][0]["status"] == "applied"


def test_llm_function_call_organization_rejects_unknown_tool_name() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization

    with pytest.raises(ValueError, match="Unknown built-in write tool"):
        LLMFunctionCallOrganization(prompt="x", tools=["ADD", "UNKNOWN"])


def test_llm_function_call_graph_organization_adds_normalized_graph_record() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation, TripleRepresentation
    from memprimitive.utils._graph_family import graph_metadata_from_record

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            return represented, {"source": "test_seed", "entities": entities, "triple_count": len(triples)}

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Store graph memory for {{ unit.text }}",
        tools=["GRAPH_ADD"],
        target_layer="knowledge_graph",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"text": "Alice likes tea."})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    record = store.iter_records("knowledge_graph")[0]
    graph = graph_metadata_from_record(record)
    assert graph["layer"] == "knowledge_graph"
    assert graph["shape"] == "node"
    assert graph["entities"] == ["Alice", "tea"]
    assert graph["triples"] == [("Alice", "likes", "tea")]
    assert graph["links"] == []
    assert graph["link_count"] == 0
    assert record.metadata["llm_tool"]["action"] == "GRAPH_ADD"
    assert packet_out.trace["organization"]["written_record_ids"] == [record.record_id]


def test_llm_function_call_organization_graph_tools_declare_graph_contracts() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization
    from memprimitive.contracts import RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT

    graph_module = LLMFunctionCallOrganization(prompt="x", tools=["GRAPH_ADD"], target_layer="knowledge_graph")
    plain_module = LLMFunctionCallOrganization(prompt="x", tools=["ADD"], target_layer="default")

    assert graph_module.get_requires_contracts() == frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    assert graph_module.get_produces_contracts() == frozenset({RECORD_GRAPH_LINKS_CONTRACT})
    assert plain_module.get_requires_contracts() == frozenset()
    assert plain_module.get_produces_contracts() == frozenset()


def test_llm_function_call_evolution_updates_selected_records_from_decisions_store() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["old profile"])
    module = LLMFunctionCallEvolution(
        prompt="Rewrite {{ selected_records.0.text }}",
        tools=["UPDATE"],
        source_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Rewrite old profile"
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "text": "new profile"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(
        Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}),
        store,
    )

    assert store.iter_records("profile")[0].text == "new profile"
    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions_store"
    assert packet_out.trace["memory_evolution"]["selected_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["updated_record_ids"] == ["rec-1"]


def test_llm_function_call_evolution_deletes_records_from_source_layer_scan() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["delete profile"])
    module = LLMFunctionCallEvolution(
        prompt="Delete {{ selected_records.0.text }}",
        tools=["DELETE"],
        source_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "reason": "cleanup"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(), store)

    assert store.iter_records("profile") == []
    assert packet_out.trace["memory_evolution"]["decision_source"] == "source_layer_scan"
    assert packet_out.trace["memory_evolution"]["deleted_record_ids"] == ["rec-1"]


def test_llm_function_call_evolution_update_rejects_record_outside_selected_records() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["selected profile", "other profile"])
    module = LLMFunctionCallEvolution(
        prompt="Rewrite {{ selected_records.0.text }}",
        tools=["UPDATE"],
        source_layer="profile",
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Rewrite selected profile"
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "should fail"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="Record 'rec-2' is not in the current evolution candidate set."):
        module.run(
            Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}),
            store,
        )


def test_llm_function_call_evolution_delete_rejects_record_outside_selected_records() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["selected profile", "other profile"])
    module = LLMFunctionCallEvolution(
        prompt="Delete {{ selected_records.0.text }}",
        tools=["DELETE"],
        source_layer="profile",
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Delete selected profile"
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "reason": "should fail"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="Record 'rec-2' is not in the current evolution candidate set."):
        module.run(
            Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}),
            store,
        )


def test_llm_function_call_evolution_graph_update_normalizes_graph_metadata() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution
    from memprimitive.utils._graph_family import graph_metadata_from_record

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "links": [], "triples": []}},
        )
    )
    module = LLMFunctionCallEvolution(
        prompt="Rewrite {{ selected_records.0.text }}",
        tools=["GRAPH_UPDATE"],
        source_layer="knowledge_graph",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(
            tools[0],
            {
                "record_id": "rec-1",
                "text": "Alice likes green tea",
                "metadata_patch": {
                    "graph": {
                        "entities": ["Alice", "green tea"],
                        "triples": [["Alice", "likes", "green tea"]],
                        "links": ["rec-99"],
                    }
                },
            },
        )
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(), store)

    record = store.iter_records("knowledge_graph")[0]
    graph = graph_metadata_from_record(record)
    assert record.text == "Alice likes green tea"
    assert graph["layer"] == "knowledge_graph"
    assert graph["entities"] == ["Alice", "green tea"]
    assert graph["triples"] == [("Alice", "likes", "green tea")]
    assert graph["links"] == ["rec-99"]
    assert graph["link_count"] == 1
    assert packet_out.trace["memory_evolution"]["updated_record_ids"] == ["rec-1"]


def test_llm_function_call_evolution_graph_delete_cleans_dangling_links() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution
    from memprimitive.utils._graph_family import graph_metadata_from_record

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "links": ["rec-2"], "triples": []}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="seed-2",
            layer="knowledge_graph",
            text="Tea is warm",
            timestamp="2026-01-01T00:00:02Z",
            metadata={"graph": {"entities": ["tea"], "links": [], "triples": []}},
        )
    )
    module = LLMFunctionCallEvolution(
        prompt="Delete {{ selected_records.1.text }}",
        tools=["GRAPH_DELETE"],
        source_layer="knowledge_graph",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "reason": "cleanup"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(), store)

    remaining = store.iter_records("knowledge_graph")
    assert [record.record_id for record in remaining] == ["rec-1"]
    assert graph_metadata_from_record(remaining[0])["links"] == []
    assert graph_metadata_from_record(remaining[0])["link_count"] == 0
    assert packet_out.trace["memory_evolution"]["deleted_record_ids"] == ["rec-2"]
    assert packet_out.trace["memory_evolution"]["updated_record_ids"] == ["rec-1"]
    assert any(
        effect.get("effect_type") == "graph_link_cleanup"
        for effect in packet_out.trace["memory_evolution"]["effects"]
    )


def test_llm_function_call_evolution_graph_delete_rejects_record_outside_selected_records() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "links": [], "triples": []}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="seed-2",
            layer="knowledge_graph",
            text="Tea is warm",
            timestamp="2026-01-01T00:00:02Z",
            metadata={"graph": {"entities": ["tea"], "links": [], "triples": []}},
        )
    )
    module = LLMFunctionCallEvolution(
        prompt="Delete {{ selected_records.0.text }}",
        tools=["GRAPH_DELETE"],
        source_layer="knowledge_graph",
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Delete Alice likes tea"
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "reason": "should fail"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="Record 'rec-2' is not in the current evolution candidate set."):
        module.run(
            Packet(decisions_store={"knowledge_graph": {"record_ids": ["rec-1"]}}),
            store,
        )


def test_llm_function_call_organization_graph_add_link_only_updates_links() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation
    from memprimitive.utils._graph_family import graph_metadata_from_record

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "triples": [["Alice", "likes", "tea"]], "links": []}},
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="link Alice to Bob", source="dialogue")),
        store,
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Patch graph links for {{ unit.text }}",
        tools=["GRAPH_ADD_LINK"],
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "links": ["rec-2", "rec-3", "rec-2"]})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    record = store.iter_records("knowledge_graph")[0]
    graph = graph_metadata_from_record(record)
    assert record.text == "Alice likes tea"
    assert graph["entities"] == ["Alice"]
    assert graph["triples"] == [("Alice", "likes", "tea")]
    assert graph["links"] == ["rec-2", "rec-3"]
    assert graph["link_count"] == 2
    assert record.metadata["llm_tool"]["action"] == "GRAPH_ADD_LINK"
    assert packet_out.trace["organization"]["updated_record_ids"] == ["rec-1"]
    assert packet_out.trace["organization"]["effects"][0]["added_links"] == ["rec-2", "rec-3"]


def test_llm_function_call_organization_graph_link_tool_rejects_non_graph_layer() -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    store = MemoryStore()
    _seed_layer(store, "default", ["plain note"])
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="try graph link", source="dialogue")),
        store,
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Patch links",
        tools=["GRAPH_ADD_LINK"],
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "links": ["rec-2"]})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="GRAPH_ADD_LINK requires target layer 'default' to be Graph."):
        module.run(packet, store)


def test_llm_function_call_evolution_graph_add_link_rejects_record_outside_selected_records() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "links": [], "triples": []}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="seed-2",
            layer="knowledge_graph",
            text="Bob likes coffee",
            timestamp="2026-01-01T00:00:02Z",
            metadata={"graph": {"entities": ["Bob"], "links": [], "triples": []}},
        )
    )
    module = LLMFunctionCallEvolution(
        prompt="Patch links for {{ selected_records.0.text }}",
        tools=["GRAPH_ADD_LINK"],
        source_layer="knowledge_graph",
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Patch links for Alice likes tea"
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "links": ["rec-9"]})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="Record 'rec-2' is not in the current evolution candidate set."):
        module.run(
            Packet(decisions_store={"knowledge_graph": {"record_ids": ["rec-1"]}}),
            store,
        )


def test_llm_function_call_evolution_graph_update_link_replaces_links() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution
    from memprimitive.utils._graph_family import graph_metadata_from_record

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "links": ["rec-2"], "triples": []}},
        )
    )
    module = LLMFunctionCallEvolution(
        prompt="Replace links for {{ selected_records.0.text }}",
        tools=["GRAPH_UPDATE_LINK"],
        source_layer="knowledge_graph",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "links": ["rec-3", "rec-4"]})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(), store)

    record = store.iter_records("knowledge_graph")[0]
    graph = graph_metadata_from_record(record)
    assert graph["links"] == ["rec-3", "rec-4"]
    assert graph["link_count"] == 2
    assert packet_out.trace["memory_evolution"]["updated_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["effects"][0]["previous_links"] == ["rec-2"]
    assert packet_out.trace["memory_evolution"]["effects"][0]["current_links"] == ["rec-3", "rec-4"]
    assert record.metadata["llm_tool"]["action"] == "GRAPH_UPDATE_LINK"


def test_llm_function_call_evolution_graph_delete_link_removes_only_requested_links() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution
    from memprimitive.utils._graph_family import graph_metadata_from_record

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="seed-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"graph": {"entities": ["Alice"], "links": ["rec-2", "rec-3"], "triples": []}},
        )
    )
    module = LLMFunctionCallEvolution(
        prompt="Delete links for {{ selected_records.0.text }}",
        tools=["GRAPH_DELETE_LINK"],
        source_layer="knowledge_graph",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "links": ["rec-2", "rec-9"]})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(), store)

    record = store.iter_records("knowledge_graph")[0]
    graph = graph_metadata_from_record(record)
    assert graph["links"] == ["rec-3"]
    assert graph["link_count"] == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["removed_links"] == ["rec-2"]
    assert record.metadata["llm_tool"]["action"] == "GRAPH_DELETE_LINK"


def test_llm_function_call_evolution_graph_tools_declare_graph_contracts() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution
    from memprimitive.contracts import RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT

    graph_module = LLMFunctionCallEvolution(prompt="x", tools=["GRAPH_DELETE"], source_layer="knowledge_graph")
    plain_module = LLMFunctionCallEvolution(prompt="x", tools=["DELETE"], source_layer="default")

    assert graph_module.get_requires_contracts() == frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    assert graph_module.get_produces_contracts() == frozenset({RECORD_GRAPH_LINKS_CONTRACT})
    assert plain_module.get_requires_contracts() == frozenset()
    assert plain_module.get_produces_contracts() == frozenset()


def test_llm_function_call_graph_link_tools_declare_graph_contracts() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution, LLMFunctionCallOrganization
    from memprimitive.contracts import RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT

    graph_org = LLMFunctionCallOrganization(prompt="x", tools=["GRAPH_ADD_LINK"])
    graph_evo = LLMFunctionCallEvolution(prompt="x", tools=["GRAPH_DELETE_LINK"], source_layer="knowledge_graph")

    assert graph_org.get_requires_contracts() == frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    assert graph_org.get_produces_contracts() == frozenset({RECORD_GRAPH_LINKS_CONTRACT})
    assert graph_evo.get_requires_contracts() == frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    assert graph_evo.get_produces_contracts() == frozenset({RECORD_GRAPH_LINKS_CONTRACT})


def test_llm_function_call_evolution_rejects_missing_tool_calls_when_required() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["old profile"])
    module = LLMFunctionCallEvolution(
        prompt="Maybe do nothing",
        tools=["UPDATE"],
        source_layer="profile",
        allow_no_tool_call=False,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        return "NO_ACTION"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="at least one successful or attempted tool call"):
        module.run(Packet(), store)


def test_llm_function_call_custom_tool_executes_and_appears_in_trace() -> None:
    from memprimitive import WriteToolCallContext, WriteToolResult, WriteToolSpec
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    def _custom_executor(context: WriteToolCallContext, arguments: dict[str, Any]) -> WriteToolResult:
        record = MemoryRecord(
            record_id=f"rec-{context.store.next_sequence_id()}",
            unit_id="custom-unit",
            layer="profile",
            text=str(arguments["message"]),
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"custom": True},
        )
        context.store.append(record)
        return WriteToolResult(
            effects=[{"action": "add", "record_id": record.record_id, "layer": "profile", "status": "applied"}],
            store=context.store,
        )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="custom tool", source="notes")),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")])),
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(
        prompt="Use custom tool for {{ unit.text }}",
        tools=[
            WriteToolSpec(
                name="CUSTOM_ADD",
                description="Write one custom profile record.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
                executor=_custom_executor,
            )
        ],
        target_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"message": "custom record"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    assert store.iter_records("profile")[0].text == "custom record"
    assert packet_out.trace["organization"]["tool_calls"][0]["tool_name"] == "CUSTOM_ADD"
    assert packet_out.trace["organization"]["effects"][0]["action"] == "add"


def test_llm_function_call_custom_tool_strict_schema_validation_raises() -> None:
    from memprimitive import WriteToolResult, WriteToolSpec
    from memprimitive.baselines import LLMFunctionCallEvolution

    module = LLMFunctionCallEvolution(
        prompt="Use custom tool",
        tools=[
            WriteToolSpec(
                name="CUSTOM",
                description="Needs a message string.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
                executor=lambda context, arguments: WriteToolResult(effects=[], store=context.store),
            )
        ],
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"message": 123})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="must have JSON type string"):
        module.run(Packet(), MemoryStore())

