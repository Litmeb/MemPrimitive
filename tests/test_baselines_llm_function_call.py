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
from memprimitive.utils import _runtime as runtime_module

from baselines_test_helpers import (
    _graph_store,
    _invoke_runtime_tool,
    _seed_layer,
)


class _FakeEmbeddingRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        normalized = str(text).strip()
        self.calls.append(normalized)
        return [float(len(normalized)), float(len(normalized.split()))]


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
    original_store = store
    packet_out, store = module.run(packet, store)

    assert store is original_store
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


def test_llm_function_call_tools_use_store_managed_embedding_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    fake_runtime = _FakeEmbeddingRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(
                    name="profile",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                ),
            ]
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="trigger managed embedding", source="notes")),
        store,
    )
    packet = replace(packet, decisions=[True])
    module = LLMFunctionCallOrganization(prompt="Write profile memory", tools=["ADD"], target_layer="profile")

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"text": "Alice profile note"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    record = store.iter_records("profile")[0]
    assert record.embedding == [18.0, 3.0]
    assert fake_runtime.calls == ["Alice profile note"]
    assert packet_out.trace["organization"]["written_record_ids"] == [record.record_id]


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
    original_store = store
    packet_out, store = module.run(
        Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}),
        store,
    )

    assert store is original_store
    assert store.iter_records("profile")[0].text == "new profile"
    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions_store"
    assert packet_out.trace["memory_evolution"]["selected_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["updated_record_ids"] == ["rec-1"]


def test_llm_function_call_evolution_add_preserves_shared_store_identity() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    module = LLMFunctionCallEvolution(
        prompt="Add a profile memory",
        tools=["ADD"],
        target_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"text": "Caroline attended an LGBTQ support group on 7 May 2023."})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, returned_store = module.run(Packet(), store)

    assert returned_store is store
    assert [record.text for record in store.iter_records("profile")] == [
        "Caroline attended an LGBTQ support group on 7 May 2023."
    ]
    assert packet_out.trace["memory_evolution"]["written_record_ids"] == ["rec-1"]


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


def test_llm_function_call_evolution_default_tool_failure_is_recorded_without_raising() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["selected profile", "other profile"])
    module = LLMFunctionCallEvolution(prompt="Try invalid update", tools=["UPDATE"])

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "should be skipped"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, updated_store = module.run(Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}), store)

    assert [record.text for record in updated_store.iter_records("profile")] == ["selected profile", "other profile"]
    trace = packet_out.trace["memory_evolution"]
    assert trace["raise_on_tool_error"] is False
    assert trace["retry_count"] == 0
    assert trace["tool_calls"][0]["status"] == "failed"
    assert "current evolution candidate set" in trace["failed_tool_calls"][0]["error"]


def test_llm_function_call_evolution_raise_on_tool_error_raises_after_retry_exhaustion() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["selected profile", "other profile"])
    module = LLMFunctionCallEvolution(
        prompt="Try invalid update",
        tools=["UPDATE"],
        max_retry=1,
        raise_on_tool_error=True,
    )
    calls: list[dict[str, Any]] = []

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        calls.append(context)
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "should fail"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="LLMFunctionCallEvolution tool calls failed"):
        module.run(Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}), store)

    assert len(calls) == 2
    assert calls[1]["retry"]["previous_failed_tool_calls"][0]["tool_name"] == "UPDATE"
    assert [record.text for record in store.iter_records("profile")] == ["selected profile", "other profile"]


def test_llm_function_call_evolution_max_retry_reruns_agent_and_commits_only_final_attempt() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["selected profile", "other profile"])
    module = LLMFunctionCallEvolution(prompt="Retry update", tools=["UPDATE"], max_retry=1)
    calls: list[dict[str, Any]] = []

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        calls.append(context)
        if len(calls) == 1:
            _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "text": "partial update"})
            _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "invalid update"})
            return "FAILED"
        assert context["retry"]["previous_failed_tool_calls"][0]["tool_name"] == "UPDATE"
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "text": "final update"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, updated_store = module.run(Packet(decisions_store={"profile": {"record_ids": ["rec-1"]}}), store)

    assert [record.text for record in updated_store.iter_records("profile")] == ["final update", "other profile"]
    assert len(calls) == 2
    trace = packet_out.trace["memory_evolution"]
    assert trace["retry_count"] == 1
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["effects"][0]["record_id"] == "rec-1"
    assert trace["effects"] == trace["attempts"][1]["effects"]


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


def test_llm_function_call_organization_records_prompt_recall_visible_scope() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMFunctionCallOrganization, PassThroughUnitFormation, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import structured_prompt, text_prompt

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(store, "default", ["general note"])
    _seed_layer(store, "profile", ["recalled profile"])
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="trigger recall-scoped update", source="notes")),
        store,
    )
    packet = replace(packet, decisions=[True])
    recall_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=store,
    )
    module = LLMFunctionCallOrganization(
        prompt=structured_prompt(
            {
                "blocks": [
                    {"id": "task", "title": "Task", "template": "Update the recalled profile memory if needed."},
                    {"id": "recalled", "title": "Recalled", "template": "{{ candidate }}"},
                ]
            },
            labeled_recall_plans={"candidate": text_prompt("{{ candidate }}")},
            labeled_sub_recall_pipelines={"candidate": recall_pipeline},
            labeled_recall_query_builders={"candidate": lambda packet, store, context: "profile lookup"},
            visible_record_recall_labels=("candidate",),
        ),
        tools=["UPDATE"],
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "recalled profile" in rendered_prompt
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "updated recalled profile"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(packet, store)

    assert store.iter_records("profile")[0].text == "updated recalled profile"
    per_unit = packet_out.trace["organization"]["per_unit"][0]
    assert per_unit["retrieved_record_ids_by_label"] == {"candidate": ["rec-2"]}
    assert per_unit["visible_record_ids_by_label"] == {"candidate": ["rec-2"]}
    assert per_unit["visible_record_source"] == "store_plus_prompt_recall"


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


def test_llm_function_call_evolution_update_allows_prompt_recalled_record_outside_selected_records() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMFunctionCallEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import structured_prompt, text_prompt

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="selected"), StoreLayerSpec(name="profile")]))
    _seed_layer(store, "selected", ["selected record"])
    _seed_layer(store, "profile", ["recalled profile"])
    recall_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=store,
    )
    module = LLMFunctionCallEvolution(
        prompt=structured_prompt(
            {
                "blocks": [
                    {"id": "task", "title": "Task", "template": "Use the recalled candidate when needed."},
                    {"id": "candidate", "title": "Candidate", "template": "{{ candidate }}"},
                ]
            },
            labeled_recall_plans={"candidate": text_prompt("{{ candidate }}")},
            labeled_sub_recall_pipelines={"candidate": recall_pipeline},
            labeled_recall_query_builders={"candidate": lambda packet, store, context: "profile recall"},
            visible_record_recall_labels=("candidate",),
        ),
        tools=["UPDATE"],
        source_layer="selected",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "recalled profile" in rendered_prompt
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "updated via prompt recall"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(decisions_store={"selected": {"record_ids": ["rec-1"]}}), store)

    assert store.iter_records("profile")[0].text == "updated via prompt recall"
    assert packet_out.trace["memory_evolution"]["selected_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["visible_record_ids"] == ["rec-1", "rec-2"]
    assert packet_out.trace["memory_evolution"]["visible_record_source"] == "selected_records_plus_prompt_recall"


def test_llm_function_call_evolution_delete_allows_prompt_recalled_record_outside_selected_records() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMFunctionCallEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import structured_prompt, text_prompt

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="selected"), StoreLayerSpec(name="profile")]))
    _seed_layer(store, "selected", ["selected record"])
    _seed_layer(store, "profile", ["delete recalled profile"])
    recall_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=store,
    )
    module = LLMFunctionCallEvolution(
        prompt=structured_prompt(
            {
                "blocks": [
                    {"id": "task", "title": "Task", "template": "Delete the recalled candidate if needed."},
                    {"id": "candidate", "title": "Candidate", "template": "{{ candidate }}"},
                ]
            },
            labeled_recall_plans={"candidate": text_prompt("{{ candidate }}")},
            labeled_sub_recall_pipelines={"candidate": recall_pipeline},
            labeled_recall_query_builders={"candidate": lambda packet, store, context: "profile recall"},
            visible_record_recall_labels=("candidate",),
        ),
        tools=["DELETE"],
        source_layer="selected",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert "delete recalled profile" in rendered_prompt
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "reason": "prompt recalled candidate"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(decisions_store={"selected": {"record_ids": ["rec-1"]}}), store)

    assert store.iter_records("profile") == []
    assert packet_out.trace["memory_evolution"]["deleted_record_ids"] == ["rec-2"]


def test_llm_function_call_evolution_prompt_recall_default_visibility_includes_primary_and_labeled_branches() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMFunctionCallEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import PRIMARY_RECALL_LABEL, structured_prompt, text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="primary_layer"),
                StoreLayerSpec(name="alpha_layer"),
                StoreLayerSpec(name="beta_layer"),
            ]
        )
    )
    _seed_layer(store, "primary_layer", ["primary visible"])
    _seed_layer(store, "alpha_layer", ["alpha visible"])
    _seed_layer(store, "beta_layer", ["beta visible"])
    primary_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="primary_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    alpha_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="alpha_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    beta_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="beta_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    module = LLMFunctionCallEvolution(
        prompt=structured_prompt(
            {
                "blocks": [
                    {"id": "task", "title": "Task", "template": "All recalled branches should be visible."},
                    {"id": "primary", "title": "Primary", "template": "{{ recalled_prompt }}"},
                    {"id": "alpha", "title": "Alpha", "template": "{{ alpha }}"},
                    {"id": "beta", "title": "Beta", "template": "{{ beta }}"},
                ]
            },
            recall_plan=text_prompt("{{ recalled_prompt }}"),
            recall_query_builder=lambda packet, store, context: "primary",
            sub_recall_pipeline=primary_pipeline,
            labeled_recall_plans={"alpha": text_prompt("{{ alpha }}"), "beta": text_prompt("{{ beta }}")},
            labeled_sub_recall_pipelines={"alpha": alpha_pipeline, "beta": beta_pipeline},
            labeled_recall_query_builders={
                "alpha": lambda packet, store, context: "alpha",
                "beta": lambda packet, store, context: "beta",
            },
        ),
        tools=["UPDATE"],
        source_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-2", "text": "alpha updated"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(decisions_store={"profile": {"record_ids": []}}), store)

    assert store.iter_records("alpha_layer")[0].text == "alpha updated"
    assert packet_out.trace["memory_evolution"]["visible_record_ids"] == ["rec-1", "rec-2", "rec-3"]
    assert packet_out.trace["memory_evolution"]["prompt_trace"]["retrieved_record_ids_by_label"] == {
        PRIMARY_RECALL_LABEL: ["rec-1"],
        "alpha": ["rec-2"],
        "beta": ["rec-3"],
    }


def test_llm_function_call_evolution_prompt_recall_visibility_can_exclude_primary_and_other_labels() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMFunctionCallEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import structured_prompt, text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="primary_layer"),
                StoreLayerSpec(name="alpha_layer"),
                StoreLayerSpec(name="beta_layer"),
            ]
        )
    )
    _seed_layer(store, "primary_layer", ["primary hidden"])
    _seed_layer(store, "alpha_layer", ["alpha hidden"])
    _seed_layer(store, "beta_layer", ["beta visible"])
    primary_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="primary_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    alpha_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="alpha_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    beta_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="beta_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    module = LLMFunctionCallEvolution(
        prompt=text_prompt(
            "Use only beta recall for visibility.",
            recall_plan=text_prompt("{{ recalled_prompt }}"),
            recall_query_builder=lambda packet, store, context: "primary",
            sub_recall_pipeline=primary_pipeline,
            labeled_recall_plans={"alpha": text_prompt("{{ alpha }}"), "beta": text_prompt("{{ beta }}")},
            labeled_sub_recall_pipelines={"alpha": alpha_pipeline, "beta": beta_pipeline},
            labeled_recall_query_builders={
                "alpha": lambda packet, store, context: "alpha",
                "beta": lambda packet, store, context: "beta",
            },
            visible_record_recall_labels=("beta",),
        ),
        tools=["UPDATE"],
        source_layer="profile",
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "text": "should fail"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="Record 'rec-1' is not in the current evolution candidate set."):
        module.run(Packet(decisions_store={"profile": {"record_ids": []}}), store)


def test_llm_function_call_evolution_prompt_recall_visibility_can_include_primary_only() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMFunctionCallEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import PRIMARY_RECALL_LABEL, structured_prompt, text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="primary_layer"),
                StoreLayerSpec(name="beta_layer"),
            ]
        )
    )
    _seed_layer(store, "primary_layer", ["primary visible"])
    _seed_layer(store, "beta_layer", ["beta hidden"])
    primary_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="primary_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    beta_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="beta_layer"),
        readout=ConcatenateReadout(),
        store=store,
    )
    module = LLMFunctionCallEvolution(
        prompt=text_prompt(
            "Use only primary recall for visibility.",
            recall_plan=text_prompt("{{ recalled_prompt }}"),
            recall_query_builder=lambda packet, store, context: "primary",
            sub_recall_pipeline=primary_pipeline,
            labeled_recall_plans={"beta": text_prompt("{{ beta }}")},
            labeled_sub_recall_pipelines={"beta": beta_pipeline},
            labeled_recall_query_builders={"beta": lambda packet, store, context: "beta"},
            visible_record_recall_labels=(PRIMARY_RECALL_LABEL,),
        ),
        tools=["UPDATE"],
        source_layer="profile",
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"record_id": "rec-1", "text": "primary updated"})
        return "DONE"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, store = module.run(Packet(decisions_store={"profile": {"record_ids": []}}), store)

    assert store.iter_records("primary_layer")[0].text == "primary updated"
    assert packet_out.trace["memory_evolution"]["visible_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["prompt_trace"]["visible_record_ids_by_label"] == {
        PRIMARY_RECALL_LABEL: ["rec-1"]
    }


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

