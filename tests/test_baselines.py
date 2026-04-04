from __future__ import annotations

from dataclasses import replace
import asyncio
import json
from typing import Any
import pytest

from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
    registered_baseline_class_names,
)
from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    Observation,
    Packet,
    Placement,
    Query,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)
from memprimitive.pipeline_slots import PRE_EVOLUTION_SLOTS


def _stored_pipeline_packet(text: str, store: MemoryStore) -> tuple[Packet, MemoryStore]:
    """Pre-evolution ingest chain; uses the same default modules as the full pipeline."""
    mods = instantiate_default_baseline_modules(top_k=2)
    packet = Packet(observation=Observation(text=text, source="dialogue"))
    for slot in PRE_EVOLUTION_SLOTS:
        packet, store = mods[slot].run(packet, store)
    return packet, store


def _represented_packet(
    text: str,
    *,
    source: str = "dialogue",
    observation_metadata: dict | None = None,
) -> tuple[Packet, MemoryStore]:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    packet = Packet(observation=Observation(text=text, source=source, metadata=observation_metadata or {}))
    packet, store = PassThroughUnitFormation().run(packet, MemoryStore())
    packet, store = BasicRepresentation().run(packet, store)
    return packet, store


def _graph_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )


def _graph_vector_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity", "vector")),
            ]
        )
    )


def _mixed_graph_vector_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity", "vector")),
                StoreLayerSpec(name="other_graph", shape="Graph", indices=("graph", "entity", "vector")),
            ]
        )
    )


def _budgeted_store(
    *,
    layer_name: str = "episodic",
    record_budget: int | None = None,
    token_budget: int | None = None,
) -> MemoryStore:
    settings: dict[str, int] = {}
    if record_budget is not None:
        settings["record_budget"] = record_budget
    if token_budget is not None:
        settings["token_budget"] = token_budget
    capacity = "unlimited"
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name=layer_name, capacity=capacity, settings=settings),
            ]
        )
    )


def _seed_layer(store: MemoryStore, layer: str, texts: list[str]) -> None:
    for index, text in enumerate(texts, start=1):
        unit = MemoryUnit(
            unit_id=f"seed-{index}",
            text=text,
            timestamp=f"2026-01-01T00:00:{index:02d}Z",
            metadata={},
        )
        store.append(MemoryRecord.from_unit(unit=unit, layer=layer, sequence_id=store.next_sequence_id()))


def _seed_layer_with_metadata(store: MemoryStore, layer: str, units: list[dict[str, object]]) -> None:
    for index, payload in enumerate(units, start=1):
        unit = MemoryUnit(
            unit_id=str(payload.get("unit_id", f"seed-{index}")),
            text=str(payload.get("text", f"seed text {index}")),
            timestamp=f"2026-01-01T00:00:{index:02d}Z",
            metadata=dict(payload.get("metadata", {})),
        )
        store.append(MemoryRecord.from_unit(unit=unit, layer=layer, sequence_id=store.next_sequence_id()))


def _invoke_runtime_tool(tool, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.on_invoke_tool(None, json.dumps(arguments, ensure_ascii=False)))


def test_unit_formation_returns_one_unit_with_provenance() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    module = PassThroughUnitFormation()
    packet = Packet(observation=Observation(text="Alice likes tea.", source="dialogue"))

    packet_out, _ = module.run(packet, MemoryStore())

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    assert packet_out.units[0].text == "Alice likes tea."
    assert packet_out.units[0].metadata["provenance"]["observation_id"] == packet.observation.observation_id


def test_unit_formation_requires_observation() -> None:
    from memprimitive.baselines import PassThroughUnitFormation

    module = PassThroughUnitFormation()

    with pytest.raises(ValueError, match="packet.observation"):
        module.run(Packet(), MemoryStore())


def test_representation_preserves_identity_and_adds_normalized_text() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="  Alice Likes Tea  ", source="dialogue")),
        MemoryStore(),
    )

    packet_out, _ = BasicRepresentation().run(unit_packet, store)

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    assert packet_out.units[0].unit_id == unit_packet.units[0].unit_id
    assert packet_out.units[0].text == "Alice Likes Tea"
    assert packet_out.units[0].normalized_text == "alice likes tea"
    assert packet_out.units[0].embedding is not None
    assert len(packet_out.units[0].embedding) > 0
    assert packet_out.units[0].representation_elements == ("embedding", "text")
    assert packet_out.trace["representation"]["elements"] == ["text", "embedding"]
    assert packet_out.trace["representation"]["per_unit"][0]["elements"] == ["embedding", "text"]
    assert packet_out.units[0].metadata["representation"]["text"] == "Alice Likes Tea"
    assert packet_out.units[0].metadata["representation"]["normalized_text"] == "alice likes tea"
    assert packet_out.units[0].metadata["representation"]["embedding"]["dim"] == len(packet_out.units[0].embedding)


def test_basic_representation_rejects_legacy_triple_element() -> None:
    from memprimitive.baselines import BasicRepresentation

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "triple"))


def test_triple_representation_direct_uses_real_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice works at OpenAI in San Francisco and collaborates with Bob on graph memory systems.",
                source="notes",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = TripleRepresentation(method="direct").run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.triples
    assert unit.metadata["representation"]["triples"] == unit.triples
    assert unit.entities
    assert len(unit.entities) >= 2
    assert "triple" in unit.representation_elements
    assert "entities" in unit.representation_elements
    assert all(len(triple) == 3 for triple in unit.triples)
    flattened = " ".join(" ".join(part for part in triple) for triple in unit.triples).casefold()
    assert "alice" in flattened or "openai" in flattened or "bob" in flattened


def test_triple_representation_two_stage_uses_real_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice mentors Bob at OpenAI, and Bob researches retrieval graphs with Carol.",
                source="notes",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = TripleRepresentation(method="two_stage").run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.triples
    assert unit.entities
    assert unit.metadata["representation"]["triples"] == unit.triples
    entity_set = {entity.casefold() for entity in unit.entities}
    assert any(subject.casefold() in entity_set for subject, _, _ in unit.triples)
    assert any(obj.casefold() in entity_set for _, _, obj in unit.triples)
    assert all(subject and predicate and obj for subject, predicate, obj in unit.triples)


def test_basic_representation_rejects_summary_and_description_elements() -> None:
    from memprimitive.baselines import BasicRepresentation

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "summary"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "description"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "kv"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "entities"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "tags"))

    with pytest.raises(ValueError, match="Unsupported representation element"):
        BasicRepresentation(elements=("text", "relation_tags"))


def test_llm_representation_requires_openai_config() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(
        field="summary",
        prompt="Extract a one-sentence summary.",
        api_key="",
        base_url="",
        model="",
    )
    with pytest.raises(ValueError, match="LLMRepresentation field 'summary'.*MEMPRIMITIVE"):
        rep.run(unit_packet, store)


def test_llm_representation_writes_known_list_field_to_unit() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="tags", prompt="Extract retrieval tags.")

    def _fake_llm_json(*, user: str) -> Any:
        payload = json.loads(user)
        assert payload["field"] == "tags"
        assert payload["prompt"] == "Extract retrieval tags."
        assert payload["unit"]["text"] == "Alice studies graph memory systems."
        return ["graph-memory", "research"]

    rep._llm_json = _fake_llm_json  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    unit = packet_out.units[0]
    assert unit.tags == ["graph-memory", "research"]
    assert "tags" in unit.representation_elements
    assert unit.metadata["representation"]["tags"] == ["graph-memory", "research"]
    assert packet_out.trace["representation"]["field"] == "tags"
    assert packet_out.trace["representation"]["per_unit"][0]["kind"] == "list"


def test_llm_representation_writes_summary_and_custom_fields_into_representation_metadata() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    summary_rep = LLMRepresentation(field="summary", prompt="Extract a one-sentence summary.")
    summary_rep._llm_text = lambda *, user: "Alice studies graph memory systems."  # type: ignore[method-assign]
    packet_out, _ = summary_rep.run(unit_packet, store)

    summary_unit = packet_out.units[0]
    assert summary_unit.metadata["representation"]["summary"] == "Alice studies graph memory systems."
    assert "summary" in summary_unit.representation_elements

    custom_rep = LLMRepresentation(field="custom_topic", prompt="Extract the main topic.")
    custom_rep._llm_text = lambda *, user: "graph memory"  # type: ignore[method-assign]
    packet_out, _ = custom_rep.run(packet_out, store)

    custom_unit = packet_out.units[0]
    assert custom_unit.metadata["representation"]["custom_topic"] == "graph memory"
    assert "custom_topic" in custom_unit.representation_elements


def test_llm_representation_prompt_template_renders_unit_context_and_trace() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(
            observation=Observation(
                text="Alice studies graph memory systems.",
                source="notes",
                metadata={"session_id": "sess-1"},
            )
        ),
        MemoryStore(),
    )
    rep = LLMRepresentation(
        field="summary",
        prompt="Extract {{ field }} for {{ unit.unit_type }}: {{ unit.text }} / {{ unit.metadata.session_id | default('none') }}",
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "Extract summary for observation: Alice studies graph memory systems. / sess-1"
        return "templated summary"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    assert packet_out.units[0].metadata["representation"]["summary"] == "templated summary"
    assert packet_out.trace["representation"]["prompt_is_template"] is True
    assert packet_out.trace["representation"]["per_unit"][0]["rendered_prompt"].startswith("Extract summary")
    assert packet_out.trace["representation"]["per_unit"][0]["missing_variables"] == []


def test_llm_representation_prompt_template_missing_variables_do_not_crash() -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory systems.", source="notes")),
        MemoryStore(),
    )
    rep = LLMRepresentation(field="summary", prompt="Extract {{ unit.metadata.unknown_key }} from {{ unit.text }}")

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "Extract  from Alice studies graph memory systems."
        return "summary with missing field"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    assert "unit.metadata.unknown_key" in packet_out.trace["representation"]["per_unit"][0]["missing_variables"]


def test_llm_representation_prompt_template_supports_recalled_prompt_from_current_store() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMRepresentation, PassThroughUnitFormation, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice is preparing a reply.", source="notes")),
        MemoryStore(),
    )
    _seed_layer(store, "default", ["CURRENT STORE PROFILE"])

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "default", ["WRONG PIPELINE STORE PROFILE"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    rep = LLMRepresentation(
        field="summary",
        prompt=text_prompt(
            "Use {{ recalled_prompt }} while summarizing {{ unit.text }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"profile for {context['unit']['text']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "Use CURRENT STORE PROFILE while summarizing Alice is preparing a reply."
        return "summary with recalled prompt"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    prompt_trace = packet_out.trace["representation"]["per_unit"][0]
    assert prompt_trace["recall_prompt"]["enabled"] is True
    assert prompt_trace["recall_prompt"]["rendered_recall_query"] == "profile for Alice is preparing a reply."
    assert prompt_trace["recalled_prompt"] == "CURRENT STORE PROFILE"


def test_llm_representation_prompt_template_empty_recalled_prompt_falls_back_to_empty_string() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMRepresentation, PassThroughUnitFormation, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice is preparing a reply.", source="notes")),
        MemoryStore(),
    )
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    rep = LLMRepresentation(
        field="summary",
        prompt=text_prompt(
            "prefix {{ recalled_prompt }} suffix",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"profile for {context['unit']['text']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == "prefix  suffix"
        return "summary without recalled prompt"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    prompt_trace = packet_out.trace["representation"]["per_unit"][0]
    assert prompt_trace["recall_prompt"]["enabled"] is True
    assert prompt_trace["recall_prompt"]["matched"] is False
    assert prompt_trace["recalled_prompt"] == ""


def test_memory_store_delete_record_removes_expected_record() -> None:
    store = MemoryStore()
    _seed_layer(store, "default", ["first", "second"])

    removed = store.delete_record("default", "rec-1")

    assert removed.record_id == "rec-1"
    assert [record.record_id for record in store.iter_records("default")] == ["rec-2"]


def test_memory_store_delete_record_rejects_unknown_record() -> None:
    store = MemoryStore()
    _seed_layer(store, "default", ["first"])

    with pytest.raises(KeyError, match="not found"):
        store.delete_record("default", "rec-missing")


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


def test_llm_function_call_evolution_graph_tools_declare_graph_contracts() -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution
    from memprimitive.contracts import RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT

    graph_module = LLMFunctionCallEvolution(prompt="x", tools=["GRAPH_DELETE"], source_layer="knowledge_graph")
    plain_module = LLMFunctionCallEvolution(prompt="x", tools=["DELETE"], source_layer="default")

    assert graph_module.get_requires_contracts() == frozenset({TOPOLOGY_GRAPH_LAYER_CONTRACT})
    assert graph_module.get_produces_contracts() == frozenset({RECORD_GRAPH_LINKS_CONTRACT})
    assert plain_module.get_requires_contracts() == frozenset()
    assert plain_module.get_produces_contracts() == frozenset()


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


def test_write_trigger_aligns_decisions_with_units() -> None:
    from memprimitive.baselines import AlwaysTrigger, BasicRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)

    packet_out, _ = AlwaysTrigger().run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["module"] == "always_write_trigger"
    assert packet_out.trace["write_trigger"]["threshold"] is None
    assert packet_out.trace["write_trigger"]["constant"] == 1.0
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is True


def test_evolution_trigger_aligns_decisions_with_units() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        AppendOrganization,
        BasicRepresentation,
        NeverTrigger,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = AppendOrganization().run(packet, store)

    packet_out, _ = NeverTrigger().run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.trace["evolution_trigger"]["module"] == "never_evolution_trigger"
    assert packet_out.trace["evolution_trigger"]["decisions"] == [False]
    assert packet_out.trace["evolution_trigger"]["threshold"] is None
    assert packet_out.trace["evolution_trigger"]["constant"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is False


def test_organization_aligns_placements_with_units_and_commits_normal_write() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)

    packet_out, updated_store = AppendOrganization().run(packet, store)

    assert packet_out.placements is not None
    assert len(packet_out.placements) == len(packet_out.units)
    assert packet_out.placements[0].target_layer == "default"
    assert updated_store.count() == 1
    assert packet_out.trace["organization"]["written_record_ids"]
    assert packet_out.trace["organization"]["written_unit_ids"] == [packet_out.units[0].unit_id]
    assert packet_out.trace["organization"]["skipped_unit_count"] == 0


def test_append_only_evolution_is_noop_when_decisions_are_false() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        decisions=[False],
        placements=packet.placements,
        trace=packet.trace,
    )

    _, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 1


def test_append_only_evolution_records_active_unit_ids_without_mutating_store() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        decisions=[True],
        placements=packet.placements,
        trace=packet.trace,
    )

    packet_out, updated_store = AppendOnlyEvolution().run(packet, store)

    assert updated_store.count() == 1
    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions"
    assert packet_out.trace["memory_evolution"]["active_unit_ids"] == [packet.units[0].unit_id]
    assert packet_out.trace["memory_evolution"]["effects"] == []


def test_append_only_evolution_requires_explicit_decisions() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    packet, store = _stored_pipeline_packet("Alice likes tea.", MemoryStore())
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        trace=packet.trace,
    )

    with pytest.raises(ValueError, match="packet.decisions"):
        AppendOnlyEvolution().run(packet, store)


def test_append_only_evolution_requires_aligned_inputs() -> None:
    from memprimitive.baselines import AppendOnlyEvolution

    with pytest.raises(ValueError, match="aligned units"):
        AppendOnlyEvolution().run(
            Packet(units=[], decisions=[True], placements=[]),
            MemoryStore(),
        )


def test_write_and_evolution_trigger_are_independent_by_default() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        AppendOrganization,
        BasicRepresentation,
        NeverTrigger,
        PassThroughUnitFormation,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    write_packet, store = AlwaysTrigger().run(packet, store)
    write_packet, store = AppendOrganization().run(write_packet, store)
    evolution_packet, _ = NeverTrigger().run(write_packet, store)

    assert write_packet.decisions == [True]
    assert evolution_packet.decisions == [False]
    assert write_packet.trace["write_trigger"]["module"] == "always_write_trigger"
    assert evolution_packet.trace["evolution_trigger"]["module"] == "never_evolution_trigger"


def test_threshold_write_trigger_respects_threshold_policy() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation, ThresholdTrigger

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)

    packet_out, _ = ThresholdTrigger(threshold=0.8, constant=0.7).run(packet, store)
    assert packet_out.decisions == [False]
    assert packet_out.trace["write_trigger"]["threshold"] == 0.8
    assert packet_out.trace["write_trigger"]["constant"] == 0.7
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is False

    packet_out, _ = ThresholdTrigger(threshold=0.7, constant=0.7).run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["decision"] is True


def test_threshold_evolution_trigger_writes_only_decisions() -> None:
    from memprimitive.baselines import (
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
        ThresholdTrigger,
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = ThresholdTrigger(slot="evolution_trigger", threshold=2.0, constant=1.0).run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.trace["evolution_trigger"]["threshold"] == 2.0
    assert packet_out.trace["evolution_trigger"]["constant"] == 1.0
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["decision"] is False


def test_boundary_event_trigger_matches_structural_events_for_both_slots() -> None:
    from memprimitive.baselines import AppendOrganization, BoundaryEventTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": ["turn_end", "session_end"]}},
    )
    write_packet, store = BoundaryEventTrigger(accepted_events=("session_end",)).run(packet, store)

    assert write_packet.decisions == [True]
    assert write_packet.trace["write_trigger"]["source"] == "boundary"
    assert write_packet.trace["write_trigger"]["matched_events"] == ["session_end"]

    organized_packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )
    evolution_packet, _ = BoundaryEventTrigger(
        slot="evolution_trigger",
        accepted_events=("session_end",),
    ).run(organized_packet, store)

    assert evolution_packet.decisions == [True]
    assert evolution_packet.trace["evolution_trigger"]["source"] == "boundary"


@pytest.mark.parametrize(
    ("event_name", "match_key", "match_value"),
    [
        ("session_end", "session_id", "sess-1"),
        ("turn_end", "turn_id", "turn-1"),
        ("chunk_end", "chunk_id", "chunk-1"),
        ("subgoal_end", "subgoal_id", "subgoal-1"),
        ("episode_end", "episode_id", "episode-1"),
    ],
)
def test_boundary_event_trigger_populates_decisions_store_for_matching_boundary(
    event_name: str,
    match_key: str,
    match_value: str,
) -> None:
    from memprimitive.baselines import BoundaryEventTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic", theme="episode"),
            ]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "default match", "metadata": {match_key: match_value}},
            {"text": "default miss", "metadata": {match_key: "other"}},
        ],
    )
    _seed_layer_with_metadata(
        store,
        "episodic",
        [
            {"text": "episodic match", "metadata": {match_key: match_value}},
            {"text": "episodic other", "metadata": {"session_id": "other-session"}},
        ],
    )
    packet, _ = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": [event_name], match_key: match_value}},
    )

    packet_out, _ = BoundaryEventTrigger(accepted_events=(event_name,)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"default", "episodic"}
    assert packet_out.decisions_store["default"]["record_ids"] == ["rec-1"]
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-3"]
    assert packet_out.trace["write_trigger"]["boundary_kind"] == match_key.removesuffix("_id")
    assert packet_out.trace["write_trigger"]["match_key"] == match_key
    assert packet_out.trace["write_trigger"]["match_value"] == match_value
    assert packet_out.trace["write_trigger"]["decisions_store_counts"] == {"default": 1, "episodic": 1}


def test_boundary_event_trigger_keeps_decisions_but_records_missing_match_key() -> None:
    from memprimitive.baselines import BoundaryEventTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": ["session_end"]}},
    )

    packet_out, _ = BoundaryEventTrigger(accepted_events=("session_end",)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is None
    assert packet_out.trace["write_trigger"]["missing_match_key"] is True
    assert packet_out.trace["write_trigger"]["match_key"] == "session_id"
    assert packet_out.trace["write_trigger"]["match_value"] is None
    assert packet_out.trace["write_trigger"]["decisions_store_layers"] == []


def test_runtime_event_trigger_uses_packet_events_when_trigger_metadata_missing() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    packet, store = _represented_packet("Alice likes tea.")
    packet = replace(packet, events=["task_failed"])
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            events=packet.events,
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(accepted_events=("task_failed",)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["evolution_trigger"]["source"] == "runtime"
    assert packet_out.trace["evolution_trigger"]["observed_events"] == ["task_failed"]


def test_runtime_event_trigger_computes_memory_pressure_event_from_record_budget() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=2)
    _seed_layer(store, "episodic", ["one", "two"])
    packet, _ = _represented_packet("Alice likes tea.")
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(
        accepted_events=("memory_pressure",),
        pressure_threshold=1.0,
    ).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["evolution_trigger"]["matched_events"] == ["memory_pressure"]
    assert packet_out.trace["evolution_trigger"]["computed_runtime_events"] == ["memory_pressure"]
    assert packet_out.trace["evolution_trigger"]["record_pressure"] == 1.5
    assert packet_out.trace["evolution_trigger"]["token_pressure"] is None
    assert packet_out.trace["evolution_trigger"]["target_layer"] == "episodic"
    assert packet_out.decisions_store is not None
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2", "rec-3"]
    assert packet_out.decisions_store["episodic"]["selector"]["kind"] == "layer_all"
    assert packet_out.trace["evolution_trigger"]["decisions_store_counts"] == {"episodic": 3}


def test_runtime_event_trigger_keeps_literal_memory_pressure_event_without_threshold() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=10)
    packet, _ = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"events": ["memory_pressure"]}},
    )
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(accepted_events=("memory_pressure",)).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["evolution_trigger"]["observed_events"] == ["memory_pressure"]
    assert packet_out.trace["evolution_trigger"]["computed_runtime_events"] == []
    assert packet_out.trace["evolution_trigger"]["pressure_threshold"] is None
    assert packet_out.decisions_store is not None
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1"]


def test_runtime_event_trigger_skips_decisions_store_when_memory_pressure_not_triggered() -> None:
    from memprimitive.baselines import AppendOrganization, RuntimeEventTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=10)
    packet, _ = _represented_packet("Alice likes tea.")
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = RuntimeEventTrigger(
        accepted_events=("memory_pressure",),
        pressure_threshold=2.0,
    ).run(packet, store)

    assert packet_out.decisions == [False]
    assert packet_out.decisions_store is None
    assert packet_out.trace["evolution_trigger"]["decisions_store_layers"] == []


def test_runtime_event_trigger_memory_pressure_requires_single_layer_for_broadcast_resolution() -> None:
    from memprimitive.baselines import RuntimeEventTrigger

    packet, store = _represented_packet("Alice likes tea.")
    packet = replace(
        packet,
        placements=[
            Placement(unit_id="unit-a", target_layer="default"),
            Placement(unit_id="unit-b", target_layer="episodic"),
        ],
        units=[replace(packet.units[0], unit_id="unit-a"), replace(packet.units[0], unit_id="unit-b")],
    )
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default", capacity="unlimited", settings={"record_budget": 2}),
                StoreLayerSpec(name="episodic", capacity="unlimited", settings={"record_budget": 2}),
            ]
        )
    )

    with pytest.raises(ValueError, match="single target layer"):
        RuntimeEventTrigger(
            accepted_events=("memory_pressure",),
            pressure_threshold=0.5,
        ).run(packet, store)


def test_scalar_rule_trigger_supports_broadcast_and_per_unit_modes() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"signals": {"importance": 0.82}}},
    )
    packet_out, _ = ScalarRuleTrigger(signal_key="importance", threshold=0.8).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["signal_key"] == "importance"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["signal_value"] == 0.82

    multi_unit = replace(
        packet,
        units=[
            replace(packet.units[0], unit_id="unit-a", metadata={"importance": 0.9}),
            replace(packet.units[0], unit_id="unit-b", metadata={"importance": 0.3}),
        ],
    )
    per_unit_packet, _ = ScalarRuleTrigger(
        signal_key="importance",
        threshold=0.5,
        aggregate="per_unit",
    ).run(multi_unit, store)

    assert per_unit_packet.decisions == [True, False]
    assert per_unit_packet.trace["write_trigger"]["aggregate"] == "per_unit"


def test_scalar_rule_trigger_memory_pressure_supports_record_and_token_budgets() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    store = _budgeted_store(layer_name="episodic", record_budget=4, token_budget=3)
    _seed_layer(store, "episodic", ["alpha beta", "gamma delta"])
    packet, _ = _represented_packet("Alice likes tea.")

    packet_out, _ = ScalarRuleTrigger(
        signal_key="memory_pressure",
        threshold=1.0,
        target_layer="episodic",
    ).run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["target_layer_mode"] == "explicit"
    assert packet_out.trace["write_trigger"]["record_pressure"] == 0.5
    assert packet_out.trace["write_trigger"]["token_pressure"] == pytest.approx(4 / 3)
    assert packet_out.trace["write_trigger"]["memory_pressure"] == pytest.approx(4 / 3)
    assert packet_out.trace["write_trigger"]["active_budget_types"] == ["record_budget", "token_budget"]
    assert packet_out.trace["write_trigger"]["per_unit"][0]["target_layer"] == "episodic"
    assert packet_out.decisions_store is not None
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2"]
    assert packet_out.decisions_store["episodic"]["selector"]["kind"] == "layer_all"
    assert packet_out.decisions_store["episodic"]["selector"]["source"] == "scalar_rule"
    assert packet_out.trace["write_trigger"]["decisions_store_counts"] == {"episodic": 2}


def test_scalar_rule_trigger_memory_pressure_supports_per_unit_resolution_from_placements() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working", capacity="unlimited", settings={"record_budget": 4}),
                StoreLayerSpec(name="semantic", capacity="unlimited", settings={"record_budget": 2}),
            ]
        )
    )
    _seed_layer(store, "working", ["one"])
    _seed_layer(store, "semantic", ["one", "two"])
    packet, _ = _represented_packet("Alice likes tea.")
    packet = replace(
        packet,
        units=[
            replace(packet.units[0], unit_id="unit-a"),
            replace(packet.units[0], unit_id="unit-b"),
        ],
        placements=[
            Placement(unit_id="unit-a", target_layer="working"),
            Placement(unit_id="unit-b", target_layer="semantic"),
        ],
    )

    packet_out, _ = ScalarRuleTrigger(
        slot="evolution_trigger",
        signal_key="memory_pressure",
        threshold=0.75,
        aggregate="per_unit",
    ).run(packet, store)

    assert packet_out.decisions == [False, True]
    assert packet_out.trace["evolution_trigger"]["target_layer_mode"] == "placement"
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["target_layer"] == "working"
    assert packet_out.trace["evolution_trigger"]["per_unit"][0]["record_pressure"] == 0.25
    assert packet_out.trace["evolution_trigger"]["per_unit"][1]["target_layer"] == "semantic"
    assert packet_out.trace["evolution_trigger"]["per_unit"][1]["record_pressure"] == 1.0
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"semantic"}
    assert packet_out.decisions_store["semantic"]["record_ids"] == ["rec-2", "rec-3"]
    assert packet_out.trace["evolution_trigger"]["decisions_store_counts"] == {"semantic": 2}


def test_scalar_rule_trigger_memory_pressure_requires_explicit_write_layer() -> None:
    from memprimitive.baselines import ScalarRuleTrigger

    packet, store = _represented_packet("Alice likes tea.")

    with pytest.raises(ValueError, match="explicit target_layer"):
        ScalarRuleTrigger(signal_key="memory_pressure", threshold=0.8).run(packet, store)


def test_model_judge_trigger_supports_injected_per_unit_and_broadcast_modes() -> None:
    from memprimitive.baselines import AppendOrganization, ModelJudgeTrigger

    packet, store = _represented_packet("Alice likes tea.")

    def per_unit_judge(payload: dict) -> dict:
        return {"decision": "alice" in payload["unit"]["text"].casefold(), "score": 0.9, "label": "write"}

    packet_out, _ = ModelJudgeTrigger(system_prompt="Judge writes.", judge_callable=per_unit_judge).run(packet, store)
    assert packet_out.decisions == [True]
    assert packet_out.trace["write_trigger"]["source"] == "model_judge"
    assert packet_out.trace["write_trigger"]["per_unit"][0]["score"] == 0.9

    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    def broadcast_judge(payload: dict) -> dict:
        assert payload["unit"] is None
        return {"score": 0.75}

    evolution_packet, _ = ModelJudgeTrigger(
        slot="evolution_trigger",
        system_prompt="Judge evolution.",
        decision_mode="score",
        threshold=0.7,
        per_unit=False,
        judge_callable=broadcast_judge,
    ).run(packet, store)
    assert evolution_packet.decisions == [True]
    assert evolution_packet.trace["evolution_trigger"]["per_unit"][0]["score"] == 0.75


def test_periodic_maintenance_trigger_runs_wrapped_trigger_when_schedule_matches() -> None:
    from memprimitive.baselines import AppendOrganization, PeriodicMaintenanceTrigger, ScalarRuleTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 12, "idle_seconds": 45.0}, "events": ["idle"]}},
    )
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=ScalarRuleTrigger(
            slot="evolution_trigger",
            signal_key="importance",
            threshold=0.5,
        ),
    ).run(
        replace(
            packet,
            observation=replace(
                packet.observation,
                metadata={"trigger": {"schedule": {"tick": 12}, "signals": {"importance": 0.9}}},
            ),
        ),
        store,
    )

    assert periodic_packet.decisions == [True]
    assert periodic_packet.trace["evolution_trigger"]["module"] == "scalar_rule_evolution_trigger"
    assert periodic_packet.trace["evolution_trigger"]["tick"] == 12
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is True
    assert periodic_packet.trace["evolution_trigger"]["wrapped_trigger_module"] == "scalar_rule_evolution_trigger"
    assert periodic_packet.trace["evolution_trigger"]["signal_key"] == "importance"


def test_periodic_maintenance_trigger_preserves_existing_decisions_on_miss() -> None:
    from memprimitive.baselines import AppendOrganization, PeriodicMaintenanceTrigger, NeverTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 11}}},
    )
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=NeverTrigger(slot="evolution_trigger"),
    ).run(packet, store)

    assert periodic_packet.decisions == [True]
    assert periodic_packet.trace["evolution_trigger"]["module"] == "periodic_maintenance_evolution_trigger"
    assert periodic_packet.trace["evolution_trigger"]["tick"] == 11
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is False
    assert periodic_packet.trace["evolution_trigger"]["wrapped_trigger_module"] == "never_evolution_trigger"


def test_periodic_maintenance_trigger_keeps_none_decisions_on_miss() -> None:
    from memprimitive.baselines import PeriodicMaintenanceTrigger, NeverTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 11}}},
    )
    packet = replace(
        packet,
        placements=[Placement(unit_id=packet.units[0].unit_id, target_layer="default")],
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=NeverTrigger(slot="evolution_trigger"),
    ).run(packet, store)

    assert periodic_packet.decisions is None
    assert periodic_packet.trace["evolution_trigger"]["decisions"] is None
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is False


def test_periodic_maintenance_trigger_uses_store_counter_when_schedule_tick_missing() -> None:
    from memprimitive.baselines import AppendOrganization, NeverTrigger, PeriodicMaintenanceTrigger

    packet, store = _represented_packet("Alice likes tea.")
    store = replace(store, metadata={**store.metadata, "ingest_count": 6})
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    periodic_packet, _ = PeriodicMaintenanceTrigger(
        every_n=3,
        trigger=NeverTrigger(slot="evolution_trigger"),
    ).run(packet, store)

    assert periodic_packet.trace["evolution_trigger"]["tick"] == 6
    assert periodic_packet.trace["evolution_trigger"]["periodic_matched"] is True
    assert periodic_packet.decisions == [False]


def test_periodic_maintenance_trigger_rejects_wrapped_trigger_slot_mismatch() -> None:
    from memprimitive.baselines import AlwaysTrigger, PeriodicMaintenanceTrigger

    with pytest.raises(ValueError, match="wrapped trigger slot"):
        PeriodicMaintenanceTrigger(
            every_n=3,
            trigger=AlwaysTrigger(slot="write_trigger"),
        )


def test_idle_maintenance_trigger_gates_evolution_from_schedule_metadata() -> None:
    from memprimitive.baselines import AppendOrganization, IdleMaintenanceTrigger

    packet, store = _represented_packet(
        "Alice likes tea.",
        observation_metadata={"trigger": {"schedule": {"tick": 12, "idle_seconds": 45.0}, "events": ["idle"]}},
    )
    packet, store = AppendOrganization().run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    idle_packet, _ = IdleMaintenanceTrigger(min_idle_seconds=30.0).run(packet, store)
    assert idle_packet.decisions == [True]
    assert idle_packet.trace["evolution_trigger"]["idle_seconds"] == 45.0


def test_store_all_trigger_preserves_existing_decisions_and_selects_all_layers() -> None:
    from memprimitive.baselines import AlwaysTrigger, StoreAllTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
                StoreLayerSpec(name="semantic"),
            ]
        )
    )
    _seed_layer(store, "default", ["default one"])
    _seed_layer(store, "episodic", ["episodic one", "episodic two"])
    _seed_layer(store, "semantic", ["semantic one"])

    packet, _ = _represented_packet("Alice likes tea.")
    packet, _ = AlwaysTrigger().run(packet, store)

    packet_out, _ = StoreAllTrigger().run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"default", "episodic", "semantic"}
    assert packet_out.decisions_store["default"]["record_ids"] == ["rec-1"]
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-2", "rec-3"]
    assert packet_out.decisions_store["semantic"]["record_ids"] == ["rec-4"]
    assert packet_out.decisions_store["semantic"]["selector"]["kind"] == "store_all"
    assert packet_out.decisions_store["semantic"]["selector"]["source"] == "store_all_trigger"
    assert packet_out.trace["write_trigger"]["module"] == "store_all_write_trigger"
    assert packet_out.trace["write_trigger"]["decisions"] == [True]
    assert packet_out.trace["write_trigger"]["decisions_store_counts"] == {
        "default": 1,
        "episodic": 2,
        "semantic": 1,
    }


def test_store_all_trigger_keeps_decisions_none_when_no_prior_decision_exists() -> None:
    from memprimitive.baselines import StoreAllTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    _seed_layer(store, "episodic", ["episodic one"])
    packet, _ = _represented_packet("Alice likes tea.")

    packet_out, _ = StoreAllTrigger().run(packet, store)

    assert packet_out.decisions is None
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"episodic"}
    assert packet_out.trace["write_trigger"]["decisions"] is None
    assert packet_out.trace["write_trigger"]["per_unit"] == []
    assert packet_out.trace["write_trigger"]["preserved_decisions"] is False


def test_store_all_trigger_supports_evolution_slot() -> None:
    from memprimitive.baselines import AppendOrganization, StoreAllTrigger

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="episodic"),
            ]
        )
    )
    _seed_layer(store, "episodic", ["prior one"])
    packet, _ = _represented_packet("Alice likes tea.")
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True],
            trace=packet.trace,
        ),
        store,
    )

    packet_out, _ = StoreAllTrigger(slot="evolution_trigger").run(packet, store)

    assert packet_out.decisions == [True]
    assert packet_out.decisions_store is not None
    assert set(packet_out.decisions_store) == {"episodic"}
    assert packet_out.decisions_store["episodic"]["record_ids"] == ["rec-1", "rec-2"]
    assert packet_out.trace["evolution_trigger"]["module"] == "store_all_evolution_trigger"


def test_new_trigger_classes_are_registered_in_baseline_exports() -> None:
    exported = registered_baseline_class_names()

    assert {
        "BoundaryEventTrigger",
        "RuntimeEventTrigger",
        "ScalarRuleTrigger",
        "StoreAllTrigger",
        "ModelJudgeTrigger",
        "PeriodicMaintenanceTrigger",
        "IdleMaintenanceTrigger",
    }.issubset(exported)


def test_hierarchical_classes_are_registered_in_baseline_exports() -> None:
    exported = registered_baseline_class_names()

    assert {
        "HierarchicalOrganization",
        "HierarchicalEvolution",
    }.issubset(exported)


class _FakeAMEMRuntime:
    def require_llm(self, *, capability: str) -> None:
        return None

    def embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            10.0 if "alice" in lowered else 0.0,
            8.0 if "tea" in lowered else 0.0,
            6.0 if "focus" in lowered else 0.0,
            4.0 if "graph" in lowered else 0.0,
            float(len(lowered)),
        ]

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        lowered_system = system.casefold()
        if "enrich memory notes" in lowered_system or "note generator" in lowered_system:
            unit_text = payload["unit_text"].casefold()
            if "alice likes tea" in unit_text:
                return {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                }
            if "tea routines improve focus" in unit_text:
                return {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                }
            return {
                "content": payload["unit_text"],
                "note_text": "Graph note",
                "context": "Graph memory context.",
                "keywords": ["graph", "memory"],
                "tags": ["graph", "memory"],
                "category": "insight",
                "attributes": {"topic": "graph"},
            }
        if "memory write controller" in lowered_system:
            return {"decision": "write", "reason": "store the note", "confidence": 0.9}
        if "choose which neighbors should receive" in lowered_system:
            return {"connections": [0], "tags": ["focus", "tea", "bridge"]}
        if "update each neighbor note's context and tags" in lowered_system:
            return {
                "updates": [
                    {
                        "context": "Alice's tea habit is now understood as a focus-supporting routine.",
                        "tags": ["preference", "habit", "focus"],
                    }
                ]
            }
        if "expand the query" in lowered_system or (
            "expand" in lowered_system and "knowledge_graph" in lowered_system
        ):
            return {
                "query_text": payload["query"],
                "content": payload["query"],
                "context": "Retrieve the most relevant enriched note.",
                "keywords": ["alice", "tea"] if "alice" in payload["query"].casefold() else ["focus", "graph"],
                "tags": ["query", "memory"],
                "category": "query",
                "attributes": {},
            }
        raise AssertionError(f"Unexpected runtime prompt: {system}")

    def rerank(self, *, query: str, candidates: list[dict[str, object]], task: str, top_k: int):
        return [
            {
                "id": str(candidate["id"]),
                "score": float(candidate.get("score", 0.0)),
                "rationale": f"selected for {query}",
            }
            for candidate in sorted(
                candidates,
                key=lambda item: (-float(item.get("score", 0.0)), str(item.get("id", ""))),
            )[:top_k]
        ]


class _WrapperShapeAMEMRuntime(_FakeAMEMRuntime):
    def json(self, *, system: str, user: str):
        payload = super().json(system=system, user=user)
        lowered_system = system.casefold()
        if "choose which neighbors should receive" in lowered_system:
            return [0]
        if "update each neighbor note's context and tags" in lowered_system:
            return payload["updates"]
        return payload


class _FakeHierarchicalRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def require_llm(self, *, capability: str) -> None:
        return None

    def json(self, *, system: str, user: str):
        payload = json.loads(user)
        self.calls.append({"system": system, "payload": payload})
        fields = payload["extract_fields"]
        records = payload["records"]
        group_key = payload["group_key"]
        if "CUSTOM HIERARCHICAL PROMPT" in system:
            return {
                field: f"custom::{field}::{group_key.get('session_id', 'all')}::{len(records)}"
                for field in fields
            }
        return {
            field: f"generated::{field}::{group_key.get('session_id', 'all')}::{len(records)}"
            for field in fields
        }




def test_retrieval_honors_top_k() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("one", "two", "three"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="items")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2


def test_retrieval_rejects_non_positive_top_k() -> None:
    from memprimitive.baselines import RecencyRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        RecencyRetrieval(top_k=0)


def test_embedding_similarity_retrieval_rejects_non_positive_top_k() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        EmbeddingSimilarityRetrieval(top_k=0)


def test_retrieval_on_empty_store_returns_empty_retrieved_set() -> None:
    from memprimitive.baselines import RecencyRetrieval

    packet_out, store_out = RecencyRetrieval(top_k=2).run(
        Packet(query=Query(text="alice")),
        MemoryStore(),
    )

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert store_out.count() == 0


def test_readout_formats_deterministic_text_and_source_ids() -> None:
    from memprimitive.baselines import ConcatenateReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    retrieved = RetrievedSet(items=list(reversed(store.iter_records())), scores=[])

    packet_out, _ = ConcatenateReadout().run(Packet(retrieved=retrieved), store)

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Alice likes tea."
    assert packet_out.readout.source_ids == [store.iter_records()[0].record_id]


def test_readout_on_empty_retrieval_returns_valid_empty_output() -> None:
    from memprimitive.baselines import ConcatenateReadout

    packet_out, _ = ConcatenateReadout().run(Packet(retrieved=RetrievedSet()), MemoryStore())

    assert packet_out.readout is not None
    assert packet_out.readout.text == ""
    assert packet_out.readout.source_ids == []


def test_retrieval_returns_latest_records_first_even_when_query_matches_older_records() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob prefers coffee", "Alice studies graphs"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), store)

    assert packet_out.retrieved is not None
    assert len(packet_out.retrieved.items) == 2
    assert [record.text for record in packet_out.retrieved.items] == [
        "Alice studies graphs",
        "Bob prefers coffee",
    ]


def test_retrieval_returns_latest_records_first_regardless_of_query_text() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("first item", "second item", "third item"):
        packet, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="unmatched")), store)

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["third item", "second item"]


def test_recency_retrieval_source_store_preserves_default_behavior() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("first item", "second item", "third item"):
        _, store = _stored_pipeline_packet(text, store)

    default_packet, _ = RecencyRetrieval(top_k=2).run(Packet(query=Query(text="ignored")), store)
    explicit_store_packet, _ = RecencyRetrieval(top_k=2, source="store").run(Packet(query=Query(text="ignored")), store)

    assert [record.record_id for record in explicit_store_packet.retrieved.items] == [
        record.record_id for record in default_packet.retrieved.items
    ]
    assert explicit_store_packet.retrieved.trace["source"] == "store"


def test_recency_retrieval_source_retrieved_reranks_existing_subset_only() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("first item", "second item", "third item", "fourth item"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2, source="retrieved").run(
        Packet(
            query=Query(text="ignored"),
            retrieved=RetrievedSet(items=[store.iter_records()[0], store.iter_records()[2]], scores=[]),
        ),
        store,
    )

    assert [record.text for record in packet_out.retrieved.items] == ["third item", "first item"]
    assert packet_out.retrieved.trace["source"] == "retrieved"
    assert packet_out.retrieved.trace["candidate_count"] == 2


def test_retrieval_does_not_mutate_store() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea", store)
    before_ids = [record.record_id for record in store.iter_records()]

    _, store_after = RecencyRetrieval(top_k=1).run(Packet(query=Query(text="Alice")), store)

    assert [record.record_id for record in store_after.iter_records()] == before_ids


def test_embedding_similarity_retrieval_ranks_records_by_query_embedding() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    records = [
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="closest",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        ),
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="second",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[0.8, 0.2],
            metadata={"representation": {"embedding": {"dim": 2}}},
        ),
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="default",
            text="opposite",
            timestamp="2026-01-01T00:00:02+00:00",
            embedding=[-1.0, 0.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        ),
    ]
    for record in records:
        store.append(record)

    packet_out, store_after = EmbeddingSimilarityRetrieval(top_k=2).run(
        Packet(query=Query(text="ignored", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1", "rec-2"]
    assert packet_out.retrieved.scores[0]["strategy"] == "embedding_similarity"
    assert packet_out.retrieved.scores[0]["record_id"] == "rec-1"
    assert packet_out.retrieved.scores[0]["rank"] == 1
    assert packet_out.retrieved.scores[0]["score"] >= packet_out.retrieved.scores[1]["score"]
    assert packet_out.trace["retrieval"]["reused_query_embedding"] is True
    assert [record.record_id for record in store_after.iter_records()] == [record.record_id for record in store.iter_records()]


def test_embedding_similarity_retrieval_computes_and_caches_query_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="alpha",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="beta",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[0.0, 1.0],
            metadata={"representation": {"embedding": {"dim": 2}}},
        )
    )

    def _fake_embed_text(self, text: str) -> list[float]:
        assert text == "alpha query"
        return [1.0, 0.0]

    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", _fake_embed_text)

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1).run(Packet(query=Query(text="alpha query")), store)

    assert packet_out.query is not None
    assert packet_out.query.embedding == [1.0, 0.0]
    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.trace["retrieval"]["reused_query_embedding"] is False
    assert packet_out.trace["retrieval"]["embedding_candidate_count"] == 2


def test_embedding_similarity_retrieval_uses_record_embedding_not_metadata_summary() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="metadata-only",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=None,
            metadata={"representation": {"embedding": {"dim": 2}}},
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1).run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.trace["retrieval"]["candidate_count"] == 1
    assert packet_out.trace["retrieval"]["embedding_candidate_count"] == 0


def test_embedding_similarity_retrieval_source_retrieved_uses_query_embedding() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    first = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="default",
        text="closest",
        timestamp="2026-01-01T00:00:00+00:00",
        embedding=[1.0, 0.0],
    )
    second = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="default",
        text="second",
        timestamp="2026-01-01T00:00:01+00:00",
        embedding=[0.0, 1.0],
    )
    outside = MemoryRecord(
        record_id="rec-3",
        unit_id="unit-3",
        layer="default",
        text="outside subset",
        timestamp="2026-01-01T00:00:02+00:00",
        embedding=[0.99, 0.01],
    )
    for record in (first, second, outside):
        store.append(record)

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=2, source="retrieved").run(
        Packet(
            query=Query(text="ignored", embedding=[0.0, 1.0]),
            retrieved=RetrievedSet(items=[first, second], scores=[]),
        ),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert packet_out.retrieved.trace["source"] == "retrieved"
    assert packet_out.retrieved.trace["candidate_count"] == 2
    assert packet_out.retrieved.trace["reused_query_embedding"] is True


def test_embedding_similarity_retrieval_source_retrieved_computes_query_embedding_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    first = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="default",
        text="closest",
        timestamp="2026-01-01T00:00:00+00:00",
        embedding=[1.0, 0.0],
    )
    second = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="default",
        text="second",
        timestamp="2026-01-01T00:00:01+00:00",
        embedding=[0.0, 1.0],
    )
    for record in (first, second):
        store.append(record)

    def _fake_embed_text(self, text: str) -> list[float]:
        assert text == "alpha query"
        return [1.0, 0.0]

    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "_embed_text", _fake_embed_text)

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1, source="retrieved").run(
        Packet(
            query=Query(text="alpha query"),
            retrieved=RetrievedSet(items=[second, first], scores=[]),
        ),
        store,
    )

    assert packet_out.query.embedding == [1.0, 0.0]
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.retrieved.trace["reused_query_embedding"] is False


def test_embedding_similarity_retrieval_skips_missing_and_mismatched_embeddings() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="usable",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="missing",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=None,
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="default",
            text="wrong-dim",
            timestamp="2026-01-01T00:00:02+00:00",
            embedding=[1.0, 0.0, 0.0],
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=3).run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.trace["retrieval"]["embedding_candidate_count"] == 1
    assert packet_out.trace["retrieval"]["skipped_dim_mismatch_count"] == 1


def test_embedding_similarity_retrieval_can_target_declared_topology_layer() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episode"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="default",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="episodic",
            text="episodic-best",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[1.0, 0.0],
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(top_k=1, layer="episodic").run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2"]
    assert packet_out.trace["retrieval"]["candidate_count"] == 1


def test_organization_can_write_into_declared_non_default_topology_layer() -> None:
    from memprimitive.baselines import AppendOrganization

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",)),
        ]
    )
    store = MemoryStore(topology=topology)
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    packet, store = AppendOrganization(target_layer="episodic").run(
        Packet(
            observation=packet.observation,
            units=packet.units,
            decisions=[True for _ in packet.units or []],
            trace=packet.trace,
        ),
        store,
    )

    assert store.count("episodic") == 1
    assert store.iter_records("episodic")[0].layer == "episodic"


def test_retrieval_can_target_declared_topology_layer() -> None:
    from memprimitive.baselines import AppendOrganization, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="default"),
            StoreLayerSpec(name="episodic", theme="episode"),
        ]
    )
    store = MemoryStore(topology=topology)
    for text in ("episodic first", "episodic second"):
        packet, store = _stored_pipeline_packet(text, store)
        packet, store = AppendOrganization(target_layer="episodic").run(
            Packet(
                observation=packet.observation,
                units=packet.units,
                decisions=[True for _ in packet.units or []],
                trace=packet.trace,
            ),
            store,
        )

    packet_out, _ = RecencyRetrieval(top_k=1, layer="episodic").run(Packet(query=Query(text="episodic")), store)

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["episodic second"]


def test_layer_aware_retrieval_merges_per_layer_results_and_applies_global_top_k() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval, LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="semantic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working hit",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-semantic-1",
            unit_id="unit-semantic-1",
            layer="semantic",
            text="semantic best",
            timestamp="2026-01-01T00:00:01+00:00",
            embedding=[1.0, 0.0],
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-semantic-2",
            unit_id="unit-semantic-2",
            layer="semantic",
            text="semantic weaker",
            timestamp="2026-01-01T00:00:02+00:00",
            embedding=[0.8, 0.2],
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=2),
        retriever_by_layer={"semantic": EmbeddingSimilarityRetrieval(top_k=2)},
        top_k=2,
    ).run(
        Packet(query=Query(text="query", embedding=[1.0, 0.0])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-semantic-1", "rec-semantic-2"]
    assert packet_out.retrieved.scores[0]["merge_rank"] == 1
    assert packet_out.retrieved.scores[0]["merge_key_type"] == "score"
    assert packet_out.retrieved.scores[0]["layer"] == "semantic"
    assert packet_out.trace["retrieval"]["merge_strategy"] == "global_rank"
    assert packet_out.trace["retrieval"]["total_merged_count"] == 3
    assert packet_out.trace["retrieval"]["final_returned_count"] == 2


def test_layer_aware_retrieval_falls_back_to_default_retriever_for_unconfigured_layers() -> None:
    from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working latest",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-episodic-1",
            unit_id="unit-episodic-1",
            layer="episodic",
            text="episodic latest",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=1),
        retriever_by_layer={"working": RecencyRetrieval(top_k=1)},
        top_k=2,
    ).run(Packet(query=Query(text="latest")), store)

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-working-1", "rec-episodic-1"]
    assert [entry["module"] for entry in packet_out.trace["retrieval"]["per_layer"]] == [
        "recency_retrieval",
        "recency_retrieval",
    ]


def test_layer_aware_retrieval_can_limit_active_layers() -> None:
    from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working memory",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-episodic-1",
            unit_id="unit-episodic-1",
            layer="episodic",
            text="episodic memory",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=1),
        active_layers=("episodic",),
        top_k=2,
    ).run(Packet(query=Query(text="memory")), store)

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-episodic-1"]
    assert packet_out.trace["retrieval"]["active_layers"] == ["episodic"]


def test_layer_aware_retrieval_uses_layer_order_to_break_rank_ties() -> None:
    from memprimitive.baselines import LayerAwareRetrieval, RecencyRetrieval

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    store.append(
        MemoryRecord(
            record_id="rec-working-1",
            unit_id="unit-working-1",
            layer="working",
            text="working rank one",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-episodic-1",
            unit_id="unit-episodic-1",
            layer="episodic",
            text="episodic rank one",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=1),
        top_k=2,
    ).run(Packet(query=Query(text="rank")), store)

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-working-1", "rec-episodic-1"]
    assert packet_out.retrieved.scores[0]["merge_key_type"] == "rank"
    assert packet_out.retrieved.scores[1]["merge_key_type"] == "rank"


def test_layer_aware_retrieval_returns_valid_empty_result_for_empty_store() -> None:
    from memprimitive.baselines import LayerAwareRetrieval

    packet_out, store_out = LayerAwareRetrieval(top_k=2).run(
        Packet(query=Query(text="query")),
        MemoryStore(),
    )

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.trace["retrieval"]["per_layer"][0]["candidate_count"] == 0
    assert store_out.count() == 0


def test_layer_aware_retrieval_validates_inputs() -> None:
    from memprimitive.baselines import LayerAwareRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        LayerAwareRetrieval(top_k=0)

    with pytest.raises(ValueError, match="merge_strategy='global_rank'"):
        LayerAwareRetrieval(merge_strategy="round_robin")

    with pytest.raises(TypeError, match="default_retriever"):
        LayerAwareRetrieval(default_retriever=object())

    with pytest.raises(TypeError, match="retriever_by_layer values"):
        LayerAwareRetrieval(retriever_by_layer={"semantic": object()})

    topology = StoreTopology.from_layers([StoreLayerSpec(name="working")])
    store = MemoryStore(topology=topology)
    with pytest.raises(ValueError, match="not declared in the store topology"):
        LayerAwareRetrieval(active_layers=("missing",)).run(Packet(query=Query(text="query")), store)


def test_store_capability_queries_reflect_declared_topology() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working", indices=("keyword",)),
                StoreLayerSpec(name="graph", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )

    assert store.has_graph_layer() is True
    assert store.has_keyword_layer() is True
    assert store.layer_supports_index("graph", "graph") is True


def test_baselines_simple_reexports_match_package_exports() -> None:
    import memprimitive.baselines as pkg
    import memprimitive.baselines.simple as legacy

    assert set(pkg.__all__) == set(legacy.__all__)
    for name in sorted(pkg.__all__):
        assert getattr(pkg, name) is getattr(legacy, name), name


def test_baselines_all_matches_registered_baseline_classes() -> None:
    """``__init__.__all__`` must list exactly the classes registered in per-module ``BASELINE_CLASSES``."""
    import memprimitive.baselines as pkg

    assert set(pkg.__all__) == registered_baseline_class_names()


def test_removed_trigger_family_symbols_are_not_registered() -> None:
    removed = {
        "MetadataGatedWriteTrigger",
        "KeyReadyWriteTrigger",
        "LLMJudgedWriteTrigger",
        "OutcomeConditionedEvolutionTrigger",
        "NewWriteEvolutionTrigger",
        "NeighborExistsEvolutionTrigger",
    }

    assert registered_baseline_class_names().isdisjoint(removed)


def test_write_false_skips_normal_write_and_leaves_evolution_noop() -> None:
    from memprimitive.baselines import (
        AppendOnlyEvolution,
        AppendOrganization,
        BasicRepresentation,
        PassThroughUnitFormation,
    )

    store = MemoryStore()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        store,
    )
    packet, store = BasicRepresentation().run(packet, store)
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[False],
        trace=packet.trace,
    )
    packet, store = AppendOrganization().run(packet, store)
    packet = Packet(
        units=packet.units,
        decisions=[False],
        placements=packet.placements,
        trace=packet.trace,
    )
    packet, store = AppendOnlyEvolution().run(packet, store)

    assert store.count() == 0
    assert packet.trace["organization"]["written_record_ids"] == []
    assert packet.trace["memory_evolution"]["effects"] == []


def test_sentence_split_unit_formation_splits_sentences_and_preserves_provenance() -> None:
    from memprimitive.baselines import SentenceSplitUnitFormation

    packet_out, _ = SentenceSplitUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea. Bob prefers coffee!", source="dialogue")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["Alice likes tea.", "Bob prefers coffee!"]
    assert all("provenance" in unit.metadata for unit in packet_out.units)


def test_line_split_unit_formation_filters_empty_lines() -> None:
    from memprimitive.baselines import LineSplitUnitFormation

    packet_out, _ = LineSplitUnitFormation().run(
        Packet(observation=Observation(text="alpha\n\n beta \n", source="notes")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["alpha", "beta"]


def test_windowed_unit_formation_creates_overlapping_windows() -> None:
    from memprimitive.baselines import WindowedUnitFormation

    packet_out, _ = WindowedUnitFormation(window_size=5, stride=3).run(
        Packet(observation=Observation(text="abcdefghij", source="notes")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["abcde", "defgh", "ghij"]
    assert packet_out.units[1].metadata["window_index"] == 1


def test_metadata_hint_unit_formation_prefers_hint_and_can_set_unit_type() -> None:
    from memprimitive.baselines import MetadataHintUnitFormation

    packet_out, _ = MetadataHintUnitFormation().run(
        Packet(
            observation=Observation(
                text="fallback",
                source="notes",
                metadata={"units": [{"text": "Alice likes tea", "unit_type": "fact"}]},
            )
        ),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["Alice likes tea"]
    assert packet_out.units[0].unit_type == "fact"
    assert packet_out.trace["unit_formation"]["mode"] == "metadata"


def test_representation_supports_new_elements_and_persists_them_into_record_metadata() -> None:
    from memprimitive.baselines import AppendOrganization, AlwaysTrigger, BasicRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice studies graph memory on 2026-03-24.", source="notes")),
        MemoryStore(),
    )
    packet, store = BasicRepresentation(
        elements=("text", "keywords", "time_anchor", "source_type")
    ).run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    _, store = AppendOrganization().run(packet, store)

    record = store.iter_records()[0]
    rep = record.metadata["representation"]
    assert "keywords" in rep
    assert "time_anchor" in rep
    assert rep["source_type"] == "notes"


def test_conditional_layer_organization_routes_entity_rich_units_to_semantic() -> None:
    from memprimitive.baselines import AlwaysTrigger, ConditionalLayerOrganization, LLMRepresentation, PassThroughUnitFormation

    class SeededEntityRepresentation(LLMRepresentation):
        def _llm_json(self, *, user: str) -> Any:
            return ["Alice", "tea"]

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="working"),
                StoreLayerSpec(name="semantic", theme="semantic", indices=("entity", "keyword")),
            ]
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="dialogue")),
        store,
    )
    packet, store = SeededEntityRepresentation(field="entities", prompt="Extract entities.").run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = ConditionalLayerOrganization(
        default_layer="working",
        rules=({"has_entity": True, "target_layer": "semantic"},),
    ).run(packet, store)

    assert packet.placements[0].target_layer == "semantic"
    assert store.count("semantic") == 1


def test_graph_append_organization_requires_graph_layer_and_writes_graph_metadata() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            return represented, {"source": "test_seed", "entities": entities, "triple_count": len(triples)}

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphAppendOrganization(target_layer="knowledge_graph").run(packet, store)

    record = store.iter_records("knowledge_graph")[0]
    assert "graph" in record.metadata
    assert record.metadata["graph"]["triples"]
    assert record.metadata["graph"]["links"] == []
    assert record.metadata["graph"]["link_count"] == 0
    assert packet.trace["organization"]["graph_metadata_schema"]
    assert packet.trace["organization"]["separate"] is False
    assert packet.trace["organization"]["source_written_record_ids"] == []


def test_graph_append_organization_separate_mode_writes_source_and_triple_layers() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            return represented, {"source": "test_seed", "entities": entities, "triple_count": len(triples)}

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="source_notes"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphAppendOrganization(
        target_layer="knowledge_graph",
        separate=True,
        separate_layer="source_notes",
    ).run(packet, store)

    source_record = store.iter_records("source_notes")[0]
    triple_record = store.iter_records("knowledge_graph")[0]
    assert source_record.text == "Alice likes tea."
    assert "graph" in triple_record.metadata
    assert "hierarchical" in triple_record.metadata
    assert triple_record.metadata["hierarchical"]["source_layer"] == "source_notes"
    assert triple_record.metadata["hierarchical"]["target_layer"] == "knowledge_graph"
    assert triple_record.metadata["hierarchical"]["source_record_ids"] == [source_record.record_id]
    assert triple_record.metadata["hierarchical"]["source_unit_ids"] == [source_record.unit_id]
    assert triple_record.metadata["hierarchical"]["field_payload"]["triples"] == [("Alice", "likes", "tea")]
    assert triple_record.metadata["hierarchical"]["relation"] == "hierarchical_extracted_triple"
    assert packet.trace["organization"]["separate"] is True
    assert packet.trace["organization"]["separate_layer"] == "source_notes"
    assert packet.trace["organization"]["source_written_record_ids"] == [source_record.record_id]
    assert packet.trace["organization"]["triple_written_record_ids"] == [triple_record.record_id]


def test_graph_append_organization_separate_mode_requires_separate_layer() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphAppendOrganization, PassThroughUnitFormation

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = AlwaysTrigger().run(packet, store)

    with pytest.raises(ValueError, match="requires separate_layer"):
        GraphAppendOrganization(target_layer="knowledge_graph", separate=True).run(packet, store)


def test_memory_store_graph_link_round_trip_returns_neighbors() -> None:
    store = _graph_store()
    first = MemoryRecord(record_id="rec-1", unit_id="unit-1", layer="knowledge_graph", text="Alice likes tea", timestamp="t1")
    second = MemoryRecord(record_id="rec-2", unit_id="unit-2", layer="knowledge_graph", text="Alice studies graphs", timestamp="t2")
    store.append(first)
    store.append(second)

    merged_links = store.add_graph_links("knowledge_graph", "rec-2", ["rec-1"])
    neighbors = store.iter_graph_neighbors("knowledge_graph", "rec-2")

    assert merged_links == ["rec-1"]
    assert [record.record_id for record in neighbors] == ["rec-1"]


def test_graph_neighbor_retrieval_handles_missing_and_present_links() -> None:
    from memprimitive.baselines import GraphNeighborRetrieval

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-seed",
        unit_id="unit-seed",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    store.append(seed)
    store.append(neighbor)

    empty_packet, _ = GraphNeighborRetrieval(top_k=3).run(
        Packet(query=Query(text="Alice", metadata={"graph_seed_record_ids": ["rec-seed"]})),
        store,
    )
    assert empty_packet.retrieved.items == []

    store.add_graph_links("knowledge_graph", "rec-seed", ["rec-neighbor"])
    linked_packet, _ = GraphNeighborRetrieval(top_k=3).run(
        Packet(query=Query(text="Alice", metadata={"graph_seed_record_ids": ["rec-seed"]})),
        store,
    )

    assert [record.record_id for record in linked_packet.retrieved.items] == ["rec-neighbor"]
    assert linked_packet.trace["retrieval"]["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_graph_seed_and_expand_retrieval_uses_candidate_set_and_neighbor_expansion() -> None:
    from memprimitive.baselines import GraphSeedAndExpandRetrieval

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-seed",
        unit_id="unit-seed",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": ["rec-neighbor"]}},
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    other = MemoryRecord(
        record_id="rec-other",
        unit_id="unit-other",
        layer="knowledge_graph",
        text="Bob studies memory retrieval",
        timestamp="2026-03-27T00:02:00+00:00",
        metadata={"graph": {"entities": ["Bob"], "links": []}},
    )
    store.append(seed)
    store.append(neighbor)
    store.append(other)

    packet_out, _ = GraphSeedAndExpandRetrieval(top_k=3, seed_top_k=1).run(
        Packet(
            query=Query(
                text="Alice graph",
                metadata={"graph_candidate_record_ids": ["rec-seed", "rec-neighbor"]},
            )
        ),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-seed", "rec-neighbor"]
    assert packet_out.trace["retrieval"]["seed_record_ids"] == ["rec-seed"]
    assert packet_out.trace["retrieval"]["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_expand_retrieved_graph_neighbors_adds_neighbors_from_retrieved_seeds() -> None:
    from memprimitive.baselines import ExpandRetrievedGraphNeighbors

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-seed",
        unit_id="unit-seed",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"links": ["rec-neighbor"]}},
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"links": []}},
    )
    store.append(seed)
    store.append(neighbor)

    packet_out, _ = ExpandRetrievedGraphNeighbors(top_k=3, layer="knowledge_graph").run(
        Packet(retrieved=RetrievedSet(items=[seed], scores=[])),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-seed", "rec-neighbor"]
    assert packet_out.retrieved.scores[0]["strategy"] == "graph_seed"
    assert packet_out.retrieved.scores[1]["strategy"] == "graph_expand_retrieved"
    assert packet_out.retrieved.trace["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_expand_retrieved_graph_neighbors_dedupes_and_filters_non_target_layers() -> None:
    from memprimitive.baselines import ExpandRetrievedGraphNeighbors

    store = _graph_store()
    seed_a = MemoryRecord(
        record_id="rec-seed-a",
        unit_id="unit-seed-a",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"links": ["rec-neighbor"]}},
    )
    seed_b = MemoryRecord(
        record_id="rec-seed-b",
        unit_id="unit-seed-b",
        layer="knowledge_graph",
        text="Alice studies retrieval",
        timestamp="2026-03-27T00:00:01+00:00",
        metadata={"graph": {"links": ["rec-neighbor"]}},
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Shared neighbor",
        timestamp="2026-03-27T00:00:02+00:00",
        metadata={"graph": {"links": []}},
    )
    other_layer = MemoryRecord(
        record_id="rec-other",
        unit_id="unit-other",
        layer="default",
        text="Other layer seed",
        timestamp="2026-03-27T00:00:03+00:00",
    )
    for record in (seed_a, seed_b, neighbor, other_layer):
        store.append(record)

    packet_out, _ = ExpandRetrievedGraphNeighbors(
        top_k=5,
        layer="knowledge_graph",
        include_seed_records=False,
        dedupe=True,
    ).run(
        Packet(retrieved=RetrievedSet(items=[seed_a, other_layer, seed_b], scores=[])),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-neighbor"]
    assert packet_out.retrieved.trace["seed_record_ids"] == ["rec-seed-a", "rec-seed-b"]
    assert packet_out.retrieved.trace["returned_count"] == 1


def test_expand_retrieved_graph_neighbors_returns_empty_when_no_seeds() -> None:
    from memprimitive.baselines import ExpandRetrievedGraphNeighbors

    packet_out, _ = ExpandRetrievedGraphNeighbors(top_k=3).run(Packet(retrieved=RetrievedSet(items=[], scores=[])), _graph_store())

    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.retrieved.trace["seed_record_ids"] == []


def test_graph_neighbor_append_evolution_only_modifies_graph_layer() -> None:
    from memprimitive.baselines import GraphNeighborAppendEvolution

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-working",
            unit_id="unit-working",
            layer="default",
            text="Working memory note",
            timestamp="2026-03-27T00:00:00+00:00",
        )
    )
    existing = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    incoming = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    store.append(existing)
    store.append(incoming)

    packet = Packet(
        units=[MemoryUnit(text="Alice studies graph memory", unit_id="unit-2")],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        decisions=[True],
    )

    packet_out, store = GraphNeighborAppendEvolution(target_layer="knowledge_graph", neighbor_limit=1).run(packet, store)

    updated_graph_records = store.iter_records("knowledge_graph")
    updated_incoming = [record for record in updated_graph_records if record.record_id == "rec-2"][0]
    assert updated_incoming.metadata["graph"]["links"] == ["rec-1"]
    assert store.iter_records("default")[0].record_id == "rec-working"
    assert packet_out.trace["memory_evolution"]["effects"][0]["target_layer"] == "knowledge_graph"


def test_graph_link_evolution_rewrites_only_graph_metadata_namespace() -> None:
    from memprimitive.baselines import GraphLinkEvolution

    store = _graph_vector_store()
    existing = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:00:00+00:00",
        embedding=[1.0, 0.0],
        metadata={"owner": "kept", "graph": {"entities": ["Alice"], "links": []}},
    )
    incoming = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:01:00+00:00",
        embedding=[0.95, 0.05],
        metadata={"owner": "kept", "graph": {"entities": ["Alice"], "links": []}},
    )
    store.append(existing)
    store.append(incoming)

    packet = Packet(
        units=[MemoryUnit(text="Alice studies graph memory", unit_id="unit-2", embedding=[0.95, 0.05])],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        decisions=[True],
    )

    packet_out, store = GraphLinkEvolution(
        target_layer="knowledge_graph",
        neighbor_limit=1,
        rewrite_neighbor_metadata=True,
    ).run(packet, store)

    updated = [record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-2"][0]
    assert updated.metadata["owner"] == "kept"
    assert updated.metadata["graph"]["links"] == ["rec-1"]
    assert updated.metadata["graph"]["neighbor_context"]["neighbor_record_ids"] == ["rec-1"]
    assert packet_out.trace["memory_evolution"]["effects"][0]["candidate_scores"][0]["record_id"] == "rec-1"


def test_graph_neighbor_context_trace_evolution_can_run_trace_only_or_rewrite() -> None:
    from memprimitive.baselines import GraphNeighborContextTraceEvolution

    store = _graph_store()
    seed = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice likes jasmine tea",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": []}},
    )
    current = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:01:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": ["rec-1"]}},
    )
    store.append(seed)
    store.append(current)

    packet = Packet(
        units=[MemoryUnit(text="Alice studies graph memory", unit_id="unit-2")],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        decisions=[True],
    )

    trace_packet, store = GraphNeighborContextTraceEvolution(target_layer="knowledge_graph").run(packet, store)
    assert trace_packet.trace["memory_evolution"]["effects"][0]["neighbor_record_ids"] == ["rec-1"]
    assert "neighbor_context" not in store.iter_records("knowledge_graph")[1].metadata["graph"]

    rewrite_packet, store = GraphNeighborContextTraceEvolution(
        target_layer="knowledge_graph",
        rewrite_metadata=True,
    ).run(packet, store)
    assert rewrite_packet.trace["memory_evolution"]["effects"][0]["rewrite_metadata"] is True
    assert store.iter_records("knowledge_graph")[1].metadata["graph"]["neighbor_context"]["neighbor_record_ids"] == ["rec-1"]


def test_graph_readout_renders_graph_metadata() -> None:
    from memprimitive.baselines import GraphReadout

    record = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="knowledge_graph",
        text="Alice studies graph memory",
        timestamp="2026-03-27T00:00:00+00:00",
        metadata={"graph": {"entities": ["Alice"], "links": ["rec-0"]}},
    )
    packet_out, _ = GraphReadout().run(Packet(retrieved=RetrievedSet(items=[record], scores=[])), _graph_store())

    assert "entities=Alice" in packet_out.readout.text
    assert "links=rec-0" in packet_out.readout.text
    assert packet_out.readout.metadata["graph_item_count"] == 1


def test_graph_baseline_pipeline_end_to_end_supports_threshold_trigger_evolution_retrieval_and_readout() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import (
        BasicRepresentation,
        GraphAppendOrganization,
        GraphLinkEvolution,
        GraphNeighborContextTraceEvolution,
        GraphReadout,
        GraphSeedAndExpandRetrieval,
        LLMRepresentation,
        PassThroughUnitFormation,
        ThresholdTrigger,
        TripleRepresentation,
    )

    class SeededTripleRepresentation(TripleRepresentation):
        _TRIPLES_BY_TEXT = {
            "Alice likes jasmine tea.": ([("Alice", "likes", "jasmine tea")], ["Alice", "jasmine tea"]),
            "Alice studies graph memory systems.": (
                [("Alice", "studies", "graph memory systems")],
                ["Alice", "graph memory systems"],
            ),
            "Bob builds retrieval tools.": ([("Bob", "builds", "retrieval tools")], ["Bob", "retrieval tools"]),
        }

        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples, entities = self._TRIPLES_BY_TEXT[unit.text.strip()]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            return represented, {"source": "test_seed", "entities": entities, "triple_count": len(triples)}

    class SeededTagRepresentation(LLMRepresentation):
        _TAGS_BY_TEXT = {
            "Alice likes jasmine tea.": ["preference", "tea"],
            "Alice studies graph memory systems.": ["graph", "memory"],
            "Bob builds retrieval tools.": ["retrieval", "tools"],
        }

        def _llm_json(self, *, user: str) -> Any:
            payload = json.loads(user)
            return list(self._TAGS_BY_TEXT[payload["unit"]["text"]])

    store = _graph_vector_store()
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=(
            BasicRepresentation(elements=("text", "embedding")),
            SeededTripleRepresentation(),
            BasicRepresentation(elements=("keywords",)),
            SeededTagRepresentation(field="tags", prompt="Extract tags."),
        ),
        organization=GraphAppendOrganization(target_layer="knowledge_graph"),
        evolution_trigger=ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0),
        memory_evolution=(
            GraphLinkEvolution(target_layer="knowledge_graph", neighbor_limit=2, rewrite_neighbor_metadata=True),
            GraphNeighborContextTraceEvolution(target_layer="knowledge_graph", rewrite_metadata=True),
        ),
        retrieval=GraphSeedAndExpandRetrieval(top_k=4, layer="knowledge_graph", seed_top_k=1),
        readout=GraphReadout(),
        store=store,
    )

    first_packet = pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
    second_packet = pipeline.ingest(Observation(text="Alice studies graph memory systems.", source="notes"))
    pipeline.ingest(Observation(text="Bob builds retrieval tools.", source="notes"))
    readout = pipeline.recall(Query(text="Alice graph"))

    graph_records = pipeline.store.iter_records("knowledge_graph")
    linked_record = [record for record in graph_records if record.unit_id == second_packet.units[0].unit_id][0]

    assert first_packet.trace["write_trigger"]["decisions"] == [True]
    assert second_packet.trace["write_trigger"]["decisions"] == [True]
    assert first_packet.decisions == [True]
    assert second_packet.decisions == [True]
    assert linked_record.metadata["graph"]["links"]
    assert linked_record.metadata["graph"]["neighbor_context"]["neighbor_record_ids"]
    assert "Alice studies graph memory systems." in readout.text or "Alice likes jasmine tea." in readout.text
    assert readout.source_ids


def test_semantic_field_enrichment_and_retrieval_embedding_repair_note_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import RetrievalOrientedEmbeddingRepresentation, SemanticFieldEnrichmentRepresentation

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    packet = Packet(
        units=[
            MemoryUnit(
                text="Alice likes tea.",
                metadata={"amem": {"context": "Alice routine only", "keywords": ["alice", "tea"]}},
            )
        ]
    )

    packet, store = SemanticFieldEnrichmentRepresentation(note_namespace="amem").run(packet, MemoryStore())
    packet, _ = RetrievalOrientedEmbeddingRepresentation(note_namespace="amem").run(packet, store)

    unit = packet.units[0]
    assert unit.metadata["amem"]["note_text"].startswith("Comprehensive note:")
    assert unit.metadata["representation"]["enhanced_embedding_text"].startswith("content: Alice likes tea.")
    assert unit.embedding == _runtime._DEFAULT_RUNTIME.embed(unit.metadata["representation"]["enhanced_embedding_text"])


def test_graph_append_link_ready_organization_does_not_eagerly_validate_graph_vector_layer() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import GraphAppendLinkReadyOrganization

    bad_store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory_graph", shape="Graph", indices=("graph", "keyword", "tag"))]
        )
    )

    pipeline = MemoryPipeline(
        store=bad_store,
        organization=GraphAppendLinkReadyOrganization(target_layer="memory_graph"),
    )

    assert isinstance(pipeline.organization, GraphAppendLinkReadyOrganization)


def test_vector_graph_seed_and_expand_retrieval_expands_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import VectorGraphSeedAndExpandRetrieval

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    store = _graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-seed",
            unit_id="unit-seed",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=_runtime._DEFAULT_RUNTIME.embed("content: Alice likes tea."),
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "context": "Alice's tea habit supports her daily routine.",
                    "enhanced_embedding_text": "content: Alice likes tea.",
                },
                "graph": {"entities": ["Alice"], "links": ["rec-neighbor"]},
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-neighbor",
            unit_id="unit-neighbor",
            layer="knowledge_graph",
            text="Tea routines improve focus.",
            timestamp="2026-03-27T00:00:01+00:00",
            embedding=_runtime._DEFAULT_RUNTIME.embed("content: Tea routines improve focus."),
            metadata={
                "amem": {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                },
                "representation": {
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "context": "Tea routines are linked to improved focus.",
                    "enhanced_embedding_text": "content: Tea routines improve focus.",
                },
                "graph": {"entities": ["Tea"], "links": []},
            },
        )
    )

    packet_out, _ = VectorGraphSeedAndExpandRetrieval(
        top_k=2,
        layer="knowledge_graph",
        candidate_k=1,
        neighbor_expansion_k=1,
        note_namespace="amem",
    ).run(Packet(query=Query(text="Alice")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-seed", "rec-neighbor"]
    assert packet_out.retrieved.trace["expanded_neighbor_ids"] == ["rec-neighbor"]


def test_vector_graph_seed_and_expand_retrieval_system_prompt_template_renders_query_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import VectorGraphSeedAndExpandRetrieval
    from memprimitive.utils._template import text_prompt

    fake_runtime = _FakeAMEMRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = _graph_vector_store()
    store.append(
        MemoryRecord(
            record_id="rec-seed",
            unit_id="unit-seed",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=fake_runtime.embed("content: Alice likes tea."),
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {"enhanced_embedding_text": "content: Alice likes tea."},
                "graph": {"entities": ["Alice"], "links": []},
            },
        )
    )

    packet_out, _ = VectorGraphSeedAndExpandRetrieval(
        top_k=1,
        layer="knowledge_graph",
        candidate_k=1,
        neighbor_expansion_k=1,
        note_namespace="amem",
        query_expand_with_llm=True,
        prompt=text_prompt(
            "Expand {{ query.text }} for {{ retrieval.layer }} "
            "with candidate_k={{ retrieval.candidate_k }} and now={{ runtime.now }}"
        ),
    ).run(Packet(query=Query(text="Alice")), store)

    assert packet_out.retrieved.trace["query_expand_with_llm"] is True
    assert packet_out.retrieved.trace["system_prompt_is_template"] is True
    assert "Expand Alice for knowledge_graph with candidate_k=1" in packet_out.retrieved.trace["query_expansion_prompt_trace"]["rendered_prompt"]


def test_vector_graph_seed_and_expand_retrieval_system_prompt_template_supports_recalled_prompt_from_current_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import ConcatenateReadout, RecencyRetrieval, VectorGraphSeedAndExpandRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    fake_runtime = _FakeAMEMRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = _graph_vector_store()
    _seed_layer(store, "default", ["CURRENT STORE CONTEXT"])
    store.append(
        MemoryRecord(
            record_id="rec-seed",
            unit_id="unit-seed",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=fake_runtime.embed("content: Alice likes tea."),
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {"enhanced_embedding_text": "content: Alice likes tea."},
                "graph": {"entities": ["Alice"], "links": []},
            },
        )
    )

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "default", ["WRONG PIPELINE STORE CONTEXT"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="default"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    packet_out, _ = VectorGraphSeedAndExpandRetrieval(
        top_k=1,
        layer="knowledge_graph",
        candidate_k=1,
        neighbor_expansion_k=1,
        note_namespace="amem",
        query_expand_with_llm=True,
        prompt=text_prompt(
            "Expand {{ query.text }} for {{ retrieval.layer }} with {{ recalled_prompt }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"context for {context['query']['text']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
    ).run(Packet(query=Query(text="Alice")), store)

    prompt_trace = packet_out.retrieved.trace["query_expansion_prompt_trace"]
    assert "CURRENT STORE CONTEXT" in prompt_trace["rendered_prompt"]
    assert "WRONG PIPELINE STORE CONTEXT" not in prompt_trace["rendered_prompt"]
    assert prompt_trace["recall_prompt"]["rendered_recall_query"] == "context for Alice"


def test_link_strengthening_and_neighbor_update_write_back_graph_and_note_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import LinkStrengtheningEvolution, NeighborContextUpdateEvolution

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _FakeAMEMRuntime())
    store = _graph_vector_store()
    first_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Alice likes tea.")
    second_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Tea routines improve focus.")
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=first_embedding,
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {"enhanced_embedding_text": "content: Alice likes tea."},
                "graph": {"entities": ["Alice"], "links": []},
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Tea routines improve focus.",
            timestamp="2026-03-27T00:00:01+00:00",
            embedding=second_embedding,
            metadata={
                "amem": {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                },
                "representation": {"enhanced_embedding_text": "content: Tea routines improve focus."},
                "graph": {"entities": ["Tea"], "links": []},
            },
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Tea routines improve focus.", unit_id="unit-2", embedding=second_embedding)],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        decisions=[True],
    )

    packet, store = LinkStrengtheningEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)
    packet, store = NeighborContextUpdateEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)

    current = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-2")
    neighbor = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-1")
    assert current.metadata["graph"]["links"] == ["rec-1"]
    assert neighbor.metadata["amem"]["context"] == "Alice's tea habit is now understood as a focus-supporting routine."
    assert neighbor.metadata["amem"]["tags"] == ["preference", "habit", "focus"]


def test_amem_evolution_repairs_list_shaped_llm_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import LinkStrengtheningEvolution, NeighborContextUpdateEvolution

    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", _WrapperShapeAMEMRuntime())
    store = _graph_vector_store()
    first_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Alice likes tea.")
    second_embedding = _runtime._DEFAULT_RUNTIME.embed("content: Tea routines improve focus.")
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-03-27T00:00:00+00:00",
            embedding=first_embedding,
            metadata={
                "amem": {
                    "content": "Alice likes tea.",
                    "note_text": "Comprehensive note: Alice likes tea and keeps a steady routine.",
                    "context": "Alice's tea habit supports her daily routine.",
                    "keywords": ["alice", "tea", "routine"],
                    "tags": ["preference", "habit", "beverage"],
                    "category": "personal_preference",
                    "attributes": {"person": "Alice"},
                },
                "representation": {"enhanced_embedding_text": "content: Alice likes tea."},
                "graph": {"entities": ["Alice"], "links": []},
            },
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Tea routines improve focus.",
            timestamp="2026-03-27T00:00:01+00:00",
            embedding=second_embedding,
            metadata={
                "amem": {
                    "content": "Tea routines improve focus.",
                    "note_text": "Comprehensive note: Tea routines improve focus during reflective work.",
                    "context": "Tea routines are linked to improved focus.",
                    "keywords": ["tea", "focus", "routine"],
                    "tags": ["productivity", "habit", "focus"],
                    "category": "insight",
                    "attributes": {"topic": "focus"},
                },
                "representation": {"enhanced_embedding_text": "content: Tea routines improve focus."},
                "graph": {"entities": ["Tea"], "links": []},
            },
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="Tea routines improve focus.", unit_id="unit-2", embedding=second_embedding)],
        placements=[Placement(unit_id="unit-2", target_layer="knowledge_graph")],
        decisions=[True],
    )

    packet, store = LinkStrengtheningEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)
    packet, store = NeighborContextUpdateEvolution(target_layer="knowledge_graph", note_namespace="amem").run(packet, store)

    current = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-2")
    neighbor = next(record for record in store.iter_records("knowledge_graph") if record.record_id == "rec-1")
    assert current.metadata["graph"]["links"] == ["rec-1"]
    assert neighbor.metadata["amem"]["context"] == "Alice's tea habit is now understood as a focus-supporting routine."
    assert neighbor.metadata["amem"]["tags"] == ["preference", "habit", "focus"]


def test_summary_rewrite_evolution_appends_summary_record() -> None:
    from memprimitive.baselines import SummaryRewriteEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    packet, store = _stored_pipeline_packet("Alice likes jasmine tea.", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = SummaryRewriteEvolution(target_layer="semantic").run(packet, store)

    assert store.count("semantic") == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["effect_type"] == "summary_append"


def test_layer_move_evolution_copy_appends_unit_to_target_layer() -> None:
    from memprimitive.baselines import LayerMoveEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    packet, store = _stored_pipeline_packet("Alice likes jasmine tea.", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = LayerMoveEvolution(target_layer="semantic").run(packet, store)

    assert store.count("semantic") == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["move_style"] == "copy_append"


def test_hierarchical_organization_copy_uses_decisions_store_selection() -> None:
    from memprimitive.baselines import HierarchicalOrganization

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "doc-a note", "metadata": {"session_id": "sess-1", "doc_id": "doc-a", "subgoal_id": "sg-1"}},
            {"text": "doc-b note", "metadata": {"session_id": "sess-2", "doc_id": "doc-b", "subgoal_id": "sg-2"}},
        ],
    )
    packet, _ = _represented_packet("incoming note")
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {
                "decision": True,
                "record_ids": ["rec-1"],
                "selector": {"kind": "boundary_match"},
            }
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id", "subgoal_id"),
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 1
    assert packet_out.placements is not None
    assert packet_out.placements[0].target_layer == "semantic"
    assert packet_out.trace["organization"]["selection_source"] == "decisions_store"
    assert packet_out.trace["organization"]["selected_record_count"] == 1
    assert packet_out.trace["organization"]["append_current_units"] is False
    assert packet_out.trace["organization"]["write_mode"] == "memory_pipeline_ingest"
    assert packet_out.trace["organization"]["writer_pipeline_mode"] == "default_target_layer"
    assert packet_out.trace["organization"]["written_record_ids"] == ["rec-3"]
    assert packet_out.trace["organization"]["sub_ingest_trace"][0]["organization"]["target_layer"] == "semantic"
    assert written[0].metadata["hierarchical"]["source_record_ids"] == ["rec-1"]
    assert written[0].metadata["hierarchical"]["field_payload"]["doc_id"] == "doc-a"


def test_hierarchical_evolution_copy_uses_evolution_decisions_store_selection() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "doc-a note", "metadata": {"session_id": "sess-1", "doc_id": "doc-a"}},
            {"text": "doc-b note", "metadata": {"session_id": "sess-2", "doc_id": "doc-b"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {
                "decision": True,
                "record_ids": ["rec-2"],
                "selector": {"kind": "boundary_match"},
            }
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 1
    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions_store"
    assert packet_out.trace["memory_evolution"]["selected_record_count"] == 1
    assert packet_out.trace["memory_evolution"]["write_mode"] == "memory_pipeline_ingest"
    assert packet_out.trace["memory_evolution"]["writer_pipeline_mode"] == "default_target_layer"
    assert packet_out.trace["memory_evolution"]["effects"][0]["source_record_ids"] == ["rec-2"]
    assert packet_out.trace["memory_evolution"]["effects"][0]["sub_ingest_trace"]["organization"]["target_layer"] == "semantic"
    assert written[0].text == "doc-b"


def test_hierarchical_copy_grouping_and_dedup_preserve_unique_values() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1", "doc_id": "doc-a"}},
            {"text": "a2", "metadata": {"session_id": "sess-1", "doc_id": "doc-a"}},
            {"text": "b1", "metadata": {"session_id": "sess-2", "doc_id": "doc-b"}},
            {"text": "b2", "metadata": {"session_id": "sess-2", "doc_id": "doc-c"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {
                "decision": True,
                "record_ids": ["rec-1", "rec-2", "rec-3", "rec-4"],
                "selector": {"kind": "boundary_match"},
            }
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        group_by=("session_id",),
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 2
    assert packet_out.trace["memory_evolution"]["group_count"] == 2
    first_payload = written[0].metadata["hierarchical"]["field_payload"]["doc_id"]
    second_payload = written[1].metadata["hierarchical"]["field_payload"]["doc_id"]
    assert first_payload == "doc-a"
    assert second_payload == ["doc-b", "doc-c"]
    assert written[0].metadata["hierarchical"]["group_key"] == {"session_id": "sess-1"}
    assert written[1].metadata["hierarchical"]["group_key"] == {"session_id": "sess-2"}


def test_hierarchical_modules_fall_back_to_source_layer_scan_when_decisions_store_missing() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"doc_id": "doc-a"}},
            {"text": "a2", "metadata": {"doc_id": "doc-b"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        target_layer="semantic",
    ).run(packet, store)

    assert packet_out.trace["memory_evolution"]["decision_source"] == "source_layer_scan"
    assert packet_out.trace["memory_evolution"]["selected_record_count"] == 3
    assert store.count("semantic") == 1
    assert store.iter_records("semantic")[0].metadata["hierarchical"]["field_payload"]["doc_id"] == ["doc-a", "doc-b", None]


def test_hierarchical_modules_noop_when_decisions_store_excludes_source_layer() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(store, "default", [{"text": "a1", "metadata": {"doc_id": "doc-a"}}])
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "other": {"decision": True, "record_ids": ["rec-1"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        target_layer="semantic",
    ).run(packet, store)

    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions_store"
    assert packet_out.trace["memory_evolution"]["selected_record_count"] == 0
    assert packet_out.trace["memory_evolution"]["effects"] == []
    assert store.count("semantic") == 0


def test_hierarchical_generate_mode_supports_default_and_custom_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1", "rec-2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary", "profile"),
        group_by=("session_id",),
        prompt="CUSTOM HIERARCHICAL PROMPT",
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 1
    assert fake_runtime.calls
    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT"
    assert written[0].text == "summary: custom::summary::sess-1::2\nprofile: custom::profile::sess-1::2"
    assert written[0].metadata["hierarchical"]["field_payload"]["summary"] == "custom::summary::sess-1::2"
    assert packet_out.trace["memory_evolution"]["active_group_keys"] == [{"session_id": "sess-1"}]


def test_hierarchical_generate_mode_uses_default_prompt_when_custom_prompt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(store, "default", [{"text": "a1", "metadata": {"session_id": "sess-1"}}])
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    _, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] != "CUSTOM HIERARCHICAL PROMPT"
    assert "higher-level hierarchical memory record" in fake_runtime.calls[0]["system"]
    assert store.iter_records("semantic")[0].text == "generated::summary::all::1"


def test_hierarchical_organization_generate_mode_supports_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalOrganization
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, _ = _represented_packet("incoming note")
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1", "rec-2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt="CUSTOM HIERARCHICAL PROMPT {{ group_key.session_id }} / {{ record_count }} / {{ records | length }}",
        target_layer="semantic",
    ).run(packet, store)

    assert "CUSTOM HIERARCHICAL PROMPT sess-1 / 2 / 2" == fake_runtime.calls[0]["system"]
    assert packet_out.trace["organization"]["prompt_is_template"] is True
    assert packet_out.trace["organization"]["prompt_trace"][0]["rendered_prompt"] == fake_runtime.calls[0]["system"]
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_organization_generate_mode_supports_recalled_prompt_from_current_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import ConcatenateReadout, HierarchicalOrganization, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils import _runtime
    from memprimitive.utils._template import text_prompt

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="semantic", theme="semantic"),
            ]
        )
    )
    _seed_layer(store, "profile", ["CURRENT STORE MEMORY"])
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, _ = _represented_packet("incoming note")
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-2", "rec-3"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "profile", ["WRONG PIPELINE STORE MEMORY"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    packet_out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt=text_prompt(
            "CUSTOM HIERARCHICAL PROMPT {{ recalled_prompt }} / {{ group_key.session_id }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"memory for {context['group_key']['session_id']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT CURRENT STORE MEMORY / sess-1"
    assert packet_out.trace["organization"]["prompt_trace"][0]["recall_prompt"]["rendered_recall_query"] == "memory for sess-1"
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_evolution_generate_mode_supports_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1", "rec-2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt="CUSTOM HIERARCHICAL PROMPT {{ source_layer }} -> {{ target_layer }} / {{ group_key.session_id }} / {{ record_count }}",
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT default -> semantic / sess-1 / 2"
    assert packet_out.trace["memory_evolution"]["prompt_is_template"] is True
    assert packet_out.trace["memory_evolution"]["prompt_trace"][0]["rendered_prompt"] == fake_runtime.calls[0]["system"]
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_evolution_generate_mode_supports_recalled_prompt_from_current_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import ConcatenateReadout, HierarchicalEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils import _runtime
    from memprimitive.utils._template import text_prompt

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="semantic", theme="semantic"),
            ]
        )
    )
    _seed_layer(store, "profile", ["CURRENT STORE MEMORY"])
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-2", "rec-3"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "profile", ["WRONG PIPELINE STORE MEMORY"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt=text_prompt(
            "CUSTOM HIERARCHICAL PROMPT {{ recalled_prompt }} / {{ source_layer }} -> {{ target_layer }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"memory for {context['group_key']['session_id']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT CURRENT STORE MEMORY / default -> semantic"
    assert packet_out.trace["memory_evolution"]["prompt_trace"][0]["recall_prompt"]["rendered_recall_query"] == "memory for sess-1"
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_constructors_require_exactly_one_target_or_pipeline() -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.pipeline import create_baseline_pipeline

    with pytest.raises(ValueError, match="Exactly one of target_layer or memory_pipeline"):
        HierarchicalEvolution(
            source_layer="default",
            extract_mode="copy",
            extract_fields=("doc_id",),
        )

    with pytest.raises(ValueError, match="Exactly one of target_layer or memory_pipeline"):
        HierarchicalEvolution(
            source_layer="default",
            extract_mode="copy",
            extract_fields=("doc_id",),
            target_layer="semantic",
            memory_pipeline=create_baseline_pipeline(),
        )


def test_hierarchical_memory_pipeline_mode_reuses_parent_store_and_custom_route() -> None:
    from memprimitive.baselines import AlwaysTrigger, AppendOrganization, HierarchicalEvolution
    from memprimitive.pipeline import MemoryPipeline

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="semantic", theme="semantic"),
                StoreLayerSpec(name="profile", theme="semantic"),
            ]
        )
    )
    _seed_layer_with_metadata(store, "default", [{"text": "a1", "metadata": {"doc_id": "doc-a"}}])
    child_pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="profile"),
        store=MemoryStore(),
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        memory_pipeline=child_pipeline,
    ).run(packet, store)

    assert child_pipeline.store is store
    assert store.count("profile") == 1
    assert store.count("semantic") == 0
    assert packet_out.trace["memory_evolution"]["writer_pipeline_mode"] == "provided"
    assert packet_out.trace["memory_evolution"]["target_layer"] == "profile"
    assert packet_out.trace["memory_evolution"]["effects"][0]["sub_ingest_trace"]["organization"]["target_layer"] == "profile"
    assert store.iter_records("profile")[0].metadata["hierarchical"]["field_payload"]["doc_id"] == "doc-a"


def test_keyword_count_retrieval_prefers_keyword_hits() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob likes coffee", "Alice studies graphs"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = KeywordCountRetrieval(top_k=2).run(Packet(query=Query(text="Alice graphs")), store)

    assert [record.text for record in packet_out.retrieved.items] == ["Alice studies graphs", "Alice likes tea"]


def test_keyword_count_retrieval_source_store_preserves_default_behavior() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob likes coffee", "Alice studies graphs"):
        _, store = _stored_pipeline_packet(text, store)

    default_packet, _ = KeywordCountRetrieval(top_k=2).run(Packet(query=Query(text="Alice graphs")), store)
    explicit_store_packet, _ = KeywordCountRetrieval(top_k=2, source="store").run(Packet(query=Query(text="Alice graphs")), store)

    assert [record.record_id for record in explicit_store_packet.retrieved.items] == [
        record.record_id for record in default_packet.retrieved.items
    ]
    assert explicit_store_packet.retrieved.trace["source"] == "store"


def test_keyword_count_retrieval_source_retrieved_only_reranks_candidate_subset() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    candidate_a = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="default",
        text="tea note",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"representation": {"keywords": ["tea"]}},
    )
    candidate_b = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="default",
        text="graph note",
        timestamp="2026-01-01T00:00:01+00:00",
        metadata={"representation": {"keywords": ["graph"]}},
    )
    better_outside = MemoryRecord(
        record_id="rec-3",
        unit_id="unit-3",
        layer="default",
        text="graph memory retrieval",
        timestamp="2026-01-01T00:00:02+00:00",
        metadata={"representation": {"keywords": ["graph", "memory", "retrieval"]}},
    )
    for record in (candidate_a, candidate_b, better_outside):
        store.append(record)

    packet_out, _ = KeywordCountRetrieval(top_k=2, source="retrieved").run(
        Packet(
            query=Query(text="graph"),
            retrieved=RetrievedSet(items=[candidate_a, candidate_b], scores=[]),
        ),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert all(record.record_id != "rec-3" for record in packet_out.retrieved.items)


def test_bm25_retrieval_prefers_stronger_lexical_matches() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    for text in ("graph memory retrieval", "graph retrieval", "tea notes"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="graph memory")), store)

    assert [record.text for record in packet_out.retrieved.items] == ["graph memory retrieval", "graph retrieval"]
    assert packet_out.retrieved.scores[0]["strategy"] == "bm25"
    assert packet_out.retrieved.scores[0]["score"] >= packet_out.retrieved.scores[1]["score"]


def test_bm25_retrieval_breaks_ties_by_recency() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    for text in ("graph memory", "graph memory"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="graph memory")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]


def test_bm25_retrieval_uses_representation_keywords() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="u1",
            layer="default",
            text="notes about tea",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"keywords": ["graph", "memory", "graph"]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="u2",
            layer="default",
            text="plain tea notes",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = BM25Retrieval(top_k=1).run(Packet(query=Query(text="graph memory")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]


def test_bm25_retrieval_on_empty_store_returns_empty_retrieved_set() -> None:
    from memprimitive.baselines import BM25Retrieval

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="Alice")), MemoryStore())

    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []


def test_bm25_retrieval_source_retrieved_returns_empty_on_empty_candidates() -> None:
    from memprimitive.baselines import BM25Retrieval

    packet_out, _ = BM25Retrieval(top_k=2, source="retrieved").run(
        Packet(query=Query(text="Alice"), retrieved=RetrievedSet(items=[], scores=[])),
        MemoryStore(),
    )

    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.trace["source"] == "retrieved"
    assert packet_out.retrieved.trace["candidate_count"] == 0


def test_bm25_retrieval_requires_query() -> None:
    from memprimitive.baselines import BM25Retrieval

    with pytest.raises(ValueError, match="packet.query"):
        BM25Retrieval(top_k=2).run(Packet(), MemoryStore())


def test_bm25_retrieval_falls_back_to_recency_when_all_scores_are_zero() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    for text in ("old note", "new note"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = BM25Retrieval(top_k=2).run(Packet(query=Query(text="graph memory")), store)

    assert [record.text for record in packet_out.retrieved.items] == ["new note", "old note"]
    assert packet_out.retrieved.trace["used_recency_fallback"] is True
    assert all(score["score"] == 0.0 for score in packet_out.retrieved.scores)


def test_bm25_retrieval_source_retrieved_only_scores_candidate_subset() -> None:
    from memprimitive.baselines import BM25Retrieval

    store = MemoryStore()
    candidate_a = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="default",
        text="tea notes only",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"representation": {"keywords": ["tea"]}},
    )
    candidate_b = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="default",
        text="graph retrieval",
        timestamp="2026-01-01T00:00:01+00:00",
        metadata={"representation": {"keywords": ["graph", "retrieval"]}},
    )
    better_outside = MemoryRecord(
        record_id="rec-3",
        unit_id="unit-3",
        layer="default",
        text="graph memory retrieval",
        timestamp="2026-01-01T00:00:02+00:00",
        metadata={"representation": {"keywords": ["graph", "memory", "retrieval"]}},
    )
    for record in (candidate_a, candidate_b, better_outside):
        store.append(record)

    packet_out, _ = BM25Retrieval(top_k=2, source="retrieved").run(
        Packet(
            query=Query(text="graph memory"),
            retrieved=RetrievedSet(items=[candidate_a, candidate_b], scores=[]),
        ),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2"]
    assert all(record.record_id != "rec-3" for record in packet_out.retrieved.items)
    assert packet_out.retrieved.trace["candidate_count"] == 2


def test_tag_retrieval_prefers_matching_tags() -> None:
    from memprimitive.baselines import AlwaysTrigger, AppendOrganization, LLMRepresentation, PassThroughUnitFormation, TagRetrieval

    class SeededTagRepresentation(LLMRepresentation):
        _TAGS_BY_TEXT = {
            "Alice likes tea": ["preference", "tea"],
            "Alice studies graph memory": ["graph", "memory"],
            "Bob likes coffee": ["preference", "coffee"],
        }

        def _llm_json(self, *, user: str) -> Any:
            payload = json.loads(user)
            return list(self._TAGS_BY_TEXT[payload["unit"]["text"]])

    store = MemoryStore()
    for text in ("Alice likes tea", "Alice studies graph memory", "Bob likes coffee"):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = SeededTagRepresentation(field="tags", prompt="Extract tags.").run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
        _, store = AppendOrganization().run(packet, store)

    packet_out, _ = TagRetrieval(top_k=1).run(Packet(query=Query(text="graph")), store)

    assert packet_out.retrieved.items[0].text == "Alice studies graph memory"


def test_entity_retrieval_prefers_entity_overlap() -> None:
    from memprimitive.baselines import AlwaysTrigger, AppendOrganization, EntityRetrieval, LLMRepresentation, PassThroughUnitFormation

    class SeededEntityRepresentation(LLMRepresentation):
        _ENTITIES_BY_TEXT = {
            "Alice likes tea": ["Alice"],
            "Bob likes coffee": ["Bob"],
            "Alice studies graph memory": ["Alice"],
        }

        def _llm_json(self, *, user: str) -> Any:
            payload = json.loads(user)
            return list(self._ENTITIES_BY_TEXT[payload["unit"]["text"]])

    store = MemoryStore()
    for text in ("Alice likes tea", "Bob likes coffee", "Alice studies graph memory"):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = SeededEntityRepresentation(field="entities", prompt="Extract entities.").run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
        _, store = AppendOrganization().run(packet, store)

    packet_out, _ = EntityRetrieval(top_k=2).run(Packet(query=Query(text="Alice")), store)

    assert all("Alice" in record.text for record in packet_out.retrieved.items)


def test_layer_aware_retrieval_supports_per_layer_top_k_and_merge_weights() -> None:
    from memprimitive.baselines import KeywordCountRetrieval, LayerAwareRetrieval, RecencyRetrieval

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="working"), StoreLayerSpec(name="semantic")])
    )
    store.append(MemoryRecord(record_id="rec-1", unit_id="u1", layer="working", text="recent working", timestamp="2026-01-01T00:00:00+00:00"))
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="u2",
            layer="semantic",
            text="Alice semantic graph",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"representation": {"keywords": ["alice", "semantic", "graph"]}},
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=2),
        retriever_by_layer={"semantic": KeywordCountRetrieval(top_k=2)},
        top_k=2,
        top_k_by_layer={"working": 1, "semantic": 1},
        merge_weight_by_layer={"semantic": 2.0},
    ).run(Packet(query=Query(text="Alice graph")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]


def test_bullet_list_readout_formats_bullets() -> None:
    from memprimitive.baselines import BulletListReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)
    retrieved = RetrievedSet(items=store.iter_records(), scores=[])

    packet_out, _ = BulletListReadout().run(Packet(retrieved=retrieved), store)

    assert packet_out.readout.text.startswith("- Alice likes tea.")


def test_buffer_retrieval_returns_latest_window_in_chronological_order() -> None:
    from memprimitive.baselines import BufferRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")]))
    for index in range(1, 5):
        store.append(
            MemoryRecord(
                record_id=f"rec-{index}",
                unit_id=f"unit-{index}",
                layer="reflections",
                text=f"Reflection {index}",
                timestamp=f"2026-01-01T00:00:0{index}+00:00",
            )
        )

    packet_out, _ = BufferRetrieval(top_k=2, layer="reflections").run(
        Packet(query=Query(text="Current question")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-3", "rec-4"]
    assert packet_out.retrieved.trace["candidate_count"] == 4


def test_grouped_by_layer_readout_groups_items() -> None:
    from memprimitive.baselines import GroupedByLayerReadout

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="working"), StoreLayerSpec(name="semantic")])
    )
    store.append(MemoryRecord(record_id="rec-1", unit_id="u1", layer="working", text="working", timestamp="2026-01-01T00:00:00+00:00"))
    store.append(MemoryRecord(record_id="rec-2", unit_id="u2", layer="semantic", text="semantic", timestamp="2026-01-01T00:00:01+00:00"))

    packet_out, _ = GroupedByLayerReadout().run(Packet(retrieved=RetrievedSet(items=store.iter_records(), scores=[])), store)

    assert "[working]" in packet_out.readout.text
    assert packet_out.readout.metadata["group_counts"] == {"working": 1, "semantic": 1}


def test_prompt_context_readout_switches_between_strategies() -> None:
    from memprimitive.baselines import PromptContextReadout

    reflection_record = MemoryRecord(
        record_id="rec-reflection",
        unit_id="unit-reflection",
        layer="reflections",
        text="Reflection: handle the empty-input edge case first.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    retrieved = RetrievedSet(items=[reflection_record], scores=[])

    reflexion_packet, _ = PromptContextReadout(memory_layer="reflections", default_strategy="reflexion").run(
        Packet(
            query=Query(
                text="Parse the input stream",
                metadata={"reflexion": {"last_attempt": "Attempt missed the edge case."}},
            ),
            retrieved=retrieved,
        ),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")])),
    )
    assert "Reflection 1:" in reflexion_packet.readout.text
    assert reflexion_packet.readout.source_ids == ["rec-reflection"]

    last_attempt_packet, _ = PromptContextReadout(memory_layer="reflections", default_strategy="reflexion").run(
        Packet(
            query=Query(
                text="Parse the input stream",
                metadata={"reflexion": {"strategy": "last_trial", "last_attempt": "Attempt missed the edge case."}},
            ),
            retrieved=retrieved,
        ),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="reflections")])),
    )
    assert "Below is the last trial you attempted" in last_attempt_packet.readout.text
    assert "Reflection 1:" not in last_attempt_packet.readout.text
    assert last_attempt_packet.readout.source_ids == []


def test_json_readout_returns_json_string() -> None:
    from memprimitive.baselines import JSONReadout

    store = MemoryStore()
    packet, store = _stored_pipeline_packet("Alice likes tea.", store)

    packet_out, _ = JSONReadout().run(Packet(retrieved=RetrievedSet(items=store.iter_records(), scores=[])), store)

    payload = json.loads(packet_out.readout.text)
    assert payload["items"][0]["text"] == "Alice likes tea."


def test_template_readout_simple_template_binds_context_filters_and_missing_variables() -> None:
    from memprimitive.baselines import TemplateReadout
    from memprimitive.utils._template import text_prompt

    episodic = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="episodic",
        text="Alice visited the tea house.",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={
            "session_id": "sess-1",
            "note": {
                "content": "Alice visited the tea house.",
                "context": "Trip note",
                "tags": ["travel", "tea"],
            },
            "representation": {
                "keywords": ["alice", "tea"],
                "entities": ["Alice", "Tea House"],
            },
            "graph": {
                "entities": ["Alice", "Tea House"],
                "links": ["rec-0"],
            },
        },
    )
    summary = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="session_summary",
        text="Alice had a productive tea-focused session.",
        timestamp="2026-01-02T00:00:00+00:00",
        metadata={
            "unit_type": "summary",
            "session_id": "sess-1",
            "hierarchical": {
                "group_key": {"session_id": "sess-1"},
                "source_record_ids": ["rec-1"],
                "source_unit_ids": ["unit-1"],
                "field_payload": {"summary": "Alice had a productive tea-focused session."},
            },
            "representation": {"summary": "Alice had a productive tea-focused session."},
        },
    )
    packet = Packet(
        query=Query(text="Alice tea", metadata={"foo": "bar", "session_id": "sess-1"}),
        retrieved=RetrievedSet(
            items=[episodic, summary],
            scores=[
                {"record_id": "rec-1", "rank": 2, "score": 0.4, "strategy": "keyword_count"},
                {"record_id": "rec-2", "rank": 1, "score": 0.9, "strategy": "embedding_similarity"},
            ],
            trace={
                "module": "layer_aware_retrieval",
                "retrieval_mode": "layer_aware",
                "candidate_count": 2,
                "active_layers": ["session_summary", "episodic"],
            },
        ),
        trace={"request_id": "req-1"},
    )

    packet_out, _ = TemplateReadout(
        prompt=text_prompt(
            "Q={{ query.text }}\n"
            "Now={{ runtime.now }}\n"
            "Foo={{ query.metadata.foo | default('x') }}\n"
            "Missing={{ runtime.user_name | default('anonymous') }}\n"
            "Top={{ retrieved.items | sort_by('timestamp', reverse=True) | topk(1) | join_text }}\n"
            "Episode={{ retrieved.by_layer.episodic | join_text }}\n"
            "Note={{ retrieved.by_record_id.rec-1.note.context }}\n"
            "Entities={{ retrieved.by_record_id.rec-1.graph.entities | join(', ') }}\n"
            "SummaryGroup={{ retrieved.by_record_id.rec-2.hierarchical.group_key.session_id }}\n"
            "Score={{ scores.by_record_id.rec-2.strategy }}\n"
            "Layers={{ trace.retrieval.active_layers | join(', ') }}"
        ),
        runtime_now_factory=lambda: "2026-04-03T12:00:00+00:00",
    ).run(packet, MemoryStore())

    assert packet_out.readout is not None
    assert "Q=Alice tea" in packet_out.readout.text
    assert "Now=2026-04-03T12:00:00+00:00" in packet_out.readout.text
    assert "Missing=anonymous" in packet_out.readout.text
    assert "Top=Alice had a productive tea-focused session." in packet_out.readout.text
    assert "Episode=Alice visited the tea house." in packet_out.readout.text
    assert "Note=Trip note" in packet_out.readout.text
    assert "Entities=Alice, Tea House" in packet_out.readout.text
    assert "SummaryGroup=sess-1" in packet_out.readout.text
    assert "Score=embedding_similarity" in packet_out.readout.text
    assert "Layers=session_summary, episodic" in packet_out.readout.text
    assert packet_out.readout.source_ids == ["rec-2", "rec-1"]
    assert "runtime.user_name | default('anonymous')" in packet_out.readout.metadata["missing_variables"]
    assert packet_out.readout.metadata["used_record_ids"] == ["rec-2", "rec-1"]
    assert "summary_with_sources" in packet_out.readout.metadata["available_views"]


def test_template_readout_structured_template_tracks_blocks_groups_and_relations() -> None:
    from memprimitive.baselines import TemplateReadout
    from memprimitive.utils._template import structured_prompt

    episodic = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="episodic",
        text="Alice debugged retrieval with graph traces.",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"session_id": "sess-1", "subgoal_id": "sg-1"},
    )
    summary = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="session_summary",
        text="Session summary for retrieval debugging.",
        timestamp="2026-01-02T00:00:00+00:00",
        metadata={
            "unit_type": "summary",
            "session_id": "sess-1",
            "hierarchical": {
                "group_key": {"session_id": "sess-1"},
                "source_record_ids": ["rec-1"],
                "source_unit_ids": ["unit-1"],
                "field_payload": {"summary": "Session summary for retrieval debugging."},
            },
            "representation": {"summary": "Session summary for retrieval debugging."},
        },
    )
    packet = Packet(
        query=Query(text="retrieval summary"),
        retrieved=RetrievedSet(
            items=[summary, episodic],
            scores=[
                {"record_id": "rec-2", "rank": 1, "strategy": "embedding_similarity"},
                {"record_id": "rec-1", "rank": 2, "strategy": "embedding_similarity"},
            ],
            trace={"retrieval_mode": "layer_aware", "candidate_count": 2},
        ),
    )

    packet_out, _ = TemplateReadout(
        prompt=structured_prompt({
            "blocks": [
                {"id": "query", "title": "Query", "template": "{{ query.text }}"},
                {
                    "id": "episodes",
                    "title": "Episodes",
                    "condition": "retrieved.by_layer.episodic | length",
                    "repeat_over": "retrieved.by_layer.episodic",
                    "item_template": "- {{ item.text }}",
                    "separator": "\n",
                },
                {
                    "id": "summaries",
                    "title": "Summaries",
                    "repeat_over": "retrieved.views.summary_with_sources",
                    "item_template": "* {{ item.summary.text }} ({{ item.group_key.session_id }}) => {{ item.sources | join_text }}",
                    "separator": "\n",
                },
                {
                    "id": "skip",
                    "title": "Skip",
                    "condition": "retrieved.by_layer.profile | length",
                    "template": "should not appear",
                },
            ]
        })
    ).run(packet, MemoryStore())

    assert packet_out.readout is not None
    assert "Query\nretrieval summary" in packet_out.readout.text
    assert "Episodes\n- Alice debugged retrieval with graph traces." in packet_out.readout.text
    assert "Summaries\n* Session summary for retrieval debugging. (sess-1) => Alice debugged retrieval with graph traces." in packet_out.readout.text
    assert "should not appear" not in packet_out.readout.text
    assert packet_out.readout.metadata["used_group_ids"] == ['group:hierarchical:{"session_id": "sess-1"}']
    assert packet_out.readout.metadata["used_record_ids"] == ["rec-1", "rec-2"]
    assert packet_out.readout.metadata["structuring_summary"]["relation_count"] >= 2
    assert any(entry["block_id"] == "skip" and entry["rendered"] is False for entry in packet_out.readout.metadata["block_trace"])


def test_template_readout_missing_value_can_render_placeholder() -> None:
    from memprimitive.baselines import TemplateReadout
    from memprimitive.utils._template import text_prompt

    packet_out, _ = TemplateReadout(
        prompt=text_prompt("User={{ runtime.user_id }}"),
        missing_value="<missing>",
    ).run(Packet(query=Query(text="hello"), retrieved=RetrievedSet()), MemoryStore())

    assert packet_out.readout is not None
    assert packet_out.readout.text == "User=<missing>"
    assert packet_out.readout.metadata["missing_variables"] == ["runtime.user_id"]


def test_template_readout_works_in_memory_pipeline_recall_flow() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import RecencyRetrieval, TemplateReadout
    from memprimitive.utils._template import text_prompt

    pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=2),
        readout=TemplateReadout(prompt=text_prompt("{{ retrieved.items | join_text }}")),
    )
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob likes coffee.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert "Bob likes coffee." in readout.text
    assert "Alice likes tea." in readout.text
    assert len(readout.source_ids) == 2


def test_render_prompt_plan_text_prompt_can_read_stored_representation_fields() -> None:
    from memprimitive.baselines import RecencyRetrieval
    from memprimitive.utils._template import render_prompt_plan, text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile", indices=("temporal",))])
    )
    store.append(
        MemoryRecord(
            record_id="rec-profile-1",
            unit_id="unit-profile-1",
            layer="profile",
            text="Alice prefers concise technical explanations.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={
                "representation": {
                    "user_profile": "Concise, concrete, technical.",
                    "response_hint": "Use short examples.",
                }
            },
        )
    )

    packet, _ = RecencyRetrieval(top_k=1, layer="profile").run(Packet(query=Query(text="Alice profile")), store)
    rendered, metadata, _ = render_prompt_plan(
        text_prompt(
            "profile={{ retrieved.items.0.representation.user_profile }}; "
            "hint={{ retrieved.items.0.representation.response_hint }}"
        ),
        packet=packet,
        store=store,
        runtime_now_factory=lambda: "2026-04-04T00:00:00+00:00",
    )

    assert rendered == "profile=Concise, concrete, technical.; hint=Use short examples."
    assert metadata["template_mode"] == "simple"
    assert metadata["missing_variables"] == []
    assert metadata["used_record_ids"] == ["rec-profile-1"]


def test_render_prompt_plan_structured_prompt_can_read_stored_representation_fields() -> None:
    from memprimitive.baselines import RecencyRetrieval
    from memprimitive.utils._template import render_prompt_plan, structured_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile", indices=("temporal",))])
    )
    store.append(
        MemoryRecord(
            record_id="rec-profile-1",
            unit_id="unit-profile-1",
            layer="profile",
            text="Alice prefers concise technical explanations.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={
                "representation": {
                    "user_profile": "Concise, concrete, technical.",
                    "response_hint": "Use short examples.",
                }
            },
        )
    )

    packet, _ = RecencyRetrieval(top_k=1, layer="profile").run(Packet(query=Query(text="Alice profile")), store)
    rendered, metadata, _ = render_prompt_plan(
        structured_prompt(
            {
                "blocks": [
                    {"id": "profile", "title": "Profile", "template": "{{ retrieved.items.0.representation.user_profile }}"},
                    {"id": "hint", "title": "Hint", "template": "{{ retrieved.items.0.representation.response_hint }}"},
                ]
            }
        ),
        packet=packet,
        store=store,
        runtime_now_factory=lambda: "2026-04-04T00:00:00+00:00",
    )

    assert "Profile\nConcise, concrete, technical." in rendered
    assert "Hint\nUse short examples." in rendered
    assert metadata["template_mode"] == "structured"
    assert metadata["missing_variables"] == []
    assert metadata["used_record_ids"] == ["rec-profile-1"]


def test_template_readout_text_prompt_can_fill_recalled_prompt_via_lightweight_retrieve_pipeline() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import RecencyRetrieval, TemplateReadout
    from memprimitive.utils._template import text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile", indices=("temporal",))])
    )
    store.append(
        MemoryRecord(
            record_id="rec-profile-1",
            unit_id="unit-profile-1",
            layer="profile",
            text="Alice profile memory",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"user_profile": "Concise, concrete, technical."}},
        )
    )

    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=TemplateReadout(
            prompt=text_prompt("{{ retrieved.items.0.representation.user_profile }}")
        ),
        store=store,
    )

    packet_out, _ = TemplateReadout(
        prompt=text_prompt(
            "Injected={{ recalled_prompt }}",
            recall_plan=text_prompt("{{ retrieved.items.0.representation.user_profile }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: "recall Alice profile",
            sub_recall_pipeline=retrieve_pipeline,
        )
    ).run(Packet(query=Query(text="How should we reply to Alice?"), retrieved=RetrievedSet()), store)

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Injected=Concise, concrete, technical."
    assert packet_out.readout.metadata["recalled_prompt"] == "Concise, concrete, technical."
    assert packet_out.readout.metadata["recall_prompt"]["matched"] is True
    assert packet_out.readout.metadata["recall_prompt"]["readout_source_ids"] == ["rec-profile-1"]


def test_template_readout_structured_prompt_can_fill_recalled_prompt_via_lightweight_retrieve_pipeline() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import RecencyRetrieval, TemplateReadout
    from memprimitive.utils._template import structured_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile", indices=("temporal",))])
    )
    store.append(
        MemoryRecord(
            record_id="rec-profile-1",
            unit_id="unit-profile-1",
            layer="profile",
            text="Alice profile memory",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"user_profile": "Concise, concrete, technical."}},
        )
    )

    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=TemplateReadout(
            prompt=structured_prompt(
                {
                    "blocks": [
                        {"id": "profile", "title": "Profile", "template": "{{ retrieved.items.0.representation.user_profile }}"},
                    ]
                }
            )
        ),
        store=store,
    )

    packet_out, _ = TemplateReadout(
        prompt=structured_prompt(
            {
                "blocks": [
                    {"id": "recalled", "title": "Recalled", "template": "{{ recalled_prompt }}"},
                ]
            },
            recall_plan=structured_prompt(
                {
                    "blocks": [
                        {"id": "profile", "title": "Profile", "template": "{{ retrieved.items.0.representation.user_profile }}"},
                    ]
                },
                metadata_mode="readout",
            ),
            recall_query_builder=lambda packet, current_store, context: "recall Alice profile",
            sub_recall_pipeline=retrieve_pipeline,
        )
    ).run(Packet(query=Query(text="How should we reply to Alice?"), retrieved=RetrievedSet()), store)

    assert packet_out.readout is not None
    assert "Recalled\nProfile\nConcise, concrete, technical." in packet_out.readout.text
    assert packet_out.readout.metadata["recalled_prompt"] == "Profile\nConcise, concrete, technical."
    assert packet_out.readout.metadata["recall_prompt"]["matched"] is True
    assert packet_out.readout.metadata["recall_prompt"]["readout_source_ids"] == ["rec-profile-1"]


def test_llm_representation_rejects_removed_recall_kwargs() -> None:
    from memprimitive.baselines import LLMRepresentation

    with pytest.raises(TypeError):
        LLMRepresentation(
            field="summary",
            prompt="Extract summary.",
            retrieve_pipeline=object(),
        )


def test_vector_graph_seed_and_expand_retrieval_rejects_removed_system_prompt_kwarg() -> None:
    from memprimitive.baselines import VectorGraphSeedAndExpandRetrieval

    with pytest.raises(TypeError):
        VectorGraphSeedAndExpandRetrieval(
            query_expand_with_llm=True,
            system_prompt="legacy prompt",
        )


def test_hierarchical_evolution_rejects_removed_recall_query_template_kwarg() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    with pytest.raises(TypeError):
        HierarchicalEvolution(
            source_layer="default",
            extract_mode="generate",
            extract_fields=("summary",),
            target_layer="semantic",
            recall_query_template="legacy",
        )


def test_template_readout_rejects_removed_simple_and_structured_template_kwargs() -> None:
    from memprimitive.baselines import TemplateReadout

    with pytest.raises(TypeError):
        TemplateReadout(simple_template="{{ retrieved.items | join_text }}")

    with pytest.raises(TypeError):
        TemplateReadout(structured_template={"blocks": []})
