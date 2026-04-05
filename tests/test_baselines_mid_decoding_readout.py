from __future__ import annotations

import json
from typing import Any

import pytest

from memprimitive import MemoryPipeline
from memprimitive.core import MemoryRecord, MemoryStore, Observation, Packet, Query, RetrievedSet, StoreLayerSpec, StoreTopology
from memprimitive.utils._template import text_prompt

from baselines_test_helpers import _invoke_runtime_tool, _seed_layer


def test_mid_decoding_memory_readout_can_call_mem_read_and_return_final_answer() -> None:
    from memprimitive.baselines import ConcatenateReadout, KeywordCountRetrieval, MidDecodingMemoryReadout

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["Alice likes concise technical answers."])

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    module = MidDecodingMemoryReadout(
        prompt=text_prompt("Question={{ query.text }}"),
        retrieve_pipeline=retrieve_pipeline,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        assert rendered_prompt == "Question=What does Alice like?"
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "Alice concise answers"}))
        assert payload["memory_text"] == "Alice likes concise technical answers."
        return "Alice likes concise technical answers."

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, _ = module.run(
        Packet(query=Query(text="What does Alice like?"), retrieved=RetrievedSet()),
        store,
    )

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Alice likes concise technical answers."
    assert packet_out.readout.source_ids == ["rec-1"]
    assert packet_out.readout.metadata["memory_read_count"] == 1
    assert packet_out.readout.metadata["memory_read_record_ids"] == ["rec-1"]
    assert packet_out.trace["readout"]["tool_calls"][0]["tool_name"] == "MEM_READ"


def test_mid_decoding_memory_readout_mem_read_uses_current_store_not_child_pipeline_store() -> None:
    from memprimitive.baselines import ConcatenateReadout, KeywordCountRetrieval, MidDecodingMemoryReadout

    current_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    stale_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(current_store, "profile", ["CURRENT STORE PROFILE"])
    _seed_layer(stale_store, "profile", ["STALE PIPELINE PROFILE"])

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=stale_store,
    )
    module = MidDecodingMemoryReadout(
        prompt=text_prompt("Question={{ query.text }}"),
        retrieve_pipeline=retrieve_pipeline,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "CURRENT PROFILE"}))
        assert payload["memory_text"] == "CURRENT STORE PROFILE"
        return payload["memory_text"]

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, _ = module.run(
        Packet(query=Query(text="Use current store"), retrieved=RetrievedSet()),
        current_store,
    )

    assert packet_out.readout is not None
    assert packet_out.readout.text == "CURRENT STORE PROFILE"
    assert packet_out.readout.source_ids == ["rec-1"]


def test_mid_decoding_memory_readout_multiple_mem_reads_count_calls_and_dedupe_source_ids() -> None:
    from memprimitive.baselines import ConcatenateReadout, KeywordCountRetrieval, MidDecodingMemoryReadout

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Alice likes tea.",
            timestamp="2026-01-01T00:00:01Z",
            metadata={"representation": {"keywords": ["alice", "tea"]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="profile",
            text="Alice likes concise answers.",
            timestamp="2026-01-01T00:00:02Z",
            metadata={"representation": {"keywords": ["alice", "concise", "answers"]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="profile",
            text="Bob likes coffee.",
            timestamp="2026-01-01T00:00:03Z",
            metadata={"representation": {"keywords": ["bob", "coffee"]}},
        )
    )

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    module = MidDecodingMemoryReadout(
        prompt=text_prompt("Question={{ query.text }}"),
        retrieve_pipeline=retrieve_pipeline,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        tea_payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "tea tea alice"}))
        repeat_payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "tea tea alice"}))
        concise_payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "concise answers"}))
        assert tea_payload["source_ids"] == ["rec-1"]
        assert repeat_payload["source_ids"] == ["rec-1"]
        assert concise_payload["source_ids"] == ["rec-2"]
        return "done"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]
    packet_out, _ = module.run(
        Packet(query=Query(text="Answer with multiple reads"), retrieved=RetrievedSet()),
        store,
    )

    assert packet_out.readout is not None
    assert packet_out.readout.text == "done"
    assert packet_out.readout.source_ids == ["rec-1", "rec-2"]
    assert packet_out.readout.metadata["memory_read_count"] == 3
    assert packet_out.readout.metadata["memory_read_record_ids"] == ["rec-1", "rec-2"]


def test_mid_decoding_memory_readout_strict_schema_validation_raises() -> None:
    from memprimitive.baselines import ConcatenateReadout, KeywordCountRetrieval, MidDecodingMemoryReadout

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["Alice likes tea."])

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    module = MidDecodingMemoryReadout(
        prompt=text_prompt("Question={{ query.text }}"),
        retrieve_pipeline=retrieve_pipeline,
        strict_tools=True,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        _invoke_runtime_tool(tools[0], {"query": 123})
        return "should not reach"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="must have JSON type string"):
        module.run(Packet(query=Query(text="bad schema"), retrieved=RetrievedSet()), store)


def test_mid_decoding_memory_readout_rejects_missing_tool_calls_when_required() -> None:
    from memprimitive.baselines import ConcatenateReadout, KeywordCountRetrieval, MidDecodingMemoryReadout

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["Alice likes tea."])

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    module = MidDecodingMemoryReadout(
        prompt=text_prompt("Question={{ query.text }}"),
        retrieve_pipeline=retrieve_pipeline,
        allow_no_tool_call=False,
    )

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        return "NO_ACTION"

    module._run_agent = _fake_run_agent.__get__(module, type(module))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="at least one successful or attempted tool call"):
        module.run(Packet(query=Query(text="no tool"), retrieved=RetrievedSet()), store)


def test_mid_decoding_memory_readout_works_in_memory_pipeline_recall_flow() -> None:
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

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    readout_module = MidDecodingMemoryReadout(
        prompt=text_prompt("Question={{ query.text }}"),
        retrieve_pipeline=retrieve_pipeline,
    )
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="profile"),
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=readout_module,
        store=MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")])),
    )
    pipeline.ingest(Observation(text="Alice prefers concise answers.", source="dialogue"))

    def _fake_run_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        payload = json.loads(_invoke_runtime_tool(tools[0], {"query": "Alice concise"}))
        return payload["memory_text"]

    readout_module._run_agent = _fake_run_agent.__get__(readout_module, type(readout_module))  # type: ignore[method-assign]
    readout = pipeline.recall(Query(text="How should we answer Alice?"))

    assert readout.text == "Alice prefers concise answers."
    assert readout.source_ids == ["rec-1"]
