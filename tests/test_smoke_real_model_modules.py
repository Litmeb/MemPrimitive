"""Optional smoke tests: exercise each LLM- or embedding-backed baseline once.

These tests are marked ``integration``. LLM paths require ``MEMPRIMITIVE_API_KEY``,
``MEMPRIMITIVE_BASE_URL``, and ``MEMPRIMITIVE_MODEL`` (see ``require_real_runtime``);
without them, affected tests are skipped. Embedding-only paths (local
``sentence-transformers`` or ``Runtime.embed``) run without those variables.

Coverage map (one pass per public module that calls into LLM or embedding):

- **representation**: ``BasicRepresentation``, ``TripleRepresentation``,
  ``LLMRepresentation``, ``SemanticFieldEnrichmentRepresentation``,
  ``ConfigurableEmbeddingRepresentation``
- **retrieval**: ``EmbeddingSimilarityRetrieval``, ``QueryRewriteRetrieval`` (llm),
  ``VectorGraphSeedAndExpandRetrieval`` (LLM query expand + ``Runtime.embed`` + graph expand)
- **organization**: ``GraphDeduplicationAppendOrganization``,
  ``GraphEntityDeduplicationAppendOrganization``, ``LLMFunctionCallOrganization``
- **trigger**: ``LLMJudgeTrigger``
- **memory_evolution**: ``LinkStrengtheningEvolution``,
  ``NeighborContextUpdateEvolution``, ``LLMFunctionCallEvolution``,
  ``HierarchicalEvolution`` (generate), ``HierarchicalOrganization`` (generate)
- **readout**: ``MidDecodingMemoryReadout``
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from memprimitive import MemoryPipeline
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
from memprimitive.utils._amem_family import DEFAULT_NOTE_NAMESPACE, repair_note_payload


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


def _note_record(
    *,
    record_id: str,
    unit_id: str,
    text: str,
    embedding: list[float],
) -> MemoryRecord:
    payload = repair_note_payload(
        {
            "content": text,
            "note_text": text,
            "context": f"context for {text}",
            "keywords": ["smoke"],
            "tags": ["smoke"],
            "category": "smoke",
            "attributes": {},
        },
        fallback_content=text,
        default_category="smoke",
    )
    enhanced = f"content: {payload['content']}"
    return MemoryRecord(
        record_id=record_id,
        unit_id=unit_id,
        layer="knowledge_graph",
        text=text,
        timestamp="2026-04-05T00:00:00+00:00",
        embedding=list(embedding),
        metadata={
            DEFAULT_NOTE_NAMESPACE: {**payload, "enhanced_embedding_text": enhanced},
            "graph": {"entities": [], "triples": [], "links": []},
        },
    )


@pytest.mark.integration
def test_smoke_basic_representation_embedding() -> None:
    from memprimitive.baselines import BasicRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Smoke test: Alice likes oolong tea.", source="notes")),
        MemoryStore(),
    )
    out, _ = BasicRepresentation(elements=("text", "embedding")).run(packet, store)
    assert out.units is not None
    assert out.units[0].embedding is not None
    assert len(out.units[0].embedding) > 0


@pytest.mark.integration
def test_smoke_triple_representation(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, TripleRepresentation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice works with Bob on graph memory in Seattle.", source="notes")),
        MemoryStore(),
    )
    out, _ = TripleRepresentation(method="direct").run(packet, store)
    assert out.units is not None
    assert out.units[0].triples
    assert all(len(t) == 3 for t in out.units[0].triples)


@pytest.mark.integration
def test_smoke_llm_representation(require_real_runtime: None) -> None:
    from memprimitive.baselines import LLMRepresentation, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice maintains a retrieval benchmark for graph memory.", source="notes")),
        MemoryStore(),
    )
    out, _ = LLMRepresentation(
        field="summary",
        prompt="Write one concise English sentence summarizing the unit.",
    ).run(packet, store)
    assert out.units is not None
    summary = out.units[0].metadata.get("representation", {}).get("summary", "")
    assert isinstance(summary, str) and summary.strip()


@pytest.mark.integration
def test_smoke_semantic_field_enrichment_representation(require_real_runtime: None) -> None:
    from memprimitive.baselines import PassThroughUnitFormation, SemanticFieldEnrichmentRepresentation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Bob stores session notes about tea preferences.", source="notes")),
        MemoryStore(),
    )
    out, _ = SemanticFieldEnrichmentRepresentation().run(packet, store)
    assert out.units is not None
    note = out.units[0].metadata.get(DEFAULT_NOTE_NAMESPACE)
    assert isinstance(note, dict) and str(note.get("content", "")).strip()


@pytest.mark.integration
def test_smoke_configurable_embedding_representation() -> None:
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, PassThroughUnitFormation
    from memprimitive.utils._amem_family import repair_note_payload
    from memprimitive.utils._template import text_prompt

    payload = repair_note_payload(
        {"content": "Carol indexes embeddings for retrieval.", "note_text": "Carol indexes embeddings for retrieval."},
        fallback_content="Carol indexes embeddings for retrieval.",
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Carol indexes embeddings for retrieval.", source="notes")),
        MemoryStore(),
    )
    unit = packet.units[0]
    unit = MemoryUnit(
        unit_id=unit.unit_id,
        text=unit.text,
        metadata={**unit.metadata, DEFAULT_NOTE_NAMESPACE: dict(payload)},
    )
    packet = _packet_with_unit(packet, unit)
    out, _ = ConfigurableEmbeddingRepresentation(
        embedding_text=text_prompt("{{ unit.metadata.note.content }}")
    ).run(packet, store)
    assert out.units is not None
    assert out.units[0].embedding is not None
    assert len(out.units[0].embedding) > 0


def _packet_with_unit(packet: Packet, unit: MemoryUnit) -> Packet:
    return replace(packet, units=[unit])


@pytest.mark.integration
def test_smoke_embedding_similarity_retrieval() -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval

    retriever = EmbeddingSimilarityRetrieval(top_k=1, layer="default")
    emb = retriever._embed_text("smoke query about alice")
    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-a",
            unit_id="u-a",
            layer="default",
            text="alice project",
            timestamp="2026-04-05T00:00:01+00:00",
            embedding=emb,
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-b",
            unit_id="u-b",
            layer="default",
            text="unrelated zebra",
            timestamp="2026-04-05T00:00:02+00:00",
            embedding=retriever._embed_text("zebra"),
        )
    )
    out, _ = retriever.run(Packet(query=Query(text="smoke query about alice")), store)
    assert out.retrieved is not None
    assert out.retrieved.items
    assert out.retrieved.items[0].record_id == "rec-a"


@pytest.mark.integration
def test_smoke_query_rewrite_retrieval_llm(require_real_runtime: None) -> None:
    from memprimitive.baselines import QueryRewriteRetrieval, RecencyRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="u-1",
            layer="default",
            text="alice graph memory",
            timestamp="2026-04-05T00:00:00+00:00",
        )
    )
    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="llm",
        prompt="Rewrite the query into a short English keyword phrase for retrieval.",
    )
    out, _ = module.run(Packet(query=Query(text="Who works on graph memory?")), store)
    assert out.query is not None
    assert out.retrieved is not None
    assert out.retrieved.items
    assert out.trace["retrieval"]["query_rewrite"]["rewrite_strategy"] == "llm"


@pytest.mark.integration
def test_smoke_vector_graph_seed_and_expand_retrieval(require_real_runtime: None) -> None:
    from memprimitive.baselines import VectorGraphSeedAndExpandRetrieval
    from memprimitive.utils._runtime import get_runtime

    rt = get_runtime()
    emb_a = list(rt.embed("alice likes tea for smoke test"))
    emb_b = list(rt.embed("bob studies graphs for smoke test"))
    store = _graph_vector_store()
    store.append(_note_record(record_id="rec-a", unit_id="u-a", text="Alice likes tea.", embedding=emb_a))
    store.append(_note_record(record_id="rec-b", unit_id="u-b", text="Bob studies graphs.", embedding=emb_b))
    store.add_graph_links("knowledge_graph", "rec-a", ["rec-b"])

    module = VectorGraphSeedAndExpandRetrieval(
        top_k=2,
        layer="knowledge_graph",
        candidate_k=2,
        neighbor_expansion_k=2,
        query_expand_with_llm=True,
        agentic_search=False,
    )
    out, _ = module.run(Packet(query=Query(text="alice tea smoke")), store)
    assert out.retrieved is not None
    trace = out.retrieved.trace
    assert trace.get("query_expand_with_llm") is True
    assert trace.get("retrieval_mode") == "embedding_similarity_plus_graph_expand"
    assert out.retrieved.items


@pytest.mark.integration
def test_smoke_graph_deduplication_append_organization_embedding() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphDeduplicationAppendOrganization, PassThroughUnitFormation

    store = _graph_store()
    from memprimitive.utils._runtime import get_runtime

    rt = get_runtime()
    emb_seed = list(rt.embed("Alice likes tea."))
    store.append(
        MemoryRecord(
            record_id="rec-seed",
            unit_id="u-seed",
            layer="knowledge_graph",
            text="Alice likes tea.",
            timestamp="2026-04-05T00:00:00+00:00",
            embedding=emb_seed,
            metadata={"graph": {"entities": ["Alice", "tea"], "triples": [("Alice", "likes", "tea")], "links": []}},
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    u = packet.units[0]
    u = MemoryUnit(
        unit_id=u.unit_id,
        text="Alice likes tea.",
        triples=[("Alice", "likes", "tea")],
        entities=["Alice", "tea"],
        metadata=u.metadata,
    )
    packet = _packet_with_unit(packet, u)
    packet, store = AlwaysTrigger().run(packet, store)
    _, store = GraphDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.99).run(packet, store)
    assert store.count("knowledge_graph") == 1


@pytest.mark.integration
def test_smoke_graph_entity_deduplication_append_organization_embedding() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphEntityDeduplicationAppendOrganization, PassThroughUnitFormation
    from memprimitive.utils._runtime import get_runtime

    rt = get_runtime()
    emb_alice = list(rt.embed("entity alice smoke"))
    emb_tea = list(rt.embed("entity tea smoke"))
    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-alice",
            unit_id="u-alice",
            layer="knowledge_graph",
            text="Alice",
            timestamp="2026-04-05T00:00:00+00:00",
            embedding=emb_alice,
            metadata={"graph": {"entities": ["Alice"], "triples": [], "links": []}},
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice enjoys tea.", source="notes")),
        store,
    )
    u = packet.units[0]
    u = MemoryUnit(
        unit_id=u.unit_id,
        text=u.text,
        triples=[("Alice", "enjoys", "tea")],
        entities=["Alice", "tea"],
        metadata={
            **u.metadata,
            "representation": {
                **u.metadata.get("representation", {}),
                "entity_embeddings": {"Alice": emb_alice, "tea": emb_tea},
            },
        },
    )
    packet = _packet_with_unit(packet, u)
    packet, store = AlwaysTrigger().run(packet, store)
    _, store = GraphEntityDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.99).run(packet, store)
    assert store.count("knowledge_graph") >= 1


@pytest.mark.integration
def test_smoke_llm_function_call_organization(require_real_runtime: None) -> None:
    from memprimitive.baselines import LLMFunctionCallOrganization, PassThroughUnitFormation

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Smoke: store one short fact about oolong.", source="notes")),
        MemoryStore(),
    )
    packet = replace(packet, decisions=[True])
    org = LLMFunctionCallOrganization(
        prompt=(
            "Call the ADD tool exactly once. Use a short one-sentence memory text derived from the unit. "
            "The ADD argument target_layer must be the literal string default (the default flat memory layer)."
        ),
        tools=["ADD"],
        target_layer="default",
        allow_no_tool_call=True,
    )
    out, store = org.run(packet, store)
    assert out.trace["organization"]["module"] == "llm_function_call_organization"
    assert store.count() >= 0


@pytest.mark.integration
def test_smoke_llm_judge_trigger(require_real_runtime: None) -> None:
    from memprimitive.baselines import LLMJudgeTrigger

    packet = Packet(
        units=[MemoryUnit(text="Judge smoke prompt.", unit_id="u1")],
        observation=Observation(text="obs", source="notes"),
    )
    trigger = LLMJudgeTrigger(
        slot="write_trigger",
        prompt='Return strict JSON: {"decision": true, "score": 0.9, "label": "smoke"}',
        decision_mode="score",
        threshold=0.5,
        per_unit=True,
    )
    out, _ = trigger.run(packet, MemoryStore())
    assert out.trace["write_trigger"]["module"] == "llm_judge_write_trigger"
    assert out.decisions == [True]


@pytest.mark.integration
def test_smoke_link_strengthening_evolution(require_real_runtime: None) -> None:
    from memprimitive.baselines import LinkStrengtheningEvolution
    from memprimitive.utils._runtime import get_runtime

    rt = get_runtime()
    emb_x = list(rt.embed("link smoke current note"))
    emb_y = list(rt.embed("link smoke neighbor note"))
    store = _graph_vector_store()
    store.append(_note_record(record_id="rec-cur", unit_id="u-smoke", text="current note", embedding=emb_x))
    store.append(_note_record(record_id="rec-nb", unit_id="u-other", text="neighbor note", embedding=emb_y))

    unit = MemoryUnit(text="current note", unit_id="u-smoke")
    packet = Packet(
        units=[unit],
        placements=[Placement(unit_id="u-smoke", target_layer="knowledge_graph")],
        decisions=[True],
    )
    out, _ = LinkStrengtheningEvolution(
        target_layer="knowledge_graph",
        candidate_k=2,
        max_links_per_record=2,
    ).run(packet, store)
    assert out.trace["memory_evolution"]["module"] == "link_strengthening_evolution"


@pytest.mark.integration
def test_smoke_neighbor_context_update_evolution(require_real_runtime: None) -> None:
    from memprimitive.baselines import NeighborContextUpdateEvolution
    from memprimitive.utils._runtime import get_runtime

    rt = get_runtime()
    emb_a = list(rt.embed("neighbor ctx smoke a"))
    emb_b = list(rt.embed("neighbor ctx smoke b"))
    store = _graph_vector_store()
    r_a = _note_record(record_id="rec-a", unit_id="u-a", text="note a", embedding=emb_a)
    r_b = _note_record(record_id="rec-b", unit_id="u-b", text="note b", embedding=emb_b)
    store.append(r_a)
    store.append(r_b)
    store.add_graph_links("knowledge_graph", "rec-a", ["rec-b"])

    unit = MemoryUnit(text="note a", unit_id="u-a")
    packet = Packet(
        units=[unit],
        placements=[Placement(unit_id="u-a", target_layer="knowledge_graph")],
        decisions=[True],
    )
    out, _ = NeighborContextUpdateEvolution(target_layer="knowledge_graph", candidate_k=2).run(packet, store)
    assert out.trace["memory_evolution"]["module"] == "neighbor_context_update_evolution"


@pytest.mark.integration
def test_smoke_llm_function_call_evolution(require_real_runtime: None) -> None:
    from memprimitive.baselines import LLMFunctionCallEvolution

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-ev",
            unit_id="u-ev",
            layer="default",
            text="evolution smoke target",
            timestamp="2026-04-05T00:00:00+00:00",
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="smoke", unit_id="u1")],
        decisions=[True],
        decisions_store={"default": {"decision": True, "record_ids": ["rec-ev"], "selector": {"kind": "manual"}}},
    )
    evo = LLMFunctionCallEvolution(
        prompt="If appropriate, call UPDATE on the selected record to append the word SMOKE to its text; otherwise respond NO_ACTION.",
        tools=["UPDATE"],
        source_layer="default",
        allow_no_tool_call=True,
    )
    out, _ = evo.run(packet, store)
    assert out.trace["memory_evolution"]["module"] == "llm_function_call_evolution"


@pytest.mark.integration
def test_smoke_hierarchical_evolution_generate(require_real_runtime: None) -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-h1",
            unit_id="u-h1",
            layer="default",
            text="first line about graphs",
            timestamp="2026-04-05T00:00:01+00:00",
            metadata={"session_id": "sess-smoke"},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-h2",
            unit_id="u-h2",
            layer="default",
            text="second line about retrieval",
            timestamp="2026-04-05T00:00:02+00:00",
            metadata={"session_id": "sess-smoke"},
        )
    )
    packet = Packet(
        units=[MemoryUnit(text="incoming", unit_id="u-in")],
        placements=[Placement(unit_id="u-in", target_layer="default")],
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-h1", "rec-h2"], "selector": {"kind": "manual"}}
        },
        trace={},
    )
    out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt="Return JSON with a single key summary whose value is one English sentence aggregating the records.",
        target_layer="semantic",
    ).run(packet, store)
    assert out.trace["memory_evolution"]["module"] == "hierarchical_evolution"
    written = store.iter_records("semantic")
    assert written
    assert str(written[0].text).strip()


@pytest.mark.integration
def test_smoke_hierarchical_organization_generate(require_real_runtime: None) -> None:
    from memprimitive.baselines import HierarchicalOrganization, PassThroughUnitFormation

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-o1",
            unit_id="u-o1",
            layer="default",
            text="org line one",
            timestamp="2026-04-05T00:00:01+00:00",
            metadata={"session_id": "sess-org"},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-o2",
            unit_id="u-o2",
            layer="default",
            text="org line two",
            timestamp="2026-04-05T00:00:02+00:00",
            metadata={"session_id": "sess-org"},
        )
    )
    packet, _ = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="hierarchical org smoke", source="notes")),
        store,
    )
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-o1", "rec-o2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )
    out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt="Return JSON with key summary: one sentence combining the selected records.",
        target_layer="semantic",
    ).run(packet, store)
    assert out.trace["organization"]["module"] == "hierarchical_organization"
    assert store.iter_records("semantic")


@pytest.mark.integration
def test_smoke_mid_decoding_memory_readout(require_real_runtime: None) -> None:
    from memprimitive.baselines import ConcatenateReadout, KeywordCountRetrieval, MidDecodingMemoryReadout
    from memprimitive.utils._template import text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile"), StoreLayerSpec(name="default")])
    )
    store.append(
        MemoryRecord(
            record_id="rec-md-1",
            unit_id="u-md-1",
            layer="profile",
            text="Alice prefers concise technical answers.",
            timestamp="2026-04-05T00:00:01+00:00",
            metadata={"representation": {"keywords": ["alice", "concise", "technical"]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-md-2",
            unit_id="u-md-2",
            layer="profile",
            text="Alice wants source provenance preserved when possible.",
            timestamp="2026-04-05T00:00:02+00:00",
            metadata={"representation": {"keywords": ["alice", "source", "provenance"]}},
        )
    )

    retrieve_pipeline = MemoryPipeline(
        retrieval=KeywordCountRetrieval(top_k=2, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    module = MidDecodingMemoryReadout(
        prompt=text_prompt(
            (
                "You must call MEM_READ exactly once before answering.\n"
                "Use the literal query string: alice concise provenance\n"
                "After the tool returns, answer in one short English sentence."
            )
        ),
        retrieve_pipeline=retrieve_pipeline,
        max_turns=4,
        allow_no_tool_call=False,
    )

    out, _ = module.run(
        Packet(
            query=Query(text="What should I remember about Alice?"),
            retrieved=RetrievedSet(),
        ),
        store,
    )

    assert out.readout is not None
    assert isinstance(out.readout.text, str) and out.readout.text.strip()
    assert out.readout.metadata["memory_read_count"] >= 1
    assert out.readout.source_ids
    assert "rec-md-1" in out.readout.source_ids or "rec-md-2" in out.readout.source_ids
    assert out.trace["readout"]["module"] == "mid_decoding_memory_readout"
    assert any(call["tool_name"] == "MEM_READ" and call["status"] == "applied" for call in out.trace["readout"]["tool_calls"])
