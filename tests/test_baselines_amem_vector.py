from __future__ import annotations

from dataclasses import replace
import pytest

from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    MemoryUnit,
    Packet,
    Placement,
    Query,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)

from baselines_test_helpers import (
    _FakeAMEMRuntime,
    _WrapperShapeAMEMRuntime,
    _graph_vector_store,
    _seed_layer,
)


def test_semantic_field_enrichment_and_retrieval_embedding_repair_note_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.utils import _runtime
    from memprimitive.baselines import ConfigurableEmbeddingRepresentation, SemanticFieldEnrichmentRepresentation

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
    packet, _ = ConfigurableEmbeddingRepresentation(
        embedding_text=(
            "{{ unit.metadata.amem.content }} | "
            "context: {{ unit.metadata.amem.context }} | "
            "keywords: {{ unit.metadata.amem.keywords | join(', ') }} | "
            "tags: {{ unit.metadata.amem.tags | join(', ') }}"
        )
    ).run(packet, store)

    unit = packet.units[0]
    assert unit.metadata["amem"]["note_text"].startswith("Comprehensive note:")
    assert unit.text == "Alice likes tea."
    assert unit.metadata["representation"]["embedding_input_text"] == (
        "Alice likes tea. | context: Alice's tea habit supports her daily routine. | "
        "keywords: alice, tea, routine | tags: preference, habit, beverage"
    )
    assert unit.embedding == _runtime._DEFAULT_RUNTIME.embed(unit.metadata["representation"]["embedding_input_text"])


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


def test_vector_graph_seed_and_expand_retrieval_delegates_to_generic_seed_and_expand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import (
        EmbeddingSimilarityRetrieval,
        ExpandRetrievedGraphNeighbors,
        VectorGraphSeedAndExpandRetrieval,
    )

    store = _graph_vector_store()
    seed = MemoryRecord(
        record_id="rec-seed",
        unit_id="unit-seed",
        layer="knowledge_graph",
        text="Alice likes tea.",
        timestamp="2026-03-27T00:00:00+00:00",
        embedding=[1.0, 0.0],
        metadata={
            "amem": {
                "content": "Alice likes tea.",
                "note_text": "Alice likes tea.",
                "context": "Preference note",
                "keywords": ["alice", "tea"],
                "tags": ["preference"],
                "category": "profile",
                "attributes": {},
            },
            "graph": {"links": ["rec-neighbor"]},
        },
    )
    neighbor = MemoryRecord(
        record_id="rec-neighbor",
        unit_id="unit-neighbor",
        layer="knowledge_graph",
        text="Tea routines improve focus.",
        timestamp="2026-03-27T00:00:01+00:00",
        embedding=[0.9, 0.1],
        metadata={
            "amem": {
                "content": "Tea routines improve focus.",
                "note_text": "Tea routines improve focus.",
                "context": "Focus note",
                "keywords": ["tea", "focus"],
                "tags": ["insight"],
                "category": "insight",
                "attributes": {},
            },
            "graph": {"links": []},
        },
    )

    calls: list[str] = []

    def _fake_seed_run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        calls.append(f"seed:{self.top_k}:{self.layer}:{self.source}")
        retrieved = RetrievedSet(
            items=[seed],
            scores=[{"record_id": seed.record_id, "rank": 1, "score": 0.9, "strategy": "embedding_similarity"}],
            trace={"module": self.spec.name, "top_k": self.top_k, "source": self.source},
        )
        return replace(packet, retrieved=retrieved), store

    def _fake_expand_run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        calls.append(
            f"expand:{self.top_k}:{self.layer}:{self.include_seed_records}:{self.per_seed_top_k}:{self.dedupe}"
        )
        retrieved = RetrievedSet(
            items=[seed, neighbor],
            scores=[
                {"record_id": seed.record_id, "rank": 1, "strategy": "graph_seed", "seed_record_id": seed.record_id, "hop": 0},
                {
                    "record_id": neighbor.record_id,
                    "rank": 2,
                    "strategy": "graph_expand_retrieved",
                    "seed_record_id": seed.record_id,
                    "hop": 1,
                },
            ],
            trace={
                "module": self.spec.name,
                "expanded_neighbor_ids": [neighbor.record_id],
                "top_k": self.top_k,
                "per_seed_top_k": self.per_seed_top_k,
            },
        )
        return replace(packet, retrieved=retrieved), store

    monkeypatch.setattr(EmbeddingSimilarityRetrieval, "run", _fake_seed_run)
    monkeypatch.setattr(ExpandRetrievedGraphNeighbors, "run", _fake_expand_run)

    packet_out, _ = VectorGraphSeedAndExpandRetrieval(
        top_k=2,
        layer="knowledge_graph",
        candidate_k=1,
        neighbor_expansion_k=1,
        note_namespace="amem",
    ).run(Packet(query=Query(text="Alice", embedding=[1.0, 0.0])), store)

    assert calls == [
        "seed:1:knowledge_graph:store",
        "expand:2:knowledge_graph:True:1:True",
    ]
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-seed", "rec-neighbor"]
    assert packet_out.retrieved.trace["seed_trace"]["module"] == "embedding_similarity_retrieval"
    assert packet_out.retrieved.trace["expand_trace"]["module"] == "expand_retrieved_graph_neighbors"


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

