from __future__ import annotations

import pytest

from memprimitive.baselines.registry import (
    registered_baseline_class_names,
)
from memprimitive.core import (
    MemoryRecord,
    MemoryStore,
    Observation,
    Packet,
    Query,
    RetrievedSet,
    StoreLayerSpec,
    StoreTopology,
)

from baselines_test_helpers import (
    _stored_pipeline_packet,
)


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


def test_metadata_retrieval_rejects_invalid_inputs() -> None:
    from memprimitive.baselines import MetadataRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        MetadataRetrieval(top_k=0, field="topic", target="graphs")
    with pytest.raises(ValueError, match="non-empty field"):
        MetadataRetrieval(field="   ", target="graphs")
    with pytest.raises(ValueError, match="match_mode"):
        MetadataRetrieval(field="topic", target="graphs", match_mode="contains")


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


def test_multi_query_recency_retrieval_dedupes_repeated_hits_in_query_order() -> None:
    from memprimitive.baselines import RecencyRetrieval

    store = MemoryStore()
    for text in ("first item", "second item", "third item"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = RecencyRetrieval(top_k=2).run(
        Packet(
            queries=[
                Query(text="first query"),
                Query(text="second query"),
            ]
        ),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["third item", "second item"]
    assert len(packet_out.retrieved.scores) == 2
    assert packet_out.retrieved.trace["query_count"] == 2
    assert packet_out.retrieved.trace["merge_strategy"] == "query_order_dedupe"
    assert [entry["returned_count"] for entry in packet_out.retrieved.trace["per_query"]] == [2, 2]


def test_multi_query_keyword_retrieval_flattens_non_overlapping_hits_in_query_order() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    for text in ("alice memory", "bob memory", "carol memory"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = KeywordCountRetrieval(top_k=1).run(
        Packet(
            queries=[
                Query(text="alice"),
                Query(text="bob"),
            ]
        ),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.text for record in packet_out.retrieved.items] == ["alice memory", "bob memory"]
    assert [score["record_id"] for score in packet_out.retrieved.scores] == [
        packet_out.retrieved.items[0].record_id,
        packet_out.retrieved.items[1].record_id,
    ]


def test_multi_query_retrieval_accepts_queries_without_primary_query() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    for text in ("alice memory", "bob memory"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = KeywordCountRetrieval(top_k=1).run(
        Packet(
            queries=[
                Query(text="bob"),
            ]
        ),
        store,
    )

    assert packet_out.query is None
    assert packet_out.queries is not None
    assert [query.text for query in packet_out.queries] == ["bob"]
    assert [record.text for record in packet_out.retrieved.items] == ["bob memory"]


def test_multi_query_retrieval_prefers_queries_field_over_primary_query() -> None:
    from memprimitive.baselines import KeywordCountRetrieval

    store = MemoryStore()
    for text in ("alice memory", "carol memory"):
        _, store = _stored_pipeline_packet(text, store)

    packet_out, _ = KeywordCountRetrieval(top_k=1).run(
        Packet(
            query=Query(text="carol"),
            queries=[Query(text="alice")],
        ),
        store,
    )

    assert packet_out.query is not None
    assert packet_out.query.text == "carol"
    assert [record.text for record in packet_out.retrieved.items] == ["alice memory"]
    assert packet_out.retrieved.trace["query_ids"] == [packet_out.queries[0].query_id]


def test_query_rewrite_retrieval_llm_single_query_rewrites_before_delegate() -> None:
    from memprimitive.baselines import KeywordCountRetrieval, QueryRewriteRetrieval

    store = MemoryStore()
    for text in ("alice memory", "bob memory"):
        _, store = _stored_pipeline_packet(text, store)

    module = QueryRewriteRetrieval(
        retriever=KeywordCountRetrieval(top_k=1),
        strategy="llm",
        prompt="Rewrite the query for retrieval.",
    )
    module._llm_json = lambda *, user: {"query": "alice"}  # type: ignore[method-assign]

    packet_out, _ = module.run(Packet(query=Query(text="who is relevant?")), store)

    assert packet_out.query is not None
    assert packet_out.query.text == "alice"
    assert packet_out.query.metadata["rewrite"]["source"] == "llm"
    assert [record.text for record in packet_out.retrieved.items] == ["alice memory"]
    assert packet_out.trace["retrieval"]["wrapped_retriever"] == "keyword_count_retrieval"
    assert packet_out.trace["retrieval"]["query_rewrite"]["returned_query_count"] == 1


def test_query_rewrite_retrieval_llm_multi_query_reuses_delegate_merge() -> None:
    from memprimitive.baselines import KeywordCountRetrieval, QueryRewriteRetrieval

    store = MemoryStore()
    for text in ("alice memory", "bob memory", "carol memory"):
        _, store = _stored_pipeline_packet(text, store)

    module = QueryRewriteRetrieval(
        retriever=KeywordCountRetrieval(top_k=1),
        strategy="llm",
        prompt="Produce retrieval sub-queries.",
        allow_multi_query=True,
        max_queries=4,
    )
    module._llm_json = lambda *, user: {"queries": ["alice", "bob"]}  # type: ignore[method-assign]

    packet_out, _ = module.run(Packet(query=Query(text="find relevant people")), store)

    assert packet_out.query is not None
    assert packet_out.query.text == "alice"
    assert packet_out.queries is not None
    assert [query.text for query in packet_out.queries] == ["alice", "bob"]
    assert [record.text for record in packet_out.retrieved.items] == ["alice memory", "bob memory"]
    assert packet_out.trace["retrieval"]["query_count"] == 2
    assert packet_out.trace["retrieval"]["merge_strategy"] == "query_order_dedupe"
    assert packet_out.trace["retrieval"]["query_rewrite"]["rewritten_query_texts"] == ["alice", "bob"]


def test_query_rewrite_retrieval_llm_normalizes_multi_query_results() -> None:
    from memprimitive.baselines import RecencyRetrieval, QueryRewriteRetrieval

    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="llm",
        prompt="Produce retrieval sub-queries.",
        allow_multi_query=True,
        max_queries=2,
    )
    module._llm_json = lambda *, user: {"queries": [" alice ", "", "alice", "bob", "carol"]}  # type: ignore[method-assign]

    packet_out, _ = module.run(Packet(query=Query(text="seed")), MemoryStore())

    assert packet_out.queries is not None
    assert [query.text for query in packet_out.queries] == ["alice", "bob"]
    rewrite_trace = packet_out.trace["retrieval"]["query_rewrite"]
    assert rewrite_trace["dropped_empty_count"] == 1
    assert rewrite_trace["duplicate_count"] == 1
    assert rewrite_trace["over_limit_count"] == 1


def test_query_rewrite_retrieval_llm_single_query_mode_takes_first_multi_query_result() -> None:
    from memprimitive.baselines import RecencyRetrieval, QueryRewriteRetrieval

    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="llm",
        prompt="Rewrite the query.",
        allow_multi_query=False,
    )
    module._llm_json = lambda *, user: {"queries": ["alice", "bob"]}  # type: ignore[method-assign]

    packet_out, _ = module.run(Packet(query=Query(text="seed")), MemoryStore())

    assert packet_out.query is not None
    assert packet_out.query.text == "alice"
    assert packet_out.queries is None
    assert packet_out.trace["retrieval"]["query_rewrite"]["returned_query_count"] == 1


def test_query_rewrite_retrieval_regex_applies_rules_in_order() -> None:
    from memprimitive.baselines import QueryRewriteRetrieval, RecencyRetrieval

    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="regex",
        regex_rules=[
            {"pattern": "teh", "repl": "the"},
            {"pattern": "\\s+", "repl": " "},
        ],
    )

    packet_out, _ = module.run(Packet(query=Query(text="  teh   graph  ")), MemoryStore())

    assert packet_out.query is not None
    assert packet_out.query.text == "the graph"
    rewrite_trace = packet_out.trace["retrieval"]["query_rewrite"]
    assert rewrite_trace["rule_count"] == 2
    assert rewrite_trace["rewritten_query_text"] == "the graph"
    assert [rule["changed"] for rule in rewrite_trace["rules"]] == [True, True]


def test_query_rewrite_retrieval_regex_supports_flags_and_count() -> None:
    from memprimitive.baselines import QueryRewriteRetrieval, RecencyRetrieval

    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="regex",
        regex_rules=[
            {"pattern": "ALICE", "repl": "bob", "flags": "IGNORECASE", "count": 1},
        ],
    )

    packet_out, _ = module.run(Packet(query=Query(text="ALICE alice ALICE")), MemoryStore())

    assert packet_out.query is not None
    assert packet_out.query.text == "bob alice ALICE"
    rule_trace = packet_out.trace["retrieval"]["query_rewrite"]["rules"][0]
    assert rule_trace["flags"] == ["IGNORECASE"]
    assert rule_trace["replacement_count"] == 1


def test_query_rewrite_retrieval_regex_rejects_empty_result_when_drop_empty_enabled() -> None:
    from memprimitive.baselines import QueryRewriteRetrieval, RecencyRetrieval

    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="regex",
        regex_rules=[{"pattern": ".+", "repl": ""}],
    )

    with pytest.raises(ValueError, match="no usable rewritten queries"):
        module.run(Packet(query=Query(text="alice")), MemoryStore())


def test_query_rewrite_retrieval_prompt_template_trace_is_preserved() -> None:
    from memprimitive.baselines import QueryRewriteRetrieval, RecencyRetrieval

    module = QueryRewriteRetrieval(
        retriever=RecencyRetrieval(top_k=1),
        strategy="llm",
        prompt="Rewrite {{ query.text }} with {{ query.metadata.topic | default('none') }}",
    )
    module._llm_json = lambda *, user: {"query": "alice"}  # type: ignore[method-assign]

    packet_out, _ = module.run(
        Packet(query=Query(text="who", metadata={"topic": "graphs"})),
        MemoryStore(),
    )

    rewrite_trace = packet_out.trace["retrieval"]["query_rewrite"]
    assert rewrite_trace["prompt_is_template"] is True
    assert rewrite_trace["missing_variables"] == []
    assert rewrite_trace["rendered_prompt"] == "Rewrite who with graphs"


def test_triple_memory_retrieval_supports_metadata_subject_relation_query() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Alice likes tea.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"triples": [("Alice", "likes", "tea")]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="profile",
            text="Alice likes coffee.",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"representation": {"triples": [("Alice", "likes", "coffee")]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="profile",
            text="Bob likes tea.",
            timestamp="2026-01-01T00:00:02+00:00",
            metadata={"representation": {"triples": [("Bob", "likes", "tea")]}},
        )
    )

    packet_out, _ = TripleMemoryRetrieval(top_k=2, layer="profile").run(
        Packet(
            query=Query(
                text="unused",
                metadata={
                    "triple_query": {
                        "subject": "Alice",
                        "relation": "likes",
                        "object": "*",
                    }
                },
            )
        ),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert packet_out.retrieved.scores[0]["strategy"] == "triple_memory_exact"
    assert packet_out.retrieved.scores[0]["matched_triples"] == [("Alice", "likes", "coffee")]
    assert packet_out.retrieved.trace["query_mode"] == "subject_relation"
    assert packet_out.retrieved.trace["query_source"] == "metadata.triple_query"
    assert packet_out.retrieved.trace["retrieval_mode"] == "exact"


def test_triple_memory_retrieval_supports_text_relation_object_query_and_graph_triples() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "entity"))]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-g-1",
            unit_id="unit-g-1",
            layer="knowledge_graph",
            text="graph node",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"graph": {"triples": [("Alice", "likes", "tea")], "links": []}},
        )
    )

    packet_out, _ = TripleMemoryRetrieval(top_k=1, layer="knowledge_graph").run(
        Packet(query=Query(text=" >> likes >> tea ")),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-g-1"]
    assert packet_out.retrieved.trace["query_mode"] == "relation_object"
    assert packet_out.retrieved.trace["query_source"] == "query.text"
    assert packet_out.retrieved.trace["query_triple"] == {
        "subject": None,
        "relation": "likes",
        "object": "tea",
    }


def test_triple_memory_retrieval_supports_single_slot_relation_query() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Alice likes tea.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"triples": [("Alice", "likes", "tea")]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="profile",
            text="Bob likes coffee.",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"representation": {"triples": [("Bob", "likes", "coffee")]}},
        )
    )

    packet_out, _ = TripleMemoryRetrieval(top_k=2, layer="profile").run(
        Packet(query=Query(text="* >> likes >> *")),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert packet_out.retrieved.trace["query_mode"] == "relation"
    assert packet_out.retrieved.trace["retrieval_mode"] == "exact"


def test_triple_memory_retrieval_supports_subject_object_query_without_relation() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Alice likes tea.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"triples": [("Alice", "likes", "tea")]}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="profile",
            text="Alice drinks tea.",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"representation": {"triples": [("Alice", "drinks", "tea")]}},
        )
    )

    packet_out, _ = TripleMemoryRetrieval(top_k=2, layer="profile").run(
        Packet(query=Query(text="Alice >> * >> tea")),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert packet_out.retrieved.trace["query_mode"] == "subject_object"
    assert packet_out.retrieved.trace["retrieval_mode"] == "exact"


def test_triple_memory_retrieval_falls_back_to_fuzzy_threshold_matching() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Washington D.C. is the capital of the United States.",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"triples": [("Washington D.C.", "capital of", "United States")]}},
        )
    )

    retriever = TripleMemoryRetrieval(
        top_k=1,
        layer="profile",
        candidate_similarity_threshold=0.7,
        final_similarity_threshold=0.8,
    )
    embedding_map = {
        "usa": [1.0, 0.0],
        "united states": [1.0, 0.0],
        "capital of": [0.0, 1.0],
    }
    retriever._embed_text = lambda text: embedding_map.get(text.strip().casefold(), [0.0, 0.0])  # type: ignore[method-assign]

    packet_out, _ = retriever.run(
        Packet(query=Query(text="* >> capital of >> USA")),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.retrieved.scores[0]["strategy"] == "triple_memory_fuzzy"
    assert packet_out.retrieved.trace["retrieval_mode"] == "fuzzy"
    assert packet_out.retrieved.trace["fallback_used"] is True


def test_triple_memory_retrieval_source_retrieved_limits_candidates() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    candidate = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="profile",
        text="Alice likes tea.",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"representation": {"triples": [("Alice", "likes", "tea")]}},
    )
    outside_match = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="profile",
        text="Alice likes green tea.",
        timestamp="2026-01-01T00:00:01+00:00",
        metadata={"representation": {"triples": [("Alice", "likes", "green tea")]}},
    )
    for record in (candidate, outside_match):
        store.append(record)

    packet_out, _ = TripleMemoryRetrieval(top_k=2, source="retrieved").run(
        Packet(
            query=Query(text="Alice >> likes >> tea"),
            retrieved=RetrievedSet(items=[candidate], scores=[]),
        ),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.retrieved.trace["source"] == "retrieved"
    assert packet_out.retrieved.trace["candidate_count"] == 1


def test_triple_memory_retrieval_rejects_unstructured_query_text() -> None:
    from memprimitive.baselines import TripleMemoryRetrieval

    with pytest.raises(ValueError, match="structured triple query"):
        TripleMemoryRetrieval().run(Packet(query=Query(text="Alice likes tea")), MemoryStore())


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


def test_embedding_similarity_retrieval_embeds_query_through_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import EmbeddingSimilarityRetrieval
    from memprimitive.utils import _runtime

    calls: list[dict[str, object]] = []

    class _FakeRuntime:
        def __init__(
            self,
            *,
            embedding_model: str | None = None,
            embedding_provider: str | None = None,
            embedding_api_key: str | None = None,
            embedding_base_url: str | None = None,
        ) -> None:
            calls.append(
                {
                    "embedding_model": embedding_model,
                    "embedding_provider": embedding_provider,
                    "embedding_api_key": embedding_api_key,
                    "embedding_base_url": embedding_base_url,
                }
            )

        def embed(self, text: str) -> list[float]:
            calls.append({"text": text})
            return [1.0, 0.0]

    monkeypatch.setattr(_runtime, "Runtime", _FakeRuntime)
    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="best",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[1.0, 0.0],
        )
    )

    packet_out, _ = EmbeddingSimilarityRetrieval(
        top_k=1,
        embedding_model="embed-model",
        embedding_provider="openai",
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example/v1",
    ).run(Packet(query=Query(text="alpha query")), store)

    assert packet_out.query.embedding == [1.0, 0.0]
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert calls == [
        {
            "embedding_model": "embed-model",
            "embedding_provider": "openai",
            "embedding_api_key": "embed-key",
            "embedding_base_url": "https://embed.example/v1",
        },
        {"text": "alpha query"},
    ]


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


def test_triple_memory_retrieval_embeds_fuzzy_terms_through_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import TripleMemoryRetrieval
    from memprimitive.utils import _runtime

    calls: list[dict[str, object]] = []

    class _FakeRuntime:
        def __init__(
            self,
            *,
            embedding_model: str | None = None,
            embedding_provider: str | None = None,
            embedding_api_key: str | None = None,
            embedding_base_url: str | None = None,
        ) -> None:
            calls.append(
                {
                    "embedding_model": embedding_model,
                    "embedding_provider": embedding_provider,
                    "embedding_api_key": embedding_api_key,
                    "embedding_base_url": embedding_base_url,
                }
            )

        def embed(self, text: str) -> list[float]:
            calls.append({"text": text})
            return [1.0, 0.0] if text == "alice" else [0.9, 0.1]

    monkeypatch.setattr(_runtime, "Runtime", _FakeRuntime)
    TripleMemoryRetrieval._term_embedding_cache.clear()
    retriever = TripleMemoryRetrieval(
        embedding_model="embed-model",
        embedding_provider="openai",
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example/v1",
    )

    assert retriever._embed_text("Alice") == [1.0, 0.0]
    assert retriever._embed_text("Alicia") == [0.9, 0.1]
    assert calls == [
        {
            "embedding_model": "embed-model",
            "embedding_provider": "openai",
            "embedding_api_key": "embed-key",
            "embedding_base_url": "https://embed.example/v1",
        },
        {"text": "alice"},
        {
            "embedding_model": "embed-model",
            "embedding_provider": "openai",
            "embedding_api_key": "embed-key",
            "embedding_base_url": "https://embed.example/v1",
        },
        {"text": "alicia"},
    ]


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


def test_parent_episode_expansion_dedupes_sentence_hits_and_uses_best_numeric_score() -> None:
    from memprimitive.baselines import ParentEpisodeExpansionRetrieval

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="episodic"),
                StoreLayerSpec(name="sentence"),
            ]
        )
    )
    parent = MemoryRecord(
        record_id="episode-1",
        unit_id="episode-unit-1",
        layer="episodic",
        text="Alice discussed graph memory, then Bob asked about tea.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    store.append(parent)
    hit_a = MemoryRecord(
        record_id="sentence-1",
        unit_id="sentence-unit-1",
        layer="sentence",
        text="Alice discussed graph memory.",
        timestamp="2026-01-01T00:00:01+00:00",
        metadata={
            "parent_episode_record_id": "episode-1",
            "provenance": {"sentence_index": 0, "source": "sentence_split"},
        },
    )
    hit_b = MemoryRecord(
        record_id="sentence-2",
        unit_id="sentence-unit-2",
        layer="sentence",
        text="Bob asked about tea.",
        timestamp="2026-01-01T00:00:02+00:00",
        metadata={
            "provenance": {
                "parent_episode_record_id": "episode-1",
                "sentence_index": 1,
                "source": "sentence_split",
            }
        },
    )

    packet_out, store_out = ParentEpisodeExpansionRetrieval(top_k=3, episode_layer="episodic").run(
        Packet(
            retrieved=RetrievedSet(
                items=[hit_a, hit_b],
                scores=[
                    {"record_id": "sentence-1", "score": 0.2, "strategy": "embedding_similarity"},
                    {"record_id": "sentence-2", "score": 0.9, "strategy": "embedding_similarity"},
                ],
            )
        ),
        store,
    )

    assert store_out.count("episodic") == 1
    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["episode-1"]
    score = packet_out.retrieved.scores[0]
    assert score["record_id"] == "episode-1"
    assert score["score"] == 0.9
    assert score["source_hit_record_id"] == "sentence-2"
    assert score["first_hit_record_id"] == "sentence-1"
    assert score["source_score"]["record_id"] == "sentence-2"
    assert score["parent_id_field"] == "parent_episode_record_id"
    assert score["hit_provenance"]["sentence_index"] == 1
    assert packet_out.retrieved.trace["input_hit_ids"] == ["sentence-1", "sentence-2"]
    assert packet_out.retrieved.trace["successful_parent_ids"] == ["episode-1"]
    assert packet_out.retrieved.trace["duplicate_parent_count"] == 1
    assert [entry["status"] for entry in packet_out.retrieved.trace["hit_to_parent"]] == ["resolved", "resolved"]


def test_parent_episode_expansion_skips_missing_and_unresolved_parent_ids() -> None:
    from memprimitive.baselines import ParentEpisodeExpansionRetrieval

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="episodic"),
                StoreLayerSpec(name="sentence"),
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="episode-1",
            unit_id="episode-unit-1",
            layer="episodic",
            text="Grounded parent episode.",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    missing_parent_hit = MemoryRecord(
        record_id="sentence-missing-parent",
        unit_id="sentence-unit-1",
        layer="sentence",
        text="A derivative hit without provenance.",
        timestamp="2026-01-01T00:00:01+00:00",
        metadata={},
    )
    unresolved_parent_hit = MemoryRecord(
        record_id="sentence-unresolved-parent",
        unit_id="sentence-unit-2",
        layer="sentence",
        text="A derivative hit for an absent parent.",
        timestamp="2026-01-01T00:00:02+00:00",
        metadata={"source_episode_record_id": "episode-missing"},
    )
    resolved_hit = MemoryRecord(
        record_id="sentence-resolved-parent",
        unit_id="sentence-unit-3",
        layer="sentence",
        text="A derivative hit for the stored parent.",
        timestamp="2026-01-01T00:00:03+00:00",
        metadata={"episode_record_id": "episode-1"},
    )

    packet_out, _ = ParentEpisodeExpansionRetrieval(top_k=3, episode_layer="episodic").run(
        Packet(retrieved=RetrievedSet(items=[missing_parent_hit, unresolved_parent_hit, resolved_hit], scores=[])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["episode-1"]
    trace = packet_out.retrieved.trace
    assert trace["missing_parent_count"] == 1
    assert trace["unresolved_parent_count"] == 1
    assert trace["successful_parent_ids"] == ["episode-1"]
    assert [entry["status"] for entry in trace["hit_to_parent"]] == [
        "missing_parent_id",
        "parent_not_found",
        "resolved",
    ]


def test_parent_episode_expansion_returns_empty_when_packet_has_no_retrieved_set() -> None:
    from memprimitive.baselines import ParentEpisodeExpansionRetrieval

    store = MemoryStore()
    before_count = store.count()

    packet_out, store_out = ParentEpisodeExpansionRetrieval(top_k=2).run(Packet(), store)

    assert store_out.count() == before_count
    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.retrieved.trace["candidate_count"] == 0
    assert packet_out.retrieved.trace["input_hit_ids"] == []
    assert packet_out.retrieved.trace["returned_count"] == 0


def test_parent_episode_expansion_retrieval_is_registered_and_exported() -> None:
    import memprimitive.baselines as pkg
    from memprimitive.baselines import ParentEpisodeExpansionRetrieval

    assert ParentEpisodeExpansionRetrieval.__name__ == "ParentEpisodeExpansionRetrieval"
    assert "ParentEpisodeExpansionRetrieval" in registered_baseline_class_names()
    assert "ParentEpisodeExpansionRetrieval" in pkg.__all__


def _temporal_episode_store(records: list[dict[str, object]]) -> tuple[MemoryStore, dict[str, MemoryRecord]]:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="episodic")]))
    by_id: dict[str, MemoryRecord] = {}
    for payload in records:
        record_id = str(payload["record_id"])
        record = MemoryRecord(
            record_id=record_id,
            unit_id=str(payload.get("unit_id", f"unit-{record_id}")),
            layer="episodic",
            text=str(payload.get("text", record_id)),
            timestamp=str(payload["timestamp"]),
            metadata=dict(payload.get("metadata", {})),
        )
        store.append(record)
        by_id[record_id] = record
    return store, by_id


def test_temporal_neighbor_expansion_expands_same_session_window() -> None:
    from memprimitive.baselines import TemporalNeighborExpansionRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": "ep-1", "timestamp": "2026-01-01T00:00:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-2", "timestamp": "2026-01-01T00:01:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-3", "timestamp": "2026-01-01T00:02:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-4", "timestamp": "2026-01-01T00:03:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-5", "timestamp": "2026-01-01T00:04:00+00:00", "metadata": {"session_id": "s1"}},
        ]
    )

    packet_out, _ = TemporalNeighborExpansionRetrieval(layer="episodic").run(
        Packet(retrieved=RetrievedSet(items=[records["ep-3"]], scores=[])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-2", "ep-3", "ep-4", "ep-5"]
    cluster = packet_out.retrieved.trace["clusters"][0]
    assert cluster["nucleus_record_id"] == "ep-3"
    assert cluster["cluster_ids"] == ["ep-2", "ep-3", "ep-4", "ep-5"]
    assert cluster["backward_ids"] == ["ep-2"]
    assert cluster["forward_ids"] == ["ep-4", "ep-5"]


def test_temporal_neighbor_expansion_does_not_cross_session_scope() -> None:
    from memprimitive.baselines import TemporalNeighborExpansionRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": "s1-prev", "timestamp": "2026-01-01T00:00:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "s2-near-prev", "timestamp": "2026-01-01T00:01:00+00:00", "metadata": {"session_id": "s2"}},
            {"record_id": "s1-nucleus", "timestamp": "2026-01-01T00:02:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "s2-near-next", "timestamp": "2026-01-01T00:03:00+00:00", "metadata": {"session_id": "s2"}},
            {"record_id": "s1-next", "timestamp": "2026-01-01T00:04:00+00:00", "metadata": {"session_id": "s1"}},
        ]
    )

    packet_out, _ = TemporalNeighborExpansionRetrieval(layer="episodic", backward=1, forward=2).run(
        Packet(retrieved=RetrievedSet(items=[records["s1-nucleus"]], scores=[])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["s1-prev", "s1-nucleus", "s1-next"]
    assert packet_out.retrieved.trace["clusters"][0]["forward_ids"] == ["s1-next"]


def test_temporal_neighbor_expansion_dedupes_overlapping_clusters() -> None:
    from memprimitive.baselines import TemporalNeighborExpansionRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": f"ep-{index}", "timestamp": f"2026-01-01T00:0{index}:00+00:00", "metadata": {"session_id": "s1"}}
            for index in range(1, 6)
        ]
    )

    packet_out, _ = TemporalNeighborExpansionRetrieval(layer="episodic", backward=1, forward=2).run(
        Packet(retrieved=RetrievedSet(items=[records["ep-2"], records["ep-3"]], scores=[])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-1", "ep-2", "ep-3", "ep-4", "ep-5"]
    assert [cluster["cluster_ids"] for cluster in packet_out.retrieved.trace["clusters"]] == [
        ["ep-1", "ep-2", "ep-3", "ep-4"],
        ["ep-2", "ep-3", "ep-4", "ep-5"],
    ]
    assert packet_out.retrieved.trace["deduped_duplicate_count"] == 3


def test_temporal_neighbor_expansion_returns_chronological_order_from_out_of_order_nuclei() -> None:
    from memprimitive.baselines import TemporalNeighborExpansionRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": "ep-3", "timestamp": "2026-01-01T00:02:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-1", "timestamp": "2026-01-01T00:00:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-2", "timestamp": "2026-01-01T00:01:00+00:00", "metadata": {"session_id": "s1"}},
            {"record_id": "ep-4", "timestamp": "2026-01-01T00:03:00+00:00", "metadata": {"session_id": "s1"}},
        ]
    )

    packet_out, _ = TemporalNeighborExpansionRetrieval(layer="episodic", backward=0, forward=1).run(
        Packet(retrieved=RetrievedSet(items=[records["ep-3"], records["ep-1"]], scores=[])),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-1", "ep-2", "ep-3", "ep-4"]
    assert packet_out.retrieved.trace["ordering_mode"] == "timestamp_record_id"
    assert packet_out.retrieved.trace["returned_ids"] == ["ep-1", "ep-2", "ep-3", "ep-4"]


def test_temporal_neighbor_expansion_retrieval_is_registered_and_exported() -> None:
    import memprimitive.baselines as pkg
    from memprimitive.baselines import TemporalNeighborExpansionRetrieval

    assert TemporalNeighborExpansionRetrieval.__name__ == "TemporalNeighborExpansionRetrieval"
    assert "TemporalNeighborExpansionRetrieval" in registered_baseline_class_names()
    assert "TemporalNeighborExpansionRetrieval" in pkg.__all__


def test_episode_cluster_rerank_prefers_high_scoring_cluster_and_resolves_store_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import EpisodeClusterRerankRetrieval
    from memprimitive.utils import _runtime

    class _FakeClusterRerankRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def rerank(self, *, query: str, candidates: list[dict[str, object]], task: str, top_k: int):
            self.calls.append({"query": query, "candidates": candidates, "task": task, "top_k": top_k})
            scores = {"ep-2": 0.2, "ep-4": 0.95}
            return [
                {
                    "id": str(candidate["id"]),
                    "score": scores[str(candidate["nucleus_record_id"])],
                    "rationale": "fake cluster rank",
                }
                for candidate in sorted(
                    candidates,
                    key=lambda item: -scores[str(item["nucleus_record_id"])],
                )[:top_k]
            ]

    fake_runtime = _FakeClusterRerankRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store, records = _temporal_episode_store(
        [
            {"record_id": "ep-1", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "ep-2", "timestamp": "2026-01-01T00:01:00+00:00"},
            {"record_id": "ep-3", "timestamp": "2026-01-01T00:02:00+00:00"},
            {"record_id": "ep-4", "timestamp": "2026-01-01T00:03:00+00:00"},
            {"record_id": "ep-5", "timestamp": "2026-01-01T00:04:00+00:00"},
        ]
    )
    retrieved = RetrievedSet(
        items=[records["ep-2"], records["ep-4"]],
        scores=[
            {"record_id": "ep-2", "score": 0.2},
            {"record_id": "ep-4", "score": 0.95},
        ],
        trace={
            "layer": "episodic",
            "clusters": [
                {
                    "nucleus_record_id": "ep-2",
                    "cluster_ids": ["ep-1", "ep-2"],
                    "backward_ids": ["ep-1"],
                    "forward_ids": [],
                },
                {
                    "nucleus_record_id": "ep-4",
                    "cluster_ids": ["ep-4", "ep-5"],
                    "backward_ids": [],
                    "forward_ids": ["ep-5"],
                },
            ],
        },
    )

    packet_out, _ = EpisodeClusterRerankRetrieval(top_k=2, cluster_top_k=1).run(
        Packet(query=Query(text="important ep-4 context"), retrieved=retrieved),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-4", "ep-5"]
    assert fake_runtime.calls[0]["top_k"] == 1
    assert packet_out.retrieved.trace["reranked_clusters"][0]["nucleus_record_id"] == "ep-4"
    assert packet_out.retrieved.trace["returned_ids"] == ["ep-4", "ep-5"]


def test_episode_cluster_rerank_dedupes_overlapping_clusters_and_merges_provenance() -> None:
    from memprimitive.baselines import EpisodeClusterRerankRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": f"ep-{index}", "timestamp": f"2026-01-01T00:0{index}:00+00:00"}
            for index in range(1, 5)
        ]
    )
    retrieved = RetrievedSet(
        items=list(records.values()),
        scores=[],
        trace={
            "layer": "episodic",
            "clusters": [
                {
                    "nucleus_record_id": "ep-2",
                    "cluster_ids": ["ep-1", "ep-2", "ep-3"],
                    "backward_ids": ["ep-1"],
                    "forward_ids": ["ep-3"],
                },
                {
                    "nucleus_record_id": "ep-3",
                    "cluster_ids": ["ep-2", "ep-3", "ep-4"],
                    "backward_ids": ["ep-2"],
                    "forward_ids": ["ep-4"],
                },
            ],
        },
    )

    packet_out, _ = EpisodeClusterRerankRetrieval(top_k=5, rerank=False, chronological=False).run(
        Packet(query=Query(text="dedupe overlaps"), retrieved=retrieved),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-1", "ep-2", "ep-3", "ep-4"]
    scores_by_id = {score["record_id"]: score for score in packet_out.retrieved.scores}
    assert scores_by_id["ep-2"]["source_nucleus_record_ids"] == ["ep-2", "ep-3"]
    assert scores_by_id["ep-2"]["roles"] == ["nucleus", "backward"]
    assert scores_by_id["ep-3"]["source_nucleus_record_ids"] == ["ep-2", "ep-3"]
    assert scores_by_id["ep-3"]["roles"] == ["forward", "nucleus"]
    assert packet_out.retrieved.trace["deduped_duplicate_count"] == 2


def test_episode_cluster_rerank_partial_budget_prefers_forward_neighbor_before_backward_on_ties() -> None:
    from memprimitive.baselines import EpisodeClusterRerankRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": f"ep-{index}", "timestamp": f"2026-01-01T00:0{index}:00+00:00"}
            for index in range(1, 6)
        ]
    )
    retrieved = RetrievedSet(
        items=list(records.values()),
        scores=[],
        trace={
            "layer": "episodic",
            "clusters": [
                {
                    "nucleus_record_id": "ep-3",
                    "cluster_ids": ["ep-1", "ep-2", "ep-3", "ep-4", "ep-5"],
                    "backward_ids": ["ep-1", "ep-2"],
                    "forward_ids": ["ep-4", "ep-5"],
                },
            ],
        },
    )

    packet_out, _ = EpisodeClusterRerankRetrieval(top_k=3, rerank=False, chronological=False).run(
        Packet(query=Query(text="tight budget"), retrieved=retrieved),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-3", "ep-4", "ep-2"]
    assert packet_out.retrieved.trace["budget_truncated_count"] == 2


def test_episode_cluster_rerank_returns_chronological_order() -> None:
    from memprimitive.baselines import EpisodeClusterRerankRetrieval

    store, records = _temporal_episode_store(
        [
            {"record_id": "ep-3", "timestamp": "2026-01-01T00:02:00+00:00"},
            {"record_id": "ep-1", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "ep-4", "timestamp": "2026-01-01T00:03:00+00:00"},
            {"record_id": "ep-2", "timestamp": "2026-01-01T00:01:00+00:00"},
        ]
    )
    retrieved = RetrievedSet(
        items=list(records.values()),
        scores=[],
        trace={
            "layer": "episodic",
            "clusters": [
                {
                    "nucleus_record_id": "ep-3",
                    "cluster_ids": ["ep-3", "ep-4"],
                    "backward_ids": [],
                    "forward_ids": ["ep-4"],
                },
                {
                    "nucleus_record_id": "ep-1",
                    "cluster_ids": ["ep-1", "ep-2"],
                    "backward_ids": [],
                    "forward_ids": ["ep-2"],
                },
            ],
        },
    )

    packet_out, _ = EpisodeClusterRerankRetrieval(top_k=4, rerank=False).run(
        Packet(query=Query(text="chronological context"), retrieved=retrieved),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["ep-1", "ep-2", "ep-3", "ep-4"]
    assert packet_out.retrieved.trace["returned_ids"] == ["ep-1", "ep-2", "ep-3", "ep-4"]


def test_episode_cluster_rerank_returns_empty_for_missing_or_unresolved_clusters() -> None:
    from memprimitive.baselines import EpisodeClusterRerankRetrieval

    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="episodic")]))
    no_cluster_packet, _ = EpisodeClusterRerankRetrieval().run(
        Packet(query=Query(text="no clusters"), retrieved=RetrievedSet(items=[], scores=[], trace={})),
        store,
    )

    assert no_cluster_packet.retrieved is not None
    assert no_cluster_packet.retrieved.items == []
    assert no_cluster_packet.retrieved.trace["empty_reason"] == "no_clusters"
    assert no_cluster_packet.retrieved.trace["input_cluster_count"] == 0

    unresolved_packet, _ = EpisodeClusterRerankRetrieval().run(
        Packet(
            query=Query(text="missing episode"),
            retrieved=RetrievedSet(
                items=[],
                scores=[],
                trace={
                    "layer": "episodic",
                    "clusters": [
                        {
                            "nucleus_record_id": "missing",
                            "cluster_ids": ["missing"],
                            "backward_ids": [],
                            "forward_ids": [],
                        }
                    ],
                },
            ),
        ),
        store,
    )

    assert unresolved_packet.retrieved is not None
    assert unresolved_packet.retrieved.items == []
    assert unresolved_packet.retrieved.trace["empty_reason"] == "no_resolved_clusters"
    assert unresolved_packet.retrieved.trace["input_cluster_count"] == 1
    assert unresolved_packet.retrieved.trace["unresolved_cluster_count"] == 1


def test_episode_cluster_rerank_retrieval_is_registered_and_exported() -> None:
    import memprimitive.baselines as pkg
    from memprimitive.baselines import EpisodeClusterRerankRetrieval

    assert EpisodeClusterRerankRetrieval.__name__ == "EpisodeClusterRerankRetrieval"
    assert "EpisodeClusterRerankRetrieval" in registered_baseline_class_names()
    assert "EpisodeClusterRerankRetrieval" in pkg.__all__


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


def test_layer_aware_retrieval_merges_multi_query_results_and_preserves_per_query_trace() -> None:
    from memprimitive.baselines import KeywordCountRetrieval, LayerAwareRetrieval

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
            text="working memory",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-semantic-1",
            unit_id="unit-semantic-1",
            layer="semantic",
            text="semantic memory",
            timestamp="2026-01-01T00:00:01+00:00",
        )
    )

    packet_out, _ = LayerAwareRetrieval(
        default_retriever=KeywordCountRetrieval(top_k=1),
        top_k=1,
    ).run(
        Packet(
            queries=[
                Query(text="semantic"),
                Query(text="working"),
            ]
        ),
        store,
    )

    assert packet_out.retrieved is not None
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-semantic-1", "rec-working-1"]
    assert packet_out.retrieved.trace["query_count"] == 2
    assert packet_out.retrieved.trace["per_query"][0]["trace"]["module"] == "layer_aware_retrieval"
    assert packet_out.retrieved.trace["per_query"][0]["trace"]["per_layer"][0]["module"] == "keyword_count_retrieval"


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
        "GraphNeighborAppendEvolution",
        "BulletListReadout",
        "GroupedByLayerReadout",
        "GraphEntityAppendOrganization",
        "TagRetrieval",
        "ConditionalLayerOrganization",
        "LineSplitUnitFormation",
        "WindowedUnitFormation",
        "MetadataHintUnitFormation",
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

