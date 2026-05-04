from __future__ import annotations

import json
from typing import Any
import pytest

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
    _seed_layer,
    _stored_pipeline_packet,
)


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


def test_metadata_retrieval_exact_match_on_scalar_metadata_field() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Graph memory note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"topic": "graphs"},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="Tea note",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"topic": "tea"},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=3, field="topic", target="graphs").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.retrieved.trace["matched_count"] == 1
    assert packet_out.retrieved.trace["match_mode"] == "exact"


def test_metadata_retrieval_exact_match_is_case_insensitive() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Alice record",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"owner": "Alice"},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=1, field="owner", target="alice").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]


def test_metadata_retrieval_regex_match_on_scalar_metadata_field() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Graph memory note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"topic": "graph-memory"},
        )
    )

    packet_out, _ = MetadataRetrieval(
        top_k=1,
        field="topic",
        target=r"graph.*memory",
        match_mode="regex",
    ).run(Packet(query=Query(text="ignored")), store)

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.retrieved.trace["match_mode"] == "regex"


def test_metadata_retrieval_iterable_field_matches_any_member() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Tagged note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"tags": ["tea", "graphs", "memory"]},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=1, field="tags", target="graphs").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]


def test_metadata_retrieval_iterable_field_skips_non_matching_members() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Tagged note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"tags": ["tea", "memory"]},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=1, field="tags", target="graphs").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.trace["matched_count"] == 0


def test_metadata_retrieval_treats_string_metadata_as_single_value() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Single string note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"code": "abc"},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=1, field="code", target="b").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert packet_out.retrieved.items == []


def test_metadata_retrieval_skips_missing_metadata_field() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Missing field note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"other": "graphs"},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=1, field="topic", target="graphs").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.trace["candidate_count"] == 1


def test_metadata_retrieval_honors_layer_filtering() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile"), StoreLayerSpec(name="history")])
    )
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="profile",
            text="Profile graph note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"topic": "graphs"},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="history",
            text="History graph note",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"topic": "graphs"},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=2, field="topic", target="graphs", layer="profile").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-1"]
    assert packet_out.retrieved.trace["layer"] == "profile"


def test_metadata_retrieval_source_retrieved_limits_candidate_subset() -> None:
    from memprimitive.baselines import MetadataRetrieval

    candidate_a = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="default",
        text="Tea note",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"topic": "tea"},
    )
    candidate_b = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="default",
        text="Graph note",
        timestamp="2026-01-01T00:00:01+00:00",
        metadata={"topic": "graphs"},
    )
    better_outside = MemoryRecord(
        record_id="rec-3",
        unit_id="unit-3",
        layer="default",
        text="Graph note outside subset",
        timestamp="2026-01-01T00:00:02+00:00",
        metadata={"topic": "graphs"},
    )
    store = MemoryStore()
    for record in (candidate_a, candidate_b, better_outside):
        store.append(record)

    packet_out, _ = MetadataRetrieval(top_k=2, field="topic", target="graphs", source="retrieved").run(
        Packet(query=Query(text="ignored"), retrieved=RetrievedSet(items=[candidate_a, candidate_b], scores=[])),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2"]
    assert packet_out.retrieved.trace["source"] == "retrieved"
    assert packet_out.retrieved.trace["candidate_count"] == 2


def test_metadata_retrieval_orders_matches_by_recency() -> None:
    from memprimitive.baselines import MetadataRetrieval

    store = MemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="Older graph note",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"topic": "graphs"},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="default",
            text="Newer graph note",
            timestamp="2026-01-01T00:00:01+00:00",
            metadata={"topic": "graphs"},
        )
    )

    packet_out, _ = MetadataRetrieval(top_k=2, field="topic", target="graphs").run(
        Packet(query=Query(text="ignored")),
        store,
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]


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


def test_reranker_retrieval_reranks_existing_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import RerankerRetrieval
    from memprimitive.utils import _runtime

    calls: list[dict[str, Any]] = []

    class FakeRuntime:
        def rerank(self, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return [
                {"id": "rec-2", "score": 0.91, "rationale": "matches tea"},
                {"id": "rec-1", "score": 0.2, "rationale": ""},
            ]

    monkeypatch.setattr(_runtime, "get_runtime", lambda: FakeRuntime())
    candidate_a = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="default",
        text="Alice likes coffee.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    candidate_b = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="default",
        text="Alice likes jasmine tea.",
        timestamp="2026-01-01T00:00:01+00:00",
    )

    packet_out, _ = RerankerRetrieval().run(
        Packet(
            query=Query(text="tea preference"),
            retrieved=RetrievedSet(items=[candidate_a, candidate_b], scores=[]),
        ),
        MemoryStore(),
    )

    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-2", "rec-1"]
    assert packet_out.retrieved.scores == [
        {
            "record_id": "rec-2",
            "rank": 1,
            "score": 0.91,
            "rationale": "matches tea",
            "strategy": "runtime_rerank",
        },
        {
            "record_id": "rec-1",
            "rank": 2,
            "score": 0.2,
            "rationale": "",
            "strategy": "runtime_rerank",
        },
    ]
    assert calls == [
        {
            "query": "tea preference",
            "candidates": [
                {"id": "rec-1", "content": "Alice likes coffee."},
                {"id": "rec-2", "content": "Alice likes jasmine tea."},
            ],
            "task": "Rerank memory records for retrieval relevance.",
            "top_k": 2,
        }
    ]
    assert packet_out.retrieved.trace["source"] == "retrieved"
    assert packet_out.retrieved.trace["top_k"] is None
    assert packet_out.retrieved.trace["effective_top_k"] == 2
    assert packet_out.retrieved.trace["returned_ids"] == ["rec-2", "rec-1"]


def test_reranker_retrieval_top_k_limits_runtime_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import RerankerRetrieval
    from memprimitive.utils import _runtime

    calls: list[dict[str, Any]] = []

    class FakeRuntime:
        def rerank(self, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return [{"id": "rec-3", "score": 0.8, "rationale": ""}]

    monkeypatch.setattr(_runtime, "get_runtime", lambda: FakeRuntime())
    candidates = [
        MemoryRecord(
            record_id=f"rec-{index}",
            unit_id=f"unit-{index}",
            layer="default",
            text=f"candidate {index}",
            timestamp=f"2026-01-01T00:00:0{index}+00:00",
        )
        for index in range(1, 4)
    ]

    packet_out, _ = RerankerRetrieval(top_k=1).run(
        Packet(query=Query(text="candidate 3"), retrieved=RetrievedSet(items=candidates, scores=[])),
        MemoryStore(),
    )

    assert calls[0]["top_k"] == 1
    assert [record.record_id for record in packet_out.retrieved.items] == ["rec-3"]
    assert packet_out.retrieved.trace["effective_top_k"] == 1
    assert packet_out.retrieved.trace["selected_count"] == 1


def test_reranker_retrieval_source_store_filters_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import RerankerRetrieval
    from memprimitive.utils import _runtime

    calls: list[dict[str, Any]] = []

    class FakeRuntime:
        def rerank(self, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return [{"id": "profile-2", "score": 0.7, "rationale": ""}]

    monkeypatch.setattr(_runtime, "get_runtime", lambda: FakeRuntime())
    store = MemoryStore(
        topology=StoreTopology.from_layers([StoreLayerSpec(name="profile"), StoreLayerSpec(name="history")])
    )
    for record in (
        MemoryRecord(
            record_id="profile-1",
            unit_id="unit-1",
            layer="profile",
            text="profile coffee",
            timestamp="2026-01-01T00:00:00+00:00",
        ),
        MemoryRecord(
            record_id="profile-2",
            unit_id="unit-2",
            layer="profile",
            text="profile tea",
            timestamp="2026-01-01T00:00:01+00:00",
        ),
        MemoryRecord(
            record_id="history-1",
            unit_id="unit-3",
            layer="history",
            text="history tea",
            timestamp="2026-01-01T00:00:02+00:00",
        ),
    ):
        store.append(record)

    packet_out, _ = RerankerRetrieval(source="store", layer="profile").run(Packet(query=Query(text="tea")), store)

    assert [candidate["id"] for candidate in calls[0]["candidates"]] == ["profile-1", "profile-2"]
    assert [record.record_id for record in packet_out.retrieved.items] == ["profile-2"]
    assert packet_out.retrieved.trace["source"] == "store"
    assert packet_out.retrieved.trace["layer"] == "profile"


def test_reranker_retrieval_empty_candidates_skip_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import RerankerRetrieval
    from memprimitive.utils import _runtime

    monkeypatch.setattr(_runtime, "get_runtime", lambda: pytest.fail("reranker should not be called"))

    packet_out, _ = RerankerRetrieval().run(
        Packet(query=Query(text="tea"), retrieved=RetrievedSet(items=[], scores=[])),
        MemoryStore(),
    )

    assert packet_out.retrieved.items == []
    assert packet_out.retrieved.scores == []
    assert packet_out.retrieved.trace["candidate_count"] == 0
    assert packet_out.retrieved.trace["selected_count"] == 0


def test_reranker_retrieval_requires_query() -> None:
    from memprimitive.baselines import RerankerRetrieval

    with pytest.raises(ValueError, match="packet.query"):
        RerankerRetrieval().run(Packet(), MemoryStore())


def test_reranker_retrieval_requires_positive_top_k() -> None:
    from memprimitive.baselines import RerankerRetrieval

    with pytest.raises(ValueError, match="top_k > 0"):
        RerankerRetrieval(top_k=0)


def test_reranker_retrieval_is_registered() -> None:
    from memprimitive.baselines import RerankerRetrieval, registered_baseline_class_names

    assert "RerankerRetrieval" in registered_baseline_class_names()
    assert RerankerRetrieval.spec.name == "reranker_retrieval"


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


def test_template_readout_text_prompt_can_fill_multiple_labeled_recalled_prompts() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import ConcatenateReadout, RecencyRetrieval, TemplateReadout
    from memprimitive.utils._template import text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile", indices=("temporal",)),
                StoreLayerSpec(name="history", indices=("temporal",)),
            ]
        )
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
    store.append(
        MemoryRecord(
            record_id="rec-history-1",
            unit_id="unit-history-1",
            layer="history",
            text="Alice prefers direct replies",
            timestamp="2026-01-02T00:00:00+00:00",
        )
    )

    profile_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=TemplateReadout(
            prompt=text_prompt("{{ retrieved.items.0.representation.user_profile }}")
        ),
        store=MemoryStore(),
    )
    history_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="history"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )

    packet_out, _ = TemplateReadout(
        prompt=text_prompt(
            "Profile={{ profile }} | History={{ history }}",
            recall_query_builder=lambda packet, current_store, context: f"memory for {context['query']['text']}",
            labeled_recall_plans={
                "profile": text_prompt("{{ retrieved.items.0.representation.user_profile }}", metadata_mode="readout"),
                "history": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            },
            labeled_sub_recall_pipelines={
                "profile": profile_pipeline,
                "history": history_pipeline,
            },
        )
    ).run(Packet(query=Query(text="Alice"), retrieved=RetrievedSet()), store)

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Profile=Concise, concrete, technical. | History=Alice prefers direct replies"
    assert packet_out.readout.metadata["recalled_prompt"] == ""
    assert packet_out.readout.metadata["labeled_recalled_prompts"] == {
        "profile": "Concise, concrete, technical.",
        "history": "Alice prefers direct replies",
    }
    assert packet_out.readout.metadata["labeled_recall_prompts"]["profile"]["rendered_recall_query"] == "memory for Alice"
    assert packet_out.readout.metadata["labeled_recall_prompts"]["history"]["rendered_recall_query"] == "memory for Alice"
    assert packet_out.readout.metadata["labeled_recall_prompts"]["profile"]["readout_source_ids"] == ["rec-profile-1"]
    assert packet_out.readout.metadata["labeled_recall_prompts"]["history"]["readout_source_ids"] == ["rec-history-1"]


def test_template_readout_structured_prompt_can_fill_multiple_labeled_recalled_prompts_with_overrides() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import ConcatenateReadout, RecencyRetrieval, TemplateReadout
    from memprimitive.utils._template import structured_prompt, text_prompt

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="profile", indices=("temporal",)),
                StoreLayerSpec(name="history", indices=("temporal",)),
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-profile-1",
            unit_id="unit-profile-1",
            layer="profile",
            text="Alice profile memory",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"representation": {"user_profile": "Profile text"}},
        )
    )

    profile_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=TemplateReadout(
            prompt=text_prompt("{{ retrieved.items.0.representation.user_profile }}")
        ),
        store=MemoryStore(),
    )
    history_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="history"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )

    packet_out, _ = TemplateReadout(
        prompt=structured_prompt(
            {
                "blocks": [
                    {"id": "profile", "title": "Profile", "template": "{{ profile }}"},
                    {"id": "history", "title": "History", "template": "{{ history }}"},
                ]
            },
            recall_query_builder=lambda packet, current_store, context: "shared query",
            labeled_recall_plans={
                "profile": text_prompt("{{ retrieved.items.0.representation.user_profile }}", metadata_mode="readout"),
                "history": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            },
            labeled_recall_query_builders={
                "history": lambda packet, current_store, context: "",
            },
            labeled_sub_recall_pipelines={
                "profile": profile_pipeline,
                "history": history_pipeline,
            },
        )
    ).run(Packet(query=Query(text="Alice"), retrieved=RetrievedSet()), store)

    assert packet_out.readout is not None
    assert "Profile\nProfile text" in packet_out.readout.text
    assert "History" in packet_out.readout.text
    assert packet_out.readout.metadata["labeled_recalled_prompts"] == {
        "profile": "Profile text",
        "history": "",
    }
    assert packet_out.readout.metadata["labeled_recall_prompts"]["profile"]["rendered_recall_query"] == "shared query"
    assert packet_out.readout.metadata["labeled_recall_prompts"]["history"]["rendered_recall_query"] == ""
    assert (
        packet_out.readout.metadata["labeled_recall_prompts"]["history"]["disabled_reason"]
        == "empty_rendered_recall_query"
    )


def test_template_readout_labeled_recalled_prompt_missing_config_degrades_without_error() -> None:
    from memprimitive.baselines import TemplateReadout
    from memprimitive.utils._template import text_prompt

    packet_out, _ = TemplateReadout(
        prompt=text_prompt(
            "Profile={{ profile }} History={{ history }}",
            recall_query_builder=lambda packet, current_store, context: "query",
            labeled_recall_plans={
                "profile": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            },
            labeled_sub_recall_pipelines={
                "history": object(),
            },
        )
    ).run(Packet(query=Query(text="Alice"), retrieved=RetrievedSet()), MemoryStore())

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Profile= History="
    assert packet_out.readout.metadata["labeled_recalled_prompts"] == {"profile": "", "history": ""}
    assert (
        packet_out.readout.metadata["labeled_recall_prompts"]["profile"]["disabled_reason"]
        == "missing_recall_plan_or_query_builder_or_pipeline"
    )
    assert (
        packet_out.readout.metadata["labeled_recall_prompts"]["history"]["disabled_reason"]
        == "missing_recall_plan_or_query_builder_or_pipeline"
    )


def test_llm_representation_prompt_template_supports_multiple_labeled_recalled_prompts() -> None:
    from memprimitive.baselines import ConcatenateReadout, LLMRepresentation, PassThroughUnitFormation, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils._template import text_prompt

    unit_packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice is preparing a reply.", source="notes")),
        MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile"), StoreLayerSpec(name="history")])),
    )
    _seed_layer(store, "profile", ["CURRENT STORE PROFILE"])
    _seed_layer(store, "history", ["CURRENT STORE HISTORY"])

    profile_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )
    history_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="history"),
        readout=ConcatenateReadout(),
        store=MemoryStore(),
    )

    rep = LLMRepresentation(
        field="summary",
        prompt=text_prompt(
            "Use {{ profile }} and {{ history }} while summarizing {{ unit.text }}",
            recall_query_builder=lambda packet, current_store, context: f"shared for {context['unit']['text']}",
            labeled_recall_plans={
                "profile": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
                "history": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            },
            labeled_sub_recall_pipelines={
                "profile": profile_pipeline,
                "history": history_pipeline,
            },
        ),
    )

    def _fake_llm_text(*, user: str) -> str:
        payload = json.loads(user)
        assert payload["prompt"] == (
            "Use CURRENT STORE PROFILE and CURRENT STORE HISTORY while summarizing Alice is preparing a reply."
        )
        return "summary with labeled recalled prompts"

    rep._llm_text = _fake_llm_text  # type: ignore[method-assign]
    packet_out, _ = rep.run(unit_packet, store)

    prompt_trace = packet_out.trace["representation"]["per_unit"][0]
    assert prompt_trace["recalled_prompt"] == ""
    assert prompt_trace["labeled_recalled_prompts"] == {
        "profile": "CURRENT STORE PROFILE",
        "history": "CURRENT STORE HISTORY",
    }
    assert prompt_trace["labeled_recall_prompts"]["profile"]["rendered_recall_query"] == (
        "shared for Alice is preparing a reply."
    )
    assert prompt_trace["labeled_recall_prompts"]["history"]["rendered_recall_query"] == (
        "shared for Alice is preparing a reply."
    )


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
