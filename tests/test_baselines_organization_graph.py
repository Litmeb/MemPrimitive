from __future__ import annotations

from dataclasses import replace
import json
from typing import Any
import pytest

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

from baselines_test_helpers import (
    _FakeAMEMRuntime,
    _graph_store,
    _graph_vector_store,
)


def test_sentence_split_unit_formation_splits_sentences_and_preserves_provenance() -> None:
    from memprimitive.baselines import SentenceSplitUnitFormation

    packet_out, _ = SentenceSplitUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea. Bob prefers coffee!", source="dialogue")),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.text for unit in packet_out.units] == ["Alice likes tea.", "Bob prefers coffee!"]
    assert all("provenance" in unit.metadata for unit in packet_out.units)


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


def test_graph_append_organization_requires_graph_layer_and_writes_graph_metadata() -> None:
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
                StoreLayerSpec(
                    name="knowledge_graph",
                    theme="semantic",
                    shape="Graph",
                    indices=("graph", "entity", "vector"),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
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


def test_graph_append_organization_preserves_standard_record_embedding_shape() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            represented = replace(represented, embedding=[1.0, 2.0, 3.0])
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata["representation"],
                        "embedding": {"dim": 3},
                        "entity_embeddings": {
                            "Alice": [1.0, 0.0, 0.0],
                            "tea": [0.0, 1.0, 0.0],
                        },
                    },
                },
            )
            return represented, {"source": "test_seed"}

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="knowledge_graph",
                    theme="semantic",
                    shape="Graph",
                    indices=("graph", "entity", "vector"),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphAppendOrganization(target_layer="knowledge_graph").run(packet, store)

    record = store.iter_records("knowledge_graph")[0]
    assert record.embedding == [1.0, 2.0, 3.0]
    assert record.metadata["representation"]["embedding"] == {"dim": 3}
    assert "embedding" not in record.metadata["graph"]
    assert packet.trace["organization"]["writes_embedding_from_record_field"] is True
    assert packet.trace["organization"]["records_with_embedding"] == 1


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


def test_graph_entity_deduplication_append_organization_writes_one_record_per_entity_when_unmatched() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphEntityDeduplicationAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            represented = replace(represented, embedding=[1.0, 2.0, 3.0])
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata["representation"],
                        "embedding": {"dim": 3},
                        "entity_embeddings": {
                            "Alice": [1.0, 0.0, 0.0],
                            "tea": [0.0, 1.0, 0.0],
                        },
                    },
                },
            )
            return represented, {"source": "test_seed"}

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphEntityDeduplicationAppendOrganization(
        target_layer="knowledge_graph",
        threshold=0.95,
    ).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert [record.text for record in records] == ["Alice", "tea"]
    embedding_by_text = {record.text: record.embedding for record in records}
    assert embedding_by_text == {"Alice": [1.0, 0.0, 0.0], "tea": [0.0, 1.0, 0.0]}
    assert all(record.metadata["representation"]["embedding"] == {"dim": 3} for record in records)
    assert all(record.metadata["graph"]["entities"] == ["Alice", "tea"] for record in records)
    assert all(record.metadata["graph"]["triples"] == [("Alice", "likes", "tea")] for record in records)
    assert packet.trace["organization"]["fanout_mode"] == "per_entity"
    assert packet.trace["organization"]["entity_written_record_ids"] == [record.record_id for record in records]
    assert packet.trace["organization"]["written_unit_ids"] == [packet.units[0].unit_id, packet.units[0].unit_id]
    assert packet.trace["organization"]["records_with_embedding"] == 2
    assert packet.trace["organization"]["skipped_unit_count"] == 0
    assert [effect["effect_type"] for effect in packet.trace["organization"]["effects"]] == ["append", "append"]


def test_graph_entity_deduplication_append_organization_separate_mode_writes_source_and_entity_layers() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphEntityDeduplicationAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            triples = [("Alice", "likes", "tea")]
            entities = ["Alice", "tea"]
            represented = self._replace_unit(unit, unit.text.strip(), unit.text.strip().casefold(), entities, triples)
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata["representation"],
                        "entity_embeddings": {
                            "Alice": [1.0, 0.0, 0.0],
                            "tea": [0.0, 1.0, 0.0],
                        },
                    },
                },
            )
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
    packet, store = GraphEntityDeduplicationAppendOrganization(
        target_layer="knowledge_graph",
        threshold=0.95,
        separate=True,
        separate_layer="source_notes",
    ).run(packet, store)

    source_record = store.iter_records("source_notes")[0]
    entity_records = store.iter_records("knowledge_graph")
    assert source_record.text == "Alice likes tea."
    assert [record.text for record in entity_records] == ["Alice", "tea"]
    assert all("graph" in record.metadata for record in entity_records)
    assert all("hierarchical" in record.metadata for record in entity_records)
    assert all(record.metadata["hierarchical"]["source_record_ids"] == [source_record.record_id] for record in entity_records)
    assert all(record.metadata["hierarchical"]["source_unit_ids"] == [source_record.unit_id] for record in entity_records)
    assert all(
        record.metadata["hierarchical"]["field_payload"]["triples"] == [("Alice", "likes", "tea")]
        for record in entity_records
    )
    assert packet.trace["organization"]["source_written_record_ids"] == [source_record.record_id]
    assert packet.trace["organization"]["entity_written_record_ids"] == [record.record_id for record in entity_records]


def test_graph_entity_deduplication_append_organization_skips_units_without_entities() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphEntityDeduplicationAppendOrganization, PassThroughUnitFormation

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="No entities here.", source="notes")),
        store,
    )
    packet = replace(
        packet,
        units=[
            replace(
                packet.units[0],
                entities=[],
                triples=[],
                metadata={
                    **packet.units[0].metadata,
                    "representation": {
                        **packet.units[0].metadata.get("representation", {}),
                        "entities": [],
                        "triples": [],
                    },
                },
            )
        ],
    )
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphEntityDeduplicationAppendOrganization(
        target_layer="knowledge_graph",
        threshold=0.95,
    ).run(packet, store)

    assert store.count("knowledge_graph") == 0
    assert packet.trace["organization"]["written_record_ids"] == []
    assert packet.trace["organization"]["entity_written_record_ids"] == []
    assert packet.trace["organization"]["skipped_unit_count"] == 1
    assert packet.trace["organization"]["effects"] == [{"unit_id": packet.units[0].unit_id, "effect_type": "skipped_no_entities"}]


def test_graph_deduplication_append_organization_merges_top1_match_and_dedupes_relation_destination_pairs() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphDeduplicationAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        _BY_TEXT = {
            "Alice likes tea.": {
                "triples": [("Alice", "likes", "tea"), ("Alice", "visits", "park")],
                "entities": ["Alice", "tea", "park"],
                "embedding": [1.0, 0.0, 0.0],
            },
            "Alice loves tea and visits library.": {
                "triples": [("Alicia", "likes", "tea"), ("Alicia", "likes", "library"), ("Alicia", "knows", "tea")],
                "entities": ["Alicia", "tea", "library"],
                "embedding": [0.95, 0.05, 0.0],
            },
        }

        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            payload = self._BY_TEXT[unit.text.strip()]
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=list(payload["entities"]),
                triples=list(payload["triples"]),
                embedding=list(payload["embedding"]),
                representation_elements=("text", "embedding", "triples"),
            )
            return represented, {"source": "test_seed"}

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text="Alice likes tea.", source="notes")), store)
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.8).run(packet, store)

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice loves tea and visits library.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.8).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert len(records) == 1
    record = records[0]
    assert record.text == "Alice loves tea and visits library."
    assert record.embedding == [0.95, 0.05, 0.0]
    assert record.metadata["graph"]["entities"] == ["Alicia", "tea", "library"]
    assert record.metadata["graph"]["triples"] == [
        ("Alicia", "likes", "tea"),
        ("Alice", "visits", "park"),
        ("Alicia", "likes", "library"),
        ("Alicia", "knows", "tea"),
    ]
    effect = packet.trace["organization"]["effects"][0]
    assert effect["effect_type"] == "merge"
    assert effect["matched_record_id"] == record.record_id
    assert effect["top1_similarity"] > 0.8
    assert effect["embedding_source"] == "existing_unit_embedding"
    assert effect["record_has_embedding"] is True
    assert packet.trace["organization"]["written_record_ids"] == [record.record_id]


def test_graph_deduplication_append_organization_appends_when_threshold_not_met() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphDeduplicationAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        _BY_TEXT = {
            "Alice likes tea.": {
                "triples": [("Alice", "likes", "tea")],
                "entities": ["Alice", "tea"],
                "embedding": [1.0, 0.0],
            },
            "Bob builds tools.": {
                "triples": [("Bob", "builds", "tools")],
                "entities": ["Bob", "tools"],
                "embedding": [0.0, 1.0],
            },
        }

        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            payload = self._BY_TEXT[unit.text.strip()]
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=list(payload["entities"]),
                triples=list(payload["triples"]),
                embedding=list(payload["embedding"]),
                representation_elements=("text", "embedding", "triples"),
            )
            return represented, {"source": "test_seed"}

    store = _graph_store()
    for text in ("Alice likes tea.", "Bob builds tools."):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = SeededTripleRepresentation().run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
        packet, store = GraphDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.95).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert len(records) == 2
    effect = packet.trace["organization"]["effects"][0]
    assert effect["effect_type"] == "append"
    assert effect["record_id"] in {record.record_id for record in records}
    assert effect["top1_similarity"] == 0.0


def test_graph_deduplication_append_organization_uses_runtime_embedding_and_skips_invalid_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphDeduplicationAppendOrganization, PassThroughUnitFormation
    from memprimitive.utils import _runtime

    fake_runtime = _FakeAMEMRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="knowledge_graph",
                    theme="semantic",
                    shape="Graph",
                    indices=("graph", "entity", "vector"),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-missing",
            unit_id="seed-missing",
            layer="knowledge_graph",
            text="Missing embedding",
            timestamp="t0",
            embedding=None,
            metadata={"graph": {"entities": ["none"], "triples": [("None", "is", "missing")], "links": []}},
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-mismatch",
            unit_id="seed-mismatch",
            layer="knowledge_graph",
            text="Wrong dim",
            timestamp="t1",
            embedding=[1.0, 2.0],
            metadata={"graph": {"entities": ["wrong"], "triples": [("Wrong", "dim", "node")], "links": []}},
        )
    )
    target_embedding = fake_runtime.embed("Alice likes tea.")
    store.append(
        MemoryRecord(
            record_id="rec-target",
            unit_id="seed-target",
            layer="knowledge_graph",
            text="Old Alice record",
            timestamp="t2",
            embedding=list(target_embedding),
            metadata={
                "representation": {"text": "Old Alice record", "normalized_text": "old alice record"},
                "graph": {"entities": ["Alice"], "triples": [("Alice", "likes", "tea")], "links": []},
            },
        )
    )

    packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text="Alice likes tea.", source="notes")), store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.99).run(packet, store)

    record = [item for item in store.iter_records("knowledge_graph") if item.record_id == "rec-target"][0]
    assert record.text == "Alice likes tea."
    assert record.embedding == target_embedding
    assert packet.trace["organization"]["effects"][0]["effect_type"] == "merge"
    assert packet.trace["organization"]["effects"][0]["embedding_source"] == "store_policy_fallback"
    assert packet.trace["organization"]["records_with_embedding"] == 1
    assert store.count("knowledge_graph") == 3


def test_graph_deduplication_append_organization_uses_triple_representation_embedding_when_enabled() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphDeduplicationAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class _FakeRuntime:
        def require_llm(self, *, capability: str) -> None:
            return None

        def embed(self, text: str) -> list[float]:
            lowered = text.casefold()
            return [10.0 if "alice" in lowered else 0.0, 5.0 if "tea" in lowered else 0.0, float(len(text))]

        def run_agent(
            self,
            *,
            name: str,
            instructions: str,
            input_text: str,
            temperature: float = 0.0,
            tools: list[Any] | None = None,
            max_turns: int = 10,
            output_type: type[Any] | None = None,
        ) -> Any:
            payload = json.loads(input_text)
            text = payload.get("text", "")
            if "jasmine" in text:
                return {
                    "entities": ["Alice", "jasmine tea"],
                    "relationships": [{"subject": "Alice", "predicate": "likes", "object": "jasmine tea"}],
                }
            return {
                "entities": ["Alice", "tea"],
                "relationships": [{"subject": "Alice", "predicate": "likes", "object": "tea"}],
            }

    store = _graph_store()
    rep = TripleRepresentation(method="direct", embed_extracted=True)
    rep._runtime = lambda: _FakeRuntime()  # type: ignore[method-assign]

    for text in ("Alice likes tea.", "Alice likes jasmine tea."):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = rep.run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
        packet, store = GraphDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.8).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert len(records) == 1
    record = records[0]
    assert record.embedding is not None
    assert record.metadata["representation"]["embedding"]["dim"] == len(record.embedding)
    assert "embedding" not in record.metadata["graph"]
    assert packet.trace["organization"]["effects"][0]["embedding_source"] == "existing_unit_embedding"


def test_graph_deduplication_append_organization_separate_mode_still_writes_source_record_on_merge() -> None:
    from memprimitive.baselines import AlwaysTrigger, GraphDeduplicationAppendOrganization, PassThroughUnitFormation, TripleRepresentation

    class SeededTripleRepresentation(TripleRepresentation):
        _BY_TEXT = {
            "Alice likes tea.": {
                "triples": [("Alice", "likes", "tea")],
                "entities": ["Alice", "tea"],
                "embedding": [1.0, 0.0],
            },
            "Alice likes jasmine tea.": {
                "triples": [("Alice", "likes", "jasmine tea")],
                "entities": ["Alice", "jasmine tea"],
                "embedding": [0.98, 0.02],
            },
        }

        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            payload = self._BY_TEXT[unit.text.strip()]
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=list(payload["entities"]),
                triples=list(payload["triples"]),
                embedding=list(payload["embedding"]),
                representation_elements=("text", "embedding", "triples"),
            )
            return represented, {"source": "test_seed"}

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="source_notes"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )

    for text in ("Alice likes tea.", "Alice likes jasmine tea."):
        packet, store = PassThroughUnitFormation().run(Packet(observation=Observation(text=text, source="notes")), store)
        packet, store = SeededTripleRepresentation().run(packet, store)
        packet, store = AlwaysTrigger().run(packet, store)
        packet, store = GraphDeduplicationAppendOrganization(
            target_layer="knowledge_graph",
            threshold=0.8,
            separate=True,
            separate_layer="source_notes",
        ).run(packet, store)

    assert store.count("source_notes") == 2
    assert store.count("knowledge_graph") == 1
    graph_record = store.iter_records("knowledge_graph")[0]
    source_records = store.iter_records("source_notes")
    assert graph_record.metadata["hierarchical"]["source_record_ids"] == [source_records[-1].record_id]
    assert packet.trace["organization"]["source_written_record_ids"] == [source_records[-1].record_id]
    assert packet.trace["organization"]["effects"][0]["effect_type"] == "merge"


def test_graph_entity_deduplication_append_organization_merges_per_entity_and_appends_unmatched_entities() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        GraphEntityDeduplicationAppendOrganization,
        PassThroughUnitFormation,
        TripleRepresentation,
    )

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=["Alice", "tea"],
                triples=[("Alice", "likes", "tea")],
                representation_elements=("entities", "triple"),
            )
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata.get("representation", {}),
                        "entity_embeddings": {
                            "Alice": [1.0, 0.0],
                            "tea": [0.0, 1.0],
                        },
                    },
                },
            )
            return represented, {"source": "test_seed"}

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-alice",
            unit_id="seed-alice",
            layer="knowledge_graph",
            text="Alice",
            timestamp="t0",
            embedding=[1.0, 0.0],
            metadata={
                "representation": {"text": "Alice", "normalized_text": "alice", "embedding": {"dim": 2}},
                "graph": {"entities": ["Alice"], "triples": [("Alice", "likes", "coffee")], "links": []},
            },
        )
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphEntityDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.8).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert len(records) == 2
    alice_record = [record for record in records if record.record_id == "rec-alice"][0]
    tea_record = [record for record in records if record.record_id != "rec-alice"][0]
    assert alice_record.text == "Alice"
    assert alice_record.embedding == [1.0, 0.0]
    assert alice_record.metadata["graph"]["triples"] == [
        ("Alice", "likes", "coffee"),
        ("Alice", "likes", "tea"),
    ]
    assert tea_record.text == "tea"
    assert tea_record.embedding == [0.0, 1.0]
    effects = packet.trace["organization"]["effects"]
    assert [effect["effect_type"] for effect in effects] == ["merge", "append"]
    assert [effect["entity"] for effect in effects] == ["Alice", "tea"]
    assert packet.trace["organization"]["entity_written_record_ids"] == ["rec-alice", tea_record.record_id]


def test_graph_entity_deduplication_append_organization_appends_when_threshold_not_met() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        GraphEntityDeduplicationAppendOrganization,
        PassThroughUnitFormation,
        TripleRepresentation,
    )

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=["Alice"],
                triples=[("Alice", "likes", "tea")],
                representation_elements=("entities", "triple"),
            )
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata.get("representation", {}),
                        "entity_embeddings": {"Alice": [0.0, 1.0]},
                    },
                },
            )
            return represented, {"source": "test_seed"}

    store = _graph_store()
    store.append(
        MemoryRecord(
            record_id="rec-old-alice",
            unit_id="seed-alice",
            layer="knowledge_graph",
            text="Alice",
            timestamp="t0",
            embedding=[1.0, 0.0],
            metadata={
                "representation": {"text": "Alice", "normalized_text": "alice", "embedding": {"dim": 2}},
                "graph": {"entities": ["Alice"], "triples": [("Alice", "likes", "coffee")], "links": []},
            },
        )
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphEntityDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.95).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert len(records) == 2
    effect = packet.trace["organization"]["effects"][0]
    assert effect["effect_type"] == "append"
    assert effect["entity"] == "Alice"
    assert effect["top1_similarity"] == 0.0


def test_graph_entity_deduplication_append_organization_skips_entities_without_entity_embedding() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        GraphEntityDeduplicationAppendOrganization,
        PassThroughUnitFormation,
        TripleRepresentation,
    )

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=["Alice", "tea"],
                triples=[("Alice", "likes", "tea")],
                representation_elements=("entities", "triple"),
            )
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata.get("representation", {}),
                        "entity_embeddings": {"Alice": [1.0, 0.0]},
                    },
                },
            )
            return represented, {"source": "test_seed"}

    store = _graph_store()
    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphEntityDeduplicationAppendOrganization(target_layer="knowledge_graph", threshold=0.8).run(packet, store)

    records = store.iter_records("knowledge_graph")
    assert [record.text for record in records] == ["Alice"]
    assert packet.trace["organization"]["skipped_entity_count"] == 1
    assert packet.trace["organization"]["effects"][1]["effect_type"] == "skipped_entity_missing_embedding"
    assert packet.trace["organization"]["effects"][1]["entity"] == "tea"


def test_graph_entity_deduplication_append_organization_separate_mode_writes_source_and_entity_records() -> None:
    from memprimitive.baselines import (
        AlwaysTrigger,
        GraphEntityDeduplicationAppendOrganization,
        PassThroughUnitFormation,
        TripleRepresentation,
    )

    class SeededTripleRepresentation(TripleRepresentation):
        def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
            represented = replace(
                unit,
                normalized_text=unit.text.strip().casefold(),
                entities=["Alice", "tea"],
                triples=[("Alice", "likes", "tea")],
                representation_elements=("entities", "triple"),
            )
            represented = replace(
                represented,
                metadata={
                    **represented.metadata,
                    "representation": {
                        **represented.metadata.get("representation", {}),
                        "entity_embeddings": {
                            "Alice": [1.0, 0.0],
                            "tea": [0.0, 1.0],
                        },
                    },
                },
            )
            return represented, {"source": "test_seed"}

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="source_notes"),
                StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-alice",
            unit_id="seed-alice",
            layer="knowledge_graph",
            text="Alice",
            timestamp="t0",
            embedding=[1.0, 0.0],
            metadata={
                "representation": {"text": "Alice", "normalized_text": "alice", "embedding": {"dim": 2}},
                "graph": {"entities": ["Alice"], "triples": [("Alice", "likes", "coffee")], "links": []},
            },
        )
    )

    packet, store = PassThroughUnitFormation().run(
        Packet(observation=Observation(text="Alice likes tea.", source="notes")),
        store,
    )
    packet, store = SeededTripleRepresentation().run(packet, store)
    packet, store = AlwaysTrigger().run(packet, store)
    packet, store = GraphEntityDeduplicationAppendOrganization(
        target_layer="knowledge_graph",
        threshold=0.8,
        separate=True,
        separate_layer="source_notes",
    ).run(packet, store)

    source_records = store.iter_records("source_notes")
    graph_records = store.iter_records("knowledge_graph")
    assert len(source_records) == 1
    assert len(graph_records) == 2
    source_record = source_records[0]
    alice_record = [record for record in graph_records if record.record_id == "rec-alice"][0]
    tea_record = [record for record in graph_records if record.record_id != "rec-alice"][0]
    assert source_record.text == "Alice likes tea."
    assert alice_record.metadata["hierarchical"]["source_record_ids"] == [source_record.record_id]
    assert tea_record.metadata["hierarchical"]["source_record_ids"] == [source_record.record_id]
    assert packet.trace["organization"]["source_written_record_ids"] == [source_record.record_id]
    assert packet.trace["organization"]["effects"][0]["effect_type"] == "merge"
    assert packet.trace["organization"]["effects"][1]["effect_type"] == "append"


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


def test_graph_link_evolution_only_modifies_graph_layer() -> None:
    from memprimitive.baselines import GraphLinkEvolution

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

    packet_out, store = GraphLinkEvolution(target_layer="knowledge_graph", neighbor_limit=1).run(packet, store)

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


def test_graph_relation_readout_renders_relation_sentences_in_retrieval_order() -> None:
    from memprimitive.baselines import GraphRelationReadout

    records = [
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes sushi.",
            timestamp="2026-03-27T00:00:00+00:00",
            metadata={"graph": {"links": ["rec-2"], "triples": [("Alice", "likes", "sushi")]}},
        ),
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Bob builds tools.",
            timestamp="2026-03-27T00:01:00+00:00",
            metadata={"graph": {"links": ["rec-1"], "triples": [("Bob", "builds", "tools")]}},
        ),
    ]

    packet_out, _ = GraphRelationReadout().run(Packet(retrieved=RetrievedSet(items=records, scores=[])), _graph_store())

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Alice likes sushi\nBob builds tools"
    assert packet_out.readout.source_ids == ["rec-1", "rec-2"]
    assert packet_out.readout.metadata["relation_sentence_count"] == 2
    assert packet_out.readout.metadata["linked_item_count"] == 2
    assert packet_out.readout.metadata["fallback_used"] is False


def test_graph_relation_readout_dedupes_repeated_triples_and_ignores_unlinked_records() -> None:
    from memprimitive.baselines import GraphRelationReadout

    records = [
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes sushi.",
            timestamp="2026-03-27T00:00:00+00:00",
            metadata={"graph": {"links": ["rec-2"], "triples": [("Alice", "likes", "sushi")]}},
        ),
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Duplicate relation.",
            timestamp="2026-03-27T00:01:00+00:00",
            metadata={"graph": {"links": ["rec-1"], "triples": [("Alice", "likes", "sushi")]}},
        ),
        MemoryRecord(
            record_id="rec-3",
            unit_id="unit-3",
            layer="knowledge_graph",
            text="Carol mentors Dana.",
            timestamp="2026-03-27T00:02:00+00:00",
            metadata={"graph": {"links": [], "triples": [("Carol", "mentors", "Dana")]}},
        ),
    ]

    packet_out, _ = GraphRelationReadout().run(Packet(retrieved=RetrievedSet(items=records, scores=[])), _graph_store())

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Alice likes sushi"
    assert packet_out.readout.source_ids == ["rec-1", "rec-2", "rec-3"]
    assert packet_out.readout.metadata["relation_sentence_count"] == 1
    assert packet_out.readout.metadata["linked_item_count"] == 2


def test_graph_relation_readout_falls_back_to_original_text_when_no_relation_sentence_exists() -> None:
    from memprimitive.baselines import GraphRelationReadout

    records = [
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes sushi.",
            timestamp="2026-03-27T00:00:00+00:00",
            metadata={"graph": {"links": ["rec-2"], "triples": []}},
        ),
        MemoryRecord(
            record_id="rec-2",
            unit_id="unit-2",
            layer="knowledge_graph",
            text="Carol mentors Dana.",
            timestamp="2026-03-27T00:01:00+00:00",
            metadata={"graph": {"links": [], "triples": [("Carol", "mentors", "Dana")]}},
        ),
    ]

    packet_out, _ = GraphRelationReadout().run(Packet(retrieved=RetrievedSet(items=records, scores=[])), _graph_store())

    assert packet_out.readout is not None
    assert packet_out.readout.text == "Alice likes sushi.\nCarol mentors Dana."
    assert packet_out.readout.source_ids == ["rec-1", "rec-2"]
    assert packet_out.readout.metadata["relation_sentence_count"] == 0
    assert packet_out.readout.metadata["linked_item_count"] == 1
    assert packet_out.readout.metadata["fallback_used"] is True
    assert packet_out.trace["readout"]["fallback_used"] is True


def test_graph_entity_retrieval_pipeline_end_to_end_supports_threshold_trigger_evolution_retrieval_and_readout() -> None:
    from memprimitive import MemoryPipeline
    from memprimitive.baselines import (
        BasicRepresentation,
        EntityRetrieval,
        ExpandRetrievedGraphNeighbors,
        GraphAppendOrganization,
        GraphLinkEvolution,
        GraphNeighborContextTraceEvolution,
        GraphReadout,
        LLMRepresentation,
        PassThroughUnitFormation,
        AlwaysTrigger,
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
        evolution_trigger=AlwaysTrigger(slot="evolution_trigger"),
        memory_evolution=(
            GraphLinkEvolution(target_layer="knowledge_graph", neighbor_limit=2, rewrite_neighbor_metadata=True),
            GraphNeighborContextTraceEvolution(target_layer="knowledge_graph", rewrite_metadata=True),
        ),
        retrieval=(
            EntityRetrieval(top_k=1, layer="knowledge_graph"),
            ExpandRetrievedGraphNeighbors(top_k=4, layer="knowledge_graph"),
        ),
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
