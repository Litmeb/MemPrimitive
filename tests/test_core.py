import pytest

from memprimitive.core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Observation, Query, StoreLayerSpec, StoreTopology


def test_observation_rejects_empty_text() -> None:
    try:
        Observation(text="   ")
    except ValueError as exc:
        assert "Observation.text" in str(exc)
    else:
        raise AssertionError("Observation should reject empty text.")


def test_query_rejects_empty_text() -> None:
    try:
        Query(text="")
    except ValueError as exc:
        assert "Query.text" in str(exc)
    else:
        raise AssertionError("Query should reject empty text.")


def test_memory_store_starts_with_empty_default_layer() -> None:
    store = MemoryStore()

    assert "default" in store.layers
    assert store.topology.layer_count == 1
    assert store.topology.get_layer("default") == StoreLayerSpec(name="default", theme="working")
    assert store.count() == 0
    assert store.count("default") == 0
    assert store.is_empty() is True


def test_store_topology_from_layers_supports_multi_layer_capabilities() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
            StoreLayerSpec(name="knowledge_graph", theme="semantic", shape="Graph", indices=("graph", "entity")),
        ]
    )

    assert topology.layer_count == 2
    assert topology.layer_names == ("working", "knowledge_graph")
    assert topology.layer_shape("knowledge_graph") == "Graph"
    assert topology.layer_supports_index("working", "keyword") is True
    assert topology.layer_supports_index("knowledge_graph", "graph") is True
    assert topology.has_graph_layer() is True
    assert topology.has_keyword_layer() is True
    assert topology.has_vector_layer() is False


def test_store_layer_spec_rejects_invalid_shape_indices_and_capacity() -> None:
    with pytest.raises(ValueError, match="StoreLayerSpec.shape"):
        StoreLayerSpec(name="bad", shape="Tree")

    with pytest.raises(ValueError, match="StoreLayerSpec.indices"):
        StoreLayerSpec(name="bad", indices=("vector", "bogus"))

    with pytest.raises(ValueError, match="StoreLayerSpec.capacity"):
        StoreLayerSpec(name="bad", capacity="limited")


def test_store_layer_spec_accepts_custom_theme_but_rejects_empty_theme() -> None:
    spec = StoreLayerSpec(name="customized", theme="domain_memory")

    assert spec.theme == "domain_memory"

    with pytest.raises(ValueError, match="StoreLayerSpec.theme"):
        StoreLayerSpec(name="invalid", theme="  ")


def test_memory_store_initializes_declared_topology_layers() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic", theme="story", indices=("temporal",)),
        ]
    )

    store = MemoryStore(topology=topology)

    assert store.layers == {"working": [], "episodic": []}
    assert store.has_layer("episodic") is True
    assert store.layer_shape("working") == "Flat"
    assert store.layer_supports_index("episodic", "temporal") is True


def test_memory_store_iter_records_preserves_topology_order_across_layers() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="working"),
            StoreLayerSpec(name="episodic"),
        ]
    )
    store = MemoryStore(topology=topology)
    working_record = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="working",
        text="working memory",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    episodic_record = MemoryRecord(
        record_id="rec-2",
        unit_id="unit-2",
        layer="episodic",
        text="episodic memory",
        timestamp="2026-01-01T00:00:01+00:00",
    )

    store.append(working_record)
    store.append(episodic_record)

    assert [record.record_id for record in store.iter_records()] == ["rec-1", "rec-2"]
    assert [record.record_id for record in store.iter_records("episodic")] == ["rec-2"]


def test_memory_store_ensure_layer_is_strict_by_default() -> None:
    store = MemoryStore()

    with pytest.raises(ValueError, match="not declared in the store topology"):
        store.ensure_layer("new_layer")


def test_memory_store_ensure_layer_can_extend_topology_with_requested_theme() -> None:
    store = MemoryStore()

    store.ensure_layer("new_layer", allow_create=True, theme="long_term")

    assert store.has_layer("new_layer") is True
    assert store.topology.get_layer("new_layer") == StoreLayerSpec(name="new_layer", theme="long_term")
    assert store.layers["new_layer"] == []


def test_memory_store_accepts_extra_layer_data_when_topology_extension_enabled() -> None:
    record = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="extra",
        text="preloaded",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    store = MemoryStore(layers={"default": [], "extra": [record]}, allow_topology_extend=True)

    assert store.has_layer("extra") is True
    assert store.topology.get_layer("extra") == StoreLayerSpec(name="extra")
    assert store.count("extra") == 1


def test_module_spec_preserves_declared_contract_fields() -> None:
    spec = ModuleSpec(
        name="demo",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items",),
        store_requirements=("vector",),
        layer_requirements=("shape:Flat",),
        side_effects=("read_only",),
    )

    assert spec.name == "demo"
    assert spec.slot == "retrieval"
    assert spec.input_requirements == ("query.text",)
    assert spec.output_guarantees == ("retrieved.items",)
    assert spec.store_requirements == ("vector",)
    assert spec.layer_requirements == ("shape:Flat",)
    assert spec.side_effects == ("read_only",)


def test_memory_unit_representation_fields_have_stable_defaults() -> None:
    unit = MemoryUnit(text="Alice likes tea.")

    assert unit.representation_elements == ()
    assert unit.normalized_text is None
    assert unit.embedding is None
    assert unit.triples == []
    assert unit.kv == {}
    assert unit.entities == []
    assert unit.tags == []
    assert unit.description is None


def test_memory_record_from_unit_carries_representation_summary_without_raw_embedding_duplication() -> None:
    unit = MemoryUnit(
        text="Alice likes tea.",
        representation_elements=("text", "embedding", "entities"),
        normalized_text="alice likes tea.",
        embedding=[0.1, 0.2, 0.3],
        entities=["Alice"],
    )

    record = MemoryRecord.from_unit(unit=unit, layer="default", sequence_id=1)

    assert record.metadata["representation"]["elements"] == ["text", "embedding", "entities"]
    assert record.metadata["representation"]["normalized_text"] == "alice likes tea."
    assert record.metadata["representation"]["entities"] == ["Alice"]
    assert record.metadata["representation"]["embedding"] == {"dim": 3}
