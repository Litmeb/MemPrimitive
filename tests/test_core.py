import pytest

from memprimitive import IncompatibleCompositionError
from memprimitive.contracts import (
    RECORD_GRAPH_LINKS_CONTRACT,
    RECORD_NOTE_PAYLOAD_CONTRACT,
    TOPOLOGY_GRAPH_LAYER_CONTRACT,
    TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT,
    TOPOLOGY_TAG_INDEX_CONTRACT,
    TOPOLOGY_VECTOR_INDEX_CONTRACT,
    UNIT_EMBEDDING_CONTRACT,
    UNIT_NOTE_PAYLOAD_CONTRACT,
    UNIT_TAGS_CONTRACT,
)
from memprimitive.core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.utils import _runtime as runtime_module


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        normalized = str(text).strip()
        self.calls.append(normalized)
        return [float(len(normalized)), float(sum(ord(ch) for ch in normalized) % 97)]


def _fake_embedding(text: str) -> list[float]:
    normalized = str(text).strip()
    return [float(len(normalized)), float(sum(ord(ch) for ch in normalized) % 97)]


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


def test_query_embedding_defaults_to_none() -> None:
    query = Query(text="alice")

    assert query.embedding is None


def test_query_embedding_is_normalized_to_float_list() -> None:
    query = Query(text="alice", embedding=[1, "2.5", 3.0])

    assert query.embedding == [1.0, 2.5, 3.0]


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


def test_memory_record_from_unit_carries_embedding_vector_and_representation_dim_summary() -> None:
    unit = MemoryUnit(
        text="Alice likes tea.",
        representation_elements=("text", "embedding", "entities"),
        normalized_text="alice likes tea.",
        embedding=[0.1, 0.2, 0.3],
        entities=["Alice"],
    )

    record = MemoryRecord.from_unit(unit=unit, layer="default", sequence_id=1)

    assert record.embedding == [0.1, 0.2, 0.3]
    assert record.metadata["representation"]["elements"] == ["text", "embedding", "entities"]
    assert record.metadata["representation"]["normalized_text"] == "alice likes tea."
    assert record.metadata["representation"]["entities"] == ["Alice"]
    assert record.metadata["representation"]["embedding"] == {"dim": 3}


def test_memory_store_append_auto_embeds_record_when_layer_policy_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="semantic",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )

    record = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="semantic",
        text="Alice likes tea",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    store.append(record)

    stored = store.iter_records("semantic")[0]
    assert stored.embedding == _fake_embedding("Alice likes tea")
    assert fake_runtime.calls == ["Alice likes tea"]


def test_memory_store_append_skips_auto_embedding_when_layer_policy_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default")]))

    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="default",
            text="plain note",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    assert store.iter_records("default")[0].embedding is None
    assert fake_runtime.calls == []


def test_memory_store_replace_record_refreshes_embedding_when_text_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="semantic",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="semantic",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    updated = MemoryRecord(
        record_id="rec-1",
        unit_id="unit-1",
        layer="semantic",
        text="Alice likes jasmine tea",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"changed": True},
    )
    store.replace_record("semantic", "rec-1", updated)

    stored = store.iter_records("semantic")[0]
    assert stored.embedding == _fake_embedding("Alice likes jasmine tea")
    assert fake_runtime.calls == ["Alice likes tea", "Alice likes jasmine tea"]


def test_memory_store_replace_record_keeps_embedding_when_only_metadata_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="semantic",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="semantic",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    original_embedding = list(store.iter_records("semantic")[0].embedding)

    store.replace_record(
        "semantic",
        "rec-1",
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="semantic",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"owner": "updated"},
        ),
    )

    stored = store.iter_records("semantic")[0]
    assert stored.embedding == original_embedding
    assert stored.metadata["owner"] == "updated"
    assert fake_runtime.calls == ["Alice likes tea"]


def test_memory_store_graph_link_updates_do_not_refresh_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="knowledge_graph",
                    theme="semantic",
                    shape="Graph",
                    indices=("graph", "vector"),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="knowledge_graph",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"graph": {"entities": ["Alice"], "links": []}},
        )
    )
    original_embedding = list(store.iter_records("knowledge_graph")[0].embedding)

    merged_links = store.add_graph_links("knowledge_graph", "rec-1", ["rec-2"])

    stored = store.iter_records("knowledge_graph")[0]
    assert merged_links == ["rec-2"]
    assert stored.embedding == original_embedding
    assert stored.metadata["graph"]["links"] == ["rec-2"]
    assert fake_runtime.calls == ["Alice likes tea"]


def test_memory_store_replace_record_preserves_explicit_embedding_over_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: fake_runtime)
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="semantic",
                    theme="semantic",
                    indices=("vector",),
                    settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
                )
            ]
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="semantic",
            text="Alice likes tea",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    store.replace_record(
        "semantic",
        "rec-1",
        MemoryRecord(
            record_id="rec-1",
            unit_id="unit-1",
            layer="semantic",
            text="Alice likes tea very much",
            timestamp="2026-01-01T00:00:00+00:00",
            embedding=[9.0, 9.0],
        ),
    )

    stored = store.iter_records("semantic")[0]
    assert stored.embedding == [9.0, 9.0]
    assert fake_runtime.calls == ["Alice likes tea"]


def test_memory_store_check_passes_without_registered_modules() -> None:
    store = MemoryStore()

    missing = store.check()

    assert missing == frozenset()
    assert store.required_contracts == frozenset()
    assert TOPOLOGY_GRAPH_LAYER_CONTRACT not in store.produced_contracts


def test_memory_store_check_reports_missing_contracts() -> None:
    store = MemoryStore()
    store.register_module_contracts(
        slot="retrieval",
        module_name="graph_neighbor_retrieval",
        requires_contracts=("record.graph_links", "topology.graph_layer"),
    )

    with pytest.raises(IncompatibleCompositionError, match="record.graph_links"):
        store.check()


def test_memory_store_topology_contracts_surface_declared_capabilities() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default", indices=("keyword",)),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "vector", "tag")),
            ]
        )
    )

    assert TOPOLOGY_GRAPH_LAYER_CONTRACT in store.topology_contracts
    assert TOPOLOGY_VECTOR_INDEX_CONTRACT in store.topology_contracts
    assert TOPOLOGY_TAG_INDEX_CONTRACT in store.topology_contracts
    assert TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT in store.topology_contracts


@pytest.mark.parametrize(
    ("requires_contracts", "produces_contracts", "expected_missing"),
    (
        pytest.param(
            (UNIT_EMBEDDING_CONTRACT,),
            (),
            {UNIT_EMBEDDING_CONTRACT},
            id="missing-unit-embedding",
        ),
        pytest.param(
            (UNIT_TAGS_CONTRACT, TOPOLOGY_TAG_INDEX_CONTRACT),
            (),
            {UNIT_TAGS_CONTRACT, TOPOLOGY_TAG_INDEX_CONTRACT},
            id="missing-tags-and-tag-index",
        ),
        pytest.param(
            (UNIT_NOTE_PAYLOAD_CONTRACT, TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT),
            (),
            {UNIT_NOTE_PAYLOAD_CONTRACT, TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT},
            id="missing-note-payload-and-graph-vector-layer",
        ),
        pytest.param(
            (RECORD_NOTE_PAYLOAD_CONTRACT, RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT),
            (RECORD_NOTE_PAYLOAD_CONTRACT,),
            {RECORD_GRAPH_LINKS_CONTRACT, TOPOLOGY_GRAPH_LAYER_CONTRACT},
            id="partial-production-still-missing-linked-graph-contracts",
        ),
    ),
)
def test_memory_store_check_rejects_batched_invalid_contract_sets(
    requires_contracts: tuple[str, ...],
    produces_contracts: tuple[str, ...],
    expected_missing: set[str],
) -> None:
    store = MemoryStore()
    store.register_module_contracts(
        slot="demo",
        module_name="invalid_demo_module",
        requires_contracts=requires_contracts,
        produces_contracts=produces_contracts,
    )

    with pytest.raises(IncompatibleCompositionError) as excinfo:
        store.check()

    contracts_meta = store.metadata["composition_contracts"]
    assert store.required_contracts == frozenset(requires_contracts)
    assert frozenset(contracts_meta["missing"]) == frozenset(expected_missing)
    for contract in sorted(expected_missing):
        assert contract in str(excinfo.value)


def test_memory_store_check_accepts_contracts_jointly_satisfied_by_modules_and_topology() -> None:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="knowledge_graph", shape="Graph", indices=("graph", "vector")),
            ]
        )
    )
    store.register_module_contracts(
        slot="representation",
        module_name="semantic_field_enrichment_representation",
        produces_contracts=(UNIT_NOTE_PAYLOAD_CONTRACT,),
    )
    store.register_module_contracts(
        slot="organization",
        module_name="graph_append_organization",
        requires_contracts=(TOPOLOGY_GRAPH_LAYER_CONTRACT,),
        produces_contracts=(RECORD_NOTE_PAYLOAD_CONTRACT, RECORD_GRAPH_LINKS_CONTRACT),
    )
    store.register_module_contracts(
        slot="retrieval",
        module_name="vector_graph_seed_and_expand_retrieval",
        requires_contracts=(RECORD_NOTE_PAYLOAD_CONTRACT, TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT),
    )

    assert store.check() == frozenset()
    assert store.metadata["composition_contracts"]["missing"] == []
