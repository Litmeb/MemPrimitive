from memprimitive.core import MemoryStore, ModuleSpec, Observation, Query


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
    assert store.count() == 0
    assert store.count("default") == 0
    assert store.is_empty() is True


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
