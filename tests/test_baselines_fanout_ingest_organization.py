from __future__ import annotations

import pytest

from memprimitive import FanoutIngestOrganization as TopLevelFanoutIngestOrganization
from memprimitive.baselines import AppendOrganization, BasicRepresentation, FanoutIngestOrganization, PassThroughUnitFormation
from memprimitive.baselines.registry import registered_baseline_class_names
from memprimitive.core import MemoryStore, Observation, Packet
from memprimitive.pipeline import MemoryPipeline


def _child_pipeline(store: MemoryStore | None = None) -> MemoryPipeline:
    return MemoryPipeline(
        store=store,
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer="default"),
    )


def test_fanout_ingest_organization_ingests_each_valid_string_in_order() -> None:
    store = MemoryStore()
    module = FanoutIngestOrganization(field="segments", pipeline=_child_pipeline())
    packet = Packet(
        observation=Observation(
            text="parent",
            source="notes",
            metadata={"segments": ["alpha", " beta "]},
        )
    )

    packet_out, updated_store = module.run(packet, store)

    assert [record.text for record in updated_store.iter_records("default")] == ["alpha", "beta"]
    assert packet_out.placements is None
    assert packet_out.trace["organization"]["fanout_count"] == 2
    assert packet_out.trace["organization"]["skipped_member_count"] == 0
    assert [entry["index"] for entry in packet_out.trace["organization"]["child_traces"]] == [0, 1]
    assert [entry["organization_trace"]["target_layer"] for entry in packet_out.trace["organization"]["child_traces"]] == [
        "default",
        "default",
    ]


def test_fanout_ingest_organization_skips_non_string_and_empty_members() -> None:
    store = MemoryStore()
    module = FanoutIngestOrganization(field="segments", pipeline=_child_pipeline())
    packet = Packet(
        observation=Observation(
            text="parent",
            source="notes",
            metadata={"segments": ["alpha", "", "   ", 3, None, "beta"]},
        )
    )

    packet_out, updated_store = module.run(packet, store)

    assert [record.text for record in updated_store.iter_records("default")] == ["alpha", "beta"]
    assert packet_out.trace["organization"]["fanout_count"] == 2
    assert packet_out.trace["organization"]["skipped_member_count"] == 4
    assert packet_out.trace["organization"]["skipped_reasons"] == {"non_string": 2, "empty_string": 2}


def test_fanout_ingest_organization_requires_metadata_field() -> None:
    module = FanoutIngestOrganization(field="segments", pipeline=_child_pipeline())
    packet = Packet(observation=Observation(text="parent", source="notes", metadata={}))

    with pytest.raises(ValueError, match="requires observation.metadata"):
        module.run(packet, MemoryStore())


def test_fanout_ingest_organization_rejects_non_iterable_metadata_value() -> None:
    module = FanoutIngestOrganization(field="segments", pipeline=_child_pipeline())
    packet = Packet(observation=Observation(text="parent", source="notes", metadata={"segments": 3}))

    with pytest.raises(ValueError, match="iterable of strings"):
        module.run(packet, MemoryStore())


def test_fanout_ingest_organization_rejects_string_metadata_value() -> None:
    module = FanoutIngestOrganization(field="segments", pipeline=_child_pipeline())
    packet = Packet(observation=Observation(text="parent", source="notes", metadata={"segments": "alpha"}))

    with pytest.raises(ValueError, match="iterable of strings"):
        module.run(packet, MemoryStore())


def test_fanout_ingest_organization_writes_into_shared_store_and_updates_child_pipeline_store() -> None:
    parent_store = MemoryStore()
    child_pipeline = _child_pipeline(store=MemoryStore())
    module = FanoutIngestOrganization(field="segments", pipeline=child_pipeline)
    packet = Packet(observation=Observation(text="parent", source="notes", metadata={"segments": ["alpha"]}))

    _, updated_store = module.run(packet, parent_store)

    assert updated_store is parent_store
    assert child_pipeline.store is parent_store
    assert parent_store.count("default") == 1


def test_fanout_ingest_organization_is_registered_and_exported() -> None:
    assert "FanoutIngestOrganization" in registered_baseline_class_names()
    assert FanoutIngestOrganization is TopLevelFanoutIngestOrganization


def test_memory_pipeline_accepts_fanout_ingest_organization() -> None:
    child_pipeline = _child_pipeline()
    pipeline = MemoryPipeline(organization=FanoutIngestOrganization(field="segments", pipeline=child_pipeline))

    assert isinstance(pipeline.organization, FanoutIngestOrganization)
