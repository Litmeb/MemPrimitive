from __future__ import annotations

import pytest

from memprimitive import Observation, Query, create_baseline_pipeline
from memprimitive.baselines.registry import (
    instantiate_default_baseline_modules,
    iter_baseline_pipeline_instances,
)
from memprimitive.core import ModuleSpec, Packet
from memprimitive.interfaces import RetrievalModule
from memprimitive.pipeline import MemoryPipeline


def test_ingesting_observations_then_recalling_query_produces_non_empty_readout() -> None:
    pipeline = create_baseline_pipeline(top_k=2)
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob likes coffee.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert readout.text
    assert len(readout.source_ids) == 1


def test_full_baseline_pipeline_preserves_trace_fields_across_ingest_stages() -> None:
    pipeline = create_baseline_pipeline(top_k=1)

    packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))

    for slot in MemoryPipeline.INGEST_SLOTS:
        assert slot in packet.trace, f"missing trace key for {slot}"


def test_repeated_ingests_accumulate_records_in_store() -> None:
    pipeline = create_baseline_pipeline(top_k=2)
    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Bob likes coffee.", source="dialogue"))
    pipeline.ingest(Observation(text="Charlie likes cocoa.", source="dialogue"))

    assert pipeline.store.count() == 3


def test_round_trip_demo_scenario_works_with_baseline_pipeline() -> None:
    pipeline = create_baseline_pipeline(top_k=2)

    readout = pipeline.run_round(
        Observation(text="Alice started learning graph memory systems.", source="notes"),
        Query(text="Alice"),
    )

    assert "Alice" in readout.text
    assert readout.metadata["item_count"] >= 1
    assert "ingest_trace" in readout.metadata


def test_memory_pipeline_rejects_wrong_abstract_type_at_slot() -> None:
    """Composition rules: each kwarg must match the expected primitive ABC."""
    m = instantiate_default_baseline_modules(top_k=2)
    with pytest.raises(TypeError, match="readout"):
        MemoryPipeline(
            unit_formation=m["unit_formation"],
            representation=m["representation"],
            write_trigger=m["write_trigger"],
            organization=m["organization"],
            memory_evolution=m["memory_evolution"],
            retrieval=m["retrieval"],
            readout=m["retrieval"],
        )


def test_memory_pipeline_rejects_wrong_module_spec_slot() -> None:
    """Even with the correct ABC, ModuleSpec.slot must match the pipeline position."""

    class MislabeledRetrieval(RetrievalModule):
        spec = ModuleSpec(name="mislabeled", slot="readout")

        def run(self, packet: Packet, store):
            return packet, store

    m = instantiate_default_baseline_modules(top_k=2)
    with pytest.raises(ValueError, match=r"expects ModuleSpec\.slot='retrieval'"):
        MemoryPipeline(
            unit_formation=m["unit_formation"],
            representation=m["representation"],
            write_trigger=m["write_trigger"],
            organization=m["organization"],
            memory_evolution=m["memory_evolution"],
            retrieval=MislabeledRetrieval(),
            readout=m["readout"],
        )


def test_every_registered_baseline_combination_runs_ingest_and_recall() -> None:
    """Cartesian product over :func:`~memprimitive.baselines.registry.baseline_factory_groups`.

    Any combination accepted by :class:`~memprimitive.pipeline.MemoryPipeline` must run
    without error; extend the factory tuples when adding alternative baselines.
    """
    for pipeline in iter_baseline_pipeline_instances(top_k=2):
        pipeline.ingest(Observation(text="combinatorial ingest.", source="dialogue"))
        readout = pipeline.recall(Query(text="combinatorial"))
        assert isinstance(readout.text, str)
        assert pipeline.store.count() == 1
