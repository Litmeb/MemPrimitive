from __future__ import annotations

import pytest

from memprimitive import Observation, Packet, Query
from memprimitive.classic_modules.amem import AMEMConfig, AMEM_GRAPH_LAYER, build_amem_pipeline


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _amem_pipeline() -> object:
    return build_amem_pipeline(
        config=AMEMConfig(
            top_k=3,
            max_hops=2,
            seed_k=1,
            max_links_per_record=2,
            link_threshold=1.0,
        )
    )


def test_amem_ingest_updates_graph_links_between_related_notes() -> None:
    pipeline = _amem_pipeline()

    first_packet = pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    second_packet = pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    third_packet = pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))

    assert pipeline.store.topology.layer_names == (AMEM_GRAPH_LAYER,)
    records = pipeline.store.iter_records(AMEM_GRAPH_LAYER)
    assert len(records) == 3

    first_record_id = first_packet.trace["organization"]["effects"][0]["record_id"]
    second_record_id = second_packet.trace["organization"]["effects"][0]["record_id"]
    third_record_id = third_packet.trace["organization"]["effects"][0]["record_id"]

    first_record = next(record for record in records if record.record_id == first_record_id)
    second_record = next(record for record in records if record.record_id == second_record_id)
    third_record = next(record for record in records if record.record_id == third_record_id)

    assert second_record_id in first_record.metadata["graph"]["links"]
    assert first_record_id in second_record.metadata["graph"]["links"]
    assert third_record_id in second_record.metadata["graph"]["links"]
    assert second_record_id in third_record.metadata["graph"]["links"]
    assert second_packet.trace["memory_evolution"]["effects"][0]["linked_record_ids"] == [first_record_id]
    assert third_packet.trace["memory_evolution"]["effects"][0]["linked_record_ids"] == [second_record_id]


def test_amem_graph_hop_retrieval_walks_two_hops_from_an_anchor() -> None:
    pipeline = _amem_pipeline()

    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))

    packet, _ = pipeline.retrieval.run(Packet(query=Query(text="Alice")), pipeline.store)

    assert packet.retrieved is not None
    assert [score["hop"] for score in packet.retrieved.scores] == [0, 1, 2]
    assert packet.retrieved.trace["seed_record_ids"]
    assert packet.retrieved.items[0].text == "Alice likes tea."
    assert packet.retrieved.items[1].text == "Tea routines improve focus."
    assert packet.retrieved.items[2].text == "Focus helps graph memory systems."


def test_amem_graph_readout_groups_results_by_hop_distance() -> None:
    pipeline = _amem_pipeline()

    pipeline.ingest(Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(Observation(text="Tea routines improve focus.", source="dialogue"))
    pipeline.ingest(Observation(text="Focus helps graph memory systems.", source="dialogue"))

    readout = pipeline.recall(Query(text="Alice"))

    assert readout.metadata["format"] == "graph_hop"
    assert readout.metadata["item_count"] == 3
    assert readout.metadata["hop_counts"] == {0: 1, 1: 1, 2: 1}
    assert readout.source_ids
    assert readout.text.startswith("Query: Alice")
    assert "[hop 0] direct matches" in readout.text
    assert "[hop 1] 1-hop neighbors" in readout.text
    assert "[hop 2] 2-hop neighbors" in readout.text
