from __future__ import annotations

import pytest

from memprimitive.core import (
    MemoryStore,
    Packet,
    StoreLayerSpec,
    StoreTopology,
)

from baselines_test_helpers import (
    _FakeHierarchicalRuntime,
    _represented_packet,
    _seed_layer,
    _seed_layer_with_metadata,
    _stored_pipeline_packet,
)


def test_summary_rewrite_evolution_appends_summary_record() -> None:
    from memprimitive.baselines import SummaryRewriteEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    packet, store = _stored_pipeline_packet("Alice likes jasmine tea.", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = SummaryRewriteEvolution(target_layer="semantic").run(packet, store)

    assert store.count("semantic") == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["effect_type"] == "summary_append"


def test_layer_move_evolution_copy_appends_unit_to_target_layer() -> None:
    from memprimitive.baselines import LayerMoveEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    packet, store = _stored_pipeline_packet("Alice likes jasmine tea.", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = LayerMoveEvolution(target_layer="semantic").run(packet, store)

    assert store.count("semantic") == 1
    assert packet_out.trace["memory_evolution"]["effects"][0]["move_style"] == "copy_append"


def test_hierarchical_organization_copy_uses_decisions_store_selection() -> None:
    from memprimitive.baselines import HierarchicalOrganization

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "doc-a note", "metadata": {"session_id": "sess-1", "doc_id": "doc-a", "subgoal_id": "sg-1"}},
            {"text": "doc-b note", "metadata": {"session_id": "sess-2", "doc_id": "doc-b", "subgoal_id": "sg-2"}},
        ],
    )
    packet, _ = _represented_packet("incoming note")
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {
                "decision": True,
                "record_ids": ["rec-1"],
                "selector": {"kind": "boundary_match"},
            }
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id", "subgoal_id"),
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 1
    assert packet_out.placements is not None
    assert packet_out.placements[0].target_layer == "semantic"
    assert packet_out.trace["organization"]["selection_source"] == "decisions_store"
    assert packet_out.trace["organization"]["selected_record_count"] == 1
    assert packet_out.trace["organization"]["append_current_units"] is False
    assert packet_out.trace["organization"]["write_mode"] == "memory_pipeline_ingest"
    assert packet_out.trace["organization"]["writer_pipeline_mode"] == "default_target_layer"
    assert packet_out.trace["organization"]["written_record_ids"] == ["rec-3"]
    assert packet_out.trace["organization"]["sub_ingest_trace"][0]["organization"]["target_layer"] == "semantic"
    assert written[0].metadata["hierarchical"]["source_record_ids"] == ["rec-1"]
    assert written[0].metadata["hierarchical"]["field_payload"]["doc_id"] == "doc-a"


def test_hierarchical_evolution_copy_uses_evolution_decisions_store_selection() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "doc-a note", "metadata": {"session_id": "sess-1", "doc_id": "doc-a"}},
            {"text": "doc-b note", "metadata": {"session_id": "sess-2", "doc_id": "doc-b"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {
                "decision": True,
                "record_ids": ["rec-2"],
                "selector": {"kind": "boundary_match"},
            }
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 1
    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions_store"
    assert packet_out.trace["memory_evolution"]["selected_record_count"] == 1
    assert packet_out.trace["memory_evolution"]["write_mode"] == "memory_pipeline_ingest"
    assert packet_out.trace["memory_evolution"]["writer_pipeline_mode"] == "default_target_layer"
    assert packet_out.trace["memory_evolution"]["effects"][0]["source_record_ids"] == ["rec-2"]
    assert packet_out.trace["memory_evolution"]["effects"][0]["sub_ingest_trace"]["organization"]["target_layer"] == "semantic"
    assert written[0].text == "doc-b"


def test_hierarchical_copy_grouping_and_dedup_preserve_unique_values() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1", "doc_id": "doc-a"}},
            {"text": "a2", "metadata": {"session_id": "sess-1", "doc_id": "doc-a"}},
            {"text": "b1", "metadata": {"session_id": "sess-2", "doc_id": "doc-b"}},
            {"text": "b2", "metadata": {"session_id": "sess-2", "doc_id": "doc-c"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {
                "decision": True,
                "record_ids": ["rec-1", "rec-2", "rec-3", "rec-4"],
                "selector": {"kind": "boundary_match"},
            }
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        group_by=("session_id",),
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 2
    assert packet_out.trace["memory_evolution"]["group_count"] == 2
    first_payload = written[0].metadata["hierarchical"]["field_payload"]["doc_id"]
    second_payload = written[1].metadata["hierarchical"]["field_payload"]["doc_id"]
    assert first_payload == "doc-a"
    assert second_payload == ["doc-b", "doc-c"]
    assert written[0].metadata["hierarchical"]["group_key"] == {"session_id": "sess-1"}
    assert written[1].metadata["hierarchical"]["group_key"] == {"session_id": "sess-2"}


def test_hierarchical_modules_fall_back_to_source_layer_scan_when_decisions_store_missing() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"doc_id": "doc-a"}},
            {"text": "a2", "metadata": {"doc_id": "doc-b"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        target_layer="semantic",
    ).run(packet, store)

    assert packet_out.trace["memory_evolution"]["decision_source"] == "source_layer_scan"
    assert packet_out.trace["memory_evolution"]["selected_record_count"] == 3
    assert store.count("semantic") == 1
    assert store.iter_records("semantic")[0].metadata["hierarchical"]["field_payload"]["doc_id"] == ["doc-a", "doc-b", None]


def test_hierarchical_modules_noop_when_decisions_store_excludes_source_layer() -> None:
    from memprimitive.baselines import HierarchicalEvolution

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(store, "default", [{"text": "a1", "metadata": {"doc_id": "doc-a"}}])
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "other": {"decision": True, "record_ids": ["rec-1"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        target_layer="semantic",
    ).run(packet, store)

    assert packet_out.trace["memory_evolution"]["decision_source"] == "decisions_store"
    assert packet_out.trace["memory_evolution"]["selected_record_count"] == 0
    assert packet_out.trace["memory_evolution"]["effects"] == []
    assert store.count("semantic") == 0


def test_hierarchical_generate_mode_supports_default_and_custom_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1", "rec-2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary", "profile"),
        group_by=("session_id",),
        prompt="CUSTOM HIERARCHICAL PROMPT",
        target_layer="semantic",
    ).run(packet, store)

    written = store.iter_records("semantic")
    assert len(written) == 1
    assert fake_runtime.calls
    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT"
    assert written[0].text == "summary: custom::summary::sess-1::2\nprofile: custom::profile::sess-1::2"
    assert written[0].metadata["hierarchical"]["field_payload"]["summary"] == "custom::summary::sess-1::2"
    assert packet_out.trace["memory_evolution"]["active_group_keys"] == [{"session_id": "sess-1"}]


def test_hierarchical_generate_mode_uses_default_prompt_when_custom_prompt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(store, "default", [{"text": "a1", "metadata": {"session_id": "sess-1"}}])
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    _, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] != "CUSTOM HIERARCHICAL PROMPT"
    assert "higher-level hierarchical memory record" in fake_runtime.calls[0]["system"]
    assert store.iter_records("semantic")[0].text == "generated::summary::all::1"


def test_hierarchical_organization_generate_mode_supports_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalOrganization
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, _ = _represented_packet("incoming note")
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1", "rec-2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt="CUSTOM HIERARCHICAL PROMPT {{ group_key.session_id }} / {{ record_count }} / {{ records | length }}",
        target_layer="semantic",
    ).run(packet, store)

    assert "CUSTOM HIERARCHICAL PROMPT sess-1 / 2 / 2" == fake_runtime.calls[0]["system"]
    assert packet_out.trace["organization"]["prompt_is_template"] is True
    assert packet_out.trace["organization"]["prompt_trace"][0]["rendered_prompt"] == fake_runtime.calls[0]["system"]
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_organization_generate_mode_supports_recalled_prompt_from_current_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import ConcatenateReadout, HierarchicalOrganization, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils import _runtime
    from memprimitive.utils._template import text_prompt

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="semantic", theme="semantic"),
            ]
        )
    )
    _seed_layer(store, "profile", ["CURRENT STORE MEMORY"])
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, _ = _represented_packet("incoming note")
    packet = Packet(
        observation=packet.observation,
        units=packet.units,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-2", "rec-3"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "profile", ["WRONG PIPELINE STORE MEMORY"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    packet_out, store = HierarchicalOrganization(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt=text_prompt(
            "CUSTOM HIERARCHICAL PROMPT {{ recalled_prompt }} / {{ group_key.session_id }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"memory for {context['group_key']['session_id']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT CURRENT STORE MEMORY / sess-1"
    assert packet_out.trace["organization"]["prompt_trace"][0]["recall_prompt"]["rendered_recall_query"] == "memory for sess-1"
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_evolution_generate_mode_supports_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.utils import _runtime

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="default"), StoreLayerSpec(name="semantic", theme="semantic")]
        )
    )
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1", "rec-2"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt="CUSTOM HIERARCHICAL PROMPT {{ source_layer }} -> {{ target_layer }} / {{ group_key.session_id }} / {{ record_count }}",
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT default -> semantic / sess-1 / 2"
    assert packet_out.trace["memory_evolution"]["prompt_is_template"] is True
    assert packet_out.trace["memory_evolution"]["prompt_trace"][0]["rendered_prompt"] == fake_runtime.calls[0]["system"]
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_evolution_generate_mode_supports_recalled_prompt_from_current_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memprimitive.baselines import ConcatenateReadout, HierarchicalEvolution, RecencyRetrieval
    from memprimitive.pipeline import MemoryPipeline
    from memprimitive.utils import _runtime
    from memprimitive.utils._template import text_prompt

    fake_runtime = _FakeHierarchicalRuntime()
    monkeypatch.setattr(_runtime, "_DEFAULT_RUNTIME", fake_runtime)

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="profile"),
                StoreLayerSpec(name="semantic", theme="semantic"),
            ]
        )
    )
    _seed_layer(store, "profile", ["CURRENT STORE MEMORY"])
    _seed_layer_with_metadata(
        store,
        "default",
        [
            {"text": "a1", "metadata": {"session_id": "sess-1"}},
            {"text": "a2", "metadata": {"session_id": "sess-1"}},
        ],
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-2", "rec-3"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    pipeline_store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="default"), StoreLayerSpec(name="profile")]))
    _seed_layer(pipeline_store, "profile", ["WRONG PIPELINE STORE MEMORY"])
    retrieve_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=pipeline_store,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="generate",
        extract_fields=("summary",),
        group_by=("session_id",),
        prompt=text_prompt(
            "CUSTOM HIERARCHICAL PROMPT {{ recalled_prompt }} / {{ source_layer }} -> {{ target_layer }}",
            recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
            recall_query_builder=lambda packet, current_store, context: f"memory for {context['group_key']['session_id']}",
            sub_recall_pipeline=retrieve_pipeline,
        ),
        target_layer="semantic",
    ).run(packet, store)

    assert fake_runtime.calls[0]["system"] == "CUSTOM HIERARCHICAL PROMPT CURRENT STORE MEMORY / default -> semantic"
    assert packet_out.trace["memory_evolution"]["prompt_trace"][0]["recall_prompt"]["rendered_recall_query"] == "memory for sess-1"
    assert store.iter_records("semantic")[0].text == "custom::summary::sess-1::2"


def test_hierarchical_constructors_require_exactly_one_target_or_pipeline() -> None:
    from memprimitive.baselines import HierarchicalEvolution
    from memprimitive.pipeline import create_baseline_pipeline

    with pytest.raises(ValueError, match="Exactly one of target_layer or memory_pipeline"):
        HierarchicalEvolution(
            source_layer="default",
            extract_mode="copy",
            extract_fields=("doc_id",),
        )

    with pytest.raises(ValueError, match="Exactly one of target_layer or memory_pipeline"):
        HierarchicalEvolution(
            source_layer="default",
            extract_mode="copy",
            extract_fields=("doc_id",),
            target_layer="semantic",
            memory_pipeline=create_baseline_pipeline(),
        )


def test_hierarchical_memory_pipeline_mode_reuses_parent_store_and_custom_route() -> None:
    from memprimitive.baselines import AlwaysTrigger, AppendOrganization, HierarchicalEvolution
    from memprimitive.pipeline import MemoryPipeline

    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(name="default"),
                StoreLayerSpec(name="semantic", theme="semantic"),
                StoreLayerSpec(name="profile", theme="semantic"),
            ]
        )
    )
    _seed_layer_with_metadata(store, "default", [{"text": "a1", "metadata": {"doc_id": "doc-a"}}])
    child_pipeline = MemoryPipeline(
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="profile"),
        store=MemoryStore(),
    )
    packet, store = _stored_pipeline_packet("incoming note", store)
    packet = Packet(
        units=packet.units,
        placements=packet.placements,
        decisions=[True],
        decisions_store={
            "default": {"decision": True, "record_ids": ["rec-1"], "selector": {"kind": "manual"}}
        },
        trace=packet.trace,
    )

    packet_out, store = HierarchicalEvolution(
        source_layer="default",
        extract_mode="copy",
        extract_fields=("doc_id",),
        memory_pipeline=child_pipeline,
    ).run(packet, store)

    assert child_pipeline.store is store
    assert store.count("profile") == 1
    assert store.count("semantic") == 0
    assert packet_out.trace["memory_evolution"]["writer_pipeline_mode"] == "provided"
    assert packet_out.trace["memory_evolution"]["target_layer"] == "profile"
    assert packet_out.trace["memory_evolution"]["effects"][0]["sub_ingest_trace"]["organization"]["target_layer"] == "profile"
    assert store.iter_records("profile")[0].metadata["hierarchical"]["field_payload"]["doc_id"] == "doc-a"

