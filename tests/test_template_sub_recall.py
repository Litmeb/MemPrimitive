from __future__ import annotations

from memprimitive.core import MemoryStore, Query, StoreLayerSpec, StoreTopology
from memprimitive.pipeline import MemoryPipeline
from memprimitive.baselines import ConcatenateReadout, EmbeddingSimilarityRetrieval, RecencyRetrieval
from memprimitive.utils._template import run_child_recall_pipeline, text_prompt

from baselines_test_helpers import _graph_vector_store, _seed_layer


def test_run_child_recall_pipeline_keeps_query_text_back_compat() -> None:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    _seed_layer(store, "profile", ["legacy query text still works"])
    pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="profile"),
        readout=ConcatenateReadout(),
        store=store,
    )

    readout, _ = run_child_recall_pipeline(
        store=store,
        query_text="legacy lookup",
        retrieve_pipeline=pipeline,
        fallback_readout_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
    )

    assert readout.text == "legacy query text still works"
    assert readout.source_ids == ["rec-1"]


def test_run_child_recall_pipeline_accepts_query_with_embedding() -> None:
    store = _graph_vector_store()
    _seed_layer(store, "knowledge_graph", ["Alice likes tea."])
    record = store.iter_records("knowledge_graph")[0]
    store.replace_record(
        "knowledge_graph",
        record.record_id,
        type(record)(
            record_id=record.record_id,
            unit_id=record.unit_id,
            layer=record.layer,
            text=record.text,
            timestamp=record.timestamp,
            embedding=[1.0, 0.0],
            metadata=record.metadata,
        ),
    )
    pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=1, layer="knowledge_graph"),
        readout=ConcatenateReadout(),
        store=store,
    )

    readout, _ = run_child_recall_pipeline(
        store=store,
        query=Query(text="Alice likes tea.", embedding=[1.0, 0.0]),
        retrieve_pipeline=pipeline,
        fallback_readout_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
    )

    assert readout.text == "Alice likes tea."
    assert readout.source_ids == ["rec-1"]
