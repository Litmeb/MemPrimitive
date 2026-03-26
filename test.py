from __future__ import annotations

from memprimitive import Observation, Query
from memprimitive.baselines.registry import iter_baseline_pipeline_instances
from memprimitive.pipeline import MemoryPipeline

for idx, pipeline in enumerate(iter_baseline_pipeline_instances(top_k=2)):
    try:
        pipeline.ingest(Observation(text="combinatorial ingest.", source="dialogue"))
        readout = pipeline.recall(Query(text="combinatorial"))
        assert isinstance(readout.text, str)
        assert pipeline.store.count() >= 1
    except Exception as e:
        print(f"pipeline index: {idx}")
        print(f"error: {e}")
        slots = MemoryPipeline.INGEST_SLOTS + MemoryPipeline.RECALL_SLOTS
        for slot in slots:
            mod = getattr(pipeline, slot)
            print(f"  {slot}: {type(mod).__name__} (spec.name={mod.spec.name!r})")
        print("  store:")
        for layer in pipeline.store.topology.layers:
            print(
                f"    {layer.name}: shape={layer.shape!r} theme={layer.theme!r} "
                f"indices={layer.indices!r} capacity={layer.capacity!r}"
            )