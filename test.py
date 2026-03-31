from __future__ import annotations

from memprimitive import IncompatibleCompositionError, Observation, Query
from memprimitive.baselines.registry import iter_baseline_pipeline_instances
from memprimitive import MemoryPipeline

ok = 0
n_incompatible = 0
n_other = 0

for idx, pipeline in enumerate(iter_baseline_pipeline_instances(top_k=2)):
    try:
        pipeline.store.check()
        pipeline.ingest(Observation(text="combinatorial ingest.", source="dialogue"))
        readout = pipeline.recall(Query(text="combinatorial"))
        assert isinstance(readout.text, str)
        assert pipeline.store.count() >= 1
        ok+=1
    except Exception as e:
        
        if isinstance(e, IncompatibleCompositionError):
            n_incompatible += 1
            
        else:
            n_other += 1
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
    if idx % 100 == 0:
        print(f"ok={ok},incomp={n_incompatible},total={idx+1}")
    if idx == 2000:
        break

print(
    f"done: ok={ok} IncompatibleCompositionError={n_incompatible} "
    f"other={n_other}"
)