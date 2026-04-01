# `memprimitive.example.demonstration`

这个目录放的是可直接运行的 demonstration 脚本，用来展示 `MemoryPipeline`、`MemoryStore`、baseline primitives，以及 layered store / graph family 等机制如何以最小闭环方式组合起来。

建议在仓库根目录运行：

```text
python -m memprimitive.example.demonstration.<模块名>
```

例如：

```text
python -m memprimitive.example.demonstration.minimal_pipeline
```

## 当前说明

trigger 子系统已经暂时收缩为基础 slot trigger：

- `AlwaysWriteTrigger`
- `ThresholdWriteTrigger`
- `NeverEvolutionTrigger`
- `ThresholdEvolutionTrigger`

此前基于 `signal / scorer / gate / policy` 的 trigger-family 组合示例，以及依赖这些细分 trigger 的 Reflexion / TiM / A-MEM 风格 demonstration，已经从本目录移除。

## 基础 pipeline / store 演示

| 模块文件 | 说明 |
| --- | --- |
| `minimal_pipeline.py` | 最小 ingest -> recall 闭环。 |
| `topology_store.py` | 展示 `StoreTopology` / `MemoryStore` 的多层拓扑声明、按层写入和按层查询。 |
| `embedding_similarity_retrieval.py` | 展示带 `embedding` 的表示层，以及 `EmbeddingSimilarityRetrieval` 的检索行为。 |

## Layered / routing / dispatch 演示

| 模块文件 | 说明 |
| --- | --- |
| `layer_aware_semantic_working.py` | 写入 `working` / `semantic` 两层，并用 `LayerAwareRetrieval` 按层组合不同检索器。 |
| `layer_aware_working_graph.py` | 分离 working layer 和 graph layer，再通过 layer-aware retrieval 统一召回。 |
| `conditional_layer_routing.py` | `ConditionalLayerOrganization` 按规则把不同 unit 路由到不同 layer。 |
| `dispatch_organization_recall.py` | `DispatchOrganization` 同时驱动多个 organization child。 |
| `dispatch_organization_trace.py` | 与上例类似，但更强调 trace 输出。 |

## Graph baseline 演示

| 模块文件 | 说明 |
| --- | --- |
| `graph_append_entity_retrieval.py` | `GraphAppendOrganization` + `EntityRetrieval` 的轻量 graph 示例。 |
| `graph_baseline_pipeline.py` | graph baseline 闭环：`GraphAppendOrganization -> ThresholdEvolutionTrigger -> GraphNeighborAppendEvolution -> GraphSeedAndExpandRetrieval -> GraphReadout`。 |

## 推荐阅读顺序

1. `minimal_pipeline.py`
2. `topology_store.py`
3. `layer_aware_semantic_working.py`
4. `dispatch_organization_recall.py`
5. `graph_baseline_pipeline.py`
