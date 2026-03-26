# `memprimitive.example.demonstration`

可运行脚本，演示 `MemoryPipeline`、`MemoryStore` 与各类基元组合。建议在仓库根目录执行：

```text
python -m memprimitive.example.demonstration.<模块名>
```

例如：`python -m memprimitive.example.demonstration.minimal_pipeline`

| 模块文件 | 说明 |
|----------|------|
| `minimal_pipeline.py` | 最小闭环：手工接线基元，`AlwaysWriteTrigger` + 按时间检索 + 拼接读出。 |
| `composed_triggers.py` | 用 `compose_write_trigger` / `compose_evolution_trigger` 组合信号、门控与策略，并打印 trace。 |
| `topology_store.py` | 定义 `StoreTopology` / `MemoryStore` 多层拓扑，写入指定层并查询。 |
| `embedding_similarity_retrieval.py` | 表示层含 `embedding`，检索使用 `EmbeddingSimilarityRetrieval`。 |
| `layer_aware_semantic_working.py` | 多写入管道写入 `working` / `semantic`，`LayerAwareRetrieval` 按层选不同检索器并合并。 |
| `graph_append_entity_retrieval.py` | 图层的 `GraphAppendOrganization`，`EntityRetrieval` 在图数据上召回。 |
| `layer_aware_working_graph.py` | 分离的 working / 图写入管道，再经 `LayerAwareRetrieval` 合并检索与读出。 |
| `conditional_layer_routing.py` | `ConditionalLayerOrganization` 按规则（实体、结构化三元组等）自动路由到不同存储层。 |
| `dispatch_organization_recall.py` | `DispatchOrganization` 同时挂 `AppendOrganization` 与 `GraphAppendOrganization`，单管道内 ingest + 分层召回。 |
| `dispatch_organization_trace.py` | 同上 dispatch 结构，并打印 `dispatch` 的 organization trace，便于调试扇出行为。 |
