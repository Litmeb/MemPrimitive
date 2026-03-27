# `memprimitive.example.demonstration`

可运行脚本，演示 `MemoryPipeline`、`MemoryStore` 与各类基础原语的组合方式。建议在仓库根目录执行：

```text
python -m memprimitive.example.demonstration.<模块名>
```

例如：`python -m memprimitive.example.demonstration.minimal_pipeline`

## Trigger-family / Reflexion 风格演示

这次新增了 3 个围绕 `OutcomeCorrectnessSignal`、`FeedbackPresenceSignal`、`FeedbackSchemaGate` 的完整 pipeline 示例：

| 模块文件 | 说明 |
|----------|------|
| `reflexion_trigger_failed_trial.py` | 失败 trial：`trial_failed=1.0`，schema gate 放行，`evolution_decisions` 变成 `True`。 |
| `reflexion_trigger_success_trial.py` | 成功 trial：有 feedback schema，但 `trial_failed=0.0`，因此不触发 evolution。 |
| `reflexion_trigger_schema_gate.py` | 缺少可解析的 outcome / feedback schema：即使 policy threshold 很低，也会被 `FeedbackSchemaGate` 拦住。 |

这些脚本都使用同一类组合方式：

- `signal_providers=(OutcomeCorrectnessSignal(), FeedbackPresenceSignal())`
- `scorer=WeightedSumScorer({"trial_failed": 1.0, "feedback_present": 0.1})`
- `gate=FeedbackSchemaGate()`
- `policy=ThresholdPolicy(...)`

## 其他演示

| 模块文件 | 说明 |
|----------|------|
| `minimal_pipeline.py` | 最小闭环：手工接线基础元，`AlwaysWriteTrigger` + 按时间检索 + 拼接读出。 |
| `composed_triggers.py` | 用 `compose_write_trigger` / `compose_evolution_trigger` 组合信号、门控与策略，并打印 trace。 |
| `topology_store.py` | 定义 `StoreTopology` / `MemoryStore` 多层拓扑，写入指定层并查询。 |
| `embedding_similarity_retrieval.py` | 表示层含 `embedding`，检索使用 `EmbeddingSimilarityRetrieval`。 |
| `layer_aware_semantic_working.py` | 多写入管道写入 `working` / `semantic`，`LayerAwareRetrieval` 按层选不同检索器并合并。 |
| `graph_append_entity_retrieval.py` | 图层的 `GraphAppendOrganization`，`EntityRetrieval` 在图数据上召回。 |
| `layer_aware_working_graph.py` | 分离的 working / 图写入管道，再经 `LayerAwareRetrieval` 合并检索与读出。 |
| `graph_baseline_pipeline.py` | graph baseline family 的完整闭环：`GraphAppendOrganization` -> `GraphNeighborAppendEvolution` -> `GraphSeedAndExpandRetrieval` -> `GraphReadout`。 |
| `conditional_layer_routing.py` | `ConditionalLayerOrganization` 按规则（实体、结构化三元组等）自动路由到不同存储层。 |
| `dispatch_organization_recall.py` | `DispatchOrganization` 同时挂 `AppendOrganization` 与 `GraphAppendOrganization`，单管道做 ingest + 分层召回。 |
| `dispatch_organization_trace.py` | 同上 dispatch 结构，并打印 `dispatch` 的 organization trace，便于调试分派行为。 |

## Graph baseline 推荐组合

如果你想先验证 graph family 的最小可组合闭环，推荐优先使用下面这组 baseline：

- ingest / graph 写入：`BasicRepresentation(elements=("text", "entities", "triple", "tags", "keywords"))` + `GraphAppendOrganization`
- link evolution：`ThresholdEvolutionTrigger` + `GraphNeighborAppendEvolution`
- graph recall：`GraphSeedAndExpandRetrieval`
- readout：`GraphReadout`

这套组合覆盖了阶段 1 需要的四个动作：

- ingest 到 `Graph` layer
- 在 graph layer 内建立 links
- 按 seed + neighbor expansion 召回
- 以 graph-readable 形式输出 readout
