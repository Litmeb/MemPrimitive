# `memprimitive.example.demonstration`

这个目录放的是可直接运行的 demonstration 脚本，用来展示 `MemoryPipeline`、`MemoryStore`、baseline primitives，以及最近补齐的 trigger-family shared infrastructure 是怎样拼起来工作的。

建议在仓库根目录运行：
```text
python -m memprimitive.example.demonstration.<模块名>
```

例如：
```text
python -m memprimitive.example.demonstration.minimal_pipeline
```

## Trigger-family 演示

这一组脚本主要服务于 trigger family 的共享基础设施，以及 Reflexion / TiM 风格 trigger 的可运行示例。

| 模块文件 | 说明 |
| --- | --- |
| `composed_triggers.py` | 最简单的 trigger-family 组合方式示例：直接用 `compose_write_trigger` / `compose_evolution_trigger` 装配 signal、scorer、gate、policy，并打印 trace。 |
| `trigger_family_infrastructure.py` | 阶段 2 新增的 shared trigger-family infrastructure 演示。分别展示 TiM-like 的 metadata / unit / placement / partition readiness 信号，以及 A-MEM-like 的 neighbor-count / top-similarity 信号和 schema / embedding / vector / graph gates。 |
| `reflexion_trigger_failed_trial.py` | Reflexion 风格失败 trial：使用稳定基线 `OutcomeConditionedEvolutionTrigger`，在 `trial_failed=1.0` 时触发 `evolution_decisions=True`。 |
| `reflexion_trigger_success_trial.py` | Reflexion 风格成功 trial：同样使用 `OutcomeConditionedEvolutionTrigger`，但 `trial_failed=0.0`，因此不触发 evolution。 |
| `reflexion_trigger_schema_gate.py` | 缺少可解析 outcome / feedback schema 时，`FeedbackSchemaGate` 会直接阻断 trigger。 |
| `reflexion_reflection_cycle.py` | 完整 Reflexion-like 闭环：失败 trial 不落盘、触发 `ReflectionGenerationEvolution` 追加 reflection、下一次 recall 通过 `BufferRetrieval + PromptContextReadout` 读出上下文。 |
| `partition_ready_local_maintenance.py` | TiM 风格 partition-ready 演示：一个带 `tim.group_id` 的 `tim_thought` 放到 `thought_memory` 后，通过 `NewWriteEvolutionTrigger` 触发 local maintenance。 |

上面这些 Reflexion / TiM 风格脚本都不是黑盒触发器，而是显式复用 trigger family 四件套：

- `OutcomeConditionedEvolutionTrigger = OutcomeCorrectnessSignal + FeedbackPresenceSignal + WeightedSumScorer + FeedbackSchemaGate + ThresholdPolicy`
- `NewWriteEvolutionTrigger = UnitTypeSignal + PlacementExistsSignal + PartitionKeyPresentSignal + MinScorer + LayerAllowedGate + ThresholdPolicy`

## 基础 pipeline 与 store 演示

| 模块文件 | 说明 |
| --- | --- |
| `minimal_pipeline.py` | 最小 ingest -> recall 闭环。适合快速确认 baseline pipeline 的基本数据流。 |
| `topology_store.py` | 演示 `StoreTopology` / `MemoryStore` 的多层拓扑声明、按层写入和按层查询。 |
| `embedding_similarity_retrieval.py` | 演示带 `embedding` 的表示层，以及 `EmbeddingSimilarityRetrieval` 的检索行为。 |

## Layered / routing / dispatch 演示

| 模块文件 | 说明 |
| --- | --- |
| `layer_aware_semantic_working.py` | 把内容写入 `working` / `semantic` 两层，并用 `LayerAwareRetrieval` 按层选择不同检索器后合并结果。 |
| `layer_aware_working_graph.py` | 把 working layer 和 graph layer 分离，再通过 layer-aware 检索统一召回。 |
| `conditional_layer_routing.py` | `ConditionalLayerOrganization` 按规则把不同 unit 路由到不同 layer。 |
| `dispatch_organization_recall.py` | `DispatchOrganization` 同时驱动多个 organization child，让一次 ingest 同时写入多条组织路径。 |
| `dispatch_organization_trace.py` | 与上一个例子类似，但更强调 trace 输出，方便观察 dispatch 行为。 |

## Graph baseline 演示

| 模块文件 | 说明 |
| --- | --- |
| `graph_append_entity_retrieval.py` | `GraphAppendOrganization` + `EntityRetrieval` 的轻量 graph 示例。 |
| `graph_baseline_pipeline.py` | graph baseline family 的完整闭环：`GraphAppendOrganization` -> `GraphNeighborAppendEvolution` -> `GraphSeedAndExpandRetrieval` -> `GraphReadout`。 |
| `graph_dependent_pipeline.py` | 阶段 3 的 graph-dependent 闭环：`NeighborExistsEvolutionTrigger` -> `GraphLinkEvolution` -> `GraphNeighborContextTraceEvolution` -> `GraphSeedAndExpandRetrieval` -> `GraphReadout`。 |

如果你想先验证 graph family 的最小可组合闭环，推荐优先看 `graph_baseline_pipeline.py`。它覆盖了阶段 1 graph baseline 的四个核心动作：

- ingest 到 `Graph` layer
- 在 graph layer 内追加 / 维护 links
- 按 seed + neighbor expansion 做召回
- 以 graph-readable 形式输出 readout

如果你要给后续 A-MEM-like prompt/controller 版本找一个更完整的阶段 3 底座，推荐看 `graph_dependent_pipeline.py`。它在上面的 graph baseline 闭环之外，又补上了：

- 基于邻居存在性的 graph-dependent evolution trigger
- 基于同层 neighbor candidates 的 link evolution
- 简化版 neighbor-context trace / metadata 更新

## 推荐阅读顺序

如果你第一次看这些 demonstration，比较顺手的顺序是：

1. `minimal_pipeline.py`
2. `composed_triggers.py`
3. `trigger_family_infrastructure.py`
4. `reflexion_trigger_failed_trial.py`
5. `reflexion_trigger_success_trial.py`
6. `reflexion_reflection_cycle.py`
7. `partition_ready_local_maintenance.py`
8. `layer_aware_semantic_working.py`
9. `graph_baseline_pipeline.py`
10. `graph_dependent_pipeline.py`

这样会先建立 baseline pipeline 心智模型，再进入 trigger family、layered retrieval，最后看 graph baseline family。
