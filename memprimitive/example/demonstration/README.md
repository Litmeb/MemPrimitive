# `memprimitive.example.demonstration`

这个目录放的是可直接运行的 demonstration 脚本，用来展示 `MemoryPipeline`、`MemoryStore`、baseline primitives，以及共享 trigger family / layered store / graph family 等机制如何以最小闭环方式组合起来。

建议在仓库根目录运行：

```text
python -m memprimitive.example.demonstration.<模块名>
```

例如：

```text
python -m memprimitive.example.demonstration.minimal_pipeline
```

## 写法原则

- demonstration 优先直接展示 `MemoryPipeline + module composition` 的 DSL 式用法。
- 示例要尽量短，让人一眼能看出 topology、slot 选择、ingest、recall。
- demonstration 不要依赖高层 workflow 封装或任务脚手架；那类内容更适合放在 `memprimitive/example/classics/`。
- 如果一个 motif 能直接用 baseline/classic module 组合表达，就优先展示组合本身，而不是再包一层 helper。

## Trigger-family 演示

这一组脚本主要服务于 trigger family 的共享基础设施，以及 Reflexion / TiM 风格 trigger 的可运行示例。

| 模块文件 | 说明 |
| --- | --- |
| `composed_triggers.py` | 最简单的 trigger-family 组合方式示例：直接用 `compose_write_trigger` / `compose_evolution_trigger` 装配 signal、scorer、gate、policy，并打印 trace。 |
| `trigger_family_infrastructure.py` | shared trigger-family infrastructure 演示，分别展示 TiM-like 与 A-MEM-like 信号和 gate。 |
| `reflexion_trigger_failed_trial.py` | Reflexion 风格失败 trial：使用 `OutcomeConditionedEvolutionTrigger`，在 `trial_failed=1.0` 时触发 `evolution_decisions=True`。 |
| `reflexion_trigger_success_trial.py` | Reflexion 风格成功 trial：同样使用 `OutcomeConditionedEvolutionTrigger`，但 `trial_failed=0.0`，因此不触发 evolution。 |
| `reflexion_trigger_schema_gate.py` | 缺少可解析 outcome / feedback schema 时，`FeedbackSchemaGate` 会直接阻断 trigger。 |
| `reflexion_reflection_cycle.py` | 完整 Reflexion-like DSL 闭环：直接用 `PlacementWithoutAppendOrganization + OutcomeConditionedEvolutionTrigger + ReflectionGenerationEvolution + BufferRetrieval + PromptContextReadout` 组合失败 trial 到下一次 recall 的上下文注入；默认通过 classic runtime 调用真实 LLM 生成 reflection，需要 `MEMPRIMITIVE_API_KEY`、`MEMPRIMITIVE_BASE_URL`、`MEMPRIMITIVE_MODEL`。 |
| `partition_ready_local_maintenance.py` | TiM 风格 partition-ready 演示：一个带 `tim.group_id` 的 `tim_thought` 放到 `thought_memory` 后，通过 `NewWriteEvolutionTrigger` 触发 local maintenance。 |

这些 Reflexion / TiM 风格脚本都不是黑盒触发器，而是显式复用 trigger family 四件套：

- `OutcomeConditionedEvolutionTrigger = OutcomeCorrectnessSignal + FeedbackPresenceSignal + WeightedSumScorer + FeedbackSchemaGate + ThresholdPolicy`
- `NewWriteEvolutionTrigger = UnitTypeSignal + PlacementExistsSignal + PartitionKeyPresentSignal + MinScorer + LayerAllowedGate + ThresholdPolicy`

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
| `dispatch_organization_recall.py` | `DispatchOrganization` 同时驱动多个 organization child，让一次 ingest 同时走多条组织路径。 |
| `dispatch_organization_trace.py` | 与上例类似，但更强调 trace 输出。 |

## Graph baseline 演示

| 模块文件 | 说明 |
| --- | --- |
| `graph_append_entity_retrieval.py` | `GraphAppendOrganization` + `EntityRetrieval` 的轻量 graph 示例。 |
| `graph_baseline_pipeline.py` | graph baseline family 的完整闭环：`GraphAppendOrganization -> GraphNeighborAppendEvolution -> GraphSeedAndExpandRetrieval -> GraphReadout`。 |
| `graph_dependent_pipeline.py` | graph-dependent 闭环：`NeighborExistsEvolutionTrigger -> GraphLinkEvolution -> GraphNeighborContextTraceEvolution -> GraphSeedAndExpandRetrieval -> GraphReadout`。 |
| `amem_like_graph_cycle.py` | A-MEM-like graph 闭环，但直接用 baseline slot 组合表达：`SemanticFieldEnrichmentRepresentation -> RetrievalOrientedEmbeddingRepresentation -> LLMJudgedWriteTrigger -> GraphAppendLinkReadyOrganization -> NeighborExistsEvolutionTrigger -> LinkStrengtheningEvolution + NeighborContextUpdateEvolution -> VectorGraphSeedAndExpandRetrieval -> NoteRenderReadout`。默认复用 classic runtime 调真实 LLM，需配置 `MEMPRIMITIVE_API_KEY`、`MEMPRIMITIVE_BASE_URL`、`MEMPRIMITIVE_MODEL`。 |

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
11. `amem_like_graph_cycle.py`
