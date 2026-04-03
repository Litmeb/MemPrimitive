# MemPrimitive DSL 参考（当前实现面）

本文档只描述当前仓库里已经公开实现的 baseline module，来源以 `memprimitive.baselines` 为准。重点是回答三件事：

1. 每个 `slot` 目前有哪些 module。
2. 这些 module 构造时接受什么参数。
3. 它们在 pipeline 里的直接效果是什么。

## Pipeline Slots

标准顺序仍然是：

```text
unit_formation
-> representation
-> write_trigger
-> organization
-> evolution_trigger
-> memory_evolution
-> retrieval
-> readout
```

## `unit_formation`

把输入 observation 切成一个或多个 `MemoryUnit`。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `PassThroughUnitFormation` (`pass_through_unit_formation`) | 无额外参数 | 不切分输入；一条 observation 直接变成一个 unit。 |
| `SentenceSplitUnitFormation` (`sentence_split_unit_formation`) | 无额外参数 | 用轻量句子切分规则把一条 observation 拆成多个 sentence-level unit。 |
| `LineSplitUnitFormation` (`line_split_unit_formation`) | 无额外参数 | 按非空行切分，适合日志、笔记、逐行记录。 |
| `WindowedUnitFormation` (`windowed_unit_formation`) | `window_size=120`, `stride=80` | 把长文本切成固定窗口并允许重叠，适合长上下文分块写入。 |
| `MetadataHintUnitFormation` (`metadata_hint_unit_formation`) | 无额外参数 | 如果 `observation.metadata["units"]` 已给出候选单元，就直接按这些 hints 物化 unit。 |

## `representation`

为 unit 生成后续写入、检索、图组织会用到的表示字段。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `BasicRepresentation` (`basic_representation`) | `elements=("text", "embedding")`, `embedding_model=None`, `api_key=None`, `base_url=None`, `model=None` | 按 `elements` 生成基础表示，可组合文本、embedding、关键词、标签、摘要等字段。 |
| `KeywordRepresentation` (`keyword_representation`) | `elements=("text", "keywords", "tags")`, `embedding_model=None`, `api_key=None`, `base_url=None`, `model=None` | `BasicRepresentation` 的关键词导向封装，默认更偏向 keyword/tag 检索。 |
| `TripleRepresentation` (`triple_representation`) | `method="direct"`, `api_key=None`, `base_url=None`, `model=None`, `embedding_model=None` | 用 LLM 抽取 triple；可直接抽取，也可走两阶段实体/关系流程。 |
| `SemanticFieldEnrichmentRepresentation` (`semantic_field_enrichment_representation`) | `note_namespace="note"`, `strict_llm=True`, `default_category="Uncategorized"` | 生成 note-oriented 富字段，如标题、类别、上下文、标签等，供图组织和高级检索使用。 |
| `RetrievalOrientedEmbeddingRepresentation` (`retrieval_oriented_embedding_representation`) | `note_namespace="note"`, `default_category="Uncategorized"`, `embedding_version="content_context_keywords_tags_v2"`, `embedding_model=None` | 把 note 的多个语义字段拼成更适合检索的复合 embedding 表示。 |

## `write_trigger`

决定当前输入 unit 是否进入本轮写入/组织阶段。

说明：

- 所有 trigger module 都带 `slot` 参数，取值必须是 `write_trigger` 或 `evolution_trigger`。
- 下面这张表写的是“可挂到 `write_trigger` 的 trigger 类”。其中有些类默认更偏向 `evolution_trigger`，但代码上仍可通过 `slot="write_trigger"` 使用。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AlwaysTrigger` | `slot="write_trigger"` | 所有 unit 一律触发。 |
| `NeverTrigger` | `slot="write_trigger"` | 所有 unit 一律不触发。 |
| `ThresholdTrigger` | `slot="write_trigger"`, `threshold=0.5`, `constant=1.0` | 用常数分数和阈值比较，得到统一布尔决策。 |
| `BoundaryEventTrigger` | `slot="write_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False` | 根据结构边界事件触发，支持 `session/turn/chunk/subgoal/episode` 边界别名；命中后除广播写 `packet.decisions` 外，也会按 `*_id` 生成 `packet.decisions_store`。 |
| `RuntimeEventTrigger` | `slot="write_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False`, `target_layer=None`, `pressure_threshold=None` | 根据 runtime callback 事件触发，也可结合 store pressure 做事件级门控；当 `memory_pressure` 命中时会把目标 layer 的全部 records 写入 `packet.decisions_store`。 |
| `ScalarRuleTrigger` | `slot="write_trigger"`, `signal_key`, `threshold`, `comparator=">="`, `signal_source="auto"`, `aggregate="broadcast"`, `missing_value=None`, `target_layer=None` | 读取显式标量信号并做规则比较，例如分数、预算压力、置信度；当 `signal_key="memory_pressure"` 命中时也会把目标 layer 的全部 records 写入 `packet.decisions_store`。 |
| `ModelJudgeTrigger` | `slot="write_trigger"`, `system_prompt`, `decision_mode="bool"`, `threshold=0.5`, `per_unit=True`, `strict_json=True`, `judge_callable=None` | 调用真实模型或注入 judge 函数，判断是否应该写入。 |
| `PeriodicMaintenanceTrigger` | `slot="write_trigger"`, `every_n`, `counter_key="ingest_count"`, `decision_mode="broadcast"` | 按计数周期触发；更常用于维护，但也可以挂到写入侧。 |
| `IdleMaintenanceTrigger` | `slot="write_trigger"`, `min_idle_seconds=None`, `accepted_events=("idle", "sleep", "background_idle")`, `decision_mode="broadcast"` | 仅在 runtime 报告空闲维护窗口时触发；通常更适合演化侧。 |

## `organization`

决定 unit 写到哪个 layer、以什么写入形状落库。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AppendOrganization` (`append_organization`) | `target_layer="default"` | 把当前 unit 直接 append 到固定 layer。 |
| `ConditionalLayerOrganization` (`conditional_layer_organization`) | `default_layer="default"`, `rules=()` | 根据 tags、entities、unit metadata 等规则把 unit 路由到不同 layer，再 append。 |
| `GraphAppendOrganization` (`graph_append_organization`) | `target_layer="knowledge_graph"` | 把 unit 追加到图层，并补上 graph-oriented metadata。 |
| `PlacementWithoutAppendOrganization` (`placement_without_append_organization`) | `target_layer="trial_buffer"` | 只产出 placement，不真正落库；适合“先试算/先追踪”的写入路径。 |
| `GraphAppendLinkReadyOrganization` (`graph_append_link_ready_organization`) | `target_layer="knowledge_graph"`, `note_namespace="note"` | 把富 note 写入图层，并带上后续 link evolution 需要的 link-ready metadata。 |
| `HierarchicalOrganization` (`hierarchical_organization`) | `source_layer`, `extract_mode`, `extract_fields`, `group_by=()`, `prompt=None`, `target_layer=None`, `memory_pipeline=None` | 从 source layer 选中的 records 按组抽共性；最终写入不再直接 append，而是把抽象结果构造成 observation 后交给子 `MemoryPipeline.ingest()` 持久化，其中 `target_layer` 与 `memory_pipeline` 二选一。 |

## `evolution_trigger`

决定是否对当前写入结果继续执行 `memory_evolution`。

说明：

- 可用 trigger 类与 `write_trigger` 相同，差别只是实例化时的 `slot` 值。
- 有些 trigger 默认就面向演化侧，例如 `RuntimeEventTrigger`、`PeriodicMaintenanceTrigger`、`IdleMaintenanceTrigger`。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AlwaysTrigger` | `slot="evolution_trigger"` | 所有 unit 一律进入演化。 |
| `NeverTrigger` | `slot="evolution_trigger"` | 所有 unit 一律不进入演化。 |
| `ThresholdTrigger` | `slot="evolution_trigger"`, `threshold=0.5`, `constant=1.0` | 用常数分数和阈值控制是否演化。 |
| `BoundaryEventTrigger` | `slot="evolution_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False` | 根据结构边界事件触发演化；支持 `session/turn/chunk/subgoal/episode` 边界，并把同一 `*_id` 的 store records 写入 `packet.decisions_store`。 |
| `RuntimeEventTrigger` | `slot="evolution_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False`, `target_layer=None`, `pressure_threshold=None` | 根据 runtime 事件或 memory pressure 触发演化；`memory_pressure` 命中时会把对应 layer 的全部 records 标记到 `packet.decisions_store`。 |
| `ScalarRuleTrigger` | `slot="evolution_trigger"`, `signal_key`, `threshold`, `comparator=">="`, `signal_source="auto"`, `aggregate="broadcast"`, `missing_value=None`, `target_layer=None` | 根据显式标量规则触发演化；当 `signal_key="memory_pressure"` 命中时会把对应 layer 的全部 records 标记到 `packet.decisions_store`。 |
| `ModelJudgeTrigger` | `slot="evolution_trigger"`, `system_prompt`, `decision_mode="bool"`, `threshold=0.5`, `per_unit=True`, `strict_json=True`, `judge_callable=None` | 由模型判断哪些 unit 需要演化。 |
| `PeriodicMaintenanceTrigger` | `slot="evolution_trigger"`, `every_n`, `counter_key="ingest_count"`, `decision_mode="broadcast"` | 每累计 `every_n` 次计数触发一次维护/演化。 |
| `IdleMaintenanceTrigger` | `slot="evolution_trigger"`, `min_idle_seconds=None`, `accepted_events=("idle", "sleep", "background_idle")`, `decision_mode="broadcast"` | 只有在空闲维护窗口到来时触发。 |

## `memory_evolution`

在组织完成后，对刚写入内容或相关旧记录做追加、摘要、迁移、连边、反思生成等演化。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AppendOnlyEvolution` (`append_only_evolution`) | 无额外参数 | 不额外改写 store，只把已有组织结果沿用到 evolution 阶段。 |
| `TraceOnlyEvolution` (`trace_only_evolution`) | 无额外参数 | 不改 store，只在 trace 里记录显式“无副作用/占位 effect”。 |
| `SummaryRewriteEvolution` (`summary_rewrite_evolution`) | `target_layer="default"` | 为被激活的 unit 生成摘要记录，并写入目标 layer。 |
| `LayerMoveEvolution` (`layer_move_evolution`) | `target_layer="default"` | 把演化激活的记录复制追加到另一个 layer。 |
| `GraphLinkEvolution` (`graph_link_evolution`) | `target_layer="knowledge_graph"`, `neighbor_limit=2`, `bidirectional=True`, `min_score=0.1`, `rewrite_neighbor_metadata=False` | 在同层图记录之间找邻居并建立 link，可选双向连边与邻居 metadata 重写。 |
| `GraphNeighborAppendEvolution` (`graph_neighbor_append_evolution`) | `target_layer="knowledge_graph"`, `neighbor_limit=2`, `bidirectional=True` | `GraphLinkEvolution` 的兼容包装，偏向“邻居连边追加”基线。 |
| `GraphNeighborContextTraceEvolution` (`graph_neighbor_context_trace_evolution`) | `target_layer="knowledge_graph"`, `rewrite_metadata=False` | 追踪 linked neighbor 的上下文，并可保守地把摘要/metadata 写回。 |
| `LinkStrengtheningEvolution` (`link_strengthening_evolution`) | `target_layer="knowledge_graph"`, `candidate_k=5`, `max_links_per_record=4`, `note_namespace="note"`, `default_category="Uncategorized"`, `embedding_version="content_context_keywords_tags_v2"`, `strict_llm=True` | 基于富 note 表示先找候选邻居，再用 LLM 选择和强化更可靠的图连接。 |
| `NeighborContextUpdateEvolution` (`neighbor_context_update_evolution`) | `target_layer="knowledge_graph"`, `candidate_k=5`, `note_namespace="note"`, `default_category="Uncategorized"`, `embedding_version="content_context_keywords_tags_v2"`, `strict_llm=True` | 从当前 note 视角回写邻居上下文、标签或关系说明。 |
| `ReflectionGenerationEvolution` (`reflection_generation_evolution`) | `target_layer="reflections"`, `memory_size=3`, `window_size=None`, `reflection_generator=None`, `prompt_builder=None` | 从失败轨迹生成 strategy/reflection note，并追加到反思层。 |
| `HierarchicalEvolution` (`hierarchical_evolution`) | `source_layer`, `extract_mode`, `extract_fields`, `group_by=()`, `prompt=None`, `target_layer=None`, `memory_pipeline=None` | 消费 `packet.decisions_store` 或 source layer 全扫描结果，按组生成更高层抽象 observation，并通过子 `MemoryPipeline.ingest()` 写入；`target_layer` 模式会构造默认 pipeline，`memory_pipeline` 模式则复用传入 pipeline 与当前 store。 |

## `retrieval`

从 store 取回和 query 相关的记录。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `RecencyRetrieval` (`recency_retrieval`) | `top_k=3`, `layer=None` | 完全按时间新近性取回最近的 `top_k` 条记录。 |
| `KeywordCountRetrieval` (`keyword_count_retrieval`) | `top_k=3`, `layer=None` | 按 query token 命中数排序，再用 recency 打破平分。 |
| `EmbeddingSimilarityRetrieval` (`embedding_similarity_retrieval`) | `top_k=3`, `layer=None`, `embedding_model="sentence-transformers/all-MiniLM-L6-v2"` | 用 embedding cosine similarity 取回最相近记录。 |
| `TagRetrieval` (`tag_retrieval`) | `top_k=3`, `layer=None` | 按 query token 与 record tags 的重叠程度检索。 |
| `EntityRetrieval` (`entity_retrieval`) | `top_k=3`, `layer=None` | 按 query entity/token 与 record entities 的重叠程度检索。 |
| `BM25Retrieval` (`bm25_retrieval`) | `top_k=3`, `layer=None` | 对文本和 representation keywords 做 BM25 排序，再参考 recency。 |
| `GraphNeighborRetrieval` (`graph_neighbor_retrieval`) | `top_k=3`, `layer="knowledge_graph"`, `include_seed_records=False` | 输入 seed record id 后，沿图边扩展邻居并返回。 |
| `GraphSeedAndExpandRetrieval` (`graph_seed_and_expand_retrieval`) | `top_k=3`, `layer="knowledge_graph"`, `seed_top_k=2`, `include_seed_records=True` | 先从 query 文本里选 graph seed，再做一跳邻居扩展。 |
| `LayerAwareRetrieval` (`layer_aware_retrieval`) | `default_retriever=None`, `retriever_by_layer=None`, `active_layers=None`, `top_k=3`, `top_k_by_layer=None`, `merge_weight_by_layer=None`, `merge_strategy="global_rank"` | 对不同 layer 分别调用不同 retriever，再按全局规则合并结果。 |
| `BufferRetrieval` (`buffer_retrieval`) | `top_k=3`, `layer="reflections"`, `chronological=True` | 不做 query 搜索，只从某一 layer 读一个有界 recency window。 |
| `VectorGraphSeedAndExpandRetrieval` (`vector_graph_seed_and_expand_retrieval`) | `top_k=3`, `layer="knowledge_graph"`, `candidate_k=5`, `neighbor_expansion_k=3`, `note_namespace="note"`, `default_category="Uncategorized"`, `agentic_search=False`, `query_expand_with_llm=False` | 先做向量 seed 检索，再结合图邻居扩展；适合富 note 图记录。 |

## `readout`

把 retrieval 返回的记录渲染成最终给 agent / 下游调用方使用的输出。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `ConcatenateReadout` (`concatenate_readout`) | `separator="\n"` | 把 retrieval items 直接拼成一个字符串，并附带来源 record id。 |
| `BulletListReadout` (`bullet_list_readout`) | 无额外参数 | 把每条 retrieval item 渲染成一行 bullet list。 |
| `GroupedByLayerReadout` (`grouped_by_layer_readout`) | 无额外参数 | 按来源 layer 分组展示 retrieval items。 |
| `JSONReadout` (`json_readout`) | 无额外参数 | 输出 JSON 字符串，适合工具链或 agent 下游解析。 |
| `GraphReadout` (`graph_readout`) | `include_links=True` | 以稳定、可读的格式展示图记录及其 link metadata。 |
| `PromptContextReadout` (`prompt_context_readout`) | `memory_layer="reflections"`, `default_strategy="reflexion"`, `top_k=3` | 把 retrieval 结果组织成下一步 prompt context。 |
| `NoteRenderReadout` (`note_render_readout`) | `note_namespace="note"`, `default_category="Uncategorized"`, `include_context=True`, `include_tags=True` | 把富 note payload 渲染成更适合人读或 prompt 注入的 note 视图。 |

## 备注

- 这里列的是“当前公开 baseline surface”，不是所有内部 helper。
- 触发器类是统一设计：同一个类通过 `slot=` 绑定到 `write_trigger` 或 `evolution_trigger`。
- 如果后续 baseline module 有增删改，本文档也应随代码同步更新。
