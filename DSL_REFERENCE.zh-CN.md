# MemPrimitive DSL 参考（当前实现面）

本文档描述当前仓库里已经公开实现、且可直接用于 `MemoryPipeline` 的 baseline surface。内容以 `memprimitive.baselines`、`memprimitive.utils._template`、`memprimitive.utils._llm_function_tools` 的现状为准，目标是提供一份偏 API Reference 风格的中文索引，而不是设计草案。

本文档重点覆盖：

1. Pipeline 八个 slot 当前有哪些 baseline module。
2. `tool call` 写路径里的九种默认工具与自定义工具协议。
3. 如何使用模板系统编写 template prompt。

## Pipeline Slots

标准执行顺序：

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

## 通用约定

- 本文档中的“构造参数”只列常用公开参数；更细节的内部 helper 不在这里展开。
- 同一个 trigger 类通过 `slot="write_trigger"` 或 `slot="evolution_trigger"` 绑定到不同 slot。
- 模板相关模块统一接受 `PromptPlan | str`；`str` 会被当作简单文本模板处理。
- retrieval 的 query 输入当前有两个公开入口：
  - `packet.query: Query | None`
  - `packet.queries: list[Query] | None`
- retrieval 的 query 解析优先级是：
  - 若 `packet.queries` 非空，则使用 `packet.queries`
  - 否则回退到 `packet.query`
- tool-call 写路径当前公开入口是：
  - `LLMFunctionCallOrganization`
  - `LLMFunctionCallEvolution`

## `unit_formation`

把输入 `Observation` 切成一个或多个 `MemoryUnit`。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `PassThroughUnitFormation` (`pass_through_unit_formation`) | 无额外参数 | 不切分输入；一条 observation 直接变成一个 unit。 |
| `SentenceSplitUnitFormation` (`sentence_split_unit_formation`) | 无额外参数 | 按句子边界切分成长短更合适的 unit。 |
| `LineSplitUnitFormation` (`line_split_unit_formation`) | 无额外参数 | 按非空行切分，适合日志、逐行笔记。 |
| `WindowedUnitFormation` (`windowed_unit_formation`) | `window_size=120`, `stride=80` | 把长文本切成可重叠窗口。 |
| `MetadataHintUnitFormation` (`metadata_hint_unit_formation`) | 无额外参数 | 如果 `observation.metadata["units"]` 已给出 unit hints，就直接按 hints 物化。 |

## `representation`

为 unit 生成后续组织、检索、图结构和提示词可能会使用的表示字段。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `BasicRepresentation` (`basic_representation`) | `elements=("text", "embedding")`, `embedding_model=None`, `api_key=None`, `base_url=None`, `model=None` | 生成基础表示，适合文本和 embedding 等轻量字段。 |
| `TripleRepresentation` (`triple_representation`) | `method="direct"`, `prompt=None`, `api_key=None`, `base_url=None`, `model=None`, `embedding_model=None`, `embed_extracted=False`, `embed_entities=False` | 用 LLM 抽取 `(subject, relation, object)` triples；`embed_extracted=True` 时给整份抽取出的 `entities + triples` 生成一个 graph-object embedding 写回 `unit.embedding`，`embed_entities=True` 时额外把每个 entity 的 embedding 写入 `metadata["representation"]["entity_embeddings"]`。 |
| `LLMRepresentation` (`llm_representation`) | `field`, `prompt`, `api_key=None`, `base_url=None`, `model=None`, `embedding_model=None` | 用 prompt 驱动的方式为 unit 生成一个语义字段，并回写到 `metadata["representation"]` 等标准位置。 |
| `ConfigurableEmbeddingRepresentation` (`configurable_embedding_representation`) | `embedding_text=None`, `embedding_version="content_context_keywords_tags_v2"`, `embedding_model=None` | 基于可配置文本或模板渲染文本生成 embedding，并记录 embedding 输入来源。 |

说明：

- A-MEM / note-graph 风格的富表示现在推荐直接组合多个 `LLMRepresentation`，分别生成 `context`、`keywords`、`tags`、`category`、`attributes`，再接 `ConfigurableEmbeddingRepresentation`。

## `write_trigger`

决定当前输入 unit 是否进入写入/组织阶段。

说明：

- 所有 trigger 都有 `slot` 参数，必须是 `write_trigger` 或 `evolution_trigger`。
- `packet.decisions` 仍然是下游最稳定的布尔决策接口。
- `packet.decisions_store` 是并行的 store 级选择结果，用于“选中已有 records 而不是只选中当前输入 unit”的场景。
- `StoreAllTrigger` 只补 `packet.decisions_store`，不主动生成新的 `packet.decisions`。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AlwaysTrigger` | `slot="write_trigger"` | 所有 unit 均触发。 |
| `NeverTrigger` | `slot="write_trigger"` | 所有 unit 均不触发。 |
| `StoreAllTrigger` | `slot="write_trigger"` | 不改 `packet.decisions`，只把所有非空 layer 的全部 records 写入 `packet.decisions_store`。 |
| `BoundaryEventTrigger` | `slot="write_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False` | 根据 `session/turn/chunk/subgoal/episode` 等结构边界事件触发，并可映射到同类 store records。 |
| `RuntimeEventTrigger` | `slot="write_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False`, `target_layer=None`, `pressure_threshold=None` | 根据 runtime 事件或 `memory_pressure` 触发。 |
| `ScalarRuleTrigger` | `slot="write_trigger"`, `signal_key`, `threshold`, `comparator=">="`, `signal_source="auto"`, `aggregate="broadcast"`, `missing_value=None`, `target_layer=None` | 根据显式标量信号触发，如分数、预算压力、置信度。 |
| `LLMJudgeTrigger` | `slot="write_trigger"`, `prompt`, `decision_mode="bool"`, `threshold=0.5`, `per_unit=True` | 用真实模型和 `PromptPlan` 风格 prompt 做触发判断。 |
| `PeriodicMaintenanceTrigger` | `slot="write_trigger"`, `every_n`, `counter_key="ingest_count"`, `decision_mode="broadcast"` | 按计数周期触发。 |
| `IdleMaintenanceTrigger` | `slot="write_trigger"`, `min_idle_seconds=None`, `accepted_events=("idle", "sleep", "background_idle")`, `decision_mode="broadcast"` | 在空闲维护窗口触发。 |

## `organization`

决定当前 unit 写到哪个 layer，以及以什么写入形状落库。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AppendOrganization` (`append_organization`) | `target_layer="default"` | 直接 append 到固定 layer。 |
| `ConditionalLayerOrganization` (`conditional_layer_organization`) | `default_layer="default"`, `rules=()` | 按 tags、entities、metadata 等规则路由到不同 layer。 |
| `GraphAppendOrganization` (`graph_append_organization`) | `target_layer="knowledge_graph"`, `separate=False`, `separate_layer=None` | 向图层追加记录，并维护 graph metadata；保留 unit 上已有的 note payload 等 metadata，因而也可承接 A-MEM 风格的 note graph 写入；`separate=True` 时原文本与 triple 记录可分层落库。 |
| `GraphEntityAppendOrganization` (`graph_entity_append_organization`) | `target_layer="knowledge_graph"`, `separate=False`, `separate_layer=None` | 保持 `GraphAppendOrganization` 的 graph metadata / provenance 约定，但把一个 unit 展开成多条 graph record，每条 `record.text` 是一个 entity；若当前 unit 没有 entity，则跳过该 unit。 |
| `GraphDeduplicationAppendOrganization` (`graph_deduplication_append_organization`) | `target_layer="knowledge_graph"`, `threshold`, `separate=False`, `separate_layer=None` | 向图层写入前先与同 layer 现有 graph records 做 embedding top-1 相似度匹配；若 `similarity > threshold`，则原地合并节点并更新 text / embedding / triples，否则正常 append；`separate=True` 时 source 文本仍可单独落到 side layer。 |
| `GraphEntityDeduplicationAppendOrganization` (`graph_entity_deduplication_append_organization`) | `target_layer="knowledge_graph"`, `threshold`, `separate=False`, `separate_layer=None` | 保持 `GraphDeduplicationAppendOrganization` 的阈值去重与 provenance 语义，但按 entity 独立执行 merge-or-append：每个 entity 用自己的 entity embedding 与现有 graph records 比较，相似度超过阈值时合并到命中节点，否则追加一个新的 entity 节点；缺少该 entity embedding 时只跳过该 entity。 |
| `PlacementWithoutAppendOrganization` (`placement_without_append_organization`) | `target_layer="trial_buffer"` | 只给出 placement，不实际 append。 |
| `HierarchicalOrganization` (`hierarchical_organization`) | `source_layer`, `extract_mode`, `extract_fields`, `group_by=()`, `prompt=None`, `target_layer=None`, `memory_pipeline=None` | 对选中的 source-layer records 做抽象聚合，再通过子 `MemoryPipeline.ingest()` 写入高层记忆。 |
| `LLMFunctionCallOrganization` (`llm_function_call_organization`) | `prompt`, `tools`, `target_layer=None`, `max_turns=6`, `strict_tools=True`, `allow_no_tool_call=True`, `api_key=None`, `base_url=None`, `model=None`, `embedding_model=None` | 用 LLM tool call 在 organization 阶段执行 `add / update / delete` 等写操作。 |

## `evolution_trigger`

决定是否继续执行 `memory_evolution`。

说明：

- 可用 trigger 类与 `write_trigger` 相同，只是 `slot` 取值不同。
- 某些 trigger 更常见于维护/演化，如 `PeriodicMaintenanceTrigger`、`IdleMaintenanceTrigger`。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AlwaysTrigger` | `slot="evolution_trigger"` | 所有 unit 进入演化。 |
| `NeverTrigger` | `slot="evolution_trigger"` | 所有 unit 不进入演化。 |
| `StoreAllTrigger` | `slot="evolution_trigger"` | 补 `packet.decisions_store`，不改 `packet.decisions`。 |
| `BoundaryEventTrigger` | `slot="evolution_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False` | 根据结构边界事件触发演化。 |
| `RuntimeEventTrigger` | `slot="evolution_trigger"`, `accepted_events`, `match_mode="any"`, `default_decision=False`, `invert=False`, `target_layer=None`, `pressure_threshold=None` | 根据 runtime 事件或 `memory_pressure` 触发演化。 |
| `ScalarRuleTrigger` | `slot="evolution_trigger"`, `signal_key`, `threshold`, `comparator=">="`, `signal_source="auto"`, `aggregate="broadcast"`, `missing_value=None`, `target_layer=None` | 根据显式标量规则触发演化。 |
| `LLMJudgeTrigger` | `slot="evolution_trigger"`, `prompt`, `decision_mode="bool"`, `threshold=0.5`, `per_unit=True` | 用模型判断哪些 unit 或阶段需要演化。 |
| `PeriodicMaintenanceTrigger` | `slot="evolution_trigger"`, `every_n`, `counter_key="ingest_count"`, `decision_mode="broadcast"` | 周期性触发维护。 |
| `IdleMaintenanceTrigger` | `slot="evolution_trigger"`, `min_idle_seconds=None`, `accepted_events=("idle", "sleep", "background_idle")`, `decision_mode="broadcast"` | 空闲维护窗口触发。 |

## `memory_evolution`

在组织完成后，对新写入内容或旧记录做追加、重写、迁移、连边、反思生成等演化。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `AppendOnlyEvolution` (`append_only_evolution`) | 无额外参数 | 不额外改 store，只沿用组织阶段结果。 |
| `TraceOnlyEvolution` (`trace_only_evolution`) | 无额外参数 | 不改 store，只在 trace 中保留显式占位信息。 |
| `SummaryRewriteEvolution` (`summary_rewrite_evolution`) | `target_layer="default"` | 为激活内容生成摘要并写入目标层。 |
| `LayerMoveEvolution` (`layer_move_evolution`) | `target_layer="default"` | 把激活记录复制/迁移到另一层。 |
| `GraphLinkEvolution` (`graph_link_evolution`) | `target_layer="knowledge_graph"`, `neighbor_limit=2`, `bidirectional=True`, `min_score=0.1`, `rewrite_neighbor_metadata=False` | 在图层记录间建立或更新 links。 |
| `GraphNeighborAppendEvolution` (`graph_neighbor_append_evolution`) | `target_layer="knowledge_graph"`, `neighbor_limit=2`, `bidirectional=True` | 图邻居追加式兼容封装。 |
| `GraphNeighborContextTraceEvolution` (`graph_neighbor_context_trace_evolution`) | `target_layer="knowledge_graph"`, `rewrite_metadata=False` | 跟踪图邻居上下文，并可保守回写 metadata。 |
| `LinkStrengtheningEvolution` (`link_strengthening_evolution`) | `target_layer="knowledge_graph"`, `candidate_k=5`, `max_links_per_record=4`, `note_namespace="note"`, `default_category="Uncategorized"`, `embedding_version="content_context_keywords_tags_v2"`, `strict_llm=True` | 基于富 note 表示强化图连接。 |
| `NeighborContextUpdateEvolution` (`neighbor_context_update_evolution`) | `target_layer="knowledge_graph"`, `candidate_k=5`, `note_namespace="note"`, `default_category="Uncategorized"`, `embedding_version="content_context_keywords_tags_v2"`, `strict_llm=True` | 更新邻居上下文或关系说明。 |
| `ReflectionGenerationEvolution` (`reflection_generation_evolution`) | `target_layer="reflections"`, `memory_size=3`, `window_size=None`, `reflection_generator=None`, `prompt_builder=None` | 生成 reflection / strategy note。 |
| `HierarchicalEvolution` (`hierarchical_evolution`) | `source_layer`, `extract_mode`, `extract_fields`, `group_by=()`, `prompt=None`, `target_layer=None`, `memory_pipeline=None` | 基于 `packet.decisions_store` 或 source-layer 扫描，做高层抽象和分层持久化。 |
| `LLMFunctionCallEvolution` (`llm_function_call_evolution`) | `prompt`, `tools`, `source_layer=None`, `target_layer=None`, `max_turns=6`, `strict_tools=True`, `allow_no_tool_call=True`, `api_key=None`, `base_url=None`, `model=None`, `embedding_model=None` | 用 LLM tool call 对已有 records 做新增、更新、删除等演化操作。 |

## `retrieval`

从 store 取回和 query 相关的记录。

说明：

- 对“以 query 驱动”的 retrieval module，如果 `packet.queries` 非空，则默认按 query 顺序逐个执行检索，再合并回一个扁平的 `packet.retrieved`。
- `QueryRewriteRetrieval` 是一个 retrieval wrapper：它先把 `packet.query` 改写成一个或多个 query，再把改写后的 `packet.query` / `packet.queries` 交给内部 retriever。
- 默认合并策略是 `query_order_dedupe`：
  - 先保留每个 query 自己的原始排序语义
  - 合并时按 query 顺序拼接
  - 以 `record_id` 去重；同一 record 被多个 query 命中时保留第一次出现的那一份
- 合并后的 `packet.retrieved.trace` 会额外包含：
  - `query_count`
  - `query_ids`
  - `merge_strategy`
  - `per_query`
  - `final_returned_count`
- `DispatchRetrieval` 仍然只表示“多 retrieval module 分发”；它不会额外改变上述多 query 默认语义。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `RecencyRetrieval` (`recency_retrieval`) | `top_k=3`, `layer=None`, `source="store"` | 按时间新近性检索。 |
| `KeywordCountRetrieval` (`keyword_count_retrieval`) | `top_k=3`, `layer=None`, `source="store"` | 按 token 命中数排序。 |
| `EmbeddingSimilarityRetrieval` (`embedding_similarity_retrieval`) | `top_k=3`, `layer=None`, `embedding_model="sentence-transformers/all-MiniLM-L6-v2"`, `source="store"` | 按 embedding 相似度检索。 |
| `TagRetrieval` (`tag_retrieval`) | `top_k=3`, `layer=None` | 按 query 与 tags 的重叠度检索。 |
| `EntityRetrieval` (`entity_retrieval`) | `top_k=3`, `layer=None` | 按 query 与 entities 的重叠度检索。 |
| `BM25Retrieval` (`bm25_retrieval`) | `top_k=3`, `layer=None`, `source="store"` | 对文本和关键词做 BM25 排序。 |
| `GraphNeighborRetrieval` (`graph_neighbor_retrieval`) | `top_k=3`, `layer="knowledge_graph"`, `include_seed_records=False` | 从 seed 记录出发扩展图邻居。 |
| `ExpandRetrievedGraphNeighbors` (`expand_retrieved_graph_neighbors`) | `top_k=3`, `layer="knowledge_graph"`, `include_seed_records=True`, `per_seed_top_k=None`, `dedupe=True` | 以当前 `packet.retrieved.items` 为种子继续做一跳图邻居扩展。 |
| `VectorGraphSeedAndExpandRetrieval` (`vector_graph_seed_and_expand_retrieval`) | `top_k=3`, `layer="knowledge_graph"`, `candidate_k=5`, `neighbor_expansion_k=3`, `note_namespace="note"`, `default_category="Uncategorized"`, `agentic_search=False`, `query_expand_with_llm=False`, `prompt=None` | 一个 retrieval wrapper：先用 `EmbeddingSimilarityRetrieval` 做 graph-vector layer 的 seed 检索，再把 seed 交给 `ExpandRetrievedGraphNeighbors` 扩邻居；外层只额外负责 note payload 投影、可选 LLM query expansion 和可选 agentic rerank。 |
| `LayerAwareRetrieval` (`layer_aware_retrieval`) | `default_retriever=None`, `retriever_by_layer=None`, `active_layers=None`, `top_k=3`, `top_k_by_layer=None`, `merge_weight_by_layer=None`, `merge_strategy="global_rank"` | 分层调用不同 retriever，再合并结果。 |
| `BufferRetrieval` (`buffer_retrieval`) | `top_k=3`, `layer="reflections"`, `chronological=True` | 读取某个 layer 的有界 recency window。 |
| `QueryRewriteRetrieval` (`query_rewrite_retrieval`) | `retriever`, `strategy="llm"`, `prompt=None`, `allow_multi_query=False`, `regex_rules=None`, `include_original=False`, `max_queries=None`, `strip_queries=True`, `drop_empty_queries=True` | 先做 query rewrite，再委托内部 retriever 执行。`strategy="llm"` 时要求 `prompt`，返回严格 JSON 的单 query 或多 query；`strategy="regex"` 时按顺序应用 `regex_rules`，当前只产出单 query。 |

## `readout`

把 retrieval 结果整理成最终输出。

| Module | 构造参数 | 效果 |
| --- | --- | --- |
| `ConcatenateReadout` (`concatenate_readout`) | `separator="\n"` | 直接拼接 retrieval items。 |
| `BulletListReadout` (`bullet_list_readout`) | 无额外参数 | 渲染为 bullet list。 |
| `GroupedByLayerReadout` (`grouped_by_layer_readout`) | 无额外参数 | 按 layer 分组展示。 |
| `JSONReadout` (`json_readout`) | 无额外参数 | 输出 JSON 字符串。 |
| `GraphReadout` (`graph_readout`) | `include_links=True` | 展示图记录及 links。 |
| `PromptContextReadout` (`prompt_context_readout`) | `memory_layer="reflections"`, `default_strategy="reflexion"`, `top_k=3` | 把 recall 结果整理成 prompt context。 |
| `NoteRenderReadout` (`note_render_readout`) | `note_namespace="note"`, `default_category="Uncategorized"`, `include_context=True`, `include_tags=True` | 渲染富 note payload。 |
| `MidDecodingMemoryReadout` (`mid_decoding_memory_readout`) | `prompt`, `retrieve_pipeline`, `tools=None`, `max_turns=6`, `strict_tools=True`, `allow_no_tool_call=True`, `runtime_now_factory=None`, `api_key=None`, `base_url=None`, `model=None`, `embedding_model=None` | 在 readout 阶段复用 `Runtime.run_agent(..., tools=..., max_turns=...)` 做多轮 tool-calling answer generation。第一版默认只内建 `MEM_READ`：LLM 可在生成中途用 JSON 参数触发一次子 recall，再继续完成最终答案。最终仍返回标准 `Readout(text, source_ids, metadata)`，其中 `source_ids` 汇总所有 `MEM_READ` 命中的 `record_id`。 |
| `TemplateReadout` (`template_readout`) | `prompt`, `filters=None`, `missing_value=""`, `note_namespace="note"`, `default_category="Uncategorized"`, `runtime_now_factory=None` | 通过模板系统把 recall 结果渲染成可控文本，是当前 template prompt/readout 的统一公开入口。 |

## Tool Call 写路径

### 适用模块

当前支持 tool call 写操作的 baseline module：

- `LLMFunctionCallOrganization`
- `LLMFunctionCallEvolution`

二者共享同一套工具协议、参数校验规则和 trace 结构。

### 九种默认工具

当前内置九种 built-in write tools，传给 `tools=[...]` 时使用字符串名字即可：

| 工具名 | 作用 | 参数 schema 概要 | 备注 |
| --- | --- | --- | --- |
| `ADD` | 新增一条普通 memory record | `text` 必填；可带 `target_layer`、`metadata`、`unit_id`、`timestamp` | 若未传 `target_layer`，可回退到模块构造时的默认目标层。 |
| `UPDATE` | 更新一条已有 record | `record_id` 必填；可带 `text`、`metadata_patch` | 按 `record_id` 全局查找。 |
| `DELETE` | 删除一条已有 record | `record_id` 必填；可带 `reason` | 调用 `MemoryStore.delete_record(...)`。 |
| `GRAPH_ADD` | 新增一条 graph record | 与 `ADD` 相同 | 目标层必须是 `Graph`；会规范化 `metadata["graph"]`。 |
| `GRAPH_UPDATE` | 更新一条 graph record | 与 `UPDATE` 相同 | 记录所在层必须是 `Graph`；会规范化 graph metadata。 |
| `GRAPH_DELETE` | 删除一条 graph record | 与 `DELETE` 相同 | 删除后会清理同层 dangling graph links。 |
| `GRAPH_ADD_LINK` | 给一条 graph record 追加 outgoing links | `record_id`、`links` 必填 | 只改 `metadata["graph"]["links"]`，不会改节点文本、entities、triples。 |
| `GRAPH_UPDATE_LINK` | 全量替换一条 graph record 的 outgoing links | `record_id`、`links` 必填 | 只改边；当前语义是“replace whole outgoing link list”。 |
| `GRAPH_DELETE_LINK` | 从一条 graph record 删除部分 outgoing links | `record_id`、`links` 必填 | 只删指定边；未命中的 link 会被忽略。 |

### 默认工具的行为约定

- 所有 built-in tool 都要求 arguments 是 JSON object。
- 参数校验基于每个工具自己的 `parameters_json_schema`，并执行最小 JSON type 检查。
- 三个 `GRAPH_*_LINK` built-in 只允许修改 `metadata["graph"]["links"]` / `link_count` / `last_linked_at`，不改节点内容，也不自动维护反向边。
- 执行结果统一落到 trace：
  - `tool_calls`
  - `effects`
  - `written_record_ids`
  - `updated_record_ids`
  - `deleted_record_ids`
- built-in tool 写出的记录会自动补充 `metadata["llm_tool"]`，记录动作类型、来源模块和模块 slot。

### 自定义工具协议

自定义工具通过 `WriteToolSpec` 暴露给 `LLMFunctionCallOrganization` / `LLMFunctionCallEvolution`。

公开类型已在顶层导出：

```python
from memprimitive import WriteToolSpec, WriteToolCallContext, WriteToolResult
```

协议如下：

```python
@dataclass(slots=True, frozen=True)
class WriteToolSpec:
    name: str
    description: str
    parameters_json_schema: dict[str, Any]
    executor: Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]
```

```python
@dataclass(slots=True)
class WriteToolCallContext:
    packet: Packet
    store: MemoryStore
    module_slot: Literal["organization", "memory_evolution"]
    default_target_layer: str | None
    visible_records: list[MemoryRecord]
```

```python
@dataclass(slots=True)
class WriteToolResult:
    effects: list[dict[str, Any]]
    store: MemoryStore
```

### 自定义工具实现要求

- `name` 必须是非空字符串，且在当前模块的 `tools=[...]` 中不能与其他工具重名。
- `description` 必须是非空字符串，供模型理解用途。
- `parameters_json_schema` 必须是 object schema；当前运行时只支持顶层 JSON object 参数。
- `executor(context, arguments)` 负责真正修改 `context.store`，然后返回 `WriteToolResult`。
- `effects` 建议使用统一字段：
  - `action`
  - `record_id`
  - `layer`
  - `status`
  - 以及你的自定义补充字段

### 自定义工具最小示例

下面是当前仓库演示脚本使用的自定义删除整层工具风格：

```python
from memprimitive import WriteToolCallContext, WriteToolResult, WriteToolSpec

def build_delete_layer_tool() -> WriteToolSpec:
    def _executor(context: WriteToolCallContext, arguments: dict[str, object]) -> WriteToolResult:
        target_layer = str(arguments.get("target_layer", "")).strip()
        if not target_layer:
            raise ValueError("target_layer is required")

        effects = []
        for record in list(context.store.iter_records(target_layer)):
            removed = context.store.delete_record(target_layer, record.record_id)
            effects.append(
                {
                    "action": "delete",
                    "record_id": removed.record_id,
                    "layer": removed.layer,
                    "status": "applied",
                }
            )
        return WriteToolResult(effects=effects, store=context.store)

    return WriteToolSpec(
        name="DELETE_LAYER",
        description="Delete every record in one target layer.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "target_layer": {"type": "string"},
            },
            "required": ["target_layer"],
            "additionalProperties": False,
        },
        executor=_executor,
    )
```

## Template Prompt 写法

### 总览

当前模板系统的统一抽象是 `PromptPlan`。公开构造函数有两个：

```python
from memprimitive.utils._template import text_prompt, structured_prompt
```

- `text_prompt(...)`
  - 适合普通字符串 prompt。
- `structured_prompt(...)`
  - 适合“分 block 组织”的结构化 prompt / readout 模板。

任何接受 `PromptPlan | str` 的模块，都可以直接传原始字符串；字符串内部如果包含 `{{ ... }}`，会按模板表达式渲染。

### `text_prompt(...)`

最简单的模板写法：

```python
from memprimitive.utils._template import text_prompt

prompt = text_prompt(
    "Summarize this unit for long-term memory:\n"
    "unit_id={{ unit.unit_id }}\n"
    "text={{ unit.text }}\n"
    "session={{ runtime.session_id }}"
)
```

常用参数：

- `template: str`
- `context_builder=None`
- `recall_plan=None`
- `recall_query_builder=None`
- `missing_value=""`
- `metadata_mode="prompt"`
- `sub_recall_pipeline=None`

### `structured_prompt(...)`

结构化模板适合写成“块式 prompt”，便于 API 文档风格地组织任务、上下文、可用工具、候选记录。

基本写法：

```python
from memprimitive.utils._template import structured_prompt

prompt = structured_prompt(
    {
        "blocks": [
            {
                "id": "task",
                "title": "Task",
                "template": "Decide whether to write the current unit.",
            },
            {
                "id": "unit",
                "title": "Incoming Unit",
                "template": "unit_id={{ unit.unit_id }}\ntext={{ unit.text }}",
            },
        ]
    }
)
```

当前结构化 block 常用字段：

- `id`
- `title`
- `template`
- `condition`
- `repeat_over`
- `item_template`
- `separator`
- `children`

其中：

- `condition`
  - 为真时才渲染该 block。
- `repeat_over`
  - 对列表上下文做循环渲染。
- `item_template`
  - 循环项模板，循环变量名固定为 `item`。

### 模板上下文

模板渲染时会自动注入一组稳定上下文。实际可用字段会因模块位置不同而增加，但基础字段包括：

- `query`
- `runtime`
- `trace`
- `recalled_prompt`

常见模块还会额外注入：

- `LLMRepresentation`
  - 常见会有 `unit`
- `LLMFunctionCallOrganization`
  - 常见会有 `unit`、`tools`、`default_target_layer`、`visible_records`
- `LLMFunctionCallEvolution`
  - 常见会有 `selected_records`、`source_layer`、`tools`
- `TemplateReadout`
  - 常见会有 `retrieved` 以及投影后的 record/group/view 信息

### 模板表达式

模板表达式语法：

```text
{{ path.to.value }}
{{ some_list | length }}
{{ retrieved.items | join_text }}
{{ visible_records | topk(5) }}
```

当前内置 filter：

- `default(...)`
- `join_text`
- `join(...)`
- `length`
- `first`
- `topk(...)`
- `sort_by(field_name, reverse=False)`

示例：

```python
"{{ selected_records | length }}"
"{{ retrieved.items | topk(3) | join_text }}"
"{{ visible_records | sort_by('timestamp', true) | first }}"
```

### 用模板写 tool-call prompt

推荐把“任务说明”“可用工具”“精确 record_id 映射”拆成不同 block。当前仓库示例采用的写法类似：

```python
prompt = structured_prompt(
    {
        "blocks": [
            {
                "id": "task",
                "title": "Task",
                "template": (
                    "If the memory should be saved, call ADD exactly once."
                ),
            },
            {
                "id": "existing_records",
                "title": "Existing Records With Exact IDs",
                "condition": "visible_records | length",
                "repeat_over": "visible_records",
                "item_template": "- record_id={{ item.record_id }} | layer={{ item.layer }} | text={{ item.text }}",
                "separator": "\n",
            },
        ]
    }
)
```

这个模式的关键点是显式向模型展示 `record_id -> text` 映射，避免模型“猜”记录编号。

### 模板子召回（sub recall）

`PromptPlan` 还支持在渲染主 prompt 前，先执行一次或多次子 recall，再把结果注入模板变量。

核心参数：

- `recall_plan`
- `recall_query_builder`
- `sub_recall_pipeline`
- `labeled_recall_plans`
- `labeled_sub_recall_pipelines`
- `labeled_recall_query_builders`

示意：

```python
from memprimitive.utils._template import text_prompt

prompt = text_prompt(
    "Known related memory:\n{{ recalled_prompt }}\n\nNow process:\n{{ unit.text }}",
    recall_plan=text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
    recall_query_builder=lambda packet, store, context: context["unit"]["text"],
    sub_recall_pipeline=recall_pipeline,
)
```

说明：

- 子 recall 始终对当前 live store 执行。
- 没有 recall 结果时，`{{ recalled_prompt }}` 会退化为空字符串。
- 相关渲染信息会记录进 prompt trace。

多 label 版本示意：

```python
from memprimitive.utils._template import text_prompt

prompt = text_prompt(
    "Profile:\n{{ profile }}\n\nHistory:\n{{ history }}\n\nNow process:\n{{ unit.text }}",
    recall_query_builder=lambda packet, store, context: context["unit"]["text"],
    labeled_recall_plans={
        "profile": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
        "history": text_prompt("{{ retrieved.items | join_text }}", metadata_mode="readout"),
    },
    labeled_sub_recall_pipelines={
        "profile": profile_recall_pipeline,
        "history": history_recall_pipeline,
    },
)
```

补充说明：

- 旧的单路 `{{ recalled_prompt }}`、`recall_plan`、`sub_recall_pipeline` 仍然完全兼容。
- 多 label 模板变量直接使用 label 名本身，例如 `{{ profile }}`、`{{ history }}`。
- 多 label 默认复用全局 `recall_query_builder`；若需要单独覆盖，可传 `labeled_recall_query_builders={"history": ...}`。
- 每个 label 的 recall 结果与 trace 会分别记录在 metadata / prompt trace 的 `labeled_recalled_prompts` 和 `labeled_recall_prompts` 中。

## 推荐示例文件

如果需要直接看可运行范式，优先参考：

- `memprimitive/example/demonstration/llm_function_call_tools.py`
- `memprimitive/example/demonstration/llm_function_call_graph_tools.py`
- `memprimitive/example/demonstration/structured_template_readout.py`
- `memprimitive/example/demonstration/recalled_prompt_template.py`

## 备注

- 本文档描述的是“当前公开实现面”，不是历史设计稿。
- 若 baseline module、PromptPlan、tool-call 协议继续演进，本文档应与代码同步更新。
