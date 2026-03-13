# Memory Primitive 搜索空间

本文档系统枚举每个 primitive slot 的所有实现可能性，定义搜索空间的边界，并分析模块间的兼容性约束。

---

## 0. 搜索空间总览

### 维度结构

```
搜索空间 = StoreTopo × UnitFormation × Representation × WriteTrigger 
         × Organization × Update × Compression × Retrieval × Readout 
         × Maintenance × CompressTrigger × MaintainTrigger
```

每个维度是一个**离散选择集**（实现变体），每个变体内部可能有**离散/连续超参**。

### Store 拓扑选择（结构维度）

Store 拓扑决定了整个系统的骨架，是最高层的结构性选择。

| ID | 拓扑 | Store 构成 | 典型场景 |
|----|------|-----------|----------|
| ST-1 | **Single-Flat** | 1 个 Flat store | Reflexion, 极简 memory |
| ST-2 | **Dual-Store** | working + long_term | RAISE, MemoryBank |
| ST-3 | **Triple-Store** | working + episodic + semantic | MemGPT (main + archival + recall) |
| ST-4 | **Quad-Store** | working + episodic + semantic + profile | 完整认知架构 |
| ST-5 | **Graph-Centric** | 1 个 Graph store | A-MEM |
| ST-6 | **Hybrid-Graph** | Graph + Flat working buffer | 图增强架构 |
| ST-7 | **Skill-Store** | specialized store for code/skills | Voyager |

---

## A. Unit Formation — 记忆单元形成

**核心问题**：原始输入如何切分/转化为 memory units？

| ID | 实现 | 描述 | 输出 unit_type | 关键参数 |
|----|------|------|---------------|----------|
| UF-1 | **PassThrough** | 原始 observation 直接作为一个 unit | raw | — |
| UF-2 | **TurnLevel** | 每轮对话 (user+assistant) 作为一个 unit | turn | include_system=bool |
| UF-3 | **MessageLevel** | 每条消息独立成 unit | message | — |
| UF-4 | **EventExtractor** | 用 LLM/规则从文本中抽取离散事件 | event | prompt_template, min_confidence |
| UF-5 | **FactExtractor** | 抽取事实性命题 (who/what/when/where) | fact | schema, extraction_model |
| UF-6 | **EntityStateExtractor** | 抽取实体及其当前状态 | entity_state | entity_types, attribute_schema |
| UF-7 | **TripleExtractor** | 抽取 (subject, predicate, object) 三元组 | triple | ontology, max_triples |
| UF-8 | **KeyValueExtractor** | 抽取 key-value 对（偏好、属性等） | kv_pair | key_schema |
| UF-9 | **ChunkSplitter** | 按固定大小或语义边界分块 | chunk | chunk_size, overlap, strategy={fixed, semantic} |
| UF-10 | **SessionSummarizer** | 整个 session 压缩为一个 summary unit | summary | max_length |
| UF-11 | **TaskReflectionExtractor** | 从任务轨迹中提取反思 | reflection | include_trajectory, include_outcome |
| UF-12 | **SkillExtractor** | 提取可复用代码/技能 | skill | include_code, include_description |
| UF-13 | **ThoughtExtractor** | 提取内部推理步骤 | thought | granularity={step, chain} |
| UF-14 | **MultiGranularity** | 同时在多个粒度抽取（组合多个 UF） | mixed | sub_extractors: list[UF-ID] |
| UF-15 | **LLMFreeform** | LLM 自由决定切分方式和粒度 | mixed | model, instruction |

### 变异轴

- **粒度**：raw → turn → message → event/fact → triple/kv
- **结构化程度**：自由文本 ↔ 严格 schema
- **抽取方式**：规则 / 模型 / LLM

---

## B. Representation — 表示编码

**核心问题**：unit 以什么形式存储和索引？

| ID | 实现 | 描述 | 产出字段 | 关键参数 |
|----|------|------|---------|----------|
| Rep-1 | **RawText** | 仅保留原始文本 | content | — |
| Rep-2 | **TextWithEmbedding** | 文本 + dense embedding | content, embedding | model, dim |
| Rep-3 | **StructuredTriple** | 转为 (s, p, o) 三元组 | content(triple) | schema |
| Rep-4 | **KeyValuePair** | 转为 key-value 格式 | content(kv) | — |
| Rep-5 | **FrameSlotFilling** | 填入预定义的 frame schema | content(frame) | frame_schema |
| Rep-6 | **HybridTextEmbed** | 文本 + embedding + entity tags | content, embedding, entities | model, entity_linker |
| Rep-7 | **HybridStructured** | 文本 + embedding + triple + tags + entities | all fields | model, schema, entity_linker |
| Rep-8 | **CodeWithDescription** | 代码 + 自然语言描述 + embedding | content(code), description, embedding | model |
| Rep-9 | **LatentVector** | 仅保留 dense vector，不保留文本 | embedding | model, dim |
| Rep-10 | **SparseVector** | TF-IDF / BM25 sparse representation | sparse_embedding | vocab_size |
| Rep-11 | **DualEncoder** | 同时生成 dense + sparse 表示 | embedding, sparse_embedding | dense_model, sparse_method |

### 变异轴

- **模态**：text only / embedding only / hybrid
- **结构化**：自由文本 ↔ schema-bound
- **密度**：dense / sparse / dual

### 关键约束

- `Rep-9` (LatentVector) 不兼容需要文本内容的下游模块
- `Rep-3/4/5` 要求上游 UF 产出结构化内容，或本模块自带转换能力
- 向量检索 (Ret-1) 要求 `embedding` 字段非空 → 排除 `Rep-1, Rep-3, Rep-4, Rep-5`

---

## C. Write Trigger — 写入决策

**核心问题**：什么时候该把 unit 写入 store？

| ID | 实现 | 描述 | 决策方式 | 关键参数 |
|----|------|------|---------|----------|
| WT-1 | **Always** | 所有 unit 无条件写入 | deterministic | — |
| WT-2 | **Never** | 从不写入（只读 memory） | deterministic | — |
| WT-3 | **ImportanceThreshold** | importance score > θ 时写入 | score-based | threshold, scorer |
| WT-4 | **NoveltyThreshold** | 与已有 memory 差异 > θ 时写入 | score-based | threshold, similarity_metric |
| WT-5 | **ImportanceAndNovelty** | importance × novelty 联合评分 | score-based | α, β, threshold |
| WT-6 | **LLMJudge** | LLM 判断是否值得记忆 | llm-based | model, criteria, prompt |
| WT-7 | **AgentToolCall** | Agent 显式调用写入工具时才写 | agent-controlled | tool_names |
| WT-8 | **RuleBased** | 预定义规则（含特定 entity/关键词/事件类型） | rule-based | rules: list[Predicate] |
| WT-9 | **OnEvent** | 特定事件发生时写入 | event-triggered | event_types |
| WT-10 | **SurpriseGated** | 基于 prediction error / surprise 评分 | score-based | surprise_model, threshold |
| WT-11 | **SampledWrite** | 按概率采样写入（用于大量流式输入） | probabilistic | sample_rate |
| WT-12 | **DuplicateAwareWrite** | 检测重复后决定（去重写入） | dedup-based | similarity_threshold |

### 变异轴

- **控制源**：系统自动 / agent 主动 / 事件驱动
- **决策模型**：确定性 / 评分阈值 / LLM / 概率
- **信息利用**：仅看 unit / 对比 store 已有内容

---

## D. Organization — 关系归组与放置规划

**核心问题**：unit 应该放在哪里？与已有 memory 建立什么关系？

| ID | 实现 | 描述 | 产出 | 关键参数 |
|----|------|------|------|----------|
| Org-1 | **FlatAppend** | 无任何关系，直接追加到默认 store | target_store | default_store |
| Org-2 | **TemporalAppend** | 追加并建立时间链接 | target_store, temporal_links | — |
| Org-3 | **EntityLinked** | 按 entity 建立关联链接 | target_store, entity_links | link_type |
| Org-4 | **TemporalAndEntity** | 同时建立时间 + entity 链接 | target_store, both_links | — |
| Org-5 | **GraphPlacement** | 在图中创建节点和有类型的边 | node, edges | edge_types, connection_method={rule, embedding_sim, llm_inferred} |
| Org-6 | **HierarchicalPlacement** | 放入层级结构的合适层 | target_level, parent_link | hierarchy_depth, placement_policy |
| Org-7 | **DualStoreRouter** | 根据 unit type 路由到不同 store | target_store | route_rules: dict |
| Org-8 | **AgentExplicitTarget** | Agent 显式指定目标 store | target_store | tool_to_store: dict |
| Org-9 | **SimilarityCluster** | 聚类到最近的已有 cluster | cluster_id, cluster_link | similarity_threshold, max_clusters |
| Org-10 | **TagBasedPlacement** | 按 tag 分区放置 | target_partition | tag_field |

### 变异轴

- **关系类型**：无 / 时间 / 实体 / 因果 / 相似性 / 层级
- **路由策略**：固定 / 规则 / agent 决策 / 自动推断
- **结构复杂度**：flat → indexed → graph → hierarchical

### 关键约束

- `Org-5` (GraphPlacement) 要求 store 类型为 Graph 且有 `graph` index
- `Org-3/4` 要求 unit 有 `entities` 字段 → 需要 UF 产出 entity 或 Rep 补充 entity
- `Org-7` 要求 store 拓扑有多个 store

---

## E. Update — 写入执行

**核心问题**：unit 如何物理写入 store？遇到冲突怎么办？

| ID | 实现 | 描述 | 对 store 的副作用 | 关键参数 |
|----|------|------|-----------------|----------|
| Upd-1 | **Append** | 仅追加新 unit | add | — |
| Upd-2 | **Replace** | 替换匹配条件的旧 unit | add + delete | match_key, match_strategy |
| Upd-3 | **MergeByEntity** | 按 entity 合并（更新属性，保留历史） | modify | merge_strategy={latest_wins, aggregate, llm_merge} |
| Upd-4 | **UpsertByKey** | 有则更新，无则插入 | add or modify | key_field, similarity_threshold |
| Upd-5 | **DeltaUpdate** | 只记录增量变化 | add(delta) | diff_method |
| Upd-6 | **LLMRewrite** | LLM 重写已有记忆，融入新信息 | modify | model, rewrite_prompt |
| Upd-7 | **ConflictResolution** | 检测冲突并解决后写入 | add + modify | conflict_detector, resolution={recency, confidence, llm_judge} |
| Upd-8 | **AppendWithLinkUpdate** | 追加新 unit 并更新相关 unit 的关系链接 | add + modify(links) | — |
| Upd-9 | **VersionedAppend** | 追加并保留历史版本链 | add + version_link | max_versions |

### 变异轴

- **写入模式**：append-only / upsert / replace / merge
- **冲突处理**：忽略 / 覆盖 / 合并 / LLM 裁决
- **历史保留**：不保留 / delta / 版本链

### 关键约束

- `Upd-3` (MergeByEntity) 要求 unit 有 `entities` 字段
- `Upd-5` (DeltaUpdate) 要求能对齐到已有 unit → 需要 key_field 或 entity
- `Upd-6` (LLMRewrite) 增加 LLM 调用开销，不适合高频写入场景

---

## F. Compression — 抽象压缩

**核心问题**：如何从已有记忆中生成更高层抽象？

| ID | 实现 | 描述 | 产出 | 关键参数 |
|----|------|------|------|----------|
| Comp-1 | **None** | 不做压缩 | — | — |
| Comp-2 | **Summarization** | LLM 摘要一批 memory 为一条 summary | summary unit | method={extractive, abstractive}, batch_size |
| Comp-3 | **RecursiveSummarization** | 层级式递归摘要（摘要的摘要） | hierarchy of summaries | max_depth, summary_ratio |
| Comp-4 | **LLMReflection** | 从 observations 中归纳高层 insight | reflection unit | prompt_template, batch_size |
| Comp-5 | **EntityProfileUpdate** | 聚合同一 entity 的多条记忆，更新 profile | profile unit | aggregation_method={latest, majority, llm_merge} |
| Comp-6 | **ConceptExtraction** | 从多条记忆中提取抽象概念 | concept unit | concept_model, min_support |
| Comp-7 | **PrototypeFormation** | 形成原型记忆（类似于 schema/stereotype） | prototype unit | clustering_method, min_instances |
| Comp-8 | **ProgressiveSummarization** | 增量式更新已有摘要（每次来新内容就更新） | updated summary | update_strategy={append_then_rewrite, running_summary} |
| Comp-9 | **MapReduce** | 分片摘要后合并 | summary unit | map_prompt, reduce_prompt, chunk_size |
| Comp-10 | **Distillation** | 将多条 memory 蒸馏为一条高信息密度 unit | distilled unit | target_ratio, quality_metric |

### 变异轴

- **抽象层次**：摘要(同层) / 反思(升层) / 概念(跨记忆)
- **触发模式**：周期 / 事件 / 预算溢出 / 显式
- **更新模式**：批量重写 / 增量追加
- **处理范围**：全局 / 按 entity / 按时间窗口

---

## G. Retrieval — 检索策略

**核心问题**：给定 query，如何从 store 中找到相关 memory？

| ID | 实现 | 描述 | 核心机制 | 关键参数 |
|----|------|------|---------|----------|
| Ret-1 | **Similarity** | Dense embedding 向量相似度 | vector search | top_k, metric={cosine, dot, l2} |
| Ret-2 | **BM25** | 关键词/稀疏向量检索 | sparse search | top_k |
| Ret-3 | **Recency** | 按时间倒序取最新 | temporal | window_size |
| Ret-4 | **RecencyDecay** | 时间衰减加权 | temporal + score | decay_rate, half_life |
| Ret-5 | **ImportanceRanked** | 按 importance score 排序 | score-based | top_k |
| Ret-6 | **EntityLookup** | 按 entity 精确匹配 | symbolic | max_per_entity |
| Ret-7 | **GraphHop** | 从 seed 出发多跳图遍历 | graph traversal | max_hops, top_per_hop, edge_filter |
| Ret-8 | **FullContext** | 将所有 memory 注入 context | brute-force | — |
| Ret-9 | **TemporalWindow** | 取指定时间窗口内的记忆 | temporal range | window_start, window_end |
| Ret-10 | **WeightedMultiSignal** | 多信号加权组合 | hybrid | signals: list, weights: list |
| Ret-11 | **TwoStageRecallRerank** | 先粗召回、再精排 | cascade | recall_method, rerank_method, recall_k, final_k |
| Ret-12 | **LLMRerank** | 候选集经 LLM 重排 | llm-based | model, rerank_prompt, candidate_k, final_k |
| Ret-13 | **LLMControlled** | LLM 自主决定检索什么、怎么检索 | llm-agent | model, available_tools |
| Ret-14 | **AgentToolCall** | Agent 通过工具调用触发检索 | agent-controlled | tool_names, backend |
| Ret-15 | **HierarchicalRetrieval** | 先检索高层摘要，再展开到细节 | top-down | levels, expand_top_k |
| Ret-16 | **DiversityAware** | 在相关性基础上增加多样性约束 | diversity | diversity_method={MMR, DPP}, lambda |
| Ret-17 | **ContextualBandit** | 基于历史 retrieval 效果的在线学习 | learned | exploration_rate, feature_model |

### 变异轴

- **信号源**：similarity / recency / importance / entity / graph / learned
- **阶段数**：单阶段 / 两阶段(recall+rerank) / 多阶段
- **控制方式**：自动 / agent 主动 / LLM 控制
- **多样性**：无约束 / MMR / DPP

### 关键约束

- `Ret-1` (Similarity) 要求 store 有 `vector` index 且 unit 有 `embedding`
- `Ret-7` (GraphHop) 要求 store 有 `graph` index 且 `Org` 产出了 graph edges
- `Ret-6` (EntityLookup) 要求 store 有 `entity` index 且 unit 有 `entities`
- `Ret-8` (FullContext) 仅适用于小规模 memory（否则超出 context window）
- `Ret-2` (BM25) 要求 store 有 `keyword` index 或 unit 有文本内容

---

## H. Readout — 输出格式化

**核心问题**：检索到的 memory 如何转化为 agent 可消费的输入？

| ID | 实现 | 描述 | 输出格式 | 关键参数 |
|----|------|------|---------|----------|
| Read-1 | **FlatConcat** | 按序拼接检索结果为文本 | text block | separator, max_tokens, format |
| Read-2 | **StructuredSections** | 按类型/来源分区组织 | sectioned text | sections: list[str] |
| Read-3 | **SummarizedReadout** | 先摘要检索结果再输出 | summary text | model, max_length |
| Read-4 | **TemplatedPrompt** | 填入预定义 prompt 模板 | filled template | template |
| Read-5 | **PrependToContext** | 直接 prepend 到对话 context | prefixed context | prefix |
| Read-6 | **CodeInjection** | 以函数/工具定义形式注入 | code/tool defs | format={function_def, docstring} |
| Read-7 | **SelectiveByRole** | 根据 agent 角色/阶段选择性注入 | filtered output | role_rules: dict |
| Read-8 | **RankedList** | 带分数的有序列表 | ranked list | include_scores, format |
| Read-9 | **JSONStructured** | 以 JSON 结构化格式输出 | json | schema |
| Read-10 | **LatentInjection** | 以 hidden state / soft prompt 注入 | latent tensor | injection_layer |

### 变异轴

- **格式**：自由文本 / 分区 / 模板 / 结构化 / latent
- **压缩度**：全量 / 摘要后注入 / 选择性注入
- **注入方式**：prompt 内 / 工具定义 / 参数注入

---

## I. Maintenance — 维护管理

**核心问题**：如何管理 memory 的生命周期、容量和质量？

| ID | 实现 | 描述 | 副作用 | 关键参数 |
|----|------|------|--------|----------|
| Maint-1 | **None** | 不做维护 | — | — |
| Maint-2 | **FIFO** | 最旧的先淘汰 | delete | budget |
| Maint-3 | **LRU** | 最久未访问的先淘汰 | delete | budget |
| Maint-4 | **ImportanceDecay** | importance 随时间衰减，低于阈值删除 | modify + delete | decay_rate, prune_threshold |
| Maint-5 | **EbbinghausForgetting** | 模拟遗忘曲线 | modify + delete | retention_model, rehearsal_boost |
| Maint-6 | **SlidingWindow** | 只保留最近 N 条 | delete | window_size |
| Maint-7 | **TokenBudgetEnforcement** | 按 token 预算裁剪 | delete | max_tokens, eviction_policy |
| Maint-8 | **Deduplication** | 检测并合并重复记忆 | delete + modify | similarity_threshold, merge_strategy |
| Maint-9 | **ConflictCleanup** | 检测矛盾记忆并解决 | delete or modify | conflict_detector, resolution |
| Maint-10 | **ArchivalMovement** | 将低活跃记忆从 hot store 移到 cold store | move | activity_threshold, target_store |
| Maint-11 | **ContextWindowManager** | 管理 working memory 的 token 预算 | summarize + evict | strategy={summarize_and_evict, truncate}, budget |
| Maint-12 | **PeriodicConsolidation** | 定期合并碎片化记忆 | merge + delete | consolidation_method |
| Maint-13 | **RelevanceReview** | LLM 定期审查并删除不再相关的记忆 | delete | model, review_prompt |

### 变异轴

- **淘汰策略**：FIFO / LRU / 评分衰减 / 预算截断
- **质量管理**：去重 / 冲突解决 / 相关性审查
- **空间管理**：token 预算 / unit 数量限制 / 分层存储

---

## J. Trigger — 触发条件（Compression 和 Maintenance 共享）

| ID | 实现 | 描述 | 关键参数 |
|----|------|------|----------|
| Trig-1 | **Never** | 从不触发 | — |
| Trig-2 | **Periodic** | 每 N 步触发一次 | every: int |
| Trig-3 | **AfterWrite** | 每次写入后触发 | — |
| Trig-4 | **OnEvent** | 特定事件发生时触发 | event_type |
| Trig-5 | **BudgetExceeded** | 预算使用超过阈值时触发 | threshold: float |
| Trig-6 | **CountExceeded** | unit 数量超过阈值时触发 | max_count: int |
| Trig-7 | **Conditional** | 自定义条件表达式 | predicate |

---

## 1. 兼容性约束矩阵

以下约束在搜索时用于剪枝非法组合。

### 1.1 Store → Module 约束（store 拓扑限制模块选择）

| 约束 | 含义 |
|------|------|
| `requires(Ret-1, store.index contains vector)` | Similarity 检索要求向量索引 |
| `requires(Ret-2, store.index contains keyword)` | BM25 检索要求关键词索引 |
| `requires(Ret-7, store.index contains graph)` | GraphHop 要求图索引 |
| `requires(Ret-6, store.index contains entity)` | EntityLookup 要求实体索引 |
| `requires(Org-5, store.type == Graph)` | GraphPlacement 要求图类型 store |
| `requires(Org-7, len(stores) > 1)` | DualStoreRouter 要求多 store |
| `requires(Maint-10, len(stores) > 1)` | ArchivalMovement 要求多 store |

### 1.2 Module → Module 约束（上游输出限制下游选择）

| 约束 | 含义 |
|------|------|
| `requires(Ret-1, Rep-* produces embedding)` | 向量检索要求有 embedding |
| `requires(Org-3, UF/Rep produces entities)` | EntityLinked 要求有实体 |
| `requires(Upd-3, UF/Rep produces entities)` | MergeByEntity 要求有实体 |
| `requires(Upd-5, UF produces alignable units)` | DeltaUpdate 要求可对齐 |
| `requires(Ret-7, Org-5 produces graph edges)` | GraphHop 要求有图边 |
| `requires(Read-6, UF-12 produces code)` | CodeInjection 要求代码 unit |
| `requires(Ret-15, Comp produces hierarchy)` | HierarchicalRetrieval 要求层级摘要 |

### 1.3 语义耦合约束（强烈推荐的组合模式）

| 约束 | 含义 |
|------|------|
| `co_requires(UF-5/6/7, Upd-3/4)` | 结构化 unit 与 merge/upsert 强关联 |
| `co_requires(UF-2/3, Upd-1)` | 对话级 unit 与 append 强关联 |
| `co_requires(Comp-5, UF-6)` | EntityProfileUpdate 与 EntityStateExtractor 强关联 |
| `co_requires(Org-5, Ret-7)` | GraphPlacement 与 GraphHop 强关联 |
| `co_requires(Rep-8, Read-6)` | CodeWithDescription 与 CodeInjection 强关联 |

### 1.4 互斥约束

| 约束 | 含义 |
|------|------|
| `incompatible(WT-2, Upd-*)` | Never write 与任何 update 互斥 |
| `incompatible(Ret-8, Maint-1)` | FullContext 与无维护在大 store 下互斥 |
| `incompatible(Rep-9, Read-1)` | LatentVector 与 FlatConcat 互斥（无文本可拼接） |

---

## 2. 搜索空间规模估算

### 2.1 各 slot 候选数

| Slot | 候选数 | 含组合算子后 |
|------|--------|------------|
| Store Topo | 7 | 7 |
| UnitFormation | 15 | 15 + C(15,2) for MultiGranularity ≈ 120 |
| Representation | 11 | 11 |
| WriteTrigger | 12 | 12 |
| Organization | 10 | 10 |
| Update | 9 | 9 + Dispatch 组合 ≈ 30 |
| Compression | 10 | 10 |
| Retrieval | 17 | 17 + Compose/Cascade 组合 ≈ 200 |
| Readout | 10 | 10 |
| Maintenance | 13 | 13 + Compose ≈ 50 |
| CompressTrigger | 7 | 7 |
| MaintainTrigger | 7 | 7 |

### 2.2 空间上界

不含组合算子：`7 × 15 × 11 × 12 × 10 × 9 × 10 × 17 × 10 × 13 × 7 × 7 ≈ 1.2 × 10^9`

含组合算子但不含参数：`7 × 120 × 11 × 12 × 10 × 30 × 10 × 200 × 10 × 50 × 7 × 7 ≈ 4.5 × 10^12`

### 2.3 约束剪枝后的有效空间

根据兼容性约束估算约 5-10% 的组合是合法的：

- **不含组合算子**：~10^8 有效配置
- **含组合算子**：~10^11 有效配置

### 2.4 可行的搜索策略

| 空间规模 | 推荐策略 |
|---------|---------|
| 10^4 以下 | 穷举 (grid search) |
| 10^4 - 10^6 | 随机搜索 + 约束剪枝 |
| 10^6 - 10^9 | 进化算法 / Bayesian optimization |
| 10^9 以上 | 分层搜索：先搜索 structural level，固定后搜索 modular，最后调参 |

推荐策略：**分层搜索**

```
Phase 1: 固定 StoreTopo + Trigger，搜索 9 个 modular slot（基础实现，不含组合算子）
          空间 ≈ 15 × 11 × 12 × 10 × 9 × 10 × 17 × 10 × 13 ≈ 3 × 10^8
          → 进化算法，population=100, generations=200

Phase 2: 选取 top-10 配置，对 Retrieval 和 UnitFormation 展开组合算子搜索
          空间 ≈ 10 × 200 × 120 ≈ 2.4 × 10^5 per config
          → 随机搜索 + 精细调参

Phase 3: 参数调优（每个配置内部的连续参数）
          → Bayesian optimization / grid search
```

---

## 3. 已有工作的 Primitive 分解

将 DSLgrammar.md 中 8 个经典系统分解为 Primitive ID：

| System | UF | Rep | WT | Org | Upd | Comp | Ret | Read | Maint |
|--------|-----|-----|-----|------|------|-------|------|-------|--------|
| Generative Agents | UF-1 | Rep-2 | WT-1 | Org-2 | Upd-1 | Comp-4 | Compose(Ret-1,4,5) | Read-1 | Maint-1 |
| MemGPT | UF-2 | Rep-2 | WT-7 | Org-8 | Upd-1 | Comp-3 | Ret-14→Ret-1 | Read-2 | Maint-11 |
| Reflexion | UF-11 | Rep-1 | WT-9 | Org-1 | Upd-1 | Comp-1 | Ret-8 | Read-1 | Maint-6 |
| Voyager | UF-12 | Rep-8 | WT-9 | Org-10 | Upd-4 | Comp-1 | Ret-1 | Read-6 | Maint-1 |
| MemoryBank | Compose(UF-2,5) | Rep-2 | WT-1 | Org-7 | Dispatch(Upd-1,3) | Comp-2 | Compose(Ret-1,4) | Read-1 | Maint-5 |
| SCM | UF-5 | Rep-3 | WT-6 | Org-3 | Upd-4 | Comp-5 | Ret-13→Ret-6 | Read-2 | Maint-9 |
| A-MEM | UF-15 | Rep-7 | WT-1 | Org-5 | Upd-8 | Comp-1 | Compose(Ret-1,7) | Read-2 | Maint-1 |
| TiM | UF-13 | Rep-2 | WT-9 | Org-2 | Upd-1 | Comp-8 | Compose(Ret-3,1) | Read-5 | Maint-6 |

### 从中可观察到的初步 Motif

**Motif 1: Embed-and-Retrieve（出现 7/8 次）**
- `Rep-2` (TextWithEmbedding) + `Ret-1` (Similarity) 或其组合
- 几乎所有系统都需要 embedding 作为基础检索手段

**Motif 2: Append-Dominant（出现 6/8 次）**
- `Upd-1` (Append) 是最常见的写入方式
- 只有涉及结构化 entity 更新的系统才用 merge/upsert

**Motif 3: Selective-Write 分化**
- 简单系统用 `WT-1` (Always) 或 `WT-9` (OnEvent)
- 复杂系统用 `WT-6` (LLMJudge) 或 `WT-7` (AgentToolCall)
- 这是区分 passive vs active memory 的关键维度

**Motif 4: Compression 可选**
- 4/8 系统不做 compression（`Comp-1`）
- 需要 compression 的系统倾向于不同层次：摘要 / 反思 / profile 更新

**Motif 5: Maintenance 两极化**
- 要么不做（`Maint-1`），要么做 context window 管理 / 滑动窗口
- 复杂的衰减/遗忘机制（如 `Maint-5`）在实际系统中较少见

---

## 4. 潜在的未探索区域（搜索的价值所在）

通过对已有工作的分解，可以识别出搜索空间中**尚未被探索的组合**：

| 未探索组合 | 假设 | 预期价值 |
|-----------|------|---------|
| `UF-7` (Triple) + `Ret-7` (GraphHop) + `Maint-5` (Ebbinghaus) | 结构化图记忆 + 认知遗忘 | 长期对话中的知识演化 |
| `WT-10` (Surprise) + `Comp-4` (Reflection) | 只记意外 + 反思 | 高效学习型 agent |
| `Ret-17` (ContextualBandit) + `Read-7` (SelectiveByRole) | 在线学习检索 + 角色适配 | 自适应 memory |
| `UF-14` (MultiGranularity) + `Org-6` (Hierarchical) + `Ret-15` (HierarchicalRetrieval) | 多粒度 + 层级存储 + 层级检索 | 类人记忆层级 |
| `Comp-7` (Prototype) + `Maint-12` (Consolidation) | 原型形成 + 碎片整合 | 概念学习 |
| `WT-5` (Importance+Novelty) + `Upd-6` (LLMRewrite) + `Comp-8` (Progressive) | 选择性写 + 动态改写 + 渐进摘要 | 活性记忆系统 |
