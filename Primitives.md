# Memory Primitive 搜索空间

本文档系统枚举每个 primitive slot 的所有实现可能性，定义搜索空间的边界，并分析模块间的兼容性约束。

---

## 0. 搜索空间总览

### 维度结构

```
搜索空间 = LayeredStoreTopo × UnitFormation × Representation × WriteTrigger 
         × Organization × Update × Compression × Retrieval × Readout 
         × Maintenance × CompressTrigger × MaintainTrigger
```

每个维度是一个**离散选择集**（实现变体），每个变体内部可能有**离散/连续超参**。

### Store 拓扑选择（结构维度）

Store 拓扑决定了整个系统的骨架，是最高层的结构性选择。这里不再把拓扑枚举为 `Single-Flat / Dual-Store / Graph-Centric` 这类命名模板，而是改成**多层配置**：

| 结构字段 | 含义 | 典型取值 |
|----|------|----------|
| `layer_count` | 系统有几层 memory layer | 1, 2, 3, 4 |
| `layer.theme` | 该层的语义角色 | `working`, `episodic`, `semantic`, `profile`, `skill`, `reflection`, `custom` |
| `layer.shape` | 该层的存储形态 | `Flat`, `Graph` |
| `layer.indices` | 该层支持的索引能力 | `vector`, `entity`, `temporal`, `keyword`, `graph`, `tag` |
| `layer.capacity` | 该层的预算/容量策略 | `token_limited`, `sliding_window`, `unlimited` |

这种表示把过去大量重复设计拆成了更正交的组合：

- `Single-Flat` = 1 层 + `shape=Flat`
- `Dual-Store` = 2 层 + 不同 `theme`
- `Graph-Centric` = 1 层 + `shape=Graph`
- `Hybrid-Graph` = 多层，其中一层 `Flat`、一层 `Graph`

因此，搜索不再是在几个命名拓扑之间切换，而是在“层数 × 每层主题 × 每层形态 × 每层索引”上进行结构搜索。

---

## A. Unit Formation — 记忆单元形成

**核心问题**：原始输入如何切分/转化为 memory units？

| ID | 实现 | 描述 | 输出 unit_type | 关键参数 |
|----|------|------|---------------|----------|
| UF-1 | **Segment(observation)** | 原始 observation 直接形成一个 unit | raw | mode=`observation` |
| UF-2 | **Segment(message)** | 每条消息独立成 unit | message | mode=`message` |
| UF-3 | **Segment(turn)** | 每轮对话 (user+assistant) 作为一个 unit | turn | mode=`turn`, include_system=bool |
| UF-4 | **Segment(chunk)** | 按固定大小或语义边界分块 | chunk | mode=`chunk`, chunk_size, overlap, strategy={fixed, semantic} |
| UF-5 | **Extract(event)** | 从 observation 中抽取离散事件 | event | target=`event`, method, schema, granularity, source_scope |
| UF-6 | **Extract(fact)** | 抽取事实性命题 (who/what/when/where) | fact | target=`fact`, method, schema, granularity, source_scope |
| UF-7 | **Extract(entity_state)** | 抽取实体及其当前状态 | entity_state | target=`entity_state`, method, entity_types, attribute_schema |
| UF-8 | **Extract(triple)** | 抽取 (subject, predicate, object) 三元组 | triple | target=`triple`, method, ontology, max_triples |
| UF-9 | **Extract(kv)** | 抽取 key-value 对（偏好、属性等） | kv_pair | target=`kv`, method, key_schema |
| UF-10 | **Extract(skill)** | 提取可复用代码/技能 | skill | target=`skill`, method, include_code, include_description |
| UF-11 | **Extract(thought)** | 提取内部推理步骤 | thought | target=`thought`, method, granularity={step, chain} |
| UF-12 | **Abstract(summary)** | 把一个 session / 窗口压缩为 summary unit | summary | target=`summary`, method, source_scope, max_length |
| UF-13 | **Abstract(reflection)** | 从任务轨迹中形成反思 | reflection | target=`reflection`, method, include_trajectory, include_outcome |
| UF-14 | **Delegate(freeform)** | 让 LLM 自由决定切分方式、粒度和输出类型 | mixed | model, instruction |

### 变异轴

- **操作类型**：Segment / Extract / Abstract / Delegate
- **目标类型**：event / fact / entity_state / triple / kv / skill / thought / summary / reflection
- **粒度**：observation → message → turn → chunk → event/fact → triple/kv
- **结构化程度**：自由文本 ↔ 严格 schema
- **形成方法**：规则 / parser / encoder / constrained LLM / free-form LLM

### 建模说明

- `Segment` 负责切分、打包和重分块；它不改变内容语义，只改变 unit 边界。
- `Extract` 负责从输入中抽出结构化或半结构化单元；真正重要的搜索维度是 `target` 与 `method`，而不是为每种 target 单独发明一个 primitive 名字。
- `Abstract` 负责把较长范围的输入压缩为高层单元，如 `summary` 和 `reflection`。
- `Delegate` 用于让模型在开放环境下自行决定形成策略，适合作为高自由度基线。
- 多粒度形成不再单独建模为 `UF-14 MultiGranularity`，而是通过 DSL 里的 `Compose` / `Cascade` 明确表达。

### 常见 `method` 类型

| method | 含义 | 常见适用 target |
|------|------|---------------|
| `rule_based` | regex、模板、关键词、语法规则抽取 | kv, event |
| `schema_guided_llm` | 按给定 schema/slot 填充结构化字段 | fact, entity_state, skill |
| `constrained_decoding` | JSON schema / function calling / CFG 约束输出 | fact, entity_state, triple |
| `span_labeling` | NER / trigger-argument / slot tagging | event, entity_state |
| `relation_extraction` | 关系分类、OpenIE、三元组抽取 | triple, fact |
| `ontology_guided` | 在给定 ontology 中做实体/关系/属性抽取 | entity_state, triple |
| `retrieval_assisted` | 借助已有 store 做 entity disambiguation 或 state linking | entity_state, fact |
| `freeform_llm` | 只给自然语言指令，由模型自由决定输出 | event, thought, reflection |

---

## B. Representation — 表示编码

**核心问题**：unit 以什么形式存储和索引？

这里不再把表示方式枚举为一批“命名模板”，而是改为：**每个 unit 选择一个 representation element set**。  
也就是说，表示编码的核心不是选 `A`、`B`、`A+B` 这种名字，而是决定 unit 最终携带哪些表示元素。

| 元素 ID | 表示元素 | 含义 | 常见来源/参数 |
|----|------|------|----------|
| RE-1 | `text` | 原始或规范化文本内容 | normalize, text_field |
| RE-2 | `embedding` | dense embedding | model, dim |
| RE-3 | `triple` | `(subject, predicate, object)` 三元组 | schema, ontology |
| RE-4 | `kv` | key-value 结构 | key_schema |
| RE-5 | `frame` | slot / frame 填充结构 | frame_schema |
| RE-6 | `entities` | entity tags / entity links | entity_linker, entity_types |
| RE-7 | `tags` | 标签、topic、category | tagger, tag_schema |
| RE-8 | `code` | 可执行代码或代码片段 | language, parser |
| RE-9 | `description` | 对代码或技能的自然语言描述 | description_model |
| RE-10 | `sparse_embedding` | TF-IDF / BM25 稀疏表示 | vocab_size, sparse_method |

常见表示集合示例：

- `{text}`
- `{text, embedding}`
- `{triple}`
- `{text, embedding, entities}`
- `{text, embedding, triple, tags, entities}`
- `{code, description, embedding}`
- `{text, sparse_embedding}`

### 变异轴

- **元素选择**：选择哪些表示元素进入 set
- **结构化程度**：自由文本 ↔ schema-bound
- **检索支持**：dense / sparse / symbolic / hybrid
- **语义增强**：是否加入 entities / tags / description 等附加字段

### 关键约束

- `{triple}`、`{kv}`、`{frame}` 等结构化表示要求上游 UF 产出结构化内容，或 representation 模块自带转换能力
- 向量检索 (Ret-1) 要求表示 set 中包含 `embedding`
- BM25 / sparse retrieval (Ret-2) 要求表示 set 中包含 `text` 或 `sparse_embedding`
- Entity 相关组织与更新要求表示 set 中包含 `entities`，或由上游 UF 显式产出实体字段
- `code` 通常与 `description` 或 `embedding` 联合出现，否则复用和检索能力较弱

### 建模说明

- representation 的搜索对象不再是“选哪个命名实现”，而是“选哪些元素进入 set”
- 这使 `text + embedding`、`text + embedding + entities`、`code + description + embedding` 这类组合都变成同一种语法层次上的配置
- 原来的 `HybridTextEmbed`、`HybridStructured`、`CodeWithDescription` 等都被视为常见 set 模式，而不是新的 primitive 名字

---

## C. Write Trigger — 写入决策

**核心问题**：什么时候该把 unit 写入 store？

这里不再把写入决策建模成很多命名 trigger，而是统一为一个**决策框架**：

```text
WriteTrigger = signals + scorer + gates + policy
```

| 组件 | 含义 | 典型取值 |
|----|------|----------|
| `signals` | 决策时可观测的信号 | `importance`, `novelty`, `surprise`, `duplicate_risk`, `task_relevance`, `event_match`, `llm_value_estimate` |
| `scorer` | 如何把多个信号聚合为 score 或中间判断 | `identity`, `weighted_sum`, `max`, `product`, `rule_expr`, `llm_judge` |
| `gates` | 不经过打分、直接生效的硬条件 | `on_event`, `not_duplicate`, `entity_present`, `tool_called` |
| `policy` | 如何从 score / gates 变成最终写入决定 | `always`, `never`, `threshold`, `top_k_per_window`, `sample_by_score`, `boolean_gate`, `explicit_only` |

### 常见 signal

| Signal | 含义 | 常见来源 |
|------|------|----------|
| `importance` | 内容长期价值或重要性 | heuristics / model / LLM |
| `novelty` | 与已有 store 的差异度 | similarity to store |
| `surprise` | 预测误差或意外程度 | surprise model |
| `duplicate_risk` | 与已有 unit 重复的风险 | dedup detector |
| `task_relevance` | 与当前目标/任务的相关性 | task-conditioned scorer |
| `user_relevance` | 与用户稳定画像或偏好的相关性 | profile relevance scorer |
| `entity_salience` | 是否涉及高价值实体 | entity-aware scorer |
| `event_match` | 是否命中特定事件类型 | event detector |
| `llm_value_estimate` | LLM 对“值不值得记”的估计 | LLM judge |

### 常见 scorer

| scorer | 含义 | 关键参数 |
|------|------|----------|
| `identity` | 单信号直接作为 score | source |
| `weighted_sum` | 多个信号加权求和 | weights |
| `max` | 取最大信号 | sources |
| `product` | 联合门控式相乘 | sources |
| `rule_expr` | 用规则表达式组合信号和谓词 | expression |
| `llm_judge` | LLM 直接输出 score 或 yes/no | model, prompt, criteria |

### 常见 policy

| policy | 含义 | 关键参数 |
|------|------|----------|
| `always` | 无条件写入 | — |
| `never` | 从不写入 | — |
| `threshold` | score 超过阈值时写入 | threshold |
| `top_k_per_window` | 每个窗口只保留 top-k | k, window |
| `sample_by_score` | 按 score 进行概率采样 | temperature |
| `boolean_gate` | gates 满足时写入 | gate_logic |
| `explicit_only` | 只有显式工具调用才写入 | tool_names |

### 常见 gate

| gate | 含义 |
|------|------|
| `on_event("task_failure")` | 特定事件发生时允许写入 |
| `not_duplicate(threshold=...)` | 重复风险过高时阻止写入 |
| `tool_called("memory_write")` | 只有 agent 显式调用工具时写入 |
| `predicate(...)` | 命中规则条件时允许写入 |

### 变异轴

- **信号选择**：看哪些 signals
- **评分方式**：单信号 / 加权融合 / 规则 / LLM
- **决策策略**：阈值 / 采样 / 显式调用 / gate

### 建模说明

- 不是所有 trigger 都应该强行数值化；更好的统一方式是把它们都视为 `signal + gate + policy` 的组合。
- `importance`、`novelty`、`surprise` 这类项不再单独作为 primitive，而是 scorer 的输入。
- `LLMJudge` 和 `AgentToolCall` 也不再是同层级的独立 trigger，而是 `scorer/gates/policy` 的特例。
- 你提到的“agent 编排 score”可以自然表示为：agent 通过 `tool_called(...)`、动态权重或运行时配置来影响 `scorer` 与 `policy`。

### 常见策略到此框架的映射

| 原写法 | 此框架写法 |
|------|------|
| `Always` | `policy=always` |
| `Never` | `policy=never` |
| `ImportanceThreshold` | `signals={importance}, scorer=identity, policy=threshold` |
| `NoveltyThreshold` | `signals={novelty}, scorer=identity, policy=threshold` |
| `ImportanceAndNovelty` | `signals={importance, novelty}, scorer=weighted_sum, policy=threshold` |
| `LLMJudge` | `signals={unit, context, store}, scorer=llm_judge, policy=threshold` |
| `AgentToolCall` | `gates={tool_called(...)}, policy=explicit_only` |
| `RuleBased` | `gates={predicate(...)}, scorer=rule_expr, policy=boolean_gate` |
| `OnEvent` | `gates={on_event(...)}, policy=boolean_gate` |
| `SurpriseGated` | `signals={surprise}, scorer=identity, policy=threshold` |
| `SampledWrite` | `signals={...}, scorer=..., policy=sample_by_score` |
| `DuplicateAwareWrite` | `gates={not_duplicate(...)}, policy=boolean_gate` |

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
| `requires(Read-6, UF-10 produces code)` | CodeInjection 要求代码 unit |
| `requires(Ret-15, Comp produces hierarchy)` | HierarchicalRetrieval 要求层级摘要 |

### 1.3 语义耦合约束（强烈推荐的组合模式）

| 约束 | 含义 |
|------|------|
| `co_requires(UF-6/7/8/9, Upd-3/4)` | 结构化 unit 与 merge/upsert 强关联 |
| `co_requires(UF-2/3, Upd-1)` | 对话级 unit 与 append 强关联 |
| `co_requires(Comp-5, UF-7)` | EntityProfileUpdate 与 EntityState 抽取强关联 |
| `co_requires(Org-5, Ret-7)` | GraphPlacement 与 GraphHop 强关联 |
| `co_requires(Rep contains code, Read-6)` | 代码表示与 CodeInjection 强关联 |

### 1.4 互斥约束

| 约束 | 含义 |
|------|------|
| `incompatible(WT.policy=never, Upd-*)` | 从不写入与任何 update 互斥 |
| `incompatible(Ret-8, Maint-1)` | FullContext 与无维护在大 store 下互斥 |

---

## 2. 搜索空间规模估算

### 2.1 各 slot 候选数

| Slot | 候选数 | 含组合算子后 |
|------|--------|------------|
| Layered Store Topo | `S_struct` | `S_struct` |
| UnitFormation | 14 | 14 + Compose/Cascade 组合 ≈ 120 |
| Representation | `S_rep` | `S_rep` |
| WriteTrigger | `S_wt` | `S_wt` |
| Organization | 10 | 10 |
| Update | 9 | 9 + Dispatch 组合 ≈ 30 |
| Compression | 10 | 10 |
| Retrieval | 17 | 17 + Compose/Cascade 组合 ≈ 200 |
| Readout | 10 | 10 |
| Maintenance | 13 | 13 + Compose ≈ 50 |
| CompressTrigger | 7 | 7 |
| MaintainTrigger | 7 | 7 |

### 2.2 空间上界

不含组合算子：`S_struct × 14 × S_rep × S_wt × 10 × 9 × 10 × 17 × 10 × 13 × 7 × 7`

含组合算子但不含参数：`S_struct × 120 × S_rep × S_wt × 10 × 30 × 10 × 200 × 10 × 50 × 7 × 7`

其中 `S_struct` 不再是一个固定常数，而是结构生成器允许的 layer 配置数。例如，当限制为：

- 层数 `1..4`
- `theme` 从 6 个受控主题中选
- `shape ∈ {Flat, Graph}`
- 每层 index mask 从合法集合中选

则 `S_struct` 会随结构约束的收紧或放松而变化。

同理，`S_rep` 也不再是固定常数。如果基础元素词表为 10 个，则无约束时非空子集上界为 `2^10 - 1 = 1023`；但在实践中会通过能力约束过滤掉大量无意义组合。

`S_wt` 也不再是固定常数，而取决于：

- 允许使用哪些 `signals`
- `scorer` 允许哪些组合方式
- 是否启用 `gates`
- `policy` 的候选集合，以及是否启用 gates / scorer

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
Phase 1: 固定 LayeredStoreTopo + Trigger，搜索 9 个 modular slot（基础实现，不含组合算子）
          空间 ≈ 14 × S_rep × S_wt × 10 × 9 × 10 × 17 × 10 × 13
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
| Generative Agents | UF-1 | `{text, embedding}` | `signals={importance, novelty}, scorer=weighted_sum, policy=always` | Org-2 | Upd-1 | Comp-4 | Compose(Ret-1,4,5) | Read-1 | Maint-1 |
| MemGPT | UF-3 | `{text, embedding}` | `gates={tool_called(...)}, policy=explicit_only` | Org-8 | Upd-1 | Comp-3 | Ret-14→Ret-1 | Read-2 | Maint-11 |
| Reflexion | UF-13 | `{text}` | `gates={on_event("task_failure")}, policy=boolean_gate` | Org-1 | Upd-1 | Comp-1 | Ret-8 | Read-1 | Maint-6 |
| Voyager | UF-10 | `{code, description, embedding}` | `gates={on_event("task_success")}, policy=boolean_gate` | Org-10 | Upd-4 | Comp-1 | Ret-1 | Read-6 | Maint-1 |
| MemoryBank | Compose(UF-3, UF-6) | `{text, embedding}` | `policy=always` | Org-7 | Dispatch(Upd-1,3) | Comp-2 | Compose(Ret-1,4) | Read-1 | Maint-5 |
| SCM | UF-6 | `{triple, embedding}` | `signals={llm_value_estimate}, scorer=llm_judge, policy=threshold` | Org-3 | Upd-4 | Comp-5 | Ret-13→Ret-6 | Read-2 | Maint-9 |
| A-MEM | UF-14 | `{text, embedding, triple, tags, entities}` | `policy=always` | Org-5 | Upd-8 | Comp-1 | Compose(Ret-1,7) | Read-2 | Maint-1 |
| TiM | UF-11 | `{text, embedding}` | `gates={on_reasoning_step}, policy=boolean_gate` | Org-2 | Upd-1 | Comp-8 | Compose(Ret-3,1) | Read-5 | Maint-6 |

### 从中可观察到的初步 Motif

**Motif 1: Embed-and-Retrieve（出现 7/8 次）**
- 表示 set 包含 `embedding` + `Ret-1` (Similarity) 或其组合
- 几乎所有系统都需要 embedding 作为基础检索手段

**Motif 2: Append-Dominant（出现 6/8 次）**
- `Upd-1` (Append) 是最常见的写入方式
- 只有涉及结构化 entity 更新的系统才用 merge/upsert

**Motif 3: Selective-Write 分化**
- 简单系统多用 `policy=always` 或 `gates={on_event(...)}`
- 复杂系统更倾向于 `scorer=llm_judge` 或 `policy=explicit_only`
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
| `UF-8` (Extract(triple)) + `Ret-7` (GraphHop) + `Maint-5` (Ebbinghaus) | 结构化图记忆 + 认知遗忘 | 长期对话中的知识演化 |
| `signals={surprise} + policy=threshold` + `Comp-4` (Reflection) | 只记意外 + 反思 | 高效学习型 agent |
| `Ret-17` (ContextualBandit) + `Read-7` (SelectiveByRole) | 在线学习检索 + 角色适配 | 自适应 memory |
| `Compose(UF-3, UF-6, UF-12)` + `Org-6` (Hierarchical) + `Ret-15` (HierarchicalRetrieval) | 多粒度 + 层级存储 + 层级检索 | 类人记忆层级 |
| `Comp-7` (Prototype) + `Maint-12` (Consolidation) | 原型形成 + 碎片整合 | 概念学习 |
| `signals={importance, novelty} + scorer=weighted_sum + policy=threshold` + `Upd-6` (LLMRewrite) + `Comp-8` (Progressive) | 选择性写 + 动态改写 + 渐进摘要 | 活性记忆系统 |
