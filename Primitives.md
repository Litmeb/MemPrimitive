# Memory Primitive 搜索空间

本文档系统枚举每个 primitive slot 的所有实现可能性，定义搜索空间的边界，并分析模块间的兼容性约束。

---

## 0. 搜索空间总览

### 维度结构

```
搜索空间 = LayeredStoreTopo × UnitFormation × Representation × WriteTrigger 
         × Organization × MemoryEvolution × Retrieval × Readout 
         × EvolutionTrigger
```

每个维度是一个**离散选择集**（实现变体），每个变体内部可能有**离散/连续超参**。

### Store 拓扑选择（结构维度）

Store 拓扑决定了整个系统的骨架，是最高层的结构性选择。这里不再把拓扑枚举为 `Single-Flat / Dual-Store / Graph-Centric` 这类命名模板，而是改成**多层配置**：


| 结构字段             | 含义                 | 典型取值                                                                          |
| ---------------- | ------------------ | ----------------------------------------------------------------------------- |
| `layer_count`    | 系统有几层 memory layer | 1, 2, 3, 4                                                                    |
| `layer.theme`    | 该层的语义角色            | `working`, `episodic`, `semantic`, `profile`, `skill`, `reflection`, `custom` |
| `layer.shape`    | 该层的存储形态            | `Flat`, `Graph`                                                               |
| `layer.indices`  | 该层支持的索引能力          | `vector`, `entity`, `temporal`, `keyword`, `graph`, `tag`                     |
| `layer.capacity` | 该层的预算/容量策略         | `token_limited`, `sliding_window`, `unlimited`                                |


这种表示把过去大量重复设计拆成了更正交的组合：

- `Single-Flat` = 1 层 + `shape=Flat`
- `Dual-Store` = 2 层 + 不同 `theme`
- `Graph-Centric` = 1 层 + `shape=Graph`
- `Hybrid-Graph` = 多层，其中一层 `Flat`、一层 `Graph`

因此，搜索不再是在几个命名拓扑之间切换，而是在“层数 × 每层主题 × 每层形态 × 每层索引”上进行结构搜索。

---

## A. Unit Formation — 记忆单元形成

**核心问题**：原始输入如何切分/转化为 memory units？


| ID    | 实现                        | 描述                                  | 输出 unit_type | 关键参数                                                             |
| ----- | ------------------------- | ----------------------------------- | ------------ | ---------------------------------------------------------------- |
| UF-1  | **Segment(observation)**  | 原始 observation 直接形成一个 unit          | raw          | mode=`observation`                                               |
| UF-2  | **Segment(message)**      | 每条消息独立成 unit                        | message      | mode=`message`                                                   |
| UF-3  | **Segment(turn)**         | 每轮对话 (user+assistant) 作为一个 unit     | turn         | mode=`turn`, include_system=bool                                 |
| UF-4  | **Segment(chunk)**        | 按固定大小或语义边界分块                        | chunk        | mode=`chunk`, chunk_size, overlap, strategy={fixed, semantic}    |
| UF-5  | **Extract(event)**        | 从 observation 中抽取离散事件               | event        | target=`event`, method, schema, granularity, source_scope        |
| UF-6  | **Extract(fact)**         | 抽取事实性命题 (who/what/when/where)       | fact         | target=`fact`, method, schema, granularity, source_scope         |
| UF-7  | **Extract(entity_state)** | 抽取实体及其当前状态                          | entity_state | target=`entity_state`, method, entity_types, attribute_schema    |
| UF-8  | **Extract(triple)**       | 抽取 (subject, predicate, object) 三元组 | triple       | target=`triple`, method, ontology, max_triples                   |
| UF-9  | **Extract(kv)**           | 抽取 key-value 对（偏好、属性等）              | kv_pair      | target=`kv`, method, key_schema                                  |
| UF-10 | **Extract(skill)**        | 提取可复用代码/技能                          | skill        | target=`skill`, method, include_code, include_description        |
| UF-11 | **Extract(thought)**      | 提取内部推理步骤                            | thought      | target=`thought`, method, granularity={step, chain}              |
| UF-12 | **Abstract(summary)**     | 把一个 session / 窗口压缩为 summary unit    | summary      | target=`summary`, method, source_scope, max_length               |
| UF-13 | **Abstract(reflection)**  | 从任务轨迹中形成反思                          | reflection   | target=`reflection`, method, include_trajectory, include_outcome |
| UF-14 | **Delegate(freeform)**    | 让 LLM 自由决定切分方式、粒度和输出类型              | mixed        | model, instruction                                               |


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


| method                 | 含义                                                 | 常见适用 target                |
| ---------------------- | -------------------------------------------------- | -------------------------- |
| `rule_based`           | regex、模板、关键词、语法规则抽取                                | kv, event                  |
| `schema_guided_llm`    | 按给定 schema/slot 填充结构化字段                            | fact, entity_state, skill  |
| `constrained_decoding` | JSON schema / function calling / CFG 约束输出          | fact, entity_state, triple |
| `span_labeling`        | NER / trigger-argument / slot tagging              | event, entity_state        |
| `relation_extraction`  | 关系分类、OpenIE、三元组抽取                                  | triple, fact               |
| `ontology_guided`      | 在给定 ontology 中做实体/关系/属性抽取                          | entity_state, triple       |
| `retrieval_assisted`   | 借助已有 store 做 entity disambiguation 或 state linking | entity_state, fact         |
| `freeform_llm`         | 只给自然语言指令，由模型自由决定输出                                 | event, thought, reflection |


---

## B. Representation — 表示编码

**核心问题**：unit 以什么形式存储和索引？

这里不再把表示方式枚举为一批“命名模板”，而是改为：**每个 unit 选择一个 representation element set**。  
也就是说，表示编码的核心不是选 `A`、`B`、`A+B` 这种名字，而是决定 unit 最终携带哪些表示元素。


| 元素 ID | 表示元素               | 含义                                 | 常见来源/参数                     |
| ----- | ------------------ | ---------------------------------- | --------------------------- |
| RE-1  | `text`             | 原始或规范化文本内容                         | normalize, text_field       |
| RE-2  | `embedding`        | dense embedding                    | model, dim                  |
| RE-3  | `triple`           | `(subject, predicate, object)` 三元组 | schema, ontology            |
| RE-4  | `kv`               | key-value 结构                       | key_schema                  |
| RE-5  | `frame`            | slot / frame 填充结构                  | frame_schema                |
| RE-6  | `entities`         | entity tags / entity links         | entity_linker, entity_types |
| RE-7  | `tags`             | 标签、topic、category                  | tagger, tag_schema          |
| RE-8  | `code`             | 可执行代码或代码片段                         | language, parser            |
| RE-9  | `description`      | 对代码或技能的自然语言描述                      | description_model           |
| RE-10 | `sparse_embedding` | TF-IDF / BM25 稀疏表示                 | vocab_size, sparse_method   |


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


| 组件        | 含义                         | 典型取值                                                                                                         |
| --------- | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `signals` | 决策时可观测的信号                  | `importance`, `novelty`, `surprise`, `duplicate_risk`, `task_relevance`, `event_match`, `llm_value_estimate` |
| `scorer`  | 如何把多个信号聚合为 score 或中间判断     | `identity`, `weighted_sum`, `max`, `product`, `rule_expr`, `llm_judge`                                       |
| `gates`   | 不经过打分、直接生效的硬条件             | `on_event`, `not_duplicate`, `entity_present`, `tool_called`                                                 |
| `policy`  | 如何从 score / gates 变成最终写入决定 | `always`, `never`, `threshold`, `top_k_per_window`, `sample_by_score`, `boolean_gate`, `explicit_only`       |


### 常见 signal


| Signal               | 含义              | 常见来源                     |
| -------------------- | --------------- | ------------------------ |
| `importance`         | 内容长期价值或重要性      | heuristics / model / LLM |
| `novelty`            | 与已有 store 的差异度  | similarity to store      |
| `surprise`           | 预测误差或意外程度       | surprise model           |
| `duplicate_risk`     | 与已有 unit 重复的风险  | dedup detector           |
| `task_relevance`     | 与当前目标/任务的相关性    | task-conditioned scorer  |
| `user_relevance`     | 与用户稳定画像或偏好的相关性  | profile relevance scorer |
| `entity_salience`    | 是否涉及高价值实体       | entity-aware scorer      |
| `event_match`        | 是否命中特定事件类型      | event detector           |
| `llm_value_estimate` | LLM 对“值不值得记”的估计 | LLM judge                |


### 常见 scorer


| scorer         | 含义                      | 关键参数                    |
| -------------- | ----------------------- | ----------------------- |
| `identity`     | 单信号直接作为 score           | source                  |
| `weighted_sum` | 多个信号加权求和                | weights                 |
| `max`          | 取最大信号                   | sources                 |
| `product`      | 联合门控式相乘                 | sources                 |
| `rule_expr`    | 用规则表达式组合信号和谓词           | expression              |
| `llm_judge`    | LLM 直接输出 score 或 yes/no | model, prompt, criteria |


### 常见 policy


| policy             | 含义             | 关键参数        |
| ------------------ | -------------- | ----------- |
| `always`           | 无条件写入          | —           |
| `never`            | 从不写入           | —           |
| `threshold`        | score 超过阈值时写入  | threshold   |
| `top_k_per_window` | 每个窗口只保留 top-k  | k, window   |
| `sample_by_score`  | 按 score 进行概率采样 | temperature |
| `boolean_gate`     | gates 满足时写入    | gate_logic  |
| `explicit_only`    | 只有显式工具调用才写入    | tool_names  |


### 常见 gate


| gate                           | 含义                 |
| ------------------------------ | ------------------ |
| `on_event("task_failure")`     | 特定事件发生时允许写入        |
| `not_duplicate(threshold=...)` | 重复风险过高时阻止写入        |
| `tool_called("memory_write")`  | 只有 agent 显式调用工具时写入 |
| `predicate(...)`               | 命中规则条件时允许写入        |


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


| 原写法                    | 此框架写法                                                                  |
| ---------------------- | ---------------------------------------------------------------------- |
| `Always`               | `policy=always`                                                        |
| `Never`                | `policy=never`                                                         |
| `ImportanceThreshold`  | `signals={importance}, scorer=identity, policy=threshold`              |
| `NoveltyThreshold`     | `signals={novelty}, scorer=identity, policy=threshold`                 |
| `ImportanceAndNovelty` | `signals={importance, novelty}, scorer=weighted_sum, policy=threshold` |
| `LLMJudge`             | `signals={unit, context, store}, scorer=llm_judge, policy=threshold`   |
| `AgentToolCall`        | `gates={tool_called(...)}, policy=explicit_only`                       |
| `RuleBased`            | `gates={predicate(...)}, scorer=rule_expr, policy=boolean_gate`        |
| `OnEvent`              | `gates={on_event(...)}, policy=boolean_gate`                           |
| `SurpriseGated`        | `signals={surprise}, scorer=identity, policy=threshold`                |
| `SampledWrite`         | `signals={...}, scorer=..., policy=sample_by_score`                    |
| `DuplicateAwareWrite`  | `gates={not_duplicate(...)}, policy=boolean_gate`                      |


---

## D. Organization — 关系归组与放置规划

**核心问题**：unit 应该放在哪里？与已有 memory 建立什么关系？

这里不再把 organization 建模成很多命名策略，而是统一写成：

```text
Organization = routing + links + placement
```

并且显式承认：

```text
Organization is topology-constrained.
StoreTopology defines the admissible routing targets, link types, and placement modes.
```

### 组织组件


| 组件          | 含义                   | 典型取值                                                                                   |
| ----------- | -------------------- | -------------------------------------------------------------------------------------- |
| `routing`   | unit 先被送到哪一层/哪一分区    | `default`, `by_unit_type`, `by_tag`, `by_rule`, `explicit`, `agent_selected`           |
| `links`     | 写入时与已有 memory 建立哪些关系 | `temporal`, `entity`, `similarity`, `cluster_membership`, `graph_edge`, `parent_child` |
| `placement` | 到达目标层后如何安放           | `append`, `partition`, `cluster`, `graph_node`, `hierarchical_slot`                    |


### 常见 routing


| routing          | 含义                | 关键参数                 |
| ---------------- | ----------------- | -------------------- |
| `default`        | 总是写入默认 layer      | target_layer         |
| `by_unit_type`   | 按 unit_type 路由    | route_map            |
| `by_tag`         | 按 tag 或 topic 路由  | tag_field, route_map |
| `by_rule`        | 按规则条件路由           | rules                |
| `explicit`       | 上游字段中已带有目标位置      | field                |
| `agent_selected` | agent 通过工具或字段指定目标 | tool_to_layer        |


### 常见 links


| link                 | 含义                       | 关键参数                          |
| -------------------- | ------------------------ | ----------------------------- |
| `temporal`           | 建立时间邻接关系                 | window, direction             |
| `entity`             | 按共享实体建立关联                | entity_field, link_type       |
| `similarity`         | 建立相似度近邻关系                | metric, threshold             |
| `cluster_membership` | 连接到某个 cluster / centroid | cluster_key                   |
| `graph_edge`         | 在图层中创建有类型边               | edge_types, connection_method |
| `parent_child`       | 建立层级父子关系                 | level_policy                  |


### 常见 placement


| placement           | 含义              | 关键参数                               |
| ------------------- | --------------- | ---------------------------------- |
| `append`            | 直接追加到目标 layer   | —                                  |
| `partition`         | 写入某个分区 / bucket | partition_key                      |
| `cluster`           | 放入最近的 cluster   | similarity_threshold, max_clusters |
| `graph_node`        | 在图层中创建节点并接边     | node_policy                        |
| `hierarchical_slot` | 放入多层结构中的某一层/槽位  | target_level, placement_policy     |


### 变异轴

- **路由目标**：去哪一层 / 哪个分区
- **关系结构**：建立哪些 link
- **放置方式**：append / partition / cluster / graph / hierarchical
- **结构约束来源**：由 StoreTopology 决定可选能力边界

### Topology-Constrained Organization

- `StoreTopology` 决定 admissible routing targets：没有的 layer 不能被路由。
- `StoreTopology` 决定 admissible link types：例如没有 graph layer 或 `graph` index，就不该允许 `graph_edge`。
- `StoreTopology` 决定 admissible placement modes：例如单层 flat store 不应该允许 `hierarchical_slot`。
- `Organization` 负责在这些能力边界内，选择具体的 routing / links / placement 配置。

### 关键约束

- `links contains entity` 要求 unit 有 `entities` 字段 → 需要 UF 产出 entity 或 Rep 补充 entity
- `routing=by_tag` 要求 unit/representation 中有 `tags`
- `placement=cluster` 或 `links contains similarity` 通常要求 `embedding`
- `placement=graph_node` 或 `links contains graph_edge` 要求存在 `shape=Graph` 的 layer，且通常要求 `graph` index
- `placement=hierarchical_slot` 或 `links contains parent_child` 要求 layer_count > 1
- `routing=by_unit_type`、`by_rule`、`agent_selected` 等都要求目标 layer 在当前拓扑中存在

### 旧命名策略到新框架的映射


| 旧写法                     | 新写法                                                                  |
| ----------------------- | -------------------------------------------------------------------- |
| `FlatAppend`            | `routing=default, links={}, placement=append`                        |
| `TemporalAppend`        | `routing=default, links={temporal}, placement=append`                |
| `EntityLinked`          | `routing=default, links={entity}, placement=append`                  |
| `TemporalAndEntity`     | `routing=default, links={temporal, entity}, placement=append`        |
| `GraphPlacement`        | `routing=default, links={graph_edge(...)}, placement=graph_node`     |
| `HierarchicalPlacement` | `routing=default, links={parent_child}, placement=hierarchical_slot` |
| `DualStoreRouter`       | `routing=by_unit_type(...), links={}, placement=append`              |
| `AgentExplicitTarget`   | `routing=agent_selected(...), links={}, placement=append`            |
| `SimilarityCluster`     | `routing=default, links={cluster_membership}, placement=cluster`     |
| `TagBasedPlacement`     | `routing=by_tag(...), links={}, placement=partition`                 |


---

## E. Memory Evolution — 记忆演化

**核心问题**：一旦 unit 已经被组织到某个 layer，store 应如何随之演化？**

这里统一吸收原来的 `Update`、`Compression` 和 `Maintenance`。  
它们本质上都是：针对已有 memory state 施加某种演化操作，只是作用范围、目标和触发时机不同。

```text
MemoryEvolution = selection + action + effect + trigger
```

### 演化组件


| 组件          | 含义                 | 典型取值                                                                                                                                                                                |
| ----------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `selection` | 选中哪些已有 memory 参与演化 | `incoming_only`, `matched_by_key`, `matched_by_entity`, `time_window`, `layer_slice`, `low_activity`, `all`                                                                         |
| `action`    | 对选中对象施加什么演化操作      | `append`, `replace`, `merge`, `upsert`, `rewrite`, `summarize`, `reflect`, `profile_update`, `extract_concept`, `prototype_form`, `prune`, `dedup`, `move`, `consolidate`, `review` |
| `effect`    | 对 store 造成何种状态变化   | `add`, `modify`, `delete`, `move`, `summarize`, `merge`, `version`                                                                                                                  |
| `trigger`   | 何时执行该演化操作          | `after_write`, `periodic`, `on_event`, `budget_exceeded`, `count_exceeded`, `conditional`                                                                                           |


### 常见 selection


| selection           | 含义              | 关键参数                      |
| ------------------- | --------------- | ------------------------- |
| `incoming_only`     | 只处理当前新写入 unit   | —                         |
| `matched_by_key`    | 按 key 匹配已有 unit | key_field, match_strategy |
| `matched_by_entity` | 按 entity 匹配已有记忆 | entity_field              |
| `time_window`       | 处理某个时间窗口内的记忆    | start, end / window_size  |
| `layer_slice`       | 处理某层或某分区中的对象    | target_layer, filter      |
| `low_activity`      | 处理低活跃、低价值对象     | activity_threshold        |
| `all`               | 处理整个目标集合        | scope                     |


### 常见 action


| action             | 含义              | 对应旧名                   |
| ------------------ | --------------- | ---------------------- |
| `append`           | 直接追加            | Upd-1                  |
| `replace`          | 替换匹配对象          | Upd-2                  |
| `merge`            | 合并新旧对象          | Upd-3, Upd-7           |
| `upsert`           | 有则更新，无则插入       | Upd-4                  |
| `rewrite`          | LLM 或规则重写已有内容   | Upd-6                  |
| `delta`            | 只记录变化量          | Upd-5                  |
| `versioned_append` | 追加并保留版本链        | Upd-9                  |
| `summarize`        | 形成摘要            | Comp-2, Comp-8, Comp-9 |
| `reflect`          | 形成反思/insight    | Comp-4                 |
| `profile_update`   | 聚合为实体画像/profile | Comp-5                 |
| `extract_concept`  | 形成概念单元          | Comp-6                 |
| `prototype_form`   | 形成原型/schema     | Comp-7                 |
| `distill`          | 高密度蒸馏           | Comp-10                |
| `prune`            | 删除低价值对象         | Maint-2/3/4/5/6/7      |
| `dedup`            | 去重并合并           | Maint-8                |
| `move`             | 层间迁移/归档         | Maint-10               |
| `consolidate`      | 定期整合碎片          | Maint-12               |
| `review`           | 审查相关性或冲突        | Maint-9, Maint-13      |


### 常见 effect


| effect      | 含义             |
| ----------- | -------------- |
| `add`       | 新增 unit        |
| `modify`    | 修改已有 unit      |
| `delete`    | 删除已有 unit      |
| `move`      | 从一层迁移到另一层      |
| `merge`     | 合并多个 unit      |
| `summarize` | 生成高层单元并可能替换原对象 |
| `version`   | 保留版本链          |


### 变异轴

- **演化目标**：局部更新 / 高层抽象 / 生命周期管理
- **作用范围**：incoming-only / by-entity / by-window / by-layer / global
- **状态变化**：add / modify / delete / move / summarize / version
- **触发模式**：写入后 / 周期 / 事件 / 预算 / 条件

### 建模说明

- 原 `Update` 只是一类局部演化；原 `Compression` 是一种高层抽象演化；原 `Maintenance` 是一种生命周期演化。
- `AppendWithLinkUpdate` 不再单列；link 更新原则上应回到 D 槽，E 只负责 store state evolution。
- `append` 可以被视为最简单的 evolution action，而不是必须单独占据一个大类。
- `summarize`、`prune`、`move`、`consolidate` 与 `merge` 一样，都是对 store state 的不同演化方式。

### 关键约束

- `selection=matched_by_entity`、`action=merge/profile_update` 要求 unit 或 representation 提供 `entities`
- `action=delta` 要求能对齐到已有 unit → 需要 key_field 或 entity
- `action=summarize/reflect/extract_concept/prototype_form` 通常要求 selection 覆盖多个 unit
- `action=move` 要求 layer_count > 1
- `action=rewrite/review` 通常带来额外 LLM 成本，不适合高频触发
- `action=prune`、`dedup`、`review` 与 retrieval 行为强耦合，可能改变后续可读上下文

### 旧模块到新框架的映射


| 旧写法                              | 新写法                                                                     |
| -------------------------------- | ----------------------------------------------------------------------- |
| `Upd-1 Append`                   | `selection=incoming_only, action=append, effect=add`                    |
| `Upd-3 MergeByEntity`            | `selection=matched_by_entity, action=merge, effect=modify`              |
| `Upd-4 UpsertByKey`              | `selection=matched_by_key, action=upsert, effect=add+modify`            |
| `Upd-6 LLMRewrite`               | `selection=matched_by_key/entity, action=rewrite, effect=modify`        |
| `Comp-2 Summarization`           | `selection=time_window/layer_slice, action=summarize, effect=summarize` |
| `Comp-4 LLMReflection`           | `selection=time_window/all, action=reflect, effect=add`                 |
| `Comp-5 EntityProfileUpdate`     | `selection=matched_by_entity, action=profile_update, effect=modify`     |
| `Maint-8 Deduplication`          | `selection=layer_slice/all, action=dedup, effect=delete+modify`         |
| `Maint-10 ArchivalMovement`      | `selection=low_activity, action=move, effect=move`                      |
| `Maint-12 PeriodicConsolidation` | `selection=layer_slice, action=consolidate, effect=merge+delete`        |


---

## G. Retrieval — 检索策略

**核心问题**：给定 query，如何从 store 中找到相关 memory？

这里不再把 retrieval 建模成很多命名方法，而是统一写成：

```text
Retrieval = signals + ranker + flow + constraints
```

### 检索组件


| 组件            | 含义                    | 典型取值                                                                                                                              |
| ------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `signals`     | 检索时使用哪些证据             | `similarity`, `keyword_match`, `recency`, `importance`, `entity_match`, `graph_proximity`, `hierarchy_match`, `diversity_penalty` |
| `ranker`      | 如何把多个 signals 合成为最终排序 | `identity`, `weighted_sum`, `rrf`, `decay`, `rerank_llm`, `mmr`                                                                   |
| `flow`        | 检索流程结构                | `single_stage`, `two_stage`, `top_down`, `agent_invoked`                                                                          |
| `constraints` | 对返回结果施加的结构或预算约束       | `top_k`, `window`, `edge_filter`, `expand_depth`, `final_k`                                                                       |


### 常见 signal


| signal              | 含义              | 常见来源                        |
| ------------------- | --------------- | --------------------------- |
| `similarity`        | 向量相似度           | embedding search            |
| `keyword_match`     | 关键词/稀疏匹配        | BM25 / sparse retrieval     |
| `recency`           | 时间新近性           | time index                  |
| `importance`        | 重要性分数           | memory metadata             |
| `entity_match`      | 实体精确匹配或 overlap | entity index                |
| `graph_proximity`   | 图邻近、多跳关联        | graph edges                 |
| `hierarchy_match`   | 层级摘要/父子层级相关性    | hierarchy links / summaries |
| `diversity_penalty` | 多样性约束或惩罚项       | MMR / DPP-like logic        |


### 常见 ranker


| ranker         | 含义                     | 关键参数                  |
| -------------- | ---------------------- | --------------------- |
| `identity`     | 单一 signal 直接排序         | source                |
| `weighted_sum` | 多个 signal 加权求和         | weights               |
| `rrf`          | Reciprocal Rank Fusion | k                     |
| `decay`        | 对时间/分数做衰减变换            | decay_rate, half_life |
| `rerank_llm`   | 用 LLM 对候选进行重排          | model, rerank_prompt  |
| `mmr`          | 相关性和多样性联合排序            | lambda                |


### 常见 flow


| flow            | 含义               | 关键参数                 |
| --------------- | ---------------- | -------------------- |
| `single_stage`  | 单阶段直接召回和排序       | top_k                |
| `two_stage`     | 先粗召回，再精排         | recall_k, final_k    |
| `top_down`      | 先检索高层摘要，再展开细节    | levels, expand_top_k |
| `agent_invoked` | 由 agent 显式发起检索工具 | tool_names, backend  |


### 变异轴

- **信号选择**：使用哪些 retrieval signals
- **排序方式**：单信号 / 融合 / 衰减 / LLM 重排 / 多样性约束
- **流程结构**：单阶段 / 两阶段 / top-down / agent 显式触发
- **结果约束**：top_k / final_k / edge_filter / expand_depth

### 关键约束

- `signals contains similarity` 要求 store 有 `vector` index 且 unit 有 `embedding`
- `signals contains keyword_match` 要求 store 有 `keyword` index 或 unit 有文本内容
- `signals contains entity_match` 要求 store 有 `entity` index 且 unit 有 `entities`
- `signals contains graph_proximity` 要求 store 有 `graph` index 且 `Org` 产出了 graph edges
- `signals contains hierarchy_match` 或 `flow=top_down` 要求存在层级摘要/父子层级
- `ranker=rerank_llm` 会增加额外 LLM 成本
- `flow=agent_invoked` 不再视为一种独立 retrieval primitive，而只是检索的触发方式

### 建模说明

- `similarity / recency / importance / entity_match / graph_proximity` 与 C 槽里的 signal 思维保持一致，都是可复用的机制信号。
- `TwoStageRecallRerank` 不再单列，而被 `flow=two_stage` + `ranker=...` 覆盖。
- `HierarchicalRetrieval` 不再单列，而被 `signals={hierarchy_match}` + `flow=top_down` 覆盖。
- `DiversityAware` 降为 `ranker=mmr` 或 `signals` 中的 `diversity_penalty`。
- `FullContext`、`LLMControlled`、`ContextualBandit` 暂不作为基础主表的一等成员保留；如果需要，可作为后续扩展。

### 旧检索方法到新框架的映射


| 旧写法                     | 新写法                                                             |
| ----------------------- | --------------------------------------------------------------- |
| `Similarity`            | `signals={similarity}, ranker=identity, flow=single_stage`      |
| `BM25`                  | `signals={keyword_match}, ranker=identity, flow=single_stage`   |
| `Recency`               | `signals={recency}, ranker=identity, flow=single_stage`         |
| `RecencyDecay`          | `signals={recency}, ranker=decay, flow=single_stage`            |
| `ImportanceRanked`      | `signals={importance}, ranker=identity, flow=single_stage`      |
| `EntityLookup`          | `signals={entity_match}, ranker=identity, flow=single_stage`    |
| `GraphHop`              | `signals={graph_proximity}, ranker=identity, flow=single_stage` |
| `WeightedMultiSignal`   | `signals={...}, ranker=weighted_sum, flow=single_stage`         |
| `TwoStageRecallRerank`  | `flow=two_stage, ranker=...`                                    |
| `LLMRerank`             | `ranker=rerank_llm`                                             |
| `HierarchicalRetrieval` | `signals={hierarchy_match}, flow=top_down`                      |
| `AgentToolCall`         | `flow=agent_invoked`                                            |


---

## H. Readout — 输出格式化

**核心问题**：检索到的 memory 如何转化为 agent 可消费的输入？


这里不再把 readout 视为一个大的自由搜索槽。对于当前 design space，只保留少数真正会改变 memory 使用方式的 readout；其余多数更像任务接口或宿主框架层的格式选择。

| ID | 实现 | 描述 | 输出格式 | 关键参数 |
|----|------|------|---------|----------|
| Read-1 | **FlatConcat** | 按序拼接检索结果为文本 | text block | separator, max_tokens, format |
| Read-3 | **SummarizedReadout** | 先摘要检索结果再输出 | summary text | model, max_length |
| Read-4 | **TemplatedPrompt** | 填入预定义 prompt 模板 | filled template | template |
| Read-6 | **CodeInjection** | 以函数/工具定义形式注入 | code/tool defs | format={function_def, docstring} |


### 变异轴

- **输出形式**：直接拼接 / 摘要后输出 / 模板包装
- **注入位置**：prompt 文本 / tool/code 定义
- **压缩度**：全量 / 摘要后注入

### 建模说明

- `FlatConcat` 是最基础、最通用的 readout 基线。
- `SummarizedReadout` 是少数真正会改变 memory 使用路径的 readout，因此保留。
- `TemplatedPrompt` 可以看作对 text readout 的包装层，保留它是为了覆盖真实系统中的 prompt 接口差异。
- `CodeInjection` 保留，但应视为**特殊 readout mode**：通常仅在 skill/tool memory 中使用，不参与默认搜索空间。
- `StructuredSections`、`PrependToContext`、`SelectiveByRole`、`RankedList`、`JSONStructured`、`LatentInjection` 暂降为任务/框架层格式选择，不再作为主搜索项。

---

## J. Evolution Trigger — 触发条件（Memory Evolution 共享）


| ID     | 实现                 | 描述             | 关键参数             |
| ------ | ------------------ | -------------- | ---------------- |
| Trig-1 | **Never**          | 从不触发           | —                |
| Trig-2 | **Periodic**       | 每 N 步触发一次      | every: int       |
| Trig-3 | **AfterWrite**     | 每次写入后触发        | —                |
| Trig-4 | **OnEvent**        | 特定事件发生时触发      | event_type       |
| Trig-5 | **BudgetExceeded** | 预算使用超过阈值时触发    | threshold: float |
| Trig-6 | **CountExceeded**  | unit 数量超过阈值时触发 | max_count: int   |
| Trig-7 | **Conditional**    | 自定义条件表达式       | predicate        |


---

## 1. 兼容性约束矩阵

以下约束在搜索时用于剪枝非法组合。

### 1.1 Store → Module 约束（store 拓扑限制模块选择）


| 约束                                                                             | 含义                    |
| ------------------------------------------------------------------------------ | --------------------- |
| `requires(Ret.signals contains similarity, store.index contains vector)`       | similarity 检索要求向量索引   |
| `requires(Ret.signals contains keyword_match, store.index contains keyword)`   | keyword 检索要求关键词索引     |
| `requires(Ret.signals contains graph_proximity, store.index contains graph)`   | graph proximity 要求图索引 |
| `requires(Ret.signals contains entity_match, store.index contains entity)`     | entity match 要求实体索引   |
| `requires(Org.placement=graph_node, exists layer.shape == Graph)`              | 图节点放置要求存在 graph layer |
| `requires(Org.links contains graph_edge, exists layer.indices contains graph)` | 图边要求 graph index      |
| `requires(Org.placement=hierarchical_slot, layer_count > 1)`                   | 层级放置要求多层拓扑            |
| `requires(Org.routing=by_unit_type, layer_count > 1)`                          | 按类型路由通常要求多个目标 layer   |
| `requires(Evo.action=move, layer_count > 1)`                                   | 层间迁移要求多层拓扑            |


### 1.2 Module → Module 约束（上游输出限制下游选择）


| 约束                                                                                                   | 含义                               |
| ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| `requires(Ret.signals contains similarity, Rep-* produces embedding)`                                | 向量检索要求有 embedding                |
| `requires(Org.links contains entity, UF/Rep produces entities)`                                      | entity link 要求有实体                |
| `requires(Org.routing=by_tag, Rep/UF produces tags)`                                                 | tag routing 要求 tags              |
| `requires(Org.placement=cluster, Rep produces embedding)`                                            | cluster 放置要求 embedding           |
| `requires(Evo.action in {merge, profile_update}, UF/Rep produces entities)`                          | entity 合并/画像更新要求有实体              |
| `requires(Evo.action=delta, UF produces alignable units)`                                            | delta 演化要求可对齐                    |
| `requires(Ret.signals contains graph_proximity, Org.links contains graph_edge)`                      | graph proximity 要求有图边            |
| `requires(Read-6, UF-10 produces code)`                                                              | CodeInjection 要求代码 unit          |
| `requires(Ret.flow=top_down or Ret.signals contains hierarchy_match, Evo.action produces hierarchy)` | top-down/hierarchy 检索要求层级摘要/父子层级 |


### 1.3 语义耦合约束（强烈推荐的组合模式）


| 约束                                                                                 | 含义                            |
| ---------------------------------------------------------------------------------- | ----------------------------- |
| `co_requires(UF-6/7/8/9, Evo.action in {merge, upsert, profile_update})`           | 结构化 unit 与实体/键控演化强关联          |
| `co_requires(UF-2/3, Evo.action=append)`                                           | 对话级 unit 与 append 强关联         |
| `co_requires(Evo.action=profile_update, UF-7)`                                     | profile 更新与 EntityState 抽取强关联 |
| `co_requires(Org.links contains graph_edge, Ret.signals contains graph_proximity)` | 图边组织与图邻近检索强关联                 |
| `co_requires(Rep contains code, Read-6)`                                           | 代码表示与 CodeInjection 强关联       |


### 1.4 互斥约束


| 约束                                                                              | 含义          |
| ------------------------------------------------------------------------------- | ----------- |
| `incompatible(WT.policy=never, Evo.action in {append, merge, upsert, rewrite})` | 从不写入与增量演化互斥 |


---

## 2. 搜索空间规模估算

### 2.1 各 slot 候选数


| Slot               | 候选数        | 含组合算子后                        |
| ------------------ | ---------- | ----------------------------- |
| Layered Store Topo | `S_struct` | `S_struct`                    |
| UnitFormation      | 14         | 14 + Compose/Cascade 组合 ≈ 120 |
| Representation     | `S_rep`    | `S_rep`                       |
| WriteTrigger       | `S_wt`     | `S_wt`                        |
| Organization       | `S_org`    | `S_org`                       |
| MemoryEvolution    | `S_evo`    | `S_evo`                       |
| Retrieval          | `S_ret`    | `S_ret`                       |
| Readout            | 3 + `Read-6*` | 3 + `Read-6*`                |
| EvolutionTrigger   | 7          | 7                             |

`Read-6*` 表示 `CodeInjection` 作为特殊 readout mode 保留，但默认不纳入常规搜索空间。


### 2.2 空间上界

不含组合算子：`S_struct × 14 × S_rep × S_wt × S_org × S_evo × S_ret × 3 × 7`

含组合算子但不含参数：`S_struct × 120 × S_rep × S_wt × S_org × S_evo × S_ret × 3 × 7`

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

`S_org` 同样不再是固定常数，而取决于：

- 可选的 `routing` 家族
- 可选的 `links` 子集
- 可选的 `placement`
- 以及 `StoreTopology` 提供的 capability 边界

`S_evo` 也不再是固定常数，而取决于：

- 可选的 `selection`
- 可选的 `action`
- 可选的 `effect`
- 以及 `evolution_trigger` 的触发条件

`S_ret` 也不再是固定常数，而取决于：

- 可选的 `signals`
- 可选的 `ranker`
- 可选的 `flow`
- 以及 `constraints` 参数空间

### 2.3 约束剪枝后的有效空间

根据兼容性约束估算约 5-10% 的组合是合法的：

- **不含组合算子**：~10^8 有效配置
- **含组合算子**：~10^11 有效配置

### 2.4 可行的搜索策略


| 空间规模        | 推荐策略                                         |
| ----------- | -------------------------------------------- |
| 10^4 以下     | 穷举 (grid search)                             |
| 10^4 - 10^6 | 随机搜索 + 约束剪枝                                  |
| 10^6 - 10^9 | 进化算法 / Bayesian optimization                 |
| 10^9 以上     | 分层搜索：先搜索 structural level，固定后搜索 modular，最后调参 |


推荐策略：**分层搜索**

```
Phase 1: 固定 LayeredStoreTopo + Trigger，搜索 9 个 modular slot（基础实现，不含组合算子）
          空间 ≈ 14 × S_rep × S_wt × S_org × S_evo × S_ret × 3
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


| System            | UF                  | Rep                                         | WT                                                                  | Org                                                                           | Evo                                                                                                                                     | Ret                                                                                 | Read   |
| ----------------- | ------------------- | ------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------ |
| Generative Agents | UF-1                | `{text, embedding}`                         | `signals={importance, novelty}, scorer=weighted_sum, policy=always` | `routing=default(reflections/episodes), links={temporal}, placement=append`   | `selection=time_window, action=reflect, effect=add, trigger=periodic`                                                                   | `signals={similarity, recency, importance}, ranker=weighted_sum, flow=single_stage` | Read-1 |
| MemGPT            | UF-3                | `{text, embedding}`                         | `gates={tool_called(...)}, policy=explicit_only`                    | `routing=agent_selected(...), links={}, placement=append`                     | `selection=layer_slice(main_context), action=summarize+move, effect=summarize+move, trigger=budget_exceeded`                            | `signals={similarity}, ranker=identity, flow=agent_invoked`                         | Read-4 |
| Reflexion         | UF-13               | `{text}`                                    | `gates={on_event("task_failure")}, policy=boolean_gate`             | `routing=default(reflection_layer), links={}, placement=append`               | `selection=layer_slice(reflection_layer), action=prune, effect=delete, trigger=after_write`                                             | `signals={}, ranker=identity, flow=single_stage`                                    | Read-1 |
| Voyager           | UF-10               | `{code, description, embedding}`            | `gates={on_event("task_success")}, policy=boolean_gate`             | `routing=by_tag(skill_type), links={}, placement=partition`                   | `selection=matched_by_key, action=upsert, effect=add+modify, trigger=after_write`                                                       | `signals={similarity}, ranker=identity, flow=single_stage`                          | Read-6 |
| MemoryBank        | Compose(UF-3, UF-6) | `{text, embedding}`                         | `policy=always`                                                     | `routing=by_unit_type(...), links={}, placement=append`                       | `selection=matched_by_entity + time_window, action=merge+summarize+prune, effect=modify+summarize+delete, trigger=session_end/periodic` | `signals={similarity, recency}, ranker=weighted_sum, flow=single_stage`             | Read-1 |
| SCM               | UF-6                | `{triple, embedding}`                       | `signals={llm_value_estimate}, scorer=llm_judge, policy=threshold`  | `routing=default(semantic_layer), links={entity}, placement=append`           | `selection=matched_by_entity, action=upsert+profile_update+review, effect=modify, trigger=after_write/periodic`                         | `signals={entity_match}, ranker=rerank_llm, flow=two_stage`                         | Read-4 |
| A-MEM             | UF-14               | `{text, embedding, triple, tags, entities}` | `policy=always`                                                     | `routing=default(graph_layer), links={graph_edge(...)}, placement=graph_node` | `selection=incoming_only, action=append, effect=add, trigger=after_write`                                                               | `signals={similarity, graph_proximity}, ranker=identity, flow=single_stage`         | Read-4 |
| TiM               | UF-11               | `{text, embedding}`                         | `gates={on_reasoning_step}, policy=boolean_gate`                    | `routing=default(thought_layer), links={temporal}, placement=append`          | `selection=layer_slice(thought_layer), action=summarize+prune, effect=summarize+delete, trigger=budget_exceeded`                        | `signals={recency, similarity}, ranker=weighted_sum, flow=single_stage`             | Read-1 |


### 从中可观察到的初步 Motif

**Motif 1: Embed-and-Retrieve（出现 7/8 次）**

- 表示 set 包含 `embedding` + retrieval signals 中包含 `similarity`
- 几乎所有系统都需要 embedding 作为基础检索手段

**Motif 2: Append-Dominant（出现 6/8 次）**

- `action=append` 是最常见的局部演化方式
- 只有涉及结构化 entity/profile 更新的系统才显式使用 `merge/upsert/profile_update`

**Motif 3: Selective-Write 分化**

- 简单系统多用 `policy=always` 或 `gates={on_event(...)}`
- 复杂系统更倾向于 `scorer=llm_judge` 或 `policy=explicit_only`
- 这是区分 passive vs active memory 的关键维度

**Motif 4: High-Level Evolution 可选**

- 并非所有系统都做高层演化；很多系统只做局部 append 或轻量 prune
- 需要高层演化的系统倾向于不同层次：`summarize` / `reflect` / `profile_update`

**Motif 5: Lifecycle Control 两极化**

- 要么只做非常轻的演化，要么明确做 `prune / move / summarize`
- 复杂的遗忘模型在实际系统中仍相对少见

---

## 4. 潜在的未探索区域（搜索的价值所在）

通过对已有工作的分解，可以识别出搜索空间中**尚未被探索的组合**：


| 未探索组合                                                                                                                                                                                                | 假设                 | 预期价值        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----------- |
| `UF-8` (Extract(triple)) + `Retrieval(signals={graph_proximity}, ranker=identity, flow=single_stage)` + `Evo(action=prune, selection=low_activity, trigger=periodic)`                                | 结构化图记忆 + 认知遗忘      | 长期对话中的知识演化  |
| `signals={surprise} + policy=threshold` + `Evo(action=reflect, selection=time_window, trigger=periodic)`                                                                                             | 只记意外 + 反思          | 高效学习型 agent |
| `Retrieval(signals={similarity, recency, diversity_penalty}, ranker=mmr, flow=two_stage)` + `Read-4` (TemplatedPrompt)                                                                               | 多信号重排 + 模板注入       | 自适应 memory  |
| `Compose(UF-3, UF-6, UF-12)` + `Organization(routing=default(...), links={parent_child}, placement=hierarchical_slot(...))` + `Retrieval(signals={hierarchy_match}, ranker=identity, flow=top_down)` | 多粒度 + 层级存储 + 层级检索  | 类人记忆层级      |
| `Evo(action=prototype_form, selection=layer_slice, trigger=periodic)` + `Evo(action=consolidate, selection=layer_slice, trigger=periodic)`                                                           | 原型形成 + 碎片整合        | 概念学习        |
| `signals={importance, novelty} + scorer=weighted_sum + policy=threshold` + `Evo(action=rewrite+summarize, selection=matched_by_key+time_window, trigger=after_write)`                                | 选择性写 + 动态改写 + 渐进摘要 | 活性记忆系统      |
