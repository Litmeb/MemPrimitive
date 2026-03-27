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

抽象上仍是 **representation element set**（每个 unit 选择携带哪些表示元素）。下面 **B.1** 与当前 `memprimitive.baselines.representation` 实现一一对应；**B.2** 保留搜索空间里尚未在基线中落地的元素，便于和 DSL 对照。

### B.1 Stage-1 基线：`BasicRepresentation` / `KeywordRepresentation`

- **模块**：`BasicRepresentation`、`KeywordRepresentation`（后者是薄封装，默认 `elements=("text", "keywords", "tags")`，其余构造函数参数与前者相同）。
- **输入**：`run` 要求 `packet.units` 已存在；**不修改** `MemoryStore`。
- **配置**：构造函数 `elements: tuple[str, ...]`，会去重并保持顺序；非法名字在 `__init__` 时 `ValueError`。
- **合法元素名**：与源码 `_VALID_ELEMENTS` 一致，共 12 个（**没有** `frame` / `code` / `sparse_embedding` 等抽象表中的项）。

下列表格说明：**元素名**、**写入位置**（`MemoryUnit` 字段或 `metadata["representation"]`）、**是否进入** `representation_elements`、**实现逻辑摘要**。

| 元素名 | 主要写入位置 | 进入 `representation_elements` | 行为摘要 |
| ------ | ------------ | -------------------------------- | -------- |
| `text` | `unit.text`（strip 后） | 是（始终可标） | 规范化空白文本；`normalized_text` 为 casefold 全文。 |
| `embedding` | `unit.embedding: list[float]` | 是（成功编码后） | `sentence_transformers.SentenceTransformer`（默认 `MEMPRIMITIVE_EMBEDDING_MODEL` 或 `all-MiniLM-L6-v2`），`normalize_embeddings=True`；模型按名缓存。 |
| `triple` | `unit.triples` | 仅当列表非空 | 优先 `unit.metadata["triples"]` 中形如 `[s,p,o]` 的项；否则用正则从正文抽 “X likes/prefers/… Y”“X is Y” 模式。 |
| `kv` | `unit.kv` | 仅当 dict 非空 | 优先 `unit.metadata["kv"]`；否则 `Key: value` 行模式 + likes/is 模式生成键值。 |
| `entities` | `unit.entities` | 仅当列表非空 | 优先 `unit.metadata["entities"]`；否则大写起头的连续词块（启发式 NER），过滤 the/a/an。 |
| `tags` | `unit.tags` | 仅当列表非空 | 优先 `unit.metadata["tags"]`；否则 `unit_type` + 关键词表（graph/memory/code/…）+ 若已有 entities/kv/triples 则加 `entity_rich` / `structured_kv` / `structured_triple`。 |
| `keywords` | `metadata["representation"]["keywords"]` | 仅当列表非空 | 优先 `unit.metadata["keywords"]`；否则正文词频（去停用词，最多 6 个）并并入 entities/tags 的补充词。 |
| `summary` | `metadata["representation"]["summary"]` | 仅当生成非空 | 启发式一句摘要：优先首条 triple，否则首条 kv，否则前两个实体 + 文本前缀，否则截断正文（≤96 字符）。 |
| `time_anchor` | `metadata["representation"]["time_anchor"]` | 仅当 dict 非空 | 优先 `unit.metadata["time_anchor"]`（dict）；否则用 `unit.timestamp` 拆出 `timestamp` / `date`。 |
| `relation_tags` | `metadata["representation"]["relation_tags"]` | 仅当列表非空 | 优先 `unit.metadata["relation_tags"]`；否则由 triple 的 predicate 生成 `predicate:...`，实体数≥2 时加 `multi_entity`。 |
| `source_type` | `metadata["representation"]["source_type"]` | 仅当非空 | 来自 `unit.metadata["source"]` 字符串 strip。 |
| `description` | `unit.description` | 仅当最终非空 | 若 `unit.description` 已有则保留；否则若配置了 `api_key` + `base_url` + `model` 则调 OpenAI Chat 生成一句描述；否则退回与 `summary` 相同的启发式或原文。 |

此外每次 `run` 都会把合并后的摘要写入 `unit.metadata["representation"]`（含 `_representation_summary_from_unit` 与上表中的扩展键）。`representation` slot 的 trace 在 `packet.trace["representation"]`（模块名、`elements`、逐 unit 的 `representation_elements`）。

### B.2 抽象搜索空间中的其它表示元素（基线未实现）

下列在概念文档/检索设计中常见，但 **当前 `_VALID_ELEMENTS` 不包含**，需自定义 `RepresentationModule` 或其它阶段补齐：

| 概念元素 | 含义 |
| -------- | ---- |
| `frame` | slot / frame 填充结构 |
| `code` | 可执行代码或片段 |
| `sparse_embedding` | TF-IDF / BM25 等稀疏向量 |

### B.3 变异轴与约束（与实现对齐部分）

- **元素选择**：在 B.1 的 12 个名字中做子集选择；`KeywordRepresentation` 默认偏检索：`text + keywords + tags`。
- **向量检索**：需要 `embedding` 元素（及 `EmbeddingSimilarityRetrieval` 等）；依赖本地 SentenceTransformer。
- **关键词/标签类检索**：`keywords` / `tags` / `entities` 与 `KeywordCountRetrieval`、`TagRetrieval`、`EntityRetrieval` 等配合；部分 signal（如 `KeywordMatchSignal`）读 `metadata["representation"]["keywords"]`。
- **结构化**：`triple` / `kv` 可由 UF 的 metadata 预填，也可全靠 representation 内规则从文本抽。

---

## C. Trigger — 触发决策

**核心问题**：什么时候该触发某个 memory 生命周期动作？

Stage-1 把可复用逻辑收在 **`memprimitive.baselines._trigger_family`**，由四个**组件角色**拼成一条每个 unit 的决策链（不是 pipeline 的 DSL slot 名，而是 family 内部子结构）：

```text
TriggerFamily = signal_providers（多路 signal） + scorer + gate + policy
```

运行时 **`TriggerFamilyRunner`** 对每个 `packet.units[i]` 顺序执行：合并所有 `SignalProvider.provide` → `ScoreAggregator.score` → `Gate.evaluate` → `DecisionPolicy.decide`，得到 `bool` 写入 `Packet.decisions`（write）或 `Packet.evolution_decisions`（evolution）。

* **`write_trigger`**：默认基线 `AlwaysWriteTrigger`、`ThresholdWriteTrigger`；可用 `compose_write_trigger(...)` 自定义 family。
* **`evolution_trigger`**：默认 `NeverEvolutionTrigger`、`ThresholdEvolutionTrigger`；可用 `compose_evolution_trigger(...)`。适配器默认要求 `packet.placements` 非空（`required_fields`）。

共享上下文 **`TriggerContext`**（frozen dataclass）：`packet`、`store`、`output_field`、`trace_key`（供 gate/signal 读 packet 状态；`store` 当前 runner 内未强制使用，但保留扩展点）。

### C.1 四个组件的抽象接口（必须实现的方法）

| 角色 | ABC | 实例属性 | 抽象方法 |
| ---- | --- | -------- | -------- |
| Signal | `SignalProvider` | `name: str`（实现类用 `@property`） | `provide(self, context: TriggerContext, unit_index: int) -> SignalMap`，`SignalMap = dict[str, float \| bool]`；可对同一 unit 返回多个键。 |
| Scorer | `ScoreAggregator` | `name: str` | `score(self, signals: SignalMap) -> float` |
| Gate | `Gate` | `name: str` | `evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool`（`True` 表示通过硬门控） |
| Policy | `DecisionPolicy` | `name: str` | `decide(self, *, score: float, gate_open: bool) -> bool`（最终是否触发） |

Runner 内对多个 signal provider 的返回值做 **`signals.update(provided)`**，故后者同名键会覆盖前者。

### C.2 Signal 组件（`SignalProvider` 实现类）

| 类名 | 实现的抽象方法 | 默认输出信号键 | 行为与前置条件 |
| ---- | -------------- | -------------- | -------------- |
| `ConstantSignal` | `provide` | `signal_name`（默认 `"constant"`） | 常数 `float(value)`。 |
| `UnitLengthSignal` | `provide` | `signal_name`（默认 `"unit_length"`） | 当前 unit 文本 strip 后长度；若 `normalize_by>0` 则除以该值。 |
| `KeywordMatchSignal` | `provide` | `signal_name`（默认 `"keyword_match"`） | **`packet.query` 必须非空**；query 词与 `representation["keywords"]` 及 unit 文本词的交集数量（float）。 |
| `HasEntitySignal` | `provide` | `signal_name`（默认 `"has_entity"`） | 有实体则 `1.0` 否则 `0.0`。 |
| `HasTripleSignal` | `provide` | `signal_name`（默认 `"has_triple"`） | 有 triple 则 `1.0` 否则 `0.0`。 |
| `HasKVSignal` | `provide` | `signal_name`（默认 `"has_kv"`） | 有 kv 则 `1.0` 否则 `0.0`。 |
| `TagMatchSignal` | `provide` | `signal_name`（默认 `"tag_match"`） | **`packet.query` 必须非空**；query 词与 `unit.tags` 交集数量。 |
| `LayerTargetSignal` | `provide` | `signal_name`（默认 `"layer_target"`） | **`packet.placements` 必须非空**；`placements[unit_index].target_layer` 是否在 `allowed_layers` 内，是则 `1.0` 否则 `0.0`。 |
| `QueryOverlapSignal` | `provide` | `signal_name`（默认 `"query_overlap"`） | **`packet.query` 必须非空**；query 与 unit 文本的 token 交集数量。 |

### C.3 Scorer 组件（`ScoreAggregator` 实现类）

| 类名 | 实现的抽象方法 | 行为 |
| ---- | -------------- | ---- |
| `IdentityScorer` | `score` | `signals[source]` 转为 `float`；缺键则 `ValueError`。 |
| `WeightedSumScorer` | `score` | `sum(signals[k] * weights[k])`；缺任一键则 `ValueError`。 |
| `MaxScorer` | `score` | `sources` 中信号的最大值。 |
| `MinScorer` | `score` | `sources` 中信号的最小值。 |
| `AverageScorer` | `score` | `sources` 中信号的算术平均。 |
| `ClippedWeightedSumScorer` | `score` | 先按 `WeightedSumScorer` 求和，再 clip 到 `[min_score, max_score]`。 |

### C.4 Gate 组件（`Gate` 实现类）

| 类名 | 实现的抽象方法 | 行为与前置条件 |
| ---- | -------------- | -------------- |
| `AlwaysOpenGate` | `evaluate` | 恒为 `True`。 |
| `RequireEntityGate` | `evaluate` | 当前 unit 有 `entities` 才 `True`。 |
| `RequireTripleGate` | `evaluate` | 当前 unit 有 `triples` 才 `True`。 |
| `RequireTagGate` | `evaluate` | `unit.tags` 与 `required_tags`（casefold）有交集才 `True`。 |
| `LayerAllowedGate` | `evaluate` | **`packet.placements` 必须非空**；`target_layer` 必须在 `allowed_layers` 中。 |
| `QueryPresentGate` | `evaluate` | `packet.query is not None` 时 `True`。 |

### C.5 Policy 组件（`DecisionPolicy` 实现类）

| 类名 | 实现的抽象方法 | 行为 |
| ---- | -------------- | ---- |
| `AlwaysPolicy` | `decide` | 恒 `True`（忽略 score/gate）。 |
| `NeverPolicy` | `decide` | 恒 `False`。 |
| `ThresholdPolicy` | `decide` | `gate_open and score >= threshold`。 |
| `BooleanGatePolicy` | `decide` | 直接返回 `gate_open`。 |
| `BandPassThresholdPolicy` | `decide` | `gate_open and lower <= score <= upper`。 |
| `ThresholdOrGatePolicy` | `decide` | `gate_open or score >= threshold`。 |

### C.6 `TriggerFamilyRunner.run` 与 trace

- **前置**：`packet.units` 非空；并按适配器传入的 `required_fields` 检查 `query` / `placements` 等字段非 `None`。
- **输出**：在 `packet` 上设置 `output_field`（`decisions` 或 `evolution_decisions`）为 `list[bool]`；`trace[trace_key]` 含 `family`、`policy`/`scorer`/`gate` 的 `name`、`per_unit`（每单元的 signals/score/gate/decision）。

### C.7 默认基线与组合 API

| Pipeline slot | 注册类 | 内置 family（摘要） |
| ------------- | ------ | ------------------- |
| `write_trigger` | `AlwaysWriteTrigger` | `ConstantSignal(1.0)` + `IdentityScorer("constant")` + `AlwaysOpenGate` + `AlwaysPolicy` |
| `write_trigger` | `ThresholdWriteTrigger` | 同上 signal/scorer/gate + `ThresholdPolicy(threshold)`（score 实为 `constant` 加权 1.0） |
| `evolution_trigger` | `NeverEvolutionTrigger` | 同上 + `NeverPolicy` |
| `evolution_trigger` | `ThresholdEvolutionTrigger` | 同上 + `ThresholdPolicy(threshold)` |

自定义： **`compose_write_trigger`** / **`compose_evolution_trigger`**（`write_trigger.py` / `evolution_trigger.py`），传入 `signal_providers`、`scorer`、`gate`、`policy` 及可选 `input_requirements`（非 `units` 的项会进入 `required_fields`，在 runner 里强制 packet 字段存在）。

### C.8 概念扩展（当前代码中未提供的组件形态）

下表便于与更宽的搜索空间对照；**未在 `_trigger_family.py` 中实现**：

- Signal：`importance` / `novelty` / `surprise` / `duplicate_risk`、LLM 估计等。
- Scorer：`product`、`rule_expr`、纯 `llm_judge`。
- Policy：`top_k_per_window`、`sample_by_score`、`explicit_only`（工具调用门控）等。
- Gate：`on_event`、`not_duplicate`、`tool_called` 等事件/工具谓词。

### C.9 常见策略到此框架的映射（概念）


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

### Stage-1 Runtime Mapping

在当前 `memprimitive` 实现中，ingest 顺序为：

```text
unit_formation -> representation -> write_trigger -> organization -> evolution_trigger -> memory_evolution
```

对应的数据面约定是：

* `write_trigger` 产出 `Packet.decisions`
* `organization` 读取 `decisions` 并完成常规写入
* `evolution_trigger` 产出 `Packet.evolution_decisions`
* `memory_evolution` 读取 `evolution_decisions` 并执行额外演化

如果某个 runtime 仍允许 `memory_evolution` 在 `evolution_decisions` 为空时回退到 `decisions`，应视为向后兼容行为，而不是目标语义。


---

## D. Organization — 关系归组、放置与常规写入

**核心问题**：unit 应该放在哪里？与已有 memory 建立什么关系？在 ingest-time 常规写入路径上如何正式落入 store？

这里不再把 organization 建模成很多命名策略，而是统一写成：

```text
Organization = placement + links
```

其中 **`placement` 表示 unit 被送到哪里**（目标 layer / 分区 / 显式落点），不再单独使用 `routing` 一词；到达目标层之后**如何写入、如何与结构结合**（append、分桶、聚类、建图节点等）在文档里称为**层内写入形态**，与 `placement` 的目标选择正交，见下节第二张表。

但它的 contract 需要改成：

```text
Organization handles ingest-time organization and normal write.
```

并且显式承认：

```text
Organization is topology-constrained.
StoreTopology defines the admissible placement targets, link types, and within-layer write shapes.
```

### 组织组件


| 组件          | 含义                   | 典型取值                                                                                   |
| ----------- | -------------------- | -------------------------------------------------------------------------------------- |
| `placement` | unit **送到哪一层 / 哪一分区**（目标落点） | `default`, `by_unit_type`, `by_tag`, `by_rule`, `explicit`, `agent_selected`           |
| `links`     | 写入时与已有 memory 建立哪些关系 | `temporal`, `entity`, `similarity`, `cluster_membership`, `graph_edge`, `parent_child` |


### 常见 placement（目标层 / 分区）


| placement        | 含义                | 关键参数                 |
| ---------------- | ----------------- | -------------------- |
| `default`        | 总是写入默认 layer      | target_layer         |
| `by_unit_type`   | 按 unit_type 选择目标层   | route_map            |
| `by_tag`         | 按 tag 或 topic 选择目标层 | tag_field, route_map |
| `by_rule`        | 按规则条件选择目标层       | rules                |
| `explicit`       | 上游字段中已带有目标位置      | field                |
| `agent_selected` | agent 通过工具或字段指定目标 | tool_to_layer        |


### 层内写入形态（到达目标层之后）

与「送到哪」正交：先由 `placement` 选定目标层/分区，再决定在该层内如何落地。


| 写入形态              | 含义              | 关键参数                               |
| ------------------- | --------------- | ---------------------------------- |
| `append`            | 直接追加到目标 layer   | —                                  |
| `partition`         | 写入某个分区 / bucket | partition_key                      |
| `cluster`           | 放入最近的 cluster   | similarity_threshold, max_clusters |
| `graph_node`        | 在图层中创建节点并接边     | node_policy                        |
| `hierarchical_slot` | 放入多层结构中的某一层/槽位  | target_level, slot_policy          |


### 常见 links


| link                 | 含义                       | 关键参数                          |
| -------------------- | ------------------------ | ----------------------------- |
| `temporal`           | 建立时间邻接关系                 | window, direction             |
| `entity`             | 按共享实体建立关联                | entity_field, link_type       |
| `similarity`         | 建立相似度近邻关系                | metric, threshold             |
| `cluster_membership` | 连接到某个 cluster / centroid | cluster_key                   |
| `graph_edge`         | 在图层中创建有类型边               | edge_types, connection_method |
| `parent_child`       | 建立层级父子关系                 | level_policy                  |


### 变异轴

- **目标落点（placement）**：去哪一层 / 哪个分区
- **关系结构（links）**：建立哪些 link
- **层内写入形态**：append / partition / cluster / graph / hierarchical
- **常规写入方式**：由 organization strategy 隐含决定的 ingest-time update
- **结构约束来源**：由 StoreTopology 决定可选能力边界

### Topology-Constrained Organization

- `StoreTopology` 决定 admissible placement targets：没有的 layer 不能作为落点。
- `StoreTopology` 决定 admissible link types：例如没有 graph layer 或 `graph` index，就不该允许 `graph_edge`。
- `StoreTopology` 约束层内写入形态：例如单层 flat store 不应该允许 `hierarchical_slot`。
- `Organization` 负责在这些能力边界内，选择具体的 placement / links / 层内写入形态，并完成常规写入。
- **层内 `append`** 不再意味着“只生成计划”，而意味着 append-style normal write（与当前 stage-1 基线中 `MemoryStore.append` 一致）。
- 某些 organization 变体可以包含与 placement（目标或层内形态）强耦合的 ingest-time merge / upsert / replace。
- 这些 ingest-time update 不单独拆成新 slot，因为它们不构成独立搜索轴。

### 关键约束

- `links contains entity` 要求 unit 有 `entities` 字段 → 需要 UF 产出 entity 或 Rep 补充 entity
- `placement=by_tag` 要求 unit/representation 中有 `tags`
- 层内形态 `cluster` 或 `links contains similarity` 通常要求 `embedding`
- 层内形态 `graph_node` 或 `links contains graph_edge` 要求存在 `shape=Graph` 的 layer，且通常要求 `graph` index
- 层内形态 `hierarchical_slot` 或 `links contains parent_child` 要求 layer_count > 1
- `placement=by_unit_type`、`by_rule`、`agent_selected` 等都要求对应目标 layer 在当前拓扑中存在

### 旧命名策略到新框架的映射

下文 **`placement=`** 仅表示**目标落点**（送到哪一层 / 哪一分区）；**`write=`** 表示**层内写入形态**（与上文「层内写入形态」表一致），避免与 `placement` 语义混淆。

| 旧写法                     | 新写法                                                                  |
| ----------------------- | -------------------------------------------------------------------- |
| `FlatAppend`            | `placement=default, links={}, write=append`                        |
| `TemporalAppend`        | `placement=default, links={temporal}, write=append`                |
| `EntityLinked`          | `placement=default, links={entity}, write=append`                  |
| `TemporalAndEntity`     | `placement=default, links={temporal, entity}, write=append`        |
| `GraphPlacement`        | `placement=default, links={graph_edge(...)}, write=graph_node`     |
| `HierarchicalPlacement` | `placement=default, links={parent_child}, write=hierarchical_slot` |
| `DualStoreRouter`       | `placement=by_unit_type(...), links={}, write=append`              |
| `AgentExplicitTarget`   | `placement=agent_selected(...), links={}, write=append`            |
| `SimilarityCluster`     | `placement=default, links={cluster_membership}, write=cluster`     |
| `TagBasedPlacement`     | `placement=by_tag(...), links={}, write=partition`                 |


---

## E. Memory Evolution — 记忆演化

**核心问题**：在常规写入已经完成之后，store 是否还需要额外的演化？**

这里统一吸收原来的 `Compression`、`Maintenance`，以及那些**默认不启动、额外触发** 的重写/重整操作。  
它不再承担普通 append 式落库。

```text
MemoryEvolution = selection + action + effect + trigger
```

### 演化组件


| 组件          | 含义                 | 典型取值                                                                                                                                                                                |
| ----------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `selection` | 选中哪些已有 memory 参与演化 | `matched_by_key`, `matched_by_entity`, `time_window`, `layer_slice`, `low_activity`, `all`                                                                                           |
| `action`    | 对选中对象施加什么演化操作      | `replace`, `merge`, `upsert`, `rewrite`, `summarize`, `reflect`, `profile_update`, `extract_concept`, `prototype_form`, `prune`, `dedup`, `move`, `consolidate`, `review`         |
| `effect`    | 对 store 造成何种状态变化   | `add`, `modify`, `delete`, `move`, `summarize`, `merge`, `version`                                                                                                                  |
| `trigger`   | 何时执行该演化操作          | `periodic`, `on_event`, `budget_exceeded`, `count_exceeded`, `conditional`                                                                                                          |


### 常见 selection


| selection           | 含义              | 关键参数                      |
| ------------------- | --------------- | ------------------------- |
| `matched_by_key`    | 按 key 匹配已有 unit | key_field, match_strategy |
| `matched_by_entity` | 按 entity 匹配已有记忆 | entity_field              |
| `time_window`       | 处理某个时间窗口内的记忆    | start, end / window_size  |
| `layer_slice`       | 处理某层或某分区中的对象    | target_layer, filter      |
| `low_activity`      | 处理低活跃、低价值对象     | activity_threshold        |
| `all`               | 处理整个目标集合        | scope                     |


### 常见 action


| action             | 含义              | 对应旧名                   |
| ------------------ | --------------- | ---------------------- |
| `replace`          | 替换匹配对象          | Upd-2                  |
| `merge`            | 合并新旧对象          | Upd-3, Upd-7           |
| `upsert`           | 有则更新，无则插入       | Upd-4                  |
| `rewrite`          | LLM 或规则重写已有内容   | Upd-6                  |
| `delta`            | 只记录变化量          | Upd-5                  |
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

- **演化目标**：额外重整 / 高层抽象 / 生命周期管理
- **作用范围**：by-entity / by-window / by-layer / global
- **状态变化**：add / modify / delete / move / summarize / version
- **触发模式**：周期 / 事件 / 预算 / 条件

### 建模说明

- 常规写入已经由 D 槽 `organization` 完成。
- E 槽只表示额外触发的 memory evolution。
- 原 `Compression` 是一种高层抽象演化；原 `Maintenance` 是一种生命周期演化。
- `AppendWithLinkUpdate` 不再单列；link 更新原则上应回到 D 槽，E 只负责 extra evolution over existing memory。
- `summarize`、`prune`、`move`、`consolidate` 与 `merge` 一样，都是对 store state 的不同额外演化方式。
- `replace/merge/upsert/rewrite` 只有在被理解为额外触发的已有记忆重整时，才属于 E 槽。

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
| `requires(Org.write=graph_node, exists layer.shape == Graph)`              | 图节点层内写入要求存在 graph layer |
| `requires(Org.links contains graph_edge, exists layer.indices contains graph)` | 图边要求 graph index      |
| `requires(Org.write=hierarchical_slot, layer_count > 1)`                   | 层级槽位写入要求多层拓扑            |
| `requires(Org.placement=by_unit_type, layer_count > 1)`                          | 按 unit 类型分落点通常要求多个目标 layer   |
| `requires(Evo.action=move, layer_count > 1)`                                   | 层间迁移要求多层拓扑            |


### 1.2 Module → Module 约束（上游输出限制下游选择）


| 约束                                                                                                   | 含义                               |
| ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| `requires(Ret.signals contains similarity, Rep-* produces embedding)`                                | 向量检索要求有 embedding                |
| `requires(Org.links contains entity, UF/Rep produces entities)`                                      | entity link 要求有实体                |
| `requires(Org.placement=by_tag, Rep/UF produces tags)`                                                 | 按 tag 选落点要求 tags              |
| `requires(Org.write=cluster, Rep produces embedding)`                                            | cluster 层内写入要求 embedding           |
| `requires(Evo.action in {merge, profile_update}, UF/Rep produces entities)`                          | entity 合并/画像更新要求有实体              |
| `requires(Evo.action=delta, UF produces alignable units)`                                            | delta 演化要求可对齐                    |
| `requires(Ret.signals contains graph_proximity, Org.links contains graph_edge)`                      | graph proximity 要求有图边            |
| `requires(Read-6, UF-10 produces code)`                                                              | CodeInjection 要求代码 unit          |
| `requires(Ret.flow=top_down or Ret.signals contains hierarchy_match, Evo.action produces hierarchy)` | top-down/hierarchy 检索要求层级摘要/父子层级 |


### 1.3 语义耦合约束（强烈推荐的组合模式）


| 约束                                                                                 | 含义                            |
| ---------------------------------------------------------------------------------- | ----------------------------- |
| `co_requires(UF-6/7/8/9, Evo.action in {merge, upsert, profile_update})`           | 结构化 unit 与额外实体/键控演化强关联        |
| `co_requires(UF-2/3, Org.write=append)`                                        | 对话级 unit 与 append-style organization 强关联 |
| `co_requires(Evo.action=profile_update, UF-7)`                                     | profile 更新与 EntityState 抽取强关联 |
| `co_requires(Org.links contains graph_edge, Ret.signals contains graph_proximity)` | 图边组织与图邻近检索强关联                 |
| `co_requires(Rep contains code, Read-6)`                                           | 代码表示与 CodeInjection 强关联       |


### 1.4 互斥约束


| 约束                                                                              | 含义          |
| ------------------------------------------------------------------------------- | ----------- |
| `incompatible(WT.policy=never, Org.write=append)`                           | 从不写入与 append-style organization 互斥 |


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

- 可选的 **placement**（目标落点）策略家族
- 可选的 **links** 子集
- 可选的**层内写入形态**（`append` / `partition` / …）
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


| System            | UF                  | Rep                                         | WT                                                                  | Org                                                                                                        | Evo                                                                                                                             | Ret                                                                                 | Read   |
| ----------------- | ------------------- | ------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------ |
| MemGPT            | UF-3                | `{text, embedding}`                         | `gates={tool_called(...)}, policy=explicit_only`                    | `placement=agent_selected(...), links={}, write=append`                                                  | `selection=layer_slice(main_context), action=summarize+move, effect=summarize+move, trigger=budget_exceeded`                    | `signals={similarity}, ranker=identity, flow=agent_invoked`                         | Read-4 |
| Reflexion         | UF-13               | `{text}`                                    | `gates={on_event("task_failure")}, policy=boolean_gate`             | `placement=default(reflection_layer), links={}, write=append`                                            | `selection=layer_slice(reflection_layer), action=prune, effect=delete, trigger=on_event("task_failure")`                        | `signals={}, ranker=identity, flow=single_stage`                                    | Read-1 |
| A-MEM             | UF-14               | `{text, embedding, triple, tags, entities}` | `policy=always`                                                     | `placement=default(graph_layer), links={graph_edge(...)}, write=graph_node`                              | `Trig-1 Never`                                                                                                                  | `signals={similarity, graph_proximity}, ranker=identity, flow=single_stage`         | Read-4 |


### 从中可观察到的初步 Motif

**Motif 1: Embed-and-Retrieve（出现 7/8 次）**

- 表示 set 包含 `embedding` + retrieval signals 中包含 `similarity`
- 几乎所有系统都需要 embedding 作为基础检索手段

**Motif 2: Organization-Driven Normal Write（出现 6/8 次）**

- append-style normal write 往往由 `organization` 完成
- 只有涉及额外 entity/profile 更新或高层整理的系统才显式使用 `memory_evolution`

**Motif 3: Selective-Write 分化**

- 简单系统多用 `policy=always` 或 `gates={on_event(...)}`
- 复杂系统更倾向于 `scorer=llm_judge` 或 `policy=explicit_only`
- 这是区分 passive vs active memory 的关键维度

**Motif 4: High-Level Evolution 可选**

- 并非所有系统都做高层演化；很多系统只做 organization-driven normal write，额外演化保持关闭或极轻量
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
| `Compose(UF-3, UF-6, UF-12)` + `Organization(placement=default(...), links={parent_child}, write=hierarchical_slot(...))` + `Retrieval(signals={hierarchy_match}, ranker=identity, flow=top_down)` | 多粒度 + 层级存储 + 层级检索  | 类人记忆层级      |
| `Evo(action=prototype_form, selection=layer_slice, trigger=periodic)` + `Evo(action=consolidate, selection=layer_slice, trigger=periodic)`                                                           | 原型形成 + 碎片整合        | 概念学习        |
| `signals={importance, novelty} + scorer=weighted_sum + policy=threshold` + `Evo(action=rewrite+summarize, selection=matched_by_key+time_window, trigger=after_write)`                                | 选择性写 + 动态改写 + 渐进摘要 | 活性记忆系统      |
