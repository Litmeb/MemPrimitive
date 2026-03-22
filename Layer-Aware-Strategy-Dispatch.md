# Layer-Aware Strategy Dispatch 草案

本文档讨论一个紧接在 Store Topology 之后的设计问题：

> 当 memory store 已经具备显式 topology 之后，不同 layer 是否应该拥有不同的 `representation`、`retrieval`、`memory_evolution` 策略？

本文的结论是：

- 这个方向是合理的，而且是 topology 语义自然延伸出来的结果
- 但不建议立刻把系统做成“每层一条完整独立 pipeline”
- 更合适的方式是：**保持全局 slot 不变，在 slot 内部引入 layer-aware dispatch**

---

## 1. 问题背景

在最初的 stage-1 实现里，`MemoryStore` 的 layer 更像是：

- 一个逻辑分组
- 一个 record bucket
- 一个 `target_layer` 标签

这时不同 layer 即使名称不同，行为也未必真正不同。

但当 topology 被显式建模后，每个 layer 开始具有自己的：

- `theme`
- `shape`
- `indices`
- `capacity`

此时 layer 已经不只是“放在哪”，而是在逐渐变成“具备不同能力边界的 memory region”。

因此，一个自然的问题就出现了：

> 既然 layer 的能力和角色不同，是否也应该允许它们拥有不同的表示、检索、演化策略？

答案是：**应该允许，但要控制实现方式。**

---

## 2. 为什么这是合理的

不同 memory layer 通常承担不同职能。

例如：

- `working` 层更偏近期、快速访问、轻量 append
- `episodic` 层更偏时间序列回忆和局部摘要
- `semantic` 层更偏概念融合、事实合并、结构化更新
- `profile` 层更偏实体属性、偏好、稳定状态

这些差异天然对应不同策略。

### 2.1 不同 layer 的 `retrieval` 可能不同

例如：

- `working`：更偏 `recency`
- `episodic`：更偏 `recency + keyword`
- `semantic`：更偏 `similarity + entity_match`
- `profile`：更偏 `exact match / key lookup`

### 2.2 不同 layer 的 `memory_evolution` 可能不同

例如：

- `working`：`append + prune`
- `episodic`：`append + periodic summarize`
- `semantic`：`merge / upsert / consolidate`
- `profile`：`overwrite / profile_update / conflict_resolution`

### 2.3 不同 layer 的 `representation` 也可能不同

例如：

- `working`：`{text}`
- `episodic`：`{text, normalized_text}`
- `semantic`：`{text, embedding, entities, triple}`
- `profile`：`{kv}` 或 `{entity_state}`

所以，从概念上讲，layer-specific strategy 不是额外复杂化，而是拓扑设计真正开始产生行为后果。

---

## 3. 需要避免的错误方向

虽然 layer-aware strategy 很合理，但有一种实现方式不建议采用：

> 为每一个 layer 建立一整条独立 pipeline

即把系统变成：

```text
working_pipeline
episodic_pipeline
semantic_pipeline
profile_pipeline
```

这种方式的问题是：

- 配置和组合空间迅速爆炸
- 许多逻辑被复制到每层
- 跨层行为难以统一描述
- 调试和实验成本大幅上升
- `MemoryPipeline` 的当前 slot 结构会被破坏

因此，本文不建议“per-layer full pipeline”设计。

---

## 4. 推荐方向：slot 不变，slot 内部 layer-aware

更好的设计是：

```text
one pipeline, but some slots are layer-aware
```

也就是说：

- `MemoryPipeline` 仍然保持全局固定 slot
- 但 slot 内部可以根据 layer 选择不同策略

例如：

- `retrieval` 仍然是一个 slot
- 但它内部会：
  - 决定要访问哪些 layer
  - 对不同 layer 使用不同 retriever
  - 合并各层结果

同理：

- `memory_evolution` 仍然是一个 slot
- 但它内部会根据 `placement.target_layer` 分派给不同 evolution 策略

再比如：

- `representation` 仍然是一个 slot
- 但它可以根据目标 layer 或 route 结果，应用不同 representation builder

这保持了两个重要优点：

1. 外部组合边界不变
2. layer-specific 行为可以逐步增长

---

## 5. 一个核心判断：layer 是“存储位置”还是“行为域”

这个问题对架构非常关键。

如果 layer 只是：

- 一个 record 存放位置

那么其实不需要 per-layer strategy。

但如果 layer 是：

- 一个具有 capability、policy、capacity 的 memory region

那么 layer-specific strategy 就是自然的。

本文建议把 layer 理解为：

> **具有能力边界和策略约束的 memory region**

这意味着：

- layer 不直接决定具体算法
- 但它定义了模块选择的上下文边界

例如：

- `semantic` 层可以支持 `embedding`、`entity`、`merge`
- `working` 层可能只支持简单 text append
- `graph` 形态的 layer 才允许 graph-specific organization / retrieval

这样，layer 不是 module bundle，但它也不只是 bucket。

---

## 6. 建议采用的抽象关系

推荐保持下面这个关系：

```text
topology defines capability boundaries
module chooses behavior
slot may dispatch behavior by layer
```

换句话说：

- topology 定义 layer 的边界与能力
- module 决定具体策略
- module 可以在内部按 layer dispatch
- 但 layer 本身不等于一个固定策略包

这有助于避免一个常见问题：

> layer 名称偷偷编码一整套系统行为

例如，如果你把：

- `semantic layer = triple rep + entity retrieval + upsert evolution`

直接写死进 layer 语义，那么 topology 和 module 就重新耦合了。

更稳妥的做法是：

- topology 只说 semantic layer 允许什么
- module 决定当前是否真的使用这些能力

---

## 7. 最适合先做 layer-aware 的 slot

建议按以下顺序推进。

### 7.1 第一优先级：`retrieval`

这是最适合最先做 layer-aware dispatch 的 slot。

原因：

- retrieval 天然就是“从哪些 layer 取”
- 不同 layer 用不同检索逻辑很容易解释
- 这一步不必立刻改变写入路径

一个自然的结构是：

```text
layer-aware retrieval =
    select layers
  + run retriever per layer
  + merge results
```

例如：

- `working -> RecencyRetrieval`
- `episodic -> RecencyKeywordRetrieval`
- `semantic -> SimilarityRetrieval`
- `profile -> ExactMatchRetrieval`

然后统一 merge。

### 7.2 第二优先级：`memory_evolution`

这也是非常自然的 layer-aware slot。

原因：

- evolution 本来就和目标 layer 强相关
- `placement.target_layer` 已经提供了分派信号

一个自然的结构是：

```text
layer-aware evolution =
  group units by target_layer
  dispatch to per-layer evolution strategy
```

例如：

- 写到 `working` 的 unit 用 `AppendOnlyEvolution`
- 写到 `semantic` 的 unit 用 `UpsertEvolution`
- 写到 `profile` 的 unit 用 `ProfileUpdateEvolution`

### 7.3 第三优先级：`representation`

representation 也适合 layer-aware，但应放得更晚一些。

原因：

- 它和 `unit_formation`、`organization`、`retrieval` 都有较强联动
- 它需要更清楚的“目标 layer 已确定”语义

一个更稳的做法是：

- 先让 routing / placement 更成熟
- 再让 representation 依据目标 layer 应用不同 element builders

---

## 8. 推荐的配置思路：default + override

为了避免配置爆炸，建议采用：

```text
global default strategy
+ per-layer overrides
```

例如：

```text
retrieval:
  default = RecencyRetrieval
  overrides:
    semantic = SimilarityRetrieval
    profile = ExactMatchRetrieval
```

```text
memory_evolution:
  default = AppendOnlyEvolution
  overrides:
    semantic = UpsertEvolution
    profile = ProfileUpdateEvolution
```

```text
representation:
  default = BasicRepresentation
  overrides:
    semantic = StructuredRepresentation
    profile = KVRepresentation
```

这种方式的好处是：

- 单层系统天然兼容
- 多层系统不需要每层都写一遍
- 只有少数特别层需要 override
- 搜索空间更容易控制

---

## 9. 运行时草案

下面给出一个偏 stage-1 / stage-2 过渡的运行时草案。

### 9.1 Layer-aware Retrieval

```text
Query
  -> choose active layers
  -> run retriever per layer
  -> merge retrieved sets
  -> readout
```

一个可能的接口草案：

```python
class LayerAwareRetrieval(RetrievalModule):
    default_retriever: RetrievalModule
    retriever_by_layer: dict[str, RetrievalModule]
    layer_selector: LayerSelector
    merger: RetrievalMerger
```

其中：

- `layer_selector` 决定本次 query 访问哪些层
- `retriever_by_layer` 允许层级 override
- `merger` 负责合并多层结果

### 9.2 Layer-aware Memory Evolution

```text
units + placements + decisions
  -> group by target_layer
  -> pick evolution strategy per layer
  -> apply updates
```

一个可能的接口草案：

```python
class LayerAwareEvolution(MemoryEvolutionModule):
    default_evolution: MemoryEvolutionModule
    evolution_by_layer: dict[str, MemoryEvolutionModule]
```

### 9.3 Layer-aware Representation

更稳妥的长期方向是：

```python
class LayerAwareRepresentation(RepresentationModule):
    default_representation: RepresentationModule
    representation_by_layer: dict[str, RepresentationModule]
```

但它依赖一个关键问题先回答：

> 在 representation 发生时，target layer 是否已经确定？

如果还没有确定，那么 representation 更适合先按 unit type dispatch，而不是按 layer dispatch。

因此 representation 的 layer-aware 化要更谨慎。

---

## 10. 当前代码库中的现实落点

结合当前 `memprimitive` 的结构，最现实的落点是：

### 10.1 先不要改动全局 slot 顺序

仍然保持：

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

### 10.2 先让 `retrieval` 和 `memory_evolution` layer-aware

因为：

- retrieval 在 recall 路径上天然按 layer 查询
- evolution 在 ingest 路径上天然按 `placement.target_layer` 分派

### 10.3 `representation` 暂时只预留设计口

如果 `representation` 还发生在 routing / placement 之前，那么：

- 它还不知道最终 target layer
- 因此不适合立刻强行 layer-aware

更合理的选择是：

- 先保持通用 representation
- 等 organization 或 routing 更成熟后，再考虑 layer-aware override

---

## 11. 建议的最小实现路线

### Phase A：先做 Layer-aware Retrieval

目标：

- 支持按 layer 配置不同 retriever
- 支持 merge 多层结果
- 不改动其他 slot

### Phase B：再做 Layer-aware Evolution

目标：

- 按 `placement.target_layer` 选择 evolution 策略
- working/semantic/profile 可以拥有不同 update 机制

### Phase C：最后评估 Layer-aware Representation

前提：

- routing 更稳定
- target layer 语义更清晰
- representation element 体系更成熟

---

## 12. 最终结论

不同 layer 拥有不同 `retrieve`、`evolve`、`representation` 策略，这个想法本身是非常合理的。

但更合适的实现方式不是：

- 每层一整条 pipeline

而是：

- 保持全局 slot 不变
- 在 slot 内部做 layer-aware dispatch

具体建议是：

1. 先做 `retrieval` 的 layer-aware dispatch
2. 再做 `memory_evolution` 的 layer-aware dispatch
3. 最后再考虑 `representation` 的 layer-aware override

一句话总结：

> topology 负责定义 layer 的能力边界，module 负责做策略选择，而 layer-aware dispatch 是两者之间最稳妥的桥梁。
