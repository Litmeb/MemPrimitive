# 模块拆分优先级与演化建议

本文档回答一个更偏实现的问题：

> 在当前 `memprimitive` 代码基础上，哪些模块值得现在立刻细拆，哪些模块应该先保持粗粒度？

核心结论先写在前面：

- 当前阶段应以 **slot 级模块组合** 作为主搜索空间
- 细粒度拆分不是全面展开，而是 **优先拆那些已经具有真实独立性的模块**
- 最值得先拆的是：
  - `write_trigger`
  - `evolution_trigger`
  - `representation`
- 目前不建议优先细拆的是：
  - `retrieval`
  - `organization`
- `unit_formation` 和 `memory_evolution` 可以先保持粗粒度，但在模块内部预留未来细化点

---

## 1. 判断标准：什么叫“现在值得拆”

不是文档里能拆出几个概念，就说明这些概念已经是代码里的独立 primitive。

一个模块或子机制，只有在满足下面 3 个条件时，才值得现在就拆出来：

1. **可独立变化**
   - 替换它时，不需要同时重写多个相邻 slot。
2. **有稳定的数据承载**
   - 它的输入输出能够被当前 `Packet` / `Store` / `ModuleSpec` 明确表达。
3. **组合不是伪自由**
   - 它与其他组件组合后，大多数组合都语义成立，而不是大量依赖特殊约束才能工作。

如果一个细粒度组件一旦替换，就会牵动：

- `Packet` 字段设计
- `MemoryStore` 结构
- 上下游 slot 的假设
- 大量 compatibility constraints

那么它更像是“模块内部配置”，而不是当前阶段应该独立搜索的 primitive。

---

## 2. 现阶段的总原则

建议采用下面的设计路线：

1. **对外保持 slot 级组合**
   - 也就是当前 `MemoryPipeline` 的 8 个固定 slot 仍然是主接口。
2. **对内优先拆出少数高价值子结构**
   - 只在最有把握的模块中引入细粒度组合。
3. **先做半细粒度，而不是一次性完全 DSL 化**
   - 先验证哪些子机制真的能稳定组合，再决定是否升级为更显式的搜索维度。

这意味着：

- 现在的主搜索空间仍然是“每个 slot 选一个 module”
- 但部分 module 内部可以开始采用组合式实现
- 文档中的细粒度设计，暂时作为“模块内部坐标系”，而不是全部变成一等模块

---

## 3. 第一梯队：现在就值得拆

### 3.1 `write_trigger`

这是当前最值得拆的模块。

原因：

- 它天然具有清晰的组合结构：`signals + scorer + gates + policy`
- 它对上下游接口影响最小
- 它的输入输出很稳定：
  - 输入仍然是 `units`、必要时加 `store`
  - 输出仍然是 `Packet.decisions`
- 即使内部细化，`MemoryPipeline` 和其他 slot 的外部接口几乎不用改

这类拆分不是“为了抽象而抽象”，而是真正存在稳定决策骨架。

#### 建议现在拆到的粒度

- `SignalProvider`
  - 负责产出每个 unit 的 signal 值
- `ScoreAggregator`
  - 负责把多个 signal 聚合成 score
- `Gate`
  - 负责硬条件筛选
- `DecisionPolicy`
  - 负责从 score / gate 结果得到最终布尔决策

#### 当前适合先支持的最小子集

- signals:
  - `constant`
  - `novelty_like`
  - `importance_like`
- scorer:
  - `identity`
  - `weighted_sum`
- gates:
  - `always_open`
  - `not_duplicate`
- policy:
  - `always`
  - `threshold`
  - `boolean_gate`

#### 为什么值得先做

因为这一拆分的收益立刻可见：

- 可以更自然地表达 `Always`
- 可以逐步表达 `ImportanceThreshold`
- 可以共享到 `evolution_trigger`
- 可以把“写入决策”从一个黑箱类，变成有 trace、可分析、可 ablate 的组合结构

---

### 3.2 `evolution_trigger`

`evolution_trigger` 应该和 `write_trigger` 一起拆，甚至共用同一 trigger family。

原因：

- 两者本质上都是“是否触发动作”的判断
- 当前代码里已经出现共享迹象
- slot 虽然分开，但机制非常接近

这不是新的抽象发明，而是顺着现有实现自然演化。

#### 建议做法

- 保留 `write_trigger` / `evolution_trigger` 两个独立 slot
- 共享底层 trigger 组件族
- 用 slot-specific adapter 负责把结果分别写入：
  - `Packet.decisions`
  - `Packet.evolution_decisions`

#### 为什么这是高优先级

因为它可以带来两个好处：

1. 避免把“触发机制”在两个 slot 里重复发明一次
2. 让 `Trigger` 在文档中第一次真正成为“被 runtime 承载的组合机制”

---

### 3.3 `representation`

`representation` 值得拆，但建议只做 **半细粒度拆分**。

原因：

- 它确实天然包含多个可区分元素，如 `text`、`normalized_text`、未来的 `embedding`
- 这些元素之间相对容易叠加
- 但它又和 `retrieval`、`organization`、`store indices` 有较强联动

所以不适合一步到位把 `RE-*` 全部升级成完全自由组合。

#### 建议现在拆到的粒度

把 `representation` 理解为：

```text
representation = a set of representation elements
```

但当前只先支持少数真正稳的 element：

- `text`
- `normalized_text`
- `embedding`（未来）
- `entities`（等上游与 store 更稳定时再加）

#### 更合适的实现方式

不要先把每个 element 都做成独立 slot。

更适合的是：

- 一个 `RepresentationModule`
- 内部组合多个 `RepresentationElementBuilder`

例如：

- `RawTextElement`
- `NormalizedTextElement`
- `EmbeddingElement`

#### 为什么它是第二梯队末尾、但仍值得做

因为它是很多下游能力的前置条件：

- retrieval 要不要 embedding
- organization 能不能按 entity / tag 路由
- evolution 能不能做结构化 merge

但目前它还不适合完全放开搜索，否则会立即把系统拖入大量跨模块约束。

---

## 4. 第二梯队：可以预留接口，但先别全面拆

### 4.1 `unit_formation`

`unit_formation` 很重要，但现在不建议优先彻底细拆。

原因：

- 它的细粒度维度很多，但很多并不是真独立
- `Segment / Extract / Abstract / Delegate` 之间语义差异非常大
- 不同 unit 类型会强烈影响后续 `representation`、`memory_evolution`、`retrieval`

也就是说，`unit_formation` 虽然表面上看是一个大搜索空间，但现阶段非常容易出现“能写 DSL、不能稳定组合”的问题。

#### 建议当前做法

- 先保留 slot 级模块，如：
  - `PassThrough`
  - `TurnSplitter`
  - `FactExtractor`
  - `SummaryExtractor`
- 在模块内部再保留少量配置项：
  - segmentation granularity
  - extraction method
  - schema strictness

#### 什么时候再进一步拆

等你已经有至少 2-3 类不同风格的 `unit_formation` baseline 之后，再看哪些内部决策是真正复用的。

---

### 4.2 `memory_evolution`

`memory_evolution` 暂时更适合保持粗粒度。

原因：

- 它是副作用最强的 slot
- `append / merge / upsert / rewrite / summarize+move` 在行为层面差别很大
- 一旦细拆，马上会碰到：
  - selection 如何表达
  - conflict resolution 如何表达
  - store update trace 如何表达
  - 与 organization / trigger / store structure 的联动

#### 建议当前做法

先维持几个清晰的大模块：

- `AppendOnlyEvolution`
- `MergeByKeyEvolution`
- `UpsertByEntityEvolution`
- `RewriteSummaryEvolution`

也就是说，先把 evolution 当作“策略类模块”，而不是先拆成 `selection + action + effect` 的完全组合框架。

#### 为什么不宜太早细拆

因为这类拆分很容易在框架上看起来正交，但实际上大量组合没有明确语义。

---

## 5. 暂缓拆分：现在最容易出现伪自由组合的模块

### 5.1 `retrieval`

这是当前最不建议优先细拆的模块之一。

原因：

- 在现实系统里，retrieval 的 `signals`、`ranker`、`flow` 往往强耦合
- 当前代码中的 `RecencyRetrieval` 已经同时绑定了：
  - query token matching
  - recency fallback
  - ranking logic
- 如果现在硬拆成
  - `signals={...}`
  - `ranker=...`
  - `flow=...`
  很容易形成“语法可组合，语义并不自由”的状态

#### 一个典型问题

如果系统里还没有：

- vector index
- graph links
- entity index
- rerank trace standard

那很多 retrieval 子组合只是理论存在，代码里并没有真实支撑。

#### 当前建议

先保留粗粒度 retrieval 模块，如：

- `RecencyRetrieval`
- `SimilarityRetrieval`
- `HybridRetrieval`

等 store / representation / trace 体系更成熟后，再把 retrieval 内部拆开。

---

### 5.2 `organization`

`organization` 也是一个现在不宜优先细拆的模块。

原因：

- 文档中把它拆成 `routing + links + placement` 很合理
- 但当前 runtime 的真实承载仍然较弱
- 现有 `Placement` 主要承载的是目标 layer，而不是完整的 link/update plan

也就是说，文档中的 `links` 与 `placement` 在概念上已被区分，但在当前代码里还没有被充分实体化。

#### 现在强拆的风险

如果立刻把下面这些都做成独立可搜索组件：

- routing
- temporal/entity/graph links
- placement mode

那么你很快就需要同步重构：

- `Placement` 数据结构
- `MemoryStore` 拓扑与索引
- retrieval 对 links 的消费方式
- evolution 对 placement plan 的解释方式

这类改动跨度太大，不适合作为最早的细拆对象。

#### 当前建议

先保留几个整体式 organization 模块：

- `AppendOrganization`
- `RouteByTypeOrganization`
- `GraphPlacementOrganization`

等 `PlacementPlan` 之类的中间表示稳定后，再做更正式的内部拆分。

---

## 6. 推荐的拆分顺序

建议按下面顺序推进：

### Phase 1：先把 Trigger 家族拆出来

目标：

- 把 `write_trigger` 从单一类变成组合式决策框架
- `evolution_trigger` 复用同一 trigger family

这一步最稳，且收益最大。

---

### Phase 2：把 `representation` 做成 element-set 风格

目标：

- 不改变 slot 边界
- 但允许一个 representation module 内部组合多个 elements

这一步会为后面的 retrieval / organization / evolution 打基础。

---

### Phase 3：等中间表示成熟后，再评估 `organization`

前提是你已经拥有：

- 更明确的 `PlacementPlan`
- 更丰富的 `MemoryStore` capability 表达
- retrieval 对 links / indices 的稳定消费方式

到那时，`organization = routing + links + placement` 才更像真实可组合结构。

---

### Phase 4：最后再考虑 `retrieval` 的深度细拆

前提是你已经具备：

- 多种 representation elements
- 多种 store indices
- 可标准化的 retrieval trace
- 更清晰的合法组合约束

否则 retrieval 很容易变成一堆看似正交、实则互相绑死的配置项。

---

## 7. 一份更具体的执行建议

如果接下来要动代码，我建议按下面的最小方案实施：

### 现在就做

- 将 `write_trigger` 抽象为：
  - signal providers
  - scorer
  - gates
  - policy
- 将 `evolution_trigger` 复用同一 trigger family
- 将 `representation` 改造成“内部可组合 element builders”

### 现在先不做

- 不把 `retrieval` 拆成完整的 `signals + ranker + flow`
- 不把 `organization` 完整拆成 `routing + links + placement`
- 不把 `memory_evolution` 完整拆成 `selection + action + effect`
- 不把 `unit_formation` 完整拆成文档中的全部生成算子

### 代码层面的原则

- 对外接口继续保持 slot 不变
- 对内优先引入小型组合对象
- 优先让 trace 和中间表示先长出来
- 不要让文档里的每个概念都立刻变成 runtime 一级模块

---

## 8. 最终结论

最值得现在直接拆分的模块有三个：

1. `write_trigger`
2. `evolution_trigger`
3. `representation`

其中：

- `write_trigger` 是第一优先级
- `evolution_trigger` 应与其共同演化
- `representation` 适合半细粒度拆分

而下面两个模块，虽然在 `Primitives.md` 中被拆得很漂亮，但目前更适合作为“未来目标形式”，而不是立刻全面实现：

- `retrieval`
- `organization`

一句话总结：

> 先拆真正已经具有独立性的机制，不要先拆那些仍然依赖大规模上下游重构才能成立的概念。

这样做既不会把系统锁死在粗粒度模块里，也能避免过早建立一套“看似自由、实际上并不自由”的框架。
