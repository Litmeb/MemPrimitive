# Ingest Write and Memory Evolution Semantics

这份文档用于给后续 coding agent 一个明确、统一、可执行的语义基线。

目标不是追求传统软件分层下的“最纯职责拆分”，而是维持 `MemPrimitive` 作为 **可搜索 memory mechanism space** 的切分方式。

## 1. 这次语义重定义解决什么问题

旧表述里，`organization` 更像“写前规划”，`memory_evolution` 同时承担了：

- 常规写入的真正落库
- 追加/合并/重写等 ingest-time update
- 周期性 forget / summarize / abstract / consolidate 等额外演化

这会带来一个问题：

- `write_trigger`
- `organization`
- `evolution_trigger`
- `memory_evolution`

之间的边界开始重叠，尤其是 `evolution` 这个词同时覆盖了“每次写入都会发生的默认更新”和“默认不启动的额外维护/抽象流程”。

新的语义重定义是：

- `write_trigger` 决定是否进行常规写入
- `organization` 负责 ingest-time 的组织与常规写入提交
- `evolution_trigger` 决定是否启动额外的 memory evolution
- `memory_evolution` 只负责默认关闭、额外触发的演化流程

## 2. 为什么不新增 `write_commit`

本项目的 primitive 不是按工程实现细节拆，而是按 **是否构成真实可搜索维度** 来拆。

如果所谓“写入提交”并没有独立于 `representation` / `organization` 的可搜索自由度，而是天然被它们共同决定，那么把它单独拆成 `write_commit` slot 只会制造一个空洞维度。

因此：

- 不新增 `write_commit` slot
- 而是把“常规写入如何提交”吸收到 `organization` 的语义里

这仍然保留了搜索空间，因为不同 `organization` 实现依然可以代表不同的 ingest-time organization/update strategy。

## 3. 新的 slot 语义

### 3.1 `write_trigger`

定义：

- 判断当前 observation 形成的 unit 是否值得进入常规写入路径

它回答的问题是：

- “这次要不要记？”

它不负责：

- 决定写到哪
- 决定如何组织关系
- 决定额外演化是否发生

### 3.2 `organization`

新定义：

- `organization` = ingest-time organization and normal write

它回答的问题是：

- “这些 unit 应该被放到哪里？”
- “和已有 memory 建立什么关系？”
- “在常规写入路径上，应当如何把这些 unit 正式落入 store？”

这里的“常规写入”包括：

- append 到目标 layer
- 在图层创建 node / edge
- 在分区层写入对应 bucket
- 与当前 organization strategy 紧耦合的 ingest-time merge / upsert / replace

关键点：

- 这些行为仍被视为 `organization` 的一部分
- 因为它们并不是独立搜索轴，而是与 routing / links / placement 强耦合的同一机制族

### 3.3 `evolution_trigger`

新定义：

- 控制是否启动额外的 `memory_evolution`

默认语义：

- 默认关闭
- 不应被理解为“每次常规写入都必须走的第二层写入 trigger”

它回答的问题是：

- “是否还要在常规写入之外，再做一轮额外演化？”

典型触发方式：

- periodic
- budget exceeded
- session end
- task failure
- explicit reflection / maintenance signal

### 3.4 `memory_evolution`

新定义：

- `memory_evolution` 只表示默认不启动、额外触发的 memory evolution

它回答的问题是：

- “在已有 memory 上，是否需要做额外的整理、抽象、维护、遗忘或高层重组？”

典型行为：

- forget / prune
- summarize
- reflect
- consolidate
- dedup
- move / archive
- rewrite existing memories
- profile / concept abstraction

非目标：

- 不再把普通 append 视为 `memory_evolution` 的默认职责
- 不再把“每次 observation 到来后都发生的常规落库”视为 `memory_evolution`

## 4. 新的 ingest 语义

新的 stage-1 ingest 语义是：

```text
unit_formation
-> representation
-> write_trigger
-> organization
-> evolution_trigger
-> memory_evolution
```

但它的含义已经变成：

```text
unit_formation
-> representation
-> decide whether to perform normal write
-> organize and commit normal write
-> decide whether extra evolution should run
-> run optional extra evolution over existing memory
```

因此：

- `organization` 成为默认常规写入的完成点
- `memory_evolution` 成为 optional extra pass

## 5. 对 `Packet` / runtime 的约束建议

为保持向后兼容，建议分两层处理：

### 5.1 语义层

- `Packet.decisions` 仍然表示 write-side decisions
- `Packet.evolution_decisions` 仍然表示 extra-evolution decisions

### 5.2 执行层

- `organization` 应允许修改 `store`
- `organization` 的 trace 应记录常规写入实际造成的影响
- `memory_evolution` 不应再依赖 `Packet.decisions` 作为默认回退来源
- 长期目标是让 `memory_evolution` 只看 `evolution_decisions`

如果为了平滑迁移暂时保留 fallback，也应在文档中明确标记为：

- backward-compatibility only
- not the target semantics

## 6. 搜索空间如何重划

这次调整的核心不是减少搜索空间，而是把搜索空间重新划到更符合研究意图的位置。

### 6.1 `organization` 变大

它现在覆盖：

- routing
- links
- placement
- normal ingest-time update behavior

### 6.2 `memory_evolution` 变窄

它现在只覆盖：

- extra evolution over existing memory
- maintenance / abstraction / consolidation
- non-default reflective or periodic updates

### 6.3 `evolution_trigger` 更有意义

以前它容易显得像和 `write_trigger` 冗余的第二层开关。

现在它的语义清晰了：

- `write_trigger` 管常规写入
- `evolution_trigger` 管额外演化

## 7. coding guidance

后续实现和文档修改都应遵守以下规则：

1. 不要再把 `organization` 描述成“只做 placement plan、不落库”。
2. 不要再把 `memory_evolution` 描述成“默认承接普通 append”。
3. 不要再把 `evolution_trigger` 描述成“每次写入都应参与的第二层通用写触发器”。
4. 如果某种 ingest-time update 与 routing / links / placement 强耦合，应优先作为 `organization` 变体。
5. 只有当某类变化是默认关闭、额外触发、面向已有 memory 的整理/抽象/维护时，才归入 `memory_evolution`。

## 8. 一句话版本

一句话总结新的语义：

```text
organization handles normal ingest-time organization-and-write;
memory_evolution handles optional extra evolution over already-organized memory.
```
