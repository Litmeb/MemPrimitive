# MemPrimitive

## 项目简介

`MemPrimitive` 是一个面向 agent memory research 的系统化研究框架。这个项目的核心目标，不是再提出一个单独的 memory 方法，而是试图回答一个更基础的问题：

**我们能否用统一、机制级、可组合的方式来描述 agent memory 的设计空间，从而把现有方法放到同一个框架下比较、重组与搜索？**

当前关于 agent memory 的研究已经出现了大量方法，它们在表面形式上差异很大，有的强调 episodic memory，有的强调 semantic memory，有的依赖向量检索，有的强调反思、摘要、图结构、技能库或 agent 主动控制的记忆读写。但如果从底层机制来看，这些方法往往都在重复若干相似的设计动作，例如：

- 把输入切成某种记忆单元
- 决定哪些内容值得写入
- 决定记忆如何组织与更新
- 在需要时从记忆中检索相关内容
- 对记忆做压缩、抽象、维护和遗忘

现有工作的问题在于，这些动作通常被封装在单篇论文提出的具体结构中，导致 memory design space 难以被显式表达。结果是：

- 不同方法之间难以公平比较
- 相似机制往往被不同术语重复命名
- 模块无法方便地拆解、替换和复用
- 新方法的设计更多依赖经验，而非系统搜索
- 很难从多个成功方法中归纳 recurring motifs

`MemPrimitive` 的出发点，就是把 memory system 从“单一方法”重写为“可组合系统”。

---

## 核心主张

本项目提出一个 **compositional memory DSL**，将 agent memory system 分解为一组可组合的 primitive。每个 primitive 对应 memory 流程中的一种基础机制，而不是某篇论文中的整体架构。

这些 primitive 包括但不限于：

- `unit formation`：如何把原始 observation 形成记忆单元
- `representation`：记忆单元如何编码、结构化和索引化
- `write trigger`：系统何时决定写入记忆
- `organization`：ingest-time 如何决定放置位置、关系结构并完成常规写入
- `memory evolution`：默认不启动时，已有记忆如何被额外地重写、压缩、迁移、清理和整理
- `retrieval`：给定 query 如何选取相关记忆
- `readout`：检索结果如何转化为 agent 可使用的上下文

这里最重要的思想是：

**memory system 不是一个整体模块，而是一条由多个 primitive 组成的机制链。**

一旦将系统拆成这些 primitive，就可以把“方法设计”转化为“模块组合问题”，把“新方法发现”转化为“配置搜索问题”，把“经验观察”转化为“motif 归纳问题”。

---

## 这个项目想解决什么问题

### 1. 统一描述问题

本项目首先想解决的是一个表示层问题：如何用统一语言重表达已有 agent memory 方法。

如果没有统一描述框架，那么不同工作之间往往只能停留在自然语言层面的粗略比较，例如“这个方法用了长期记忆，那个方法用了反思”。但这样的比较并不能揭示机制差异。

我们希望将每个 memory system 明确分解为：

- 它如何形成记忆单元
- 它如何表示记忆
- 它何时写入
- 它如何组织并常规写入记忆
- 它如何检索与使用记忆
- 它是否进行额外的压缩、反思或维护记忆

这样，像 MemGPT、Reflexion、A-MEM 等经典系统，就可以被看作同一语言中的不同配置，而不是彼此孤立的方法名。

### 2. 可比较问题

如果不同系统能被重写成同一种结构化表示，就可以做真正的控制变量比较。

例如：

- 固定 `unit formation`，只比较不同 `retrieval`
- 固定 `retrieval`，只比较不同 `write trigger`
- 固定 `representation`，比较不同额外 `memory evolution`

这使得 memory 研究从“方法对方法”的比较，转向“机制对机制”的比较。

### 3. 可搜索问题

本项目的第二个核心问题是：**memory architecture 能否像 program space 一样被系统搜索？**

一旦每个 primitive 都被写成标准模块接口，并且每个 slot 都有明确的候选实现，那么一个 memory system 就可以被看作：

`LayeredStoreTopology × PrimitiveChoices × Hyperparameters × Constraints`

这意味着我们不再只是手工设计一个 memory 方法，而是可以：

- 枚举某一类 memory configuration
- 在约束下自动组合不同模块
- 搜索比现有文献更优或更稳健的配置
- 观察哪些 primitive 组合反复在不同任务上出现

### 4. 机制归纳问题

如果搜索能在大规模配置空间中找到多个高性能 memory systems，那么更高层的问题就变成：

**高性能 agent memory 是否存在 recurring motifs？**

例如，未来可能观察到：

- 很多有效系统都采用“结构化抽取 + selective write + hybrid retrieval”
- 某些任务更偏好“append-only + periodic summarization”
- 某些长期交互场景更需要“entity merge + profile update + extra memory evolution”

这类结论不是从单篇方法中得到的，而是从系统搜索和机制比较中归纳出来的。

---

## 项目的研究视角

`MemPrimitive` 关注的是 **mechanism-level memory design**，而不是具体工程实现。

也就是说，这个项目关心的问题主要是：

- memory system 的构成维度有哪些
- 各维度之间的边界应如何划分
- 哪些模块应该独立出来作为 primitive
- 模块之间有哪些兼容性约束
- 哪些组合构成现有经典工作
- 哪些区域尚未被探索

因此，本项目不是一个单纯的“memory 库”或“agent 框架”，而更接近一个研究基础设施。它的价值在于提供：

- 一个统一的描述语言
- 一个清晰的模块接口层
- 一个可分析、可搜索的设计空间
- 一个帮助归纳 memory 机制规律的研究视角

---

## 整体系统观

本项目将 agent memory system 理解为一个带有副作用的数据流系统。

从外部输入到最终被 agent 使用，memory 的生命周期大致包括以下阶段：

1. 外部 observation 到达系统
2. observation 被切分或抽取为 memory units
3. memory units 被编码为可存储、可索引、可比较的表示
4. 系统决定哪些 unit 值得写入
5. unit 被放置到合适的 store，并与已有记忆建立关系
6. store 被更新，可能发生追加、替换、合并或重写
7. 在后台或额外触发路径中执行压缩、抽象、整理、遗忘或迁移
8. 当 agent 有 query 或当前任务需要时，系统检索相关记忆
9. 检索结果被组织成 agent 可消费的 readout
10. agent 将 readout 纳入自己的推理、规划和响应过程

这个视角强调两点：

- memory 不是静态数据库，而是动态演化系统
- memory 研究不应只看 retrieval，而应覆盖写入、组织、抽象和维护的完整闭环

---

## 为什么需要 DSL

本项目使用 DSL，不是为了追求语法形式本身，而是因为自然语言描述不足以支撑系统研究。

一个合格的 memory DSL 至少应该支持四件事：

### 1. 重表达已有方法

DSL 应该能把已有 memory 工作重新写成统一配置。对当前阶段而言，重点不再只是挑选少数经典工作做展示，而是尽可能把调研范围从经典代表方法扩展到更广泛、更多样的 agent memory 文献，并用统一语言持续吸纳这些方法。

当前更具体的目标是：

- 将调研覆盖面扩展到约 40 篇不同风格的 memory 工作，而不只停留在少数经典案例
- 尽量让其中约 1/4 的方法可以被 **完全重表达** 为统一的 primitive 组合与配置
- 对暂时还无法完全重表达的方法，也至少做到结构化拆解、定位缺失机制，并反过来推动 module 边界继续扩展

只有这样，这套语言才不是空洞 taxonomy，而是真正具有解释力、扩展性和文献承载能力的研究工具。

### 2. 表达模块组合

很多 memory 方法不是单模块决策，而是多机制组合，例如：

- 检索中同时使用 similarity、recency、importance
- 不同 observation type 使用不同 unit formation
- recall 和 rerank 分两阶段执行

因此 DSL 不能只支持“单字段配置”，还应支持组合、级联和条件分发。

### 3. 支持约束

memory primitive 之间并非完全自由组合。

例如：

- graph retrieval 需要图结构或图链接
- entity-based memory evolution 需要 entity-aware units
- similarity retrieval 需要 embedding
- hierarchical retrieval 往往需要层级压缩或层级组织

所以 DSL 必须不仅表达“选什么”，还要表达“哪些组合是合法的”。

### 4. 面向搜索

如果未来要做自动搜索，那么 DSL 里的每一个选择都应该是可枚举、可验证、可采样的。也就是说，DSL 应该天然对应一个可操作的 configuration space，而不仅仅是人类可读的说明文档。

---

## 本项目中的 design space

在 `MemPrimitive` 中，memory design space 由三层共同构成。

### 1. 结构层

结构层描述 memory store 的骨架，包括：

- 有几层 memory layer
- 每层的主题是什么
- 每层是 `Flat` 还是 `Graph`
- 各层有哪些 index

这是整个系统的 structural prior。这里的 hierarchical 不再被视为一种单独的 store type，而是“存在多层 layer”这一结构事实本身。

### 2. 模块层

模块层描述每个 primitive slot 选择哪种实现。

例如：

- `unit formation` 选择 turn-level、fact extraction 还是 multi-granularity
- `write trigger` 选择 always write、importance gate、LLM judge 还是 agent-controlled write
- `retrieval` 选择 similarity、graph hop、LLM-controlled retrieval 还是 hybrid retrieval

模块层是 design space 的主体。

### 3. 参数层

参数层描述每个 primitive 内部的超参数、阈值和策略选项。

例如：

- top-k
- similarity threshold
- decay rate
- summary batch size
- merge policy

参数层决定的是同一机制家族内部的行为差异。

---

## 搜索不是穷举，而是受约束的组合探索

本项目所设想的搜索，不是对所有模块进行无差别穷举，而是 **constraint-aware search**。

原因很简单：primitive 虽然被接口化了，但它们之间仍然存在语义耦合。

例如：

- `representation` 和 `retrieval` 往往强耦合
- `unit formation` 与 `organization` / `memory evolution` 常常强耦合
- `organization` 决定了后续能否进行 graph 或 hierarchical retrieval，也承担常规写入

因此，一个有效的搜索空间必须同时包含：

- 可组合性
- 类型一致性
- 能力约束
- 模块间兼容性

换言之，`MemPrimitive` 追求的不是“完全自由组合”，而是“在形式化约束下的可控组合”。

当前实现处于一次约束机制重构过渡期：

- `MemoryPipeline` 仍然会在构造期检查 slot 抽象类型与 `ModuleSpec.slot` 是否对齐
- 旧的 baseline 侧 store/topology eager compatibility check 已移除
- baseline/runtime 侧现已改为以 `MemoryStore.check()` 为中心的组合合法性检验：`MemoryPipeline` 负责把模块声明的 `requires_contracts` / `produces_contracts` 注册到共享 store，随后由调用方在需要时显式执行 `store.check()`

---

## 这个项目希望覆盖哪些已有 memory 研究

这个项目的目标之一，是让不同风格的 agent memory 方法都能被放到一个统一框架中理解。当前阶段会主动把文献池从经典工作扩展到尽可能多的代表性与异质性工作，目标规模约为 40 篇；其中会重点挑选约 1/4 作为完全重表达对象，其余工作也会纳入统一拆解与对照分析。覆盖范围包括但不限于：

- 以 observation stream 为核心的 episodic memory
- 以事实抽取、属性更新为核心的 semantic/profile memory
- 以反思、总结和抽象为核心的 reflective memory
- 以技能、代码片段为核心的 skill memory
- 以图节点与关系链接为核心的 graph memory
- 以 agent 主动工具调用为核心的 self-managed memory
- 以工作记忆、长期记忆、归档记忆分层为核心的 multi-store memory

这些系统虽然表面形式不同，但理想状态下都应被还原为一组 primitive 的不同取值和不同组合方式。若现有模块还不足以承载某一类方法，那么扩展 module 本身就是当前阶段的重要研究工作，而不是例外情况。

---

## 这个项目最终想产出什么

从研究目标上看，`MemPrimitive` 希望最终支持以下几类产出。就当前阶段而言，最优先的中期目标是：把文献覆盖面扩展到约 40 篇，并尽量使其中约 1/4 能够被比较有说服力地声称为“完全重表达”；为此，项目会继续扩展 module families 与组合边界，让更多过去方法可以落到统一框架中。

### 1. 一套统一的 memory 描述语言

用于表达、比较和重构已有 agent memory 系统。

### 2. 一套机制级 primitive ontology

明确 memory system 由哪些基础机制组成，每种机制有哪些实现变体，它们之间如何兼容或冲突。

### 3. 一个可搜索的 memory configuration space

使 memory design 从手工启发式设计，转向可约束的系统探索。

### 4. 一组 recurring motifs

通过对经典方法和搜索结果的分析，归纳出高性能 memory systems 中反复出现的设计模式。

### 5. 一个新的研究问题框架

即把 agent memory 研究从“提出一个新架构”，推进到“研究 memory design space 的组织规律、结构偏置与搜索方法”。

---

## 本仓库当前文档的分工

- `memprimitive/baselines/README.md`
  说明阶段一 baseline 代码如何按 primitive slot 拆分到多个 `.py` 文件、`__init__.py` 的聚合导出，以及扩展新实现时的约定。
  当前 `RecencyRetrieval` 语义已固定为“直接返回最新的 top-k 记录”，不再根据 query token 先做文本过滤。
  当前 trigger-family baseline 已覆盖 metadata-gated write、key-ready write、outcome-conditioned evolution trigger 与 new-write local-maintenance trigger 这类非 graph motif，可直接用稳定类名和 builder 表达 TiM、Reflexion、MemGPT 风格触发语义。
  同时，Reflexion-like back-half motif 现在也已有通用 baseline：`PlacementWithoutAppendOrganization`、`ReflectionGenerationEvolution`、`BufferRetrieval`、`PromptContextReadout`，classic Reflexion wrapper 直接复用这些 slot-level 实现。

- `memprimitive/example/demonstration/README.md`
  汇总可运行 demonstration，包括失败 / 成功 trial 的 Reflexion 风格触发、完整的 failed-trial -> reflection -> next-recall context Reflexion-like 闭环，以及 partition-ready local maintenance 的 TiM 风格触发演示。
  这些 demonstration 默认应以最简洁的 `MemoryPipeline + module composition` 形式展示 DSL 用法，而不是依赖更高层的 workflow 封装。

- `DSLIO.md`
  讨论 memory system 各模块的标准输入输出接口，明确系统中的共享对象、模块签名、副作用与能力约束。

- `DSLgrammar.md`
  将接口层抽象成声明式语言，定义 memory system 的形式语法、组合算子、约束表达方式，以及如何用该语言重表达经典方法。

- `Primitives.md`
  枚举每个 primitive slot 的可能实现，刻画搜索空间、兼容性约束、已有工作分解和潜在未探索组合。

这三个文档共同构成了项目的研究基础：

- `DSLIO.md` 回答“模块接口是什么”
- `DSLgrammar.md` 回答“系统如何被语言化描述”
- `Primitives.md` 回答“搜索空间里到底有什么”

---

## 适用场景

`MemPrimitive` 适合以下几类研究工作：

- 想系统整理 agent memory 文献的人
- 想比较不同 memory 机制而不是仅比较完整方法的人
- 想把已有 memory 架构重写成统一形式的人
- 想开展 memory architecture search 的人
- 想从搜索结果中归纳 recurring motifs 的人
- 想研究 long-term memory、semantic memory、reflective memory、skill memory 之间关系的人

---

## 不在当前范围内的内容

本项目当前关注的是系统设计与研究抽象，因此暂不把重点放在以下方面：

- 具体工程实现细节
- 具体模型调用与代码组织
- 具体 benchmark 的运行脚本
- 特定框架下的部署方式
- 面向生产环境的优化细节

这些内容未来都可以建立在当前研究抽象之上，但不是当前阶段的核心目标。

---

## 一句话概括

`MemPrimitive` 想做的事是：

**把 agent memory 从“方法集合”重构为“可组合、可搜索、可归纳的机制空间”；而当前阶段最核心的目标，是把尽可能多的既有方法纳入统一重表达框架中，扩展到约 40 篇文献，并争取其中约 1/4 达到完全重表达。**
