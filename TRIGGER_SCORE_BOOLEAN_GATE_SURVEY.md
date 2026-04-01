# Write / Evolution Trigger Implementation Survey

更新时间：2026-04-01

## 这次重写和旧版有什么不同

这份文档不再按 `threshold(score)`、`boolean gate` 这种概念标签做主分类，而是按**触发的实现机制**重写。

同时我明确收紧了 trigger 的范围：

- 只讨论 `write trigger`
- 只讨论 `evolution trigger`
- 不讨论 retrieval trigger
- 不讨论“选什么内容去写 / 去演化”的内部 scoring、ranking、retrieval 细节，除非它本身就是“是否触发动作”的实现

换句话说，这里关心的是：

- memory action 是被谁触发的
- 触发是代码里的哪种实现面
- 不同论文虽然语义上判断的东西不同，但如果实现方式相同，就应该归为同一类

## 方法

1. 以 [TRIGGERS.md](./TRIGGERS.md) 的 40 篇候选文献清单为母表
2. 回到原论文页面、摘要或可公开正文，优先核对“什么时候发生 write / evolution”
3. 优先抽取**实现机制**，不优先抽象成 trigger 语义名词
4. 对证据不足的条目标成“弱证据 / 不宜写死”

## 一句话结论

如果按实现机制而不是按语义概念来分，40 篇工作里的 write/evolution trigger 基本可以压到下面 7 类：

1. `PassThroughHook`
   - 每个 turn / step / observation / document 到来时默认执行
2. `StructuralBoundaryHook`
   - 在 turn / session / subgoal / episode 边界执行
3. `RuntimeCallback`
   - 由失败、成功、memory pressure、系统 warning、环境反馈等运行时事件回调触发
4. `ExplicitScalarRule`
   - 由显式数值、公式或阈值规则触发
5. `LLMJudge`
   - 单个 LLM / VLM / LM-based detector 直接判断“现在要不要触发”
6. `BackgroundScheduler`
   - 周期、空闲窗口、sleep-time 等后台调度触发
7. `ControllerOrchestrator`
   - 专门的 controller / scheduler / 多 agent 管理器决定什么时候做 write/evolution

旧版里看起来很异质的 `threshold(score)`、`boolean signal gate`，在实现层其实经常会塌缩到同一类：

- 如果都是“prompt 一个 LLM 让它打分或判定”，那它们都应归到 `LLMJudge`
- 如果都是“代码里维护一个分数并在阈值上比较”，那它们都应归到 `ExplicitScalarRule`
- 如果都是“session 结束后固定做压缩”，那它们都应归到 `StructuralBoundaryHook`

## 实现机制词表

### 1. `PassThroughHook`

定义：
- 新输入单元一到，就默认进入 write/evolution 管线
- 没有单独的 trigger 决策器
- 常见载体：`on_new_turn`、`on_new_observation`、`on_new_document`

识别标准：
- 论文没有强调“是否触发”，而是默认每次都写
- 差异主要在 representation / graph update / store format，而不是 trigger

### 2. `StructuralBoundaryHook`

定义：
- 由会话边界、turn 边界、chunk 边界、subgoal 结束、episode 边界这类结构化事件触发

识别标准：
- 触发时机由任务/对话结构定义
- 不依赖环境失败或后台 scheduler

### 3. `RuntimeCallback`

定义：
- 由运行时事件直接触发
- 典型事件：任务失败、测试失败、成功验收、context overrun、memory warning、显式 interrupt

识别标准：
- 触发来自环境/系统/执行结果
- 像 event handler，而不是定时器或独立评分器

### 4. `ExplicitScalarRule`

定义：
- 代码维护一个显式数值或公式
- 当其超过阈值、下降到阈值以下，或按解析式更新时触发动作

识别标准：
- 有明确 score / utility / retention / surprise / importance
- 不依赖 LLM 充当判断器

### 5. `LLMJudge`

定义：
- 由一个 LLM / LM-based module 直接判断是否触发
- 它可以输出 yes/no、score、label、boundary decision、route decision

识别标准：
- “prompt the model to decide/rate/classify whether...”
- 即使输出的是 score，只要 score 本身由 LLM 生成，主实现仍归这里

### 6. `BackgroundScheduler`

定义：
- 由固定周期、空闲窗口、sleep-time、后台低负载时间驱动

识别标准：
- 论文明确说有 periodic maintenance、idle consolidation、sleep-time update

### 7. `ControllerOrchestrator`

定义：
- 专门的 scheduler、controller、多 agent 管理器、OS-style memory manager 负责决策时机

识别标准：
- trigger 不只是单个 if/threshold，而是由控制模块统一编排

## 40 篇工作：按实现重映射

说明：

- `写入实现类` / `演化实现类` 是这次调研的主输出
- `具体实现` 只描述 trigger 是怎么落地的
- `范围判断` 用于说明是否真的属于 write/evolution trigger，而不是读侧控制
- `证据强度`：
  - `高`：原文明确
  - `中`：原文足够支持，但实现细节未完全展开
  - `低`：只能保守归纳

| # | 论文 | 写入实现类 | 演化实现类 | 具体实现 | 范围判断 | 证据强度 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Generative Agents | `LLMJudge` | `LLMJudge` + `ExplicitScalarRule` | 记忆创建时用 LLM 给 memory 打 `importance/poignancy` 分；reflection 在“近期 importance 累积和超过阈值”时触发。这里虽然有 threshold，但 score 是 LLM 产的，所以主类应算 `LLMJudge`。 | 属于 write/evolution | 高 |
| 2 | Reflexion | `RuntimeCallback` | `RuntimeCallback` | 写入 reflective memory 和后续 verbal reinforcement 都由任务失败、测试失败、环境负反馈等执行结果回调触发。 | 属于 write/evolution | 高 |
| 3 | Voyager | `RuntimeCallback` | `RuntimeCallback` + `PassThroughHook` | skill library 的新增通常发生在成功技能/程序被验证后；后续 skill 改写由执行失败或任务反馈驱动。对已有写入的修订可看成 `after_write` 式自然延续。 | 属于 write/evolution | 中 |
| 4 | MemGPT | `LLMJudge` + `RuntimeCallback` | `RuntimeCallback` + `ControllerOrchestrator` | “写到哪层 memory”由 LLM 在函数调用框架内决定；当系统发出 context pressure / warning / interrupt 时触发迁移、分页、写出。其 evolution 更像 OS-style 管理器 + 运行时告警。 | 属于 write/evolution | 高 |
| 5 | MemoryBank | `PassThroughHook` | `ExplicitScalarRule` | 对话记录本身持续进入 memory；演化/遗忘由 Ebbinghaus-style retention function 驱动，随时间衰减、回忆次数增强。 | 属于 write/evolution | 高 |
| 6 | RET-LLM | `PassThroughHook` | `PassThroughHook` | 新文本进入系统就抽取 triplet 并写入；后续结构维护基本是写后自然发生，没有独立 trigger 设计。 | 属于 write/evolution | 中 |
| 7 | My Agent Understands Me Better | `RuntimeCallback` | `ExplicitScalarRule` | write 更接近 cue hit 时的 recall/write-back；evolution 则由 consolidation model 驱动，量化考虑 relevance、elapsed time、recall frequency。这里演化主类是显式模型而不是 LLM 判定。 | write/evolution 都有，但 write 侧证据较弱 | 中 |
| 8 | Compress to Impress | `StructuralBoundaryHook` | `StructuralBoundaryHook` | 压缩与总结在 session / conversation segment 收束时启动。 | 属于 write/evolution | 高 |
| 9 | Toward Conversational Agents with Context and Time Sensitive Long-term Memory | `PassThroughHook` | `PassThroughHook` | 以对话 turn 持续累积记忆；论文重点不在 trigger，而在时间敏感 retrieval。 | write/evolution 触发很弱 | 中 |
| 10 | Agent Workflow Memory | `RuntimeCallback` + `LLMJudge` | `PassThroughHook` + `LLMJudge` | 输入是 completed task / past experiences；workflow induction 本身是 LM-based。也就是说，触发动作的“入口事件”是任务完成，而“是否抽成 workflow”主要由 LM induction 实现。 | 属于 write/evolution | 高 |
| 11 | HiAgent | `PassThroughHook` | `StructuralBoundaryHook` + `LLMJudge` | step 级 working memory 默认更新；当 subgoal / task phase 闭合时，再用 LLM 做 chunk-level summarization / abstraction。 | 属于 write/evolution | 中 |
| 12 | Memory OS of AI Agent | `LLMJudge` / `ControllerOrchestrator` | `RuntimeCallback` + `ControllerOrchestrator` | 核心是 memory manager 判定信息进入短/中/长期哪层；存满/超限时触发迁移或淘汰。 | 属于 write/evolution | 中 |
| 13 | Mem0 | `LLMJudge` | `PassThroughHook` + `LLMJudge` | selective memory formation 的入口是 ongoing conversation；“值不值得记 / 是否 merge-update”主要由 LLM 抽取与更新判断完成，因此实现上是 `LLMJudge`，不是手写阈值规则。 | 属于 write/evolution | 高 |
| 14 | A-MEM | `LLMJudge` | `PassThroughHook` + `LLMJudge` | agent 判断新信息是否值得成为 note、应该链接到哪里；图维护通常在新 note 写入后自然发生。 | 属于 write/evolution | 中 |
| 15 | HippoRAG | `PassThroughHook` | `PassThroughHook` | 文本来了就抽 triple / build graph；演化主要是写后图更新，没有独立 trigger 层。 | 属于 write/evolution，但较弱 | 中 |
| 16 | From RAG to Memory / HippoRAG 2 | `PassThroughHook` | `PassThroughHook` / `BackgroundPipeline` | 段落进入时直接参与建图；离线 integration 更像管线阶段，不太像在线 trigger。 | evolution 侧更像离线管线 | 低 |
| 17 | AriGraph | `PassThroughHook` | `PassThroughHook` | 新 observation 持续写入 episodic/semantic graph；每步后做 graph/world-model update。 | 属于 write/evolution | 高 |
| 18 | Zep / Graphiti | `PassThroughHook` | `PassThroughHook` | 新事件/新对话到来时更新时间知识图；后台维护不是论文强调的显式 trigger。 | 属于 write/evolution | 中 |
| 19 | Bridging Intuitive Associations and Deliberate Recall | `PassThroughHook` | `PassThroughHook` | event-centric graph 随新对话持续形成；主要创新在 retrieval，不在 write/evolution trigger。 | 触发很弱 | 中 |
| 20 | From Experience to Strategy | `PassThroughHook` | `ExplicitScalarRule` / `TrainingLoop` | experience 写入更像默认收集；strategy 固化受 reward / empirical utility 权重更新驱动，更像训练式 utility rule。 | evolution 属于弱 trigger | 中 |
| 21 | Hierarchical Memory Organization for Wikipedia Generation | `PassThroughHook` | `BackgroundPipeline` | 文档进入后进入层级组织管线；更像离线组织流程，不像在线 trigger。 | 仅弱相关 | 低 |
| 22 | Optimizing the Interface Between KGs and LLMs | `PassThroughHook` | `PassThroughHook` | 图构建与接口优化默认进行，没有值得抽出来的独立 write/evolution trigger 实现。 | 基本不以 trigger 为重点 | 低 |
| 23 | AI-native Memory 2.0: Second Me | `LLMJudge` + `ControllerOrchestrator` | `PassThroughHook` + `BackgroundPipeline` | 写入带有 memory type / layer routing；后台参数化和迁移存在，但公开语义更像系统后处理阶段。 | write 明显，evolution 较弱 | 中 |
| 24 | O-Mem | `LLMJudge` | `PassThroughHook` + `BackgroundPipeline` | persona、event、topic/context 层级的落库需要模型抽取/分类；后续层级整理更像后台组织阶段。 | write 明显，evolution 较弱 | 中 |
| 25 | In Prospect and Retrospect | `StructuralBoundaryHook` | `StructuralBoundaryHook` + `RuntimeCallback` | prospective reflection 在 utterance / turn / session 边界生成多粒度 memory；retrospective refinement 则在 retrieval/error signal 暴露后更新。由于你要求排除 retrieval，本轮只保留“边界生成 memory”与“误差反馈后更新”这两层。 | 属于 write/evolution，但部分机制贴近 retrieval | 高 |
| 26 | Nemori | `LLMJudge` | `LLMJudge` + `RuntimeCallback` | LLM-based boundary detector 为每条新消息输出 boundary decision 与 confidence；semantic memory generation 又由 predict-calibrate / prediction-gap 驱动。前者是 `LLMJudge`，后者带明显运行误差回调色彩。 | 属于 write/evolution | 高 |
| 27 | MemOS | `ControllerOrchestrator` | `BackgroundScheduler` + `ControllerOrchestrator` | MemOS 的写入不只是单个 if，而是 memory OS/scheduler 统一管理多种 memory form；maintenance 由持续调度和资源编排驱动。 | 属于 write/evolution | 中 |
| 28 | MIRIX | `LLMJudge` + `ControllerOrchestrator` | `ControllerOrchestrator` + `BackgroundScheduler` | 六类 memory 的更新由 multi-agent framework 动态协调；写入时也存在 type classifier / controller 决策。 | 属于 write/evolution | 高 |
| 29 | SEDM | `RuntimeCallback` + `ExplicitScalarRule` | `BackgroundScheduler` + `ExplicitScalarRule` + `ControllerOrchestrator` | self-scheduling controller 根据 verifiable replay、utility ranking 等决定何时吸收/扩散经验；其 consolidation 带明显调度器色彩。 | 属于 write/evolution | 中 |
| 30 | LightMem | `RuntimeCallback` | `BackgroundScheduler` + `PassThroughHook` | 写入受 buffer/capacity 条件触发；离线巩固明确发生在 sleep-time / idle-like window，且对已有写入做后续整理。 | 属于 write/evolution | 高 |
| 31 | Towards LifeSpan Cognitive Systems | `PassThroughHook` | `BackgroundPipeline` | 经验吸收是总框架概念，但没有可确认的具体在线 trigger 机制。 | 弱相关 | 低 |
| 32 | Human-inspired Episodic Memory for Infinite Context LLMs / EM-LLM | `ExplicitScalarRule` | `ExplicitScalarRule` + `PassThroughHook` | event segmentation 由 token surprise 超阈值触发；形成 episodic units 后，后续 memory structure 基本按写后流程更新。这里即便 `surprise` 来自 LM logits，也不是 prompt judge，而是显式数值规则。 | 属于 write/evolution | 高 |
| 33 | MemVerse | `PassThroughHook` | `BackgroundScheduler` + `ExplicitScalarRule` | 输入侧默认进入 multimodal memory；distillation/consolidation 在周期性后台阶段进行，并伴随重要样本选择。 | 属于 write/evolution | 中 |
| 34 | MGA | `PassThroughHook` | `PassThroughHook` | GUI observation / trajectory step 到来即形成外部 memory；更新只是随轨迹持续追加。 | 属于 write/evolution | 中 |
| 35 | KARMA | `LLMJudge` / `ControllerOrchestrator` | `RuntimeCallback` + `ExplicitScalarRule` | 写入前要先决定是进 short-term 还是 long-term store；短期 memory replacement 受容量与对象状态变化触发。 | 属于 write/evolution | 中 |
| 36 | VideoAgent | `OutOfScope` | `OutOfScope` | 主要 trigger 是 query-driven evidence gathering 与 stop-search，不是 write/evolution trigger。旧 survey 可保留它作为“边界案例”，但如果严格按你这次的范围，应排除出主统计。 | 不属于本轮主范围 | 高 |
| 37 | WorldMM | `OutOfScope` | `OutOfScope` / `ControllerOrchestrator` | 非平凡控制主要在 retrieval routing 与 temporal granularity 选择；不是 write/evolution trigger 的代表论文。 | 不属于本轮主范围 | 高 |
| 38 | Seeing, Listening, Remembering, and Reasoning / M3-Agent | `LLMJudge` | `PassThroughHook` | 多模态输入落到 entity-centric memory 前，需要实体/模态层面的分类与组织；之后持续更新 episodic/semantic memory。 | 属于 write/evolution | 中 |
| 39 | HippoMM | `LLMJudge` / `StructuralBoundaryHook` | `BackgroundScheduler` + `PassThroughHook` | adaptive temporal segmentation 负责形成 episodic unit；后续短长时巩固更像后台 consolidation。 | 属于 write/evolution | 中 |
| 40 | Episodic Memory Representation for Long-form Video Understanding | `StructuralBoundaryHook` | `StructuralBoundaryHook` | 关键是按 episodic event boundary 切视频片段；但其主问题更偏视频理解。若只看 memory action，trigger 是边界事件而非评分器。 | 边界案例，弱相关 | 中 |

## 重新归纳：真正值得实现成哪几类

如果只按**实现**来设计 `MemPrimitive` 的 write/evolution trigger，我建议不要先实现一堆语义类，而是先实现下面这些运行机制：

### A. `OnWrite` / `OnInput` 直通钩子

覆盖：
- `PassThroughHook`

代表：
- RET-LLM
- HippoRAG
- AriGraph
- Graphiti
- MGA

工程含义：
- 这是默认值
- 很多论文根本没有“要不要触发”这层判断

### B. `BoundaryEventTrigger`

覆盖：
- `StructuralBoundaryHook`

代表：
- Compress to Impress
- HiAgent
- In Prospect and Retrospect
- EM-LLM
- Video episodic memory 类工作

工程含义：
- 这类触发不需要复杂 scorer
- 更像“有一个上游边界检测器，然后在边界到达时执行”

### C. `RuntimeEventTrigger`

覆盖：
- `RuntimeCallback`

代表：
- Reflexion
- Voyager
- MemGPT
- LightMem

工程含义：
- 最适合统一成 event bus / callback
- 事件名可包括：
  - `task_failed`
  - `task_succeeded`
  - `memory_pressure`
  - `interrupt_received`
  - `buffer_full`

### D. `ScalarRuleTrigger`

覆盖：
- `ExplicitScalarRule`

代表：
- MemoryBank
- EM-LLM
- Generative Agents 的 importance-sum 部分
- My Agent Understands Me Better 的 consolidation model

工程含义：
- 它不关心 score 的语义名字
- 只关心：
  - 数值来自哪里
  - 比较器是什么
  - crossing 之后触发什么动作

### E. `ModelJudgeTrigger`

覆盖：
- `LLMJudge`

代表：
- Generative Agents
- Mem0
- A-MEM
- Nemori
- O-Mem
- M3-Agent

工程含义：
- 这是旧版 survey 里最容易被低估的一类
- 很多表面上不同的 trigger，其实都是：
  - prompt 一个模型
  - 输出 yes/no 或 score/label
  - 根据输出执行 write/evolution

### F. `ScheduledMaintenanceTrigger`

覆盖：
- `BackgroundScheduler`

代表：
- LightMem
- MemOS
- MIRIX
- SEDM
- MemVerse

工程含义：
- 这是 evolution trigger 的关键实现类
- 比 `PeriodicTrigger` / `IdleTrigger` 更贴近实现
- 具体 schedule 再作为参数

### G. `ControllerTrigger`

覆盖：
- `ControllerOrchestrator`

代表：
- MemGPT
- MemOS
- MIRIX
- SEDM

工程含义：
- 这类不是简单 trigger object
- 更像“统一的 memory control plane”

## 哪些语义类应该降级

这次按实现重看之后，下面这些旧语义类不适合做成一等实现类：

### 1. `threshold(score)` 不应直接是顶层实现类

原因：
- `importance threshold` 可能是 `LLMJudge`
- `surprise threshold` 可能是 `ExplicitScalarRule`
- `utility threshold` 可能是 `TrainingLoop` 或 controller 决策

更合理的做法：
- 先问“score 是谁算的”
- 再决定落到 `ModelJudgeTrigger` 还是 `ScalarRuleTrigger`

### 2. `boolean signal gate` 不应直接是顶层实现类

原因：
- `failure` 来自 runtime callback
- `session end` 来自结构边界
- `event boundary` 可能来自 LLM detector
- `memory warning` 来自系统 runtime

更合理的做法：
- 先问“信号是谁发出来的”
- 再决定落到 `BoundaryEventTrigger`、`RuntimeEventTrigger` 或 `ModelJudgeTrigger`

### 3. `new-write conditioned` 更适合降成 hook

原因：
- 它在实现上通常只是：
  - 写入后自动触发的 graph update
  - 写入后自动触发的 merge/consolidate
- 它更像 `after_write` edge，而不是一整个 trigger 家族

## 对 `MemPrimitive` 的直接实现建议

如果下一步真要往代码层推进，我建议 trigger 接口先按实现层统一，而不是先按论文语义命名。

### 最小接口集合

1. `OnInputTrigger`
   - 默认直通写入
2. `BoundaryEventTrigger`
   - 接收外部边界事件
3. `RuntimeEventTrigger`
   - 接收执行/系统事件
4. `ScalarRuleTrigger`
   - 接收标量并做比较
5. `ModelJudgeTrigger`
   - 调模型做 yes/no、score、label 判断
6. `ScheduledMaintenanceTrigger`
   - 固定周期 / idle / sleep-time
7. `ControllerTrigger`
   - 交给外部 controller 决定

### 统一元数据建议

不管哪类 trigger，都统一成：

- `source`
  - `runtime`
  - `boundary`
  - `scheduler`
  - `model_judge`
  - `scalar_rule`
  - `controller`
- `scope`
  - `write`
  - `evolution`
- `event_name`
  - 可选
- `comparator`
  - 仅 `ScalarRuleTrigger` 需要
- `model_spec`
  - 仅 `ModelJudgeTrigger` 需要
- `schedule_spec`
  - 仅 `ScheduledMaintenanceTrigger` 需要

### 这比旧设计更接近文献现实的原因

- 文献里真正稳定复现的是“谁来发 trigger”
- 不是“trigger 在语义上叫 importance 还是 cue recall”
- 实现层统一后，文献对齐反而更容易

## 当前最需要谨慎的几类边界案例

### 1. retrieval-heavy 论文

像下面这些工作，旧 trigger survey 里可以保留，但如果严格按这次范围，应该降权：

- VideoAgent
- WorldMM
- 部分视频 episodic memory 论文

原因：
- 它们的大部分 trigger 都在“是否继续检索/取证”
- 不是 write/evolution trigger

### 2. offline pipeline 论文

像下面这些工作，最好不要硬塞进 scheduler 类：

- HippoRAG 2
- Hierarchical Memory Organization for Wikipedia Generation
- Towards LifeSpan Cognitive Systems

原因：
- “有后台阶段”不等于“有一个明确 trigger”

### 3. route / classifier 混合系统

像下面这些工作，最容易误判成独特语义 trigger：

- MemGPT
- MemOS
- MIRIX
- O-Mem
- M3-Agent

其实它们更像：
- `ModelJudgeTrigger`
- 或 `ControllerTrigger`

## 最后的结论

如果只看 write/evolution trigger 的**实现面**，而不是看论文口头上说了什么信号名词，40 篇工作并不支持继续扩张一个很厚的 trigger 语义体系。

更贴近论文现实的实现分层是：

1. 默认直通
2. 结构边界
3. 运行时回调
4. 显式标量规则
5. LLM 判定
6. 后台调度
7. controller 编排

其中真正值得在 `MemPrimitive` 里优先做成一等实现的，是：

- `RuntimeEventTrigger`
- `ModelJudgeTrigger`
- `ScalarRuleTrigger`
- `ScheduledMaintenanceTrigger`
- `BoundaryEventTrigger`

而不是继续围绕 `threshold(score)`、`boolean gate`、`new-write conditioned` 这些语义标签去加类。

## 参考来源

### 核心论文

- Generative Agents: Interactive Simulacra of Human Behavior. https://arxiv.org/abs/2304.03442
- Reflexion: Language Agents with Verbal Reinforcement Learning. https://arxiv.org/abs/2303.11366
- Voyager: An Open-Ended Embodied Agent with Large Language Models. https://arxiv.org/abs/2305.16291
- MemoryBank: Enhancing Large Language Models with Long-Term Memory. https://arxiv.org/abs/2305.10250
- "My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents. https://arxiv.org/abs/2404.00573
- Agent Workflow Memory. https://arxiv.org/abs/2409.07429
- VideoAgent: Long-form Video Understanding with Large Language Model as Agent. https://arxiv.org/abs/2403.10517
- In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents. https://arxiv.org/abs/2503.08026
- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. https://arxiv.org/abs/2504.19413
- Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science. https://arxiv.org/abs/2508.03341
- MIRIX: Multi-Agent Memory System for LLM-Based Agents. https://arxiv.org/abs/2507.07957
- Human-like Episodic Memory for Infinite Context LLMs. https://arxiv.org/abs/2407.09450

### 说明

- 这次文档以“实现面重排”为目标，因此大量工作使用了 [TRIGGERS.md](./TRIGGERS.md) 已完成的 40 篇原文核对结果作为底表，再补查一手来源做实现向重解释。
- 凡是写成 `OutOfScope`、`BackgroundPipeline`、`TrainingLoop` 的地方，意思是它不适合直接当作在线 write/evolution trigger primitive，而不是说论文没有 memory update。
