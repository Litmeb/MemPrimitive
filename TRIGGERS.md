# Trigger Survey for MemPrimitive

> Note
> This file is a literature-survey / design-space note, not a description of the current public baseline API.
> The baseline runtime has been simplified and now only publicly exposes `AlwaysWriteTrigger`, `ThresholdWriteTrigger`, `NeverEvolutionTrigger`, and `ThresholdEvolutionTrigger`.

更新时间：2026-03-31

## 这份文档是做什么的

这份文档只回答一个架构问题：

在当前 40 篇候选文献里，`trigger` 到底是不是一个值得在 `MemPrimitive` 里单独建成厚重子系统的核心维度？

这次更新不再把判断停留在 `core / secondary / generic` 三档，而是进一步把每篇论文落到更具体的：

- `write trigger` 原型
- `evolution / maintenance trigger` 原型
- 对应的 `signal`
- 触发机制在整篇论文中的真实作用

这里讨论的 trigger 包括：

- `write_trigger`
- `evolution_trigger`
- 更广义的“何时路由、压缩、反思、巩固、整理、迁移、离线维护”

本轮工作方式：

- 逐篇搜索并阅读论文原文页面，优先使用 arXiv / ACL Anthology / OpenReview / 项目主页等一手来源
- 目标不是重建完整方法细节，而是抽取“何时触发 memory 动作”这一层
- 对少数方法细节仍不完全公开的论文，保持保守概括，不擅自补不存在的复杂 trigger

## 本文使用的 trigger 原型词表

为了避免把每篇都写成自定义语言，这里统一使用一组较薄的 trigger 原型：

1. `always`
  - 每个 observation / step / turn 默认都进入写入或维护流程
2. `threshold(score)`
  - 由某个显式分数超过阈值触发
3. `boolean signal gate`
  - 某个布尔事件出现时触发
4. `failure/outcome-conditioned`
  - 失败、负反馈、结果不达标时触发
5. `capacity/budget-conditioned`
  - 缓冲区、token、槽位、存储预算到阈值时触发
6. `PeriodicTrigger`
  - 以固定周期、固定轮数、固定调度节拍触发
7. `SessionEndTrigger`
  - 在 session / chunk / stage / turn 收束时触发
8. `IdleTrigger`
  - 在 sleep-time、空闲窗口、后台低负载时触发
9. `new-write conditioned`
  - 只有出现新写入后，后续整理/压缩/链接才触发
10. `type-routing / multi-label routing`
  - 先判断应写入哪一类 memory，再路由到不同 store
11. `subgoal-completion conditioned`
  - 子目标完成、阶段闭合、episode 边界到来时触发

## 总体结论

把 40 篇论文逐篇落到具体 trigger 原型之后，结论比旧版更明确：

- 真正强依赖独特 trigger 的论文，仍然是少数
- 多数论文的 trigger 都能压缩到 `always`、`threshold(score)`、`boolean gate`、`capacity`、少量 `Periodic/SessionEnd/Idle` 这几类
- 旧的 `scheduled/offline` 过于宽泛：其中只有一部分真能稳定落到新的三类，剩下不少其实只是“存在后台阶段”，不宜硬编码成 trigger
- 真正拉开异质性的，大多仍然是：
  - `representation`
  - `organization`
  - `memory_evolution`
  - `retrieval`

因此，对 `MemPrimitive` 更合理的方向仍然是：

- 保留少数高价值 trigger 原型
- 不把 trigger 建成比 organization / evolution 更复杂的一层
- 把复杂性更多放在“触发后做什么”而不是“触发语言本身多复杂”

## 40 篇逐项映射

说明：

- `判断` 仍保留，方便和旧版衔接
- `写入原型` / `演化原型` 是建议映射到 `MemPrimitive` 的 trigger prototype
- `signal` 只写论文里真正能对应动作时机的关键信号
- `trigger 摘要` 聚焦“触发点到底扮演什么角色”


| #   | 论文                                                                            | 判断        | 写入原型                                                      | 演化原型                                                                      | 关键 signal                                                        | 详细 trigger 摘要                                                                                                                                                                           |
| --- | ----------------------------------------------------------------------------- | --------- | --------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Generative Agents                                                             | secondary | `threshold(score)`                                        | `threshold(score)`                                                        | `importance score`、最近性、相关性、累计 reflection 分数                      | 重新查原文后，更稳妥的说法是：Generative Agents 的核心 maintenance trigger 仍是累计 `importance` 超阈值后的 reflection。文中虽有日程规划与“新一天”节奏，但那更像 agent loop 的时间结构，不宜直接记成 `PeriodicTrigger` 或 `SessionEndTrigger`。      |
| 2   | Reflexion                                                                     | core      | `failure/outcome-conditioned`                             | `failure/outcome-conditioned`                                             | 任务失败、环境反馈、单元测试失败、奖励/结果不达标                                        | 这是最典型的 `failure-triggered reflection`。memory 写入与后续 verbal reinforcement 都直接由失败结果触发；没有失败就不会生成那条“下次该怎么做”的反思记忆。trigger 本身就是论文辨识度的一部分。                                                      |
| 3   | Voyager                                                                       | generic   | `boolean signal gate`                                     | `failure/outcome-conditioned` + `new-write conditioned`                   | 新技能成功发现、执行失败、任务完成反馈                                              | Voyager 的记忆主轴是 skill library。真正写入常发生在发现可复用技能或成功程序后，后续迭代则由执行结果驱动修正。它更像“技能抽取与程序改写”的 workflow，不是专门设计 trigger 机制的论文；可近似成“新技能出现才写入 + 失败时修订”。                                                 |
| 4   | MemGPT                                                                        | core      | `type-routing / multi-label routing`                      | `capacity/budget-conditioned` + `boolean signal gate`                     | memory pressure、上下文窗口压力、工具调用意图、显式中断                              | MemGPT 的关键不是单一写入阈值，而是 agent 在运行时决定该把信息留在 main context、写到 recall memory 还是 archival memory，并在 memory pressure / interrupt 下做迁移或分页。trigger 与 store 管理、路由和控制流紧耦合，是少数真正需要显式 trigger 原型族的系统。 |
| 5   | MemoryBank                                                                    | secondary | `threshold(score)`                                        | `threshold(score)` / `无法表示（连续时间衰减）`                                       | 遗忘曲线强度、重要度、时间衰减、重复提及                                             | 原文更接近“带时间衰减的 retention function”，而不是固定周期整理。写成 `scheduled/offline` 偏重了；更稳妥的是保留 `importance/retention threshold`，并注明一部分演化由连续时间变量驱动，不能被这套离散 trigger 干净表示。                                  |
| 6   | RET-LLM                                                                       | generic   | `always`                                                  | `new-write conditioned`                                                   | 新抽取 triplet、文档进入系统                                               | RET-LLM 更像“读写型知识存储接口”。新文本进入时默认抽取 triplet 并写入 memory，之后围绕这些写入做读取和时间型 QA；触发层几乎没有独特设计，接近 `always write`。                                                                                   |
| 7   | My Agent Understands Me Better                                                | secondary | `boolean signal gate`                                     | `threshold(score)` / `无法表示（时间流逝驱动巩固）`                                     | `cue recall` 命中、上下文相关性、时间流逝、回忆频次、巩固度                             | 重新核对后，文章更像 cue-based recall 加 consolidation score 演化，而非固定后台调度。原文档里的 `scheduled/offline` 有些过度概括；这里改为“可部分落到阈值，剩余由时间流逝驱动，不强行表示”。                                                           |
| 8   | Compress to Impress                                                           | secondary | `SessionEndTrigger`                                       | `SessionEndTrigger`                                                       | session 边界、对话轮段结束、压缩阶段启动                                         | 复核后更适合纯粹写成 `SessionEndTrigger`。COMEDY 强调的是 session-specific summary 与对话段落收束后的压缩，不是“子目标完成后触发”的 agentic subgoal 语义。                                                         |
| 9   | Toward Conversational Agents with Context and Time Sensitive Long-term Memory | generic   | `always`                                                  | `always` / `new-write conditioned`                                        | 新对话 turn、时间戳、事件顺序约束                                              | 论文主轴是 temporal/context-sensitive retrieval，不是写入触发。对话内容基本按 turn 持续进入长期记忆，关键差异在 retrieval 侧如何按时间、事件顺序和局部上下文重构查询。trigger 可近似为 `always write`。                                              |
| 10  | Agent Workflow Memory                                                         | secondary | `threshold(score)` / `boolean signal gate`                | `new-write conditioned` + `threshold(score)`                              | workflow 重用频率、成功率、任务相似性                                          | AWM 关心的是“什么时候把重复出现的 routine 抽象成 workflow memory，并在后续任务中选择性提供给 agent”。这不是纯 `always`，但也不是复杂 trigger 体系；更适合看成 `workflow utility / reuse threshold`，达到阈值后固化为 procedural memory。             |
| 11  | HiAgent                                                                       | core      | `always`（step 级 working memory 更新）                        | `subgoal-completion conditioned`                                          | 子目标完成、任务阶段闭合、trajectory chunk 完成                                 | HiAgent 最关键的 trigger 不在逐步写入，而在“什么时候把底层执行轨迹总结成上层 working memory chunk”。它明确以 subgoal 为 memory chunk，因此 summarization / abstraction 是 `subgoal-completion conditioned`，这是论文核心之一。           |
| 12  | Memory OS of AI Agent                                                         | secondary | `type-routing / multi-label routing`                      | `capacity/budget-conditioned`                                             | 短/中/长期层级判断、访问频率、生命周期阶段                                           | 触发点其实都是“存满/超限再迁移或淘汰”                                                                                                                                                                    |
| 13  | Mem0                                                                          | generic   | `threshold(score)`                                        | `new-write conditioned`                                                   | 记忆抽取置信、相关性、更新冲突检测                                                | Mem0 强调 selective memory formation，但 selectivity 主要落在“是否抽取出值得存的 memory item、是否与旧记忆 merge/update”。这更像 extraction/update 逻辑，不是独立 trigger 家族；可用 `memory-worthiness threshold` 概括。          |
| 14  | A-MEM                                                                         | secondary | `threshold(score)` + `type-routing / multi-label routing` | `new-write conditioned` + `boolean signal gate`                           | 重要度、关联邻居、可链接性、agent 决策                                           | A-MEM 的 agent 会判断新信息值不值得成为 note、应如何链接到现有 note 网络；后续图维护常由新写入或邻居关联触发。其复杂度主要在 graph-note organization，不在 trigger 语言本身。                                                                     |
| 15  | HippoRAG                                                                      | generic   | `always`                                                  | `new-write conditioned`                                                   | 新段落进入、triplet 抽取完成、query association                             | HippoRAG 的离线阶段基本是“文本来了就抽 triple / build graph”，在线阶段是图检索与联想式 recall。触发近似 `always write`，关键异质性在 graph retrieval 和 propagation。                                                            |
| 16  | From RAG to Memory / HippoRAG 2                                               | generic   | `always`                                                  | `new-write conditioned` / `无法表示（离线建图阶段）`                                  | 离线建图阶段、paragraph integration、query-time association              | HippoRAG 2 的“offline”更像系统管线阶段，而不是周期/会话结束/空闲触发。原文档把它记为 `scheduled/offline` 不算错，但在新分类下更适合明确写成“离线建图阶段无法由现有 trigger 原型精确表示”。                                                                |
| 17  | AriGraph                                                                      | generic   | `always`                                                  | `new-write conditioned`                                                   | 新 observation、图更新需求、graph pruning                                | AriGraph 在探索环境时持续把 observation 写入 episodic + semantic graph，并在每步后做 world-model 更新。更贴近 `always write` + `new-write conditioned graph update`；真正难点在图结构如何建模和裁剪。                            |
| 18  | Zep / Graphiti                                                                | secondary | `always`                                                  | `new-write conditioned`                                                   | 新对话/业务事件、时间戳、历史关系变化                                              | 复查后更稳妥的说法是：Graphiti 主要由新事件到达驱动时间知识图更新，原文并没有把后台整理明确写成固定周期或 idle 维护。因此去掉旧版里偏宽的 `scheduled/offline`，只保留 `new-write conditioned`。                                                           |
| 19  | Bridging Intuitive Associations and Deliberate Recall                         | generic   | `always`                                                  | `always` / `new-write conditioned`                                        | 新 utterance、事件抽取、查询意图                                            | Associa 的关键是 event-centric graph 与两阶段 retrieval。memory graph 的形成对新对话基本是持续的，主要复杂度在“直觉联想 + 深思检索”的读取过程，而不是何时写入。                                                                            |
| 20  | From Experience to Strategy                                                   | secondary | `always`                                                  | `threshold(score)` / `无法表示（训练式策略蒸馏阶段）`                                    | reward feedback、经验 utility、strategy weight                       | 这篇的高层 strategy abstraction 更像训练或蒸馏阶段，而不是明确的周期、session-end 或 idle 触发。因此旧版的 `scheduled/offline` 需要收回，保留 `utility threshold` 更稳妥。                                                          |
| 21  | Hierarchical Memory Organization for Wikipedia Generation                     | generic   | `always`                                                  | `无法表示（离线层级组织管线）`                                                          | 文档 chunk 进入、层级组织流程启动                                             | MOG 的层级组织是离线 pipeline，本质上不是 trigger 论文。旧版写 `scheduled/offline` 容易让它看起来像存在明确 maintenance 触发；这里改成直接承认“当前 trigger 词表无法精确表示”。                                                               |
| 22  | Optimizing the Interface Between KGs and LLMs                                 | generic   | `always`                                                  | `always`                                                                  | ingestion、检索超参数、提示参数                                             | 这篇本质是系统调参与接口优化，不是 memory trigger 论文。可以视为图构建与检索默认执行，没有值得单列的 trigger 原型。                                                                                                                  |
| 23  | AI-native Memory 2.0: Second Me                                               | secondary | `type-routing / multi-label routing`                      | `new-write conditioned` / `无法表示（后台参数化与迁移）`                                | 用户信息类型、记忆层级、参数化时机                                                | Second Me 确有后台参数化与层间迁移，但原文没有给出足够明确的周期、session-end 或 idle 条件。旧版 `scheduled/offline` 偏泛化；这里保留路由与写后演化，把剩余部分记为无法表示更稳妥。                                                                      |
| 24  | O-Mem                                                                         | secondary | `type-routing / multi-label routing`                      | `new-write conditioned` / `无法表示（后台层级整理）`                                  | persona 属性抽取、event record 抽取、topic/context 层级                    | O-Mem 的后续整理更像系统组织阶段，而不是新三类里某个稳定 trigger。这里顺手修正旧版，把 `scheduled/offline` 去掉，避免把“有后台整理”误写成“有明确 schedule”。                                                                                  |
| 25  | In Prospect and Retrospect                                                    | secondary | `SessionEndTrigger`                                       | `boolean signal gate` + `failure/outcome-conditioned`                     | session/turn/utterance 粒度边界、LLM cited evidence、检索误差              | 复核后应去掉 `subgoal-completion conditioned`。Prospective Reflection 是按 utterance / turn / session 多粒度边界生成总结，不是任务子目标完成触发。                                            |
| 26  | Nemori                                                                        | core-ish  | `boolean signal gate`                                     | `failure/outcome-conditioned` + `boolean signal gate`                     | event boundary、episode 对齐信号、prediction gap、calibration error     | 复核后不应再写 `subgoal-completion conditioned`。Nemori 的写入边界来自 event segmentation / episode alignment，而不是显式子目标完成；其演化则由 prediction gap 等信号驱动。                                  |
| 27  | MemOS                                                                         | generic   | `type-routing / multi-label routing`                      | `PeriodicTrigger` + `capacity/budget-conditioned`                         | memory form、版本/来源元数据、迁移成本                                        | MemOS 确实存在面向 memory scheduling 的后台调度；在新分类下，把它收窄为 `PeriodicTrigger` 比 `scheduled/offline` 更贴切，因为论文讨论的是持续调度与资源编排，而不是 session-end 或 sleep-time 语义。                                         |
| 28  | MIRIX                                                                         | core      | `type-routing / multi-label routing`                      | `capacity/budget-conditioned` + `PeriodicTrigger` + `boolean signal gate` | memory type classifier、模态来源、活跃使用度、存储预算、multi-agent controller 决策 | MIRIX 明确由多代理协同控制六类 memory 的更新与检索。把旧的 `scheduled/offline` 拆开后，这里更接近 controller 驱动的周期性维护，而不是 session-end 或 idle-only 语义，因此落到 `PeriodicTrigger` 更稳妥。                                       |
| 29  | SEDM                                                                          | core-ish  | `boolean signal gate` + `threshold(score)`                | `PeriodicTrigger` + `threshold(score)`                                    | verifiable replay 通过与否、经验 utility 排名、跨域扩散价值                      | SEDM 明确写到 self-scheduling controller；在新分类下，这更像 `PeriodicTrigger` 驱动的 consolidation，而不是泛泛的 offline。原文档的大方向是对的，但需要拆细。                                                                     |
| 30  | LightMem                                                                      | core      | `capacity/budget-conditioned`                             | `IdleTrigger` + `new-write conditioned`                                   | buffer 容量、topic grouping 完成、sleep-time 窗口                        | LightMem 是最明确的 trigger-rich 论文之一。把旧类拆开后，这里的离线巩固应直接写成 `IdleTrigger`，因为论文明确使用 sleep-time window；这一点比旧的 `scheduled/offline` 更准确。                                                           |
| 31  | Towards LifeSpan Cognitive Systems                                            | generic   | `always`                                                  | `无法表示（长期吸收阶段）`                                                            | 新 experience、长期吸收阶段                                              | 这更像上位蓝图。虽然讨论 experience absorbing 与 response generation，但没有给出足够具体的周期、session-end 或 idle 触发，因此这里不再强行放进新三类。                                                                               |
| 32  | Human-inspired Episodic Memory for Infinite Context LLMs / EM-LLM             | generic   | `boolean signal gate`                                     | `boolean signal gate` + `new-write conditioned`                           | `surprise`、event boundary、两阶段 retrieval 命中                       | EM-LLM 的重要控制点在 event segmentation：上下文不是任意定长切块，而是由 `surprise` 信号和边界精炼机制切成 episodic units。它确实有 signal，但更像 unit formation signal，而不是复杂 memory-action trigger 体系。                           |
| 33  | MemVerse                                                                      | secondary | `always`                                                  | `PeriodicTrigger` + `threshold(score)`                                    | multimodal distillation 周期、重要样本选择                                | MemVerse 的关键更像 multimodal memory + periodic distillation。这里可以明确落到 `PeriodicTrigger`，旧版 `scheduled/offline` 的意思基本正确，但新写法更精确。                                                             |
| 34  | MGA                                                                           | generic   | `always`                                                  | `new-write conditioned`                                                   | GUI observation、trajectory step                                  | MGA 面向 observation-centric GUI memory，通常每个 step / state 都持续形成结构化外部记忆。后续更新只是随着新轨迹写入自然发生，trigger 不构成单独创新点。                                                                                |
| 35  | KARMA                                                                         | secondary | `type-routing / multi-label routing`                      | `capacity/budget-conditioned`                                             | short-term vs long-term scene 信息、对象状态变化、短期槽位替换压力                 | KARMA 区分长期 3D scene graph 和记载局部变化的短期记忆，并明确有 short-term replacement 策略。写入需要先判断进哪类 store，短期更新又受容量和重要变化影响，因此可映射成 `type-routing` + `capacity-conditioned replacement`。                      |
| 36  | VideoAgent                                                                    | generic   | `boolean signal gate`                                     | `failure/outcome-conditioned` / `boolean signal gate`                     | query-driven frame request、当前证据不足                                | VideoAgent 不是典型长期 memory 论文，但如果硬要映射 trigger，其控制点在 agent 何时继续调用视觉工具、何时检索更多帧。这个 trigger 是 query-driven evidence gathering，不是 memory 写入 trigger。                                           |
| 37  | WorldMM                                                                       | secondary | `type-routing / multi-label routing`                      | `boolean signal gate` / `无法表示（读侧时序控制）`                                    | query 类型、所需模态、时间粒度选择、证据充分性                                       | WorldMM 的非平凡控制主要在 retrieval routing 与 temporal granularity 选择。旧版把它并入 `scheduled/offline` 偏宽；更准确的是承认这里主要是读侧控制，不适合硬塞进新的 maintenance trigger 三分法。                                          |
| 38  | Seeing, Listening, Remembering, and Reasoning / M3-Agent                      | generic   | `type-routing / multi-label routing`                      | `new-write conditioned`                                                   | 实体抽取、模态来源、实体级更新                                                  | M3-Agent 会将视觉、听觉输入组织为 entity-centric multimodal memory，并持续更新 episodic / semantic memory。它确实有“按实体/模态落库”的路由信号，但这更多是表示层决定，不是复杂 trigger 设计。                                                 |
| 39  | HippoMM                                                                       | generic   | `boolean signal gate`                                     | `IdleTrigger` + `new-write conditioned`                                   | adaptive temporal segmentation、跨模态联想需求                           | HippoMM 用 adaptive temporal segmentation 形成 episodic units，再做短长时巩固与跨模态检索。若必须落到新三类，较稳妥的是把类似 consolidation 的离线巩固写成 `IdleTrigger`；但这仍然不是论文主轴。                                              |
| 40  | Episodic Memory Representation for Long-form Video Understanding              | generic   | `boolean signal gate`                                     | `boolean signal gate`                                                     | episodic event boundary、CoT 选择的关键 episodic subset                | Video-EM 的关键是把关键帧视为 temporally ordered episodic events，并由 CoT 迭代挑出最小但信息量最高的 episodic subset。这里的“触发”更多是事件分段和检索迭代条件，不是长期 memory action trigger。                                           |


## 更严格的统计

如果只把“trigger 明确构成系统骨架”的论文算作严格 `core`：

- Reflexion
- MemGPT
- HiAgent
- MIRIX
- LightMem

这是保守的 `5 / 40`。

如果把 trigger 设计也非常突出、但更偏控制原则而非完整工程语义的条目算进来：

- Nemori
- SEDM

则是 `7 / 40`。

如果再把“有明确而必要的非平凡 trigger，但主创新仍在别处”的 `secondary` 也纳入：

- Generative Agents
- MemoryBank
- My Agent Understands Me Better
- Compress to Impress
- Agent Workflow Memory
- Memory OS of AI Agent
- A-MEM
- Zep
- From Experience to Strategy
- Second Me
- O-Mem
- In Prospect and Retrospect
- MemVerse
- KARMA
- WorldMM

则大致是 `22 / 40` 至少具有“可命名的 trigger 原型”，但其中绝大多数仍能被薄 trigger 词表覆盖。

重新逐篇核对后，关于旧的 `scheduled/offline` 还需要补一层说明：

- 真正能明确落到 `PeriodicTrigger` 的，只占少数，代表性条目是 MemOS、MIRIX、SEDM、MemVerse
- 真正能明确落到 `SessionEndTrigger` 的也不多，代表性条目是 Compress to Impress、In Prospect and Retrospect
- 真正能明确落到 `IdleTrigger` 的更少，最典型的是 LightMem；HippoMM 只能算保守近似
- 其余不少旧条目虽然“有离线阶段/后台维护”，但原文并没有给出足够明确的周期、会话结束或空闲窗口语义，因此更诚实的写法是 `无法表示`

## 对 MemPrimitive 的直接启发

### 1. 最值得保留的高价值 trigger 原型

如果目标是“开始重设计整个 trigger 类”，我建议把“高频出现”与“值得做成一等实现”分开看。

先说结论：

- **建议做成一等 trigger 类的**：
  - `threshold(score)`
  - `boolean signal gate`
  - `failure/outcome-conditioned`
  - `capacity/budget-conditioned`
  - `PeriodicTrigger`
  - `SessionEndTrigger`
  - `IdleTrigger`
  - `type-routing / multi-label routing`
  - `subgoal-completion conditioned`
- **建议保留，但不要做成厚重类层次的**：
  - `always`
  - `new-write conditioned`

原因是：

- `always` 虽然出现多，但它更像默认空触发器，通常不值得作为复杂类实现
- `new-write conditioned` 也出现很多，但本质上常常只是“某次 write 之后自动挂接的后处理”，更适合做事件钩子或 pipeline edge，而不是独立策略对象
- 真正值得单独建模的，是那些会显著改变系统控制流、memory routing 或 maintenance scheduling 的 trigger

下面按出现次数统计，并保留出现在哪些论文中的信息。


| trigger 原型                           | 出现篇数 | 是否建议作为一等实现 | 说明                                                                                          | 出现于                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------ | ---- | ---------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PeriodicTrigger`                    | 4    | 是          | 真正明确写成周期调度的并不多，但一旦出现通常就是系统维护骨架。                                                             | MemOS；MIRIX；SEDM；MemVerse                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `SessionEndTrigger`                  | 2    | 是          | 频次不高，但很稳定地对应“chunk / turn / session 收束后再压缩或总结”。                                             | Compress to Impress；In Prospect and Retrospect                                                                                                                                                                                                                                                                                                                                                                                              |
| `IdleTrigger`                        | 2    | 是          | 最典型的是 sleep-time / idle-window consolidation。明确出现的论文很少，但语义非常清楚。                             | LightMem；HippoMM（保守近似）                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `new-write conditioned`              | 18   | 否，建议降为事件钩子 | 这类触发高频出现，但多数不是论文主创新，而是“新记忆写入后自动做 link / merge / consolidate / organize”。更适合做 `on_write` 钩子。 | Voyager；RET-LLM；Toward Conversational Agents with Context and Time Sensitive Long-term Memory；Agent Workflow Memory；Mem0；A-MEM；HippoRAG；From RAG to Memory / HippoRAG 2；AriGraph；Zep / Graphiti；Bridging Intuitive Associations and Deliberate Recall；AI-native Memory 2.0: Second Me；O-Mem；LightMem；Human-inspired Episodic Memory for Infinite Context LLMs / EM-LLM；MGA；Seeing, Listening, Remembering, and Reasoning / M3-Agent；HippoMM |
| `always`                             | 14   | 否，建议作为默认值  | 出现频率高说明很多论文根本不依赖复杂触发；但这恰恰说明它不该是复杂类，而应是默认 no-op / passthrough trigger。                       | RET-LLM；Toward Conversational Agents with Context and Time Sensitive Long-term Memory；HiAgent；HippoRAG；From RAG to Memory / HippoRAG 2；AriGraph；Zep / Graphiti；Bridging Intuitive Associations and Deliberate Recall；From Experience to Strategy；Hierarchical Memory Organization for Wikipedia Generation；Optimizing the Interface Between KGs and LLMs；Towards LifeSpan Cognitive Systems；MemVerse；MGA                                    |
| `boolean signal gate`                | 14   | 是          | 这是最通用、最该保留的基础原型。很多论文其实不需要 scorer/gate/policy 全套，只需要“某个信号出现就触发”。                             | Voyager；MemGPT；My Agent Understands Me Better；Agent Workflow Memory；A-MEM；In Prospect and Retrospect；Nemori；MIRIX；SEDM；Human-inspired Episodic Memory for Infinite Context LLMs / EM-LLM；VideoAgent；WorldMM；HippoMM；Episodic Memory Representation for Long-form Video Understanding                                                                                                                                                        |
| `type-routing / multi-label routing` | 10   | 是          | 这是少数真正会改变 memory topology 的 trigger 原型。对于多 store / 多 memory type 系统非常关键。                    | MemGPT；Memory OS of AI Agent；A-MEM；AI-native Memory 2.0: Second Me；O-Mem；MemOS；MIRIX；KARMA；WorldMM；Seeing, Listening, Remembering, and Reasoning / M3-Agent                                                                                                                                                                                                                                                                                 |
| `threshold(score)`                   | 9    | 是          | 这是最常见的“轻量智能选择”原型。比起复杂 policy，很多论文只需要 importance / utility / retention / worthiness 过阈值。     | Generative Agents；MemoryBank；My Agent Understands Me Better；Agent Workflow Memory；Mem0；A-MEM；From Experience to Strategy；SEDM；MemVerse                                                                                                                                                                                                                                                                                                      |
| `capacity/budget-conditioned`        | 6    | 是          | 出现频率不算最高，但工程价值极高，因为它正好对应 buffer、window、slot、storage budget 这些真实系统压力。                        | MemGPT；Memory OS of AI Agent；MemOS；MIRIX；LightMem；KARMA                                                                                                                                                                                                                                                                                                                                                                                     |
| `failure/outcome-conditioned`        | 5    | 是          | 虽然只在少数论文中出现，但这些论文辨识度很高，尤其是 Reflexion 系。这个原型必须保留。                                            | Reflexion；Voyager；In Prospect and Retrospect；Nemori；VideoAgent                                                                                                                                                                                                                                                                                                                                                                              |
| `subgoal-completion conditioned`     | 1    | 是          | 频次很低，但一旦出现通常就是系统核心。严格复核后，当前 40 篇里最稳妥的典型主要就是 HiAgent。                                         | HiAgent                                                                                                                                                                                                                                                                                                                                                                                                                                      |


因此，如果只保留一个**最小但够用**的 trigger 类集合，我建议是：

1. `AlwaysTrigger`
  - 作为默认实现存在，但尽量薄，不要继续堆复杂子类
2. `ThresholdTrigger`
  - 覆盖 importance / utility / retention / worthiness
3. `BooleanSignalTrigger`
  - 覆盖 cue、event boundary、prediction gap、evidence sufficiency 等布尔信号
4. `FailureTrigger`
  - 专门保留给 Reflexion / outcome-driven systems
5. `CapacityTrigger`
  - 覆盖 buffer / token / slot / budget pressure
6. `PeriodicTrigger`
  - 覆盖固定周期、固定轮数、调度节拍触发
7. `SessionEndTrigger`
  - 覆盖 session / turn / chunk / stage 收束
8. `IdleTrigger`
  - 覆盖 sleep-time / idle-window / 后台空闲维护
9. `RoutingTrigger`
  - 覆盖 type-routing / multi-label routing / store selection
10. `SubgoalTrigger`
  - 覆盖阶段闭合、子目标完成、episode close

其中：

- `NewWriteConditioned` 不建议做成独立 heavyweight trigger class
- 更建议把它降成：
  - `on_write` hook
  - `after_write` maintenance edge
  - 或者 `PeriodicTrigger` / `SessionEndTrigger` / `IdleTrigger` / `RoutingTrigger` 内部可组合条件

如果还要再砍一刀，只保留**真正必要**的一组，我会去掉 `AlwaysTrigger` 的“类地位”，把它当默认配置值，只保留：

1. `ThresholdTrigger`
2. `BooleanSignalTrigger`
3. `FailureTrigger`
4. `CapacityTrigger`
5. `PeriodicTrigger`
6. `SessionEndTrigger`
7. `IdleTrigger`
8. `RoutingTrigger`
9. `SubgoalTrigger`

这 9 类已经足够覆盖当前 40 篇里绝大多数非平凡 trigger 设计，同时不会把 trigger 子系统重新做重；而那些只有“后台阶段”但没有明确调度语义的系统，则应直接允许写成 `无法表示`。

### 2. 哪些 signal 值得在框架里作为一等输入

如果 trigger 层要薄，但仍保留论文对齐能力，最值得显式支持的 signal 是：

- `importance`
- `salience`
- `relevance`
- `novelty`
- `retention / forgetting strength`
- `consolidation strength`
- `memory pressure / capacity usage`
- `subgoal_done`
- `failure_detected`
- `prediction_gap`
- `event_boundary`
- `cue_match`
- `type_label`
- `utility / reward`
- `evidence_sufficiency`

### 3. 更合理的系统边界

这一轮逐篇映射之后，更清楚的一点是：

- 多数论文并不需要 `signal provider -> scorer -> gate -> policy` 这套厚 trigger 堆栈全开
- 很多所谓 trigger 差异，最终都只是：
  - 一个简单 signal
  - 一个阈值
  - 一个布尔条件
  - 一个 schedule
  - 一个路由标签

因此比较自然的方向是：

- trigger 只负责“何时发生”
- representation / organization / evolution 负责“发生后怎么做”
- 不把大量论文的组织逻辑误塞进 trigger 层

## 结论

从 40 篇论文的原文级逐篇筛看，`MemPrimitive` 没有必要长期维护一套高度展开的 trigger 子语言。

更好的做法是：

- 用少数稳定 trigger 原型覆盖大多数论文
- 把复杂性集中在 memory 单元形成、组织、演化和检索
- 只对极少数 trigger-heavy 系统保留更明确的 failure / capacity / routing / subgoal / offline 语义

也就是说，trigger 依然重要，但更像“薄控制层”，而不是整个 memory ontology 里最重的主战场。
