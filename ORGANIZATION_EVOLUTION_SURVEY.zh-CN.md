# 40 篇候选文献的 organization / evolution 速查

更新时间：2026-04-02

## 这份文档做什么

这是一份面向 `MemPrimitive` 文献池的快速检索笔记，只关心两件事：

- `organization`：记忆是怎样被分层、分型、成图、成段、成库、成工作流来组织的
- `evolution`：记忆是怎样被压缩、巩固、更新、重组、淘汰、迁移或自我优化的

说明：

- 本文档刻意少做理论延伸，主要依据论文摘要、官方项目页、ACL/ICLR/arXiv/OpenReview 页面做逐篇概括。
- 个别 2025-2026 论文公开信息仍偏摘要级，因此这里写的是“当前可确认的方法级总结”，不是 full implementation reverse engineering。

---

## A. 经典锚点与高契合主线

| # | 论文 | organization | evolution |
|---|---|---|---|
| 1 | Generative Agents | 以自然语言 `memory stream` 顺序存储经历，再通过 `importance / recency / relevance` 检索；reflection 作为比原始 observation 更高层的记忆条目进入同一流。 | 主要通过周期性/阈值式 `reflection synthesis` 把低层经历抽象成高层洞见；新 reflection 会回写并继续参与后续检索与计划。 |
| 2 | Reflexion | 组织上很薄，核心是一个 `episodic buffer`，存放失败轨迹后的 verbal feedback / self-reflection。 | 演化核心是“失败后生成 lesson”，把 trial-level feedback 压缩成反思文本；记忆不是复杂重组，而是不断追加更有效的 reflective hints。 |
| 3 | Voyager | 记忆主体不是对话片段，而是 `skill library`；以可执行代码技能为单位组织，形成 procedural memory 库。 | 通过成功执行后的技能抽取、去重、泛化和复用来演化；本质上是把一次次交互轨迹蒸馏成越来越稳定的技能程序。 |
| 4 | MemGPT | 明确分成 `working / recall / archival` 多层存储，由 agent 自己管理 paging 与不同层之间的访问。 | 演化体现为跨层迁移与内存管理：热点内容留在上下文，旧内容转入外部存储；通过中断与 memory management policy 持续重排工作集。 |
| 5 | MemoryBank | 以长期对话为主，组织上接近 `chronological repository + event summary + personality/profile memory`。 | 演化依赖遗忘曲线式 memory strength 更新、回忆强化、事件总结和人格画像逐步累积；记忆会随时间衰减但被再次召回时增强。 |
| 6 | RET-LLM | 将知识写成可读写的 `triplets`，因此组织核心是可聚合、可更新、可解释的结构化三元组记忆单元。 | 演化主要是对 triplet memory 的追加、聚合与更新；时间信息也可进入记忆，从而支持时态相关知识修订。 |
| 7 | “My agent understands me better” | 记忆库存每条内容及其时间上下文，并结合 cue-based recall；更像 `temporal event memory database`。 | 明确引入 consolidation 模型：根据相关性、时间流逝、回忆频次动态更新巩固度；高巩固记忆衰减更慢，召回本身会再次强化。 |
| 8 | Compress to Impress / COMEDY | 不强调显式 memory DB，而是把多轮会话压缩成统一的 `compressive memory`，其中融合 session summary、user-bot dynamics、past events。 | 演化是持续压缩与重写：不是保留大量原子记忆，而是反复把历史重整成更短、更可用的压缩表征。 |
| 9 | Toward Conversational Agents with Context and Time Sensitive Long-term Memory | 面向会话检索，把记忆组织成带时间/事件顺序索引的历史对话存储，并强调 surrounding context。 | 演化重点不在复杂写后维护，而在保留事件顺序、时间标签和上下文邻域，以便后续基于时序条件重新取回。 |
| 10 | Agent Workflow Memory | 把记忆单元组织成 `workflow / routine`，即由多个步骤组成、可重复使用的程序性工作流。 | 演化来自 offline/online workflow induction：从过去成功案例中归纳流程，在后续任务中选择性注入；工作流会因复用而沉淀、因新案例而扩充。 |
| 11 | HiAgent | working memory 不是平铺历史，而是按 `subgoal / chunk` 分层组织，形成 hierarchical working memory。 | 演化主要是工作记忆重组：随着子目标推进，不断 chunk、压缩、上卷旧步骤，只把当前相关层级保留在高带宽工作区。 |
| 12 | Memory OS of AI Agent | 三层结构很明确：`short-term / mid-term / long-term personal memory`，并配套 storage / updating / retrieval / generation 四模块。 | short→mid 采用对话链 FIFO 更新，mid→long 用 segmented page organization；核心是跨层迁移和分页式巩固。 |
| 13 | Mem0 | 通常把记忆抽取成事实/偏好/长期有用信息并存入外部记忆库，组织上偏 `production fact memory + retrieval index`。 | 演化主要是 agentic extraction、去噪、更新与覆盖旧事实；不是原样存对话，而是持续从交互中提炼“值得长期保留”的稳定记忆。 |
| 14 | A-MEM | 明确受 Zettelkasten 启发，把记忆做成带 `context / keywords / tags / embedding / links` 的 note network。 | 演化核心很强：代理不仅写入新 note，还会自主重连、重组、整理和扩展知识网络，使记忆结构随新经验自组织增长。 |

---

## B. 图记忆、结构化记忆、层级检索

| # | 论文 | organization | evolution |
|---|---|---|---|
| 15 | HippoRAG | 用知识图/关联图组织长期记忆，把实体、关系、文档片段组织成可联想检索的图式记忆。 | 演化主要体现在图构建与关联强化；新增知识会被接到已有图上，使后续检索可通过图传播完成 associative recall。 |
| 16 | From RAG to Memory / HippoRAG 2 | 从“文档检索”推进到“non-parametric continual memory”，组织上仍是可增长的结构化外部记忆，但更强调 continual acquisition。 | 演化是持续学习式的外部记忆扩展与重整：新知识进入非参数记忆后，不靠重训模型，而靠记忆更新支持 continual learning。 |
| 17 | AriGraph | 用 `memory graph` 统一 semantic memory 与 episodic memory，形成 world-model 式知识图。 | 演化通过探索过程中的图更新完成：新事件/状态写入 episodic 与 semantic 子图，链接、状态与世界模型会持续修正。 |
| 18 | Zep | 把 agent memory 组织成 `bi-temporal knowledge graph`，显式处理实体、关系及其时间有效性。 | 演化重点是 temporal graph maintenance：图节点/边会随着新事件增量更新，并保留时间维上的版本与历史关系。 |
| 19 | Bridging Intuitive Associations and Deliberate Recall | 组织核心是 `graph-structured long-term memory`，面向 personal assistant 的 event-centric 图式存储。 | 演化主要是把长历史对话转成图并随交互扩展，再用图支持“联想式召回 + 审慎式回忆”的两阶段取回。 |
| 20 | From Experience to Strategy | 记忆组织成 `multi-layer graph memory`：底层是经验路径/状态机，上层是更高层的 strategy / meta-cognition。 | 演化很强：通过 reward feedback 优化图中高层策略权重，把经验蒸馏成可训练、可强化的策略记忆。 |
| 21 | Hierarchical Memory Organization for Wikipedia Generation | 从网页文档中抽取 fine-grained memory units，再递归组织成与条目提纲对齐的层级树。 | 演化主要表现为 recursive organization 与按层聚合，而不是在线 agent 式自我维护；低层材料会不断被归并到更高层主题结构。 |
| 22 | Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning | 记忆主体是 Cognee 式 KG pipeline，组织维度包括 chunking、graph construction、retrieval interface 与 prompting interface。 | 演化不是记忆自治更新，而是通过系统性超参数优化改进图构造与接口配置；更像“memory pipeline tuning”。 |
| 23 | AI-native Memory 2.0: Second Me | 组织上是面向个人知识的 persistent memory intermediary，强调结构化用户知识、上下文推理和跨系统调用。 | 演化体现在动态利用与 memory parameterization：个人知识会持续积累、结构化、参数化，并在新场景中被自适应调用。 |
| 24 | O-Mem | 组织上明确双轨：`active user profiling + event records`，并支持 persona 属性与主题上下文的分层检索。 | 演化是动态抽取并更新用户画像与事件记录；随主动交互持续改写 profile，同时保留主题相关长期经历。 |
| 25 | In Prospect and Retrospect / RMM | personalized memory bank 采用多粒度组织：utterance、turn、session 三层 prospective summary。 | 演化分两部分：前瞻式 reflection 负责形成多粒度总结；回顾式 reflection 依据引用证据在线强化检索策略，相当于 retrieval policy 自适应更新。 |
| 26 | Nemori | 组织核心是受 Event Segmentation Theory 启发的 `semantically coherent episodes`，不是任意固定窗口。 | 演化核心是 Predict-Calibrate：系统根据 prediction gap 主动校正和更新知识，超越被动规则式提取；记忆会因预测失败而重估和重组。 |

---

## C. 系统化、分布式、自演化与多存储

| # | 论文 | organization | evolution |
|---|---|---|---|
| 27 | MemOS | 显式统一 `plaintext / activation / parameter` 三类 memory form；基本单元是带 provenance/versioning 的 `MemCube`。 | 演化能力很强：MemCube 可组合、迁移、融合，在不同 memory form 之间转化，形成从外部检索到参数学习的桥。 |
| 28 | MIRIX | 组织成六类记忆：`Core / Episodic / Semantic / Procedural / Resource / Knowledge Vault`，并由多代理协调管理。 | 演化体现在多代理控制更新与检索，各 memory type 持续增量累积；系统会随多模态经历扩展并重组不同记忆子库。 |
| 29 | SEDM | 分布式记忆组织上包含可验证写入、控制器管理的 memory pool，以及跨域可扩散的抽象知识。 | 演化是全文重点：可复现回放驱动 write admission，自调度 controller 按经验效用排序与 consolidation，并把可复用知识扩散到新域。 |
| 30 | LightMem | 采用三阶段组织：`sensory memory / topic-aware short-term memory / long-term memory`，对应过滤、整理、长期保存。 | 演化通过 topic grouping、summary consolidation 和 `sleep-time offline update` 完成；把在线响应与离线巩固解耦。 |
| 31 | Towards LifeSpan Cognitive Systems | 不是单一实现，而是把记忆组织问题提升到 `experience absorption + response generation` 的系统蓝图，并讨论多类存储技术协同。 | 演化的核心命题是 experience merging、incremental updates、长期保留与准确回忆；强调单一技术不足，需要跨机制联动吸收一生经验。 |
| 32 | Human-inspired Episodic Memory for Infinite Context LLMs / EM-LLM | 按 surprise-based event segmentation 把长上下文切成 episodes，再做两阶段 episodic retrieval。 | 演化主要是事件边界 refinement 与 episodic store 扩展；新上下文不断被重分段、入库，并在取回阶段按 episode 而非 token 重组。 |
| 33 | MemVerse | 面向 lifelong multimodal agent 的统一多模态记忆架构，组织上强调分层、多模态、长期可累积的 memory substrate。 | 演化依赖持续学习和 multimodal knowledge management，通常包含长期积累、跨模态整合和阶段性 distillation。 |
| 34 | MGA | 每一步状态被组织为三元组：`current screenshot + task-agnostic spatial info + dynamically updated structured memory`。 | 演化体现在 GUI 交互中持续更新 structured memory summary，让后续决策不必依赖整条历史轨迹；更像 observation-centric 在线重写。 |

---

## D. 多模态 / 具身 / 边界扩展

| # | 论文 | organization | evolution |
|---|---|---|---|
| 35 | KARMA | 明确区分 `short-term memory` 与 `long-term memory`，服务于 embodied planning 的双存储协作。 | 演化主要是短期经历进入长期记忆并在后续规划中重用；通过 memory-augmented prompting 持续刷新当前计划上下文。 |
| 36 | VideoAgent | 组织上更像 agentic retrieval workspace：围绕长视频的 segment / evidence 逐步查询、收集和汇总。 | 演化较弱，重点不是长期自治维护，而是多轮 agent 式检索中不断补充、修正证据集合。 |
| 37 | WorldMM | 明确三类 memory：`episodic / semantic / visual`，并且 episodic 还采用多时间尺度组织。 | 演化上，semantic memory 会持续吸收高层概念与习惯，episodic/visual 库随视频推进扩展；检索 agent 还会按 query 自适应切换不同 memory source。 |
| 38 | Seeing, Listening, Remembering, and Reasoning / M3-Agent | 组织为 `entity-centric multimodal memory`，同时维护 episodic 与 semantic memories，融合视觉和听觉输入。 | 演化是随着实时多模态输入不断 build and update episodic/semantic memory，逐步积累 world knowledge。 |
| 39 | HippoMM | 组织上是 hippocampal-inspired multimodal memory：通过自适应时间分段和 dual-process encoding 形成 integrated episodic representations。 | 演化核心包括 short-to-long consolidation、pattern separation/completion 和跨模态关联强化；细节感知会逐步抽象成可检索语义记忆。 |
| 40 | Video-EM（原表述：Episodic Memory Representation for Long-form Video Understanding） | 组织核心是 `event-centric episodic memories`：先定位 query 相关时刻，再分组成 temporally coherent events，编码为带 `when / where / what / entity` 的 event timeline。 | 演化明确包含 self-reflection loop：迭代检查证据是否充分、是否跨事件一致、是否冗余，并自适应调整 event granularity，最后得到最小充分 episodic set。 |

---

## 横向观察

只看 `organization`，这 40 篇大致可归到下面几类：

- `stream / buffer / chronological store`：Generative Agents、Reflexion、MemoryBank
- `hierarchical stores`：MemGPT、Memory OS、MemOS、LightMem、HiAgent
- `graph-structured memory`：HippoRAG、AriGraph、Zep、A-MEM、Bridging、Trainable Graph Memory
- `workflow / procedural memory`：Voyager、Agent Workflow Memory、MIRIX 的部分模块
- `profile + event dual-track memory`：Mem0、O-Mem、Second Me、RMM
- `event-centric episodic organization`：Nemori、EM-LLM、HippoMM、Video-EM、WorldMM
- `multimodal layered memory`：MIRIX、WorldMM、M3-Agent、HippoMM、MemVerse

只看 `evolution`，最常见的模式是：

- `summary / reflection`：Generative Agents、Reflexion、RMM、Video-EM
- `consolidation / forgetting`：MemoryBank、My agent understands me better、LightMem、HippoMM
- `cross-store migration / paging`：MemGPT、Memory OS、MemOS
- `graph update / relinking`：A-MEM、AriGraph、Zep、HippoRAG 系
- `profile update`：Mem0、O-Mem、Second Me
- `policy / controller optimization`：Trainable Graph Memory、SEDM、RMM
- `event segmentation refinement`：Nemori、EM-LLM、Video-EM

## 对 MemPrimitive 的直接含义

如果只从 `organization` 和 `evolution` 两个维度看，当前文献池最稳定的高频需求不是更复杂的 trigger，而是：

1. `hierarchical organization`
2. `event-centric episodic unit formation`
3. `profile/event dual-track memory`
4. `summary/consolidation families`
5. `graph relinking and graph update`
6. `cross-store migration`
7. `workflow/procedural memory units`
8. `multimodal memory partitioning`

---

## 本轮检索主要来源

- arXiv / OpenReview / ACL Anthology 官方摘要页
- 官方项目页：Voyager、EM-LLM、WorldMM 等
- 少量补充来源用于交叉确认公开方法描述

如果下一轮继续推进，最自然的下一步是把这里的 `organization / evolution` 进一步翻译成 `MemPrimitive` 的 `organization module family` 和 `memory evolution module family` 候选表。
