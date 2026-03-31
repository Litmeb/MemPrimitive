# MemPrimitive 文献候选池（第一版）

更新时间：2026-03-31

## 这份文档是做什么的

这是一份面向 `MemPrimitive` 的 agent memory 文献候选池草案，目标不是做“最全 paper list”，而是为“统一重表达既有工作”服务。

当前版本的目标：

1. 先给出约 40 篇候选，覆盖 episodic / semantic / reflective / graph / working / multi-store / multimodal 等主要风格。
2. 先筛出一批和 `MemPrimitive` 现有 primitive slots 最契合的工作，优先支撑“约 1/4 可完全重表达”的中期目标。
3. 明确哪些论文更像“直接重表达对象”，哪些更像“补模块压力测试对象”。

---

## 读综述后的筛选原则

### 1. 从两篇综述里提炼出来的最有用视角

- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
  - 把 agent memory 放在 `forms / functions / dynamics` 三个视角下看：
  - `forms`: token-level / parametric / latent
  - `functions`: factual / experiential / working
  - `dynamics`: formation / evolution / retrieval
- [A Survey on the Memory Mechanism of Large Language Model based Agents](https://arxiv.org/abs/2404.13501)
  - 更强调 memory module 的设计、评测和应用拆解，适合拿来做初始候选池扩展。

对 `MemPrimitive` 来说，最直接可映射的是 `dynamics` 视角，因为它几乎可以顺手落到：

- `unit formation`
- `representation`
- `write trigger`
- `organization`
- `evolution trigger`
- `memory evolution`
- `retrieval`
- `readout`

### 2. 本文档的筛选偏置

我刻意优先保留以下几类论文：

- 明确有“写入-组织-更新-检索”闭环的
- 有可辨认 memory unit / memory store / retrieval policy 的
- 能较自然映射到现有 baseline/classic motif 的
- 已经在 LoCoMo、LongMemEval、对话、长时任务、图记忆、GUI/具身任务里形成代表性的

我刻意降低优先级的类型：

- 只有长上下文、没有显式 memory lifecycle 的
- 只有参数记忆，没有可拆解外部 memory mechanics 的
- 纯 benchmark / 纯安全 / 纯评测论文

### 3. 这里的“影响力”怎么理解

这里的“有一定影响力”不是只看引用数。当前 agent memory 领域很新，2025-2026 的不少工作还来不及形成稳定 citation。这里综合考虑：

- 是否是经典锚点
- 是否被综述、awesome list、后续系统反复提及
- 是否进入主流 benchmark / demo / 开源框架讨论
- 是否对 `MemPrimitive` 的 primitive ontology 有补模块价值

因此，列表里既有“成熟锚点”，也有“新近但非常契合框架”的强相关工作。

---

## 我建议优先做“完全重表达”尝试的 10 篇

这 10 篇最适合先做第一批 fully re-expressed target：

1. [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
2. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
3. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
4. [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)
5. A-MEM: Agentic Memory for LLM Agents
6. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)
7. AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents
8. Memory OS of AI Agent
9. [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
10. HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model

原因很简单：

- 它们都能被拆成比较清楚的 memory pipeline，而不是只能当成整体 black box。
- 它们基本都能落到 `MemPrimitive` 已有 slots 上，最多只需要补少量 module family。
- 它们一起覆盖了 observation-stream、reflection、graph、hierarchy、workflow、working memory、self-managed memory 这几条主线。

---

## 约 40 篇候选清单

优先级说明：

- `A`：建议尽快进入重表达主线
- `B`：强相关候选，适合作为第二梯队
- `C`：边界扩展/补模块压力测试对象

### A. 经典锚点与高契合主线（14 篇）

| # | 论文 | 年份 | 记忆主轴 | 与 MemPrimitive 的契合点 | 优先级 |
|---|---|---:|---|---|---|
| 1 | [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) | 2023 | episodic + reflection + planning | observation、reflection、retrieval、readout 都很清楚 | A |
| 2 | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | 2023 | failure-triggered reflective memory | 已有 Reflexion-like motifs，最适合做标杆重表达 | A |
| 3 | [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | 2023 | skill/procedural memory | 适合推动 skill memory / reusable workflow memory | A |
| 4 | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | 2023 | multi-store / self-managed memory | 已有 classic wrapper；适合继续补强 store migration 与 paging 语义 | A |
| 5 | [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250) | 2023 | long-term dialogue memory | profile/summary/event memory 的基础锚点 | A |
| 6 | [RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/abs/2305.14322) | 2023 | explicit read-write memory | 很适合支撑“记忆单元可写可读”的 primitive 抽象 | B |
| 7 | "My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents | 2024 | recall + consolidation | 对 write trigger / forgetting / consolidation 很有帮助 | B |
| 8 | Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-Term Conversations | 2024 | compressive conversational memory | 适合扩展 summary/compression/evolution families | B |
| 9 | Toward Conversational Agents with Context and Time Sensitive Long-term Memory | 2024 | temporal conversational retrieval | 适合补时间敏感 retrieval / event-indexing | B |
| 10 | Agent Workflow Memory | 2024 | workflow/procedural memory | 很适合把 workflow 当 memory unit 的路线做实 | A |
| 11 | HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model | 2024 | hierarchical working memory | 对 working memory slot 化很重要 | A |
| 12 | Memory OS of AI Agent | 2025 | short/mid/long hierarchical storage | 和 topology/store/check 机制高度契合 | A |
| 13 | [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) | 2025 | production memory extraction/retrieval | 工程与 benchmark 影响力都很高，适合作为现实系统对照 | A |
| 14 | A-MEM: Agentic Memory for LLM Agents | 2025 | graph-note + agentic evolution | 仓库已经有 A-MEM-like 路线，值得继续做 paper-faithful 对齐 | A |

### B. 图记忆、结构化记忆、层级检索（12 篇）

| # | 论文 | 年份 | 记忆主轴 | 与 MemPrimitive 的契合点 | 优先级 |
|---|---|---:|---|---|---|
| 15 | [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) | 2024 | graph-enhanced associative retrieval | 图式组织与 recall 路径都可拆成 primitives | A |
| 16 | From RAG to Memory: Non-Parametric Continual Learning for Large Language Models | 2025 | continual non-parametric memory / HippoRAG 2 | 适合把 retrieval 从“查库”推进到“记忆系统”叙事 | B |
| 17 | AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents | 2024 | episodic + semantic world-model graph | 图层、link evolution、graph retrieval 都很贴合 | A |
| 18 | [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) | 2025 | temporal KG memory | 对时间化 graph store 与历史关系管理很关键 | A |
| 19 | [Bridging Intuitive Associations and Deliberate Recall: Empowering LLM Personal Assistant with Graph-Structured Long-term Memory](https://aclanthology.org/2025.findings-acl.901/) | 2025 | event-centric graph recall | 非常适合补 event graph + multi-stage retrieval | B |
| 20 | From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory | 2025 | trainable graph memory | 适合压测 graph memory 与 strategy abstraction 边界 | B |
| 21 | [Hierarchical Memory Organization for Wikipedia Generation](https://aclanthology.org/2025.acl-long.1423/) | 2025 | hierarchical memory organization | 虽不是 agent memory 核心论文，但对 organization primitives 很有启发 | C |
| 22 | [Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning](https://arxiv.org/abs/2505.24478) | 2025 | graph-memory interface tuning | 更像 graph memory 系统工程压力测试对象 | C |
| 23 | [AI-native Memory 2.0: Second Me](https://arxiv.org/abs/2503.08102) | 2025 | layered personal memory + parametric personalization | 对 factual/profile/parametric 边界很重要 | B |
| 24 | O-Mem: Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents | 2025 | persona memory + event memory + self-evolving retrieval | 适合 profile/event 双轨 memory 的统一建模 | B |
| 25 | In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents | 2025 | prospective + retrospective reflection | 反思不只用于失败学习，也用于 retrieval refinement | B |
| 26 | Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science | 2025 | self-organizing episodes + predictive calibration | 对 unit formation / evolution trigger 非常有价值 | B |

### C. 系统化、分布式、自演化与多存储（8 篇）

| # | 论文 | 年份 | 记忆主轴 | 与 MemPrimitive 的契合点 | 优先级 |
|---|---|---:|---|---|---|
| 27 | [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2507.03724) | 2025 | plaintext / activation / parameter unified memory | 很适合做 forms-level 扩展边界参照 | B |
| 28 | [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/abs/2507.07957) | 2025 | multi-agent modular memory managers | 对 multi-agent shared memory 很关键 | B |
| 29 | SEDM: Scalable Self-Evolving Distributed Memory for Agents | 2025 | distributed, self-optimizing memory | 适合探索 verifiable write / consolidation controller | B |
| 30 | LightMem: Lightweight and Efficient Memory-Augmented Generation | 2025 | sensory/short-term/long-term staged memory | 对轻量三段式 memory pipeline 很有参考价值 | B |
| 31 | Towards LifeSpan Cognitive Systems | 2024 | lifelong experience absorption and response | 更像上位系统蓝图，但适合校准项目叙事 | C |
| 32 | [Human-inspired Episodic Memory for Infinite Context LLMs](https://openreview.net/forum?id=BI2int5SAC) | 2025 | event segmentation + episodic retrieval | 偏长上下文，但其 event segmentation 很适合 unit formation | C |
| 33 | MemVerse: Multimodal Memory for Lifelong Learning Agents | 2025 | hierarchical multimodal memory + periodic distillation | 对未来 multimodal + parametric bridge 很有用 | C |
| 34 | MGA: Memory-Driven GUI Agent for Observation-Centric Interaction | 2025 | structured external memory for GUI trajectories | 对 observation-centric memory 很贴合 | C |

### D. 多模态/具身/边界扩展（6 篇）

| # | 论文 | 年份 | 记忆主轴 | 与 MemPrimitive 的契合点 | 优先级 |
|---|---|---:|---|---|---|
| 35 | KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems | 2024 | embodied short/long memory cooperation | 适合测试具身任务上的 multi-store memory | C |
| 36 | VideoAgent: Long-form Video Understanding with Large Language Model as Agent | 2024 | iterative retrieval over long-form video | 更像 agentic retrieval baseline，但可作 multimodal memory 前身 | C |
| 37 | WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning | 2025 | episodic + semantic + visual memory | 很适合未来 multimodal memory store 分层 | C |
| 38 | Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory | 2025 | entity-centric multimodal long-term memory | 对 entity/graph/multimodal 融合很有启发 | C |
| 39 | HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding | 2025 | audiovisual episodic memory | 偏 multimodal benchmark，但机制拆解价值高 | C |
| 40 | Episodic Memory Representation for Long-form Video Understanding | 2025 | episodic event representation for video | 对 event-based memory unit 定义很有帮助 | C |

---

## 对这 40 篇的一个机制级粗分

如果按 `MemPrimitive` 的问题意识来分，我建议这样看：

### 1. 以 observation stream 为核心的 episodic memory

- Generative Agents
- MemoryBank
- My agent understands me better
- Compress to Impress
- Toward Conversational Agents with Context and Time Sensitive Long-term Memory
- O-Mem
- Nemori
- EM-LLM / Human-inspired Episodic Memory for Infinite Context LLMs

### 2. 以 reflection / feedback / self-improvement 为核心的 reflective memory

- Reflexion
- In Prospect and Retrospect
- Agent Workflow Memory
- LightMem
- SEDM

### 3. 以 graph / relation / event structure 为核心的 structured memory

- A-MEM
- HippoRAG
- HippoRAG 2
- AriGraph
- Zep / Graphiti
- Bridging Intuitive Associations and Deliberate Recall
- From Experience to Strategy

### 4. 以层级存储 / self-managed memory 为核心的 multi-store memory

- MemGPT
- Memory OS of AI Agent
- Mem0
- MemOS
- HiAgent
- KARMA

### 5. 以 persona / profile / personalization 为核心的 factual-semantic memory

- MemoryBank
- Second Me
- O-Mem
- RMM
- My agent understands me better

### 6. 以 multimodal / embodied 为核心的边界扩展

- KARMA
- VideoAgent
- MGA
- WorldMM
- MemVerse
- Seeing, Listening, Remembering, and Reasoning
- HippoMM
- Episodic Memory Representation for Long-form Video Understanding

---

## 对 MemPrimitive 的直接启发

### 已经基本覆盖得不错的方向

- Reflexion-like failure-triggered evolution
- graph-note memory
- graph retrieval / link evolution
- multi-layer topology-backed store
- baseline retrieval/readout composition

### 很可能还需要继续补的模块族

1. `skill / workflow memory`
   - Voyager、Agent Workflow Memory 会逼着我们把“可复用过程”当作 memory unit。
2. `temporal retrieval`
   - 时间约束、事件顺序、session-level recall 目前还不够强。
3. `profile / persona evolution`
   - O-Mem、Second Me、RMM 需要更明确的人物画像与事实更新语义。
4. `compression / consolidation`
   - Compress to Impress、LightMem、My agent understands me better 都会要求更正式的 consolidation family。
5. `working memory`
   - HiAgent 说明 working memory 最好不要只被当作 prompt 拼接问题。
6. `distributed / multi-agent memory`
   - MIRIX、SEDM 是后续 phase-2/3 的关键边界。
7. `multimodal memory`
   - 当前可以先不做主线，但需要在 ontology 上给出落点。

---

## 下一步我建议怎么推进

如果继续做第二轮筛选，我建议直接进入下面三步：

1. 先把这 40 篇再压缩成 12-15 篇“核心池”
   - 目标是覆盖机制异质性，而不是平均分布。
2. 对这 12-15 篇逐篇做 slot-level 拆解
   - 至少写出 `unit / representation / trigger / organization / evolution / retrieval / readout`
3. 再从中选 8-10 篇做 fully re-expressed 路线图
   - 这样会更贴近项目当前的中期目标

---

## 本轮主要参考

- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [A Survey on the Memory Mechanism of Large Language Model based Agents](https://arxiv.org/abs/2404.13501)
- [Awesome Agent Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory)

备注：

- 这份清单是第一轮“框架契合导向”的候选池，不是最终 bibliography。
- 2025-2026 的部分论文影响力仍在形成中，因此这里把“新近但高契合”的工作保留了较多。
