# 轻量型与算法型 Memory 代码复杂度调研

日期：2026-04-14

## 目标

这份文档继续前一轮平台型 memory 调研，但这次**有意把数据库/多后端/服务化难点放到次要位置**，重点回答三个问题：

1. 算法型 memory 代码的复杂度核心到底来自哪里。
2. `MemPrimitive` 对算法型工作的复杂度降低到底有多大。
3. 对轻量型 memory，作者自己随手写一段实现是否通常比引入 `MemPrimitive` 更高效。

## 样本与方法

本次调研主要使用本地 `.tmp_repo_survey/` 与 `.tmp_repo_survey_more/` 中的代码快照，样本包括：

- `reflexion`
- `recurrentgpt`
- `memory_sharing`
- `memgpt`（当前快照实际是 Letta / MemGPT 路线）
- `memllm`
- `hipporag`
- `lightmem`
- `comorag`
- `memochat`
- `memorybank-siliconfriend`
- `arigraph`
- `moom-roleplay-dialogue`
- `Think-in-Memory`（本轮未确认到官方公开代码；仅将 `.tmp_repo_survey_more/TiMSystem` 作为低置信辅助样本）

说明：

- 这里的行数统计只用于帮助形成“重量级直觉”，**不是严格 apples-to-apples 比较**。
- “轻量型 / 算法型”不是二元切分，中间还存在一类“控制器/维护型”系统。
- 本文同时参考了当前仓库里的 `MemPrimitive` baseline 与 classics 重表达代码，以判断框架真实吸收了多少复杂度，而不是只看 README 描述。

## 一页结论

### 先说最重要的判断

- **轻量型 memory** 的核心通常不难，往往就是 prompt 拼接、列表追加、简单相似度检索、少量格式解析。
- **算法型 memory** 的难点通常不在“多写几段 ingest/retrieve 胶水”，而在下面几类更深的复杂度：
  - 结构化 memory schema 与索引协同
  - 图构建 / OpenIE / PPR / 邻居扩展等检索算法
  - learned retriever / online update / 训练环
  - context compaction / summarization / queue-manager / maintenance scheduler
  - 外层 agent loop、工具调用与格式恢复
- `MemPrimitive` **最擅长吃掉的是“机制编排复杂度”**，不是“定制算法复杂度”。
- 新增样本把中间地带看得更清楚了：
  - `MemoChat` / `Memory Bank` 这类**摘要/角色画像型**系统，不算“随手 30 行写完”，但也还没到图算法那种重量级；这是 `MemPrimitive` 最可能真正帮上忙的一档。
  - `AriGraph` / `ComoRAG` 这类**图 + episodic + probe/planning** 系统说明，很多算法型复杂度来自控制环和检索策略本体，而不是存储接口。
  - `MOOM` 说明“方法复杂度”和“评测/服务层复杂度”经常混在一起；统一适配与评测层本身确实有现实需求，但不等于作者会愿意迁移到一个全量框架里。
- 所以：
  - 对**轻量型**方法，如果只是作者自己在一个项目里用一次，通常自己写更快。
  - 对**中等复杂度**方法，尤其是摘要/角色画像/轻维护型系统，如果目标是重表达、对照实验、统一 benchmark、共享 runtime，`MemPrimitive` 有明显价值。
  - 对**重算法型**方法，`MemPrimitive` 依然有价值，但更像“统一外壳 / 评测接口 / 机制坐标系”，而不是“把核心实现难度直接打掉”。

## 样本分层

| 工作 | 代表文件 | 观察到的主类型 | 复杂度主来源 |
| --- | --- | --- | --- |
| Reflexion | `.tmp_repo_survey/reflexion/webshop_runs/generate_reflections.py` | 轻量型 | prompt memory + append-only buffer |
| RecurrentGPT | `.tmp_repo_survey/recurrentgpt/recurrentgpt.py` | 轻量型偏中等 | 向量检索 + 外层 writer/human loop |
| Interactive Memory Sharing | `.tmp_repo_survey/memory_sharing/OnePool/Agent_MS.py` / `Plan Generation/Retrival_model.py` | 中等，带学习器 | 在线 retriever 训练与样本筛选 |
| MemGPT / Letta | `.tmp_repo_survey/memgpt/letta/agent.py` | 控制器/维护型 | memory pressure、summary compaction、消息队列管理 |
| MemLLM / RET-LLM | `.tmp_repo_survey/memllm/utils/memory/memory_server_v3.py` | 结构化算法型 | triple schema、entity/relation index、结构化查询 |
| HippoRAG | `.tmp_repo_survey/hipporag/src/hipporag/HippoRAG.py` | 图算法型 | OpenIE、graph augmentation、retrieval propagation |
| LightMem | `.tmp_repo_survey/lightmem/src/lightmem/memory/lightmem.py` | 混合型 | segmentation、buffer、offline update、retriever orchestration |
| MemoChat | `.tmp_repo_survey_more/MemoChat/code/codes/api/gpt_memochat.py` | 摘要/主题型中等系统 | topic summary 写入、LLM topic 选择、recent-dialog 阈值维护 |
| Memory Bank | `.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/memory_retrieval/forget_memory.py` | 摘要/人格型中等系统 | 分日期 summary、personality 汇总、FAISS 检索、forgetting curve |
| AriGraph | `.tmp_repo_survey_more/AriGraph/graphs/contriever_graph.py` | 图控制型算法系统 | triplet 抽取、过时事实删除、图检索、episodic recall、planning loop |
| ComoRAG | `.tmp_repo_survey_more/ComoRAG/src/comorag/ComoRAG.py` | 多通道图控制型算法系统 | VER/SEM/EPI 三通道检索、自探测、fusion memory、迭代控制环 |
| MOOM | `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/app_module.py` | 双分支维护型系统 | 多时间尺度 summary、角色画像 KV、importance/inhibition 遗忘、服务化对比层 |
| Think-in-Memory（低置信） | `.tmp_repo_survey_more/TiMSystem/TiMSystem.py` | 桶化 thought memory demo | LSH 分桶、thought recall、forget/merge 组织；但当前仅有社区 demo |

## 1. 算法型 memory 的复杂度核心来自哪里

### 1.1 结构化 memory schema 与索引协同

这是 `MemLLM / RET-LLM` 这类显式 memory 的核心难点。

从 `.tmp_repo_survey/memllm/utils/memory/memory_server_v3.py:41-59,88-100` 可以看到，上游实现不是“把三元组丢进一个列表里”这么简单，而是显式维护：

- triples / entities / relation_types 三张表
- 唯一约束与二级索引
- entity / relation type 的 HNSW 向量索引

再看 `.tmp_repo_survey/memllm/utils/memory/memory_controller_v3.py:159-197`，查询也不是普通相似度 top-k，而是：

- 先找实体候选
- 再找关系候选
- 再组合结构化 query
- 再回查 relationship

这里的复杂度来自：

- 数据表示是“结构化对象”，不是平面文本块
- 检索需要跨 schema 协同，而不是单一向量库召回
- 正确性依赖 query 解析、候选筛选、结构化回查的一致性

### 1.2 图构建与传播检索算法

这是 `HippoRAG` 的核心难点。

`.tmp_repo_survey/hipporag/src/hipporag/HippoRAG.py:220-279` 展示的不是普通 ingest，而是一整个 offline indexing 流程：

- OpenIE
- entity/fact embedding
- 图构建
- 事实边、passage 边、synonymy 边扩充
- 保存图快照

`.tmp_repo_survey/hipporag/src/hipporag/HippoRAG.py:1150-1188` 还显示 retrieval 前要专门准备 graph/embedding 对齐对象。

这里的复杂度不是“memory API 怎么写”，而是：

- 预处理管线本身复杂
- graph augmentation 是算法本体
- retrieval 依赖图拓扑而不是普通 memory lookup
- 运行时必须保证 embedding store 与 graph state 一致

### 1.3 控制器、压缩与维护调度

这是 `MemGPT / Letta` 和 `LightMem` 更核心的难点。

在 `.tmp_repo_survey/memgpt/letta/agent.py:944-1042,1107-1177` 里，重点不是“怎么写进 memory”，而是：

- 如何感知 memory pressure
- 何时发 warning
- 何时触发 summarizer
- 如何裁剪旧消息
- 如何把 summary 重新塞回上下文

这其实是“memory maintenance controller”问题，而不是简单存储问题。

`LightMem` 也类似。`.tmp_repo_survey/lightmem/src/lightmem/memory/lightmem.py:107-191,204-255` 说明它的复杂度主要来自：

- 消息标准化
- topic segmentation
- sensory / short-term buffer
- extraction 时机控制
- 在线 / 离线 update 路径
- 多类 retriever 组合

这里的核心难点是**维护策略和调度**，不是 append/retrieve 本身。

### 1.4 Learned retriever / online update

这是 `Interactive Memory Sharing` 的关键。

`.tmp_repo_survey/memory_sharing/OnePool/Agent_MS.py:184-224` 里，写入 memory 不是单纯 append，而是：

- 先打分
- 低分样本不进 memory
- 进 memory 前触发 retriever 训练
- 再落到 JSONL

`.tmp_repo_survey/memory_sharing/Plan Generation/Retrival_model.py:49-86,172-209` 里还可以看到：

- BM25 初筛
- LLM 打分
- BERT 相似度模型训练

这类复杂度的关键不是 memory store，而是**retriever 自身是被训练和更新的对象**。

### 1.5 外层 agent loop 与格式恢复

即使 memory 本体不重，外层 loop 也可能显著增复杂度。

例如 `RecurrentGPT` 上游核心文件 `.tmp_repo_survey/recurrentgpt/recurrentgpt.py:11-34,85-135` 中，memory 自身其实相对轻：

- `short_memory` 是字符串
- `long_memory` 是段落列表
- retrieval 是 embedding + cosine similarity
- update 是 append paragraph 再 encode

但真正烦人的地方在：

- writer prompt 很长
- 输出格式严格
- 需要解析多个 section
- 还要接 human simulator / plan-selection loop

这说明：**有些工作“看起来是 memory paper”，但实现难点一半在 outer loop，而不是 memory 内核。**

### 1.6 摘要/角色画像型系统的复杂度：不是很重，但也绝不只是 append

`MemoChat` 和 `Memory Bank` 把“轻量型 memory”与“重算法 memory”之间的中间地带暴露得更清楚。

`MemoChat` 的核心 memory 逻辑主要集中在 `.tmp_repo_survey_more/MemoChat/code/codes/api/gpt_memochat.py:73-130`：

- `run_summary(...)` 会把最近对话压缩成 `topic -> summary -> dialogs` 结构，写入 `memo`
- `run_retrieval(...)` 会把所有 topic/summary/dialog 展平，再让 LLM 从 topic 选项里挑相关主题
- 当 `Recent Dialogs` 超过阈值时才触发摘要，因此它已经有一个小型 maintenance policy

它的复杂度不是数据库或图，而是：

- memo schema 设计
- topic-level 摘要与反向检索 prompt
- recent-dialog window 与摘要更新时机
- 训练/指令数据与推理脚本强绑定

`Memory Bank` 则更进一步。`.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/summarize_memory.py:109-146`、`.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/build_memory_index.py:24-67` 与 `.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/memory_retrieval/forget_memory.py:63-146,232-360` 共同显示：

- 按日期维护 `history`、`summary`、`personality`
- 会生成 `overall_history` 与 `overall_personality`
- 检索侧使用 FAISS / embedding store
- 遗忘侧不是简单 FIFO，而是给每条记忆维护 `memory_strength`、`last_recall_date`，再按 forgetting curve 采样删除
- 被检索到的记忆会回写 strength 与 recall date

这一类系统的难点，主要来自：

- typed summary/personality state
- summary 与 raw history 的双轨维护
- recall 改变 memory state 这一“读写耦合”
- 轻量维护策略而非重图算法

所以它们不是 `HippoRAG` / `ComoRAG` 那种重算法系统，但也明显比 `Reflexion` 的 append-only buffer 更复杂。

### 1.7 图 + episodic + planning loop：AriGraph 的复杂度主要在控制闭环

`AriGraph` 的 `ContrieverGraph.update(...)` 在 `.tmp_repo_survey_more/AriGraph/graphs/contriever_graph.py:26-84` 已经把复杂度暴露得很彻底：

- 从 observation 中抽 triplet
- 根据关联 triplet 让模型预测哪些旧 triplet 过时，并执行删除
- 为 triplet 和 item 建 embedding
- 基于 query + depth 做图检索
- 再基于 plan embedding 做 episodic memory 检索

而 `.tmp_repo_survey_more/AriGraph/pipeline_arigraph.py:67-115` 说明它不是一个“先 retrieve 再 answer”的静态系统，而是：

- 每一步都更新图与 episodic memory
- 再把 subgraph + episodic memory 送入 planning
- 再据此选择动作
- 再和环境交互，进入下一轮

这里的复杂度核心是：

- graph state 的在线维护
- outdated-fact deletion
- semantic graph recall 与 episodic recall 的并行协同
- memory module 深度嵌入到 planning / acting loop

这类复杂度，`MemPrimitive` 很难通过统一 `ingest/retrieve` 直接消掉。

### 1.8 多通道 memory pool、自探测与 fusion：ComoRAG 的复杂度比图检索还多一层

`ComoRAG` 比 `AriGraph` 更进一步。`.tmp_repo_survey_more/ComoRAG/src/comorag/ComoRAG.py:265-392` 展示的是一个显式的 meta-control loop：

- 初始先做三通道检索
- 再把检索结果编码进临时 memory pool
- 若当前 answer 仍然是 `"*"`, 则把临时 pool 合并进主 pool
- 让 probe agent 继续提新 probe
- 再检索、再 memory encoding、再 fusion
- 直到得到最终 answer 或达到迭代上限

`.tmp_repo_survey_more/ComoRAG/src/comorag/utils/memory_utils.py:72-302` 则显示它的 memory 结构也不是普通 list，而是：

- `VER / SEM / EPI / FUSION` 四类节点
- 主 pool + temp pool 双池语义
- probe + cue 的联合 embedding
- 历史节点相似检索
- 基于 retrieved nodes 的 fusion node 生成

所以 `ComoRAG` 的复杂度来源至少有四层：

- 图 / summary / timeline 三通道 retrieval
- 自探测 probe 策略
- 双 memory pool 组织语义
- fusion memory 的生成与回流

这已经非常接近“memory-as-controller”而不只是“memory-as-storage”。

### 1.9 双分支记忆、typed merge 与抑制式遗忘：MOOM 的复杂度部分在算法，部分在评测/服务层

`MOOM` 是这次新增样本里最值得小心解读的一个。

先看方法侧：`.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/README.md:7` 明确把它描述为双分支 memory plugin：

- 一支做多时间尺度 plot conflict summary
- 一支做 user character profile extraction
- 再叠加 competition-inhibition 风格的 forgetting mechanism

代码里这个结构也能看到：

- `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/app_module.py:161-234` 的 `MemoryManager` 同时管理 KV memory、summary memory、importance pool、以及 `mem0` / `memorybank` / `memochat` 等外部对比路径
- `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/app_module.py:449-508` 显示 retrieval 之后还要更新 importance、执行 inhibition、按容量裁剪
- `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/utils/memory_combine_new.py:67-264` 里有 typed merge / conflict resolution 逻辑，不同 key 走 replace / add / trajectory update / LLM fusion 不同路径
- `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/utils/forget_mechanism.py:440-487` 则把 forgetting 具体化成 score 更新、抑制和 top-n 过滤

但同时，`MOOM` 还暴露了另一件很重要的事：

- `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/main.py:104-165` 同时暴露了 `/rag-memory/`、`/rag-summary/`、`/rag-mem0/`、`/rag-memorybank/`、`/rag-memochat/` 等接口

这说明它的 repo 复杂度**一部分来自方法本身，一部分来自把多种 memory 方法拉到同一服务/评测壳里**。

这对你的项目判断很关键：

- 它支持“统一 adapter / benchmark 层有真实需求”
- 但不支持“所有方法作者都会愿意把自己的 memory 全量迁进统一框架”

### 1.10 Think-in-Memory：当前只能作为低置信辅助样本

本轮公开检索没有确认到官方 `Think-in-Memory` 代码仓库；本地 `.tmp_repo_survey_more/TiMSystem/` 更像是一个社区演示版，而不是正式实现。

这个 demo 的 `.tmp_repo_survey_more/TiMSystem/TiMSystem.py:173-352` 仍然有一点参考价值：

- 用 LSH 风格 hash 把 thought 分桶
- recall 时先找桶，再做桶内简单相似检索
- 组织阶段支持 forget / merge
- 新 thought 在 response 之后通过 post-thinking 写回

它和论文记忆机制的方向大体一致，但因为不是官方实现，所以**只能作为辅助证据，不能与前面那些官方 repo 同权重使用**。

## 2. `MemPrimitive` 对算法型工作的复杂度降低有多大

### 2.1 `MemPrimitive` 已经能吸收的复杂度

从当前代码看，`MemPrimitive` 已经覆盖了不少“机制层”能力：

- `TripleRepresentation` 与 `TripleMemoryRetrieval`
  - `memprimitive/baselines/representation.py`
  - `memprimitive/baselines/retrieval.py:1083-1165`
- `VectorGraphSeedAndExpandRetrieval`
  - `memprimitive/baselines/retrieval.py:1904-2022`
- `LLMFunctionCallEvolution`
  - `memprimitive/baselines/memory_evolution.py:756-966`
- `HierarchicalEvolution`
- 各类 `EmbeddingSimilarityRetrieval` / `BufferRetrieval` / `GraphNeighbor...`

也就是说，框架已经能把很多工作统一拆成：

- representation
- write organization
- evolution / maintenance
- retrieval
- readout

这对**机制重表达、消融实验、统一 benchmark 接口**非常有帮助。

### 2.2 但框架吸收不了哪些复杂度

它吸收不了下面这些：

- OpenIE、PPR、graph augmentation 这类算法本体
- learned retriever 的训练与增量更新
- queue manager / memory pressure / summarizer scheduler
- 训练驱动的工具调用控制器
- 严格贴合某篇论文的外层 agent loop
- 后端实体表 / 关系表 / ANN index 的工程语义

换句话说，`MemPrimitive` 主要吃掉的是**编排复杂度**，不是**定制算法复杂度**。

### 2.3 分工作判断

| 工作 | `MemPrimitive` 帮助度 | 原因 |
| --- | --- | --- |
| Reflexion | 低到中 | 机制很简单，框架主要提供统一表示和 benchmark 接口；对一篇单独工作本身降本不大 |
| RecurrentGPT | 低到中 | 可复用向量存储/召回，但 writer/human loop、格式解析仍然大量保留 |
| Memory Sharing | 中 | store/retrieve 轮廓能被框架吸收，但 retriever 训练环仍然是核心 bespoke 复杂度 |
| MemGPT / Letta | 中（memory-only）/ 低（全系统） | core/recall/summary 可以抽象，但 queue-manager 和 memory-pressure 控制仍是主难点 |
| MemLLM / RET-LLM | 中 | triple extraction / structured retrieval 外壳可复用，但显式 schema 与 controller 复杂度仍在 |
| HippoRAG | 低 | OpenIE、图增广、propagation 检索才是本体，框架只能接壳，不能消掉核心难点 |
| LightMem | 低到中 | buffer / segmentation / update 策略仍重；框架对统一接口和 benchmark 更有帮助 |
| MemoChat | 中 | 主题摘要写入、topic 检索、recent-dialog maintenance 都能被 pipeline 吸收不少，但训练/提示词设计仍然重要 |
| Memory Bank | 中到高（memory slice） | history/summary/personality/forgetting-curve 这些机制与框架槽位比较契合，是新增样本里最像“可被系统化吸收”的一类 |
| AriGraph | 低 | graph update、过时 triplet 删除、episodic recall 与 planning loop 是主复杂度，远超统一存储接口 |
| ComoRAG | 低 | tri-channel retrieval、自探测 probe、temp/main pool、fusion memory 是核心算法与控制逻辑 |
| MOOM | 中 | 双分支 memory、typed merge、importance/inhibition 可以部分被统一，但 repo 复杂度还混入了服务层与多基线对比壳 |
| Think-in-Memory（仅社区 demo） | 低置信：中 | bucketed recall / forget / merge 的 memory slice 能映射到框架，但当前缺少官方代码证据，不宜高估 |

### 2.4 一个关键观察：框架最有希望吃掉的是“中间地带”的复杂度

新增样本让这个判断更清楚：

- 对 `MemoChat` / `Memory Bank` 这类系统，复杂度已经不止是几十行 append/retrieve，但又还没有重到必须自带图算法和控制器。
- 这一档系统里，`MemPrimitive` 最有可能真正降低总成本，因为：
  - 写入、组织、维护、召回、readout 的边界已经相对清楚
  - novelty 往往在“机制组合”而不是超定制算法
  - 很多逻辑本来就在围绕 summary / profile / retrieval / forgetting 这些框架槽位旋转

也就是说，`MemPrimitive` 的最佳用武之地，未必是最轻的方法，也未必是最重的方法，而是**中等复杂度、机制可分解、需要反复对照实验的方法**。

### 2.5 一个关键观察：框架会把复杂度“换位置”，不一定“消灭复杂度”

从当前 classics 代码也能看出这一点。

比如 `RET-LLM` 的重表达文件 `memprimitive/example/classics/ret_llm_memory.py:315-505` 中：

- triple ingest/retrieval 复用了框架 primitive
- 但仍然需要自定义 structured query parsing
- 仍然需要 fallback term resolution
- 仍然需要 outer answer loop 与 `MEM_READ` 工具桥接

这说明对于算法型工作，框架的价值更像：

- 把“通用 memory 管线”沉到底座
- 让真正定制的算法部分更集中、更显式

而不是把所有代码都变短。

## 3. 对轻量型 memory，作者自己写会不会通常更高效

### 3.1 大多数时候，答案是“会”

如果作者只想在自己的系统里加一个轻量型 memory，通常自己写会更快。

最直观的证据是上游 memory-core 代码和当前 classics 示例的量级对比：

| 工作 | 上游核心文件行数 | `MemPrimitive` 对应示例行数 | 直觉 |
| --- | --- | --- | --- |
| Reflexion | `38 + 43 = 81` | `286` | 上游非常轻；引入框架后抽象成本明显增加 |
| RecurrentGPT | `106` | `707` | memory 内核不重，但重表达需要大量 orchestration / parsing |
| Memory Sharing | `234 + 177 = 411` | `377` | 这是例外：框架对机制切分有一定吸收作用 |
| MemLLM / RET-LLM | `337 + 159 = 496` | `576` | 量级相近，说明框架并没有让算法型复杂度凭空消失 |

这些数字只用于帮助形成直觉，但已经很说明问题：

- **对非常轻的 memory**，框架常常比原实现更重。
- **对中等复杂度 memory**，框架可能把复杂度压平。
- **对重算法 memory**，框架通常只能把复杂度重新组织，而不是大幅减少。

### 3.2 为什么轻量型方法经常不值得引入重框架

轻量型 memory 的上游实现通常只有这些动作：

- 拼 prompt
- append 一条文本
- top-k 相似度检索
- 把结果重新塞回 prompt

这类事情单独实现往往只要几十到一两百行，学习和调试成本都很低。

而引入 `MemPrimitive` 的额外成本包括：

- 学习 pipeline slot 与 store topology
- 组装 representation / organization / retrieval / readout
- 适配 runtime / embedding / prompt-plan
- 理解 trace / metadata contract

所以如果目标只是“我需要一个能跑的轻量 memory”，很多作者自己写确实更高效。

### 3.3 什么时候轻量型方法反而适合放进 `MemPrimitive`

只有在下面这些目标出现时，框架开销才更可能值得：

- 不是做一个 memory，而是要比较多个 memory
- 需要统一 benchmark pipeline
- 需要和别的方法共享 store / retrieval / readout
- 需要做 slot-level ablation
- 需要把论文方法重表达成统一机制语言
- 未来想把方法纳入 search space / motif 分析

也就是说，`MemPrimitive` 对轻量型 memory 的优势，更多是**研究组织方式优势**，而不是**单次开发效率优势**。

### 3.4 `MemoChat` / `Memory Bank` 是不是例外

新增样本说明，前面那句“轻量型作者自己写通常更快”需要再精确一点：

- 对 `Reflexion` 这种非常轻的 append/prompt memory，这个判断基本稳固成立。
- 但对 `MemoChat` / `Memory Bank` 这种**中轻量但已经带 maintenance 的系统**，情况开始接近 break-even：
  - 如果作者只是做一个一次性原型，自己写往往还是更快。
  - 但如果作者已经需要 topic summary、personality memory、forgetting、统一评测、以及多方法对照，那么框架开始有机会真正省事。

换句话说：

- “自己写更快”主要对**非常轻**的 memory 成立。
- 到了 `MemoChat` / `Memory Bank` 这一档，框架不一定马上赢，但已经不是显然输。

## 4. 对项目定位的含义

### 4.1 不应该把项目主要卖点放在这里

不建议把 `MemPrimitive` 的核心叙事放成：

> “如果你要做轻量型 memory，用我的框架会比自己写更省事。”

从这轮代码调研看，这句话大多数时候并不成立。

### 4.2 更合理的定位

更强的叙事应该是：

1. **统一机制表示层**
   - 把不同 memory paper 重写到同一 primitive 语言中
2. **统一 benchmark / evaluation adapter**
   - 即使作者不想把 memory 全部改写到框架里，也可以只接入 `ingest/retrieve` 或更薄的评测接口
3. **中等复杂度方法的 reusable runtime**
   - 尤其是那些 novelty 在机制组合，而不是在 bespoke 算法内核里的工作
4. **重算法方法的外壳与对照实验底座**
   - 对 HippoRAG、MemLLM 这类方法，框架更多提供统一边界，而不是替代算法本体

### 4.3 MOOM 给出的额外启发

`MOOM` 的 repo 结构本身就在说明一件事：就算单个 memory 方法未必愿意迁移到统一框架，**统一服务层 / adapter 层 / benchmark 层** 仍然可能有真实价值。

因为在 `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/main.py:104-165` 里，作者实际上已经把：

- 自己的方法
- `mem0`
- `memorybank`
- `memochat`

一起放进了同一套 API 与实验壳里。

这对 `MemPrimitive` 的启发不是“去做一个更重的平台”，而是：

- 优先把**统一接入与统一评测层**做薄、做稳
- 不要一开始假设别人会愿意直接采用整套内部抽象

### 4.4 一个很实际的产品建议

如果你希望未来有人把轻量型 memory 接到你的 benchmark 里，最值得做的不是强推全框架依赖，而是提供一层**超薄适配面**：

- 一个最小 `ingest(...) / retrieve(...)` 协议
- 或者几个开箱即用的轻量 preset
  - `SimpleReflectionMemory`
  - `SimpleSemanticMemory`
  - `SimpleSessionSummaryMemory`

这样：

- 轻量型作者不用先理解整个 primitive 体系
- 仍然能进入你的评测 pipeline
- 真正需要机制分析的人，再往 `MemPrimitive` 全量接口升级

## 5. 最终判断

### 对问题 1

算法型 memory 的核心复杂度主要来自：

- 结构化 schema 与索引协同
- 图构建与传播检索
- learned retriever / online training
- compaction / scheduler / controller
- 外层 agent loop 与严格格式控制

而不是简单的 `ingest/retrieve` API 本身。

### 对问题 2

`MemPrimitive` 对算法型工作的帮助是**分层的且有边界的**：

- 对 `MemoChat` / `Memory Bank` / `MOOM` 这类中等复杂度、维护规则相对显式的方法，帮助可以到中甚至中高
- 对机制编排、统一重表达、benchmark 接口帮助明显
- 对 bespoke 算法本体、训练环、图算法、probe/planning 控制器帮助有限

### 对问题 3

对轻量型 memory，如果只是作者自己做一个系统里的单次实现，**自己写通常更高效**；但新增样本说明，这句话主要适用于非常轻的 `Reflexion` 式方法，对 `MemoChat` / `Memory Bank` 这类中轻量维护型系统则开始接近 break-even。`MemPrimitive` 的优势更多体现在：

- 统一表示
- 对照实验
- benchmark 接入
- 长期研究组织

而不是一次性开发速度。

## 附：本次调研用到的关键代码锚点

- Reflexion
  - `.tmp_repo_survey/reflexion/webshop_runs/generate_reflections.py:12-45`
  - `.tmp_repo_survey/reflexion/webshop_runs/env_history.py:42-50`
- RecurrentGPT
  - `.tmp_repo_survey/recurrentgpt/recurrentgpt.py:11-34`
  - `.tmp_repo_survey/recurrentgpt/recurrentgpt.py:85-135`
- Interactive Memory Sharing
  - `.tmp_repo_survey/memory_sharing/OnePool/Agent_MS.py:184-224`
  - `.tmp_repo_survey/memory_sharing/Plan Generation/Retrival_model.py:49-86`
  - `.tmp_repo_survey/memory_sharing/Plan Generation/Retrival_model.py:172-209`
- MemGPT / Letta snapshot
  - `.tmp_repo_survey/memgpt/letta/agent.py:154-157`
  - `.tmp_repo_survey/memgpt/letta/agent.py:944-1042`
  - `.tmp_repo_survey/memgpt/letta/agent.py:1107-1177`
- MemLLM / RET-LLM
  - `.tmp_repo_survey/memllm/utils/memory/memory_server_v3.py:41-59`
  - `.tmp_repo_survey/memllm/utils/memory/memory_server_v3.py:88-100`
  - `.tmp_repo_survey/memllm/utils/memory/memory_controller_v3.py:159-197`
- HippoRAG
  - `.tmp_repo_survey/hipporag/src/hipporag/HippoRAG.py:220-279`
  - `.tmp_repo_survey/hipporag/src/hipporag/HippoRAG.py:1150-1188`
- LightMem
  - `.tmp_repo_survey/lightmem/src/lightmem/memory/lightmem.py:107-191`
  - `.tmp_repo_survey/lightmem/src/lightmem/memory/lightmem.py:204-255`
- MemoChat
  - `.tmp_repo_survey_more/MemoChat/code/codes/api/gpt_memochat.py:73-130`
- Memory Bank
  - `.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/summarize_memory.py:109-146`
  - `.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/build_memory_index.py:24-67`
  - `.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/memory_retrieval/forget_memory.py:63-146`
  - `.tmp_repo_survey_more/MemoryBank-SiliconFriend/memory_bank/memory_retrieval/forget_memory.py:232-360`
- AriGraph
  - `.tmp_repo_survey_more/AriGraph/graphs/contriever_graph.py:26-84`
  - `.tmp_repo_survey_more/AriGraph/pipeline_arigraph.py:67-115`
- ComoRAG
  - `.tmp_repo_survey_more/ComoRAG/src/comorag/ComoRAG.py:265-392`
  - `.tmp_repo_survey_more/ComoRAG/src/comorag/ComoRAG.py:456-554`
  - `.tmp_repo_survey_more/ComoRAG/src/comorag/utils/memory_utils.py:72-302`
- MOOM
  - `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/README.md:7`
  - `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/main.py:104-165`
  - `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/app_module.py:161-234`
  - `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/app_module.py:449-508`
  - `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/utils/memory_combine_new.py:67-264`
  - `.tmp_repo_survey_more/MOOM-Roleplay-Dialogue/utils/forget_mechanism.py:440-487`
- Think-in-Memory（社区 demo，低置信）
  - `.tmp_repo_survey_more/TiMSystem/README.md:5-17`
  - `.tmp_repo_survey_more/TiMSystem/README.md:81-83`
  - `.tmp_repo_survey_more/TiMSystem/TiMSystem.py:173-352`
- 当前 `MemPrimitive` 底座
  - `memprimitive/baselines/retrieval.py:1083-1165`
  - `memprimitive/baselines/retrieval.py:1904-2022`
  - `memprimitive/baselines/memory_evolution.py:756-966`
  - `memprimitive/example/classics/reflexion_memory.py:89-143`
  - `memprimitive/example/classics/recurrentgpt_memory.py`
  - `memprimitive/example/classics/ret_llm_memory.py:315-505`

## 附：本地快照对应的上游仓库

- Reflexion: `https://github.com/noahshinn/reflexion`
- RecurrentGPT: `https://github.com/aiwaves-cn/RecurrentGPT`
- Interactive Memory Sharing: `https://github.com/GHupppp/InteractiveMemorySharingLLM`
- MemGPT: `https://github.com/cpacker/MemGPT`
- Letta: `https://github.com/letta-ai/letta`
- MemLLM: `https://github.com/amodaresi/MemLLM`
- HippoRAG: `https://github.com/OSU-NLP-Group/HippoRAG`
- LightMem: `https://github.com/zjunlp/LightMem`
- ComoRAG: `https://github.com/EternityJune25/ComoRAG`
- MemoChat: `https://github.com/LuJunru/MemoChat`
- Memory Bank: `https://github.com/zhongwanjun/MemoryBank-SiliconFriend`
- AriGraph: `https://github.com/AIRI-Institute/AriGraph`
- MOOM: `https://github.com/cows21/MOOM-Roleplay-Dialogue`
- Think-in-Memory（本轮未确认到官方公开代码，以下仅为社区 demo）: `https://github.com/tractorjuice/TiMSystem`
