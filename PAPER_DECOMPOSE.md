> Note
> This file is a paper-to-primitive analysis note.
> Richer trigger names mentioned here are literature-mapping labels, not the current public baseline trigger API.

## HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

论文链接: <https://arxiv.org/abs/2405.14831>

官方 repo: <https://github.com/OSU-NLP-Group/HippoRAG>

### 论文侧 memory 机制速写

HippoRAG 把外部文档记忆建模成一个受 hippocampal indexing theory 启发的长期记忆系统。其 memory 核心不是“把 passage 向量化后直接查库”，而是先把 passage 处理成开放式知识图谱，再在查询时把 query 中的关键实体链接到图节点，随后用 Personalized PageRank 在图上做一次受查询偏置的扩散，最后把节点激活分数聚合回 passage 排序。

从 MemPrimitive slot 角度看，它的主链路可以概括为：

- `unit_formation`: 以 passage 为基本写入单元。
- `representation`: 对每个 passage 先做 named entity extraction，再做 NER-conditioned OpenIE triple extraction，并为 noun phrase / entity 节点建立 embedding。
- `write_trigger`: 默认离线全量写入。
- `organization`: 把 passage、noun phrase / entity、triple 关系组织成可检索的图式索引，并保留 node-to-passage 归属关系。
- `evolution_trigger`: 论文主体没有单独强调在线演化触发；索引构建阶段的图增强更像写入时/写后处理。
- `memory_evolution`: 增补 similarity/synonymy edges，使相近但不完全相同的 noun phrases 互联。
- `retrieval`: query 先抽 named entities，再链接到 KG 节点，做 node specificity 加权和 PPR 扩散，最后把节点分数聚合成 passage 分数。
- `readout`: 输出 top-k passages 给下游 reader/QA。

### 按 MemPrimitive slot 的拆解

#### unit_formation

- 论文里做了什么
  - 基本以 retrieval corpus 中的单个 passage 为索引单位，图谱构建是逐 passage 执行 OpenIE。
- MemPrimitive 现有哪些模块可直接复用
  - `PassThroughUnitFormation`：如果上游已经把输入准备为单 passage observation，可直接表达“一条输入对应一个写入单元”。
- 哪些模块只能部分复用
  - `SentenceSplitUnitFormation`、`LineSplitUnitFormation`、`WindowedUnitFormation`：它们能做切分，但 HippoRAG 论文核心并不依赖这些切分策略。
- 当前缺失什么能力
  - 无关键缺口。该 slot 不是 HippoRAG 的机制瓶颈。
- 你的判断依据是什么
  - 论文明说：offline indexing 是对 retrieval corpus 逐 passage 处理并抽取 triples。
  - repo 实现可确认：`HippoRAG.index(docs)` 接收 `List[str]` 文档/段落并逐条做 OpenIE。
  - 推断：在 MemPrimitive 中把每个 passage 视为一个 `Observation`/`MemoryUnit` 已足够承载该 slot。

#### representation

- 论文里做了什么
  - 对每个 passage 先抽 named entities，再把 named entities 放回 prompt 里做 NER-conditioned triple extraction。
  - 同时为 entity / noun phrase 建 embedding，用于后续 query linking 和 synonymy edge 构造。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。当前没有一个现有 representation 能一次性产出“query-independent 的 passage OpenIE 图节点 + 图边 + 专用于节点链接的 embedding 载体”。
- 哪些模块只能部分复用
  - `BasicRepresentation(elements=("text", "embedding", "entities", "triple"))`
    - 能覆盖“抽实体、抽三元组、做 embedding”这一外观，但当前语义更像把结果附着在单个 `MemoryUnit` 上，不等于 HippoRAG 所需的 NER-conditioned OpenIE 索引表示。
  - `KeywordRepresentation`
    - 只能覆盖 lexical side information，与论文核心表示不足够匹配。
- 当前缺失什么能力
  - 缺少“先 NER、再用 NER 约束 triple extraction”的两阶段表示模块。
  - 缺少把 noun phrase / entity 作为图节点级对象稳定输出的表示边界。
  - 缺少“passage-node incidence 信息”在表示阶段的明确产物。
  - 缺少 query 侧 named entity extraction 的对称机制；当前 recall 路径没有单独的 query representation slot，只能落到 retrieval 内部实现。
- 你的判断依据是什么
  - 论文明说：先 extract named entities，再把 named entities 加入 OpenIE prompt 以抽 triples。
  - repo 实现可确认：`prompts/templates/ner.py` 与 `prompts/templates/triple_extraction.py` 显示确有 NER 与 NER-conditioned triple extraction 两阶段 prompt；`openie_vllm_offline.py` 也按该顺序批量执行。
  - 推断：MemPrimitive 当前 `BasicRepresentation` 虽然有 `entities` 和 `triple`，但没有稳定暴露“图节点对象化 + 后续链接专用表示”的能力边界，因此只能部分复用。

#### write_trigger

- 论文里做了什么
  - 论文主体把 indexing 视为离线全量构建过程，没有复杂的 selective write 机制；新 passage 通常直接纳入索引。
- MemPrimitive 现有哪些模块可直接复用
  - `AlwaysWriteTrigger`：可直接表达“所有 passage 均进入索引”。
- 哪些模块只能部分复用
  - `ThresholdWriteTrigger`：技术上能用，但不是论文核心机制。
- 当前缺失什么能力
  - 无关键缺口。HippoRAG 的创新点不在 write gating。
- 你的判断依据是什么
  - 论文明说：offline indexing 处理整个 retrieval corpus。
  - repo 实现可确认：`index(docs)` 对输入 docs 批量执行 OpenIE 和图构建，没有单独的 selective write 判定阶段。
  - 推断：该 slot 可直接用最简单的全写入表达。

#### organization

- 论文里做了什么
  - 组织成一个开放式 KG / hippocampal index。
  - 图里至少包含 noun phrase / entity 层面的节点、triple 诱导的关系边，以及 node-to-passage 的归属统计；论文还显式提到一个记录 noun phrase 在各 passage 中出现次数的矩阵，用于把节点激活聚合回 passage。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `GraphAppendOrganization`
    - 能把记录放进 graph layer，并写一些 graph metadata；现在也能承接 note graph/A-MEM 风格写入。
    - 但它的“图”仍然是 record-centric metadata graph，不是 HippoRAG 那种显式 node/edge/index 结构，也不是 passage-entity-triple 异构索引。
- 当前缺失什么能力
  - 缺少异构图组织能力：需要同时表示 passage 节点、entity/noun phrase 节点、relation/fact 边，且这些对象不是同一种普通 `MemoryRecord` 就能自然表达。
  - 缺少 node-to-passage incidence matrix 或等价统计结构。
  - 缺少“图节点级 embedding store”与 record store 的明确分离。
  - 缺少 synonymy edge 将加入何处、如何与原始 triple edges 并存的正式组织边界。
- 你的判断依据是什么
  - 论文明说：HippoRAG builds an open KG and defines a matrix containing the number of times each noun phrase appears in each original passage。
  - repo 实现可确认：`EmbeddingStore` 分别维护 chunk/entity/fact 三套存储；`HippoRAG.index()` 明确构建图并单独维护实体、事实、passage 的 embedding 与图连接。
  - 推断：MemPrimitive 现有 graph organization 只够表达“记录之间有图链接”，不足以直接表达 HippoRAG 的异构图索引。

#### evolution_trigger

- 论文里做了什么
  - 论文没有把“何时做额外演化”单独当成一个在线控制问题来讲；同义/相似节点连边更像索引构建流水线中的固定步骤。
- MemPrimitive 现有哪些模块可直接复用
  - `NeverEvolutionTrigger`：如果把图增强视为 organization 内部完成，则可以直接不使用额外 evolution trigger。
- 哪些模块只能部分复用
  - `NeighborExistsEvolutionTrigger`
    - 语义上像“已有图邻居时再做额外图维护”，但 HippoRAG 的 synonymy edge augmentation 不是由现成邻居存在触发，而是由 embedding similarity 规则触发。
  - `ThresholdEvolutionTrigger`
    - 只能充当占位控制，不是论文语义。
- 当前缺失什么能力
  - 若坚持把 synonymy augmentation 放入 `memory_evolution` slot，则缺少一个“新 passage / 新节点写入后，对候选 entity 节点做 similarity linking”的专用触发器。
- 你的判断依据是什么
  - 论文明说：synonymy relations 是 indexing process 的组成部分。
  - repo 实现可确认：`index()` 在图构建后调用 `add_synonymy_edges()`，并非单独的在线检索期触发。
  - 推断：该 slot 在 HippoRAG 中较弱，但若要在 MemPrimitive 内干净落位，最好补一个 graph augmentation trigger。

#### memory_evolution

- 论文里做了什么
  - 在基础 OpenIE 图之上，用 retrieval encoder 为相似但不完全相同的 noun phrases/entity 增补 synonymy/similarity edges，帮助 pattern completion。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `GraphLinkEvolution`
    - 能在 graph layer 里补 link，但当前假设是 record-to-record 邻接，不是 entity-node embedding 相似驱动的节点级 augmentation。
  - `GraphNeighborAppendEvolution`
    - 只是 `GraphLinkEvolution` 的兼容包装，能力边界相同。
  - `LinkStrengtheningEvolution`
    - 能做图链接强化，但设计前提是 note graph + LLM judge，不是 HippoRAG 的 retrieval-encoder synonymy edge。
- 当前缺失什么能力
  - 缺少“基于节点 embedding 相似度阈值批量建立 synonymy edges”的显式演化模块。
  - 缺少对异构图节点级别而非 record 级别的边写回。
  - 缺少把 node specificity 所需的 local support statistics 作为图演化副产物持久化。
- 你的判断依据是什么
  - 论文明说：额外边来自 retrieval encoders，在 entity representations cosine similarity 超过阈值时加入。
  - repo 实现可确认：`index()` 在 `add_fact_edges` / `add_passage_edges` 之后调用 `add_synonymy_edges()`。
  - 推断：当前 MemPrimitive 图演化家族更接近 record graph 或 note graph，不足以无新增模块表达 HippoRAG 的 synonymy augmentation。

#### retrieval

- 论文里做了什么
  - query 先抽 named entities。
  - 把 query named entities 链接到 KG 节点，形成 query nodes。
  - 用 query nodes 作为 Personalized PageRank 种子，并做 node specificity 加权。
  - 将 PPR 后的节点概率聚合到 passage 分数，输出 top passages。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `ExpandRetrievedGraphNeighbors`
    - 覆盖了“从 seed records 做一跳图扩散”的粗轮廓，但不负责 query entity linking、query-side seed weighting 或 PPR。
  - `VectorGraphSeedAndExpandRetrieval`
    - 覆盖了“vector seed + graph expansion”的粗轮廓，但服务对象是 enriched note graph，不是 HippoRAG 的 noun phrase/entity KG。
  - `EmbeddingSimilarityRetrieval`
    - 可复用其中的 embedding 相似度计算直觉，但它只做向量检索，不做图上传播和 passage 聚合。
  - `LayerAwareRetrieval`
    - 只是在层间分发结果，不触及 HippoRAG 的核心检索算法。
- 当前缺失什么能力
  - 缺少 query named entity extraction + entity-to-node linking 的 retrieval 前半段。
  - 缺少 Personalized PageRank 检索器。
  - 缺少 node specificity / local support statistics 加权。
  - 缺少 node-to-passage incidence 聚合，把节点激活分数映射回 passage 排序。
  - 缺少与原论文一致的“单步多跳”图扩散检索语义。
- 你的判断依据是什么
  - 论文明说：query named entities -> query nodes -> PPR -> multiply by passage incidence matrix 得 passage ranking。
  - repo 实现可确认：`HippoRAG.retrieve()` 与 `run_ppr()` 表明 repo 确实有 PPR 检索主干；同时当前主分支还引入了 fact scoring、DSPy rerank、passage node weight 等更偏 HippoRAG 2 的控制逻辑。
  - 依据论文与 repo 做出的合理推断：即使不照搬 repo 的新细节，MemPrimitive 仍缺少 original HippoRAG 所需的 PPR retrieval primitive。

#### readout

- 论文里做了什么
  - 把 top-ranked passages 返回给 reader/QA 模块；readout 本身不是论文创新点。
- MemPrimitive 现有哪些模块可直接复用
  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
  - 这些都足以把检索出的 passages 线性整理给下游。
- 哪些模块只能部分复用
  - `GraphReadout`
    - 适合调试图结构，但不是 HippoRAG 论文默认的 passage readout。
- 当前缺失什么能力
  - 无关键缺口。只要 retrieval 已经产出正确的 passage ranking，readout 用现有模块即可。
- 你的判断依据是什么
  - 论文明说：PPR 最终用于 rank passages for retrieval。
  - repo 实现可确认：`retrieve()` 返回 top_k docs 文本列表。
  - 推断：HippoRAG 的 readout 可以由 MemPrimitive 现有通用文本 readout 直接承担。

### 与 MemPrimitive 现有组件的对照结论

| slot | 结论 | 说明 |
| --- | --- | --- |
| `unit_formation` | 直接复用 | passage 级写入单元可由 `PassThroughUnitFormation` 直接表达 |
| `representation` | 部分复用 | 有 entities/triples/embedding，但缺 NER-conditioned OpenIE 与节点级表示边界 |
| `write_trigger` | 直接复用 | `AlwaysWriteTrigger` 足够 |
| `organization` | 部分复用 | 有 graph layer，但不是 HippoRAG 所需异构图索引 |
| `evolution_trigger` | 部分复用 | 论文弱化该 slot，但若单列 synonymy augmentation 触发，现有触发器不贴合 |
| `memory_evolution` | 部分复用 | 有 graph link evolution，但没有节点 embedding 相似驱动的 synonymy augmentation |
| `retrieval` | 部分复用 | 有 seed-and-expand 雏形，但缺 query linking、PPR、node specificity、passage 聚合 |
| `readout` | 直接复用 | 通用 passage readout 即可 |

### 重表达判断

只能部分映射。

原因不是 slot 数量不够，而是当前缺的正好是 HippoRAG 最核心的几段机制：

- 异构图索引而非普通 record graph
- query 实体抽取与节点链接
- Personalized PageRank 检索
- 节点激活到 passage 排名的聚合矩阵/统计结构

如果只用现有模块强行拼装，最多能做出“有 triples、有 graph、有 seed-expand”的近似版，但很难把 HippoRAG 最关键的单步多跳 pattern completion 检索语义表达准确。

### 备注与证据边界

- 论文明说
  - 离线阶段: passage -> named entities -> NER-conditioned OpenIE triples -> open KG。
  - 额外用 retrieval encoders 添加 synonymy relations。
  - 在线阶段: query named entities -> query nodes -> Personalized PageRank -> passage ranking。
  - node specificity 用于调节 query node 概率。
- repo 实现可确认
  - repo 主分支保留了 OpenIE、独立的 entity/fact/passage embedding stores、图构建与 `run_ppr()`。
  - `prompts/templates/ner.py` 与 `prompts/templates/triple_extraction.py` 证实了两阶段 NER + triple extraction。
  - `HippoRAG.retrieve()` 证实当前实现仍以图检索和 PPR 为骨架。
- 依据论文与 repo 做出的合理推断
  - 需要在 MemPrimitive 中新增的，不只是一个 `PPRRetrieval`，还包括与之配套的异构图组织与 node-to-passage 聚合结构。
  - query entity extraction 在当前 slot 体系下更适合落入 retrieval module 内部，而不是强行塞进 ingest-side representation。
- 当前证据不足、不能下结论的点
  - 官方 repo 当前主分支已明显带有 HippoRAG 2 演化痕迹，包含 fact scoring、DSPy rerank、passage node weight 等逻辑；这些不应直接当成 2024 原论文的严格机制。
  - 论文没有把所有工程性图数据结构完全形式化到可一一映射 MemPrimitive contract 的粒度，因此某些“矩阵是 organization 产物还是 retrieval 辅助索引”只能做合理落位，不能声称唯一正确。

## AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents

论文链接: <https://arxiv.org/abs/2407.04363>

官方 repo: <https://github.com/AIRI-Institute/AriGraph>

### 论文侧 memory 机制速写

AriGraph 不是离线文档索引型 memory，而是面向交互式环境的 world model memory。其核心是把每一步 observation 同时写进两套相互连接的记忆：

- semantic memory：把 observation 中抽取出的 triplet `(object1, relation, object2)` 并入语义图。
- episodic memory：把该步 observation 本身作为 episodic vertex 保存，并用 episodic edge 把“这一步同时出现的语义 triplets”与该 observation 连接起来。

在写入时，AriGraph 不只是 append 新 triplets，还会先定位与当前 observation 提到对象相关的既有 semantic edges，识别其中已过时的事实并删除，再把新 triplets 写入。  
在检索时，论文采用两阶段过程：

1. semantic search：先按 query 找相关 triplets，再沿 semantic graph 做受深度/宽度控制的扩展。
2. episodic search：再根据这些 triplets 反查相关 episodic observations，输出最相关的 past experiences。

从 MemPrimitive 的 slot 视角看，它更像：

- `unit_formation`：每步 observation 作为写入单元。
- `representation`：把 observation 解析成 semantic triplets。
- `write_trigger`：每步 observation 默认都写入。
- `organization`：同时维护 semantic graph 与 episodic observation linkage。
- `evolution_trigger`：每步 observation 都触发对已有 semantic memory 的更新。
- `memory_evolution`：删除与新 observation 冲突或已过时的旧 triplets。
- `retrieval`：先 semantic graph retrieval，再 episodic memory retrieval。
- `readout`：把检索出的语义 facts 与 episodic observations 整理给规划/决策模块。

### 按 MemPrimitive slot 的拆解

#### unit_formation

- 论文里做了什么
  - 每个时间步输入是一条新的环境 observation；该 observation 既会成为一个新的 episodic memory entry，也会触发 semantic triplet 抽取。
- MemPrimitive 现有哪些模块可直接复用
  - `PassThroughUnitFormation`
    - 如果把每步 observation 作为一个 `Observation` 输入，则可直接表达“一步 observation -> 一个 memory unit”。
- 哪些模块只能部分复用
  - `SentenceSplitUnitFormation`
  - `LineSplitUnitFormation`
  - `WindowedUnitFormation`
    - 这些模块能做切分，但 AriGraph 的核心写入粒度不是 observation 内分句，而是 step-level observation。
- 当前缺失什么能力
  - 无关键缺口。该 slot 不是 AriGraph 的机制瓶颈。
- 你的判断依据是什么
  - 论文明说：每一步 agent 接收 observation `o_t`，并基于它更新 semantic 与 episodic memory。
  - repo 实现可确认：`pipeline_arigraph.py` 中每一步都调用 `graph.update(observation, ...)`。
  - 推断：把 step-level observation 作为单个 `MemoryUnit` 足以承载该 slot。

#### representation

- 论文里做了什么
  - 从 observation 中抽取 semantic triplets `(object1, relation, object2)`，这些 triplets 会成为 semantic memory 的增量。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `BasicRepresentation(elements=("text", "triple"))`
    - 外观上最接近“把 observation 变成 triples”。
    - 但当前 MemPrimitive 中的 `triple` 表示能力并不等价于 AriGraph 论文里的 observation-conditioned world-fact extraction，而且项目进展文档已明确把 triple extraction 仍偏 heuristic 视为待改进项。
  - `KeywordRepresentation`
    - 只能提供 lexical side information，不覆盖 AriGraph 的核心 triplet representation。
- 当前缺失什么能力
  - 缺少一个面向 observation 的、显式输出 world-fact triplets 的表示模块。
  - 缺少把 triplets 作为后续 semantic graph 写入契约稳定暴露出来的表示边界。
- 你的判断依据是什么
  - 论文明说：AriGraph continuously learns world model by extracting semantic triplets from textual observations。
  - repo 实现可确认：`prompts/prompts.py` 中 `prompt_extraction_current` 明确要求从 observation 抽取 `"subject, relation, object"` 序列；`graphs/contriever_graph.py` 的 `update()` 调用该 prompt 并解析 triplets。
  - 推断：MemPrimitive 当前虽然有 `triple` 元素，但还不足以视为已落地了与 AriGraph 同语义的 observation-triplet extraction primitive。

#### write_trigger

- 论文里做了什么
  - 每一步 observation 都触发 memory learning；论文没有设计额外的 selective write gate。
- MemPrimitive 现有哪些模块可直接复用
  - `AlwaysWriteTrigger`
- 哪些模块只能部分复用
  - `ThresholdWriteTrigger`
    - 可以通过常量阈值模拟“总是写入”，但这只是绕写，不是最直接表达。
- 当前缺失什么能力
  - 无关键缺口。
- 你的判断依据是什么
  - 论文明说：Every observation triggers learning that updates agent’s world model。
  - repo 实现可确认：`pipeline_arigraph.py` 在每个 step 都无条件调用 `graph.update(...)`。
  - 推断：该 slot 可由现有“全写入”模块直接表达。

#### organization

- 论文里做了什么
  - 组织一个混合 memory graph `G = (V_s, E_s, V_e, E_e)`：
    - semantic vertices / edges 承载 object-level facts；
    - episodic vertices 存 observation 文本；
    - episodic edges 把“同一步 observation 提取出的 semantic triplets”与该 observation 连接起来。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `GraphAppendOrganization`
    - 能把记录放入 graph layer，并维护一些图元数据；现在也可服务 note graph/A-MEM 风格写入。
    - 但它表达的仍是 record-centric graph append，不是 AriGraph 所需的“semantic triplets + episodic observation + 跨两类记忆的连接”，也不是 semantic/episodic 双记忆结构。
- 当前缺失什么能力
  - 缺少同时组织 semantic graph 与 episodic observation memory 的统一 organization primitive。
  - 缺少“observation -> 当步 triplets”这种跨两类记忆对象的连接结构。
  - 缺少对 episodic edge 这种“连接 observation 与一组 semantic edges”的正式承载边界。
- 你的判断依据是什么
  - 论文明说：AriGraph world model 由 semantic vertices/edges 与 episodic vertices/edges 共同组成。
  - repo 实现可确认：
    - `graphs/parent_graph.py` / `graphs/contriever_graph.py` 维护 semantic triplets 图；
    - `graphs/contriever_graph.py` 还维护 `obs_episodic`，把 observation 与其对应 triplets/embedding 关联保存。
  - 依据论文与 repo 做出的合理推断：
    - repo 中 episodic memory 更像“observation -> associated triplets”的字典，而不是论文图示里的显式 episodic hyperedge，但两者表达的 memory linkage 是同一类机制。

#### evolution_trigger

- 论文里做了什么
  - 每次新 observation 到来时，都会触发对相关既有 semantic facts 的检查与更新。
- MemPrimitive 现有哪些模块可直接复用
  - `ThresholdEvolutionTrigger(threshold=0.5, constant=1.0)`
    - 通过常量触发可以表达“每次写入后都执行演化”。
- 哪些模块只能部分复用
  - `NewWriteEvolutionTrigger`
    - 有“新写入后触发局部维护”的轮廓，但它当前面向 keyed/local-maintenance 家族，不是 AriGraph 的 observation-conditioned fact revision 语义。
- 当前缺失什么能力
  - 无必须新增的 trigger 缺口；该 slot 的关键不在 trigger，而在后续 evolution 本体。
- 你的判断依据是什么
  - 论文明说：Given new observation `o_t`，系统会先找相关已有知识，再移除过时边并扩展 semantic memory。
  - repo 实现可确认：`graph.update(...)` 在每次 observation 到来时都执行“抽取 -> 找相关子图 -> 识别 outdated -> 删除 -> 写入”。
  - 推断：这个 slot 可以通过现有 always-like evolution trigger 组合表达。

#### memory_evolution

- 论文里做了什么
  - 对 observation 涉及到的对象，先找到已有相关 semantic edges；
  - 再识别哪些旧事实已过时，与当前 observation 的新 triplets 冲突；
  - 删除这些 outdated edges，然后再并入新知识。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `GraphLinkEvolution`
    - 能对 graph layer 做额外图写回，但偏向补 link，不支持“根据新 observation 删除/替换过时 semantic facts”。
  - `GraphNeighborAppendEvolution`
    - 只是 `GraphLinkEvolution` 的兼容包装，能力边界相同。
  - `NeighborContextUpdateEvolution`
    - 能重写邻居上下文，但其目标是 note graph 邻居改写，不是 AriGraph 的事实级冲突消解。
- 当前缺失什么能力
  - 缺少“基于新 observation triplets，对相关旧 semantic edges 做冲突检测与删除”的 memory evolution primitive。
  - 缺少删除式 graph evolution，而不仅是 append/link-strengthening。
- 你的判断依据是什么
  - 论文明说：outdated edges in related semantic edges are detected by comparing them with new triplets and removed from the graph。
  - repo 实现可确认：
    - `prompts/prompts.py` 中 `prompt_refining_items` 专门要求判断 existing triplets 中哪些应被新 triplets 替换；
    - `graphs/contriever_graph.py` 的 `update()` 会调用 `parse_triplets_removing(...)` 和 `delete_triplets(...)`。
  - 推断：AriGraph 的关键 evolution 不是“增补相似边”，而是“局部事实修订”；这在 MemPrimitive 当前模块集里是明确缺口。

#### retrieval

- 论文里做了什么
  - 检索分两段：
    - semantic search：先按 query 找最相关 triplets，再沿 semantic graph 递归扩展；
    - episodic search：再把这些 triplets 回连到 past episodic observations，给 observation-level experiences 打分并返回 top-k。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `ExpandRetrievedGraphNeighbors`
    - 覆盖了“先有 seed 再沿图扩展”的粗轮廓。
    - 但它当前仍只是一跳 seed-expand 组件，不包含 AriGraph 的“semantic triplet retrieval -> episodic retrieval”两段式输出语义。
  - `EmbeddingSimilarityRetrieval`
    - 可类比 AriGraph semantic search 中的 embedding-based relevance，但它只返回相似记录，不会继续图扩展，也不会回连 episodic memory。
  - `GraphNeighborRetrieval`
    - 只适合已知 seed id 的邻居检索，不适合 AriGraph 这种从 query 文本出发的 triplet-level semantic retrieval。
  - `LayerAwareRetrieval`
    - 只能做层间分发，不能表达 AriGraph 的 semantic/episodic 联动检索流程。
- 当前缺失什么能力
  - 缺少“semantic graph retrieval + episodic memory retrieval”一体化检索 primitive。
  - 缺少基于 triplet 匹配结果给 episodic observations 打分并返回的机制。
  - 缺少对“query set / 多个查询实体”输入的原生支持边界。
- 你的判断依据是什么
  - 论文明说：Algorithm 1 先做 `SemanticSearch(q, V_s, E_s, d, w)`，再做 `EpisodicSearch(E_s^Q, V_e, E_e, k)`。
  - repo 实现可确认：
    - `utils/retriever_search_drafts.py` 的 `graph_retr_search(...)` 通过 embedding search over triplet strings + BFS 扩展返回 associated subgraph；
    - `utils/utils.py` 的 `find_top_episodic_emb(...)` 根据 retrieved triplets 与 observation 关联 triplets 的重合度，并结合 observation/plan embedding similarity，返回 top episodic memories；
    - `agents/parent_agent.py` 还会先从 observation/plan 中提取“crucial items”作为 retrieval queries。
  - 备注
    - repo 的 episodic scoring 比论文更工程化，除了 triplet overlap 还混入了 observation-plan embedding similarity；这应视为 repo 细化，而不是论文主机制本身。

#### readout

- 论文里做了什么
  - 把 relevant semantic memories 与 relevant episodic memories 放入 working memory，供 planning 和 decision making 使用。
- MemPrimitive 现有哪些模块可直接复用
  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
    - 这些模块都可以承担“把检索出的文本化记忆拼接给下游”的职责。
- 哪些模块只能部分复用
  - `GraphReadout`
    - 适合调试图结构，但 AriGraph 默认 readout 面向 working memory 的文本上下文，不是图调试输出。
  - `PromptContextReadout`
    - 具备“为下游 prompt 组上下文”的方向，但当前是 Reflexion 风格上下文模板，不是 AriGraph 的 semantic+episodic 组合格式。
- 当前缺失什么能力
  - 若只要求“把检索结果交给下游”，无关键缺口。
  - 若要求更贴近论文中的 working-memory 组织格式，则缺少一个能显式区分 semantic memories 与 episodic memories 的 readout 模板。
- 你的判断依据是什么
  - 论文明说：working memory 包含 current observation、recent history、relevant semantic memories、relevant episodic memories、goal 与 current plan。
  - repo 实现可确认：`pipeline_arigraph.py` 在 planning / action prompt 中分别注入 `subgraph` 和 `top_episodic`。
  - 推断：readout 不是 AriGraph 的主要机制瓶颈，主要缺口仍在 retrieval 之前。

### 与 MemPrimitive 现有组件的对照结论

| slot | 结论 | 说明 |
| --- | --- | --- |
| `unit_formation` | 直接复用 | `PassThroughUnitFormation` 足以表达 step-level observation 写入 |
| `representation` | 部分复用 | 有 `triple` 表示壳子，但缺 observation-level、非 heuristic 的 triplet extraction primitive |
| `write_trigger` | 直接复用 | `AlwaysWriteTrigger` 足够 |
| `organization` | 部分复用 | 有 graph append，但缺 semantic memory + episodic memory 联合组织 |
| `evolution_trigger` | 直接复用 | 可用现有 always-like trigger 组合表达“每步都演化” |
| `memory_evolution` | 部分复用 | 有 graph evolution 外壳，但缺 AriGraph 核心的 outdated fact pruning |
| `retrieval` | 部分复用 | 有 graph seed-expand 雏形，但缺 semantic retrieval 与 episodic retrieval 的两段联动 |
| `readout` | 直接复用 | 通用文本 readout 可承载基础 working-memory 注入 |

### 重表达判断

只能部分映射。

主要原因不是 slot 体系不适配，而是当前缺失的正好是 AriGraph 的三类核心机制：

- semantic graph 与 episodic observation memory 的联合组织结构
- 基于新 observation 的 outdated fact pruning
- semantic retrieval 之后回连 episodic memory 的两段式检索

如果只用现有模块强行拼装，可以得到一个“有 triplets、有 graph、能做一些 graph retrieval”的近似版，但还不足以忠实重表达 AriGraph 作为 world-model memory 的关键闭环。

### 备注与证据边界

- 论文明说
  - AriGraph 由 semantic vertices/edges 与 episodic vertices/edges 共同组成。
  - 新 observation 到来后，会提取 triplets、识别相关旧知识、移除过时边，再扩展 semantic memory。
  - 检索分为 semantic search 与 episodic search 两段。
  - episodic relevance 由匹配 triplet 数量与 observation 信息量共同决定。
- repo 实现可确认
  - `graphs/contriever_graph.py` 的 `update()` 实现了 triplet 抽取、相关子图检索、outdated triplet 删除与新 triplet 写入。
  - `utils/retriever_search_drafts.py` 实现了基于 embedding search + BFS 的 semantic subgraph retrieval。
  - `utils/utils.py` 的 `find_top_episodic_emb(...)` 实现了 episodic memory 排序。
  - `pipeline_arigraph.py` 将 retrieved semantic memories 与 episodic memories 一起注入 planning / action prompt。
- 依据论文与 repo 做出的合理推断
  - 在 MemPrimitive 中，AriGraph 最自然的缺失模块不是单一的 graph retriever，而是“semantic/episodic graph organization + outdated fact pruning + two-stage retrieval”这一整套 primitive 边界。
  - `readout` 可以继续复用通用文本 readout，不需要为 AriGraph 单独新增重型 readout family。
- 当前证据不足、不能下结论的点
  - 论文中的 episodic edge 更像显式图结构；repo 当前更偏工程实现，使用 observation 到 triplet 列表/embedding 的关联存储。两者语义接近，但数据结构并不一一同构。
  - repo 的 episodic scoring 混入了 observation/plan embedding similarity，这比论文主文中的公式更强；不能把这部分直接当成论文明确主张。

## HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model

论文链接: <https://aclanthology.org/2025.acl-long.1575/>

官方 repo: <https://github.com/HiAgent2024/HiAgent>

### 论文侧 memory 机制速写

HiAgent 关注的不是跨 trial 长期记忆，而是单次任务尝试中的 working memory 管理。它把 subgoal 作为 working-memory chunk：当前 subgoal 保留完整的 action-observation 细节，已完成 subgoal 则被压缩成一个 summarized observation，与 subgoal 一起留在上下文里。这样，历史轨迹不会被整段原样塞进 prompt，而是以“`(subgoal, summary)` 为主，当前 subgoal 保留细节”的层级结构暴露给模型。

如果模型判断某个旧 subgoal 的详细轨迹对当前决策仍然关键，HiAgent 允许模型显式生成 `retrieve(subgoal_id)`，把对应 subgoal 的完整 action-observation pairs 再拉回上下文。按 MemPrimitive slot 看，它更像：

- `unit_formation`: 每一步交互形成新的 working-memory 增量；
- `representation`: 主要保持自然语言 action / observation / subgoal 文本表示；
- `write_trigger`: 每一步默认写入当前 working-memory 轨迹；
- `organization`: 按 subgoal 切 chunk，区分当前 detailed chunk 与历史 summarized chunks；
- `evolution_trigger`: 当模型判断当前 subgoal 已完成时，触发压缩归档；
- `memory_evolution`: 把已完成 subgoal 的详细轨迹总结成 summarized observation，并以 summary 取代默认暴露的旧细节；
- `retrieval`: 按需按 subgoal id 取回某个旧 chunk 的完整详细轨迹；
- `readout`: 把“历史 summary + 当前细节 + 按需恢复的旧细节”拼成工作记忆上下文。

### 按 MemPrimitive slot 的拆解

#### unit_formation

- 论文里做了什么
  - 每个时间步都会把新的交互结果并入当前 trial 的 working memory；repo 中实际保存的是按时间追加的 action-observation 对，以及必要时插入的 `Subgoal` 标记。
- MemPrimitive 现有哪些模块可直接复用
  - `PassThroughUnitFormation`
    - 如果把每一步 observation 视为一次 ingest，或把 action 一并编码进 observation 文本/metadata，则足以承载“每步产生一个新的 working-memory 增量”这一最小写入单位。
- 哪些模块只能部分复用
  - `SentenceSplitUnitFormation`
  - `LineSplitUnitFormation`
  - `WindowedUnitFormation`
    - 它们能做切分，但 HiAgent 的关键粒度不是句子或窗口，而是交互 step 与 subgoal chunk。
- 当前缺失什么能力
  - 无关键缺口。HiAgent 的主要创新不在 unit formation。
- 你的判断依据是什么
  - 论文明说：agent 在每个时间步维护 working memory，并围绕 subgoal 组织 action-observation history。
  - repo 实现可确认：`agentboard/agents/cme_final.py` 中 `self.memory` 按步追加 `[("Action", action), ("Observation", state)]`，并在需要时插入 `[("Subgoal", subgoal)]`。
  - 推断：MemPrimitive 现有 step-level unit 形成能力足以承载该 slot。

#### representation

- 论文里做了什么
  - HiAgent 的 memory 表示核心仍是自然语言文本，而不是 embedding / graph / triples：包括 subgoal 文本、action-observation 对文本，以及已完成 subgoal 的 summarized observation。
- MemPrimitive 现有哪些模块可直接复用
  - `BasicRepresentation(elements=("text",))`
    - 足以表达 HiAgent 对 working-memory 内容的最基本文本表示。
- 哪些模块只能部分复用
  - `BasicRepresentation(elements=("text", "summary"))`
    - 表面上有 summary 能力，但该 summary 是对单个 unit 文本的表示增强，不是 HiAgent 这种“对一个完整 subgoal trajectory 做 subgoal-conditioned summarization”。
  - 多个 `LLMRepresentation`（例如 `context` / `keywords` / `tags` / `category` / `attributes`）
  - `ConfigurableEmbeddingRepresentation`
    - 可为 unit 附加结构化 note payload，但它服务的是 note-like enrichment，不是 HiAgent 的 subgoal / summarized observation 语义。
- 当前缺失什么能力
  - 无必须单独新增到 `representation` slot 的关键能力。HiAgent 的关键 summary 更适合落在 `memory_evolution`，而不是 ingest-side representation。
- 你的判断依据是什么
  - 论文明说：summarized observation `s_i = S(g_i, o_0, a_0, ..., o_t)` 是围绕 subgoal 对历史轨迹的压缩表示。
  - repo 实现可确认：`cme_final.py` 中 memory 主要由 `"Subgoal"`、`"Action"`、`"Observation"` 这类文本标签及其字符串内容构成。
  - 依据论文与 repo 做出的合理推断：HiAgent 没有依赖复杂的底层表示学习；其关键不在表示种类，而在后续如何按 subgoal 管理这些文本块。

#### write_trigger

- 论文里做了什么
  - 默认每一步交互都会写入当前 working-memory 轨迹，没有专门的 selective write 机制。
- MemPrimitive 现有哪些模块可直接复用
  - `AlwaysWriteTrigger`
- 哪些模块只能部分复用
  - `ThresholdWriteTrigger`
    - 技术上可以模拟全写入，但不是最直接表达。
- 当前缺失什么能力
  - 无关键缺口。
- 你的判断依据是什么
  - 论文明说：HiAgent 的区别主要在 working-memory 的层级管理，而不是“哪些 step 值得写入”。
  - repo 实现可确认：`cme_final.py` 的 `update()` 在每个 step 都把 action-observation 对追加进 `self.memory`。
  - 推断：该 slot 可由现有全写入模块直接表达。

#### organization

- 论文里做了什么
  - 以 subgoal 为 chunk 组织 working memory：
    - 当前 subgoal 保留完整 action-observation pairs；
    - 已完成 subgoal 只保留 `(subgoal, summarized observation)`；
    - 需要时可再把某个旧 subgoal 的详细轨迹恢复到上下文中。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `AppendOrganization`
    - 能顺序写入记录，但不能表达“当前 chunk 保留细节、历史 chunk 默认只暴露 summary”的层级 chunk 组织。
  - `ConditionalLayerOrganization`
    - 可以按规则分层，但缺少 subgoal chunk 边界、subgoal id、当前/历史状态切换等控制逻辑。
  - `PlacementWithoutAppendOrganization`
    - 能发 placement，但不能独立完成 HiAgent 的 chunk 组织。
- 当前缺失什么能力
  - 缺少显式的 subgoal-chunk organization primitive。
  - 缺少“当前 detailed chunk / 历史 summarized chunk / 按需恢复旧 detailed chunk”的统一组织边界。
  - 缺少稳定的 subgoal id 与 chunk 生命周期承载方式。
- 你的判断依据是什么
  - 论文明说：working memory 形式从直接保留完整历史，变成以 subgoal 为 chunk 的层级表示，过去 subgoal 只保留 summary，当前 subgoal 保留细节。
  - repo 实现可确认：`cme_final.py` 中 `serialize_history()` 会扫描 `Subgoal` 边界，对旧 subgoal 改写成编号后的 `Subgoal + Observation(summary)`，而最后一个 subgoal 之后的详细轨迹保持展开。
  - 依据论文与 repo 做出的合理推断：HiAgent 的核心组织机制不是简单 append，而是“chunked working-memory exposure policy”。

#### evolution_trigger

- 论文里做了什么
  - 当模型判断当前 subgoal 已完成时，触发把该 subgoal 的详细轨迹压缩成 summarized observation，并进入下一个 subgoal。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `ThresholdEvolutionTrigger`
    - 可以模拟“每次都做演化”，但不能表达 HiAgent 需要的“只在 subgoal 完成时触发压缩”。
  - `OutcomeConditionedEvolutionTrigger`
    - 轮廓上有“基于结果信号触发”的方向，但它面向 Reflexion 式 trial 结果，不是 HiAgent 的 subgoal-completion 判断。
  - `NewWriteEvolutionTrigger`
    - 有“新写入后做维护”的轮廓，但不是 HiAgent 的 chunk 压缩时机。
- 当前缺失什么能力
  - 缺少面向“subgoal 完成”这一事件的 evolution trigger。
  - 缺少把 LLM 的 subgoal-completion 判断显式转成 `evolution_decisions` 的通用边界。
- 你的判断依据是什么
  - 论文明说：LLM 可以在每步要么继续当前 subgoal，要么在判断当前 subgoal 已完成后生成新 subgoal。
  - repo 实现可确认：`cme_final.py` 中当模型输出包含 `Subgoal:` 时，会把它视为进入新 subgoal 的切换点；旧 subgoal 随后会在序列化阶段被压缩。
  - 推断：现有 evolution trigger 家族还没有直接覆盖这种 subgoal-level switch 语义。

#### memory_evolution

- 论文里做了什么
  - 对已完成 subgoal 的完整 action-observation trajectory 做 summarization，得到 summarized observation；
  - 默认上下文不再展开旧 detailed trajectory，而是用 `(subgoal, summary)` 取代；
  - 当前 subgoal 的详细轨迹继续保持展开。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `SummaryRewriteEvolution`
    - 有“追加 summary record”的轮廓，但它对齐的是单个 unit 的 summary rewrite，而不是 HiAgent 的多步 subgoal trajectory summarization。
  - `LayerMoveEvolution`
    - 可以把内容复制到另一层，但只是 copy-append，不具备“旧细节默认隐藏、summary 取代其上下文暴露”的替换语义。
- 当前缺失什么能力
  - 缺少对完整 subgoal chunk 做 summarization 的 evolution primitive。
  - 缺少“summary 取代旧细节为默认上下文暴露形式，但旧细节仍可按需恢复”的 replacement/archive 机制。
  - 缺少对 chunk 级而非 unit 级 memory evolution 的明确承载边界。
- 你的判断依据是什么
  - 论文明说：已完成 subgoal 的 action-observation pairs 会被 summarized observation 替换；当前 subgoal 保留细节。
  - repo 实现可确认：
    - `cme_final.py` 在 `serialize_history()` 中对除最后一个之外的旧 subgoal 做压缩；
    - 它调用 `TrajectorySummarizer.generate_summary([trajectory], [subgoal])` 生成 summary；
    - 旧 subgoal 在上下文中被改写成 `Subgoal + Observation(summary)`。
  - 当前证据不足、不能下结论的点
    - 公开 repo 中 `TrajectorySummarizer` 的源码当前未找到，因此 summary 模型内部实现细节无法确认。

#### retrieval

- 论文里做了什么
  - 当模型觉得某个旧 subgoal 的详细轨迹仍然关键时，显式生成 retrieval function，按 subgoal id 取回对应的完整 action-observation pairs。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `BufferRetrieval`
    - 只能取最近窗口，不能按 subgoal id 精确回取某个旧 chunk。
  - `RecencyRetrieval`
    - 只能按时间顺序读最近记录，不理解 subgoal 边界。
  - `LayerAwareRetrieval`
    - 只能在多个 retriever 间分发，不能单独承担“按 subgoal 标识符恢复详细轨迹”。
- 当前缺失什么能力
  - 缺少按 subgoal id 检索旧 detailed trajectory 的 retrieval primitive。
  - 缺少“summary chunk -> detailed chunk”的按需 rehydrate/reveal 语义。
  - 缺少与模型显式 `retrieve(subgoal_id)` 调用对接的 retrieval 边界。
- 你的判断依据是什么
  - 论文明说：trajectory retrieval module 允许模型在需要时取回过去某个 subgoal 的完整 action-observation pairs。
  - repo 实现可确认：
    - `cme_final.py` 通过正则解析 `retrieve(3)` 这类动作；
    - 命中后把 subgoal id 记入 `self.subgoal_idx`，再重新构造 prompt；
    - `serialize_history()` 对被点名的旧 subgoal 不再压缩，而是把原始详细轨迹重新拼回上下文。
  - 依据论文与 repo 做出的合理推断：HiAgent 的 retrieval 不是语义相似检索，而是 chunk id-addressed trajectory recall。

#### readout

- 论文里做了什么
  - 把 working memory 渲染成 prompt 上下文：
    - 历史 subgoal 以 `subgoal + summary` 形式出现；
    - 当前 subgoal 以完整详细轨迹出现；
    - 若触发 retrieval，则某些旧 subgoal 也会恢复为完整详细轨迹。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `ConcatenateReadout`
    - 能做纯文本拼接，但不能显式保留层级 working-memory 结构。
  - `GroupedByLayerReadout`
    - 轮廓上最接近“按层显示”，但仍不能表达 subgoal 编号、summary-vs-detail 切换、按需恢复旧 chunk 等格式。
  - `PromptContextReadout`
    - 有“拼 prompt context”的方向，但当前更偏 Reflexion/readout 语义，不是 HiAgent 的层级工作记忆呈现。
- 当前缺失什么能力
  - 缺少一个显式面向 hierarchical working memory 的 readout 模板。
  - 缺少稳定呈现“历史 summary + 当前细节 + 恢复旧细节”的读出格式。
- 你的判断依据是什么
  - 论文明说：HiAgent 的 working memory 本质上就是一种层级上下文组织方式。
  - repo 实现可确认：`cme_final.py` 的 `make_prompt()` 直接通过 `serialize_history()` 把压缩后的旧 subgoal、当前详细轨迹、以及被 retrieval 指定的旧详细轨迹拼成最终 prompt。
  - 推断：该 slot 不是 HiAgent 的最难点，但当前通用 readout 还不足以忠实重表达其层级上下文格式。

### 与 MemPrimitive 现有组件的对照结论

| slot | 结论 | 说明 |
| --- | --- | --- |
| `unit_formation` | 直接复用 | `PassThroughUnitFormation` 足以承载 step-level working-memory 增量 |
| `representation` | 直接复用 | HiAgent 主要依赖自然语言文本表示，复杂机制不在该 slot |
| `write_trigger` | 直接复用 | `AlwaysWriteTrigger` 足够 |
| `organization` | 部分复用 | 有 append / routing 外壳，但缺 subgoal-chunk 层级组织 |
| `evolution_trigger` | 部分复用 | 有通用 trigger 外壳，但缺 subgoal-completion 触发 |
| `memory_evolution` | 部分复用 | 有 `SummaryRewriteEvolution` / `LayerMoveEvolution` 轮廓，但缺 chunk-level summarization 与 replacement/archive 语义 |
| `retrieval` | 部分复用 | 有 recency/buffer/layer-aware 轮廓，但缺按 subgoal id 恢复旧 detailed trajectory |
| `readout` | 部分复用 | 有通用文本/Prompt readout，但缺 hierarchical working-memory 呈现格式 |

### 重表达判断

只能部分映射。

主要原因不是 HiAgent 需要新的顶层 slot，而是当前缺失的正好是它的核心 working-memory 机制边界：

- subgoal-chunked hierarchical organization
- subgoal-completion-triggered chunk summarization
- summary 替代旧细节的默认上下文暴露语义
- 按 subgoal id 恢复旧 detailed trajectory 的 retrieval

如果只用现有模块强行拼装，最多能做出“有文本 summary、有 append、有简单 readout”的近似版，但还不足以忠实重表达 HiAgent 作为层级 working-memory 管理方法的关键机制。

### 备注与证据边界

- 论文明说
  - HiAgent 把 subgoal 作为 working-memory chunk。
  - 当前 subgoal 保留详细 action-observation pairs，过去 subgoal 保留 summarized observation。
  - summarized observation 由 `S(g_i, o_0, a_0, ..., o_t)` 生成，并需要评估 subgoal 是否完成。
  - 当需要旧细节时，模型可生成 retrieval function 取回某个 subgoal 的完整 trajectory。
- repo 实现可确认
  - `agentboard/agents/cme_final.py` 维护 `self.memory`，其中混合保存 `Subgoal`、`Action`、`Observation` 文本条目。
  - `serialize_history()` 会压缩最后一个 subgoal 之前的旧 subgoal，只保留 `Subgoal + Observation(summary)`。
  - `retrieve(subgoal_id)` 会被显式解析，命中的旧 subgoal 不再压缩，而是把原始 detailed trajectory 重新拼回 prompt。
- 依据论文与 repo 做出的合理推断
  - HiAgent 的关键 primitive 边界更接近“hierarchical working-memory chunk lifecycle”，而不是传统的 semantic retrieval 或 long-term memory indexing。
  - 在 MemPrimitive 里，最自然的新增模块应集中在 `organization`、`evolution_trigger`、`memory_evolution`、`retrieval`、`readout`，而不是 `representation`。
- 当前证据不足、不能下结论的点
  - 公开 repo 中 `TrajectorySummarizer` 被调用，但对应源码当前未找到，因此摘要模型的具体实现方式无法确认。
  - 公开 eval config 使用的 agent 名称是 `ContextEfficientAgent`，而代码里注册的是 `ContextEfficientAgentV2`；repo 当前存在命名不一致，说明公开实现可能有缺失或整理不完整之处。

## Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

论文链接: <https://arxiv.org/abs/2504.19413>

官方 repo: <https://github.com/mem0ai/mem0>

### 论文侧 memory 机制速写

Mem0 论文主体的核心不是“把整段对话直接塞进向量库”，而是一个两阶段 memory pipeline：

1. extraction phase  
   - 以“当前消息对”作为处理单位；
   - 结合会话摘要与最近若干条消息作为上下文；
   - 用 LLM 抽取 candidate facts / candidate memories。
2. update phase  
   - 对每个 candidate fact 检索 top-k 相似旧记忆；
   - 再由 LLM 决策对知识库执行 `ADD / UPDATE / DELETE / NONE`；
   - 最终把结果写回向量记忆库，并维护时间一致性与去冗余。

论文还提出了 graph-memory 增强版：把 memory 表示成带实体类型、实体 embedding 和关系边的有向标注图；新信息进入时会做冲突检测，把过时关系标为无效；检索时同时支持 query 实体锚定扩展与 triplet-level 语义匹配。

如果按 MemPrimitive 当前 slot 体系强行落位，最自然的拆法是：

- `unit_formation`: 以一对新消息为交互单元。
- `representation`: 从新交互中抽 salient facts；graph 版则抽实体类型与关系 triplets。
- `write_trigger`: 对已抽出的 candidate facts 默认都进入后续评估。
- `organization`: 基础版是平坦的 scoped vector memory；graph 版是 typed entity-relation graph。
- `evolution_trigger`: 每个 candidate fact 都触发一次“与旧记忆比较并决定如何维护”的流程。
- `memory_evolution`: 真正执行 `ADD / UPDATE / DELETE / NONE`；graph 版还要做冲突关系失效化。
- `retrieval`: 基础版是 embedding similarity recall；graph 版是 entity-centric + triplet similarity 的图检索。
- `readout`: 返回相关 memory 文本，graph 版附带关系上下文。

### 按 MemPrimitive slot 的拆解

#### unit_formation

- 论文里做了什么
  - 论文把“当前消息与其前一条消息”组成一个 interaction unit；抽取时还会引入 conversation summary 与 recent messages 作为辅助上下文。
- MemPrimitive 现有哪些模块可直接复用
  - 无完全直接复用模块。
- 哪些模块只能部分复用
  - `PassThroughUnitFormation`
    - 如果上游已经把“消息对 + 必要上下文”预先打包成一个 `Observation`，可以承载 Mem0 的最小处理单元。
  - `MetadataHintUnitFormation`
    - 可以从 metadata hints 中构造 unit，但它并不是为“对话双消息单元”设计的。
- 当前缺失什么能力
  - 缺少面向 conversation turn-pair 的原生 unit formation。
  - 缺少把 recent-message window / conversation summary 稳定并入当前 unit 的形成边界。
- 你的判断依据是什么
  - 论文明说：Mem0 以新的 message pair 为处理单位，并结合 conversation summary 与 recent messages 做 extraction。
  - repo 实现可确认：当前 OSS `Memory.add(...)` 直接接收 message 列表，`parse_messages(...)` 把对话串成文本，但没有一个显式的“turn-pair unit formation”模块。
  - 依据论文与 repo 做出的合理推断：在 MemPrimitive 中，这一语义可以借 `PassThroughUnitFormation` 承载，但需要上游手动打包，因此只能部分复用。

#### representation

- 论文里做了什么
  - 基础 Mem0：从新交互中抽取 salient facts / memories，且抽取要感知会话摘要与最近上下文。
  - graph 版：先抽实体及类型，再抽实体间关系 triplets。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `BasicRepresentation(elements=("text", "summary"))`
    - 有“文本 + 摘要”外观，但它是对单个 unit 做局部表示增强，不是 Mem0 那种“基于新交互 + 会话上下文抽 candidate facts”的 extraction。
  - `BasicRepresentation(elements=("text", "triple", "entities"))`
    - 对 graph 版只覆盖了“看起来像有 entities/triples”的表面结构，不等于带实体类型的关系抽取。
- 当前缺失什么能力
  - 缺少 conversation-context-aware 的 salient fact extraction primitive。
  - 缺少“从消息对中直接产出 candidate facts 列表”的表示模块。
  - 缺少 graph 版所需的 typed entity extraction + relation triplet extraction primitive。
- 你的判断依据是什么
  - 论文明说：基础 Mem0 的 extraction phase 结合 `S_t` 与 recent messages，从新 message pair 中抽 memories；graph 版明确分为 entity extraction 与 relationship generation 两步。
  - repo 实现可确认：`mem0/memory/main.py` 里 `get_fact_retrieval_messages(...)` 驱动 LLM 抽取 `facts`；`mem0/memory/graph_memory.py` 中 `_retrieve_nodes_from_data(...)` 与 `_establish_nodes_relations_from_data(...)` 分别做实体与关系抽取。
  - repo 实现可确认：当前 OSS 路径没有看到论文所说的独立 async summary refresh 模块暴露在开源主链路中。
  - 推断：MemPrimitive 现有 representation 家族主要做 unit-local 表示增强，不足以直接表达 Mem0 的 extraction 核心。

#### write_trigger

- 论文里做了什么
  - 论文没有单独强调一个复杂 write gate；一旦 extraction phase 产出 candidate facts，这些候选就会进入 update evaluation。
- MemPrimitive 现有哪些模块可直接复用
  - `AlwaysWriteTrigger`
- 哪些模块只能部分复用
  - `ThresholdWriteTrigger`
    - 可以充当 always-like 触发，但不是最自然表达。
- 当前缺失什么能力
  - 无关键缺口。
- 你的判断依据是什么
  - 论文明说：candidate facts 会进入 update phase，与相似旧记忆比较后决定最终操作。
  - repo 实现可确认：`_add_to_vector_store(...)` 中只要抽出了 `new_retrieved_facts`，就会继续走相似检索和 memory action 决策。
  - 推断：Mem0 的 selectivity 主要发生在 extraction 与 update-resolution，不在独立 write gate。

#### organization

- 论文里做了什么
  - 基础 Mem0：把 memory 维护在一个 scoped 的平坦记忆库里，依赖 embedding 检索、时间戳和稳定 ID。
  - graph 版：把 memory 组织成有向标注图，节点含实体类型、embedding 与元数据，边为关系 triplets。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `AppendOrganization`
    - 对基础版 ADD 路径是接近的，能表达“把新 fact 作为 record 写入某层”。
    - 但它不内建 Mem0 的 scoped mutable memory 语义，也不处理与后续 update/delete 生命周期的耦合。
  - `GraphAppendOrganization`
    - 对 graph 版只提供了 record-centric graph append 外壳，不等于 typed entity-relation graph。
- 当前缺失什么能力
  - 缺少面向基础 Mem0 的“平坦但可后续原位维护”的 memory organization 语义边界。
  - 缺少 graph 版的 typed entity-relation graph organization。
  - 缺少对实体节点 embedding、关系边状态、作用域 metadata 的统一组织 contract。
- 你的判断依据是什么
  - 论文明说：基础版使用向量 memory database；graph 版使用 directed labeled graph，节点含 type/embedding/metadata。
  - repo 实现可确认：`mem0/memory/main.py` 把 memory 作为向量库记录管理；`mem0/memory/graph_memory.py` 用图数据库维护实体节点与关系边。
  - 推断：基础版的静态组织不复杂，但如果按 MemPrimitive 的组织 slot 严格划界，当前模块仍只覆盖了 append 外壳。

#### evolution_trigger

- 论文里做了什么
  - 每个 candidate fact 都会触发一次 update evaluation；graph 版新关系进入时也会触发 conflict detection。
- MemPrimitive 现有哪些模块可直接复用
  - 可直接用现有 always-like 组合表达。
  - `ThresholdEvolutionTrigger`
    - 设定成恒真时，可表达“每个 candidate fact 都进入后续 evolution”。
- 哪些模块只能部分复用
  - `NewWriteEvolutionTrigger`
    - 有“新写入后做维护”的轮廓，但不等于 Mem0 的每-fact update evaluation。
- 当前缺失什么能力
  - 无必须新增的触发 primitive；现有组合足够承载该 slot。
- 你的判断依据是什么
  - 论文明说：update phase 是 extraction 后的固定后续阶段，而不是条件很复杂的稀疏触发。
  - repo 实现可确认：`_add_to_vector_store(...)` 对每个新 fact 都执行相似检索与 memory action 决策。
  - 推断：该 slot 在 Mem0 中不是主要瓶颈。

#### memory_evolution

- 论文里做了什么
  - 基础 Mem0 的真正核心在这里：对每个 candidate fact 先找 top-k 相似旧记忆，再由 LLM 决定执行 `ADD / UPDATE / DELETE / NONE`。
  - graph 版还要对冲突关系做 invalidation，而不是只会追加。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `SummaryRewriteEvolution`
    - 有“基于已有内容做改写”的轮廓，但与 Mem0 的 action-resolution 完全不同。
  - `LayerMoveEvolution`
    - 有 record rewrite / move 的味道，但不负责相似检索与 `ADD/UPDATE/DELETE/NONE` 决策。
  - `GraphLinkEvolution`
    - graph 版可借其“图上做额外维护”的外壳，但不支持 conflict invalidation。
- 当前缺失什么能力
  - 缺少 `ADD / UPDATE / DELETE / NONE` 四路 memory maintenance primitive。
  - 缺少“candidate fact -> top-k similar old memories -> LLM decision -> store mutation”的完整 evolution module。
  - 缺少 graph 版“把冲突关系标记失效而非直接删除”的 evolution 语义。
- 你的判断依据是什么
  - 论文明说：基础 Mem0 的 update phase 明确有四种操作；graph 版对冲突关系做 obsolete/invalidation。
  - repo 实现可确认：`mem0/memory/main.py` 中 `_add_to_vector_store(...)` 先对每个新 fact 做 vector search，再用 `get_update_memory_messages(...)` 让 LLM 输出 `ADD / UPDATE / DELETE / NONE`，并分别调用 `_create_memory` / `_update_memory` / `_delete_memory`。
  - repo 实现可确认：`mem0/memory/graph_memory.py` 中 `_get_delete_entities_from_search_output(...)` + `_delete_entities(...)` 会把关系标成 `valid = false`。
  - 推断：这是当前 MemPrimitive 与 Mem0 之间最关键、也最集中的缺口。

#### retrieval

- 论文里做了什么
  - 基础 Mem0：以 embedding similarity 做 memory recall。
  - graph 版：同时有 entity-centric graph traversal 与 triplet semantic matching 两条 retrieval 路。
- MemPrimitive 现有哪些模块可直接复用
  - `EmbeddingSimilarityRetrieval`
    - 对基础版 user-facing recall 足够接近。
- 哪些模块只能部分复用
  - `VectorGraphSeedAndExpandRetrieval`
    - 能表达“图中找 seed 再扩展”的粗轮廓，但不等于 Mem0 graph 版的 entity-anchor traversal。
  - `BM25Retrieval`
    - 对 graph repo 的 BM25 relation rerank only 是很弱的局部近似，不是论文主干。
- 当前缺失什么能力
  - 如果要覆盖 graph 版，需要一个“entity-centric + triplet similarity”的混合图检索 primitive。
  - 当前 retrieval slot 也没有直接表达“ingest-time top-k similar memories for update evaluation”的位置；该能力更适合下沉到 `memory_evolution` 内部。
- 你的判断依据是什么
  - 论文明说：基础版用向量 memory database 做相似检索；graph 版同时有 entity-centric 与 semantic triplet retrieval。
  - repo 实现可确认：`search(...)` 主路径调用 `_search_vector_store(...)`，图检索并行返回 `relations`；`graph_memory.py` 中 `search(...)` 先抽 query entities，再在图上找相近节点与关系，最后用 BM25 做关系序列重排。
  - 推断：基础版 retrieval 可直接复用，graph 版只能部分映射。

#### readout

- 论文里做了什么
  - 基础版把相关 memories 交给下游对话模型作为补充上下文。
  - graph 版会额外返回关系上下文。
- MemPrimitive 现有哪些模块可直接复用
  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
- 哪些模块只能部分复用
  - `GraphReadout`
    - 对 graph 版调试可用，但不是 Mem0 默认的结果形态。
- 当前缺失什么能力
  - 基础版无关键缺口。
  - 如果希望忠实对齐 graph 版“向量命中 + relations 并行返回”的接口形式，可能需要一个更贴近 Mem0 API 的 readout 适配层，但这不是主机制缺口。
- 你的判断依据是什么
  - 论文明说：retrieved memories 被注入后续回答过程；graph 版提供额外 relational context。
  - repo 实现可确认：`search(...)` 返回 `{"results": original_memories, "relations": graph_entities}`。
  - 推断：readout 不构成 Mem0 的主要重表达障碍。

### 与 MemPrimitive 现有组件的对照结论

| slot | 结论 | 说明 |
| --- | --- | --- |
| `unit_formation` | 部分复用 | `PassThroughUnitFormation` 可承载预打包的 turn-pair，但缺原生 conversation pair / context bundling |
| `representation` | 只能部分复用 | 缺上下文感知 fact extraction；graph 版缺 typed entity + relation triplet extraction |
| `write_trigger` | 直接复用 | `AlwaysWriteTrigger` 足够表达“候选进入 update evaluation” |
| `organization` | 部分复用 | 基础 append 外壳可用，但 graph 版 typed entity-relation graph 仍缺明确承载 |
| `evolution_trigger` | 直接复用 | 可用 always-like 触发表达“每个 candidate fact 都进入维护” |
| `memory_evolution` | 当前缺失关键能力 | 缺 Mem0 核心的 similarity-resolved `ADD/UPDATE/DELETE/NONE` 维护 primitive |
| `retrieval` | 部分复用 | 基础向量 recall 可复用；graph 版 retrieval 只能局部映射 |
| `readout` | 直接复用 | 通用 memory 文本 readout 足以承载基础版输出 |

### 重表达判断

可大部分重表达。

更准确地说：

- **基础 Mem0** 已经很接近 MemPrimitive 当前能力边界；
- 真正阻碍“完整重表达”的，是它最核心的 update-resolution 没有现成 primitive；
- **graph-memory 增强版** 则进一步暴露出 typed graph organization、关系失效化、图检索混合策略这些新增缺口。

所以如果只看论文的基础 Mem0 主体，离完整重表达只差少量但关键的模块；如果把 graph-memory 变体也算进同一条目，则整体仍应保守写成“可大部分重表达”，而不是“可完整重表达”。

### 备注与证据边界

- 论文明说
  - Mem0 主体分 extraction 与 update 两阶段。
  - extraction 使用当前 message pair、conversation summary、recent messages。
  - update 对每个 candidate fact 先找 top-k 相似旧记忆，再做 `ADD / UPDATE / DELETE / NOOP`。
  - graph-memory 变体把记忆表示成 directed labeled graph，并对冲突关系做失效化。
  - graph 检索包含 entity-centric retrieval 与 semantic triplet retrieval。
- repo 实现可确认
  - `mem0/memory/main.py` 中 `_add_to_vector_store(...)` 先用 LLM 抽 `facts`，再对每个 fact 做向量检索，再让 LLM 输出 `ADD / UPDATE / DELETE / NONE`，最后分别执行 create/update/delete。
  - `mem0/memory/main.py` 的 `search(...)` 以 vector search 为主，graph 搜索并行返回在 `relations`。
  - `mem0/memory/graph_memory.py` 中实体抽取与关系抽取是两步；图更新支持把冲突边标记为 `valid = false`。
- 依据论文与 repo 做出的合理推断
  - 在 MemPrimitive 中，Mem0 的核心新增 primitive 最自然应落在 `representation` 与 `memory_evolution`，而不是 `write_trigger`。
  - 论文里的“ingest-time similarity retrieval”更像 `memory_evolution` 内部子过程，而不是当前 recall-side `retrieval` slot 的直接对应物。
- 当前证据不足、不能下结论的点
  - 论文宣称的 async conversation-summary module，在当前 OSS 主路径里没有看到同样清晰的独立实现边界；它可能是论文系统或平台侧能力，而不是当前开源代码中的同级模块。
  - 当前开源 repo 已混入平台化能力，如 `user_id/agent_id/run_id` 作用域、reranker、procedural memory、graph 并行返回等；这些不应全部反推为论文主文的核心 memory 机制。
## LightMem: Lightweight and Efficient Memory-Augmented Generation

论文链接: <https://openreview.net/forum?id=8QOO8Ufq2M>

官方 repo: <https://github.com/zjunlp/LightMem>

本次 repo 证据主要核对版本: `zjunlp/LightMem` `main` 分支，临时检查到的提交为 `ca39c30`。

### 论文侧 memory 机制速写

LightMem 的 memory 主机制不是“更复杂的召回器”，而是把长期记忆构建拆成三段：

- `Light1 / sensory memory`：新对话 turn 先经过预压缩，只保留信息密度更高的 token，再进入一个有 token 容量上限的感官缓冲区。
- `Light2 / topic-aware STM`：当感官缓冲区达到容量阈值后，系统基于 attention + 相邻 turn 语义相似度做 topic segmentation，把若干 turn 组织成 topic segment；随后把这些 segment 暂存进 STM，STM 到达阈值后再按 topic 粒度做总结，形成进入 LTM 的记忆条目。
- `Light3 / LTM with sleep-time update`：在线阶段对新 entry 先做 soft insert，不阻塞交互；更新被推迟到“sleep time”，先为每个 entry 构造一个带时间约束的相似候选更新队列，再离线并行执行 update/delete/ignore。

如果按 MemPrimitive 当前 slot 体系硬映射，最自然的主链路是：

- `unit_formation`：以增量对话 turn（通常是一轮 user/assistant 交换）作为进入 sensory memory 的基本输入单元。
- `representation`：先对 turn 做预压缩，再在 topic 粒度上形成可总结、可索引的文本表示；进入 LTM 的 entry 至少带 topic-aware summary 与 embedding。
- `write_trigger`：不是每条 turn 立即写入长期记忆，而是由 buffer capacity 触发分段与总结。
- `organization`：核心是 sensory buffer -> topic-aware STM buffer -> LTM 三层组织，而不是单层 append。
- `evolution_trigger`：长期记忆更新由 sleep-time / offline trigger 启动，而非每次在线写入时立即完成。
- `memory_evolution`：先按相似度与时间戳为每个 entry 构造 update queue，再并行决定 update/delete/ignore。
- `retrieval`：面向用户查询时主要是 embedding retrieval；论文并不把 retrieval 设计当作主要创新点。
- `readout`：把检索到的 memory 文本拼接返回给下游问答即可。

### 按 MemPrimitive slot 的拆解

#### unit_formation

- 论文里做了什么
  - 在实验设定里采用 incremental dialogue turn feeding；每次输入是一轮一轮到来的对话 turn。
  - 从机制上看，进入 sensory memory 的最小处理单元是 turn 级内容，而不是整段会话一次性写入。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `PassThroughUnitFormation`
    - 如果上游已经把一次 user/assistant 交互预先打包成一个 `Observation`，可以承载“turn 级输入单元”这一最小外观。
  - `MetadataHintUnitFormation`
    - 也可以靠上游 hints 人工构造多单元，但这不是 LightMem 原生的 turn ingestion 语义。
- 当前缺失什么能力
  - 缺少显式的“对话 turn / turn-pair 形成单元”能力边界。
  - 缺少与后续 sensory buffer 协作的 unit formation 约定；当前 unit_formation 不知道 buffer token 预算、speaker 对齐或 turn 粒度。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：采用 turn-level incremental feeding。
  - repo 实现可确认：`src/lightmem/memory/lightmem.py` 的 `add_memory(...)` 以新消息批进入；`sensory_memory.py` 里按 user/assistant 成对 turn 组织切段。
  - 依据论文与 repo 做出的合理推断：在 MemPrimitive 里，这一 slot 若要忠实表达 LightMem，至少需要把“turn 级输入单元”显式化，而不只是一般文本 passthrough。

#### representation

- 论文里做了什么
  - 先做 iterative pre-compression，保留更高信息密度 token。
  - 在 STM/LTM 侧，topic segment 会被总结成更紧凑的 summary 表示，并生成 embedding 供长期索引和检索。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `BasicRepresentation(elements=("summary", "embedding"))`
    - 能覆盖“生成摘要 + 生成向量”这一粗轮廓。
  - `BasicRepresentation(elements=("keywords", "entities", "tags"))`
    - 只能提供辅助元信息，不能表达 LightMem 的 token-level pre-compression。
- 当前缺失什么能力
  - 缺少“预压缩”这一 representation primitive；现有表示增强没有 token retention / entropy-style compression 语义。
  - 缺少“topic-aware segment summary”这一稳定表征边界；当前 summary 是 unit-local 的，不是 segment-local 的。
  - 缺少把“原始 turn 内容 + 压缩内容 + topic summary + summary embedding”打成同一类长期记忆 entry 表示的机制。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：Light1 先做 pre-compression；Light2 把 topic 结构总结后形成进入 LTM 的条目，条目包含 summary embedding。
  - repo 实现可确认：`lightmem.py` 里 `pre_compress=True` 时先调用 `compress(...)`；topic segment 进入 `meta_text_extract(...)` 后被转成 `MemoryEntry`，并在插入向量库时依赖 embedding。
  - 当前证据边界：repo 的 extraction prompt 更偏“事实抽取”，而论文正文在机制层更强调 topic-aware summarization；两者不能完全画等号。

#### write_trigger

- 论文里做了什么
  - 写入不是“每个 turn 都直接总结并入 LTM”。
  - 感官缓冲区达到容量阈值时触发 topic segmentation；STM 缓冲区再次达到阈值时触发 topic-level extraction / summarization 并形成长期记忆条目。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接复用模块。
- 哪些模块只能部分复用
  - `ThresholdWriteTrigger`
    - 只有“阈值触发”的外壳相似，但当前分数来源是常量/信号，不是 buffer occupancy 或 token budget。
  - `AlwaysWriteTrigger`
    - 只适合表达“entry 一旦形成就入库”，不适合表达 LightMem 的批量缓冲触发。
- 当前缺失什么能力
  - 缺少 buffer-capacity / token-budget 驱动的 write trigger。
  - 缺少多级触发：sensory -> STM 的分段触发，以及 STM -> LTM 的总结触发。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：sensory buffer 达到容量后触发 segmentation；STM 达到阈值后再做 summarization。
  - repo 实现可确认：`SenMemBufferManager.add_messages(...)` 与 `ShortMemBufferManager.add_segments(...)` 都是按 token 上限决定是否触发后续阶段。
  - 依据论文与 repo 做出的合理推断：LightMem 的写触发本质是“缓存阈值触发”，不是现有 MemPrimitive 触发族所覆盖的 metadata/key/常量阈值。

#### organization

- 论文里做了什么
  - 组织结构是三层的：sensory memory buffer、topic-aware STM、LTM。
  - STM 中间态不是简单 append，而是 `{topic, message turns}` 这样的 topic-segment 索引结构；LTM entry 则进一步变成 topic summary + raw turn support 的长期条目。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `AppendOrganization`
    - 只能表达“把条目写到某层”，不能表达层间缓冲与 topic segment 结构。
  - `ConditionalLayerOrganization`
    - 可做简单分层路由，但不能表达“先在 sensory/STM 暂存，后批量提升到 LTM”的生命周期。
- 当前缺失什么能力
  - 缺少显式的分层 buffer organization primitive。
  - 缺少 topic-segment 级数据结构及其与 LTM entry 的绑定关系。
  - 缺少“在线先 soft insert，后离线更新”的 LTM 生命周期组织边界。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：LightMem 由 sensory、STM、LTM 三模块组成；STM 维护 `{topic, message turns}`，LTM 存 `{topic, {sum_i, user_i, model_i}}`。
  - repo 实现可确认：`SenMemBufferManager`、`ShortMemBufferManager`、`LightMemory` 的向量 LTM 插入路径分别对应三层。
  - 依据论文与 repo 做出的合理推断：当前 MemPrimitive 的 layer 概念能表达“多层”，但不能直接表达“多层缓冲 + topic segment 生命周期”。

#### evolution_trigger

- 论文里做了什么
  - 长期记忆更新被延后到 sleep time。
  - 当所有 entries 插入完成或收到 update trigger 时，系统为 LTM entries 构造 update queue，并启动离线更新。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接复用模块。
- 哪些模块只能部分复用
  - `NewWriteEvolutionTrigger`
    - 有“新写入后触发维护”的轮廓，但它是局部在线维护，不是 LightMem 的离线批处理触发。
  - `ThresholdEvolutionTrigger`
    - 只能做抽象阈值触发，缺少 sleep-time / batch-update 语义。
- 当前缺失什么能力
  - 缺少显式的 sleep-time / offline batch evolution trigger。
  - 缺少“构造 update queue”和“执行离线更新”两个阶段之间的触发边界。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：更新与在线推理解耦，等所有 entries 插入完成或更新触发到来后，再计算 update queue。
  - repo 实现可确认：`construct_update_queue_all_entries(...)` 与 `offline_update_all_entries(...)` 是独立的离线步骤。
  - 依据论文与 repo 做出的合理推断：在 MemPrimitive 里，把这一路径强行塞进现有 online evolution trigger 会丢失 LightMem 的关键效率语义。

#### memory_evolution

- 论文里做了什么
  - 在线阶段只做 soft update，即新 entry 直接插入 LTM。
  - 离线阶段先为每个 entry 建立 top-k 相似候选队列，并加入时间约束，只允许较新的 entry 更新较旧的 entry。
  - 随后每个目标 entry 独立执行 update/delete/ignore，因此可以并行。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `SummaryRewriteEvolution`
    - 有“基于已有内容生成新版本”的表面相似性，但不具备 update queue 与时间约束。
  - `LayerMoveEvolution`
    - 能做记录改写/迁移，但没有“相似候选 + delete/update/ignore”决策。
  - `TraceOnlyEvolution`
    - 只能表达 no-op/记录痕迹，无法承载真实离线更新。
- 当前缺失什么能力
  - 缺少 timestamp-constrained similarity update queue 构造模块。
  - 缺少“target entry + candidate source entries -> update/delete/ignore”的离线维护 primitive。
  - 缺少并行 batch evolution 的能力边界。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：`Q(e_i)` 由相似度 top-k 与时间约束组成，且更新可并行执行。
  - repo 实现可确认：`construct_update_queue_all_entries(...)` 先为每个 entry 写入 `update_queue`；`offline_update_all_entries(...)` 再并行调用 LLM 决定 `update` 或 `delete`，否则跳过。
  - 当前证据不足、不能下结论的点：repo 当前 `UPDATE_PROMPT` 只显式暴露 `update/delete/ignore` 三类动作；论文正文若把“直接插入新 entry”也视作 update 体系的一部分，则 repo 的动作标签与论文抽象层级并不完全同构。

#### retrieval

- 论文里做了什么
  - 论文没有把 retrieval 设计当作主要创新点；方法贡献主要集中在 memory construction 与 sleep-time update。
  - 从系统使用角度看，LTM entry 通过 embedding semantic search 被召回。
- MemPrimitive 现有哪些模块可直接复用
  - `EmbeddingSimilarityRetrieval`
    - 足以表达“对长期记忆条目做向量相似检索”这一核心 recall 机制。
- 哪些模块只能部分复用
  - `LayerAwareRetrieval`
    - 如果以后想显式区分 LTM 与 summary store，可作为分层路由外壳，但不是论文必要条件。
- 当前缺失什么能力
  - 对 LightMem 论文主体而言，无关键缺口。
  - 若把 repo 后续的 summary store / StructMem 扩展也算进来，则需要额外的层间检索设计；但那不是这篇论文的核心记忆机制。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：效率分析主要对比 Summary/Update 成本，并明确说明 retrieval stage 不是其关注重点。
  - repo 实现可确认：`retrieve(...)` 直接对 query 做 embedding，然后查向量库返回 memory 文本。
  - 依据论文与 repo 做出的合理推断：在 MemPrimitive 中，LightMem 的 retrieval 可被现有 embedding retrieval 直接承载。

#### readout

- 论文里做了什么
  - 把检索出的 memory 条目文本交给下游问答或代理使用。
  - readout 不是论文创新点。
- MemPrimitive 现有哪些模块可直接复用
  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
- 哪些模块只能部分复用
  - `JSONReadout`
    - 只是在需要结构化下游接口时可选，不是论文必要。
- 当前缺失什么能力
  - 无关键缺口。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：LightMem 的重点在 memory bank construction 与 update efficiency，不在特殊 readout 格式。
  - repo 实现可确认：`retrieve(...)` 最终把结果格式化成字符串列表/拼接文本。
  - 依据论文与 repo 做出的合理推断：只要 retrieval 产出正确，现有通用 readout 即可承载 LightMem。

### 与 MemPrimitive 现有组件的对照结论

| slot | 结论 | 说明 |
| --- | --- | --- |
| `unit_formation` | 部分复用 | `PassThroughUnitFormation` 可承载预打包 turn，但缺原生 turn / turn-pair 形成语义 |
| `representation` | 部分复用 | 有 summary/embedding 外壳，但缺 pre-compression 与 topic-segment summary 表征 |
| `write_trigger` | 缺失 | 当前没有 buffer-capacity / token-budget 驱动触发器 |
| `organization` | 部分复用 | 有多层/append 外壳，但缺 sensory -> STM -> LTM 的分层缓冲组织 |
| `evolution_trigger` | 部分复用 | 可勉强表达“新写入后维护”，但缺 sleep-time batch trigger |
| `memory_evolution` | 缺失 | 当前没有 timestamp-constrained update queue + offline parallel update primitive |
| `retrieval` | 直接复用 | `EmbeddingSimilarityRetrieval` 足以表达论文主体 recall |
| `readout` | 直接复用 | 通用文本 readout 足够 |

### 重表达判断

只能部分映射。

原因不在于 slot 数量不够，而在于 LightMem 最关键的两段机制当前都没有真实落地承载物：

- 缓冲阈值驱动的三层 memory organization
- sleep-time 的 update queue 构造与离线并行更新

如果只用现有模块强行拼装，最多能做出“对话写入 + 摘要 + 向量检索”的近似版，但很难把 LightMem 的效率导向核心机制完整表达出来。

### 备注与证据边界

- 论文明说
  - LightMem 由 sensory memory、topic-aware STM、sleep-time updated LTM 三部分组成。
  - 新输入先经 pre-compression，再进入 sensory buffer。
  - sensory buffer 达到阈值后触发基于 attention + similarity 的 topic segmentation。
  - STM 达阈值后按 topic 粒度总结，形成进入 LTM 的条目。
  - LTM 更新先 soft insert，再在 sleep time 构造带时间约束的相似 update queue，并离线并行更新。
  - retrieval stage 不是论文主要优化对象。
- repo 实现可确认
  - `src/lightmem/factory/memory_buffer/sensory_memory.py` 实现了有 token 上限的 sensory buffer，以及 topic segmentation 后切段。
  - `src/lightmem/factory/topic_segmenter/llmlingua_2.py` 实现了 attention-based boundary proposal。
  - `src/lightmem/factory/memory_buffer/short_term_memory.py` 实现了 STM token 阈值触发。
  - `src/lightmem/memory/lightmem.py` 中 `construct_update_queue_all_entries(...)` 会为条目写入 `update_queue`；`offline_update_all_entries(...)` 会并行执行离线 update。
  - `retrieve(...)` 是直接的 embedding retrieval。
- 依据论文与 repo 做出的合理推断
  - 在 MemPrimitive 中，pre-compression 更适合作为 `representation`，而 topic-segment 生命周期更适合作为 `organization`。
  - sleep-time update 在 slot 上应拆成 `evolution_trigger` + `memory_evolution` 两部分，而不是塞进单一 organization side effect。
- 当前证据不足、不能下结论的点
  - repo 当前 extraction prompt 明显比论文正文更偏“事实抽取”；这是否等同于论文中的 topic summary 具体实现，证据不足。
  - repo 当前 `UPDATE_PROMPT` 暴露的动作是 `update/delete/ignore`；论文抽象层说的是 soft insert + offline update 框架。两者在动作标签上并非完全一一对应，不能过度等同。
  - repo 2026 年主分支已经包含 StructMem、baseline toolkit 等后续扩展；这些不应反推为 LightMem 原论文的核心 memory 机制。

## MIRIX: Multi-Agent Memory System for LLM-Based Agents

论文链接: <https://arxiv.org/abs/2507.07957>

官方 repo: <https://github.com/MIRIX-AI/MIRIX>

本次 repo 证据主要核对版本: `MIRIX-AI/MIRIX` 默认分支，临时检查时重点读取 `docs/ARCHITECTURE.md`、`mirix/functions/function_sets/memory_tools.py`、各 memory schema 与 manager 实现。

### 论文侧 memory 机制速写

MIRIX 的 memory 机制重点，不是“单一记忆库上再叠一层检索”，而是把长时记忆显式拆成多类、再交给多 agent 分工维护。论文与官方文档一致强调六类 memory：

- `Core Memory`：面向 persona / human profile 的高优先级块状记忆
- `Episodic Memory`：事件化经历
- `Semantic Memory`：关于实体、偏好、事实、关系的概念性知识
- `Procedural Memory`：可复用的操作步骤与工作流
- `Resource Memory`：文档、网页、文件等资源内容
- `Knowledge Vault`：带敏感级别的机密或受限信息

对应的写入链路是：

- 先由 `Meta Memory Manager Agent` 读取累积对话，判断这次交互应更新哪些 memory types
- 再由各 memory-type 专用 agent 产出对应 schema 的 memory item，并决定是插入、更新、合并还是跳过
- 最后由各 memory manager 持久化，并在需要时重算 embedding

对应的维护与读取链路是：

- `Reflexion Agent` 负责跨 memory types 的去重、整理、从 episode 中提炼更高层 pattern
- 查询时按 memory type 分别检索，再把不同类型的结果按类型分组注入 system prompt
- 检索并不是统一的单库 recall，而是多 store、字段感知、类型感知的 BM25 / embedding / fuzzy match 组合

如果硬映射到 MemPrimitive 当前 slot 体系，MIRIX 的核心新意主要压在四处：

- `write_trigger`：不是“写不写”二值，而是“一次交互要路由到哪些 memory types”
- `organization`：不是单一 layer append，而是异构多记忆库并存
- `memory_evolution`：不是简单追加，而是类型化 update / merge / rewrite / dedupe
- `retrieval/readout`：不是单一召回器，而是多 store 检索后再按 memory type 组织读出

### 按 MemPrimitive slot 的拆解

#### unit_formation

- 论文里做了什么
  - MIRIX 的上游输入并不是单句事实，而是“累积到当前时刻的一段交互上下文”。`Meta Memory Manager Agent` 会读取 accumulated messages / recent interaction，再决定哪些 memory types 需要更新。
  - 进入专用 memory agent 之前，最小可感知单元更像“conversation bundle / interaction chunk”，而不是纯文本句子切片。
- MemPrimitive 现有哪些模块可直接复用
  - `PassThroughUnitFormation`
    - 如果上游已经把一段对话上下文打包成单个 `Observation`，可以勉强承载 MIRIX 的输入外观。
- 哪些模块只能部分复用
  - `MetadataHintUnitFormation`
    - 可借助 hints 人工补上会话边界或来源标记，但它不原生表达“累积交互包”这一输入语义。
  - `WindowedUnitFormation`
    - 只能做局部窗口切片，不能稳定表达 MIRIX 那种面向 memory-router 的交互打包边界。
- 当前缺失什么能力
  - 缺少显式的“conversation bundle / interaction chunk”形成 primitive。
  - 缺少把多轮文本、附件引用、时间范围、参与者信息一起封装为后续 typed-memory extraction 输入的标准 contract。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：MIRIX 由 MetaAgent 根据交互内容决定更新哪些 memory types。
  - repo 实现可确认：`meta_memory_agent.txt` 直接要求 agent 从 accumulated messages 判断应更新的一组 memory types。
  - 依据论文与 repo 做出的合理推断：对 MemPrimitive 来说，这里的自然输入单元不是句子，而是“待路由的一段交互包”。

#### representation

- 论文里做了什么
  - MIRIX 的表示不是统一的 `summary + embedding`，而是先经由专用 agent 抽成不同 memory family 的 typed payload。
  - 例如 episodic memory 会形成事件摘要、细节、参与者、时间；semantic memory 会形成实体名、摘要、细节、来源；procedural memory 会形成步骤化流程；resource memory 会形成标题、摘要、正文；knowledge vault 会形成敏感度与密钥型内容。
  - 多数 memory types 在落库时还会补 embedding，但 embedding 只是 typed representation 的附属，不是主体。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `BasicRepresentation`
    - 可提供 `summary`、`embedding`、`keywords` 等通用增强，但不能产出 MIRIX 所需的多 schema typed memory payload。
  - `ConfigurableEmbeddingRepresentation`
    - 可覆盖“表示后附 embedding 供检索”的局部需求，但不负责把交互抽成 episodic / semantic / procedural / resource / vault 这些不同结构。
  - 多个 `LLMRepresentation`（例如 `context` / `keywords` / `tags` / `category` / `attributes`）
  - `ConfigurableEmbeddingRepresentation`
    - 只是在统一记录上补语义字段，不是按 memory family 分化 schema。
- 当前缺失什么能力
  - 缺少“一次交互 -> 多种 memory schema 候选”的 typed extraction primitive。
  - 缺少对 block memory、event memory、concept memory、procedure memory、resource memory、sensitive secret memory 的统一表示边界。
  - 缺少“不同 memory type 使用不同字段集与后续索引字段”的 representation contract。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：MIRIX 使用多类 memory，并由不同 memory agents 管理。
  - repo 实现可确认：各 schema 文件分别定义了 episodic / semantic / procedural / resource / knowledge vault 的字段结构，且工具层存在分 memory type 的 insert/update 接口。
  - 依据论文与 repo 做出的合理推断：MIRIX 在 slot 上更接近“typed schema extraction representation”，而不是单一通用表示增强。

#### write_trigger

- 论文里做了什么
  - MIRIX 的写触发核心不是简单判断“写不写”，而是由 MetaAgent 对当前交互做多标签路由，决定本次应更新哪些 memory types。
  - 当某个 memory type 被选中后，还会进入该类型专用 agent 的后续判断与更新。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `LLMJudgedWriteTrigger`
    - 可以借 LLM 做“是否值得写”判断，但它没有 MIRIX 所需的 one-to-many memory-type routing 输出。
  - `MetadataGatedWriteTrigger`
    - 如果上游已写好目标 memory types，可按 metadata 过滤；但 MIRIX 自身关键在于触发器内部产出路由结果，而不是消费既有标签。
  - `AlwaysWriteTrigger`
    - 只能表达“全部送入后续流程”，无法表达“只更新 episodic + semantic，不更新 resource / vault”。
- 当前缺失什么能力
  - 缺少 multi-label、type-selective 的 write trigger。
  - 缺少“当前交互 -> 目标 memory type 集合”的触发结果格式。
  - 缺少把路由决策继续传给各 store / agent 的显式控制边界。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：Meta Memory Manager Agent 决定 memory update 的类型分发。
  - repo 实现可确认：`trigger_memory_update` 会先收集 memory types，再并行触发对应子 agent。
  - 依据论文与 repo 做出的合理推断：在 MemPrimitive slot 里，这应落在 `write_trigger`，且现有触发器只覆盖“是否写”，未覆盖“写到哪些 type”。

#### organization

- 论文里做了什么
  - MIRIX 不是单一 memory table，而是六类异构 memory store 并存。
  - 其中 `Core Memory` 还是块状 memory block 管理；其他几类更像 typed record store，但字段、索引域、可更新方式都不同。
  - 不同 memory 还带有 scope / owner / sensitivity 等过滤维度，尤其 knowledge vault 有额外访问边界。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `ConditionalLayerOrganization`
    - 可近似表达“按类别分层”，但无法表达 MIRIX 这种 heterogeneous stores with distinct schemas 的组织方式。
  - `AppendOrganization`
    - 只能承载单一 record append，不足以表示 block memory 与 typed stores 并存。
  - `PlacementWithoutAppendOrganization`
    - 可表达路由/落位外观，但没有 MIRIX 所需的实际异构存储语义。
- 当前缺失什么能力
  - 缺少“异构多记忆库组织” primitive。
  - 缺少 block-style core memory 与 record-style typed memories 并存的组织抽象。
  - 缺少 per-type schema、scope、sensitivity、owner constraints 的组织边界。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：MIRIX 采用多种 memory types。
  - repo 实现可确认：`ARCHITECTURE.md`、schemas、memory managers 都显示六种 memory store 分离；core memory 通过 blocks 管理，knowledge vault 带 sensitivity。
  - 依据论文与 repo 做出的合理推断：MemPrimitive 当前 layer 概念只能部分近似 MIRIX 的 store 分化，不能忠实承载其异构组织。

#### evolution_trigger

- 论文里做了什么
  - MIRIX 的维护触发不是单一模式，而是至少有三类：
  - `Core Memory` 在接近容量上限时触发 rewrite / condense。
  - 各 typed memories 在检测到重复、重叠或同一对象时触发 update / merge / replace。
  - `Reflexion Agent` 还会在额外时机对所有 memory types 做去重、整理与高层 pattern 提炼。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `ThresholdEvolutionTrigger`
    - 可近似表达“达到阈值后触发”，但现有触发信号不是 core block fullness。
  - `NewWriteEvolutionTrigger`
    - 可表达“新写入后做维护”，但 MIRIX 中很多维护是 type-aware update / merge，而非统一后处理。
  - `OutcomeConditionedEvolutionTrigger`
    - 对 Reflexion 类触发有一点外形相似，但 MIRIX Reflexion 不是试错反馈触发，而是面向 memory hygiene 的整理触发。
- 当前缺失什么能力
  - 缺少容量驱动的 core-block rewrite trigger。
  - 缺少类型化 memory update / merge trigger。
  - 缺少 scheduled / per-query reflexion-style maintenance trigger。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：系统包含 Reflexion agent，并强调多 memory maintenance。
  - repo 实现可确认：core tools 里存在 rewrite block；各 memory tools 存在 insert/update/merge；`reflexion_agent.txt` 明确写了 cleanup / dedup / pattern extraction。
  - 依据论文与 repo 做出的合理推断：MIRIX 的 evolution trigger 明显是多源触发族，而不是当前单一阈值或单一新写入触发可覆盖。

#### memory_evolution

- 论文里做了什么
  - MIRIX 的 memory evolution 包括多类 type-specific 维护动作：
  - core memory 会做 block rewrite / condense。
  - episodic memory 会 merge / replace 事件条目。
  - semantic / procedural / resource / knowledge vault 会对已有条目做 update，且会跳过重复项。
  - Reflexion 还会做跨 memory types 的 dedupe、整理，以及从 episodic 中提炼 lifestyle / behavior pattern 写回更高层 memory。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `SummaryRewriteEvolution`
    - 与 core block rewrite 有表面相似，但它不是 block-capacity-conditioned rewrite，也不面向 persona/human blocks。
  - `ReflectionGenerationEvolution`
    - 对 Reflexion 式“从经历生成更高层洞见”有局部相似，但不支持跨 store dedupe 与 typed write-back。
  - `AppendOnlyEvolution`
    - 只能表达新增，无法表达 MIRIX 的 update / merge / replace / dedupe 主体。
- 当前缺失什么能力
  - 缺少 typed memory update / merge / replace primitive。
  - 缺少跨 memory stores 的 deduplication / cleanup primitive。
  - 缺少“从 episodic pattern 提炼 semantic insight 并写回”的跨类型演化能力。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：MIRIX 通过多类 memory 协同获得更强长期记忆管理。
  - repo 实现可确认：memory tools 提供分类型 insert/update/merge/check；Reflexion prompt 明确要求 dedup、cleanup、pattern extraction。
  - 依据论文与 repo 做出的合理推断：MIRIX 的核心维护语义落在 typed evolution，而不是当前 append/rewrite/graph-link 家族能完整覆盖的范围。

#### retrieval

- 论文里做了什么
  - MIRIX 查询时不是单一向量召回，而是按 memory type 分别检索。
  - episodic memory 会同时区分 recent events 与 relevant events。
  - 其他 memory types 使用字段感知的 BM25 / embedding / fuzzy matching；knowledge vault 还带敏感级别过滤，避免不该暴露给当前 agent 的秘密被读出。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `BM25Retrieval`
    - 可覆盖 MIRIX 检索中的一条主路径，但只面向统一记录语义。
  - `EmbeddingSimilarityRetrieval`
    - 可覆盖向量检索部分，但不表达 per-type fields、recent-vs-relevant dual view 与 sensitivity constraints。
  - `LayerAwareRetrieval`
    - 可做多层路由外壳，但离 MIRIX 的多 store、字段感知检索 orchestration 仍有明显距离。
  - `RecencyRetrieval`
    - 仅能覆盖 episodic recent branch 的一部分。
- 当前缺失什么能力
  - 缺少跨异构 stores 的联合检索 orchestrator。
  - 缺少 per-type searchable fields、hybrid search policy 与 sensitivity filtering。
  - 缺少“recent episodic + relevant episodic + other typed memories”并行召回的统一接口。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：MIRIX 用多类 memory 提供更完整上下文。
  - repo 实现可确认：各 manager 支持 BM25 / embedding / fuzzy 等不同检索；agent 构 prompt 时分别拉取 episodic recent / relevant 与其他 memory types；knowledge vault 会限制敏感内容暴露。
  - 依据论文与 repo 做出的合理推断：当前 MemPrimitive 的 retrieval family 可复用部分局部检索器，但缺少 MIRIX 风格的多 store orchestrator。

#### readout

- 论文里做了什么
  - MIRIX 的读出不是简单拼接全部命中文本，而是按 memory type 分组装配到 system prompt。
  - episodic memory 会分 recent 与 relevant 两块呈现；其他 memory 则按类型列出；部分场景还会保留 item id 供后续更新。
  - knowledge vault 的读出受可见性与敏感度约束。
- MemPrimitive 现有哪些模块可直接复用
  - 无可直接完整复用模块。
- 哪些模块只能部分复用
  - `GroupedByLayerReadout`
    - 可近似表达“分组呈现”，但组的语义只是 layer，不是 MIRIX 的 typed memory families。
  - `PromptContextReadout`
    - 可承载最终 prompt 注入外观，但不自带 MIRIX 那种 recent/relevant split 与敏感信息抑制。
  - `ConcatenateReadout`
    - 只能做最粗粒度拼接。
- 当前缺失什么能力
  - 缺少 typed memory prompt assembly primitive。
  - 缺少 episodic recent/relevant 双区块 readout。
  - 缺少与 sensitivity / owner filtering 协同的 readout contract。
- 你的判断依据是什么（论文 / repo / 推断）
  - 论文明说：系统在推理时调用不同 memory 以增强上下文。
  - repo 实现可确认：`build_system_prompt_with_memories` 会按 memory type 构造不同片段，并对 knowledge vault 做限制。
  - 依据论文与 repo 做出的合理推断：对 MemPrimitive 而言，readout 不是零难点，但它更像 retrieval 之后的 typed formatting 缺口。

### 与 MemPrimitive 现有组件的对照结论

| slot | 结论 | 说明 |
| --- | --- | --- |
| `unit_formation` | 部分复用 | `PassThroughUnitFormation` 可勉强承载预打包交互，但缺 conversation bundle contract |
| `representation` | 缺失 | 当前没有“一次交互 -> 多 memory schema typed payload”模块 |
| `write_trigger` | 缺失 | 当前没有多标签 memory-type routing trigger |
| `organization` | 部分复用 | 可用 layer/router 外壳近似，但缺异构多 store 与 core-block 组织 |
| `evolution_trigger` | 部分复用 | 有阈值/新写入/结果驱动触发外壳，但缺 capacity-triggered rewrite 与 reflexion scheduling |
| `memory_evolution` | 缺失 | 当前没有 typed update/merge/replace/dedupe 跨 store 维护 primitive |
| `retrieval` | 部分复用 | BM25/embedding/recency 可局部复用，但缺多 store、字段感知、敏感度约束 orchestrator |
| `readout` | 部分复用 | 现有 prompt/group readout 可做近似，但缺 typed prompt assembly 与 episodic dual-view readout |

### 重表达判断

只能部分映射。

原因不在于 MIRIX 无法放进八 slot 框架，而在于它的关键机制并不是单点缺口，而是整条“多类型路由 -> 异构组织 -> 类型化维护 -> 多 store 检索/读出”链条都比当前模块族更强：

- 前半段缺 multi-memory-type routing trigger
- 中段缺 heterogeneous memory-store organization
- 后半段缺 typed update / dedupe / cross-store reflexion evolution
- recall 侧虽能复用部分 BM25 / embedding / recency primitive，但离 MIRIX 的 typed orchestration 仍有明显差距

因此，当前 MemPrimitive 可以表达 MIRIX 的一些局部思想，例如：

- 用 layer-aware 路由近似多 memory categories
- 用 embedding / BM25 recall 近似若干单 store 检索
- 用 summary/rewrite/reflection 局部近似少数维护动作

但还不能把 MIRIX 忠实重表达成一个“只靠现有模块组合即可成立”的完整系统。

### 备注与证据边界

- 论文明说
  - MIRIX 采用 multi-agent memory architecture。
  - 系统维护六类 memory，并用专门 agent 协同管理。
  - 系统还包含 Reflexion 机制以提升长期记忆管理质量。
- repo 实现可确认
  - `ARCHITECTURE.md` 明确列出六类 memory 与 MetaAgent / Reflexion Agent。
  - 各 schema 与 manager 明确显示 episodic / semantic / procedural / resource / knowledge vault 的真实字段、索引与检索方法。
  - `memory_tools.py` 明确存在按 memory type 的 insert / update / merge / rewrite / dedupe 相关工具入口。
  - `agent.py` 中的 prompt 构造逻辑确实按 memory type 读取，并把 episodic recent / relevant 分开呈现。
- 依据论文与 repo 做出的合理推断
  - 在 MemPrimitive slot 映射中，MetaAgent 的“决定写入哪些 memory types”最自然落在 `write_trigger` 而不是 `organization`。
  - MIRIX 的主要缺口不在单个 retriever，而在 typed routing 与 heterogeneous organization。
  - Reflexion 在 slot 上更适合作为 `evolution_trigger + memory_evolution` 的组合，而不是单独新增顶层 slot。
- 当前证据不足、不能下结论的点
  - 论文正文与 repo 文档都没有把“交互包的精确定义”形式化到可直接等价为某个 unit schema，故 `unit_formation` 的细粒度边界仍有解释空间。
  - repo 某些检索路径当前默认更偏 BM25，而论文层面的 memory 使用描述更高层；不能据此反推“论文只主张 BM25”。
  - Reflexion 的真实生产触发频率、调度策略、与在线推理的严格时序边界，在当前公开材料中证据不够充分，不能写死为唯一机制。
