# Motif 到 DSL 层的放置指南

本文档服务于**未来实现**。

目标不是重述某篇 paper，而是回答两个更工程化的问题：

1. 一个 motif 应该进入哪一层 DSL。
2. 如果要把它真正做成可组合 primitive，它需要实现什么功能、依赖什么输入与结构前提。

这里默认沿用 `README.md` 中已经确立的 design space 分层：

- 结构层（store topology / layer shape / index prior）
- 模块层（slot-level primitive choices）
- 参数层（同一家族内部的行为差异）
- 约束层（哪些组合合法，哪些只是 paper residual）

---

## 0. 使用原则

把 motif 放进哪一层，不取决于它“看起来有多重要”，而取决于它改变的对象是什么：

- 如果它改变的是 **memory store 的骨架**，它属于结构层。
- 如果它改变的是 **某个 slot 内执行什么机制**，它属于模块层。
- 如果它改变的是 **同一种机制的力度、预算、阈值、策略开关**，它属于参数层。
- 如果它表达的是 **兼容性、前置能力、语义耦合**，它属于约束层。

一个实践上的判断标准是：

- “换掉它，会不会要求你改 store 形状？”如果会，优先放结构层。
- “换掉它，只是同一 slot 里换一个实现？”如果会，优先放模块层。
- “换掉它，只是 top-k / threshold / budget 不同？”如果会，优先放参数层。
- “它不是功能本身，而是在说什么能和什么配？”如果会，优先放约束层。

---

## 1. 结构层 Motif

结构层只放那些会决定 memory store 骨架的 motif。它们不是某个 slot 的局部小技巧，而是全局 structural prior。

### 1.1 Layer Role Partitioning

- 典型来源：
  - MemGPT 的 role-partitioned multi-layer store
  - Reflexion 的 `trial_buffer` / `reflections`
  - TiM 的 `thought_memory`
  - A-MEM 的单图层 `memory_graph`
- 实现功能：
  - 预先声明不同 layer 承担的 memory role。
  - 让 organization / retrieval / evolution 有明确目标层。
- 依赖：
  - `StoreTopology`
  - 明确 layer name / theme
- 适合编码成 DSL 的部分：
  - layer 数量
  - layer 名称
  - layer theme
  - layer 间角色分工
- 不该放在这里的内容：
  - 某个 layer 内具体如何检索
  - 某个 layer 的 prompt 格式

### 1.2 Graph Memory Topology

- 典型来源：
  - A-MEM
- 实现功能：
  - 声明某 layer 为 `Graph`，允许 record 间显式 link。
  - 为 graph-based evolution / retrieval 提供载体。
- 依赖：
  - `shape="Graph"`
  - `index:graph`
- 适合编码成 DSL 的部分：
  - graph layer 是否存在
  - graph layer 名称
  - graph index 是否启用
- 典型下游模块依赖：
  - graph organization
  - neighbor expansion retrieval
  - graph link evolution

### 1.3 Partition-Addressable Flat Memory

- 典型来源：
  - TiM bucketed thought layer
  - MemGPT keyed block memory
- 实现功能：
  - 虽然底层仍是 flat layer，但 record 可被某种 partition key 稳定寻址。
  - 让后续模块能做 local maintenance 或 keyed overwrite。
- 依赖：
  - 某种 record-level key
  - 该 key 可由 representation 或 metadata 生成
- 适合编码成 DSL 的部分：
  - 是否要求可分区寻址
  - 使用哪类 partition key：`hash` / `entity` / `slot_key`
- 说明：
  - 这不是 retrieval 策略本身。
  - 它是让“局部更新”和“局部检索”成为可能的结构前提。

### 1.4 Required Index Family

- 典型来源：
  - A-MEM 需要 `graph` + `vector`
  - MemGPT archival search 需要 `vector`
  - tag / keyword augmentation 需要 `keyword` / `tag`
- 实现功能：
  - 声明哪些 index 能力是某类 motif 的前置条件。
- 依赖：
  - `StoreLayerSpec.indices`
- 适合编码成 DSL 的部分：
  - `vector`
  - `graph`
  - `keyword`
  - `tag`
  - `temporal`

---

## 2. 模块层 Motif

模块层是 motif 的主战场。这里按 slot 分类，因为未来实现最可能就是落回现有 `MemoryPipeline` 的 slot 接口。

### 2.1 Unit Formation 层

#### Motif: Structured Extraction

- 典型来源：
  - TiM post-thought extraction
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L367)
- 实现功能：
  - 把一个 observation 拆成更高价值的 memory units。
  - 常见输出不是 raw turn，而是 thought / fact / relation-aware unit。
- 依赖：
  - `observation.text`
  - 可选的 observation metadata hint
  - 如需 LLM 抽取，则依赖 classic runtime LLM
- 未来实现建议：
  - 把“抽取什么粒度”做成 family
  - 把具体 schema 放到 metadata namespace 中

#### Motif: Provenance Attachment

- 典型来源：
  - TiM unit 里保留 `observation_id` / `source`
- 实现功能：
  - 给所有派生 unit 附加来源信息，便于后续 evolution 与 trace。
- 依赖：
  - `observation.observation_id`
  - `observation.source`
- 说明：
  - 这更像一个可复用子 motif，可内嵌到多个 unit formation 家族。

### 2.2 Representation 层

#### Motif: Semantic Field Enrichment

- 典型来源：
  - A-MEM comprehensive note
  - TiM summary / keywords / triple metadata
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L427)
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L279)
- 实现功能：
  - 除原始 text 外，再生成可供组织/检索/读出的结构化字段。
- 依赖：
  - `units`
  - 可选 LLM
  - 可选 embedder
- 推荐输出：
  - `summary`
  - `keywords`
  - `tags`
  - `context`
  - `attributes`

#### Motif: Partition Key Derivation

- 典型来源：
  - TiM `hash_index` / `group_id`
  - MemGPT `memgpt_key` 的上游生成逻辑
- 实现功能：
  - 为后续 organization / retrieval / maintenance 生成稳定分区键。
- 依赖：
  - text 或 enriched representation
  - embedder 或规则函数
- 说明：
  - 这是很值得独立命名的 motif，因为它解释了“为什么后续能 localize”。

#### Motif: Retrieval-Oriented Embedding Construction

- 典型来源：
  - A-MEM 的 enhanced embedding text
- 实现功能：
  - 不是直接 embed 原始文本，而是 embed 一个为检索优化的复合表示。
- 依赖：
  - text
  - context
  - keywords / tags
  - embedder
- 适合实现为：
  - `embedding_source = raw_text | enriched_text | structured_projection`

### 2.3 Write Trigger 层

在当前代码里，`write_trigger` 与 `evolution_trigger` 已经有一个很好的实现容器：

- `signal_providers: tuple[SignalProvider, ...]`
- `scorer: ScoreAggregator`
- `gate: Gate`
- `policy: DecisionPolicy`

未来如果要把 classic motif 真正对齐到 `memprimitive/baselines/_trigger_family.py`，建议都先投影到这四件套。

这里的运行顺序固定为：

1. `signal_providers` 产出 per-unit signals
2. `scorer` 把 signals 聚合成一个 score
3. `gate` 做 hard constraint 判断
4. `policy` 用 `score + gate_open` 产出最终布尔决策

下面的分解里：

- 若某篇 paper / repo 已直接体现该机制，我直接按原机制抽象。
- 若原文没有显式拆成四件套，我会按论文语义**亲自做结构化分解**，并明确写出这是 inferred decomposition。

#### Motif: Metadata-Gated Write

- 典型来源：
  - TiM 的 `metadata.tim.write`
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L491)
- 实现功能：
  - 基于 upstream metadata 决定是否写入。
- 依赖：
  - `units`
  - 某个 metadata flag
- 适合场景：
  - 上游已经做过抽取与筛选
  - write policy 本身不想再引入复杂判断
- 四件套分解：
  - `signal_providers`
    - `UnitTypeSignal(tim_thought)`
    - `MetadataFlagSignal(path="metadata.tim.write", default=True)`
  - `scorer`
    - `MinScorer("is_tim_thought", "write_flag")`
  - `gate`
    - `AlwaysOpenGate`
  - `policy`
    - `ThresholdPolicy(threshold=1.0)` 或 `BooleanGatePolicy` 配合布尔化 scorer
- 如何拼接出当前 TiM write trigger：
  - 当前实现实际上把 “是 `tim_thought` 且 write 不为 false” 直接写在一个判定里。
  - 若改写成 trigger family，可等价表示为：
    1. 先生成 `is_tim_thought` 与 `write_flag`
    2. 用 `scorer` 把两者合成“是否满足 TiM 写入前提”
    3. `gate` 不额外拦截
    4. `policy` 直接把结果转成 decision
- 说明：
  - 这一步分解主要来自当前实现与 TiM 论文的 post-think write 语义，不是论文原文显式给出的组件边界。

#### Motif: LLM-Judged Write

- 典型来源：
  - A-MEM write controller
- 代表实现：
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L371)
- 实现功能：
  - 让 LLM 依据内容、上下文和标签判断是否值得写入。
- 依赖：
  - `units`
  - classic runtime LLM
  - 足够丰富的 unit metadata
- 风险：
  - 成本高
  - 行为漂移大
  - 强依赖 prompt
- 四件套分解：
  - `signal_providers`
    - `ContentRichnessSignal`
    - `TagDensitySignal`
    - `KeywordDensitySignal`
    - `NoveltyHintSignal`
    - 或一个合并版 `LLMWriteAssessmentSignal`，直接返回 `store_likelihood`
  - `scorer`
    - `WeightedSumScorer`
    - 或 `IdentityScorer(source="store_likelihood")`
  - `gate`
    - `SchemaPresentGate(amem note fields ready)`
    - 可选 `MinContentGate`
  - `policy`
    - `ThresholdPolicy`
- 如何拼接出当前 A-MEM write trigger：
  - 当前代码把 LLM 返回的 `decision / reason / confidence` 直接视为最终写入结果。
  - 若改写成 trigger family，我建议把 LLM 的输出拆成：
    1. `signal_providers` 先提供 note content、tags、keywords 等显式信号
    2. 再用一个 agentic `SignalProvider` 或 scorer-side helper 生成 `store_likelihood`
    3. `gate` 保证 A-MEM note schema 已经存在
    4. `policy` 用阈值把 `confidence` 或 `store_likelihood` 转成最终写入决策
- 说明：
  - A-MEM 论文与官方仓库明确支持“agent-driven decision making for adaptive memory management”和“生成 note 后再做 memory organization / evolution”。
  - 但它们没有把 write controller 明确拆成四件套，因此上面的边界属于 paper-grounded inference。

### 2.4 Organization 层

#### Motif: Append to Layer

- 典型来源：
  - TiM thought append
  - A-MEM note append
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L571)
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L441)
- 实现功能：
  - 把写入 unit 放到目标层。
- 依赖：
  - target layer 存在
  - `units` 与 `decisions` 对齐

#### Motif: Keyed Upsert

- 典型来源：
  - MemGPT core / working block 更新
- 代表实现：
  - [memgpt.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/memgpt.py#L139)
- 实现功能：
  - 若 key 不存在则插入，存在则覆盖更新。
- 依赖：
  - 目标 layer
  - `unit.metadata[key_name]`
  - key 可稳定复用
- 特别适合：
  - persona
  - profile
  - working summary
  - entity card

#### Motif: Graph Append with Link-Ready Metadata

- 典型来源：
  - A-MEM graph organization
- 实现功能：
  - 写入 graph layer，并初始化 link-ready metadata。
- 依赖：
  - graph layer
  - `index:graph`
  - `index:vector`
- 说明：
  - 它不是“已经做了 graph evolution”。
  - 它只是让 graph evolution 之后有地方写回 links。

#### Motif: Placement Without Append

- 典型来源：
  - Reflexion trial organization
- 代表实现：
  - [reflexion.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py#L223)
- 实现功能：
  - 给 packet 发 placement，但不把 trial 本身持久化到 store。
- 依赖：
  - `units`
  - `decisions`
- 价值：
  - 明确区分“当前 trial 轨迹”和“值得沉淀的 memory”。

### 2.5 Evolution Trigger 层

这一层最适合被系统化改写成 trigger family。

原因是：

- 它们本质上都在回答“额外演化要不要触发”
- 只是不同方法依赖的 signal 来源不同
- 这些差异很自然就能映射到 `signal -> score -> gate -> policy`

#### Motif: Outcome-Conditioned Trigger

- 典型来源：
  - Reflexion failure-only trigger
- 代表实现：
  - [reflexion.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py#L255)
- 实现功能：
  - 只有当外部反馈显示失败时，才触发额外 memory evolution。
- 依赖：
  - `observation.metadata`
  - task outcome / feedback schema
- 四件套分解：
  - `signal_providers`
    - `OutcomeCorrectnessSignal`
    - `FeedbackPresenceSignal`
    - 可选 `TrialIndexSignal`
  - `scorer`
    - `IdentityScorer(source="trial_failed")`
    - 或 `WeightedSumScorer({"trial_failed": 1.0, "feedback_present": 0.1})`
  - `gate`
    - `FeedbackSchemaGate`
  - `policy`
    - `ThresholdPolicy(threshold=1.0)` 或 `BooleanGatePolicy`
- 如何拼接出当前 Reflexion evolution trigger：
  - 当前实现实际上是：
    1. 从 `observation.metadata` 推断 `is_correct`
    2. `should_reflect = not is_correct`
    3. 对每个 unit 复制同一 decision
  - 若改写成 trigger family，可表示为：
    1. `OutcomeCorrectnessSignal` 生成 `trial_failed`
    2. `scorer` 直接读取这个信号
    3. `gate` 检查 metadata 里至少有可解析的 feedback/outcome
    4. `policy` 将 “失败” 变为 `True`
- 来源依据：
  - Reflexion 论文明确说 agent 会 “reflect on task feedback signals” 并将 reflective text 放入 episodic buffer。
  - 官方仓库 README 还明确区分了 `LAST_ATTEMPT` / `REFLEXION` / `LAST_ATTEMPT_AND_REFLEXION` 这些使用 memory 的策略。

#### Motif: New-Write Trigger

- 典型来源：
  - TiM 只要新 thought 写入就触发后续维护
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L528)
- 实现功能：
  - 新 memory 进入后立刻触发局部维护。
- 依赖：
  - `units`
  - `placements`
  - unit family identity
- 四件套分解：
  - `signal_providers`
    - `UnitTypeSignal(tim_thought)`
    - `PlacementExistsSignal`
    - `PartitionKeyPresentSignal(group_id/hash_index)`
  - `scorer`
    - `MinScorer("is_tim_thought", "has_placement", "has_partition_key")`
  - `gate`
    - `LayerAllowedGate(allowed_layers=("thought_memory",))`
  - `policy`
    - `ThresholdPolicy(threshold=1.0)`
- 如何拼接出当前 TiM evolution trigger：
  - 当前实现等价于“只要 unit 是新写入的 TiM thought，就触发 evolution”。
  - 若映射到 trigger family：
    1. `signal_providers` 收集 `is_tim_thought`、是否已放入 thought layer、是否有 bucket key
    2. `scorer` 聚合成“是否满足本地维护前提”
    3. `gate` 限制只在 TiM thought layer 上生效
    4. `policy` 一旦条件满足就触发
- 说明：
  - TiM 论文没有把“触发器”单列成模块，它只说 post-thinking 之后用 insert / forget / merge 更新 memory。
  - 因此这里是根据论文里的 update loop 和当前实现，做出的 inferred decomposition。

#### Motif: Neighbor-Exists Trigger

- 典型来源：
  - A-MEM 只有附近已有相似记忆时才进 evolution
- 代表实现：
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L518)
- 实现功能：
  - 只有当新写入 record 周围存在候选邻居时，才触发 graph-level rewrite/linking。
- 依赖：
  - embeddings
  - vector index
  - candidate retrieval
- 四件套分解：
  - `signal_providers`
    - `NeighborCountSignal(top_k=candidate_k)`
    - `TopNeighborSimilaritySignal`
    - 可选 `GraphConnectivitySignal`
  - `scorer`
    - `WeightedSumScorer({"neighbor_count": 1.0, "top_neighbor_similarity": alpha})`
    - 或最简 `IdentityScorer(source="neighbor_exists")`
  - `gate`
    - `HasEmbeddingGate`
    - `VectorIndexReadyGate`
    - `GraphLayerGate`
  - `policy`
    - `ThresholdPolicy(threshold=1.0)` 或 `BooleanGatePolicy`
- 如何拼接出当前 A-MEM evolution trigger：
  - 当前实现是：
    1. 以新 unit embedding 做 candidate retrieval
    2. 只要存在近邻，就把 evolution decision 设为 `True`
  - 若改写成 trigger family：
    1. `NeighborCountSignal` 负责从 vector index 统计候选数
    2. `scorer` 把候选存在性或相似度压成一个分数
    3. `gate` 保证 graph/vector 能力存在
    4. `policy` 在“有邻居”时触发 evolution
- 来源依据：
  - A-MEM 论文明确说新记忆加入后，系统会分析历史记忆、识别 relevant connections，并在 meaningful similarities 存在时建立 links 与更新旧记忆。
  - 论文附录 prompt 还把 “Should this memory be evolved?” 建立在新记忆与 several nearest neighbors 上。

### 2.5.1 Trigger Family 映射建议

如果未来你要把 classic trigger 系统性迁移进 `memprimitive/baselines/_trigger_family.py`，我建议先补一批通用 signal / gate：

- signal providers:
  - `MetadataFlagSignal`
  - `UnitTypeSignal`
  - `OutcomeCorrectnessSignal`
  - `FeedbackPresenceSignal`
  - `PlacementExistsSignal`
  - `PartitionKeyPresentSignal`
  - `NeighborCountSignal`
  - `TopNeighborSimilaritySignal`
- scorers:
  - 现有 `IdentityScorer`
  - 现有 `WeightedSumScorer`
  - 现有 `MinScorer`
- gates:
  - `FeedbackSchemaGate`
  - `HasEmbeddingGate`
  - `VectorIndexReadyGate`
  - `GraphLayerGate`
  - `SchemaPresentGate`
- policies:
  - 现有 `ThresholdPolicy`
  - 现有 `BooleanGatePolicy`

这会让下面三类 classic trigger 都能以统一骨架实现：

- Reflexion: `feedback-derived signal -> failure score -> feedback/schema gate -> trigger`
- TiM: `new-thought signal -> local-maintenance readiness score -> thought-layer gate -> trigger`
- A-MEM: `neighbor-availability signal -> connectivity score -> graph/vector gate -> trigger`

### 2.6 Memory Evolution 层

#### Motif: Local Conflict Forgetting

- 典型来源：
  - TiM bucket 内 forget conflicts
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L642)
- 实现功能：
  - 只在局部 partition 内删除矛盾或被覆盖的旧记忆。
- 依赖：
  - stable partition key
  - 能枚举局部邻域
  - 可选 LLM judge

#### Motif: Local Merge / Compression

- 典型来源：
  - TiM bucket 内 merge
- 实现功能：
  - 把局部多条相近 memory 合并成更紧凑的 representation。
- 依赖：
  - partition-local candidate set
  - merge policy
  - 可选 fallback merge heuristic

#### Motif: Reflection Generation

- 典型来源：
  - Reflexion reflection append
- 代表实现：
  - [reflexion.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py#L296)
- 实现功能：
  - 将失败经验压缩成下一轮可消费的 strategy note。
- 依赖：
  - question
  - scratchpad / last attempt
  - evaluator feedback
  - LLM

#### Motif: Link Strengthening and Neighbor Update

- 典型来源：
  - A-MEM agentic evolution
- 代表实现：
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L571)
- 实现功能：
  - 为新旧记忆建立 links，并重写邻居的 context / tags。
- 依赖：
  - graph layer
  - vector candidate search
  - LLM rewrite
- 说明：
  - 这是一个强耦合 motif，适合以 bundle 进入搜索。

### 2.7 Retrieval 层

#### Motif: Buffer Retrieval

- 典型来源：
  - Reflexion reflection buffer
- 代表实现：
  - [reflexion.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py#L403)
- 实现功能：
  - 直接读取一个 bounded memory window，而不做 query search。
- 依赖：
  - target layer
  - temporal ordering

#### Motif: Partition-First Retrieval

- 典型来源：
  - TiM bucket first, in-bucket rank second
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L854)
- 实现功能：
  - 先定位 query 最相关的 partition，再只在该 partition 内精排。
- 依赖：
  - partition key
  - query projection 到 partition 的能力
  - 局部 ranker

#### Motif: Paged Similarity Retrieval

- 典型来源：
  - MemGPT paged embedding search
- 代表实现：
  - [memgpt.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/memgpt.py#L229)
- 实现功能：
  - 在一个目标 layer 内做 similarity search，并提供 page/page_size 访问。
- 依赖：
  - embeddings
  - page metadata contract

#### Motif: Seed-and-Expand Retrieval

- 典型来源：
  - A-MEM vector seed + neighbor expansion
- 代表实现：
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L750)
- 实现功能：
  - 先用向量拿 seed，再沿 graph links 扩展候选，最后可 rerank。
- 依赖：
  - vector index
  - graph links
  - 可选 reranker / LLM query expansion

### 2.8 Readout 层

#### Motif: Plain List Readout

- 典型来源：
  - TiM
- 代表实现：
  - [tim.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py#L953)
- 实现功能：
  - 把 retrieved items 直接渲染成简洁文本列表。
- 依赖：
  - `retrieved.items`

#### Motif: Prompt Context Readout

- 典型来源：
  - Reflexion
- 代表实现：
  - [reflexion.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py#L454)
- 实现功能：
  - 把 memory 转成下一轮 prompt 的 prepend context。
- 依赖：
  - query text
  - retrieved items
  - strategy metadata

#### Motif: Tool Payload Readout

- 典型来源：
  - MemGPT search tool JSON
- 代表实现：
  - [memgpt.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/memgpt.py#L327)
- 实现功能：
  - 输出 agent 工具可直接消费的结构化 payload。
- 依赖：
  - retrieved items
  - page trace metadata
  - 下游 agent loop 的工具接口约定

#### Motif: Note Render Readout

- 典型来源：
  - A-MEM
- 代表实现：
  - [amem.py](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py#L906)
- 实现功能：
  - 以 note/context/tag 形式渲染 memory，强调可读性和 agentic search trace。
- 依赖：
  - enriched note payload

---

## 3. 参数层 Motif

参数层不定义新的机制家族，只表达同一 motif 家族内部的行为变化。

### 3.1 常见参数类型

- budget:
  - TiM bucket budget
  - Reflexion memory size
- threshold:
  - write threshold
  - trigger threshold
- top-k:
  - retrieval candidate count
  - final selected count
- page controls:
  - `page`
  - `page_size`
- expansion controls:
  - `neighbor_expansion_k`
  - `candidate_k`
- policy flags:
  - `agentic_search`
  - `query_expand_with_llm`
  - `write_decision_enabled`

### 3.2 参数层的落点规则

下面这些不要误升格成模块层：

- `budget=4`
- `top_k=5`
- `memory_size=3`
- `page_size=20`
- `max_links_per_record=4`

它们属于“同一个 motif 的数值配置”，而不是新的 primitive。

---

## 4. 约束层 Motif

约束层不是在“做功能”，而是在定义哪些 motif 组合是合法的。

### 4.1 Capability Constraints

- 示例：
  - graph retrieval 需要 graph layer
  - similarity retrieval 需要 embeddings
  - seed-and-expand retrieval 需要 graph links + vector index
- 实现上建议：
  - 挂到 `ModuleSpec.store_requirements`
  - 或额外的 searchable metadata 中

### 4.2 Metadata Contract Constraints

- 示例：
  - keyed upsert 需要稳定 key
  - outcome-conditioned trigger 需要 outcome schema
  - reflection generation 需要 question / scratchpad / feedback
  - partition-first retrieval 需要 partition key
- 说明：
  - 这是当前 runtime 还没完全形式化的部分。
  - 但从 motif DSL 角度，它们必须被显式记录。

### 4.3 Family Coupling Constraints

- 示例：
  - TiM 的 local maintenance 强依赖 TiM-style partition metadata
  - A-MEM 的 evolution 强依赖 graph note payload
  - MemGPT 的 readout 强依赖 tool-facing downstream
- 建议：
  - 为每个 motif 额外声明：
    - `family_id`
    - `coupling_level`
    - `safe_search_mode`

### 4.4 Residual Boundary

有些内容不应直接成为 DSL primitive，而应被标为 residual：

- paper-specific prompt wording
- benchmark-specific metadata schema
- 某篇方法特有的 agent loop protocol
- 某篇实现里偶然选定的字段名

DSL 只需要表达：

- “有 failure-triggered reflection”
- “有 keyed overwrite”
- “有 seed-and-expand retrieval”

而不需要把：

- `memgpt_key`
- `reflexion.feedback`
- `amem.note_text`

这些具体字段名当成语言核心。

---

## 5. 当前最值得优先 DSL 化的 Motif

如果要考虑实现优先级，建议优先做下面这些，因为它们跨方法重复出现、且组合价值高：

- `structured_extraction`
- `semantic_field_enrichment`
- `partition_key_derivation`
- `metadata_gated_write`
- `llm_judged_write`
- `keyed_upsert`
- `outcome_conditioned_trigger`
- `local_conflict_forgetting`
- `local_merge_compression`
- `buffer_retrieval`
- `partition_first_retrieval`
- `seed_and_expand_retrieval`
- `prompt_context_readout`
- `tool_payload_readout`
