# Primitive 扩展清单（当前框架可直接落地）

本文只列 **不改动大框架**、可在当前 `memprimitive` runtime 中直接实现的扩展。

默认约束：

- 不新增 slot。
- 不修改 `MemoryPipeline` 顺序。
- 不修改 `Packet` 主字段结构。
- 新实现优先放进现有 `memprimitive/baselines/*.py` 对应 slot 文件，并加入 `BASELINE_CLASSES`。
- layer 相关扩展必须契合当前 `StoreTopology` / `StoreLayerSpec`：`name`、`theme`、`shape in {Flat, Graph}`、`indices`、`capacity`。

---

## 0. 当前可扩展边界

当前稳定边界：

- `unit_formation`
- `representation`
- `write_trigger` / `evolution_trigger` 的组件家族
- `organization`
- `memory_evolution`
- `retrieval`
- `readout`
- layer-aware dispatch（slot 内部做，不改 slot）

当前不要做：

- 新 DSL 语法
- 新顶层 pipeline
- 新 store 类型
- 超出 `Flat` / `Graph` 的 layer shape
- 需要新增 `Packet` 主字段的复杂机制

---

## A. Unit Formation

### 可直接新增的实现

#### 1. `SentenceSplitUnitFormation`

- 形式：将一个 observation 按句切成多个 unit。
- 产物：多个 `MemoryUnit`，共享 provenance。
- 执行：
  1. 新建类到 `memprimitive/baselines/unit_formation.py`
  2. 读取 `packet.observation.text`
  3. 用简单句界规则切分
  4. 为每句生成一个 unit
  5. 记录 `trace["unit_formation"]["unit_ids"]`
  6. 加入 `BASELINE_CLASSES`

#### 2. `LineSplitUnitFormation`

- 形式：按换行切 unit，适合 notes / logs。
- 执行：
  1. 复用 `PassThroughUnitFormation` 的 provenance 写法
  2. 过滤空行
  3. 每行输出一个 unit
  4. 加测试：空行过滤、顺序保持、trace 对齐

#### 3. `WindowedUnitFormation`

- 形式：固定窗口切分长 observation。
- 参数：`window_size`、`stride`
- 执行：
  1. 用字符窗或 token 近似窗切文本
  2. 每窗一个 unit
  3. 在 metadata 里写 `window_index`
  4. 加测试：overlap、尾窗、单窗

#### 4. `MetadataHintUnitFormation`

- 形式：优先读取 `observation.metadata["units"]` 作为显式 unit 列表；无 hint 时回退 pass-through。
- 执行：
  1. 先判断 metadata hint
  2. 有 hint 就直接 materialize
  3. 无 hint 就回退单 unit
  4. 加测试：hint 覆盖、回退路径

---

## B. Representation

### 可直接新增的 element / 模块实现

#### 1. `BasicRepresentation` 新 element

当前最顺手的扩展方式是继续扩 `elements`，不必立刻拆新类。

可直接加：

- `keywords`
- `summary`
- `time_anchor`
- `relation_tags`
- `source_type`

执行：

1. 在 `memprimitive/baselines/representation.py` 扩 `_VALID_ELEMENTS`
2. 为每个新 element 增加 `_extract_`* / `_build_*`
3. 写回 `MemoryUnit.metadata["representation"]`
4. 保持不新增 `Packet` 字段
5. 加测试覆盖 element 单独开启与混合开启

#### 2. `KeywordRepresentation`

- 形式：从文本抽 keyword 列表，便于 keyword/tag 检索。
- 建议落点：
  - 直接做成 `BasicRepresentation(elements=("text", "tags"))` 的增强版
  - 或新类 `KeywordRepresentation`
- 执行：
  1. 从 text 抽词并去重
  2. 写入 `metadata["representation"]["keywords"]`
  3. 可同步并入 `tags`

#### 3. `EntityCentricRepresentation`

- 形式：强化 `entities`、`kv`、`triples`，不依赖新框架。
- 适合：
  - profile
  - semantic layer
- 执行：
  1. 新类或扩 `BasicRepresentation`
  2. 优先从 metadata hint 读取结构化字段
  3. 缺失时再规则抽取
  4. 保证输出仍落在现有 `MemoryUnit` 字段中

#### 4. `LightweightSummaryRepresentation`

- 形式：基于规则压缩生成短描述，不依赖外部 LLM。
- 执行：
  1. 从 text / triples / kv 生成单句 summary
  2. 写入 `description`
  3. 保持 `description` 可由 rule-based 或 API 两路生成

---

## C. Trigger Family Components

这里优先扩 **组件**，不要先扩“完整 trigger 类”。

### 1. SignalProvider

可直接新增：

- `UnitLengthSignal`
- `KeywordMatchSignal`
- `HasEntitySignal`
- `HasTripleSignal`
- `HasKVSignal`
- `TagMatchSignal`
- `LayerTargetSignal`
- `QueryOverlapSignal`

执行：

1. 在 `memprimitive/baselines/_trigger_family.py` 新增 `SignalProvider` 子类
2. 从 `packet.units`、`packet.query`、`packet.placements`、`store` 取值
3. 输出命名 signal map
4. 为缺失前置字段时报错或在组合时通过 `required_fields` 约束

### 2. ScoreAggregator

可直接新增：

- `MaxScorer`
- `MinScorer`
- `AverageScorer`
- `ClippedWeightedSumScorer`

执行：

1. 新增 `ScoreAggregator` 子类
2. 输入仍是 `signals`
3. 只产一个 float
4. 加测试：缺 signal、边界值、组合结果

### 3. Gate

可直接新增：

- `RequireEntityGate`
- `RequireTripleGate`
- `RequireTagGate`
- `LayerAllowedGate`
- `QueryPresentGate`

执行：

1. 新增 `Gate` 子类
2. 只做硬条件判断
3. 不直接写 decision
4. 用 `BooleanGatePolicy` 或 `ThresholdPolicy` 组合

### 4. DecisionPolicy

可直接新增：

- `TopKPolicy`
- `TopFractionPolicy`
- `ThresholdWithFallbackPolicy`

执行：

1. 若 policy 需要跨 unit 比较，可优先做成新 runner 或先不做
2. 当前最稳妥的是只加 per-unit policy
3. 因此建议先做：
  - `BandPassThresholdPolicy`
  - `ThresholdOrGatePolicy`

### 5. Trigger 组合产物

在组件足够后，再用现有组合入口生成具体 trigger：

- `compose_write_trigger(...)`
- `compose_evolution_trigger(...)`

执行：

1. 先补组件
2. 再在 `write_trigger.py` / `evolution_trigger.py` 写少量预制组合类
3. 每个预制类仍只是一组组件配置

---

## D. Organization

### 可直接新增的实现

#### 1. `ConditionalLayerOrganization`

- 形式：根据 unit 内容选择目标 layer。
- 兼容当前 topology：是。
- 执行：
  1. 参数支持 `default_layer` + 简单规则
  2. 规则基于 `unit_type` / `tags` / `entities` / `metadata`
  3. 输出 `placements`
  4. 对 `decision=True` 的 unit 正常 append

#### 2. `DispatchOrganization`

- 形式：给不同 layer 绑定不同子 organization，但仍在 `organization` 一个 slot 内。
- 兼容当前 topology：是。
- 执行：
  1. 做成 slot 内 dispatcher
  2. 先决定 target layer
  3. 再调用 layer-specific 组织逻辑
  4. 初版可只支持“不同 layer，不同 append 规则”

#### 3. `DuplicateToLayersOrganization`

- 形式：一个 unit 可写入多个已声明 layer。
- 兼容当前 topology：是。
- 注意：当前 `Placement` 是单 layer，初版不要改数据结构。
- 执行：
  1. 不建议现在做真正多 placement
  2. 如果必须做，建议暂缓
  3. 当前更推荐拆成多个 unit 再分别写入

#### 4. `GraphAppendOrganization`

- 形式：写入 `shape="Graph"` 的 layer，同时把图相关信息写进 record metadata。
- 兼容当前 topology：是，前提是 layer 已声明为 `Graph`。
- 执行：
  1. 检查 `store.layer_shape(target_layer) == "Graph"`
  2. 从 unit.triples / entities 生成图相关 metadata
  3. 仍然 append 为 `MemoryRecord`
  4. 不要求引入真正图存储引擎

---

## E. Memory Evolution

### 可直接新增的实现

#### 1. `TraceOnlyEvolution`

- 形式：比 `AppendOnlyEvolution` 多记录更详细 effect trace，但不改 store。
- 执行：
  1. 复用当前 no-op 语义
  2. 增加 effect 类型标记
  3. 为后续真实 evolution 保留 trace 结构

#### 2. `SummaryRewriteEvolution`

- 形式：对命中的 unit 生成 summary record，追加回某个 layer。
- 兼容当前框架：是。
- 执行：
  1. 读取 `packet.evolution_decisions`
  2. 对 active unit 生成 summary text
  3. `MemoryRecord.from_unit(...)` 后 append 到目标 layer
  4. `trace["memory_evolution"]["effects"]` 记录新增 record id

#### 3. `TagCleanupEvolution`

- 形式：清理或补充 tags / description，然后重写 record。
- 当前风险：现有 store 没有 record 更新接口。
- 执行：
  1. 先不要做原地 rewrite
  2. 可做 append-new-record + effect trace
  3. 若要求原位更新，暂缓

#### 4. `LayerMoveEvolution`

- 形式：把 active unit 的演化结果写去另一层，例如 `working -> semantic`。
- 兼容当前 topology：是。
- 执行：
  1. 参数给出 `target_layer`
  2. 生成新 record 追加到目标层
  3. 不删除原记录
  4. trace 标记 `move_style="copy_append"`

---

## F. Retrieval

### 可直接新增的实现

#### 1. `KeywordCountRetrieval`

- 形式：按 query token 命中数排序。
- 执行：
  1. 从 `query.text` 分词
  2. 统计每条 record 命中数
  3. 按命中数、再按 recency 排序
  4. 写 numeric `score`

#### 2. `TagRetrieval`

- 形式：按 `record.metadata["representation"]["tags"]` 或 record text 中的 tag 信号检索。
- 执行：
  1. 从 query 抽 tag-like token
  2. 命中 tags 的记录优先
  3. 回退 recency

#### 3. `EntityRetrieval`

- 形式：按 entity overlap 检索。
- 执行：
  1. 从 query 抽 entities
  2. 与 record metadata 中 entities 做 overlap
  3. 用 overlap 数排序

#### 4. `HybridScoreRetrieval`

- 形式：融合 keyword / embedding / recency。
- 执行：
  1. 计算多个子分数
  2. 做 weighted sum
  3. 输出统一 numeric score
  4. 不改 `RetrievedSet`

#### 5. `LayerAwareRetrieval` 扩展

当前最值得继续加的点：

- 每层不同 `top_k`
- 每层不同 merge weight
- active layer 自动筛选

执行：

1. 扩 `LayerAwareRetrieval` 参数
2. 保持 `retriever_by_layer` 结构不变
3. merge 仍使用当前 `global_rank`
4. 不改 `MemoryPipeline`

---

## G. Readout

### 可直接新增的实现

#### 1. `BulletListReadout`

- 形式：每条 record 一行。
- 执行：
  1. 遍历 `retrieved.items`
  2. 生成 bullet text
  3. 保留 `source_ids`

#### 2. `GroupedByLayerReadout`

- 形式：按 layer 分组格式化。
- 兼容当前框架：是。
- 执行：
  1. 读取 `record.layer`
  2. 分组后拼接文本
  3. metadata 中写每组 item 数

#### 3. `JSONReadout`

- 形式：输出 JSON 风格字符串，适合下游 agent/tool。
- 执行：
  1. 组装结构化 dict
  2. `json.dumps(..., ensure_ascii=False)`
  3. `Readout.text` 仍为字符串

---

## H. Layer / Topology 兼容扩展

这里只列 **符合当前 topology 构造法** 的扩展。

### 1. 当前可直接支持的 layer 设计

- 多个 `Flat` layer
- `Flat + Graph` 混合
- 不同 layer 使用不同 `theme`
- 不同 layer 声明不同 `indices`
- 不同 layer 做 retrieval override
- 不同 layer 做 organization target
- 不同 layer 做 evolution target

执行：

1. 用 `StoreTopology.from_layers([...])`
2. layer 名称显式声明
3. `shape` 只用 `Flat` / `Graph`
4. `indices` 只用当前允许值：`vector`、`entity`、`temporal`、`keyword`、`graph`、`tag`

### 2. 当前适合先实现的 layer 模板

#### `working + episodic`

- `working`: `Flat`, `indices=("temporal", "keyword")`
- `episodic`: `Flat`, `indices=("temporal", "vector")`

#### `working + semantic`

- `working`: `Flat`, `indices=("temporal",)`
- `semantic`: `Flat`, `indices=("entity", "keyword", "vector")`

#### `episodic + knowledge_graph`

- `episodic`: `Flat`, `indices=("temporal", "vector")`
- `knowledge_graph`: `Graph`, `indices=("graph", "entity")`

### 3. 当前不要做的 layer 方向

- 隐式层级树
- 动态生成大量匿名 layer
- 需要新增 shape 的 layer
- 依赖真正图数据库语义的 layer

## I. 落地规则

每新增一个 primitive 或组件，统一这样执行：

1. 在对应 slot 文件实现类。
2. 保持 `ModuleSpec.slot` 正确。
3. 加入对应 `BASELINE_CLASSES`。
4. 在 `tests/test_baselines.py` 增加行为测试。
5. 若影响组合枚举，确认 `iter_baseline_pipeline_instances()` 仍可运行。
6. 若涉及 layer，测试 declared topology、missing layer、multi-layer 三种情况。