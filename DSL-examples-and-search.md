# 经典记忆架构与当前 stage-1 实现的对应关系

本文档把原先 MemDSL 风格的「系统块」改写为**当前仓库里真实存在的 Python 写法**：`MemoryStore` / `StoreTopology` / `MemoryPipeline`，以及 `memprimitive.baselines` 中注册的基元类。编排方式对齐 `memprimitive/example/` 下的示例（显式 `from memprimitive import …`、`MemoryPipeline(..., store=store)` 等）。

**与「完全按论文重实现」还差多少（摘要）**

- **已有**：固定 8 个流水线槽位（ingest：`unit_formation` → `representation` → `write_trigger` → `organization` → `evolution_trigger` → `memory_evolution`；recall：`retrieval` → `readout`）、`StoreLayerSpec` 声明层与索引、`Dispatch*` 扇出、多种检索与读出类。槽位名与顺序以 `memprimitive.pipeline_slots` 为准。
- **尚未覆盖旧 DSL 中大量算子**：独立 `update` / `compression` / `maintenance` 槽位不存在；周期性事件、工具调用路由、多信号加权融合检索、图跳检索、上下文 token 预算强制执行等多在**缺失模块**中列出。`StoreLayerSpec.capacity` 仅有枚举取值（`unlimited` / `token_limited` / `sliding_window`），**当前 `MemoryStore` 不会在写入时按容量裁剪**（见文末说明）。

---

## 5. 经典方法 → 当前 Python 表达（近似）

下列代码为**概念对齐**用的骨架；可直接运行的最小闭环见 `memprimitive/example/example.py`（`ingest` + `recall`）。未给出的槽位由 `MemoryPipeline` 默认填充（如 `NeverEvolutionTrigger`、`AppendOnlyEvolution`）。

### 5.1 Generative Agents (Park et al., 2023)

**Motif**：流式观察 + 反思摘要 + 多因素检索；当前可用「向量相似度 + 近邻/关键词」近似，无独立 importance / 指数遗忘融合算子。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    PassThroughUnitFormation,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="observation_stream", theme="working", indices=("temporal", "keyword")),
        StoreLayerSpec(name="reflections", theme="semantic", indices=("temporal", "keyword", "vector")),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(
        elements=("text", "embedding", "tags", "keywords", "summary", "entities", "triple"),
    ),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(target_layer="observation_stream"),
    # 反思层需 evolution / 应用层写入；此处检索与写入层对齐为观察流
    retrieval=EmbeddingSimilarityRetrieval(top_k=50, layer="observation_stream"),
    readout=ConcatenateReadout(separator="\n"),
)
```

**说明**：论文中的周期性 LLM 反思写入 `reflections`、三信号 `WeightedSum` 检索，需额外逻辑或未来模块；可用 `evolution_trigger` + `SummaryRewriteEvolution` 做「摘要追加到另一层」的弱化版（见 5.8 / 文末缺失表）。

---

### 5.2 MemGPT (Packer et al., 2023)

**Motif**：多区记忆 + 工具驱动读写 + 上下文管理；当前无「按工具名路由 store」的基元。

```python
from memprimitive import DispatchOrganization, MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    LayerAwareRetrieval,
    PassThroughUnitFormation,
    RecencyRetrieval,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="main_context", theme="working", indices=("temporal", "keyword")),
        StoreLayerSpec(name="archival", theme="semantic", indices=("vector", "keyword", "temporal")),
        StoreLayerSpec(name="recall", theme="semantic", indices=("temporal",)),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "embedding", "summary", "keywords")),
    write_trigger=AlwaysWriteTrigger(),
    organization=DispatchOrganization(
        (
            AppendOrganization(target_layer="main_context"),
            AppendOrganization(target_layer="archival"),
        ),
        primary_index=0,
    ),
    retrieval=LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=10, layer="archival"),
        retriever_by_layer={"archival": RecencyRetrieval(top_k=10, layer="archival")},
        top_k=10,
    ),
    readout=ConcatenateReadout(separator="\n\n"),
)
```

**说明**：`AgentToolCall` / `AgentExplicitTarget` / `ContextWindowManager` 均无对应类；`DispatchOrganization` 会**顺序**跑子模块并共享同一 packet，与「agent 选工具写不同层」不等价。

---

### 5.3 Reflexion (Shinn et al., 2023)

**Motif**：失败时写反思、滑窗、读回时前缀；当前无 `OnEvent("task_failure")`，无专用滑窗维护。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(
            name="reflections",
            theme="working",
            capacity="sliding_window",  # 声明用；写入路径不自动截断
            indices=("temporal",),
        ),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "summary")),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(target_layer="reflections"),
    retrieval=RecencyRetrieval(top_k=100, layer="reflections"),
    readout=ConcatenateReadout(separator="\n"),
)
```

**说明**：需业务层在失败时调用 `ingest` 或改用 `ThresholdWriteTrigger`/`compose_write_trigger` 组合信号近似「是否写入」；滑窗需在应用层删旧记录或待维护模块。

---

### 5.4 Voyager (Wang et al., 2023)

**Motif**：技能库 + 向量 + 标签；可用 `TagRetrieval` / `EmbeddingSimilarityRetrieval` + 图组织近似「索引库」，无代码专用表示与 upsert。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    ConcatenateReadout,
    GraphAppendOrganization,
    PassThroughUnitFormation,
    TagRetrieval,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(
            name="skill_library",
            theme="knowledge_graph",
            shape="Graph",
            indices=("graph", "entity", "tag", "vector"),
        ),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "embedding", "tags", "keywords", "description")),
    write_trigger=AlwaysWriteTrigger(),
    organization=GraphAppendOrganization(target_layer="skill_library"),
    retrieval=TagRetrieval(top_k=5, layer="skill_library"),
    readout=ConcatenateReadout(separator="\n\n"),
)
```

**说明**：`SkillExtractor` / `CodeWithDescription` / `UpsertByKey` / `CodeInjection` 均无；`BasicRepresentation` 的 `description` 在配置 OpenAI 环境变量时可走 LLM，否则为启发式（见 `memprimitive/baselines/representation.py`）。

---

### 5.5 MemoryBank (Zhong et al., 2024)

**Motif**：短期/长期分流 + 压缩 + 遗忘曲线；当前有分层组织与 BM25/向量检索，无 LLM 摘要到长期层、无 Ebbinghaus 维护。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    BM25Retrieval,
    ConcatenateReadout,
    ConditionalLayerOrganization,
    PassThroughUnitFormation,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="short_term", theme="working", capacity="token_limited", indices=("temporal",)),
        StoreLayerSpec(name="long_term", theme="semantic", indices=("vector", "temporal", "keyword")),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "embedding", "entities", "triple", "tags")),
    write_trigger=AlwaysWriteTrigger(),
    organization=ConditionalLayerOrganization(
        default_layer="short_term",
        rules=(
            {"has_entity": True, "target_layer": "long_term"},
        ),
    ),
    retrieval=BM25Retrieval(top_k=20, layer="long_term"),
    readout=ConcatenateReadout(separator="\n"),
)
```

**说明**：`DualStoreRouter` / `MergeByEntity` / `Summarization` / `EbbinghausForgetting` 无直接对应；`ConditionalLayerOrganization` 规则仅支持 `has_entity`、`unit_type`、`tag_contains`、`metadata_key`（见 `organization.py`）。

---

### 5.6 SCM — Self-Controlled Memory (Wang et al., 2024)

**Motif**：结构化三元组 + 门控写入 + 实体检索；可用表示层 `triple`/`entities` + `EntityRetrieval` 部分覆盖。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EntityRetrieval,
    PassThroughUnitFormation,
    ThresholdWriteTrigger,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="short_term", theme="working", indices=("temporal",)),
        StoreLayerSpec(name="long_term", theme="semantic", indices=("entity", "vector")),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "triple", "entities", "embedding", "kv")),
    write_trigger=ThresholdWriteTrigger(threshold=0.6),
    organization=AppendOrganization(target_layer="long_term"),
    retrieval=EntityRetrieval(top_k=10, layer="long_term"),
    readout=ConcatenateReadout(separator="\n"),
)
```

**说明**：`LLMJudge` / `LLMStructuredExtractor` / `UpsertByEntity` / `LLMControlled` 均无；`ThresholdWriteTrigger` 的分数来自触发族加权信号，不是论文中的判别准则（见 `memprimitive/baselines/write_trigger.py`、`_trigger_family.py`）。

---

### 5.7 A-MEM — Agentic Memory (Xu et al., 2025)

**Motif**：图记忆 + 多跳检索；当前仅有 `GraphAppendOrganization` 写入图元数据，**无** `GraphHop` 或多跳检索算子。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    GraphAppendOrganization,
    PassThroughUnitFormation,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(
            name="memory_graph",
            theme="knowledge_graph",
            shape="Graph",
            indices=("graph", "entity", "vector", "tag"),
        ),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "embedding", "tags", "entities", "triple")),
    write_trigger=AlwaysWriteTrigger(),
    organization=GraphAppendOrganization(target_layer="memory_graph"),
    retrieval=EmbeddingSimilarityRetrieval(top_k=10, layer="memory_graph"),
    readout=ConcatenateReadout(separator="\n"),
)
```

---

### 5.8 TiM — Think-in-Memory (Liu et al., 2023)

**Motif**：推理步写入 + 渐进摘要 + 预算触发；可用 `SummaryRewriteEvolution` + `ThresholdEvolutionTrigger` 表达「部分单元摘要追加到层」，无 `BudgetExceeded` 触发器。

```python
from memprimitive import MemoryPipeline, MemoryStore, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    KeywordCountRetrieval,
    PassThroughUnitFormation,
    SummaryRewriteEvolution,
    ThresholdEvolutionTrigger,
)

topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="thought_memory", theme="working", capacity="token_limited", indices=("temporal", "vector")),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    store=store,
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "embedding", "summary")),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(target_layer="thought_memory"),
    evolution_trigger=ThresholdEvolutionTrigger(threshold=0.5),
    memory_evolution=SummaryRewriteEvolution(target_layer="thought_memory"),
    retrieval=KeywordCountRetrieval(top_k=5, layer="thought_memory"),
    readout=ConcatenateReadout(separator="\n"),
)
```

**说明**：`AfterReasoningStep` / `PrependToContext` / 预算阈值触发均无；`SummaryRewriteEvolution` 对 `evolution_decisions` 为真的单元生成摘要记录（见 `memory_evolution.py`）。

---

## 6. 对搜索的支持分析（按当前实现）

### 6.1 搜索空间的三个维度

| 层 | 维度 | 说明 |
|----|------|------|
| **Structural** | `StoreTopology` | 若干 `StoreLayerSpec`：`shape` ∈ {`Flat`, `Graph`}；`indices` ∈ {`vector`, `entity`, `temporal`, `keyword`, `graph`, `tag`}；`capacity` 枚举见 `memprimitive.core` |
| **Modular** | 流水线槽位 | 共 **8** 个槽位（非旧 DSL 的 9+2 触发器拆分）；见 `memprimitive.pipeline_slots.ALL_PIPELINE_SLOTS` |
| **Parametric** | 各类 `__init__` | 如 `RecencyRetrieval(top_k, layer)`、`BasicRepresentation(elements=...)`、`ConditionalLayerOrganization(rules=...)` 等 |

一次「配置」可视为各槽位所选**具体类**及其构造参数；组合测试可参考 `memprimitive.baselines.registry.baseline_factory_groups` 与 `tests/test_baselines.py`。

### 6.2 约束剪枝

仍适用 `ModuleSpec` 上的 `input_requirements`、`store_requirements`、`layer_requirements` 等（见各基元 `spec`）。例如：`EmbeddingSimilarityRetrieval` 声明 `store_requirements=("record.embedding",)`；`GraphAppendOrganization` 要求拓扑中目标层 `shape="Graph"`（`MemoryPipeline._validate_store_compatibility`）。

### 6.3 配置向量化（更新）

```
config_vector ≈ [
    store_topo_encoding,     # 各层 shape / indices / capacity 的离散编码
    unit_formation_id, ...,
    representation_id, elements_bitmap_or_hash, ...,
    write_trigger_id, ...,
    organization_id, ...,
    evolution_trigger_id, ...,
    memory_evolution_id, ...,
    retrieval_id, ...,
    readout_id, ...,
]
```

旧 DSL 中的 `compress_trigger` / `maintain_trigger` 在实现上并入 **`evolution_trigger` + `memory_evolution`** 两槽（或默认 `Never` + `AppendOnly` 无额外行为）。

### 6.4 Motif 归纳

频率 / 共现 / 消融 / 聚类等方法仍适用于上述向量；需注意当前检索侧**无**统一的 `Compose([...], merge=WeightedSum)` 基元，多策略合并需自定义 `RetrievalModule` 或用 `LayerAwareRetrieval` 做分层不同检索器再合并。

---

## 7. 相对旧 DSL 与论文流程仍缺失或未落地的能力

下列为旧文档中的名称或常见论文构件，在 **`memprimitive.baselines` 当前源码中不存在**或**仅有声明无运行时语义**；使用前请以源码为准（`README.md` 为索引，类行为以 `.py` 为准）。

### 7.1 槽位 / 算子级（无同名或等价单一类）

- **独立槽位**：旧 DSL 的 `update`、`compression`、`maintenance`；现由 `organization`（正常写入）与 `memory_evolution`（演化写入）分担部分语义，无通用 `Dispatch`/`MergeByEntity`/`UpsertByKey` 等。
- **写入 / 组织**：`TurnLevel`、`AgentToolCall`、`AgentExplicitTarget`、`RecursiveSummarization`（MemGPT 式）、`FlatAppend`/`TemporalAppend`（独立类）、`TagBasedPlacement`、`DualStoreRouter`、`EntityLinked`、`GraphPlacement`、`AppendWithLinkUpdate`。
- **检索**：`Similarity`（泛称）、`RecencyDecay`、`ImportanceScore`、`FullContext`、`Compose`+`WeightedSum`/`Union`、`GraphHop`、`LLMControlled`。
- **读出类**：`StructuredSections`、`FlatConcat(max_tokens=...)`、`CodeInjection`、`PrependToContext`（`ConcatenateReadout` 仅有 `separator`，无 max_tokens / 分区标题）。
- **触发（旧 compress/maintain）**：`Periodic`、`OnEvent`、`AfterWrite`、`BudgetExceeded`、`Never`/`Always` 作为独立 DSL 关键字；实现侧为 `NeverEvolutionTrigger`、`ThresholdEvolutionTrigger` 等与 `compose_*_trigger` 的触发族组合。
- **其它**：`LLMReflection`、`ContextWindowManager`、`EbbinghausForgetting`、`LLMJudge`、`TaskReflectionExtractor`、`SkillExtractor`、`ThoughtExtractor`、`AfterReasoningStep` 等。

### 7.2 数据面 / 拓扑

- **`StoreLayerSpec.capacity`**：`token_limited` / `sliding_window` **不会**在 `MemoryStore.append` 中自动截断或淘汰；若需 MemGPT/TiM 式预算，应在应用层或未来扩展存储逻辑。
- **无 `token_budget` 字段**：旧 DSL 中的 `token_budget=4000` 在当前 `StoreLayerSpec` 中无对应参数，仅可作为注释或上层约定。

### 7.3 已有且宜直接查阅源码的模块（示例）

| 槽位 | 模块文件 | 导出类（注册在 `BASELINE_CLASSES`） |
|------|-----------|-------------------------------------|
| `unit_formation` | `unit_formation.py` | `PassThroughUnitFormation`, `SentenceSplitUnitFormation`, `LineSplitUnitFormation`, `WindowedUnitFormation`, `MetadataHintUnitFormation` |
| `representation` | `representation.py` | `BasicRepresentation`, `KeywordRepresentation` |
| `write_trigger` | `write_trigger.py` | `AlwaysWriteTrigger`, `ThresholdWriteTrigger` |
| `organization` | `organization.py` | `AppendOrganization`, `ConditionalLayerOrganization`, `GraphAppendOrganization` |
| `evolution_trigger` | `evolution_trigger.py` | `NeverEvolutionTrigger`, `ThresholdEvolutionTrigger` |
| `memory_evolution` | `memory_evolution.py` | `AppendOnlyEvolution`, `TraceOnlyEvolution`, `SummaryRewriteEvolution`, `LayerMoveEvolution` |
| `retrieval` | `retrieval.py` | `RecencyRetrieval`, `KeywordCountRetrieval`, `EmbeddingSimilarityRetrieval`, `TagRetrieval`, `EntityRetrieval`, `BM25Retrieval`, `LayerAwareRetrieval` |
| `readout` | `readout.py` | `ConcatenateReadout`, `BulletListReadout`, `GroupedByLayerReadout`, `JSONReadout` |

扇出与多子模块：`memprimitive.dispatch` 中的 `DispatchOrganization`、`DispatchRepresentation` 等（参见 `example10.py`）。

---

*（原稿自 `DSLgrammar.md` 拆出的 MemDSL 块已按上表迁移为可对照实现的 Python 形态；若需与论文一一可运行复现，需在上列缺失项上补充模块或应用层逻辑。）*
