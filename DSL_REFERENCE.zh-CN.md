# MemPrimitive DSL 参考手册

这是一份“查阅型”文档。目标不是讲理念，而是回答三类问题：

1. 这个工程里现在真实存在的 DSL-like 结构是什么。
2. 每个 slot 可以填哪些模块，它们的功能、IO 和写法是什么。
3. pipeline 怎么组，怎么传列表，怎么做 dispatch。

本文全部以当前代码为准，主要对应：

- [`memprimitive/core.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/core.py)
- [`memprimitive/interfaces.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/interfaces.py)
- [`memprimitive/pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py)
- [`memprimitive/dispatch.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/dispatch.py)
- [`memprimitive/baselines`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines)

## 1. 当前“DSL-like 结构”到底是什么

当前仓库还没有一个完整的声明式 DSL parser/compiler。真正可执行的“DSL-like 结构”是：

- 固定顺序的 pipeline slot
- 统一的 `Packet` 中间表示
- 统一的 `MemoryStore` / `StoreTopology`
- 每个模块自带 `ModuleSpec`
- 支持“单模块 / 同 slot 顺序列表 / dispatch fan-out”的组装方式

也就是说，当前的 DSL 不是文本语言，而是“受约束的 Python 组装语言”。

## 2. Pipeline 槽位

当前标准顺序见 [`memprimitive/pipeline_slots.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline_slots.py)：

```text
unit_formation
-> representation
-> write_trigger
-> organization
-> evolution_trigger
-> memory_evolution
-> retrieval
-> readout
```

其中前 6 个是 ingest，后 2 个是 recall。

## 3. 共享数据对象

### 3.1 Observation

入口输入对象。

- 类型：`Observation`
- 主要字段：`text`、`observation_id`、`timestamp`、`source`、`metadata`
- 创建方式：

```python
Observation(text="Alice likes jasmine tea.", source="notes")
```

### 3.2 MemoryUnit

写入前的中间记忆单元。

- 类型：`MemoryUnit`
- 常见字段：
  - `text`
  - `unit_id`
  - `unit_type`
  - `timestamp`
  - `representation_elements`
  - `normalized_text`
  - `embedding`
  - `triples`
  - `kv`
  - `entities`
  - `tags`
  - `description`
  - `metadata`

### 3.3 MemoryRecord

真正落到 store 里的记录。

- 类型：`MemoryRecord`
- 关键区别：多了 `record_id` 和 `layer`
- 常见写入路径：由 organization / memory_evolution 内部调用 `MemoryRecord.from_unit(...)`

### 3.4 Placement

表示某个 unit 将被放到哪个 layer。

- 类型：`Placement`
- 字段：`unit_id`、`target_layer`

### 3.5 Query

检索输入。

- 类型：`Query`
- 字段：`text`、`query_id`、`timestamp`、`embedding`、`metadata`

### 3.6 RetrievedSet

检索结果。

- 类型：`RetrievedSet`
- 字段：
  - `items`
  - `scores`
  - `trace`

### 3.7 Readout

给下游 agent 的最终输出。

- 类型：`Readout`
- 字段：`text`、`source_ids`、`metadata`

### 3.8 Packet

当前 runtime 最关键的中间表示。

可读写字段包括：

- `observation`
- `units`
- `decisions`
- `evolution_decisions`
- `placements`
- `query`
- `retrieved`
- `readout`
- `trace`

可以把它理解为 slot 间共享的 IR。

## 4. Store 与 topology

### 4.1 StoreLayerSpec

定义一个 layer：

- `name`
- `theme`
- `shape`
- `indices`
- `capacity`
- `settings`

`shape` 当前支持：

- `Flat`
- `Graph`

`indices` 当前支持：

- `vector`
- `entity`
- `temporal`
- `keyword`
- `graph`
- `tag`

### 4.2 StoreTopology

声明整个 store 有哪些 layer。

常见写法：

```python
topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
        StoreLayerSpec(
            name="knowledge_graph",
            theme="knowledge_graph",
            shape="Graph",
            indices=("graph", "entity"),
        ),
    ]
)
```

### 4.3 MemoryStore

真实的内存存储对象。

常见能力：

- `append(record)`
- `iter_records(layer=None)`
- `count(layer=None)`
- `has_layer(layer)`
- `layer_shape(layer)`
- `layer_supports_index(layer, index_name)`
- `replace_record(...)`
- `upsert_record(...)`
- `add_graph_links(...)`
- `iter_graph_neighbors(...)`

## 5. 模块接口与约束

所有 slot 模块都继承 [`memprimitive/interfaces.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/interfaces.py) 中的 ABC，并统一实现：

```python
run(packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]
```

每个模块还有一个 `spec: ModuleSpec`，用于声明：

- `name`
- `slot`
- `input_requirements`
- `output_guarantees`
- `store_requirements`
- `layer_requirements`
- `side_effects`

`MemoryPipeline` 会同时检查：

1. 模块 ABC 类型是否匹配 slot。
2. `ModuleSpec.slot` 是否匹配 slot 名字。
3. store / layer 约束是否满足。

## 6. 各 slot 模块总表

### 6.1 `unit_formation`

输入通常依赖 `packet.observation`，输出是 `packet.units`。

#### `PassThroughUnitFormation`

- 功能：一条 observation 直接变成一个 unit。
- 输入：`observation.text`
- 输出：`units`
- 副作用：无
- 典型写法：

```python
PassThroughUnitFormation()
```

#### `SentenceSplitUnitFormation`

- 功能：按句子切分 observation。
- 输入：`observation.text`
- 输出：多个 sentence-level unit
- 写法：

```python
SentenceSplitUnitFormation()
```

#### `LineSplitUnitFormation`

- 功能：按非空行切分 observation。
- 输入：`observation.text`
- 输出：多个 line-level unit
- 写法：

```python
LineSplitUnitFormation()
```

#### `WindowedUnitFormation`

- 功能：按固定窗口和步长切 observation。
- 输入：`observation.text`
- 输出：多个 window unit
- 参数：
  - `window_size`
  - `stride`
- 写法：

```python
WindowedUnitFormation(window_size=120, stride=80)
```

#### `MetadataHintUnitFormation`

- 功能：优先读取 `observation.metadata["units"]` 中的 hints 生成 unit；没有 hint 时回退为 passthrough。
- 支持 hint：
  - 字符串
  - 至少包含 `text` 的 dict
- 写法：

```python
MetadataHintUnitFormation()
```

### 6.2 `representation`

输入通常依赖 `packet.units`，输出仍是 `packet.units`，但 unit 被增强。

#### `BasicRepresentation`

- 功能：为 unit 填充一组 representation elements。
- 输入：`units`
- 输出：
  - `units.representation_elements`
  - `units.normalized_text`
  - `units.embedding`
  - `units.triples`
  - `units.kv`
  - `units.entities`
  - `units.tags`
  - `units.description`
  - `units.metadata["representation"]`
- 可选 elements：
  - `text`
  - `embedding`
  - `triple`
  - `kv`
  - `entities`
  - `tags`
  - `description`
  - `keywords`
  - `summary`
  - `time_anchor`
  - `relation_tags`
  - `source_type`
- 参数：
  - `elements`
  - `embedding_model`
  - `api_key`
  - `base_url`
  - `model`
- 说明：
  - `embedding` 使用 `sentence-transformers`
  - `summary` / `description` 需要 OpenAI-compatible API
- 写法：

```python
BasicRepresentation(elements=("text", "embedding", "entities", "tags", "keywords"))
```

#### `KeywordRepresentation`

- 功能：`BasicRepresentation` 的 keyword-oriented 包装。
- 默认 elements：`("text", "keywords", "tags")`
- 写法：

```python
KeywordRepresentation()
```

#### `SemanticFieldEnrichmentRepresentation`

- 功能：为 note-like unit 生成结构化 note payload。
- 输出主要写入：`unit.metadata[note_namespace]`
- 常见字段：
  - `content`
  - `note_text`
  - `context`
  - `keywords`
  - `tags`
  - `category`
  - `attributes`
- 参数：
  - `note_namespace`
  - `strict_llm`
  - `default_category`
- 说明：当前要求 `strict_llm=True`
- 写法：

```python
SemanticFieldEnrichmentRepresentation(note_namespace="note")
```

#### `RetrievalOrientedEmbeddingRepresentation`

- 功能：把 enriched note 转成面向检索的 embedding 表示。
- 输出：
  - `unit.embedding`
  - `unit.metadata["representation"]`
  - `unit.metadata[note_namespace]["enhanced_embedding_text"]`
- 参数：
  - `note_namespace`
  - `default_category`
  - `embedding_version`
  - `embedding_model`
- 写法：

```python
RetrievalOrientedEmbeddingRepresentation(note_namespace="note")
```

### 6.3 `write_trigger`

输出写到 `packet.decisions`。

#### `AlwaysWriteTrigger`

- 功能：全部写入。
- 写法：

```python
AlwaysWriteTrigger()
```

#### `ThresholdWriteTrigger`

- 功能：常数信号阈值触发。
- 参数：
  - `threshold`
  - `constant`
- 写法：

```python
ThresholdWriteTrigger(threshold=0.5, constant=1.0)
```

#### `MetadataGatedWriteTrigger`

- 功能：按 unit type 和 metadata flag 共同决定是否写入。
- 典型语义：TiM-like metadata gated write
- 参数：
  - `expected_unit_type`
  - `metadata_path`
  - `metadata_source`
  - `default_write`
- 写法：

```python
MetadataGatedWriteTrigger(
    expected_unit_type="tim_thought",
    metadata_path="tim.write",
)
```

#### `KeyReadyWriteTrigger`

- 功能：只有当 unit 已经带有某种 key / partition key 时才写。
- 典型语义：MemGPT-like keyed write / upsert 前置条件
- 参数：
  - `expected_unit_type`
  - `key_paths`
  - `key_source`
  - `strict_missing`
- 写法：

```python
KeyReadyWriteTrigger(key_paths=("memgpt_key",))
```

#### `LLMJudgedWriteTrigger`

- 功能：让 LLM 判断 enriched note 是否值得写入。
- 输出 trace 中包含 `decision`、`reason`、`confidence`
- 参数：
  - `note_namespace`
  - `enabled`
  - `strict_llm`
  - `default_category`
- 写法：

```python
LLMJudgedWriteTrigger(note_namespace="note", enabled=True)
```

#### 组合 builder

除了类，还可以直接组 trigger：

- `compose_write_trigger(...)`
- `compose_metadata_gated_write_trigger(...)`
- `compose_key_ready_write_trigger(...)`

最通用写法：

```python
compose_write_trigger(
    name="demo_trigger",
    signal_providers=(...,),
    scorer=...,
    gate=...,
    policy=...,
)
```

### 6.4 `organization`

这个 slot 负责 ingest-time 正常写入。它既决定 placement，也可能直接修改 store。

#### `AppendOrganization`

- 功能：把每个 unit 放到固定 layer，并把 `decision=True` 的 unit append 成 record。
- 输入：
  - `units`
  - `decisions`
- 输出：
  - `placements`
- 副作用：
  - `modify_store`
  - `append_records`
- 参数：
  - `target_layer`
- 写法：

```python
AppendOrganization(target_layer="default")
```

#### `ConditionalLayerOrganization`

- 功能：按规则把 unit 路由到不同 layer，然后 append。
- 支持规则键：
  - `has_entity`
  - `unit_type`
  - `tag_contains`
  - `metadata_key`
- 参数：
  - `default_layer`
  - `rules`
- 写法：

```python
ConditionalLayerOrganization(
    default_layer="working",
    rules=(
        {"unit_type": "profile", "target_layer": "semantic"},
        {"has_entity": True, "target_layer": "entity_layer"},
    ),
)
```

#### `GraphAppendOrganization`

- 功能：把 unit append 到 graph layer，并补 graph metadata。
- 约束：
  - 目标层必须存在
  - 目标层 `shape="Graph"`
  - 目标层支持 `graph` index
- 参数：
  - `target_layer`
- 写法：

```python
GraphAppendOrganization(target_layer="knowledge_graph")
```

#### `PlacementWithoutAppendOrganization`

- 功能：只产生 `placements`，不真正落库。
- 典型用途：Reflexion-style trial packet 只走 placement contract，不 append 当前 trial。
- 参数：
  - `target_layer`
- 写法：

```python
PlacementWithoutAppendOrganization(target_layer="trial_buffer")
```

#### `GraphAppendLinkReadyOrganization`

- 功能：把 enriched note 写入 graph layer，并初始化 link-ready graph metadata。
- 约束：
  - graph layer
  - graph index
  - vector index
- 参数：
  - `target_layer`
  - `note_namespace`
- 写法：

```python
GraphAppendLinkReadyOrganization(target_layer="knowledge_graph", note_namespace="note")
```

### 6.5 `evolution_trigger`

输出写到 `packet.evolution_decisions`。

#### `NeverEvolutionTrigger`

- 功能：默认不触发额外演化。
- 写法：

```python
NeverEvolutionTrigger()
```

#### `ThresholdEvolutionTrigger`

- 功能：常数信号阈值触发演化。
- 参数：
  - `threshold`
  - `constant`
- 写法：

```python
ThresholdEvolutionTrigger(threshold=0.5, constant=1.0)
```

#### `OutcomeConditionedEvolutionTrigger`

- 功能：按 observation outcome 是否失败来触发演化。
- 典型语义：Reflexion-style failed-trial trigger
- 输入额外要求：
  - `observation`
- 参数：
  - `threshold`
  - `feedback_bonus`
- 写法：

```python
OutcomeConditionedEvolutionTrigger(threshold=1.0, feedback_bonus=0.1)
```

#### `NewWriteEvolutionTrigger`

- 功能：当新写入的 unit 满足 family、placement、partition key 条件时触发局部维护。
- 典型语义：TiM-like local maintenance
- 参数：
  - `expected_unit_type`
  - `allowed_layers`
  - `partition_key_paths`
  - `partition_key_source`
  - `strict_missing_partition_key`
- 写法：

```python
NewWriteEvolutionTrigger(
    expected_unit_type="tim_thought",
    allowed_layers=("thought_memory",),
)
```

#### `NeighborExistsEvolutionTrigger`

- 功能：当前 unit 在 graph layer 有邻居候选时才触发 graph evolution。
- 约束：
  - graph layer
  - vector index
- 参数：
  - `target_layer`
  - `candidate_top_k`
  - `similarity_threshold`
- 写法：

```python
NeighborExistsEvolutionTrigger(target_layer="knowledge_graph", candidate_top_k=3)
```

#### 组合 builder

- `compose_evolution_trigger(...)`
- `compose_outcome_conditioned_evolution_trigger(...)`
- `compose_new_write_evolution_trigger(...)`
- `compose_graph_neighbor_evolution_trigger(...)`

### 6.6 `memory_evolution`

这个 slot 是“额外演化”，不是普通 ingest append。

#### `AppendOnlyEvolution`

- 功能：额外演化的 no-op，占位实现。
- 读取：
  - `units`
  - `placements`
  - `evolution_decisions`
- 写法：

```python
AppendOnlyEvolution()
```

#### `TraceOnlyEvolution`

- 功能：只写 trace，不改 store。
- 写法：

```python
TraceOnlyEvolution()
```

#### `SummaryRewriteEvolution`

- 功能：为被激活 unit 生成 summary record 并 append 到目标层。
- 参数：
  - `target_layer`
- 写法：

```python
SummaryRewriteEvolution(target_layer="summary")
```

#### `LayerMoveEvolution`

- 功能：把激活 unit copy-append 到另一个 layer。
- 参数：
  - `target_layer`
- 写法：

```python
LayerMoveEvolution(target_layer="archive")
```

#### `GraphLinkEvolution`

- 功能：在 graph layer 基于候选邻居补 link。
- 参数：
  - `target_layer`
  - `neighbor_limit`
  - `min_score`
  - `rewrite_neighbor_metadata`
- 写法：

```python
GraphLinkEvolution(target_layer="knowledge_graph", neighbor_limit=2)
```

#### `GraphNeighborAppendEvolution`

- 功能：`GraphLinkEvolution` 的兼容包装，保留旧类名与旧 trace 名称。
- 参数：
  - `target_layer`
  - `neighbor_limit`
  - `bidirectional`
- 写法：

```python
GraphNeighborAppendEvolution(target_layer="knowledge_graph", neighbor_limit=2)
```

#### `GraphNeighborContextTraceEvolution`

- 功能：跟踪已链接邻居上下文，并可选写入保守的 `graph.neighbor_context`。
- 参数：
  - `target_layer`
  - `rewrite_metadata`
- 写法：

```python
GraphNeighborContextTraceEvolution(target_layer="knowledge_graph", rewrite_metadata=True)
```

#### `LinkStrengtheningEvolution`

- 功能：通过 LLM 在 enriched note graph 中挑选并加强链接。
- 参数：
  - `target_layer`
  - `candidate_k`
  - `max_links_per_record`
  - `note_namespace`
  - `default_category`
- 写法：

```python
LinkStrengtheningEvolution(target_layer="knowledge_graph", candidate_k=5)
```

#### `NeighborContextUpdateEvolution`

- 功能：从当前 note 视角重写其邻居 note 的 context/tags。
- 参数：
  - `target_layer`
  - `note_namespace`
  - `default_category`
  - `candidate_k`
- 写法：

```python
NeighborContextUpdateEvolution(target_layer="knowledge_graph", note_namespace="note")
```

#### `ReflectionGenerationEvolution`

- 功能：根据 failed trial 生成 reflection 并写入某层，同时做滑窗裁剪。
- 典型语义：Reflexion-like reflection writeback
- 输入额外要求：
  - `observation`
- 参数：
  - `target_layer`
  - `memory_size`
  - `reflection_generator`
  - `prompt_builder`
- 写法：

```python
ReflectionGenerationEvolution(target_layer="reflection_memory", memory_size=3)
```

### 6.7 `retrieval`

输入通常依赖 `query`，输出写到 `packet.retrieved`。

#### `RecencyRetrieval`

- 功能：直接按 recency 取最新的 `top_k` 条记录，不使用 query token 过滤。
- 参数：
  - `top_k`
  - `layer`
- 写法：

```python
RecencyRetrieval(top_k=3, layer="default")
```

#### `KeywordCountRetrieval`

- 功能：按 query token 与 record 关键词重叠数排序。
- 参数：
  - `top_k`
  - `layer`

#### `EmbeddingSimilarityRetrieval`

- 功能：按 embedding cosine similarity 检索。
- 要求：record 有 embedding
- 参数：
  - `top_k`
  - `layer`
  - `embedding_model`

#### `TagRetrieval`

- 功能：按 query token 与 record tags 重叠检索。
- 参数：
  - `top_k`
  - `layer`

#### `EntityRetrieval`

- 功能：按 query 实体或 token 与 record entities 重叠检索。
- 参数：
  - `top_k`
  - `layer`

#### `BM25Retrieval`

- 功能：用 `rank-bm25` 在 text + keywords 上检索。
- 参数：
  - `top_k`
  - `layer`

#### `GraphNeighborRetrieval`

- 功能：基于指定 seed record id 直接取 graph 邻居。
- 典型用法：query.metadata 里带 graph seed id

#### `GraphSeedAndExpandRetrieval`

- 功能：先对 graph record 做 seed scoring，再做一跳扩展。
- 参数常见项：
  - `top_k`
  - `layer`
  - `seed_top_k`
  - `neighbor_expansion_k`

#### `VectorGraphSeedAndExpandRetrieval`

- 功能：先向量召回 seed，再做 graph expansion，可选 agentic rerank。
- 约束：
  - graph layer
  - vector index
- 参数：
  - `top_k`
  - `layer`
  - `candidate_k`
  - `neighbor_expansion_k`
- `note_namespace`
- `default_category`
- `agentic_search`
- `query_expand_with_llm`

#### `LayerAwareRetrieval`

- 功能：对每个 layer 分别检索，再做全局 merge。
- 关键参数：
  - `default_retriever`
  - `retriever_by_layer`
  - `active_layers`
  - `top_k`
  - `top_k_by_layer`
  - `merge_weight_by_layer`
  - `merge_strategy`
- 写法：

```python
LayerAwareRetrieval(
    default_retriever=RecencyRetrieval(top_k=2),
    retriever_by_layer={"knowledge_graph": EntityRetrieval(top_k=2)},
    top_k=4,
)
```

#### `BufferRetrieval`

- 功能：不看 query 内容，直接读某层最近窗口。
- 典型语义：Reflexion-style buffer read
- 参数：
  - `top_k`
  - `layer`
  - `chronological`

### 6.8 `readout`

输入通常依赖 `retrieved`，输出写到 `packet.readout`。

#### `ConcatenateReadout`

- 功能：简单拼接文本。
- 参数：
  - `separator`

#### `BulletListReadout`

- 功能：一条记录一条 bullet。

#### `GroupedByLayerReadout`

- 功能：按 layer 分组渲染。

#### `JSONReadout`

- 功能：输出 JSON 字符串。

#### `GraphReadout`

- 功能：渲染 graph record 的 entities / links。
- 参数：
  - `include_links`

#### `PromptContextReadout`

- 功能：把检索结果渲染成下一步 prompt context。
- 典型语义：Reflexion-like prompt context
- 参数：
  - `memory_layer`
  - `default_strategy`
  - `top_k`

#### `NoteRenderReadout`

- 功能：把 enriched note 渲染成可读文本。
- 参数：
  - `note_namespace`
  - `default_category`
  - `include_context`
  - `include_tags`

## 7. Trigger family 四件套

`write_trigger` 和 `evolution_trigger` 的很多模块不是各写一套逻辑，而是复用同一个 trigger family 机制。

角色分成四层：

1. `SignalProvider`
2. `ScoreAggregator`
3. `Gate`
4. `DecisionPolicy`

执行顺序是：

```text
signals -> score -> gate -> final decision
```

这也是为什么可以用 `compose_write_trigger(...)` / `compose_evolution_trigger(...)` 直接在 pipeline 里“现场组 trigger”。

当前执行容器是 [`memprimitive/baselines/_trigger_family.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/_trigger_family.py) 里的 `TriggerFamilyRunner`。它会对每个 unit 按固定顺序执行：

1. 跑所有 `SignalProvider.provide(...)`
2. 把 signal map 交给 `ScoreAggregator.score(...)`
3. 把 `signals + score` 交给 `Gate.evaluate(...)`
4. 把 `score + gate_open` 交给 `DecisionPolicy.decide(...)`

最终输出会被写入：

- `write_trigger` 场景：`packet.decisions`
- `evolution_trigger` 场景：`packet.evolution_decisions`

### 7.1 小 slot 一：`SignalProvider`

职责：为当前 unit 生成一个或多个可命名 signal，供 scorer 使用。  
统一接口：

```python
provide(context: TriggerContext, unit_index: int) -> SignalMap
```

#### `ConstantSignal`

- 功能：输出一个固定常数 signal。
- 输出：`{signal_name: value}`
- 参数：
  - `signal_name`
  - `value`
- 写法：

```python
ConstantSignal(signal_name="constant", value=1.0)
```

#### `UnitLengthSignal`

- 功能：输出当前 unit 文本长度，可选按 `normalize_by` 归一化。
- 输入：`packet.units[unit_index].text`
- 参数：
  - `signal_name`
  - `normalize_by`
- 写法：

```python
UnitLengthSignal(signal_name="unit_length", normalize_by=100.0)
```

#### `KeywordMatchSignal`

- 功能：输出 query token 与 unit text / representation keywords 的重叠数。
- 依赖：
  - `packet.query`
  - `packet.units`
- 参数：
  - `signal_name`
- 写法：

```python
KeywordMatchSignal(signal_name="keyword_match")
```

#### `HasEntitySignal`

- 功能：如果 unit 有 entity，输出 `1.0`，否则 `0.0`。
- 依赖：`unit.entities`
- 参数：
  - `signal_name`
- 写法：

```python
HasEntitySignal(signal_name="has_entity")
```

#### `HasTripleSignal`

- 功能：如果 unit 有 triple，输出 `1.0`，否则 `0.0`。
- 依赖：`unit.triples`
- 参数：
  - `signal_name`

#### `HasKVSignal`

- 功能：如果 unit 有 key-value，输出 `1.0`，否则 `0.0`。
- 依赖：`unit.kv`
- 参数：
  - `signal_name`

#### `TagMatchSignal`

- 功能：输出 query token 与 unit tags 的重叠数。
- 依赖：
  - `packet.query`
  - `unit.tags`
- 参数：
  - `signal_name`
- 写法：

```python
TagMatchSignal(signal_name="tag_match")
```

#### `LayerTargetSignal`

- 功能：当当前 placement 的 `target_layer` 落在 `allowed_layers` 中时输出 `1.0`。
- 依赖：`packet.placements`
- 参数：
  - `allowed_layers`
  - `signal_name`
- 写法：

```python
LayerTargetSignal(allowed_layers=("working", "semantic"))
```

#### `QueryOverlapSignal`

- 功能：输出 query text 和当前 unit text 的 token overlap 数。
- 依赖：
  - `packet.query`
  - `unit.text`
- 参数：
  - `signal_name`

#### `OutcomeCorrectnessSignal`

- 功能：从 `observation.metadata` 里解析 trial outcome；如果 trial 失败，输出 `1.0`，否则 `0.0`。
- 典型语义：Reflexion-style failed-trial signal
- 依赖：`packet.observation`
- 参数：
  - `signal_name`

#### `FeedbackPresenceSignal`

- 功能：如果 `observation.metadata` 里存在支持的 feedback 文本，输出 `1.0`。
- 依赖：`packet.observation`
- 参数：
  - `signal_name`

#### `MetadataFlagSignal`

- 功能：从某个 source 上按 dotted path 读 bool-ish / numeric 字段，并转成 signal。
- 可用 source：
  - `unit.metadata`
  - `observation.metadata`
  - `query.metadata`
- 参数：
  - `path`
  - `signal_name`
  - `source`
  - `default`
- 说明：
  - 找不到 path 且 `default is None` 时会报错
  - 这是 TiM-style metadata-gated trigger 常用组件
- 写法：

```python
MetadataFlagSignal(
    path="tim.write",
    source="unit.metadata",
    default=True,
    signal_name="metadata_write_flag",
)
```

#### `UnitTypeSignal`

- 功能：当 `unit.unit_type == expected_unit_type` 时输出 `1.0`。
- 参数：
  - `expected_unit_type`
  - `signal_name`
- 写法：

```python
UnitTypeSignal(expected_unit_type="tim_thought", signal_name="unit_type_ready")
```

#### `PlacementExistsSignal`

- 功能：当前 unit 有对齐的 placement 时输出 `1.0`。
- 依赖：`packet.placements`
- 参数：
  - `signal_name`

#### `PartitionKeyPresentSignal`

- 功能：如果配置的多个 dotted path 里至少有一个存在且有值，则输出 `1.0`。
- 典型语义：MemGPT/TiM 风格 keyed 或 partition-ready signal
- 可用 source：
  - `unit.metadata`
  - 其他 `_resolve_source_object` 支持的 source
- 参数：
  - `paths`
  - `signal_name`
  - `source`
  - `strict_missing`
- 写法：

```python
PartitionKeyPresentSignal(
    paths=("tim.group_id", "tim.hash_index"),
    source="unit.metadata",
    signal_name="partition_key_ready",
)
```

#### `NeighborCountSignal`

- 功能：统计当前 unit 在目标 layer 里有多少个可比较的 vector neighbor。
- 依赖：
  - 当前 unit 有 embedding
  - 目标 layer 支持 vector index
- 参数：
  - `top_k`
  - `signal_name`
  - `layer`
  - `similarity_threshold`
- 说明：
  - 如果显式传了 `layer`，优先使用该 layer
  - 否则从 `packet.placements` 推导目标 layer
- 写法：

```python
NeighborCountSignal(
    top_k=3,
    layer="knowledge_graph",
    similarity_threshold=0.5,
    signal_name="neighbor_count",
)
```

#### `TopNeighborSimilaritySignal`

- 功能：输出当前 unit 最相似邻居的 cosine similarity。
- 依赖同 `NeighborCountSignal`
- 参数：
  - `top_k`
  - `signal_name`
  - `layer`
  - `similarity_threshold`

### 7.2 小 slot 二：`ScoreAggregator`

职责：把一组 signals 聚合成单个数值分数。  
统一接口：

```python
score(signals: SignalMap) -> float
```

#### `IdentityScorer`

- 功能：直接读取某个 signal 作为 score。
- 参数：
  - `source`
- 写法：

```python
IdentityScorer(source="constant")
```

#### `WeightedSumScorer`

- 功能：按权重对多个 signal 做加权和。
- 参数：
  - `weights`
- 写法：

```python
WeightedSumScorer(weights={"importance_hint": 1.0, "feedback_present": 0.1})
```

#### `MaxScorer`

- 功能：取多个 signal 的最大值。
- 参数：
  - `sources`

#### `MinScorer`

- 功能：取多个 signal 的最小值。
- 典型语义：多个 readiness signal 都必须满足
- 参数：
  - `sources`
- 写法：

```python
MinScorer(sources=("unit_type_ready", "metadata_write_flag"))
```

#### `AverageScorer`

- 功能：取多个 signal 的平均值。
- 参数：
  - `sources`

#### `ClippedWeightedSumScorer`

- 功能：先做加权和，再把最终得分裁剪到 `[min_score, max_score]`。
- 参数：
  - `weights`
  - `min_score`
  - `max_score`

### 7.3 小 slot 三：`Gate`

职责：在 scorer 得分之后再做硬约束过滤。  
统一接口：

```python
evaluate(context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool
```

#### `AlwaysOpenGate`

- 功能：永远放行。
- 写法：

```python
AlwaysOpenGate()
```

#### `RequireEntityGate`

- 功能：只有当前 unit 含有 entity 才放行。

#### `RequireTripleGate`

- 功能：只有当前 unit 含有 triple 才放行。

#### `RequireTagGate`

- 功能：只有当前 unit 的 tags 与 `required_tags` 有交集才放行。
- 参数：
  - `required_tags`
- 写法：

```python
RequireTagGate(required_tags=("preference", "profile"))
```

#### `LayerAllowedGate`

- 功能：只有当前 placement 的 target layer 落在 `allowed_layers` 中才放行。
- 依赖：`packet.placements`
- 参数：
  - `allowed_layers`
- 写法：

```python
LayerAllowedGate(allowed_layers=("thought_memory",))
```

#### `QueryPresentGate`

- 功能：只有 packet 上存在 query 才放行。

#### `SchemaPresentGate`

- 功能：检查 source 对象上某些 dotted path 是否存在。
- 可用 source：
  - `unit`
  - `unit.metadata`
  - `observation`
  - `observation.metadata`
  - `query`
  - `query.metadata`
- 参数：
  - `paths`
  - `source`
  - `require_all`
- 写法：

```python
SchemaPresentGate(
    paths=("tim.group_id", "tim.hash_index"),
    source="unit.metadata",
    require_all=False,
)
```

#### `FeedbackSchemaGate`

- 功能：只有 observation 上存在可解析的 outcome 或 feedback schema 时才放行。
- 典型语义：Reflexion-style failure/feedback readiness gate

#### `HasEmbeddingGate`

- 功能：只有 source 对象暴露出非空 embedding 才放行。
- 参数：
  - `source`
- 默认 source：`unit`
- 写法：

```python
HasEmbeddingGate(source="unit")
```

#### `VectorIndexReadyGate`

- 功能：只有目标 layer 存在并且支持 vector index 才放行。
- 参数：
  - `layer`
- 说明：
  - 如果显式给 `layer`，优先检查该层
  - 否则回退到 placement 推导

#### `GraphLayerGate`

- 功能：只有目标 layer 存在并且 `shape == "Graph"` 才放行。
- 参数：
  - `layer`

#### `AllGate`

- 功能：只有内部所有 gate 都放行时才放行。
- 参数：
  - `gates`
- 典型语义：把 graph layer、vector index、embedding 等 readiness gate 打包成一个复合 gate
- 写法：

```python
AllGate((HasEmbeddingGate(), VectorIndexReadyGate(layer="knowledge_graph"), GraphLayerGate(layer="knowledge_graph")))
```

### 7.4 小 slot 四：`DecisionPolicy`

职责：把 `score` 和 `gate_open` 变成最终布尔 decision。  
统一接口：

```python
decide(*, score: float, gate_open: bool) -> bool
```

#### `AlwaysPolicy`

- 功能：总是输出 `True`。
- 说明：忽略 gate 和 score

#### `NeverPolicy`

- 功能：总是输出 `False`。

#### `ThresholdPolicy`

- 功能：只有 `gate_open and score >= threshold` 时输出 `True`。
- 参数：
  - `threshold`
- 写法：

```python
ThresholdPolicy(threshold=1.0)
```

#### `BooleanGatePolicy`

- 功能：直接把 `gate_open` 当成最终 decision。
- 说明：适合“gate 就是主判断”的情形

#### `BandPassThresholdPolicy`

- 功能：只有 `gate_open` 且 `lower <= score <= upper` 时输出 `True`。
- 参数：
  - `lower`
  - `upper`

#### `ThresholdOrGatePolicy`

- 功能：只要 `gate_open` 为真，或 `score >= threshold`，就输出 `True`。
- 参数：
  - `threshold`

### 7.5 常见 trigger family 组合模板

#### 最小恒真 write trigger

```python
compose_write_trigger(
    name="always_write_like",
    signal_providers=(ConstantSignal(signal_name="constant", value=1.0),),
    scorer=IdentityScorer(source="constant"),
    gate=AlwaysOpenGate(),
    policy=AlwaysPolicy(),
)
```

#### TiM-like metadata-gated write

```python
compose_write_trigger(
    name="metadata_gated_write",
    signal_providers=(
        UnitTypeSignal(expected_unit_type="tim_thought", signal_name="unit_type_ready"),
        MetadataFlagSignal(
            path="tim.write",
            source="unit.metadata",
            default=True,
            signal_name="metadata_write_flag",
        ),
    ),
    scorer=MinScorer(sources=("unit_type_ready", "metadata_write_flag")),
    gate=AlwaysOpenGate(),
    policy=ThresholdPolicy(threshold=1.0),
)
```

#### Reflexion-like failed-trial evolution trigger

```python
compose_evolution_trigger(
    name="outcome_conditioned_evolution",
    signal_providers=(OutcomeCorrectnessSignal(), FeedbackPresenceSignal()),
    scorer=WeightedSumScorer(weights={"trial_failed": 1.0, "feedback_present": 0.1}),
    gate=FeedbackSchemaGate(),
    policy=ThresholdPolicy(threshold=1.0),
    input_requirements=("units", "placements", "observation"),
)
```

#### Graph neighbor-triggered evolution

```python
compose_evolution_trigger(
    name="graph_neighbor_trigger",
    signal_providers=(
        NeighborCountSignal(top_k=3, layer="knowledge_graph", signal_name="neighbor_count"),
        TopNeighborSimilaritySignal(top_k=3, layer="knowledge_graph", signal_name="top_neighbor_similarity"),
    ),
    scorer=IdentityScorer(source="neighbor_count"),
    gate=AllGate(
        (
            HasEmbeddingGate(),
            VectorIndexReadyGate(layer="knowledge_graph"),
            GraphLayerGate(layer="knowledge_graph"),
        )
    ),
    policy=ThresholdPolicy(threshold=1.0),
    input_requirements=("units", "placements"),
)
```

## 8. Pipeline 组装方法

### 8.1 单模块写法

```python
pipeline = MemoryPipeline(
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(),
    evolution_trigger=NeverEvolutionTrigger(),
    memory_evolution=AppendOnlyEvolution(),
    retrieval=RecencyRetrieval(top_k=2),
    readout=ConcatenateReadout(),
)
```

### 8.2 传入列表：同 slot 顺序串行

`MemoryPipeline` 的每个 slot 都可以传单个模块，也可以传“同 slot 模块列表/元组”。

语义不是 fan-out，而是串行执行：

```python
pipeline = MemoryPipeline(
    representation=(
        BasicRepresentation(elements=("text", "entities")),
        BasicRepresentation(elements=("text", "entities", "tags", "keywords")),
    ),
)
```

规则：

- 必须是同一 slot 的模块
- 空 iterable 会报错
- 后一个模块读前一个模块的 `Packet` 输出

### 8.3 Dispatch：同 slot fan-out

如果你需要一份输入同时喂给多个子模块，应使用 `Dispatch*`，而不是普通列表。

可用 dispatch 类：

- `DispatchUnitFormation`
- `DispatchRepresentation`
- `DispatchWriteTrigger`
- `DispatchOrganization`
- `DispatchEvolutionTrigger`
- `DispatchMemoryEvolution`
- `DispatchRetrieval`
- `DispatchReadout`

语义：

- 对输入 `Packet` 做深拷贝
- 每个 child 各自运行
- 所有 child 共享同一个持续演化的 `MemoryStore`
- `primary_index` 决定下游继续走哪条 `Packet`
- 分支 trace 合并到 `trace["dispatch"][slot]`

例子：

```python
DispatchOrganization(
    (
        AppendOrganization(target_layer="working"),
        GraphAppendOrganization(target_layer="knowledge_graph"),
    ),
    primary_index=0,
)
```

### 8.4 Topology + layer-aware retrieval 组装

```python
topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
        StoreLayerSpec(
            name="knowledge_graph",
            theme="knowledge_graph",
            shape="Graph",
            indices=("graph", "entity"),
        ),
    ]
)
store = MemoryStore(topology=topology)

pipeline = MemoryPipeline(
    organization=DispatchOrganization(
        (
            AppendOrganization(target_layer="working"),
            GraphAppendOrganization(target_layer="knowledge_graph"),
        )
    ),
    retrieval=LayerAwareRetrieval(
        default_retriever=RecencyRetrieval(top_k=2),
        retriever_by_layer={"knowledge_graph": EntityRetrieval(top_k=2)},
        top_k=4,
    ),
    store=store,
)
```

### 8.5 Trigger 现场组装

```python
pipeline = MemoryPipeline(
    write_trigger=compose_write_trigger(
        name="demo_threshold_write_trigger",
        signal_providers=(ConstantSignal(signal_name="importance_hint", value=0.8),),
        scorer=WeightedSumScorer(weights={"importance_hint": 1.0}),
        gate=AlwaysOpenGate(),
        policy=ThresholdPolicy(threshold=0.5),
    ),
)
```

## 9. 常见组合模板

### 9.1 最小 append + recency

```python
MemoryPipeline(
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(),
    retrieval=RecencyRetrieval(top_k=2),
    readout=ConcatenateReadout(),
)
```

### 9.2 Layered memory

```python
MemoryPipeline(
    organization=ConditionalLayerOrganization(...),
    retrieval=LayerAwareRetrieval(...),
)
```

### 9.3 Reflexion-like

```python
MemoryPipeline(
    organization=PlacementWithoutAppendOrganization(...),
    evolution_trigger=OutcomeConditionedEvolutionTrigger(...),
    memory_evolution=ReflectionGenerationEvolution(...),
    retrieval=BufferRetrieval(...),
    readout=PromptContextReadout(...),
)
```

### 9.4 Graph baseline

```python
MemoryPipeline(
    organization=GraphAppendOrganization(target_layer="knowledge_graph"),
    evolution_trigger=ThresholdEvolutionTrigger(...),
    memory_evolution=GraphNeighborAppendEvolution(target_layer="knowledge_graph"),
    retrieval=GraphSeedAndExpandRetrieval(layer="knowledge_graph"),
    readout=GraphReadout(),
)
```

### 9.5 A-MEM-like enriched note graph

```python
MemoryPipeline(
    representation=(
        SemanticFieldEnrichmentRepresentation(note_namespace="note"),
        RetrievalOrientedEmbeddingRepresentation(note_namespace="note"),
    ),
    write_trigger=LLMJudgedWriteTrigger(note_namespace="note"),
    organization=GraphAppendLinkReadyOrganization(target_layer="knowledge_graph", note_namespace="note"),
    evolution_trigger=NeighborExistsEvolutionTrigger(target_layer="knowledge_graph"),
    memory_evolution=(
        LinkStrengtheningEvolution(target_layer="knowledge_graph", note_namespace="note"),
        NeighborContextUpdateEvolution(target_layer="knowledge_graph", note_namespace="note"),
    ),
    retrieval=VectorGraphSeedAndExpandRetrieval(layer="knowledge_graph", note_namespace="note"),
    readout=NoteRenderReadout(note_namespace="note"),
)
```

## 10. 使用时要记住的约束

1. 列表是串行，不是 dispatch。
2. dispatch 会共享同一个 store，所以各分支 side effect 会累积。
3. graph 模块要提前声明 graph layer。
4. vector 检索/graph-note 检索需要 vector index 或 embedding。
5. `organization` 是正常 ingest-time write；`memory_evolution` 是额外演化。
6. `write_trigger` 写 `decisions`；`evolution_trigger` 写 `evolution_decisions`。

## 11. 对应示例

- 最小闭环：
  [`memprimitive/example/demonstration/minimal_pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/minimal_pipeline.py)
- trigger 组合：
  [`memprimitive/example/demonstration/composed_triggers.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/composed_triggers.py)
- dispatch + layer-aware retrieval：
  [`memprimitive/example/demonstration/dispatch_organization_recall.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/dispatch_organization_recall.py)
- graph baseline：
  [`memprimitive/example/demonstration/graph_baseline_pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/graph_baseline_pipeline.py)
- A-MEM-like graph cycle：
  [`memprimitive/example/demonstration/amem_like_graph_cycle.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/amem_like_graph_cycle.py)
