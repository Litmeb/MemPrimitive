# 基于 store 声明 embedding 的写入期改造可行性分析

## 背景

目标改造是：

- 不再把“是否生成 embedding”主要绑定在 representation slot。
- 改为在定义 `store` / `layer` 时声明该层需要 embedding。
- 在写入 `store` 或改写 `store record` 时直接生成或刷新 embedding。

我重点核对了两件事：

1. 写入/改写 `store` 的入口是否足够集中，是否只需要改 `append` 和 `replace_record`
2. 这种方案能否覆盖当前仓库里的全部 embedding 使用需求

---

## 结论摘要

### 1. 写入入口是否足够少

结论：`大体集中，但“只改 append 和 replace_record”只能算必要条件，不是完整答案。`

- 从底层 store API 看，新增和改写确实主要收敛在 [`memprimitive/core.py`](./memprimitive/core.py) 的 `MemoryStore.append()`、`MemoryStore.replace_record()`、`MemoryStore.upsert_record()` 上。
- 其中 `upsert_record()` 本身又调用了 `append()` / `replace_record()`，所以如果只看普通 record 新增/更新，改这两个底层入口是合理的第一步。
- 但仓库里存在不少“高层先构造好特殊 record，再写回 store”的路径；这些路径虽然最终也会落到 `append/replace_record`，但它们对 embedding 的语义并不相同，不能简单用“对 record.text 做一次 embed”统一处理。

所以，对问题 1 的直接回答是：

- `如果目标只是统一普通 record 的写入期 embedding 生成，append + replace_record 基本够作为核心切入点。`
- `如果目标是无脑替代现有所有 embedding 逻辑，只改这两个函数不够。`

---

## 对问题 1 的详细判断

### 底层入口确实比较集中

核心证据：

- [`memprimitive/core.py#L517`](./memprimitive/core.py#L517) `append()` 是底层新增入口。
- [`memprimitive/core.py#L638`](./memprimitive/core.py#L638) `replace_record()` 是底层改写入口。
- [`memprimitive/core.py#L659`](./memprimitive/core.py#L659) `upsert_record()` 最终还是走前两者。

绝大多数组织阶段、演化阶段、LLM tool 写入，最后都落到这两个入口。例如：

- [`memprimitive/baselines/organization.py#L75`](./memprimitive/baselines/organization.py#L75) `AppendOrganization`
- [`memprimitive/baselines/organization.py#L108`](./memprimitive/baselines/organization.py#L108) `_append_records_for_placements`
- [`memprimitive/baselines/organization.py#L316`](./memprimitive/baselines/organization.py#L316) `GraphAppendOrganization`
- [`memprimitive/baselines/organization.py#L786`](./memprimitive/baselines/organization.py#L786) `GraphEntityDeduplicationAppendOrganization`
- [`memprimitive/baselines/memory_evolution.py#L179`](./memprimitive/baselines/memory_evolution.py#L179) `SummaryRewriteEvolution`
- [`memprimitive/baselines/memory_evolution.py#L257`](./memprimitive/baselines/memory_evolution.py#L257) `LayerMoveEvolution`
- [`memprimitive/baselines/memory_evolution.py#L424`](./memprimitive/baselines/memory_evolution.py#L424) `GraphLinkEvolution`
- [`memprimitive/utils/_llm_function_tools.py#L403`](./memprimitive/utils/_llm_function_tools.py#L403) 通用 `ADD`
- [`memprimitive/utils/_llm_function_tools.py#L466`](./memprimitive/utils/_llm_function_tools.py#L466) 通用 `UPDATE`
- [`memprimitive/utils/_mem0_family.py#L121`](./memprimitive/utils/_mem0_family.py#L121) Mem0 固定 profile tools

### 但还有两个现实限制

#### 限制 A：有些 rewrite 不是“文本变了就该重算 embedding”

例如 graph link 相关更新：

- [`memprimitive/core.py#L673`](./memprimitive/core.py#L673) `add_graph_links()` 会构造一个新 `MemoryRecord`，但复用旧 `embedding`
- [`memprimitive/utils/_llm_function_tools.py#L630`](./memprimitive/utils/_llm_function_tools.py#L630) `_apply_graph_link_change()` 只改 graph metadata，也不该强制重算 embedding

这说明 `replace_record()` 里如果做自动 embedding，不能简单写成：

- “凡是 replace 就重算 embedding”

更合理的是：

- 只在“目标 layer 声明需要 embedding 且语义内容字段发生变化”时重算
- 对纯 metadata / graph links 更新默认保留旧 embedding

#### 限制 B：当前 `append/replace_record` 没有 embedding 策略信息

`StoreLayerSpec` 现在有 `indices` 和 `settings`，但没有现成的 embedding 策略字段：

- [`memprimitive/core.py#L243`](./memprimitive/core.py#L243) `StoreLayerSpec`
- [`memprimitive/core.py#L274`](./memprimitive/core.py#L274) `get_setting()`

也就是说，架构上能把 embedding 需求挂到 layer settings 里，但当前实现还没有：

- embed 用哪个模型
- embed 的输入文本取 `record.text` 还是某个 payload
- update 时什么变化才触发重算

所以只改 `append/replace_record` 之前，还得先补一层“layer 级 embedding policy”。

---

## 对问题 2 的详细判断

### 能覆盖的需求

如果把目标收窄为：

- “某些 layer 的 `record.embedding` 由 store 写入期统一维护”

那么这套方案是可行的，并且能覆盖很大一部分需求。

典型覆盖范围：

- 普通 flat record 的向量检索
  - [`memprimitive/baselines/retrieval.py#L784`](./memprimitive/baselines/retrieval.py#L784) `EmbeddingSimilarityRetrieval` 只看 `record.embedding`
- 普通 append-only organization 写入出来的 record
- 大多数 summary / move / reflection 这类 append 新 record 的 evolution
- LLM tool 对普通 record 的 add / update

换句话说，`store-declared record embedding` 很适合接管：

- “这个 layer 的 record 就该始终带一个可检索向量”

### 不能完全覆盖的需求

结论：`不能覆盖所有 embedding 使用需求。`

至少有四类需求仍然超出“只在 store 写入点自动补 record.embedding”的范围。

#### 1. unit 级 embedding 仍然存在真实用途

当前 composition contract 里把某些能力建模成 `unit.embedding`：

- [`memprimitive/contracts.py#L7`](./memprimitive/contracts.py#L7) `UNIT_EMBEDDING_CONTRACT`
- [`memprimitive/baselines/representation.py#L196`](./memprimitive/baselines/representation.py#L196) `BasicRepresentation` 在包含 `embedding` 时会产出该 contract
- [`memprimitive/baselines/retrieval.py#L806`](./memprimitive/baselines/retrieval.py#L806) `EmbeddingSimilarityRetrieval` 目前要求这个 contract

虽然 `EmbeddingSimilarityRetrieval` 实际打分看的是 `record.embedding`，但系统的 composition 校验仍然把 embedding 能力挂在 representation slot 上。

这意味着：

- 如果你彻底拿掉 representation slot 对 embedding 的责任，现有 contract 体系会失配。
- 至少还需要同步重构 contract 命名与校验逻辑，把一部分能力从 `unit.embedding` 转成 `record.embedding` / `layer provides embedding`。

#### 2. entity 级 embedding 不能只靠 store layer 声明自动推导

graph entity 路径显式依赖每个 entity 的 embedding，而不是整个 record 的 embedding：

- [`memprimitive/baselines/organization.py#L156`](./memprimitive/baselines/organization.py#L156) `_entity_embedding_map_from_unit()`
- [`memprimitive/baselines/organization.py#L827`](./memprimitive/baselines/organization.py#L827) `GraphEntityDeduplicationAppendOrganization` 读取 `metadata["representation"]["entity_embeddings"]`

这里需要的是：

- `entity -> embedding` 的映射

而不是：

- “给最终写入的 record 生成一个 embedding”

所以这部分仍然要么保留在 representation 阶段，要么单独设计一种更细粒度的 write policy；仅靠 layer 声明不够。

#### 3. note/A-MEM 风格 embedding 不是简单对 `record.text` 做 embed

note-family 里 embedding 来自特制的复合文本，而不是原始 `record.text`：

- [`memprimitive/utils/_amem_family.py#L196`](./memprimitive/utils/_amem_family.py#L196) `rewrite_record_from_note_payload()`
- [`memprimitive/utils/_amem_family.py#L211`](./memprimitive/utils/_amem_family.py#L211) 先 `representation_from_note_payload(...)`
- [`memprimitive/utils/_amem_family.py#L216`](./memprimitive/utils/_amem_family.py#L216) 再对 `enhanced_embedding_text` 做 embed

这类需求说明，写入期生成 embedding 没问题，但 policy 不能只有一种：

- `text` 型：直接 embed `record.text`
- `note_payload` 型：先从 payload 构造增强文本再 embed

如果你打算把 embedding 逻辑下沉到 store，A-MEM 这类路径最好保留“specialized embedding builder”接口，而不是一刀切。

#### 4. query embedding 仍然不属于 store 写入期

检索时 query 本身仍然需要 embedding：

- [`memprimitive/baselines/retrieval.py#L827`](./memprimitive/baselines/retrieval.py#L827) query 没 embedding 时会现场编码

所以即便 store 侧改完，也不能把“embedding 责任”整体理解为完全脱离 representation/runtime。

---

## 对“是否只需要改 append 和 replace 函数”的最终回答

### 短答案

- `工程入口上：基本是。`
- `语义完整性上：不是。`

### 更准确的说法

如果你要做的是第一阶段改造，我建议把目标定义成：

- 让“需要 record-level embedding 的 layer”在 `append()` / `replace_record()` 统一维护 `record.embedding`

这样第一阶段确实可以主要改：

- [`memprimitive/core.py#L517`](./memprimitive/core.py#L517) `append()`
- [`memprimitive/core.py#L638`](./memprimitive/core.py#L638) `replace_record()`

同时补一个内部 helper，例如：

- `_apply_layer_embedding_policy(record, previous_record=None)`

但如果你的目标是：

- 完全移除 representation slot 中所有与 embedding 相关的职责

那就至少还要一起处理：

- composition contracts
- entity embedding
- note payload embedding
- query embedding
- 某些只改 metadata 的 rewrite 路径

---

## 推荐的落地方向

### 建议可行方案：做“部分下沉”，不要“一次性全替”

建议把这次改造定义为：

- `record-level embedding 下沉到 store write path`
- `unit/entity/query/specialized payload embedding 暂时保留在现有模块中`

这样风险最小，也最符合现有代码结构。

### 建议新增的 layer 配置语义

可在 `StoreLayerSpec.settings` 中声明类似：

- `embedding_enabled: true`
- `embedding_mode: "text" | "note_payload"`
- `embedding_model: "..."`
- `embedding_refresh_on_update: "text_change" | "always" | "never"`

如果后面真要覆盖 graph entity，再考虑更高级模式；第一版不建议直接支持。

### 建议保留的例外

以下能力不建议在第一版中强行并入通用 store hook：

- entity embeddings
- triple extraction 产物上的 graph/object embedding
- query embeddings
- 纯 graph link metadata 更新时的 embedding 重算

---

## 最终判断

### 对问题 1

`是，写入/改写入口已经足够集中，append + replace_record 可以作为这次改造的核心落点。`

但要补充一句：

`这只是代码入口层面的集中，不代表语义层面只改两处就自然完整。`

### 对问题 2

`不能完全覆盖所有 embedding 使用需求。`

它可以很好覆盖：

- 普通 layer 的 record-level embedding
- 大多数 append/update 型写入场景

但不能天然覆盖：

- entity-level embedding
- note payload 派生 embedding 的专门构造逻辑
- query embedding
- 现有 `unit.embedding` composition contract

---

## 一句话建议

这条路 `可行，但更适合作为“record embedding 下沉”而不是“embedding 机制完全去 representation 化”`。第一版把普通 record embedding 收口到 `append/replace_record`，同时保留 entity/note/query 这些特例，会更稳。
