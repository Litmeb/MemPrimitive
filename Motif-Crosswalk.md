# Classic Memory Motif Crosswalk

本文档服务于**展示与说明**。

目标是把当前已经实现的经典 family 中出现的 recurring motifs 摊开来展示，并明确区分：

- 哪些是值得放进 DSL 的 recurring motif
- 哪些只是具体方法留下的 residual

推荐阅读方式：

- 把每一行看成一个“候选 primitive family”
- 不要求任何一行都能单独重建一篇 paper
- 重点看：这个 motif 是否跨方法复现，以及它是否值得进入可搜索空间

---

## 1. 总表

| Motif | 来自哪些方法 | 放在哪个 DSL 层 | 需要哪些约束 | residual 是什么 |
| --- | --- | --- | --- | --- |
| `structured_extraction` | TiM | 模块层 `unit_formation` | 需要 `observation.text`；若为 relation-aware 版本，需定义 unit metadata namespace | TiM 的具体 `head/relation/tail/triple` 字段形状与提示词 |
| `provenance_attachment` | TiM，广义上也适合其他方法 | 模块层 `unit_formation` 的通用子 motif | 需要 observation id / source 可用 | 具体 provenance 字段名 |
| `semantic_field_enrichment` | TiM，A-MEM | 模块层 `representation` | 需要 unit text；如使用 LLM 需 LLM runtime；如写 embedding 字段需 embedder | A-MEM note JSON 的精确字段、TiM 的字段命名习惯 |
| `partition_key_derivation` | TiM，MemGPT 的上游逻辑 | 模块层 `representation` | 需要可稳定生成的 partition key；下游 organization/retrieval 必须消费该 key | TiM 的 hash 细节、MemGPT 的具体 key 名称 |
| `retrieval_oriented_embedding_construction` | A-MEM | 模块层 `representation` | 需要 enriched fields 与 embedder | A-MEM enhanced embedding text 的模板写法 |
| `metadata_gated_write` | TiM | 模块层 `write_trigger` | 需要 upstream metadata flag | flag 的具体字段路径 |
| `llm_judged_write` | A-MEM | 模块层 `write_trigger` | 需要 LLM；最好已有 context/tags/keywords | 具体 write-controller prompt 与 JSON schema |
| `append_to_layer` | TiM，A-MEM，也可泛化到多数方法 | 模块层 `organization` | 需要 target layer 存在；`units/decisions` 对齐 | family-specific metadata 的具体落盘方式 |
| `keyed_upsert` | MemGPT | 模块层 `organization` | 需要稳定 key；需要目标 layer；需要 replace 语义 | MemGPT 的五区命名、`memgpt_key` 字段名 |
| `graph_append_link_ready` | A-MEM | 模块层 `organization` | 需要 graph layer；需要 `index:graph` 和 `index:vector` | graph metadata 的具体 JSON 形状 |
| `placement_without_append` | Reflexion | 模块层 `organization` | 需要 packet placements 但不落盘 trial；下游 evolution 需能读取 observation metadata | trial buffer 的具体命名与 benchmark 语义 |
| `outcome_conditioned_trigger` | Reflexion | 模块层 `evolution_trigger` | 需要 outcome/feedback schema | `is_correct`、`feedback` 等具体字段名 |
| `new_write_trigger` | TiM | 模块层 `evolution_trigger` | 需要知道哪些 unit 已写入；通常与 family-specific unit type 绑定 | TiM 对 `tim_thought` 的具体假设 |
| `neighbor_exists_trigger` | A-MEM | 模块层 `evolution_trigger` | 需要 embeddings；需要 vector retrieval | A-MEM 的 candidate 选择细节 |
| `local_conflict_forgetting` | TiM | 模块层 `memory_evolution` | 需要 stable partition；需要 local candidate set；常需 LLM 或规则 judge | TiM forget prompt 与 bucket 内冲突定义 |
| `local_merge_compression` | TiM | 模块层 `memory_evolution` | 需要 stable partition；需要 merge policy | TiM merge prompt、fallback triple merge 细节 |
| `reflection_generation` | Reflexion | 模块层 `memory_evolution` | 需要 question、last attempt、feedback、LLM | Reflexion 的 reflection wording 与 HotPotQA-style schema |
| `link_strengthening` | A-MEM | 模块层 `memory_evolution` | 需要 graph layer；需要 neighbor candidates；需要 link writeback | 连接选择 prompt、tag 更新 prompt |
| `neighbor_context_update` | A-MEM | 模块层 `memory_evolution` | 需要可重写 neighbor payload；需要 LLM | 邻居 context/tags 的具体 schema |
| `buffer_retrieval` | Reflexion | 模块层 `retrieval` | 需要目标 layer 与时间顺序；通常不需要 query-dependent search | reflection layer 的具体用途设定 |
| `partition_first_retrieval` | TiM | 模块层 `retrieval` | 需要 partition key；query 也可投影到 partition；需要 partition-local ranker | TiM hash / bucket distance 设计 |
| `paged_similarity_retrieval` | MemGPT | 模块层 `retrieval` | 需要 embeddings；需要 page/page_size query contract | tool 名、分页元数据格式 |
| `seed_and_expand_retrieval` | A-MEM | 模块层 `retrieval` | 需要 vector index、graph links；可选 reranker | query expansion prompt、agentic rerank 策略 |
| `plain_list_readout` | TiM | 模块层 `readout` | 需要 `retrieved.items` | TiM 风格文本排版 |
| `prompt_context_readout` | Reflexion | 模块层 `readout` | 需要 query、retrieved items、strategy metadata | Reflexion header wording 与拼接格式 |
| `tool_payload_readout` | MemGPT | 模块层 `readout` | 需要 retrieved items 与分页 trace；下游必须接受 tool JSON | MemGPT agent loop 的工具协议 |
| `note_render_readout` | A-MEM | 模块层 `readout` | 需要 enriched note payload | A-MEM 输出格式的文案细节 |
| `layer_role_partitioning` | MemGPT，Reflexion，TiM，A-MEM | 结构层 | 需要显式 `StoreTopology`；layer role 要与下游模块一致 | 某篇 paper 的具体层命名与角色解释 |
| `graph_memory_topology` | A-MEM | 结构层 | 需要 `shape=\"Graph\"` 和 graph index | A-MEM 单图层方案的具体命名 |
| `partition_addressable_flat_memory` | TiM，MemGPT | 结构层 | 需要 record 可按 hash/key/entity 等稳定寻址 | 具体 partition key 算法 |
| `required_index_family` | A-MEM，MemGPT，也可泛化 | 结构层 / 约束层交界 | 需要 layer index 声明；下游模块要校验 | 某 family 对 index 名称的硬编码 |
| `budget_control` | TiM，Reflexion，A-MEM | 参数层 | 需要模块支持 budget / memory size / candidate_k 等参数 | 各 paper 选定的默认值 |
| `agentic_search_toggle` | A-MEM | 参数层 | 需要 retrieval 模块支持 rerank / query expansion | prompt 与 reranker 细节 |
| `safe_search_mode` | 全部 family | 约束层 | 需要额外 searchable metadata；需要 runtime 或 search planner 理解耦合级别 | 当前代码中尚未统一声明的 family-specific 假设 |

---

## 1.1 Trigger 四件套 Crosswalk

这一节专门回答：

> 如果把 classic 方法里的 trigger 强制改写成  
> `signal_providers + scorer + gate + policy`，每家会长什么样？

注意：

- Reflexion / A-MEM / TiM 原论文并没有都显式采用这四件套名称。
- 下表是基于原论文与官方仓库语义做的**结构化分解**。
- 因此它的价值主要是帮助你未来实现，而不是声称论文作者原本就是这样写的。

| 方法 / trigger | signal_providers | scorer | gate | policy | 如何拼接成当前 trigger |
| --- | --- | --- | --- | --- | --- |
| TiM `write_trigger` | `UnitTypeSignal(tim_thought)` + `MetadataFlagSignal(metadata.tim.write)` | `MinScorer` 或布尔 identity | `AlwaysOpenGate` | `ThresholdPolicy(1.0)` 或 `BooleanGatePolicy` | 当前实现相当于把“是 TiM thought 且 write 没被禁用”写成了单个判定；四件套化后就是先生成两个条件信号，再聚合成最终写入决策 |
| Reflexion `evolution_trigger` | `OutcomeCorrectnessSignal` + `FeedbackPresenceSignal` | `IdentityScorer(trial_failed)` | `FeedbackSchemaGate` | `BooleanGatePolicy` 或 `ThresholdPolicy(1.0)` | 当前实现先从 metadata 推断 `is_correct`，再对所有 unit 复制 `not is_correct`；四件套化后就是 failure signal 经 scorer/policy 进入最终触发 |
| TiM `evolution_trigger` | `UnitTypeSignal(tim_thought)` + `PlacementExistsSignal` + `PartitionKeyPresentSignal` | `MinScorer` | `LayerAllowedGate(thought_memory)` | `ThresholdPolicy(1.0)` | 当前实现的“新写入 thought 就触发局部维护”在四件套里可表达为：只要已落到 thought layer 且带 bucket key，就触发 insert/forget/merge 后续链路 |
| A-MEM `write_trigger` | `ContentRichnessSignal` + `KeywordDensitySignal` + `TagDensitySignal`，或合并成 `LLMWriteAssessmentSignal` | `WeightedSumScorer` 或 `IdentityScorer(store_likelihood)` | `SchemaPresentGate(amem note ready)` | `ThresholdPolicy` | 当前实现把 LLM 的 `decision/confidence` 直接当最终写入结果；四件套化后可把 confidence 当 score，把 note schema 完整性当 gate |
| A-MEM `evolution_trigger` | `NeighborCountSignal` + 可选 `TopNeighborSimilaritySignal` | `IdentityScorer(neighbor_exists)` 或 `WeightedSumScorer` | `HasEmbeddingGate` + `VectorIndexReadyGate` + `GraphLayerGate` | `ThresholdPolicy(1.0)` 或 `BooleanGatePolicy` | 当前实现先做 candidate retrieval，只要存在邻居就触发；四件套化后就是邻居存在性 signal 经 gate/policy 进入 evolution |

### Trigger 分解的直接启发

从这张表能看出，三家 trigger 的真正差异主要不在 policy，而在 signal 来源：

- Reflexion 依赖 **external feedback signal**
- TiM 依赖 **new write / partition readiness signal**
- A-MEM 依赖 **neighbor availability / graph readiness signal**

这意味着未来实现时，`scorer` / `policy` 很可能可以高度复用，而 `signal_providers` 与部分 `gate` 才是 classic motif 的主要分化点。

---

## 2. 按方法看 motif 组合

这一节不是为了“重表达整篇 paper”，而是为了把方法拆成：

`method = motif composition + residual`

### 2.1 TiM

- 可归纳出的 motif：
  - `structured_extraction`
  - `semantic_field_enrichment`
  - `partition_key_derivation`
  - `metadata_gated_write`
  - `append_to_layer`
  - `new_write_trigger`
  - `local_conflict_forgetting`
  - `local_merge_compression`
  - `partition_first_retrieval`
  - `plain_list_readout`
- 主要 residual：
  - thought schema 的具体字段
  - hash bucket 的具体设计
  - merge/forget 的 prompt 和 fallback 逻辑
- trigger 四件套说明：
  - `write_trigger` 更像 metadata-gated write
  - `evolution_trigger` 更像 new-write-triggered local maintenance
  - 两者都不需要复杂 policy，核心在 signal 是否表明“新 thought 已准备好进入局部维护链路”

### 2.2 Reflexion

- 可归纳出的 motif：
  - `placement_without_append`
  - `outcome_conditioned_trigger`
  - `reflection_generation`
  - `buffer_retrieval`
  - `prompt_context_readout`
- 主要 residual：
  - HotPotQA / trial-style metadata schema
  - reflection 的文案模板
  - benchmark-specific feedback 字段
- trigger 四件套说明：
  - Reflexion trigger 的本体不是复杂打分，而是 “failure signal exists”
  - 因此它天然适合 `OutcomeCorrectnessSignal -> IdentityScorer -> FeedbackSchemaGate -> Boolean/ThresholdPolicy`

### 2.3 MemGPT

- 可归纳出的 motif：
  - `layer_role_partitioning`
  - `keyed_upsert`
  - `paged_similarity_retrieval`
  - `tool_payload_readout`
- 主要 residual：
  - 五区 store 的具体命名与 agent loop 协议
  - tool 名称
  - `memgpt_key` 字段路径与 block conventions

### 2.4 A-MEM

- 可归纳出的 motif：
  - `graph_memory_topology`
  - `semantic_field_enrichment`
  - `retrieval_oriented_embedding_construction`
  - `llm_judged_write`
  - `graph_append_link_ready`
  - `neighbor_exists_trigger`
  - `link_strengthening`
  - `neighbor_context_update`
  - `seed_and_expand_retrieval`
  - `note_render_readout`
- 主要 residual：
  - comprehensive note 的具体字段规范
  - evolution prompt 细节
  - agentic rerank / query expand 的 prompt 设计
- trigger 四件套说明：
  - `write_trigger` 的核心是 agentic assessment
  - `evolution_trigger` 的核心是 neighbor existence
  - 因此 A-MEM 最适合把复杂性留在 `signal_providers`，而不是把整个 trigger 做成一个不可拆的黑箱

---

## 3. 哪些 motif 最像 recurring motifs

如果从“跨方法复现 + 适合进入搜索空间”两个角度同时看，最值得优先关注的是：

- `semantic_field_enrichment`
- `partition_key_derivation`
- `metadata_gated_write` / `llm_judged_write`
- `keyed_upsert`
- `outcome_conditioned_trigger`
- `local_merge_compression`
- `buffer_retrieval`
- `partition_first_retrieval`
- `seed_and_expand_retrieval`
- `prompt_context_readout`
- `tool_payload_readout`

这些 motif 的共同点是：

- 它们不是只出现于一篇方法中的“命名技巧”。
- 它们确实改变 memory 行为。
- 它们有希望变成可组合、可搜索、可约束的 primitive family。

---

## 4. 哪些东西应该继续留在 residual

下面这些目前更适合被明确标注为 residual，而不是直接进入 DSL 核心：

- paper-specific prompt wording
- benchmark-specific metadata schema
- 某篇实现特有的字段命名
- 某篇方法绑定的 agent loop protocol
- 某个 hash / merge / rerank 细节的具体启发式

换句话说，DSL 应该说：

- “这里用了 keyed overwrite”
- “这里用了 failure-triggered reflection”
- “这里用了 vector seed + graph expansion”

而不应该一开始就说：

- “这里必须有 `memgpt_key`”
- “这里必须有 `reflexion.feedback`”
- “这里必须有 `amem.note_text`”

---

## 5. 一个推荐的表述模板

以后如果你要在论文、README、搜索配置或实现文档里描述一个方法，建议统一用下面这个模板：

`Method = structural priors + motif composition + residual`

例如：

- `TiM = partition-addressable flat memory + structured extraction + local maintenance + partition-first retrieval + residual(TiM thought schema)`
- `Reflexion = reflection layer + outcome-conditioned trigger + reflection generation + buffer retrieval + residual(task feedback schema)`
- `MemGPT = role-partitioned layers + keyed upsert + paged similarity retrieval + tool payload readout + residual(agent tool protocol)`
- `A-MEM = graph memory topology + rich note representation + neighbor-triggered graph evolution + seed-and-expand retrieval + residual(agentic prompts)`

这样有三个好处：

- 你不再被“必须完整重演 paper”卡住。
- 你可以清楚区分可组合机制与 paper residual。
- 未来搜索时可以直接对 motif composition 做约束化枚举。
