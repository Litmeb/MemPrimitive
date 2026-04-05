# 计划新增MODULE

## unit_formation

### `DialogueTurnPairFormation`

- 属于哪个 slot
  - `unit_formation`
- 要补的能力是什么
  - 把增量到来的 user/assistant 一轮交互稳定组织成一个 turn-level memory unit，并保留 speaker 对齐、turn 边界、时间戳与后续 buffer 计数所需字段。
- 为什么现有模块不够
  - `PassThroughUnitFormation` 只能承载上游已经打包好的文本。
  - `MetadataHintUnitFormation` 可以靠 hints 人工构造单元，但没有 LightMem 所需的对话 turn 语义，也不和 buffer 容量机制自然对接。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 conversational memory 方法都需要把“单条消息”提升为“单轮交互”作为真实写入单元。

### `ConversationBundleFormation`

- 属于哪个 slot
  - `unit_formation`
- 要补的能力是什么
  - 把一段待处理的交互上下文稳定封装成 `conversation bundle / interaction chunk`，显式保留消息列表、参与者、时间范围、附件/资源引用、来源 agent 等信息，供后续 memory-type 路由与 typed extraction 使用。
- 为什么现有模块不够
  - `PassThroughUnitFormation` 只能承载上游已打包好的文本。
  - `WindowedUnitFormation` 只能做窗口切分。
  - 当前没有模块把“多轮交互包”作为一等 unit contract 暴露出来，因此无法忠实承载 MIRIX 这类先做 memory-type routing 再做 typed extraction 的写入链路。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是 conversational / multi-agent memory 系统，都可能先对一段交互包做路由和抽取，而不是逐句直接写库。

## representation

### `NERConditionedOpenIERepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 对 passage 先做 named entity extraction，再把 named entities 作为约束条件做第二阶段 triple extraction，稳定产出 entity/noun-phrase、triples、以及后续图组织所需的中间表示。
- 为什么现有模块不够
  - `BasicRepresentation` 虽然能产出 `entities`/`triples`/`embedding`，但没有“先 NER 再受 NER 约束的 OpenIE”这一两阶段机制，也没有显式把图节点级表示边界稳定暴露出来。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 graph-memory / OpenIE-memory 系统都需要“受约束的结构抽取”而不是一次性松散表示。

### `ObservationTripletExtractionRepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 把 step-level observation 稳定解析为 world-fact triplets，作为后续 semantic graph 写入与检索的标准中间表示。
- 为什么现有模块不够
  - `BasicRepresentation` 虽然可以声明 `triple` 元素，但当前仓库里的 triplet extraction 仍不应被视为与 AriGraph 同语义的 observation-conditioned triplet parser；它也没有把“供 semantic graph 组织/修订使用的 triplet payload”作为明确 contract 暴露出来。
- 这是为 AriGraph 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是“交互 observation -> 结构化事实图”的 memory 系统都可能复用。

### `ConversationContextSalientFactRepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 以“当前消息对 + recent-message window + 可选 conversation summary”为输入，抽取 candidate facts / candidate memories，服务 Mem0 这类上下文感知的事实抽取。
- 为什么现有模块不够
  - `BasicRepresentation` 只做 unit-local 的 text/embedding/entities/triples/summary 增强，不会把多轮对话上下文融合成一组待维护的 candidate facts。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 conversational memory 系统都需要“从新交互中抽值得记住的事实”，而不是只对单条文本做表示增强。

### `TypedEntityRelationExtractionRepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 从对话文本中先抽实体及实体类型，再抽关系 triplets，产出可直接送入图组织与图更新的 typed graph payload。
- 为什么现有模块不够
  - `BasicRepresentation(elements=("entities", "triple"))` 只给出松散实体/三元组外观，没有 graph-memory 所需的实体类型边界，也没有稳定的 graph payload contract。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是图记忆、知识图谱式 memory、关系检索式 memory 都可能复用。

### `IterativePreCompressionRepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 在写入前对 turn 文本做 token-level 预压缩，只保留高信息密度内容，同时保留原文与压缩后文本的对应关系，供后续分段和总结使用。
- 为什么现有模块不够
  - `BasicRepresentation` 可以做 summary/embedding/keywords 等增强，但没有“保留哪些 token、丢弃哪些 token”的预压缩语义。
  - LightMem 的效率收益首先来自这一步，而不是普通摘要。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。任何希望在 memory construction 前做轻量压缩降本的系统都可能需要。

### `TopicSegmentSummaryRepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 对 topic segment 级别的多 turn 内容生成 topic-aware summary，并同时产出长期索引用的 embedding-ready 表示。
- 为什么现有模块不够
  - `BasicRepresentation(elements=("summary", "embedding"))` 是 unit-local 的；它不会把“一个 topic segment”作为稳定表示对象。
  - LightMem 的长期记忆 entry 来自 segment-level 总结，不是单 turn 局部增强。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多分段式对话记忆方法都会把 segment summary 当作长期记忆主表示。

### `TypedMemorySchemaExtractionRepresentation`

- 属于哪个 slot
  - `representation`
- 要补的能力是什么
  - 以交互包为输入，抽取一组带 `memory_type` 标签的 typed memory payloads，例如 core block update、episodic event、semantic fact/profile、procedural workflow、resource note、knowledge-vault secret，并为每类 payload 生成后续组织/检索所需字段。
- 为什么现有模块不够
  - `BasicRepresentation` 与现有增强表示模块都默认输出统一记录上的通用字段。
  - MIRIX 需要的是“一次输入，产出多种 schema 候选”的表示能力，而不是在单一 record 上补 `summary` 或 `embedding`。
  - 当前没有模块显式承载 `memory_type -> schema fields -> downstream storage contract` 这一层。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是 typed-memory / multi-store memory 系统都可能需要它。

## write_trigger

> Note
> This document discusses possible future trigger/module additions from the literature side.
> It does not reflect the current public baseline trigger API, which has been simplified to basic slot triggers only.

### `BufferCapacityWriteTrigger`

- 属于哪个 slot
  - `write_trigger`
- 要补的能力是什么
  - 当指定缓冲层的 token 数或条目数达到阈值时触发后续写入阶段，可用于表达 sensory buffer 满触发 segmentation、STM 满触发 summary-to-LTM。
- 为什么现有模块不够
  - `ThresholdWriteTrigger` 只能消费已有 signal 分值，没有现成的 buffer occupancy / token budget 信号。
  - `AlwaysWriteTrigger` 无法表达 LightMem 的批量缓冲触发语义。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是“缓存满了再处理”的 memory pipeline 都会需要。

### `MultiMemoryTypeRoutingWriteTrigger`

- 属于哪个 slot
  - `write_trigger`
- 要补的能力是什么
  - 对当前交互包做 one-to-many 路由，输出本次应更新的 memory type 集合，例如 `episodic + semantic`、`resource only`、`core + knowledge_vault`，供后续组织与维护链路使用。
- 为什么现有模块不够
  - 现有 `write_trigger` 家族最多表达“写 / 不写”或消费已有 metadata。
  - MIRIX 的关键不是是否值得写，而是“应该把这次交互分发到哪些 memory stores”。
  - 当前没有任何 trigger 能稳定输出多标签路由结果。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。多类型记忆系统、工具记忆系统、权限分区记忆系统都可能需要这种路由触发器。

## organization

### `EntityFactPassageGraphOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 把 passage、entity/noun phrase、fact/triple 组织成异构图索引，并保留 node-to-passage incidence 信息，供后续图检索和 passage 聚合使用。
- 为什么现有模块不够
  - `GraphAppendOrganization` 与 `GraphAppendLinkReadyOrganization` 都是 record-centric graph 写入；它们没有 HippoRAG 所需的异构节点类型、fact 边语义、以及节点到 passage 的聚合统计结构。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是“graph over extracted concepts, readout over source documents”的 memory 系统都可能复用。

### `SemanticEpisodicGraphOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 同时维护 semantic memory 与 episodic memory：
    - 把 observation 抽出的 triplets 写入 semantic graph；
    - 把 observation 本身写成 episodic entry；
    - 建立“该 observation 对应这批 triplets”的关联结构。
- 为什么现有模块不够
  - `GraphAppendOrganization` 只能表达普通 graph-layer append；
  - `GraphAppendLinkReadyOrganization` 面向 note graph；
  - 两者都不能自然表达 AriGraph 的 semantic / episodic 双记忆联合组织。
- 这是为 AriGraph 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 world-model memory、事件图 memory、agent episode memory 都需要这种双层组织能力。

### `HierarchicalSubgoalChunkOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 把 working memory 组织成“当前 subgoal 的详细 action-observation chunk + 已完成 subgoal 的归档 chunk”两层结构，并稳定维护 subgoal id、chunk 边界、当前/历史状态。
- 为什么现有模块不够
  - `AppendOrganization` 只能顺序追加；
  - `ConditionalLayerOrganization` 只能按规则分层；
  - `PlacementWithoutAppendOrganization` 只能发 placement；
  - 它们都不能自然表达 HiAgent 所需的“按 subgoal 切 chunk，并让当前 chunk 与历史 chunk 使用不同保真度”的层级 working-memory 组织。
- 这是为 HiAgent 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是“计划-执行-压缩-必要时回看”的 agent working-memory 系统都可能复用。

### `TypedEntityRelationGraphOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 把 typed entities、entity embeddings、relation triplets 组织成 graph-memory 所需的节点/边结构，并稳定保留实体类型、时间戳、作用域 metadata。
- 为什么现有模块不够
  - `GraphAppendOrganization` 只支持 record-centric graph append；
  - 它不能自然表达 Mem0 graph 版所需的“实体节点 + 关系边 + 节点 embedding + 边状态”这套 typed graph organization。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多图记忆方法都需要 typed node/edge organization，而不只是普通 record graph。

### `HierarchicalBufferMemoryOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 显式组织 sensory buffer、topic-aware STM、LTM 三层结构，支持条目先进入短期缓冲，再以批处理方式提升到长期记忆。
- 为什么现有模块不够
  - `AppendOrganization` 只能直接落层。
  - `ConditionalLayerOrganization` 只能做静态路由，不能表达“先缓冲、后提升”的生命周期。
  - 当前没有一个模块能把 LightMem 的三层结构视为一个整体组织原语。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 memory 系统都有 cache/working/LTM 分层，只是 LightMem 把这一点做得更明确。

### `TopicSegmentIndexOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 把一批 turn 组织成 `{topic, message turns}` 形式的 topic segment 索引结构，并维护 segment 到后续 LTM entry 的来源绑定。
- 为什么现有模块不够
  - 当前 organization 家族没有“segment-level 索引对象”概念，只有 record append 或 graph append。
  - LightMem 的 STM 不是普通 layer，而是 topic-aware segment store。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。所有 topic/session/chunk 级记忆系统都可能需要类似的中间组织层。

### `HeterogeneousMemoryStoreOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 把带 `memory_type` 的 typed payload 分发到异构 memory stores，允许不同 store 拥有不同 schema、不同主键/更新策略、不同过滤维度与不同检索字段。
- 为什么现有模块不够
  - `AppendOrganization` 与 `ConditionalLayerOrganization` 都默认在同构记录空间里做追加或路由。
  - MIRIX 不是“同一 record 换个 layer”，而是 block memory、event memory、concept memory、resource memory、sensitive vault 等并存。
  - 当前没有模块把“异构 store family”视为组织层的一等对象。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 production memory systems 都会把不同类型的记忆分到不同 store，而不是共享单一 schema。

### `CoreBlockMemoryOrganization`

- 属于哪个 slot
  - `organization`
- 要补的能力是什么
  - 显式承载 bounded block-style memory，例如 persona block、human-profile block、system-preference block，支持块级重写、容量度量与高优先级提示注入。
- 为什么现有模块不够
  - 当前 organization 家族主要面向 record append、graph append 或 layer placement。
  - 它们都不把“固定少量高价值块状记忆”当作一种独立组织形态，因此无法忠实表达 MIRIX 的 core memory。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 agent memory 系统都会保留一小组高优先级 profile / persona blocks。

## evolution_trigger

### `EmbeddingSimilarityAugmentationTrigger`

- 属于哪个 slot
  - `evolution_trigger`
- 要补的能力是什么
  - 在新 passage/entity 节点写入后，判断是否需要执行基于 embedding 相似度的节点连边增强。
- 为什么现有模块不够
  - `NeighborExistsEvolutionTrigger` 假设已有图邻居或 record-level graph candidate；HippoRAG 更像“新节点写入后，对节点 embedding 空间做候选检索和阈值判断”。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive，但最初需求来自 HippoRAG。

### `SubgoalCompletionEvolutionTrigger`

- 属于哪个 slot
  - `evolution_trigger`
- 要补的能力是什么
  - 当模型或上游控制逻辑判断“当前 subgoal 已完成，应压缩并归档该 chunk”时，触发后续 memory evolution。
- 为什么现有模块不够
  - `ThresholdEvolutionTrigger` 只能做常量/阈值触发；
  - `OutcomeConditionedEvolutionTrigger` 面向 Reflexion 式试次结果；
  - `NewWriteEvolutionTrigger` 面向 keyed/local-maintenance；
  - 当前没有一个显式面向“subgoal 完成”这一 working-memory 切换事件的触发 primitive。
- 这是为 HiAgent 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多分层计划型 agent 都需要“阶段完成后再压缩记忆”的触发边界。

### `AlwaysUpdateEvaluationTrigger`

- 属于哪个 slot
  - `evolution_trigger`
- 要补的能力是什么
  - 对每个已抽出的 candidate fact 统一触发一次“相似旧记忆比较 + 决策维护”的后续流程，明确表达 Mem0 这类固定 update phase。
- 为什么现有模块不够
  - 现有 `ThresholdEvolutionTrigger` 可以勉强充当恒真触发，但语义上不够清晰；Mem0 的特点不是阈值触发，而是 extraction 之后固定进入 update evaluation。
- 这是为 Mem0 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 memory-maintenance 流水线都存在“候选一旦生成，就必须过一次维护决策”的固定后处理阶段。

### `SleepTimeBatchEvolutionTrigger`

- 属于哪个 slot
  - `evolution_trigger`
- 要补的能力是什么
  - 显式表达“在线写入之后不立即维护，等到 sleep-time / offline window / 外部 update signal 到来时，再统一触发 evolution”。
- 为什么现有模块不够
  - `NewWriteEvolutionTrigger` 是在线、局部的新写入触发。
  - `ThresholdEvolutionTrigger` 没有离线批处理或 sleep-time 语义。
  - LightMem 的关键点恰恰是把 update 从在线路径拆出去。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。任何把维护延后到后台或空闲时段的 memory 系统都可能需要。

### `CapacityConditionedBlockRewriteTrigger`

- 属于哪个 slot
  - `evolution_trigger`
- 要补的能力是什么
  - 当 block-style core memory 接近容量上限时，触发 block rewrite / condense / consolidation，而不是继续无约束追加。
- 为什么现有模块不够
  - `ThresholdEvolutionTrigger` 只能消费一般性分数阈值，没有 block fullness / token occupancy 语义。
  - MIRIX 的 core memory rewrite 是 block-aware 维护，而不是普通记录的后处理。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是采用 bounded profile blocks 或 memory blocks 的系统都可能需要。

### `ScheduledReflexionEvolutionTrigger`

- 属于哪个 slot
  - `evolution_trigger`
- 要补的能力是什么
  - 在指定时机触发跨 memory stores 的 cleanup / dedupe / pattern extraction，例如 per-query、periodic、idle-time 或 explicit maintenance window。
- 为什么现有模块不够
  - `OutcomeConditionedEvolutionTrigger` 面向试错反馈。
  - `NewWriteEvolutionTrigger` 面向写后局部维护。
  - 当前没有一个显式表达“为了 memory hygiene 与 higher-order synthesis 而定期整理”的触发器。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多长期运行 agent 都需要独立的维护窗口来做反思、去重和提炼。

## memory_evolution

### `SynonymyEdgeAugmentationEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 基于 entity/noun phrase embedding 的相似度阈值，为图节点批量增加 synonymy/similarity edges。
- 为什么现有模块不够
  - `GraphLinkEvolution` 家族当前面向 record graph/link metadata，不是节点级 embedding augmentation；也不维护 HippoRAG 需要的同义边语义。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它适用于很多“概念图 + 相似节点桥接”的长期记忆方案。

### `OutdatedSemanticEdgePruningEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 基于当前 observation 提取出的新 triplets，对相关旧 semantic edges 做冲突检测、过时事实识别与删除，然后把 semantic memory 修订到最新状态。
- 为什么现有模块不够
  - 当前 `GraphLinkEvolution` / `GraphNeighborAppendEvolution` / `NeighborContextUpdateEvolution` 都偏向新增或改写上下文，不覆盖 AriGraph 关键的“删除式事实修订”。
- 这是为 AriGraph 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。任何持续交互、世界状态会变化的 graph memory 都可能需要局部事实修订，而不只是 append。

### `SubgoalTrajectorySummarizationEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 对“已完成 subgoal”对应的整段 action-observation trajectory 做 subgoal-conditioned summarization，产出可替代旧细节的 summarized observation。
- 为什么现有模块不够
  - `SummaryRewriteEvolution` 虽然能追加 summary record，但它是基于单个 unit 已有 summary/description 字段做 append-only 重写；
  - HiAgent 需要的是“对一个 subgoal chunk 的多步轨迹做摘要”，而不是对单条 unit 文本做轻量改写。
- 这是为 HiAgent 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 working-memory 压缩方案都需要 chunk-level 而非 unit-level 的摘要演化。

### `HierarchicalChunkReplacementEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 在完成 subgoal 后，把旧 detailed chunk 切换成“默认隐藏/归档、但仍可按需恢复”的状态，并让 summarized observation 成为默认暴露给工作记忆的版本。
- 为什么现有模块不够
  - `LayerMoveEvolution` 和 `SummaryRewriteEvolution` 都是 copy-append 风格；
  - 它们不会真正表达 HiAgent 需要的“旧细节被 summary 取代为默认上下文，但详细轨迹仍可回取”的替换/归档语义。
- 这是为 HiAgent 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它对应的是一类“压缩后默认隐藏、但保持可恢复”的 working-memory 生命周期管理。

### `SimilarityResolvedMemoryUpdateEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 对每个 candidate fact 先检索 top-k 相似旧记忆，再由 LLM 或等价决策器输出 `ADD / UPDATE / DELETE / NONE`，并执行实际的 create/update/delete/no-op。
- 为什么现有模块不够
  - 当前 `SummaryRewriteEvolution`、`LayerMoveEvolution`、`GraphLinkEvolution` 都不具备“相似记忆比较 + 四路维护动作决策 + 实际 store mutation”这一组合能力；
  - 这正是 Mem0 主体最核心的机制。
- 这是为 Mem0 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它对应的是一类“candidate fact 经过 similarity-based maintenance policy 后再决定落库”的长期记忆维护模式。

### `GraphConflictInvalidationEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 对新进入的关系 triplets 检测与既有图关系的冲突，并把过时关系标记为失效而不是简单追加或硬删除。
- 为什么现有模块不够
  - `GraphLinkEvolution` 偏向新增链接；
  - `NeighborContextUpdateEvolution` 偏向重写邻居上下文；
  - 当前没有一个明确表达“冲突关系失效化 / temporal invalidation”的图维护 primitive。
- 这是为 Mem0 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是需要保留时间一致性或历史状态的图记忆系统都可能需要这种 invalidation 机制。

### `TimestampConstrainedUpdateQueueEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 为每个长期记忆条目构造一个基于 embedding 相似度的 top-k update queue，并加入时间约束，只允许较新的条目成为较旧条目的候选更新来源。
- 为什么现有模块不够
  - 当前 `memory_evolution` 家族没有“先生成全局 update queue 再更新”的阶段化机制。
  - `SummaryRewriteEvolution`、`LayerMoveEvolution` 都不会产出可复用的候选更新队列。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是采用“先找候选，再做维护决策”的离线更新系统都可能复用。

### `ParallelQueueResolvedUpdateEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 以 `target entry + update_queue candidate sources` 为输入，执行离线并行的 `update / delete / ignore` 决策与 store mutation。
- 为什么现有模块不够
  - 当前没有任何 evolution 模块能把“候选来源集合”作为更新上下文，并支持批量并行执行。
  - 这正是 LightMem 与在线逐条维护系统的关键差异。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多低延迟 memory 系统都会受益于把重维护搬到后台并行处理。

### `TypedMemoryUpdateEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 针对不同 memory type 执行 `update / merge / replace / skip` 维护动作，并允许每类 memory 使用自己的字段对齐、重复检测与 embedding 刷新策略。
- 为什么现有模块不够
  - `SummaryRewriteEvolution`、`AppendOnlyEvolution`、`LayerMoveEvolution` 都默认面对较统一的记录操作。
  - MIRIX 需要的是 typed maintenance family，而不是单一 rewrite 或 append 模式。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。多 schema memory 系统普遍需要按类型区分维护语义。

### `CrossStoreDeduplicationEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 对多个 memory stores 进行跨库去重、清理与一致性整理，避免同一信息在 episodic / semantic / resource 等 store 中以失控方式重复堆积。
- 为什么现有模块不够
  - 现有 evolution 模块都主要在单 layer 或单图结构内部工作。
  - 当前没有模块显式把“多个 store 之间的重复与冗余”当作维护对象。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。只要系统同时维护多类记忆库，就可能需要这种 hygiene primitive。

### `BehaviorPatternSynthesisEvolution`

- 属于哪个 slot
  - `memory_evolution`
- 要补的能力是什么
  - 从 episodic / interaction histories 中提炼更高层行为模式、偏好或生活规律，并把结果回写到 semantic 或其他更稳定的 memory store。
- 为什么现有模块不够
  - `ReflectionGenerationEvolution` 只能局部覆盖“生成反思文本”的外观。
  - 它不负责从一个 store 读、提炼 pattern、再写回另一个 typed store。
  - MIRIX Reflexion 的关键之一正是这种跨类型的 higher-order synthesis。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 agent memory 系统都希望把经历逐步沉淀成更稳定的偏好、规则或 profile。

## retrieval

### `QueryEntityLinkingPPRRetrieval`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 在 retrieval 内完成 query named entity extraction、entity-to-node linking、query seed weighting、Personalized PageRank 扩散，以及 passage 排名输出。
- 为什么现有模块不够
  - `VectorGraphSeedAndExpandRetrieval` 与 `ExpandRetrievedGraphNeighbors` 只有 seed + expand 的近似骨架，没有 query linking、PPR、node specificity 与 passage incidence 聚合。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它对应的是一类“graph propagation retrieval”而不只是一篇论文。

### `NodeToPassageAggregationRetrievalAdapter`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 把图节点分数按 node-to-passage incidence 统计聚合成 source passage 分数，形成真正的文档级检索输出。
- 为什么现有模块不够
  - 当前 retrieval 模块默认直接对 record 打分或做 record 邻接扩展，没有“先节点打分、再回投到 passage”的两级检索接口。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它是 graph-memory 文档检索中的常见桥接层。

### `SemanticEpisodicGraphRetrieval`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 先从 query 中检索相关 semantic triplets 并做受控图扩展，再基于这些 triplets 回连 episodic memories，对 past observations 打分并返回 top-k。
- 为什么现有模块不够
  - 现有 graph retrieval 仍只有“先选 seed 再做一跳 expand”的轮廓，没有 AriGraph 所需的“semantic retrieval -> episodic retrieval”两段式检索；
  - `EmbeddingSimilarityRetrieval` 也不能把 semantic 检索结果进一步转成 episodic recall。
- 这是为 AriGraph 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它对应的是一类“图事实检索后回连事件/轨迹记忆”的 agent-memory 模式。

### `SubgoalTrajectoryRetrieval`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 按 subgoal id 检索某个已归档 subgoal 的完整 action-observation trajectory，并把它重新暴露到当前工作记忆上下文。
- 为什么现有模块不够
  - `BufferRetrieval` 只会读最近窗口；
  - `RecencyRetrieval` 也是时间顺序取回；
  - `LayerAwareRetrieval` 只能分发已有 retriever；
  - 当前没有一个“按 chunk 标识符精确回取旧 detailed trajectory”的 retrieval primitive。
- 这是为 HiAgent 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是采用 chunk/archive 记忆组织的 agent 都可能需要按 chunk id 精确恢复细节。

### `EntityCentricTripletHybridRetrieval`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 同时支持：
    - 从 query 中抽关键实体并锚定到图节点；
    - 沿图做实体中心的关系扩展；
    - 再对 triplet 文本表示做语义匹配与重排；
  - 作为 Mem0 graph 版的 retrieval 主干。
- 为什么现有模块不够
  - `EmbeddingSimilarityRetrieval` 只覆盖基础向量检索；
  - `VectorGraphSeedAndExpandRetrieval` 与 `ExpandRetrievedGraphNeighbors` 只覆盖通用 seed-expand 骨架，没有 Mem0 graph 版的 entity-centric + triplet semantic retrieval 双路径。
- 这是为 Mem0 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它对应的是一类“图锚定检索 + triplet 语义匹配”的图记忆检索模式。

### `MultiStoreFieldAwareRetrieval`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 在多个异构 memory stores 上统一编排检索，允许每个 store 使用不同 searchable fields、不同 search method 组合、不同过滤条件，并把结果整合成 typed retrieval result。
- 为什么现有模块不够
  - `BM25Retrieval`、`EmbeddingSimilarityRetrieval` 只覆盖单一检索策略。
  - `LayerAwareRetrieval` 只能做较浅层的多层分发。
  - MIRIX 需要的是“按 memory type 选择字段、策略与过滤器”的 retrieval orchestrator。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。凡是 multi-store memory 架构都会需要类似能力。

### `DualViewEpisodicRetrieval`

- 属于哪个 slot
  - `retrieval`
- 要补的能力是什么
  - 对 episodic memory 同时产出 `recent` 与 `relevant` 两路结果，分别服务时间近因与主题相关性，并把两者作为不同视图交给 readout。
- 为什么现有模块不够
  - `RecencyRetrieval` 与 `EmbeddingSimilarityRetrieval` 各自只能做一条路。
  - 当前没有模块把“同一 episodic store 上的双视图召回”作为一个稳定接口输出。
- 这是为 HippoRAG 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。很多 episodic-memory 系统都同时需要“最近发生了什么”和“最相关的旧经历”。

## readout

### `HierarchicalWorkingMemoryReadout`

- 属于哪个 slot
  - `readout`
- 要补的能力是什么
  - 把“历史 subgoal 的 summary、当前 subgoal 的详细轨迹、按需恢复的旧 subgoal 细节”组装成稳定 prompt 上下文，并显式保留 subgoal 编号与层级关系。
- 为什么现有模块不够
  - `ConcatenateReadout`、`BulletListReadout`、`GroupedByLayerReadout` 只能做通用拼接；
  - `PromptContextReadout` 当前偏向 Reflexion 场景；
  - 它们都不能直接表达 HiAgent 的层级 working-memory 呈现格式。
- 这是为 HiAgent 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。它服务的是一类“压缩历史 + 保留当前细节 + 按需再展开”的 prompt-memory 读出模式。

### `TypedMemoryPromptReadout`

- 属于哪个 slot
  - `readout`
- 要补的能力是什么
  - 把多 store retrieval results 按 memory type 组装为稳定 prompt 片段，支持 typed section headers、字段裁剪、可选 item ids 与每类记忆的展示模板。
- 为什么现有模块不够
  - `GroupedByLayerReadout` 只会按 layer 分组。
  - `PromptContextReadout` 只提供通用 prompt 注入外观。
  - MIRIX 需要的是显式的 typed-memory prompt assembly，而不是普通文本拼接。
- 这是为 MIRIX 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。多类型记忆系统几乎都需要按类型组织给下游模型看的上下文。

### `SensitivityAwareMemoryReadout`

- 属于哪个 slot
  - `readout`
- 要补的能力是什么
  - 在读出阶段结合 sensitivity、owner、scope 等权限信息，对 retrieval results 做过滤、降级展示或屏蔽，避免把不该暴露的信息直接注入 prompt。
- 为什么现有模块不够
  - 当前 readout 家族默认“检索到了就读出”。
  - MIRIX 的 knowledge vault 明显存在额外的可见性边界，这不是普通 `ConcatenateReadout` 或 `PromptContextReadout` 能表达的。
- 这是为 MIRIX 特化，还是可抽象为通用 primitive
  - 可抽象为通用 primitive。只要记忆库里混有敏感信息、私有信息或不同 agent 的作用域信息，就会需要。
