# MemPrimitive DSL 参考（当前实现面，详细版）

本文档按当前仓库代码重写，目标是把它当成一份可直接查阅的 API / DSL 参考，而不是设计草案。内容以 `memprimitive.pipeline`、`memprimitive.baselines`、`memprimitive.utils._template`、`memprimitive.utils._llm_function_tools`、`memprimitive.utils._mid_decoding_tools` 的现状为准。

本文档覆盖三类内容：

1. `MemoryPipeline` 的真实 slot 顺序与 `Packet`/`MemoryStore` I/O 约定。
2. 当前公开 baseline surface 中每个 module 的功能、输入、输出、关键参数、特殊行为。
3. 与 module 紧密耦合的两套协议：
   1. 写路径 tool call 协议。
   2. `PromptPlan` / 模板系统。

## 1. Pipeline 总览

### 1.1 标准 slot 顺序

`MemoryPipeline` 的执行顺序固定为：

1. `unit_formation`
2. `representation`
3. `write_trigger`
4. `organization`
5. `evolution_trigger`
6. `memory_evolution`
7. `retrieval`
8. `readout`

其中：

1. `ingest(observation)` 只跑前六个 slot。
2. `recall(query)` 只跑后两个 slot。
3. `run_round(observation, query)` 先 ingest 再 recall，并把 ingest trace 塞进最终 `Readout.metadata["ingest_trace"]`。

### 1.2 `Packet` 是所有 module 的共享 I/O 容器

当前 `Packet` 主要字段如下：

1. `packet.observation: Observation | None`
   ingest 入口输入。
2. `packet.units: list[MemoryUnit] | None`
   unit 级中间表示。
3. `packet.decisions: list[bool] | None`
   当前 trigger 生成的 active mask。
4. `packet.decisions_store: dict[str, dict[str, Any]] | None`
   trigger 选中的“已有 store records”的并行结果。
5. `packet.placements: list[Placement] | None`
   organization 为每个 unit 规划的目标 layer。
6. `packet.query: Query | None`
   recall 单 query 入口。
7. `packet.queries: list[Query] | None`
   recall 多 query 入口。
8. `packet.retrieved: RetrievedSet | None`
   retrieval 输出。
9. `packet.readout: Readout | None`
   readout 输出。
10. `packet.trace: dict[str, Any]`
   每个 slot 的 trace 都写在这里。

### 1.3 `decisions` 和 `decisions_store` 的区别

这两个字段经常一起出现，但语义不同：

1. `packet.decisions`
   面向“当前这批 unit”。
   典型用途是告诉 organization / memory_evolution 哪些 unit 当前激活。
2. `packet.decisions_store`
   面向“store 里已有 records”。
   典型用途是告诉层级聚合、演化、tool-call 维护阶段，哪些历史记录被选中。

### 1.4 retrieval 的 query 解析优先级

所有支持多 query 的 retrieval module 都遵守同一套规则：

1. 如果 `packet.queries` 非空，则逐个 query 跑检索。
2. 否则回退到 `packet.query`。
3. 多 query 合并策略固定为 `query_order_dedupe`：
   1. 先保留每个 query 自己的排序。
   2. 按 query 顺序拼接结果。
   3. 用 `record_id` 去重，重复命中只保留第一次出现的那条。

合并后，`packet.retrieved.trace` 会额外带上：

1. `query_count`
2. `query_ids`
3. `merge_strategy`
4. `per_query`
5. `final_returned_count`

### 1.5 文档里的“输入 / 输出”怎么读

后面每个 module 都按下面四个维度介绍：

1. 功能
2. 读取哪些变量
3. 写回哪些变量
4. 主要参数与特殊行为

这里的“写回”默认指：

1. 是否改 `packet`
2. 是否改 `store`
3. 是否写 trace

## 2. `unit_formation`

当前公开 module：

1. `PassThroughUnitFormation`
2. `SentenceSplitUnitFormation`

### 2.1 `PassThroughUnitFormation`

1. 功能
   把一条 `Observation` 原样转成一个 `MemoryUnit`，不做切分。
2. 读取
   读取 `packet.observation`。
3. 写回
   写 `packet.units`，长度恒为 1。
   写 `packet.trace["unit_formation"]`。
   不改 `store`。
4. 重要输出位置
   输出 unit 的文本在 `packet.units[0].text`。
   unit id 在 `packet.units[0].unit_id`。
5. 特殊行为
   这是 `MemoryPipeline` 的默认 `unit_formation`。
   适合所有“不需要先切分文本”的场景。

### 2.2 `SentenceSplitUnitFormation`

1. 功能
   按句子边界把 `Observation.text` 拆成多个 `MemoryUnit`。
2. 读取
   读取 `packet.observation`。
3. 写回
   写 `packet.units`，每个句子一个 unit。
   写 `packet.trace["unit_formation"]`，包括切分后的 unit 数量。
   不改 `store`。
4. 特殊行为
   如果原始文本只包含一条句子，结果可能仍然只有一个 unit。
   适合句子级写入、句子级筛选、句子级摘要的 pipeline。

## 3. `representation`

当前公开 module：

1. `BasicRepresentation`
2. `TripleRepresentation`
3. `LLMRepresentation`
4. `ConfigurableEmbeddingRepresentation`

### 3.1 `BasicRepresentation`

1. 功能
   为 unit 构造基础表示，支持文本归一化、embedding、关键词、时间锚点、来源类型。
2. 读取
   读取 `packet.units`。
   可能读取每个 unit 的已有字段和 `unit.metadata` 中的提示值，比如 `metadata["keywords"]`、`metadata["time_anchor"]`、`metadata["source"]`。
3. 写回
   回写 `packet.units` 中每个 unit 的：
   1. `normalized_text`
   2. `embedding`
   3. `representation_elements`
   4. `metadata["representation"]`
   写 `packet.trace["representation"]`。
   不改 `store`。
4. 主要参数
   1. `elements`
      控制要生成哪些表示元素。当前合法值是：
      1. `text`
      2. `embedding`
      3. `keywords`
      4. `time_anchor`
      5. `source_type`
   2. `embedding_model`
      文本 embedding 模型。默认读环境变量，否则回退到 `sentence-transformers/all-MiniLM-L6-v2`。
   3. `embedding_provider` / `embedding_api_key` / `embedding_base_url`
      可选 embedding 后端配置。默认 `sentence_transformers`，走本地 `sentence-transformers`；设为 `openai` 时，使用 OpenAI-compatible embeddings API，并读取独立的 `MEMPRIMITIVE_EMBEDDING_API_KEY` / `MEMPRIMITIVE_EMBEDDING_BASE_URL` / `MEMPRIMITIVE_EMBEDDING_MODEL`。
   4. `api_key` / `base_url` / `model`
      当前这个类本身主要用于统一 runtime 配置来源；真正用到的是 embedding 路径。
5. 输出字段说明
   1. `unit.normalized_text`
      当前是 `unit.text.strip().casefold()`。
   2. `unit.embedding`
      如果启用了 `embedding`，这里会写入向量。
   3. `unit.metadata["representation"]`
      会同步写入紧凑摘要，至少包含：
      1. `elements`
      2. `text`
      3. `normalized_text`
      如果有 embedding，还会带 `embedding.dim`。
6. 特殊行为
   1. `keywords`
      如果 `unit.metadata["keywords"]` 已经给出列表，会优先用它；否则从文本里抽取高频词。
   2. `time_anchor`
      优先用 `unit.metadata["time_anchor"]`，否则从 `unit.timestamp` 生成。
   3. `source_type`
      读的是 `unit.metadata["source"]`，不是 `Observation.source` 直读。
   4. 如果重复运行，不会清空已有 triples / entities，只是在已有 unit 上补字段。

### 3.2 `TripleRepresentation`

1. 功能
   用 LLM 或 metadata hint 抽取图谱三元组表示，把结果写到 unit 的 `entities`、`triples`、可选 embedding 中。
2. 读取
   读取 `packet.units`。
   可能读取：
   1. `unit.metadata["triples"]`
   2. `unit.metadata["entities"]`
   3. `self.prompt` 对应的模板上下文
3. 写回
   回写每个 unit 的：
   1. `unit.entities`
   2. `unit.triples`
   3. `unit.embedding`（可选）
   4. `unit.representation_elements`
   5. `unit.metadata["representation"]`
   写 `packet.trace["representation"]`。
   不改 `store`。
4. 主要参数
   1. `method`
      取值只能是：
      1. `direct`
         一次 LLM 调用同时抽 entities 和 relationships。
      2. `two_stage`
         先抽 entities，再以这些 entities 为 anchor 抽 relationships。
   2. `prompt`
      可选。可以是字符串模板或完整 `PromptPlan`。传入后会影响三元组抽取标准。
   3. `embed_extracted`
      是否对“抽取出的 entities + triples 组合文本”再做一次 embedding，并把结果写到 `unit.embedding`。
   4. `embed_entities`
      是否给每个 entity 单独做 embedding，并写入 `unit.metadata["representation"]["entity_embeddings"]`。
   5. `embedding_model` / `api_key` / `base_url` / `model`
      控制 LLM 与 embedding runtime。
5. 输出字段说明
   1. `unit.entities`
      去重后的实体字符串列表。
   2. `unit.triples`
      标准三元组列表，元素是 `(subject, predicate, object)`。
   3. `unit.metadata["representation"]["entity_embeddings"]`
      只有 `embed_entities=True` 才会出现，结构是 `{entity_text: vector}`。
6. 特殊行为
   1. 如果 `unit.metadata["triples"]` 已经提供了合法 triples，会直接走 `metadata_hint` 路径，不再调用 LLM。
   2. `embed_extracted=True` 生成的是“图对象 embedding”，不是逐实体 embedding。
   3. `embed_entities=True` 时，后续 `GraphEntityDeduplicationAppendOrganization` 才能直接用到这些 entity embedding。
   4. trace 里会记录 `source`，可能是：
      1. `metadata_hint`
      2. `llm_direct`
      3. `llm_two_stage`

### 3.3 `LLMRepresentation`

1. 功能
   用 prompt 驱动的方式，为 unit 生成一个指定字段。
2. 读取
   读取 `packet.units`。
   读取当前 unit 的：
   1. `text`
   2. `unit_type`
   3. `timestamp`
   4. `tags`
   5. `entities`
   6. `description`
   7. `metadata`
   还会读取 `self.prompt`。
3. 写回
   一定会更新：
   1. `unit.representation_elements`
   2. `unit.metadata["representation"]`
   根据 `field` 具体不同，还可能写：
   1. `unit.tags`
   2. `unit.entities`
   3. `unit.kv`
   4. `unit.description`
   写 `packet.trace["representation"]`。
   不改 `store`。
4. 主要参数
   1. `field`
      要生成的字段名。不能为空。
   2. `prompt`
      该字段的抽取 prompt，可以是 `str` 或 `PromptPlan`。
   3. `value_type`
      显式声明输出结构。当前只支持：
      1. `str`
      2. `list[str]`
      3. `dict[str, str]`
      4. `list[dict[str, str]]`
   4. `embedding_model` / `api_key` / `base_url` / `model`
      控制底层 runtime。
5. 字段落点规则
   1. `field == "tags"`
      写到 `unit.tags`。
   2. `field == "entities"`
      写到 `unit.entities`。
   3. `field == "kv"`
      写到 `unit.kv`。
   4. `field == "description"`
      写到 `unit.description`。
   5. 其他字段
      写到 `unit.metadata["representation"][field]`。
6. 特殊行为
   1. 如果 `value_type` 不写，会按 `field` 推断输出类型。
      例如 `tags`/`entities`/`keywords` 默认按 `list` 处理，`kv`/`time_anchor` 默认按 `dict` 处理。
   2. `list[dict[str, str]]` 是最近扩展出来的能力，适合抽像 `thoughts` 这种结构化列表。
   3. trace 里会记录：
      1. `field`
      2. `kind`
      3. `declared_value_type`
      4. 实际 `value`

### 3.4 `ConfigurableEmbeddingRepresentation`

1. 功能
   用可配置文本生成 embedding，不强制把 unit 的主文本改掉。
2. 读取
   读取 `packet.units`。
   读取 `embedding_text` 对应的 prompt / template 上下文。
3. 写回
   回写每个 unit 的：
   1. `unit.embedding`
   2. `unit.representation_elements`
   3. `unit.metadata["representation"]`
   写 `packet.trace["representation"]`。
   不改 `store`。
4. 主要参数
   1. `embedding_text`
      决定送给 embedding 模型的文本。
      可以是：
      1. 普通字符串
      2. 模板字符串
      3. 完整 `PromptPlan`
      不传时默认等价于 `{{ unit.text }}`。
   2. `embedding_version`
      只做 provenance 标记，记录在 trace 和 representation metadata 中。
   3. `embedding_model`
      底层 embedding 模型。
5. 重要输出位置
   1. 向量写在 `unit.embedding`
   2. 输入文本写在 `unit.metadata["representation"]["embedding_input_text"]`
   3. 版本写在 `unit.metadata["representation"]["embedding_version"]`
6. 特殊行为
   1. 它不会改 `unit.text`，这点和某些“先重写文本再 embed”的表示模块不同。
   2. 很适合 A-MEM / note 风格写法：先用多个 `LLMRepresentation` 生成 `context`、`keywords`、`tags` 等，再用这个类生成面向检索的 embedding。

## 4. `write_trigger`

当前公开 module：

1. `AlwaysTrigger`
2. `StoreAllTrigger`
3. `BoundaryEventTrigger`
4. `RuntimeEventTrigger`
5. `ScalarRuleTrigger`
6. `LLMJudgeTrigger`

公共约定：

1. 这些 trigger 都写 `packet.trace["write_trigger"]`。
2. 这些 trigger 的 `slot` 参数虽然存在，但这里的正常用法是 `slot="write_trigger"` 或留空使用默认值。
3. 大多数 trigger 都会写 `packet.decisions`。
4. `StoreAllTrigger` 主要写 `packet.decisions_store`。

### 4.1 `AlwaysTrigger`

1. 功能
   所有 unit 都通过。
2. 读取
   读取 `packet.units`。
3. 写回
   把 `packet.decisions` 设成全 `True`。
   写 `packet.trace["write_trigger"]`。
   不改 `store`。
4. 特殊行为
   它是 `MemoryPipeline` 默认的 write trigger。

### 4.2 `StoreAllTrigger`

1. 功能
   选择 store 里所有现有 records，但不主动重写当前 `packet.decisions`。
2. 读取
   读取 `packet.units` 只是为了保持 trigger 接口一致。
   读取整个 `store`。
3. 写回
   主要写 `packet.decisions_store`。
   trace 中会说明：
   1. 哪些 layer 被选中
   2. 每层有多少 record ids
   如果 `packet.decisions` 原本已存在，则会原样保留。
4. 特殊行为
   适合“组织 / 演化阶段想拿到整层历史 records”而不是只关心当前 unit 的场景。

### 4.3 `BoundaryEventTrigger`

1. 功能
   根据边界事件触发，比如 `session`、`turn`、`chunk`、`subgoal`、`episode`。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.events`
   3. `packet.observation.metadata`
   4. trigger payload 中的边界信息
3. 写回
   1. 写 `packet.decisions`
   2. 如果能解析出边界匹配键和值，还会写 `packet.decisions_store`
      这时会把 store 中 metadata 同样命中该边界键值的记录选出来
   3. 写 `packet.trace["write_trigger"]`
4. 主要参数
   1. `accepted_events`
      需要匹配的事件名元组。
   2. `match_mode`
      `any` 或 `all`。
   3. `default_decision`
      没有观察到事件时的默认判定。
   4. `invert`
      是否反转结果。
5. 特殊行为
   1. 它不只是“看 event 名字”，还会尝试把边界事件映射到 metadata 键，例如：
      1. `session -> session_id`
      2. `turn -> turn_id`
      3. `chunk -> chunk_id`
   2. 只有在成功解析出 `match_key` 和 `match_value` 后，`decisions_store` 才会被填充。

### 4.4 `RuntimeEventTrigger`

1. 功能
   根据运行时事件触发，典型场景是维护、后台整理、内存压力触发。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.events`
   3. `store` 当前 layer 压力信息
3. 写回
   1. 写 `packet.decisions`
   2. 如果命中了 `memory_pressure`，还可能写 `packet.decisions_store`
      这时通常会把目标 layer 的所有 records 选出来
   3. 写 `packet.trace["write_trigger"]`
4. 主要参数
   1. `accepted_events`
      接受的运行时事件，例如 `idle`、`memory_pressure` 等。
   2. `match_mode`
      `any` 或 `all`。
   3. `default_decision`
      没事件时的默认值。
   4. `invert`
      是否反转。
   5. `target_layer`
      计算 memory pressure 时使用的目标 layer。
   6. `pressure_threshold`
      如果设置了，并且 `accepted_events` 包含 `memory_pressure`，会先根据 layer 压力计算是否自动生成一个 `memory_pressure` runtime event。
5. 特殊行为
   1. `memory_pressure` 事件可以来自外部，也可以由该模块内部根据 store 容量状态推导出来。
   2. 对 write trigger 而言，如果要按 `memory_pressure` 算压力，通常需要显式传 `target_layer`。

### 4.5 `ScalarRuleTrigger`

1. 功能
   根据显式数值信号和阈值做规则触发。
2. 读取
   可能读取：
   1. `packet.units[*].metadata[signal_key]`
   2. `packet.observation.metadata["trigger"]["signals"]`
   3. `packet.trace["signals"]`
   4. `store.metadata["trigger_signals"]`
   5. store 的 layer pressure 快照
3. 写回
   1. 写 `packet.decisions`
   2. 某些情况下写 `packet.decisions_store`
      尤其是 `signal_key == "memory_pressure"` 时
   3. 写 `packet.trace["write_trigger"]`
4. 主要参数
   1. `signal_key`
      要读取的信号名。
   2. `threshold`
      阈值。
   3. `comparator`
      支持 `>`, `>=`, `<`, `<=`, `==`。
   4. `signal_source`
      取值有：
      1. `auto`
      2. `observation`
      3. `trace`
      4. `store`
      5. `unit_metadata`
   5. `aggregate`
      取值有：
      1. `broadcast`
      2. `per_unit`
      3. `any_unit`
      4. `all_units`
   6. `missing_value`
      信号缺失时可替代使用的默认值。
   7. `target_layer`
      当 `signal_key == "memory_pressure"` 时，用于指定要看的 layer。
5. 特殊行为
   1. `memory_pressure` 是这个类的一个特殊信号，不是简单从 metadata 里取值。
   2. 当按 memory pressure 判定为真时，通常会把对应 layer 的全部 records 放进 `packet.decisions_store`，方便后续维护模块直接处理整层。

### 4.6 `LLMJudgeTrigger`

1. 功能
   用真实 LLM 判断当前是否应触发。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.observation`
   3. `packet.placements`（演化 trigger 时）
   4. `store` 的 layer 总结
   5. `packet.trace`
   6. `self.prompt`
3. 写回
   1. 写 `packet.decisions`
   2. 写 `packet.trace["write_trigger"]`
   不改 `store`。
4. 主要参数
   1. `prompt`
      Judge prompt，可为 `PromptPlan`。
   2. `decision_mode`
      取值：
      1. `bool`
      2. `score`
      3. `label`
   3. `threshold`
      对 `score` 模式或者可转 score 的输出生效。
   4. `per_unit`
      为 `True` 时逐 unit 调用 judge。
      为 `False` 时只判一次，然后广播给所有 unit。
5. 特殊行为
   1. LLM 输出不是固定只能有一个字段，它可以带 `reason`、`confidence` 等辅助字段。
   2. 但最终决定只按 `decision_mode` 的规则归一化。

## 5. `organization`

当前公开 module：

1. `AppendOrganization`
2. `FanoutIngestOrganization`
3. `GraphAppendOrganization`
4. `GraphDeduplicationAppendOrganization`
5. `GraphEntityDeduplicationAppendOrganization`
6. `HierarchicalOrganization`
7. `LLMFunctionCallOrganization`

### 5.1 `AppendOrganization`

1. 功能
   把被 `write_trigger` 选中的 unit 直接 append 到固定 layer。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.decisions`
3. 写回
   1. 为每个 unit 都写 `packet.placements`
   2. 对 `decision=True` 的 unit 调用 `store.append(...)`
   3. 写 `packet.trace["organization"]`
4. 主要参数
   1. `target_layer`
      目标 layer 名。
5. 输出变量位置
   1. `packet.placements[i].target_layer`
   2. trace 中的 `written_record_ids`
6. 特殊行为
   1. 即使某个 unit 没通过 trigger，也仍然会有 placement，只是不会写 store。
   2. 它是 `MemoryPipeline` 默认 organization。

### 5.2 `FanoutIngestOrganization`

1. 功能
   从某个 iterable string 字段里取出多条文本，逐条喂给一个子 `MemoryPipeline.ingest(...)`。
2. 读取
   优先读取 `packet.observation.metadata[field]`。
   如果 observation metadata 没有该字段，则会回退读取：
   `packet.units[0].metadata["representation"][field]`。
3. 写回
   1. 不写 `packet.placements`
   2. 会实际修改 `store`，因为子 pipeline 会执行 ingest
   3. 写 `packet.trace["organization"]`
4. 主要参数
   1. `field`
      要 fanout 的字段名。
   2. `pipeline`
      子 `MemoryPipeline`。
5. 特殊行为
   1. 只有字符串成员会被 fanout；非字符串会跳过。
   2. 空字符串也会跳过。
   3. trace 会带：
      1. `fanout_count`
      2. `skipped_reasons`
      3. `child_traces`
   4. 这个类非常适合“先抽 facts / thoughts / triples，再逐条进子 pipeline”的组织方式。

### 5.3 `GraphAppendOrganization`

1. 功能
   把 unit 追加到 Graph layer，并补齐标准化 graph metadata。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.decisions`
   3. unit 上已有的 `entities`、`triples`、embedding、note payload 等
3. 写回
   1. 写 `packet.placements`
   2. 向 `store` 写 graph record
   3. 写 `packet.trace["organization"]`
4. 主要参数
   1. `target_layer`
      必须是 Graph shape 的 layer。
   2. `separate`
      为 `True` 时，原文本与 triple graph 记录分层写。
   3. `separate_layer`
      `separate=True` 时原文本写入的 side layer。
5. 输出字段说明
   graph record 的 `record.metadata["graph"]` 至少会标准化出：
   1. `graph.layer`
   2. `graph.shape`
   3. `graph.entities`
   4. `graph.triples`
   5. `graph.links`
   6. `graph.node_count`
   7. `graph.link_count`
   8. `graph.last_linked_at`
   9. `graph.link_history`
6. 特殊行为
   1. `separate=False` 时，一个 unit 只写一条 graph record。
   2. `separate=True` 时，会先写 source 文本 record 到 `separate_layer`，再写 triple graph record 到 `target_layer`。
   3. 它保留 unit 上已有 metadata，所以也可承接 A-MEM / note-graph 风格写入。

### 5.4 `GraphDeduplicationAppendOrganization`

1. 功能
   图写入前先做 top-1 embedding 相似度去重；超过阈值就 merge，否则 append。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.decisions`
   3. `target_layer` 现有 graph records
   4. unit 自身 embedding，或者 store/runtime 补出来的 embedding
3. 写回
   1. 写 `packet.placements`
   2. 对命中的记录走 `store.replace_record(...)`
   3. 对未命中的记录走 `store.append(...)`
   4. 写 `packet.trace["organization"]`
4. 主要参数
   1. `target_layer`
      Graph layer。
   2. `threshold`
      top-1 相似度严格大于该值时才 merge。
   3. `separate`
      是否额外保留原始 source 记录。
   4. `separate_layer`
      `separate=True` 时 source record 的 layer。
5. 特殊行为
   1. 如果 unit 自身没有 embedding，会尝试：
      1. `store.embedding_for_record(...)`
      2. 再退回 runtime 实时 embed
   2. merge 时会：
      1. 更新 text
      2. 更新 embedding
      3. 合并 triples
      4. 保留已有 graph links
   3. trace 的 `effects` 会区分：
      1. `merge`
      2. `append`
      3. `skipped`

### 5.5 `GraphEntityDeduplicationAppendOrganization`

1. 功能
   不是以整条 unit 为粒度 dedup，而是按 entity 粒度逐个 merge-or-append。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.decisions`
   3. `unit.entities`
   4. `unit.metadata["representation"]["entity_embeddings"]`
   5. 目标 graph layer 的现有 records
3. 写回
   1. 写 `packet.placements`
   2. 逐 entity 更新或新增 graph records
   3. 写 `packet.trace["organization"]`
4. 主要参数
   与 `GraphDeduplicationAppendOrganization` 相同：
   1. `target_layer`
   2. `threshold`
   3. `separate`
   4. `separate_layer`
5. 特殊行为
   1. 只有存在 entity embedding 的 entity 才能参与 dedup。
   2. 某个 entity 没 embedding 时，会记录 `skipped_entity_missing_embedding`。
   3. 新写入的 record 文本可能直接是 entity 文本，而不是原始整句 unit 文本。
   4. 非常适合“每个实体一条节点”的图谱写入。

### 5.6 `HierarchicalOrganization`

1. 功能
   对 source layer 中被选中的记录做分组、抽取、聚合，再写到更高层记忆。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.decisions`
   3. `packet.decisions_store` 或 source layer 扫描结果
   4. `store`
   5. 可选 `prompt`
   6. 可选 `memory_pipeline`
3. 写回
   1. 会生成 `packet.placements`
   2. 会通过子 `memory_pipeline.ingest(...)` 或内部层级写入逻辑修改 `store`
   3. 写 `packet.trace["organization"]`
4. 主要参数
   1. `source_layer`
      从哪一层取原始 records。
   2. `extract_mode`
      控制如何抽取聚合内容。
   3. `extract_fields`
      抽取哪些字段。
   4. `group_by`
      按哪些字段分组。
   5. `prompt`
      可选，用于 LLM / 模板驱动的聚合。
   6. `target_layer`
      高层结果写入层。
   7. `memory_pipeline`
      如果提供，就把抽出的高层结果交给这个子 pipeline 去 ingest。
5. 特殊行为
   1. 这个类的核心不是“把当前 unit 原样写进去”，而是“基于 source layer 做高层抽象”。
   2. trace 会记录：
      1. `selection_source`
      2. `selected_record_count`
      3. `group_count`
      4. `sub_ingest_trace`
      5. `prompt_trace`

### 5.7 `LLMFunctionCallOrganization`

1. 功能
   在 organization 阶段让 LLM 通过 write tools 决定 add / update / delete / graph link 操作。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.decisions`
   3. 当前 `store`
   4. `prompt`
   5. tool specs
3. 写回
   1. 一定会写 `packet.placements`
   2. 根据 tool call 实际修改 `store`
   3. 写 `packet.trace["organization"]`
4. 主要参数
   1. `prompt`
      给 LLM 的主 prompt，可是 `PromptPlan`。
   2. `tools`
      可以是内置字符串工具名，也可以是 `WriteToolSpec`。
   3. `target_layer`
      没有在 tool 参数里显式给出 layer 时的默认落点。
   4. `max_turns`
      最多几轮 tool-calling。
   5. `max_retry`
      tool 调用失败后最多额外重跑几次 LLM，默认 `0`。
   6. `raise_on_tool_error`
      默认 `False`。如果重试耗尽后仍有失败 tool call，设为 `True` 时抛异常，否则把失败记录到 trace 并跳过该调用。
   7. `strict_tools`
      兼容旧参数；显式设为 `True` 时保留旧行为，tool 执行失败会立即抛异常。
   8. `allow_no_tool_call`
      是否允许 LLM 一次 tool 都不调。
   9. `api_key` / `base_url` / `model` / `embedding_model`
      runtime 配置。
5. 重要上下文变量
   prompt 渲染时会额外提供：
   1. `unit`
   2. `tools`
   3. `default_target_layer`
   4. `visible_records`
6. 特殊行为
   1. organization 阶段的 `visible_records` 默认是整个 store。
   2. 如果 prompt 里配置了 recall plan / visible record label，它还能在渲染 prompt 时先做一轮 recall，再把 recall 命中的 records 并入 `visible_records`。
   3. 每个 unit 单独跑一遍 tool-calling，因此 trace 里是 `per_unit` 结构。
   4. 如果某次 LLM 尝试中有失败 tool call，会从尝试前 store 快照重跑；最终只提交最后一次接受尝试里的效果。

## 6. `evolution_trigger`

当前公开 module：

1. `NeverTrigger`
2. `StoreAllTrigger`
3. `BoundaryEventTrigger`
4. `RuntimeEventTrigger`
5. `ScalarRuleTrigger`
6. `LLMJudgeTrigger`
7. `PeriodicMaintenanceTrigger`
8. `IdleMaintenanceTrigger`

说明：

1. 除了 `PeriodicMaintenanceTrigger` 和 `IdleMaintenanceTrigger`，其余语义和 write_trigger 版相同，只是 trace key 改为 `packet.trace["evolution_trigger"]`。
2. evolution trigger 阶段通常要求 `packet.placements` 已经存在并与 `packet.units` 对齐。

### 6.1 `NeverTrigger`

1. 功能
   所有 unit 都不进入演化。
2. 读取
   读取 `packet.units`，以及演化阶段需要的 `packet.placements` 对齐关系。
3. 写回
   写 `packet.decisions = [False, ...]`。
   写 `packet.trace["evolution_trigger"]`。
   不改 `store`。
4. 特殊行为
   它是 `MemoryPipeline` 默认的 evolution trigger。

### 6.2 `StoreAllTrigger`

1. 功能
   与 write 阶段同名类相同，用于选出 store 中全部历史 records。
2. 读取
   读取 `store`。
3. 写回
   主写 `packet.decisions_store`，并保留已有 `packet.decisions`。

### 6.3 `BoundaryEventTrigger`

1. 功能
   根据 session / turn / episode 等边界触发维护。
2. 特殊行为
   如果能解析出边界匹配值，会把同边界的历史 records 放进 `decisions_store`，这在层级汇总和周期性维护里很常见。

### 6.4 `RuntimeEventTrigger`

1. 功能
   根据运行时事件或内存压力触发维护。
2. 特殊行为
   在演化阶段，如果按 `memory_pressure` 计算且 `packet.placements` 可解析出目标层，就可以不显式写 `target_layer`。

### 6.5 `ScalarRuleTrigger`

1. 功能
   根据分数、预算、压力等标量信号触发维护。
2. 特殊行为
   如果 `aggregate` 不是 `broadcast`，则会按 unit 或 layer 细分判定。

### 6.6 `LLMJudgeTrigger`

1. 功能
   用 LLM 判断当前是否值得做演化。
2. 特殊行为
   演化阶段 prompt 里还会多拿到 placement 信息，便于判断目标层与演化需求。

### 6.7 `PeriodicMaintenanceTrigger`

1. 功能
   只有在周期调度命中时，才调用一个“被包裹的 trigger”。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.observation.metadata["trigger"]["schedule"]`
   3. `store.metadata[counter_key]`
3. 写回
   1. 周期未命中时，不重写 `packet.decisions`，只补 trace。
   2. 周期命中时，转而运行内部 `trigger`，并把其结果写回 packet。
4. 主要参数
   1. `every_n`
      每多少 tick 触发一次。
   2. `counter_key`
      默认 `ingest_count`。
   3. `trigger`
      真正被包裹执行的 trigger，slot 必须一致。
5. 特殊行为
   1. 这是一个 wrapper，不自己做最终判断。
   2. trace 会同时带：
      1. 周期调度信息
      2. 内部 trigger 的结果

### 6.8 `IdleMaintenanceTrigger`

1. 功能
   在系统空闲窗口触发维护。
2. 读取
   读取：
   1. `packet.events`
   2. `packet.observation.metadata["trigger"]["schedule"]["idle_seconds"]`
3. 写回
   写 `packet.decisions`。
   写 `packet.trace["evolution_trigger"]`。
4. 主要参数
   1. `min_idle_seconds`
      空闲秒数阈值。
   2. `accepted_events`
      哪些 runtime event 算作 idle 窗口。
   3. `decision_mode`
      当前只影响 trace 语义，公开可选值是 `broadcast` / `per_unit`。
5. 特殊行为
   1. 只要“观察到 idle event”或者“idle_seconds 超阈值”任一成立，就会触发。

## 7. `memory_evolution`

当前公开 module：

1. `AppendOnlyEvolution`
2. `TraceOnlyEvolution`
3. `SummaryRewriteEvolution`
4. `LayerMoveEvolution`
5. `GraphLinkEvolution`
6. `GraphNeighborContextTraceEvolution`
7. `STMConsolidationEvolution`
8. `HierarchicalEvolution`
9. `LLMFunctionCallEvolution`

### 7.1 `AppendOnlyEvolution`

1. 功能
   一个显式 no-op 演化模块，只记录哪些 unit 在演化阶段是 active 的。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.placements`
   3. `packet.decisions`
3. 写回
   只写 `packet.trace["memory_evolution"]`。
   不改 `store`。
4. 特殊行为
   适合“组织阶段已经写完，演化阶段先不开启真实维护”的 pipeline。

### 7.2 `TraceOnlyEvolution`

1. 功能
   也是 no-op，但会显式写出每个 active unit 的 effect placeholder。
2. 读取
   读取 `packet.units`、`packet.placements`、`packet.decisions`。
3. 写回
   只写 `packet.trace["memory_evolution"]["effects"]`。
   不改 `store`。
4. 特殊行为
   适合测试、trace 占位、调试下游逻辑。

### 7.3 `SummaryRewriteEvolution`

1. 功能
   对 active units 生成摘要版本，并把摘要作为新 record append 到目标层。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.placements`
   3. `packet.decisions`
   4. `unit.metadata["representation"]["summary"]`
   5. `unit.metadata["representation"]["description"]`
   6. `unit.description`
3. 写回
   向 `store` append 新 record。
   写 `packet.trace["memory_evolution"]`。
4. 主要参数
   1. `target_layer`
      摘要写到哪一层。
5. 特殊行为
   摘要文本来源优先级是：
   1. `representation.summary`
   2. `representation.description`
   3. `unit.description`
   4. `unit.text`

### 7.4 `LayerMoveEvolution`

1. 功能
   把 active units 复制写入另一个 layer。
2. 读取
   读取 `packet.units`、`packet.placements`、`packet.decisions`。
3. 写回
   向 `target_layer` append 新 record。
   写 `packet.trace["memory_evolution"]`。
4. 主要参数
   1. `target_layer`
      新 record 的 layer。
5. 特殊行为
   1. 当前是 copy-append，不删除原记录。
   2. 新 record metadata 会带：
      1. `evolution_source_unit_id`
      2. `move_style = "copy_append"`

### 7.5 `GraphLinkEvolution`

1. 功能
   在 graph layer 内，为当前 active graph record 自动寻找邻居并建立 links。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.placements`
   3. `packet.decisions`
   4. `target_layer` 当前所有 graph records
3. 写回
   1. 对目标 graph record 调用 `store.add_graph_links(...)`
   2. 可选地 rewrite 当前目标 record 的 `neighbor_context`
   3. 写 `packet.trace["memory_evolution"]`
4. 主要参数
   1. `target_layer`
      必须是 graph layer。
   2. `neighbor_limit`
      最多保留多少个邻居候选。
   3. `bidirectional`
      是否建立双向 link。
   4. `min_score`
      候选邻居的最小总分阈值。
   5. `rewrite_neighbor_metadata`
      是否把邻居上下文快照写回目标 record metadata。
5. 特殊行为
   候选邻居总分由两部分组成：
   1. 结构分
      由 entity overlap 和文本 token overlap 构成。
   2. embedding 分
      当前 record embedding 和候选 embedding 的 cosine。

### 7.6 `GraphNeighborContextTraceEvolution`

1. 功能
   读取 graph record 的已连接邻居，记录邻居上下文快照。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.placements`
   3. `packet.decisions`
   4. graph 邻居关系
3. 写回
   1. 默认只写 trace
   2. `rewrite_metadata=True` 时，会把 `neighbor_context` 回写到目标 graph record
4. 主要参数
   1. `target_layer`
   2. `rewrite_metadata`
5. 特殊行为
   适合只想“追踪邻居上下文”而不想主动重连边的场景。

### 7.7 `STMConsolidationEvolution`

1. 功能
   显式执行 STM overflow / consolidation：按 session scope 保留 working layer 中最新的 STM records，把溢出的旧 raw episodes copy-append 到 LTM episode layer，重写当前 session summary，然后从 STM 删除已 consolidation 的旧 records。
2. 读取
   读取：
   1. `working_layer` 中的 records
   2. `summary_layer` 中同 scope 的旧 summary records
   3. 每条 record 的 `metadata`，默认用 `session_id` 分组
3. 写回
   1. 向 `ltm_episode_layer` append raw episode copies
   2. 向 `summary_layer` 写入新的当前 summary record
   3. 删除同 scope 的旧 summary records，使每个受影响 session 只保留一个当前 summary
   4. 从 `working_layer` 删除已 consolidation 的 evicted records
   5. 写 `packet.trace["memory_evolution"]`
4. 主要参数
   1. `working_layer`
      STM / working memory 层，默认 `"working"`。
   2. `ltm_episode_layer`
      长期 raw episode 层，默认 `"episodic"`。
   3. `summary_layer`
      session summary 层，默认 `"session_summary"`。
   4. `record_budget`
      每个 scope 在 STM 中保留的最新 record 数，默认 `20`。
   5. `token_budget`
      可选 token budget。设置后按最新记录向前累加 token，超出部分被 evict；至少保留最新一条。
   6. `summary_max_sentences`
      runtime summary rewrite 的最大句数提示，默认 `3`。
   7. `scope_metadata_keys`
      scope 分组 metadata keys，默认 `("session_id",)`；如需同时隔离用户，可传 `("session_id", "user_id")`。
5. 特殊行为
   1. 这个模块不依赖 `MemoryStore` 的 `capacity="sliding_window"`，也不调用 `trim_layer_to_*`。如果 working layer 自己开启自动 sliding-window，旧记录可能在模块运行前就已经丢失，因此 MemMachine 风格 STM consolidation 应让该模块自己控制 overflow。
   2. LTM copy 保留原始 `text`、`timestamp`、`unit_id`、embedding 和原 metadata，并额外写：
      1. `metadata["stm_consolidation_source_record_id"]`
      2. `metadata["stm_consolidation"]`
   3. Summary rewrite 使用 runtime summarization：输入包含旧 summary（如果存在）和本次 evicted raw STM records。
   4. trace 会记录 `evicted_record_ids`、`moved_ltm_record_ids`、`deleted_working_record_ids`、`summary_updates`，并在每个 effect 中记录 `session_scope`、`retained_record_ids`、`summary_old_record_id`、`summary_new_record_id`。

### 7.8 `HierarchicalEvolution`

1. 功能
   在 organization 完成后，对已有 records 做层级抽象、生成高层记录，并可做目标层保留窗口裁剪。
2. 读取
   读取：
   1. `packet.units`
   2. `packet.placements`
   3. `packet.decisions`
   4. `packet.decisions_store`
   5. `store`
3. 写回
   1. 向高层写新 records
   2. `retention_size` 生效时还会直接裁剪 `store.layers[target_layer]`
   3. 写 `packet.trace["memory_evolution"]`
4. 主要参数
   1. `source_layer`
   2. `extract_mode`
   3. `extract_fields`
   4. `group_by`
   5. `prompt`
   6. `target_layer`
   7. `memory_pipeline`
   8. `selection_mode`
      当前重要取值：
      1. `decisions_store_or_scan`
      2. `latest_active_units`
   9. `record_text_field`
      允许生成 richer payload，但指定某个字段作为落库的 `record.text`。
   10. `retention_size`
      目标层最多保留最近多少条记录。
5. 特殊行为
   1. `latest_active_units` 很适合 Reflexion / RecurrentGPT 这类“刚写完原始记录，立刻只对这批新记录做高层抽象”的模式。
   2. `retention_size` 不是逻辑标记，而是直接裁剪内存中的 layer 列表。
   3. trace 会记录：
      1. `selection_mode`
      2. `record_text_field`
      3. `pruned_record_ids`
      4. `retained_record_ids`

### 7.9 `LLMFunctionCallEvolution`

1. 功能
   在演化阶段使用 LLM tool-calling 对已有 records 做增删改。
2. 读取
   读取：
   1. `packet.decisions_store`
   2. 或 `packet.units` + `packet.placements` + `packet.decisions`
   3. 或 `source_layer`
   4. 或整个 `store`
   5. `prompt`
   6. tool specs
3. 写回
   1. 根据 tool 调用修改 `store`
   2. 写 `packet.trace["memory_evolution"]`
4. 主要参数
   1. `prompt`
   2. `tools`
   3. `source_layer`
      如果提供，会限制 selected records 的来源层。
   4. `target_layer`
      tool 未显式指定目标层时的默认值。
   5. `max_turns`
   6. `max_retry`
      tool 调用失败后最多额外重跑几次 LLM，默认 `0`。
   7. `raise_on_tool_error`
      默认 `False`。重试耗尽后仍失败时，默认把失败写入 trace 并跳过失败调用；设为 `True` 时抛异常。
   8. `strict_tools`
      兼容旧参数；显式设为 `True` 时 tool 失败立即抛异常。
   9. `allow_no_tool_call`
   10. `api_key` / `base_url` / `model` / `embedding_model`
5. 选 record 逻辑
   selected records 的优先级是：
   1. 如果 `packet.decisions_store` 有值，优先按它选。
   2. 否则如果有 `packet.units + placements + decisions`，按 active units 找对应最新 record。
   3. 再否则，如果设了 `source_layer`，扫该层全部 records。
   4. 否则扫整个 store。
6. 特殊行为
   1. evolution 阶段的 prompt 上下文里有：
      1. `selected_records`
      2. `visible_records`
      3. `source_layer`
      4. `tools`
      5. `default_target_layer`
   2. `visible_records` 默认至少包含 `selected_records`，如果 prompt recall 又命中了别的记录，会并入可见集。
   3. 如果某次 LLM 尝试中有失败 tool call，会从尝试前 store 快照重跑；最终只提交最后一次接受尝试里的效果。

## 8. `retrieval`

当前公开 module：

1. `RecencyRetrieval`
2. `KeywordCountRetrieval`
3. `MetadataRetrieval`
4. `EmbeddingSimilarityRetrieval`
5. `EntityRetrieval`
6. `TripleMemoryRetrieval`
7. `BM25Retrieval`
8. `RerankerRetrieval`
9. `GraphNeighborRetrieval`
10. `ParentEpisodeExpansionRetrieval`
11. `TemporalNeighborExpansionRetrieval`
12. `EpisodeClusterRerankRetrieval`
13. `ExpandRetrievedGraphNeighbors`
14. `VectorGraphSeedAndExpandRetrieval`
15. `LayerAwareRetrieval`
16. `BufferRetrieval`
17. `QueryRewriteRetrieval`

### 8.1 `RecencyRetrieval`

1. 功能
   纯按时间新近性返回最近 `top_k` 条记录。
2. 读取
   读取 `packet.query`，但 query 文本只为接口一致性保留，不参与排序。
   读取 `store` 或 `packet.retrieved`，取决于 `source`。
3. 写回
   写 `packet.retrieved` 与 `packet.trace["retrieval"]`。
   不改 `store`。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `source`
      只能是：
      1. `store`
      2. `retrieved`
5. 特殊行为
   1. `source="retrieved"` 时，会对已有 `packet.retrieved.items` 再做一次重排和裁剪。

### 8.2 `KeywordCountRetrieval`

1. 功能
   按 query token 与 record 文本/关键词的命中数量排序。
2. 读取
   读取：
   1. `packet.query.text`
   2. `record.text`
   3. `record.metadata["representation"]["keywords"]`
3. 写回
   写 `packet.retrieved` 和 trace。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `source`
5. 特殊行为
   同分时按 recency 作为稳定 tie-break。

### 8.3 `MetadataRetrieval`

1. 功能
   按 metadata 某个字段筛记录。
2. 读取
   读取：
   1. `record.metadata[field]`
   2. `packet.query`
      这里只是接口要求，实际不会从 query 自动推 `target`
3. 写回
   写 `packet.retrieved` 和 trace。
4. 主要参数
   1. `top_k`
   2. `field`
   3. `target`
   4. `match_mode`
      当前支持：
      1. `exact`
      2. `regex`
   5. `layer`
   6. `source`
5. 特殊行为
   1. `exact` 匹配默认大小写不敏感。
   2. 如果 metadata 字段值是 `list` / `tuple` / `set` 或其他 iterable，会逐成员匹配，只要有一个成员命中就算匹配。

### 8.4 `EmbeddingSimilarityRetrieval`

1. 功能
   用 query embedding 与 record embedding 做 cosine 相似度检索。
2. 读取
   读取：
   1. `packet.query.embedding`
   2. 若 query 没 embedding，则读取 `packet.query.text` 现场编码
   3. `record.embedding`
3. 写回
   1. 写 `packet.retrieved`
   2. 如果 query 原本没有 embedding，会把生成的 embedding 回写到 `packet.query.embedding`
   3. 写 trace
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `embedding_model`
   4. `source`
5. 特殊行为
   1. 没 embedding 的 record 不参与评分。
   2. 向量维度不一致的 record 会被跳过，trace 会记录 `skipped_dim_mismatch_count`。

### 8.5 `EntityRetrieval`

1. 功能
   按 query 实体与 record 实体的重叠度排序。
2. 读取
   读取：
   1. `packet.query.text`
   2. `record.metadata["representation"]["entities"]`
3. 写回
   写 `packet.retrieved` 和 trace。
4. 主要参数
   1. `top_k`
   2. `layer`
5. 特殊行为
   1. 首先把 query 中首字母大写的 token 当成实体候选。
   2. 如果实体重叠为 0，再退回普通 token overlap。

### 8.6 `TripleMemoryRetrieval`

1. 功能
   面向三元组记忆的检索器，先 exact match，不命中时再做 fuzzy slot similarity。
2. 读取
   读取：
   1. `packet.query.text`
   2. 或 `packet.query.metadata["triple_query"]`
   3. `record.metadata["representation"]["triples"]`
   4. `record.metadata["graph"]["triples"]`
3. 写回
   写 `packet.retrieved` 和 trace。
   不改 `store`。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `source`
   4. `embedding_model`
   5. `candidate_similarity_threshold`
      fuzzy 阶段每个 slot 的最小相似度。
   6. `final_similarity_threshold`
      fuzzy 阶段 grounded slots 平均分的最小阈值。
5. Query 结构
   支持两种写法：
   1. `Query.metadata["triple_query"] = {"subject": ..., "relation": ..., "object": ...}`
   2. `query.text = "subject >> relation >> object"`
   空字符串或 `*` 会被当成 wildcard。
6. 特殊行为
   1. exact 阶段只要有结果，就不会进入 fuzzy。
   2. fuzzy 阶段是逐 grounded slot 算语义相似度，而不是整句相似度。
   3. score 里会保留 `matched_triples`，便于下游读出真实命中的 triplets。

### 8.7 `BM25Retrieval`

1. 功能
   用 BM25 对文本和表示层关键词做排序。
2. 读取
   读取：
   1. `packet.query.text`
   2. `record.text`
   3. `record.metadata["representation"]["keywords"]`
3. 写回
   写 `packet.retrieved` 和 trace。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `source`
5. 特殊行为
   1. 如果 query token 对所有文档都没有 overlap，会退回 recency fallback。
   2. trace 会记录 `used_recency_fallback`。

### 8.8 `RerankerRetrieval`

1. 功能
   调用专门 reranker runtime 对候选 record 做二阶段重排序。
2. 读取
   读取：
   1. `packet.query.text`
   2. 默认读取 `packet.retrieved.items`
   3. 当 `source="store"` 时读取 store，可用 `layer` 限定层
3. 写回
   写 `packet.retrieved` 和 trace。
   score dict 包含 `record_id`、`rank`、`score`、`rationale`、`strategy="runtime_rerank"`。
4. 主要参数
   1. `top_k`
      默认 `None`，表示把全部候选交给 reranker，并返回 reranker 排出的全部有效结果；显式传入时必须大于 0。
   2. `layer`
      可选层过滤。
   3. `source`
      默认 `"retrieved"`；也可显式设为 `"store"`。
   4. `task`
      传给 `Runtime.rerank()` 的任务描述。
5. 特殊行为
   1. 空候选集直接返回空 `RetrievedSet`，不会调用 reranker。
   2. 真实调用只走 `MEMPRIMITIVE_RERANK_API_KEY` / `MEMPRIMITIVE_RERANK_BASE_URL` / `MEMPRIMITIVE_RERANK_MODEL` 配置的专门 reranker runtime，不做 LLM prompt fallback。
   3. trace 会记录 `effective_top_k`、`candidate_record_ids`、`returned_ids`、`missing_or_ignored_count` 等字段。
6. 示例
   可以接在一阶段召回后使用：`EmbeddingSimilarityRetrieval(top_k=30), RerankerRetrieval(top_k=5)`。

### 8.9 `GraphNeighborRetrieval`

1. 功能
   从 query 指定的 graph seed record ids 出发，取一跳邻居。
2. 读取
   读取：
   1. `packet.query.metadata["graph_seed_record_ids"]`
   2. 或 `packet.query.metadata["graph"]["seed_record_ids"]`
   3. 可选 `graph_candidate_record_ids`
   4. graph layer 的 links
3. 写回
   写 `packet.retrieved` 和 trace。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `include_seed_records`
5. 特殊行为
   1. 结果默认按 graph link 展开顺序返回。
   2. 如果 query 里给了 candidate filter，只会返回 filter 内的 records。

### 8.10 `ParentEpisodeExpansionRetrieval`

1. 功能
   把当前句子级或 derivative 检索命中扩展回显式 parent episode 记录。
2. 读取
   读取：
   1. `packet.retrieved.items`
   2. hit record 的 `metadata[parent_id_field]`
   3. 或 hit record 的 `metadata["provenance"][parent_id_field]`
   4. `episode_layer` 中的 episode records
3. 写回
   覆盖写新的 `packet.retrieved`。
   不改 `store`。
4. 主要参数
   1. `top_k`
   2. `episode_layer`
      默认 `episodic`。
   3. `parent_id_fields`
      默认依次尝试：
      1. `parent_episode_record_id`
      2. `episode_record_id`
      3. `source_episode_record_id`
   4. `source`
      当前只支持 `retrieved`。
5. 特殊行为
   1. 不会解析 record text 来猜 parent episode；必须有显式 metadata / provenance parent id。
   2. 多个 hit 指向同一 episode 时只返回一次，并保留首次命中的 episode 顺序。
   3. 如果重复 hit 有数值 `score`，父 episode score 使用最高数值分；如果没有数值分，则保留首次 hit score。
   4. trace 记录 `input_hit_ids`、`successful_parent_ids`、`missing_parent_count`、`hit_to_parent`、`duplicate_parent_count` 和 `unresolved_parent_count`。

### 8.11 `TemporalNeighborExpansionRetrieval`

1. 功能
   从当前 `packet.retrieved.items` 中的 nucleus episode 出发，取同一 scope 内的前后时间邻居，覆盖 MemMachine 式 “nucleus episode + previous/following episodes” contextualization。
2. 读取
   读取：
   1. `packet.retrieved.items`
   2. `store.iter_records(layer)`
   3. nucleus record 的 metadata scope 字段
3. 写回
   覆盖写新的 `packet.retrieved`。
   不改 `store`。
4. 主要参数
   1. `layer`
      默认 `episodic`。
   2. `backward`
      每个 nucleus 向前取多少条，默认 `1`。
   3. `forward`
      每个 nucleus 向后取多少条，默认 `2`。
   4. `scope_fields`
      默认 `("session_id", "user_id", "agent_id")`。
   5. `chronological`
      默认 `True`，最终去重后按时间顺序返回。
5. 特殊行为
   1. 这不是 graph neighbor retrieval；不会读取 graph links，也不会沿语义图扩展。
   2. 只把 `packet.retrieved.items` 中 layer 等于 `self.layer` 的记录当作 nucleus。
   3. scope 字段只有在 nucleus metadata 中存在时才作为过滤条件；存在的字段要求候选 episode metadata 中同字段同值。
   4. 时间序优先使用 episode `timestamp` 解析值加 `record_id` 排序；如果 timestamp 不可用，则退回该 layer 的 `store.iter_records(layer)` 稳定顺序。
   5. 多个 nucleus 的 cluster 可以重叠；最终输出会按 `record_id` 去重。trace 中的 `clusters` 保留每个 nucleus 的 `cluster_ids`、`backward_ids`、`forward_ids`，顶层记录 `input_record_ids`、`returned_ids`、`ordering_mode` 和 `deduped_duplicate_count`。

### 8.12 `EpisodeClusterRerankRetrieval`

1. 功能
   对 `TemporalNeighborExpansionRetrieval` 产出的 episode clusters 做 cluster-level rerank，再在 episode budget 内合并、去重，并返回最终 episode context。
2. 读取
   读取：
   1. `packet.query.text`
   2. `packet.retrieved.items`
   3. `packet.retrieved.trace["clusters"]`
   4. 必要时从 `packet.retrieved.trace["layer"]` 对应 layer 或整个 `store` 回填解析 cluster 中的 episode records
3. 写回
   覆盖写新的 `packet.retrieved`。
   不改 `store`。
4. 主要参数
   1. `top_k`
      最终 episode 返回条数，默认 `20`。
   2. `cluster_top_k`
      rerank 后最多保留多少个 cluster；不传则使用全部可解析 cluster。
   3. `rerank`
      默认 `True`，使用 runtime reranker 对 cluster 排序；设为 `False` 时保留输入 cluster 顺序。
   4. `chronological`
      默认 `True`，最终选中的 episodes 会按时间正序返回。
5. 特殊行为
   1. cluster 输入优先读取 `cluster_ids`；如果没有 `cluster_ids`，则按 `backward_ids + nucleus_record_id + forward_ids` 组装。
   2. record 解析先使用 `packet.retrieved.items`，再按 trace 中的 `layer` 到 store 回填；无法解析的 ids 会记录到 cluster trace 的 `unresolved_ids`，完全无法解析的 cluster 会被跳过。
   3. `rerank=True` 时会把每个 cluster 渲染成 `"[record_id] text"` 行并交给 `runtime.rerank`；该 runtime reranker 会调用由 `MEMPRIMITIVE_RERANK_API_KEY` / `MEMPRIMITIVE_RERANK_BASE_URL` / `MEMPRIMITIVE_RERANK_MODEL` 配置的 OpenAI-compatible `/rerank` 接口，不再用 LLM prompt 模拟排序；runtime 没返回的 cluster 会用 fallback trace 补齐。
   4. 合并多个 cluster 时按 `record_id` 去重，并在 score 中保留 `source_nucleus_record_ids` 和 `roles`。
   5. 如果剩余 budget 放不下整个 cluster，会优先保留 nucleus episode，再按 cluster 内距离 nucleus 的近远选择邻居。
   6. trace 记录 `input_cluster_count`、`resolved_cluster_count`、`unresolved_cluster_count`、`reranked_clusters`、`returned_ids`、`deduped_duplicate_count` 和 `budget_truncated_count`。

### 8.13 `ExpandRetrievedGraphNeighbors`

1. 功能
   不从 query seed ids 出发，而是从当前 `packet.retrieved.items` 继续扩一跳邻居。
2. 读取
   读取 `packet.retrieved.items` 作为 seeds。
3. 写回
   覆盖写新的 `packet.retrieved`。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `include_seed_records`
   4. `per_seed_top_k`
   5. `dedupe`
5. 特殊行为
   1. 只有 `packet.retrieved.items` 中 layer 正好等于 `self.layer` 的记录会被当成 seeds。
   2. `dedupe=True` 时，同一 record 被多个 seed 扩到也只保留一份。

### 8.14 `VectorGraphSeedAndExpandRetrieval`

1. 功能
   先用向量检索 graph notes 作为 seeds，再扩一跳 graph neighbors，最后可选做 agentic rerank。
2. 读取
   读取：
   1. `packet.query`
   2. graph vector layer
   3. note payload
   4. 可选 query expansion prompt
3. 写回
   1. 写 `packet.retrieved`
   2. 可能回写 `packet.query.embedding`
   3. 写 trace
4. 主要参数
   1. `top_k`
      最终返回条数。
   2. `layer`
      graph+vector layer。
   3. `candidate_k`
      第一阶段向量 seed 检索的候选数。
   4. `neighbor_expansion_k`
      每个 seed 扩多少一跳邻居。
   5. `note_namespace`
      从 record 中读取 note payload 的命名空间。
   6. `default_category`
      note payload 修复时的默认分类。
   7. `agentic_search`
      是否让 runtime 做二次 agentic rerank。
   8. `query_expand_with_llm`
      是否先用 LLM 把 query 改写成适合 note 检索的增强 payload。
   9. `prompt`
      query expansion 的 prompt。
5. 特殊行为
   1. 这是一个 wrapper，不是新的底层索引结构。
   2. `query_expand_with_llm=True` 时，embedding 文本不再只是原 query，而是增强后的 note-like 文本。
   3. `agentic_search=True` 时，最终排序由 runtime.rerank 决定；它会调用独立配置的 OpenAI-compatible reranker，而不是聊天 LLM JSON prompt，不再只是 seed+expand 的原始顺序。

### 8.15 `LayerAwareRetrieval`

1. 功能
   每层用不同 retriever 分别检索，再做全局合并。
2. 读取
   读取：
   1. `packet.query`
   2. 所有 active layers
   3. 每层对应 retriever 的输出
3. 写回
   写统一的 `packet.retrieved` 和 trace。
4. 主要参数
   1. `default_retriever`
      未显式覆盖的层默认用哪个 retriever。
      不传时默认是同 `top_k` 的 `RecencyRetrieval`。
   2. `retriever_by_layer`
      每层专用 retriever。
   3. `active_layers`
      不传则用 store topology 全部层。
   4. `top_k`
      全局最终返回条数。
   5. `top_k_by_layer`
      覆盖单层 retriever 的 top_k。
   6. `merge_weight_by_layer`
      合并排序时每层的权重。
   7. `merge_strategy`
      目前只支持 `global_rank`。
5. 特殊行为
   1. 有数值 score 的结果优先于只有 rank 的结果。
   2. 数值 score 会乘上 `merge_weight_by_layer[layer]` 后再参加全局排序。
   3. 每层 retriever 是在 layer-scoped store 视图上运行的。

### 8.16 `BufferRetrieval`

1. 功能
   从某一层直接读一个有界 recency window，而不是做 query 搜索。
2. 读取
   读取 `store.iter_records(layer)`。
3. 写回
   写 `packet.retrieved` 和 trace。
4. 主要参数
   1. `top_k`
   2. `layer`
   3. `chronological`
      取最近 `top_k` 条后，是否按时间正序返回。
5. 特殊行为
   1. 虽然仍要求 `packet.query` 存在，但 query 文本不参与排名。
   2. 很适合 Reflexion 这类“读最近几条 reflection”。

### 8.17 `QueryRewriteRetrieval`

1. 功能
   先改写 query，再把改写结果交给内部 retriever。
2. 读取
   读取：
   1. `packet.query`
   2. `prompt` 或 `regex_rules`
   3. 内部 retriever
3. 写回
   1. 可能写新的 `packet.query`
   2. 可能写 `packet.queries`
   3. 最终写 `packet.retrieved`
   4. 写 trace，并把内部 retriever trace 包在 `query_rewrite` 下
4. 主要参数
   1. `retriever`
      被包装的检索器。
   2. `strategy`
      `llm` 或 `regex`。
   3. `prompt`
      `strategy="llm"` 时必填。
   4. `allow_multi_query`
      是否允许 LLM 一次返回多个 query。
   5. `regex_rules`
      `strategy="regex"` 时的改写规则。
   6. `include_original`
      是否把原始 query 也并入改写结果。
   7. `max_queries`
   8. `strip_queries`
   9. `drop_empty_queries`
5. 特殊行为
   1. 改写后的 query metadata 会带 `rewrite.source`、`rewrite.from_query_id`、`rewrite.index`。
   2. 多 query 最终仍遵守统一的 `query_order_dedupe` 合并规则。

## 9. `readout`

当前公开 module：

1. `ConcatenateReadout`
2. `JSONReadout`
3. `GraphReadout`
4. `GraphRelationReadout`
5. `PromptContextReadout`
6. `NoteRenderReadout`
7. `MidDecodingMemoryReadout`
8. `TemplateReadout`

### 9.1 `ConcatenateReadout`

1. 功能
   直接把 retrieval items 的 `record.text` 拼起来。
2. 读取
   读取 `packet.retrieved.items`。
3. 写回
   写 `packet.readout` 和 `packet.trace["readout"]`。
   不改 `store`。
4. 主要参数
   1. `separator`
5. 输出位置
   1. `readout.text`
   2. `readout.source_ids`

### 9.2 `JSONReadout`

1. 功能
   把 retrieval items 渲染成 JSON 字符串。
2. 读取
   读取 `packet.retrieved.items`。
3. 写回
   写 `readout.text` 为 JSON 字符串。
4. 特殊行为
   输出 payload 当前只包含：
   1. `record_id`
   2. `layer`
   3. `text`
   4. `timestamp`
   5. `source_ids`

### 9.3 `GraphReadout`

1. 功能
   用稳定的可读文本格式渲染 graph records。
2. 读取
   读取 `packet.retrieved.items` 和每条 record 的 graph metadata。
3. 写回
   写 `packet.readout` 和 trace。
4. 主要参数
   1. `include_links`
5. 特殊行为
   1. 它可以消费 mixed retrieval results，但设计目标是 graph layer。
   2. `include_links=True` 且没有 links 时会显式输出 `links=<none>`。

### 9.4 `GraphRelationReadout`

1. 功能
   把“已连接 graph records”的 triples 渲染成关系句子。
2. 读取
   读取：
   1. `packet.retrieved.items`
   2. `record.metadata["graph"]["links"]`
   3. `record.metadata["graph"]["triples"]`
3. 写回
   写 `packet.readout` 和 trace。
4. 特殊行为
   1. 只有 links 非空的 graph records 才会贡献 relation sentence。
   2. 句子全局去重。
   3. 如果一条关系句都渲染不出来，会退回直接拼原始 `record.text`。

### 9.5 `PromptContextReadout`

1. 功能
   把 retrieval 结果整理成下一轮 prompt context，主要服务 Reflexion 风格使用。
2. 读取
   读取：
   1. `packet.query`
   2. 可选 `packet.retrieved`
   3. `query.metadata` 中的 strategy / last attempt 信息
3. 写回
   写 `packet.readout` 和 trace。
4. 主要参数
   1. `memory_layer`
      从 retrieval items 中哪些 layer 记作“reflections”。
   2. `default_strategy`
      默认上下文拼装策略。
   3. `top_k`
5. 特殊行为
   1. `strategy` 允许被 `query.metadata` 覆盖。
   2. 只有 `reflexion` 和 `last_trial_and_reflexion` 这类策略，`source_ids` 才会包含 reflection records。

### 9.6 `NoteRenderReadout`

1. 功能
   把 note payload 渲染成更适合人读的 note 视图。
2. 读取
   读取：
   1. `packet.query`
   2. `packet.retrieved`
   3. note payload
3. 写回
   写 `packet.readout` 和 trace。
4. 主要参数
   1. `note_namespace`
   2. `default_category`
   3. `include_context`
   4. `include_tags`
5. 特殊行为
   1. 输出里会先放一行 `Query: ...`。
   2. 如果没取到任何 note，会显式输出 `No enriched notes retrieved.`。

### 9.7 `MidDecodingMemoryReadout`

1. 功能
   在 readout 阶段再跑一个多轮 tool-calling answer generation，允许模型在生成中途调用记忆检索工具。
2. 读取
   读取：
   1. `packet.query`
   2. `packet.retrieved`
   3. `prompt`
   4. `retrieve_pipeline`
   5. readout tool specs
3. 写回
   1. 写 `packet.readout`
   2. `readout.source_ids` 会汇总所有 `MEM_READ` 命中的 `record_id`
   3. 写 `packet.trace["readout"]`
   4. 可能更新 `store`
      因为 child recall pipeline 自己也可能有副作用
4. 主要参数
   1. `prompt`
   2. `retrieve_pipeline`
      mid-decoding 工具真正调用的子 recall pipeline。
   3. `tools`
      默认为 `["MEM_READ"]`。
   4. `max_turns`
   5. `strict_tools`
   6. `allow_no_tool_call`
   7. `runtime_now_factory`
   8. `api_key` / `base_url` / `model` / `embedding_model`
5. 特殊行为
   1. 这是 readout 里的 agentic 模式，不是简单模板渲染。
   2. `MEM_READ` 的结果不会直接覆盖最终答案，而是作为工具结果回给 LLM，LMM 再继续生成最终自然语言回答。

### 9.8 `TemplateReadout`

1. 功能
   通过模板系统把 retrieval 结果渲染成最终文本。
2. 读取
   读取：
   1. `packet.retrieved`
   2. `packet.query`
   3. `prompt`
   4. `PromptPlan` recall 相关配置
3. 写回
   写 `packet.readout` 和 trace。
4. 主要参数
   1. `prompt`
      可以是普通模板字符串，也可以是完整 `PromptPlan`。
   2. `filters`
      当前不支持自定义 filters，传非空会报错。
   3. `missing_value`
      模板变量缺失时的替代文本。
   4. `note_namespace`
   5. `default_category`
   6. `runtime_now_factory`
5. 特殊行为
   1. `readout.source_ids` 取自模板解析 metadata 里的 `used_record_ids`，不是简单等于 `retrieved.items`。
   2. 因此一个模板如果只引用了部分 records，`source_ids` 也会更精确。

## 10. 写路径 Tool Call 协议

适用模块：

1. `LLMFunctionCallOrganization`
2. `LLMFunctionCallEvolution`

### 10.1 写路径上下文对象

公开类型如下：

```python
from memprimitive import WriteToolSpec, WriteToolCallContext, WriteToolResult
```

#### `WriteToolSpec`

```python
@dataclass(slots=True, frozen=True)
class WriteToolSpec:
    name: str
    description: str
    parameters_json_schema: dict[str, Any]
    executor: Callable[[WriteToolCallContext, dict[str, Any]], WriteToolResult]
    requires_contracts: tuple[str, ...] = ()
    produces_contracts: tuple[str, ...] = ()
```

#### `WriteToolCallContext`

```python
@dataclass(slots=True)
class WriteToolCallContext:
    packet: Packet
    store: MemoryStore
    module_slot: Literal["organization", "memory_evolution"]
    default_target_layer: str | None
    selected_records: list[MemoryRecord]
    visible_records: list[MemoryRecord]
```

注意这里有两个容易混淆的变量：

1. `selected_records`
   当前演化 / 维护明确选中的 records。
2. `visible_records`
   模型在 prompt 里可以看到、也允许当作工具决策参考的 records。
   它通常包含 `selected_records`，也可能包含 prompt recall 额外召回的记录。

#### `WriteToolResult`

```python
@dataclass(slots=True)
class WriteToolResult:
    effects: list[dict[str, Any]]
    store: MemoryStore
```

### 10.2 十一个内置 write tools

1. `ADD`
   新增普通 memory record。
   参数：
   1. `text` 必填
   2. `target_layer` 可选
   3. `metadata` 可选
   4. `unit_id` 可选
   5. `timestamp` 可选
2. `UPDATE`
   更新已有 record。
   参数：
   1. `record_id` 必填
   2. `text` 可选
   3. `metadata_patch` 可选
3. `DELETE`
   删除已有 record。
   参数：
   1. `record_id` 必填
   2. `reason` 可选
4. `GRAPH_ADD`
   新增 graph record，并标准化 `metadata["graph"]`。
5. `GRAPH_UPDATE`
   更新 graph record，并重新标准化 graph metadata。
6. `GRAPH_DELETE`
   删除 graph record，并清理同层 dangling links。
7. `GRAPH_ADD_LINK`
   给一条 graph record 追加 outgoing links。
   参数：
   1. `record_id`
   2. `links`
8. `GRAPH_UPDATE_LINK`
   全量替换 outgoing links。
9. `GRAPH_DELETE_LINK`
   删除部分 outgoing links。
10. `UPSERT_PROFILE_FEATURE`
   新增或更新一条结构化 profile feature。
   身份键为：
   1. `target_layer`
   2. `set_id`
   3. `category`
   4. `tag`
   5. `feature`
   参数：
   1. `category` 必填
   2. `tag` 必填
   3. `feature` 必填
   4. `value` 必填，作为 record text
   5. `set_id` 可选
   6. `target_layer` 可选；没有时使用模块的 `target_layer`
   7. `source_episode_record_ids` 可选
   8. `citation_record_ids` 可选
   9. `metadata` 可选
   写入时会同时维护 `metadata["profile_feature"]`、顶层 `category/tag/feature/value`，以及 `source_episode_record_ids` / `citation_record_ids` / `citations`。
11. `DELETE_PROFILE_FEATURE`
   删除一条结构化 profile feature。
   参数：
   1. `record_id` 可选
   2. 或使用 `category` / `tag` / `feature` / `set_id` / `target_layer` 做精确 key 删除
   3. `reason` 可选

### 10.3 默认工具的行为要点

1. 所有工具都要求参数是 JSON object。
2. 参数检查基于 `parameters_json_schema` 做最小 JSON type 校验。
3. profile feature 工具只把结构化 feature 存成普通 `MemoryRecord`，不新增 pipeline slot；通常配合 `LLMFunctionCallEvolution(tools=["UPSERT_PROFILE_FEATURE", "DELETE_PROFILE_FEATURE"])` 使用。
4. graph link 三个工具只改：
   1. `metadata["graph"]["links"]`
   2. `link_count`
   3. `last_linked_at`
   不会顺便改文本或 triples。
5. tool 执行结果会累计到 trace：
   1. `tool_calls`
   2. `effects`
   3. `written_record_ids`
   4. `updated_record_ids`
   5. `deleted_record_ids`
5. 内置工具写出的 record 会自动补：
   `metadata["llm_tool"]`
6. 写路径 tool 默认容忍单次调用失败：
   1. 失败调用不会修改 store。
   2. 失败会记录到 `tool_calls` / `failed_tool_calls`。
   3. `max_retry` 会把原始 prompt、context 和失败摘要一起交回 LLM 重跑。
   4. 需要失败即报错时，优先使用 `raise_on_tool_error=True`；旧的 `strict_tools=True` 仍表示立即抛异常。

## 11. Readout Tool 协议

适用模块：

1. `MidDecodingMemoryReadout`

### 11.1 公开类型

```python
from memprimitive.utils._mid_decoding_tools import (
    ReadoutToolSpec,
    ReadoutToolCallContext,
    ReadoutToolResult,
)
```

#### `ReadoutToolCallContext`

```python
@dataclass(slots=True)
class ReadoutToolCallContext:
    packet: Packet
    store: MemoryStore
    retrieve_pipeline: Any | None = None
```

### 11.2 当前唯一内置 readout tool：`MEM_READ`

1. 功能
   运行一个子 recall pipeline 做记忆读取。
2. 参数
   1. `query: str`
3. 返回结果字段
   1. `tool_name`
   2. `query`
   3. `matched`
   4. `memory_text`
   5. `source_ids`
   6. `match_count`
4. 特殊行为
   1. `MEM_READ` 要求 `retrieve_pipeline` 非空。
   2. 如果子 recall pipeline 没有自定义 readout，会使用一个默认 fallback readout plan：
      `{{ retrieved.items | join_text }}`

## 12. `PromptPlan` 与模板系统

很多 module 都接受 `PromptPlan | str`。这是当前 DSL 里最重要的 prompt/render 统一抽象。

### 12.1 `PromptPlan` 结构

```python
@dataclass(slots=True, frozen=True)
class PromptPlan:
    mode: Literal["simple", "structured"]
    template: str | dict[str, Any] | list[Any]
    context_builder: ContextBuilder | None = None
    recall_plan: PromptPlan | None = None
    labeled_recall_plans: dict[str, PromptPlan] | None = None
    recall_query_builder: RecallQueryBuilder | None = None
    labeled_recall_query_builders: dict[str, RecallQueryBuilder] | None = None
    missing_value: str = ""
    metadata_mode: Literal["prompt", "readout"] = "prompt"
    sub_recall_pipeline: Any | None = None
    labeled_sub_recall_pipelines: dict[str, Any] | None = None
    visible_record_recall_labels: tuple[str, ...] | None = None
```

### 12.2 两种模式

1. `mode="simple"`
   普通字符串模板，形如 `{{ query.text }}`。
2. `mode="structured"`
   结构化 block 模板，适合复杂 readout 组织。

辅助构造函数：

1. `text_prompt(...)`
2. `structured_prompt(...)`
3. `ensure_prompt_plan(...)`

### 12.3 默认模板上下文

`render_prompt_plan(...)` 至少会提供：

1. `query`
2. `runtime`
3. `trace.packet`
4. `trace.retrieval`

如果当前是 readout 或 packet 里已经有 retrieval 结果，还会额外提供 retrieval 上下文，例如：

1. `retrieved`
2. note render 相关上下文
3. 记录投影结果

各 module 还会用 `context_builder` 再往里补自己的局部变量，例如：

1. `unit`
2. `selected_records`
3. `visible_records`
4. `tools`
5. `default_target_layer`

### 12.4 recall plan 的作用

`PromptPlan` 可以在渲染 prompt 前先做子 recall，把 recall 输出文本再注入模板上下文。

相关字段：

1. `recall_plan`
   单条未标记 recall。
2. `labeled_recall_plans`
   多条带 label 的 recall。
3. `sub_recall_pipeline`
   执行 recall 的子 pipeline。
4. `labeled_sub_recall_pipelines`
   每个 label 对应独立子 pipeline。
5. `recall_query_builder`
   根据当前 packet/store 动态构造 recall query。
6. `labeled_recall_query_builders`
   label 级 recall query builder。

### 12.5 visible record 控制

这是最近非常重要的一点：

1. `visible_record_recall_labels`
   用来声明哪些 labeled recall 的命中记录应该并入 `visible_records`。
2. 这意味着 prompt side recall 不只是“拿回一段 recalled text”，还可以决定“哪些历史 records 对后续 tool-calling 真正可见”。
3. 该机制已经被 `LLMFunctionCallOrganization` 和 `LLMFunctionCallEvolution` 使用。

### 12.6 空 recall query 的降级行为

如果某个 labeled recall query builder 渲染出了空字符串，不会直接报错，而是记录：

1. `disabled_reason = "empty_rendered_recall_query"`

这在 prompt/readout metadata 里可见。

### 12.7 template / prompt metadata 里常见的字段

1. `template_mode`
2. `prompt_is_template`
3. `rendered_prompt`
4. `recalled_prompt`
5. `resolved_variables`
6. `missing_variables`
7. `used_record_ids`
8. `used_group_ids`
9. `filter_trace`
10. `labeled_recall_prompts`

## 13. 默认 pipeline 与公开面说明

### 13.1 `MemoryPipeline` 默认模块

如果你不传 slot 参数，默认是：

1. `unit_formation = PassThroughUnitFormation()`
2. `representation = BasicRepresentation()`
3. `write_trigger = AlwaysTrigger()`
4. `organization = AppendOrganization()`
5. `evolution_trigger = NeverTrigger()`
6. `memory_evolution = AppendOnlyEvolution()`
7. `retrieval = RecencyRetrieval()`
8. `readout = ConcatenateReadout()`

### 13.2 当前 baseline surface 的几个边界

1. `DispatchRetrieval` 当前不是公开 baseline surface 的一部分。
2. `NeverTrigger` 当前主要作为 evolution 默认值存在，但仍然是实际可用类。
3. 文档中所有 module 顺序按当前 registry 导出的公开面书写，而不是按历史文档兼容项排列。

## 14. 评测接入用 memory binding 接口

这一节不是 pipeline slot，而是 classics / benchmark 之间的稳定边界。目标是让一个 memory 系统只要实现最小接口，就能被评测脚本直接包装，而不用为每篇论文复制一套 LoCoMo loader、双 speaker session、progress、session reuse 和 JSONL 输出逻辑。

### 14.1 memory 系统必须提供的接口

一个可接入评测的 memory 系统应暴露一个 binding 对象，满足 `MemorySystemBinding` 协议：

```python
class MemorySystemBinding(Protocol):
    name: str

    def build_system(self) -> Any:
        ...

    def ingest_event(self, system: Any, event: MemoryIngestEvent) -> Any:
        ...

    def recall(self, system: Any, query: Query, *, context: RecallContext) -> MemoryRecall | str | Any:
        ...
```

含义：

1. `name`
   系统短名，写入 benchmark prediction 的 `memory_adapter_name` 和 recall metadata。
2. `build_system()`
   创建一个干净、隔离的 memory system。不要复用全局 store；评测 adapter 会按 session 需要调用它。
3. `ingest_event(system, event)`
   写入一条标准化 memory event。系统内部可以把 event 转成自己的原生 helper 调用，例如 Mem0 的 pairwise 写入或 MemMachine 的 episode 写入。
4. `recall(system, query, *, context)`
   根据 `Query` 返回召回结果。推荐返回 `MemoryRecall`；返回 `str` 或带 `text/source_ids/metadata` 字段的对象也会被 benchmark adapter 归一化。

classics 模块推荐再提供一个工厂函数：

```python
def create_memory_binding(**params) -> MemorySystemBinding:
    ...
```

评测 CLI / adapter 只需要调用这个工厂函数，不应该知道系统内部有哪些 pipeline、store layer、tool、summary 或 graph 结构。

### 14.2 `MemoryIngestEvent`

`MemoryIngestEvent` 是 benchmark 到 memory 的标准写入事件：

```python
@dataclass
class MemoryIngestEvent:
    text: str
    session_id: str
    turn_id: str
    user_id: str
    speaker: str
    role: str = "user"
    timestamp: str | None = None
    context_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

字段约定：

1. `text`
   当前目标视角下的主文本。LoCoMo 双 speaker adapter 会把目标 speaker 的话放在这里，并保留 speaker label。
2. `context_text`
   与主文本同轮相关的上下文文本。对 pairwise 系统，通常是另一位 speaker 的回复；对 episode 系统，可以和 `text` 拼接后作为一条 episode。
3. `session_id` / `turn_id` / `timestamp`
   数据集中的会话、轮次和时间信息。需要时间敏感行为的系统应把 `timestamp` 传入底层 `Observation` 或 record metadata。
4. `user_id` / `speaker` / `role`
   当前 memory system 视角下的用户标识和说话人。LoCoMo adapter 会为两个 speaker 分别生成独立 `user_id`。
5. `metadata`
   benchmark adapter 附带的额外信息。memory 系统可以保留这些字段用于 trace，但不应依赖某个 benchmark 私有字段才能工作。

### 14.3 `RecallContext`

`RecallContext` 是 benchmark 在 recall 阶段给 memory 的上下文：

```python
@dataclass
class RecallContext:
    sample_id: str
    user_id: str
    speaker: str
    speaker_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

使用边界：

1. memory 系统可以用它做 scope、trace 或输出 metadata。
2. memory 系统不应把 LoCoMo 的 `qa_category`、`adversarial_answer` 等字段写死进召回逻辑。
3. 如果系统本身不需要上下文，可以忽略 `context`，但函数签名仍应保留。

### 14.4 LoCoMo 双 speaker adapter 的职责

LoCoMo 的双 speaker 形状属于评测适配层，不属于单个 memory 系统。`DualSpeakerLoCoMoMemoryAdapter` 负责：

1. 为 `speaker_a` 和 `speaker_b` 各调用一次 `binding.build_system()`，得到两个独立系统。
2. 按目标 speaker 视角把对话 turn pair 转成 `MemoryIngestEvent(text=..., context_text=...)`。
3. 对同一个 `locomo_sample_id` 复用已加载 memory session，避免每道 QA 重复生成 memory。
4. 合并两个 speaker 的 recall，并在 metadata 中写入：
   1. `speaker_1_memories`
   2. `speaker_2_memories`
   3. `speaker_1_user_id`
   4. `speaker_2_user_id`
   5. `num_speaker_1_memories`
   6. `num_speaker_2_memories`

因此，新 classics 系统如果实现了 `create_memory_binding()`，评测脚本可以通过 `create_dual_speaker_locomo_memory_adapter(lambda: module.create_memory_binding(...))` 直接接入。

CLI 也支持同一接口：

```bash
python -m memprimitive.benchmarking.minimal_baseline \
  --benchmark locomo \
  --memory-adapter binding \
  --memory-binding memprimitive.example.classics.mem0_memory:create_memory_binding \
  --memory-binding-kwargs '{"recall_top_k": 30}'
```

### 14.5 现有示例

当前已按这个接口接入的 classics 系统：

1. `memprimitive/example/classics/mem0_memory.py`
   `Mem0MemoryBinding` 将 `MemoryIngestEvent.text/context_text` 映射到 `ingest_message_pair(user_text=..., assistant_text=...)`，并用 `recall_profile(...)` 返回 profile memory。
2. `memprimitive/example/classics/memmachine_memory.py`
   `MemMachineMemoryBinding` 将 `MemoryIngestEvent.text/context_text` 合并成 episode，调用 `ingest_episode(...)`，并用 `recall_memmachine_context(...)` 返回 STM summary、working memory、LTM episodes 和 profile memory。

## 15. 建议查阅路径

如果你需要继续往下追实现，推荐按下面顺序查：

1. `memprimitive/pipeline.py`
2. `memprimitive/core.py`
3. `memprimitive/baselines/*.py`
4. `memprimitive/utils/_template.py`
5. `memprimitive/utils/_llm_function_tools.py`
6. `memprimitive/utils/_mid_decoding_tools.py`

如果你要让 agent 自动使用这份文档，建议优先让它根据这几个关键词定位：

1. module 所属 slot
2. 读取的 `packet.*` 字段
3. 写回的 `packet.*` / `store.*` / `trace.*`
4. `PromptPlan` 是否参与
5. 是否会真实修改 `store`
