# Memory Pipeline 设计 agent 语义操作灵感清单

## 启发型操作

1. PromptPlan 时间锚点注入
   类型：启发型
   含义：把 `timestamp` / `source_timestamp` / `speaker` 显式放进生成或 readout prompt。
   触发点：看到 `PromptPlan`、`TemplateReadout`、timestamped episode readout。
   落地：`context_builder`、`structured_prompt` block、MemMachine `_timestamped_episode_prompt(...)`、`BasicRepresentation(elements=("time_anchor", "source_type"))`。

2. PromptPlan 先召回再生成字段
   类型：启发型
   含义：生成 representation / trigger / tool prompt 前，先跑一个局部 recall，把旧记忆作为上下文。
   触发点：看到 `LLMRepresentation(prompt=...)` 或 tool-call prompt。
   落地：`PromptPlan.recall_plan`、`recall_query_builder`、`sub_recall_pipeline`。

3. 多标签 sub-recall 分支
   类型：启发型
   含义：同一个 prompt 同时召回 profile、summary、recent、graph、episode 等分支，再合并决策。
   触发点：看到 Mem0 / Mem0g 的 `conversation_summary`、`recent_messages`、`topk_similar`。
   落地：`labeled_recall_plans`、`labeled_sub_recall_pipelines`、`labeled_recall_query_builders`。

4. Prompt 可见集和 tool 可改集分离
   类型：启发型
   含义：让 prompt 可以看多个 recall 分支，但只允许工具修改指定 label 的记录。
   触发点：看到 `visible_record_recall_labels`。
   落地：在 `PromptPlan.visible_record_recall_labels` 中只放 `topk_similar`、`graph_context` 或同 bucket 分支。

5. selected records 加 prompt recall 合并
   类型：启发型
   含义：维护阶段不只看 trigger 选中的记录，也补入 prompt recall 找到的相关记录。
   触发点：看到 `LLMFunctionCallEvolution` 的 `selected_records_plus_prompt_recall`。
   落地：`source_layer` / `decisions_store` 选中当前记录，`PromptPlan` 再召回候选，工具只处理合并后的 `visible_records`。

6. recall query fallback 链
   类型：启发型
   含义：构造 recall query 时按 unit text、user message、assistant message、pair text 逐级回退。
   触发点：看到 Mem0 / Mem0g 的 `labeled_recall_query_builders`。
   落地：lambda builder 从 `context["unit"]["text"]`、`user_message`、`assistant_message`、`pair_text` 中取第一个非空值。

7. readout 排序和选择信号分离
   类型：启发型
   含义：用 reranker 选择证据，但最终按时间顺序渲染给 answer prompt。
   触发点：看到 `rerank` 后直接按得分顺序输出。
   落地：`EpisodeClusterRerankRetrieval(chronological=True)`、`TemporalNeighborExpansionRetrieval(chronological=True)`。

8. record rerank 改成 cluster rerank
   类型：启发型
   含义：把单条 record 候选升级为 nucleus+neighbors 的 episode cluster 候选。
   触发点：看到普通 `RerankerRetrieval` 或 first-stage top-k。
   落地：`TemporalNeighborExpansionRetrieval` 产 `trace["clusters"]`，再接 `EpisodeClusterRerankRetrieval(rerank=True)`。

9. cluster 截断优先 nucleus
   类型：启发型
   含义：预算不够放完整 cluster 时，先保留命中 nucleus，再保留最邻近上下文。
   触发点：看到 `top_k` 小于一个 cluster 大小。
   落地：`EpisodeClusterRerankRetrieval._partial_cluster_records()` 的 partial cluster path。

10. forward neighbor tie-break
    类型：启发型
    含义：同距离前后邻居竞争预算时，优先保留 forward neighbor。
    触发点：看到 temporal expansion 或 partial cluster budget。
    落地：`EpisodeClusterRerankRetrieval` tight-budget ordering、MemMachine forward-first 语义。

11. forward-biased temporal window
    类型：启发型
    含义：时间邻居窗口不必前后对称，可以让 future context 多于 past context。
    触发点：看到 `TemporalNeighborExpansionRetrieval(backward=..., forward=...)`。
    落地：MemMachine `_neighbor_window(expand_context=...)`、`backward=expand//3`、`forward=expand-backward`。

12. temporal scope 隔离
    类型：启发型
    含义：只在同 session / user / agent scope 内扩时间邻居。
    触发点：看到 temporal neighbor 命中跨会话记录。
    落地：`TemporalNeighborExpansionRetrieval(scope_fields=("session_id", ...))`、`STMConsolidationEvolution(scope_metadata_keys=...)`。

13. timestamp 排序兜底显式化
    类型：启发型
    含义：先按 timestamp+record_id 排序；timestamp 不可解析时保留 store iteration order。
    触发点：看到 temporal retrieval 对顺序敏感。
    落地：`TemporalNeighborExpansionRetrieval` trace 的排序来源、`timestamp_record_id` vs `store_iteration`。

14. sentence hit 回 parent episode
    类型：启发型
    含义：细粒度命中只负责找入口，最终 evidence 使用完整 episode。
    触发点：看到 sentence layer、sentence split、embedding recall。
    落地：`EmbeddingSimilarityRetrieval(layer="sentence")` 接 `ParentEpisodeExpansionRetrieval(episode_layer="episodic")`。

15. parent id 从 metadata/provenance 取
    类型：启发型
    含义：parent expansion 不从文本猜 parent，而从显式字段解析。
    触发点：看到 sentence derivative index 或 raw text 中带 episode id。
    落地：`parent_episode_record_id`、`source_episode_record_id`、`metadata["provenance"]`。

16. duplicate parent score promotion
    类型：启发型
    含义：多个 sentence 命中同一个 parent 时只返回一次 parent，但保留更高分。
    触发点：看到 sentence top-k 里一个 episode 多句重复命中。
    落地：`ParentEpisodeExpansionRetrieval` 的 parent dedupe 和 score promotion。

17. immediate LTM index 快路
    类型：启发型
    含义：写入 working 后立即同步写 episodic/sentence index，避免等 STM overflow。
    触发点：看到小 STM budget 或 benchmark 刚写完就问。
    落地：MemMachine `_append_direct_ltm_index(...)`。

18. STM summary 和 raw episode 双轨
    类型：启发型
    含义：旧 working 记录既可以 raw copy 进 episodic，也可以被压缩成 session summary。
    触发点：看到 STM overflow 只做摘要或只做删除。
    落地：`STMConsolidationEvolution(copy_evicted_to_ltm=True, summary_layer=...)`。

19. 当前 summary 单记录替换
    类型：启发型
    含义：每个 scope 只保留当前 summary，旧 summary 被删除。
    触发点：看到 summary layer 越写越多。
    落地：`STMConsolidationEvolution._summary_records_for_scope(...)` 与 old summary deletion。

20. raw LTM copy 可关闭
    类型：启发型
    含义：如果已有 immediate LTM index，STM overflow 可以只做 summary，不重复 raw copy。
    触发点：看到 write path 已经同步写 episodic/sentence。
    落地：MemMachine write pipeline 中 `STMConsolidationEvolution(copy_evicted_to_ltm=False)`。

21. profile 保守 upsert
    类型：启发型
    含义：只把稳定偏好、身份、习惯、事实写入 profile，不把临时对话片段全写进去。
    触发点：看到 profile prompt 或 `UPSERT_PROFILE_FEATURE`。
    落地：`_profile_prompt()`、`build_profile_feature_tools()`、`LLMFunctionCallEvolution(target_layer="profile")`。

22. profile delete 只处理明确矛盾
    类型：启发型
    含义：删除 profile 要求显式反证，避免因为新事实缺失就删旧事实。
    触发点：看到 `DELETE_PROFILE_FEATURE` 或普通 `DELETE` 工具。
    落地：profile prompt 中限制 delete；工具参数用 `record_id` 或结构化 key。

23. profile identity key 结构化
    类型：启发型
    含义：profile 不是自由文本去重，而是按 set/category/tag/feature 定位。
    触发点：看到 profile feature upsert。
    落地：`target_layer + set_id + category + tag + feature`。

24. source/citation id 合并保留
    类型：启发型
    含义：upsert 同一 profile feature 时合并证据 id，而不是覆盖掉旧 provenance。
    触发点：看到 profile update 或 source ids 丢失。
    落地：`source_episode_record_ids`、`citation_record_ids`、selected records source merge。

25. top-k similar 限制 tool target
    类型：启发型
    含义：写 fact/update profile 时，只让 LLM 操作相似候选，不扫全库。
    触发点：看到 Mem0-style `topk_similar`。
    落地：`visible_record_recall_labels=("topk_similar",)`、`PromptRecallSelectionTrigger`。

26. tool failure feedback retry
    类型：启发型
    含义：工具调用失败后，把失败详情放进下一轮 agent context 让模型修正参数。
    触发点：看到 tool schema 严格或写工具容易失败。
    落地：`LLMFunctionCallOrganization/Evolution(max_retry=..., strict_tools=True)`。

27. no-op 作为合法动作
    类型：启发型
    含义：维护 agent 可以不调用工具，避免为满足格式制造无意义写入。
    触发点：看到 `allow_no_tool_call` 或 "If no change is needed" prompt。
    落地：`allow_no_tool_call=True`、tool prompt 中显式允许 no action。

28. merge 拆成 update keeper + delete redundant
    类型：启发型
    含义：没有专用 merge primitive 时，用 UPDATE 一个代表记录再 DELETE 冗余记录表达 merge。
    触发点：看到 TiM bucket update 或重复 thought。
    落地：TiM `_build_tim_bucket_evolution_module(...)` 的 `UPDATE` + `DELETE`。

29. bucket-local maintenance
    类型：启发型
    含义：只在同 hash bucket 内做插入、忘记、合并，不让维护扫全局。
    触发点：看到 TiM hash group 或 metadata filter。
    落地：`MetadataRetrieval(field="hash_bucket", target=...)`、full bucket visibility pipeline。

30. hash prefilter 再语义 top-k
    类型：启发型
    含义：先用 hash/metadata 锁定候选组，再在组内 embedding similarity 排序。
    触发点：看到两阶段 TiM recall。
    落地：`MetadataRetrieval(source="store")` 接 `EmbeddingSimilarityRetrieval(source="retrieved")`。

31. relation-bearing thought 抽取
    类型：启发型
    含义：memory atom 不存原回答，而存带 head/relation/tail 的归纳 thought。
    触发点：看到 TiM post-thinking extraction。
    落地：`LLMRepresentation(field="thoughts", value_type=list[dict[str,str]])`。

32. 历史 thought 参与新 thought 抽取
    类型：启发型
    含义：抽新 thought 时给模型看历史 thoughts，减少重复并保持关系 schema。
    触发点：看到 `historical_thoughts` prompt block。
    落地：TiM `_build_tim_thought_extraction_representation()` 的 `context_builder`。

33. metadata hint 跳过 LLM
    类型：启发型
    含义：如果上游已给 keywords/entities/triples，就直接用 hint，不再重复调用 LLM。
    触发点：看到 adapter 或 benchmark 已有结构字段。
    落地：`BasicRepresentation(elements=("keywords",))`、`TripleRepresentation` metadata hint path。

34. embedding 文本表面重写
    类型：启发型
    含义：不换 embedding 模型，只改变送入 embedding 的文本投影。
    触发点：看到 raw text embedding 召回差。
    落地：`ConfigurableEmbeddingRepresentation(embedding_text=text_prompt(...), embedding_version=...)`。

35. note-style embedding 同构投影
    类型：启发型
    含义：memory 侧和 query 侧都投影成 content/context/keywords/tags 风格文本。
    触发点：看到 A-MEM note payload 或 query expansion。
    落地：`build_enhanced_embedding_text(...)`、`VectorGraphSeedAndExpandRetrieval(query_expand_with_llm=True)`。

36. tags/context/keywords 分字段控制
    类型：启发型
    含义：把 context、keywords、tags、category、attributes 拆成可独立替换的小字段。
    触发点：看到 A-MEM representation pipeline。
    落地：多段 `LLMRepresentation(field=...)` 接 `ConfigurableEmbeddingRepresentation`。

37. graph source 和 graph record 分离
    类型：启发型
    含义：同时保存 raw source observation 和结构化 graph record。
    触发点：看到 graph 写入后难回溯原始证据。
    落地：`GraphAppendOrganization(separate=True, separate_layer=...)`、`GraphEntityDeduplicationAppendOrganization(separate=True)`。

38. entity-level dedup 替代 record-level dedup
    类型：启发型
    含义：合并图记忆时按实体 embedding 对齐节点，而不是合并整条 graph record。
    触发点：看到人物/地点长期聚合。
    落地：`TripleRepresentation(embed_entities=True)` 接 `GraphEntityDeduplicationAppendOrganization(threshold=...)`。

39. graph seed-expand 后再 lexical rerank
    类型：启发型
    含义：先用向量/图扩候选，再用 BM25 或 lexical signal 在扩展结果上重排。
    触发点：看到 graph recall 结果过宽。
    落地：`VectorGraphSeedAndExpandRetrieval(...)` 接 `BM25Retrieval(source="retrieved")`。

40. query rewrite 产实体列表
    类型：启发型
    含义：把自然语言 query 改写成多个 canonical entity query，而不是一整句 embed。
    触发点：看到 graph lookup 或 Mem0g graph recall。
    落地：`QueryRewriteRetrieval(strategy="llm", allow_multi_query=True, max_queries=...)`。

41. regex rewrite 清理 benchmark 噪声
    类型：启发型
    含义：用规则删掉固定题干、speaker 前缀、日期模板噪声。
    触发点：看到 query 格式固定且 LLM rewrite 成本不必要。
    落地：`QueryRewriteRetrieval(strategy="regex", regex_rules=[...])`。

42. source="retrieved" 串联检索
    类型：启发型
    含义：后一个 retriever 只处理前一个 retriever 的候选集。
    触发点：看到需要 filter -> rerank 或 seed -> expand -> rerank。
    落地：`EmbeddingSimilarityRetrieval(source="retrieved")`、`BM25Retrieval(source="retrieved")`、`RerankerRetrieval(source="retrieved")`。

43. lexical no-match recency fallback
    类型：启发型
    含义：BM25 没有正命中时回退最近记录，避免空 context。
    触发点：看到 lexical retrieval 经常返回空。
    落地：`BM25Retrieval` 的 `used_recency_fallback` trace。

44. JSON evidence surface
    类型：启发型
    含义：把 readout 给下游 agent 时保留 item/source id JSON，而不是拼自然语言。
    触发点：看到 source ids 丢失或评分需要 evidence id。
    落地：`JSONReadout()`、`TemplateReadout` metadata 的 `used_record_ids`。

45. relation sentence readout fallback
    类型：启发型
    含义：图证据优先渲染三元组关系句；没有关系句时回退原文。
    触发点：看到 graph record 里 triples 与 text 混杂。
    落地：`GraphRelationReadout()`。

46. summary-with-sources 视图
    类型：启发型
    含义：渲染 summary 时绑定它的 source records，避免摘要和证据脱钩。
    触发点：看到 hierarchical summary readout。
    落地：`_template_readout.structure_retrieved_items()` 的 `views["summary_with_sources"]`。

47. domain-locked shared memory recall
    类型：启发型
    含义：共享 memory pool 召回前先按 domain/rubric metadata 锁定子池。
    触发点：看到 Memory Sharing / INMS shared pool。
    落地：`MetadataRetrieval(field=..., target=...)` 接 `EmbeddingSimilarityRetrieval(source="retrieved")`。

48. LLMJudge score gate
    类型：启发型
    含义：写入共享示例前先用 rubric score 判断是否值得长期保存。
    触发点：看到 accepted example pool 或 few-shot memory。
    落地：`LLMJudgeTrigger(decision_mode="score", threshold=...)`。

49. outcome-conditioned reflection
    类型：启发型
    含义：只有失败或低分 trial 触发 reflection 生成。
    触发点：看到 Reflexion failed trial memory。
    落地：`ScalarRuleTrigger(signal_key="is_correct", comparator="<", threshold=0.5)`。

50. reflection 生成看旧 reflection
    类型：启发型
    含义：生成新 reflection 时召回已有 reflection，避免重复同一教训。
    触发点：看到 hierarchical generation prompt。
    落地：`HierarchicalEvolution(prompt=text_prompt(... recall_plan=..., sub_recall_pipeline=...))`。

51. high-level layer retention
    类型：启发型
    含义：高级 memory 层只保留最近 K 条，防止摘要/反思无限膨胀。
    触发点：看到 reflection layer、compressive layer、session summary layer。
    落地：`HierarchicalEvolution(retention_size=...)`。

52. record_text_field 选主文本
    类型：启发型
    含义：结构化抽取结果可保留完整 metadata，但 record.text 只用其中一个字段。
    触发点：看到 `extract_fields` 生成多个字段。
    落地：`HierarchicalEvolution(record_text_field="reflection" / "summary")`。

53. boundary close-session 聚合
    类型：启发型
    含义：不要每轮都总结 session；只在 session_end 事件把整段 session 聚合。
    触发点：看到 COMEDY session memory。
    落地：`BoundaryEventTrigger(accepted_events=("session_end",))` + `HierarchicalOrganization(group_by=("session_id",))`。

54. periodic maintenance 只在维护观测执行
    类型：启发型
    含义：周期压缩不在普通 ingest turn 上隐式扫描全层，而由 maintenance observation 驱动。
    触发点：看到 `PeriodicMaintenanceTrigger(StoreAllTrigger(...))`。
    落地：COMEDY 的 maintenance tick observation、`HierarchicalEvolution(source_layer=...)`。

55. latest compressive memory only
    类型：启发型
    含义：reply context 只读最新 compressive snapshot，不拼全部历史摘要。
    触发点：看到 compressive memory 层。
    落地：`RecencyRetrieval(top_k=1, layer="compressive_memory")`。

56. short memory sliding state
    类型：启发型
    含义：短期状态只保留最新一条，让模型每轮重写当前状态。
    触发点：看到 RecurrentGPT short memory。
    落地：short memory layer `capacity="sliding_window"`、`settings={"record_budget": 1}`、`RecencyRetrieval(top_k=1)`。

57. long memory 延迟写前一段
    类型：启发型
    含义：生成当前内容时检索 long memory，写入 long memory 的却是上一轮最终内容。
    触发点：看到 RecurrentGPT repo-style loop。
    落地：orchestration 中 `_append_long_memory(... previous paragraph ...)`。

58. labeled output parse retry
    类型：启发型
    含义：模型输出必须包含固定 label；解析失败时重试而不是静默接受。
    触发点：看到 RecurrentGPT writer/plan parser。
    落地：`_generate_with_parse_retry(...)`、`_require_labeled_sections(...)`。

59. structured triple query with wildcards
    类型：启发型
    含义：MEM_READ query 使用 `subject >> relation >> object`，未知 slot 用 `*`。
    触发点：看到 RET-LLM / explicit memory read。
    落地：`RETLLMTripleMemoryRetrieval`、`MEM_READ` prompt。

60. triple slot fuzzy fallback
    类型：启发型
    含义：结构化三元组查不到时，把 query slot 近似映射到已有 memory terms。
    触发点：看到 exact triple lookup 过严。
    落地：`_nearest_existing_term(...)`、`fallback_similarity_threshold`。

61. mid-decoding memory read
    类型：启发型
    含义：回答时不一次性塞上下文，而让 LLM 在生成中按需调用 MEM_READ。
    触发点：看到 RET-LLM 或 `MidDecodingMemoryReadout`。
    落地：`MidDecodingMemoryReadout(retrieve_pipeline=..., tools=["MEM_READ"])`。

62. graph relation 当前有效集维护
    类型：启发型
    含义：没有 edge valid=false 时，用 link add/update/delete 维护当前有效 graph。
    触发点：看到 Mem0g stale relation 处理。
    落地：`GRAPH_ADD_LINK`、`GRAPH_UPDATE_LINK`、`GRAPH_DELETE_LINK`。

63. graph neighbor snapshot 入 metadata
    类型：启发型
    含义：把 graph 邻居摘要写成 record metadata，供 readout/prompt 使用。
    触发点：看到 graph neighbor 每次 readout 都要重算。
    落地：`GraphNeighborContextTraceEvolution(rewrite_metadata=True)`、`GraphLinkEvolution(rewrite_neighbor_metadata=True)`。

64. working 和 episodic 去重合并
    类型：启发型
    含义：同时读 working 与 episodic 时，用 unit_id 去掉已在 working 中出现的 LTM 副本。
    触发点：看到 immediate LTM index 与 working recall 同时存在。
    落地：MemMachine `recall_memmachine_memory(...)` 中按 `working_unit_ids` 过滤 episodic records。

65. source ids 稳定去重
    类型：启发型
    含义：多个 recall 分支合并后保留首次出现的 source id 顺序。
    触发点：看到 profile、summary、episode readout 合并。
    落地：MemMachine `_stable_source_ids(...)`、PromptPlan recall metadata。

66. retrieved trace 作为下一阶段输入
    类型：启发型
    含义：不要只看 retrieved items，也利用 trace 中的 clusters、layer、per-query 等结构。
    触发点：看到 grouped recall shape 或 expansion trace。
    落地：`EpisodeClusterRerankRetrieval` 读取 `packet.retrieved.trace["clusters"]`。

67. grouped recall shape 保留角色
    类型：启发型
    含义：把 nucleus/backward/forward/context 角色保留到 score/readout metadata。
    触发点：看到 temporal cluster 合并。
    落地：`EpisodeClusterRerankRetrieval` scores 的 `roles`、`source_nucleus_record_ids`。

68. group summary 使用 group_key
    类型：启发型
    含义：分组摘要记录保留 group_key，后续可按 session/person/subgoal 回查。
    触发点：看到 `HierarchicalOrganization(group_by=...)`。
    落地：hierarchical metadata 的 `group_key`、`source_record_ids`。

69. Free pipeline 试组合
    类型：启发型
    含义：当标准 slot 限制阻碍探索时，先用宽松 ordered runner 试验组合形状。
    触发点：看到想把多个 retrieval/evolution stage 以非标准方式串起来。
    落地：`FreeMemoryPipeline`、但成熟后再回迁到标准 `MemoryPipeline`。

70. real runtime path 保真
    类型：启发型
    含义：涉及 embedding/rerank/LLM 语义时走真实 runtime，不用规则 stub 替代。
    触发点：看到 rerank、embedding、tool-call、summary rewrite。
    落地：`Runtime.embed()`、`Runtime.rerank()`、`get_runtime().summarize_records(...)`。

## 增加型操作

1. 增加 working memory layer
   类型：增加型
   含义：加入一个短期工作层，保存当前会话最近 raw turns。
   触发点：agent 需要把当前上下文和长期记忆分开。
   落地：`StoreLayerSpec(name="working", theme="working", indices=("temporal",))`、`AppendOrganization(target_layer="working")`。

2. 增加 episodic LTM layer
   类型：增加型
   含义：加入原始 episode 长期层，保留可审计原文。
   触发点：agent 需要完整证据而不是摘要或 profile。
   落地：`StoreLayerSpec(name="episodic", theme="episodic", indices=("temporal",))`。

3. 增加 sentence derivative layer
   类型：增加型
   含义：为 episode 派生句子级索引层，用于高召回语义命中。
   触发点：agent 需要细粒度检索入口。
   落地：`StoreLayerSpec(name="sentence", indices=("vector", "temporal"))`、`STMConsolidationEvolution.ltm_sentence_layer`。

4. 增加 session summary layer
   类型：增加型
   含义：加入每个 scope 当前摘要层，承接 STM consolidation。
   触发点：agent 需要滚动状态而不想总读 raw turns。
   落地：`StoreLayerSpec(name="session_summary", theme="semantic")`、`STMConsolidationEvolution(summary_layer=...)`。

5. 增加 profile memory layer
   类型：增加型
   含义：加入长期用户画像/偏好/稳定事实层。
   触发点：agent 需要 durable user facts。
   落地：`StoreLayerSpec(name="profile", indices=("vector", "temporal"))`、profile feature tools。

6. 增加 graph memory layer
   类型：增加型
   含义：加入 graph-shaped 语义层，保存实体、关系、links。
   触发点：agent 需要关系推理或实体聚合。
   落地：`StoreLayerSpec(shape="Graph", indices=("graph", "vector", "entity"))`。

7. 增加 graph source observation layer
   类型：增加型
   含义：为 graph 写入保留 raw source 记录层。
   触发点：agent 需要 graph relation 之外的原始证据。
   落地：`GraphEntityDeduplicationAppendOrganization(separate=True, separate_layer="graph_source_observation")`。

8. 增加 recent dialogue layer
   类型：增加型
   含义：加入最近对话 buffer，供 profile/graph extraction prompt 使用。
   触发点：agent 需要当前 pair 之外的局部上下文。
   落地：`recent_dialogue` layer、`RecencyRetrieval(top_k=recent_top_k, layer="recent_dialogue")`。

9. 增加 conversation summary layer
   类型：增加型
   含义：加入运行中的对话摘要层，给写入和抽取提供压缩上下文。
   触发点：agent 需要 profile/graph 写入时看历史概要。
   落地：`conversation_summary` layer、`SummaryRewriteEvolution(target_layer="conversation_summary")`。

10. 增加 reflection layer
    类型：增加型
    含义：加入失败经验/反思层，用于下次尝试 prompt。
    触发点：agent 面对 trial/outcome 型任务。
    落地：`HierarchicalEvolution(... record_text_field="reflection", retention_size=...)`、`PromptContextReadout`。

11. 增加 shared memory pool
    类型：增加型
    含义：加入跨任务/跨 agent 的 accepted example pool。
    触发点：agent 需要 reusable few-shot memory。
    落地：Memory Sharing example、`LLMJudgeTrigger` + `AppendOrganization` + embedding recall。

12. 增加 query rewrite stage
    类型：增加型
    含义：在检索前插入 query 改写或多 query 生成。
    触发点：agent 发现原始 query 不适合 retriever。
    落地：`QueryRewriteRetrieval(retriever=..., strategy="llm"/"regex", allow_multi_query=...)`。

13. 增加 runtime rerank stage
    类型：增加型
    含义：在 first-stage recall 后插入真实 reranker。
    触发点：agent 需要从宽候选中压缩到更精确证据。
    落地：`RerankerRetrieval(top_k=..., source="retrieved")`、`Runtime.rerank()`。

14. 增加 parent episode expansion stage
    类型：增加型
    含义：插入 sentence/child hit 到 parent episode 的扩展阶段。
    触发点：agent 已有 derivative index。
    落地：`ParentEpisodeExpansionRetrieval(parent_id_fields=..., episode_layer=...)`。

15. 增加 temporal neighbor expansion stage
    类型：增加型
    含义：插入时间邻居补全阶段。
    触发点：agent 需要命中事件前后的上下文。
    落地：`TemporalNeighborExpansionRetrieval(layer=..., backward=..., forward=..., scope_fields=...)`。

16. 增加 episode cluster rerank stage
    类型：增加型
    含义：插入 cluster 级 rerank、dedupe、budget merge 阶段。
    触发点：agent 已有 temporal clusters。
    落地：`EpisodeClusterRerankRetrieval(top_k=..., cluster_top_k=..., rerank=True)`。

17. 增加 layer-aware retrieval stage
    类型：增加型
    含义：对不同 layer 使用不同 retriever，再融合结果。
    触发点：agent 同时有 working/profile/graph/episodic 多层。
    落地：`LayerAwareRetrieval(retriever_by_layer=..., merge_weight_by_layer=...)`。

18. 增加 metadata filter stage
    类型：增加型
    含义：按 session/user/domain/hash_bucket 等 metadata 先筛候选。
    触发点：agent 需要 scope 隔离或 bucket recall。
    落地：`MetadataRetrieval(field=..., target=..., match_mode="exact"/"regex")`。

19. 增加 lexical retrieval stage
    类型：增加型
    含义：加入 BM25 或 keyword count 检索，补充 embedding blind spot。
    触发点：agent 需要精确词、实体名、ID、日期命中。
    落地：`BM25Retrieval(...)`、`KeywordCountRetrieval(...)`。

20. 增加 embedding similarity retrieval stage
    类型：增加型
    含义：加入向量相似度检索层。
    触发点：agent 需要语义相近召回。
    落地：`EmbeddingSimilarityRetrieval(top_k=..., layer=...)`、query embedding。

21. 增加 graph seed-expand retrieval stage
    类型：增加型
    含义：先找 graph seed，再扩显式邻居。
    触发点：agent 已有 graph layer 和 links。
    落地：`VectorGraphSeedAndExpandRetrieval(candidate_k=..., neighbor_expansion_k=...)`。

22. 增加 graph neighbor expansion stage
    类型：增加型
    含义：在已有 retrieved seeds 上补 graph neighbors。
    触发点：agent 已有 seed retriever，不想重写 graph retriever。
    落地：`ExpandRetrievedGraphNeighbors(include_seed_records=..., per_seed_top_k=...)`。

23. 增加 triple memory retrieval stage
    类型：增加型
    含义：按 subject/relation/object slot 检索结构化 triples。
    触发点：agent 已有三元组记忆。
    落地：`TripleMemoryRetrieval(...)`、RET-LLM structured query wrapper。

24. 增加 mid-decoding MEM_READ readout
    类型：增加型
    含义：在 readout/answer 阶段加入工具式按需检索。
    触发点：agent 不想一次性塞满上下文。
    落地：`MidDecodingMemoryReadout(prompt=..., retrieve_pipeline=..., tools=["MEM_READ"])`。

25. 增加 PromptPlan sub-recall
    类型：增加型
    含义：给任意 prompt 加一个内部 recall pipeline。
    触发点：agent 需要“写/判/读之前先查”。
    落地：`text_prompt(..., recall_plan=..., recall_query_builder=..., sub_recall_pipeline=...)`。

26. 增加 LLM field extractor
    类型：增加型
    含义：为 memory unit 生成 summary、tags、context、attributes、thoughts 等字段。
    触发点：agent 需要非 raw text 的检索或维护表面。
    落地：`LLMRepresentation(field=..., value_type=..., prompt=...)`。

27. 增加 configurable embedding projection
    类型：增加型
    含义：加入可模板化的 embedding 输入文本生成器。
    触发点：agent 想改变向量空间表面。
    落地：`ConfigurableEmbeddingRepresentation(embedding_text=text_prompt(...))`。

28. 增加 two-stage triple extraction
    类型：增加型
    含义：先抽实体，再围绕实体抽关系。
    触发点：agent 需要更稳定的 graph 写入。
    落地：`TripleRepresentation(method="two_stage", embed_entities=True)`。

29. 增加 fanout ingest stage
    类型：增加型
    含义：把一个 list 字段拆成多条 child observations 进入子 pipeline。
    触发点：agent 已从一轮对话抽出 fact_list / claims / thoughts。
    落地：`FanoutIngestOrganization(field=..., pipeline=child_pipeline)`。

30. 增加 hierarchical aggregation stage
    类型：增加型
    含义：把 selected records 按 group 聚合成高级记忆。
    触发点：agent 需要 session/subgoal/person 级摘要。
    落地：`HierarchicalOrganization(...)` 或 `HierarchicalEvolution(...)`。

31. 增加 STM consolidation stage
    类型：增加型
    含义：加入 working overflow 处理、LTM copy、summary rewrite、sentence index。
    触发点：agent 需要容量受限工作记忆。
    落地：`STMConsolidationEvolution(record_budget=..., token_budget=...)`。

32. 增加 summary rewrite stage
    类型：增加型
    含义：把 active units 或 summary field 写成新的摘要记录。
    触发点：agent 需要维护 running summary。
    落地：`SummaryRewriteEvolution(target_layer=...)`。

33. 增加 layer promotion stage
    类型：增加型
    含义：把当前活跃记录复制提升到另一个 layer。
    触发点：agent 需要从 working 迁移到 semantic/profile-candidate 层。
    落地：`LayerMoveEvolution(target_layer=...)`。

34. 增加 graph link maintenance stage
    类型：增加型
    含义：维护 graph links，而不是只追加 graph records。
    触发点：agent 需要关系随新事实更新。
    落地：`GraphLinkEvolution(...)` 或 `LLMFunctionCallEvolution(tools=["GRAPH_ADD_LINK", ...])`。

35. 增加 graph neighbor context maintenance stage
    类型：增加型
    含义：把邻居上下文写入 trace 或 metadata。
    触发点：agent 需要 graph 状态影响后续 prompt。
    落地：`GraphNeighborContextTraceEvolution(rewrite_metadata=True)`。

36. 增加 tool-call write/evolution controller
    类型：增加型
    含义：用 LLM function calling 执行 ADD/UPDATE/DELETE 或自定义工具。
    触发点：agent 需要选择性写入、合并、删除、profile 维护。
    落地：`LLMFunctionCallOrganization(...)`、`LLMFunctionCallEvolution(...)`。

37. 增加 structured profile tools
    类型：增加型
    含义：加入 profile-specific upsert/delete 工具。
    触发点：agent 需要画像层不是普通自由文本 CRUD。
    落地：`build_profile_feature_tools()`、`UPSERT_PROFILE_FEATURE`、`DELETE_PROFILE_FEATURE`。

38. 增加 A-MEM note evolution tools
    类型：增加型
    含义：加入当前 note link strengthening 与 neighbor context/tag update 工具。
    触发点：agent 采用 note graph 记忆。
    落地：`build_amem_evolution_tools(...)`、A-MEM write pipeline。

39. 增加 LLM judge write trigger
    类型：增加型
    含义：加入 LLM 判断是否写入的门控。
    触发点：agent 需要质量过滤或选择性长期保存。
    落地：`LLMJudgeTrigger(prompt=..., decision_mode="score"/"bool"/"label")`。

40. 增加 scalar outcome trigger
    类型：增加型
    含义：按 observation/runtime scalar 信号触发写入或演化。
    触发点：agent 处理 reward、correctness、score、pressure。
    落地：`ScalarRuleTrigger(signal_key=..., comparator=..., threshold=...)`。

41. 增加 boundary event trigger
    类型：增加型
    含义：按 session_end、turn_end、task_end 等边界触发聚合。
    触发点：agent 需要事件边界上的 memory update。
    落地：`BoundaryEventTrigger(accepted_events=(...))`。

42. 增加 periodic / idle maintenance trigger
    类型：增加型
    含义：把维护改成周期或空闲时执行。
    触发点：agent 不希望每次 ingest 都维护全层。
    落地：`PeriodicMaintenanceTrigger(every_n=..., trigger=...)`、`IdleMaintenanceTrigger(...)`。

43. 增加 JSON readout
    类型：增加型
    含义：把 retrieved items 渲染为结构化 JSON evidence。
    触发点：agent 下游需要 source id、layer、timestamp。
    落地：`JSONReadout()`。

44. 增加 structured template readout
    类型：增加型
    含义：用 block、condition、repeat_over 控制 evidence surface。
    触发点：agent 需要分层、带标签、带时间的 readout。
    落地：`TemplateReadout(prompt=structured_prompt(...))`。

45. 增加 note payload readout
    类型：增加型
    含义：按 note schema 渲染 content/context/tags。
    触发点：agent 已经使用 note-style representation。
    落地：`NoteRenderReadout(note_namespace=..., include_context=True, include_tags=True)`。

46. 增加 graph relation readout
    类型：增加型
    含义：把 graph triples 渲染成关系句。
    触发点：agent 需要结构化关系证据进入 answer prompt。
    落地：`GraphRelationReadout()`。

47. 增加 prompt context readout
    类型：增加型
    含义：把 reflection / last trial / memory records 组装成下一步 prompt context。
    触发点：agent 做 Reflexion-style trial loop。
    落地：`PromptContextReadout(default_strategy=..., memory_layer=...)`。

48. 增加 grouped evidence readout shape
    类型：增加型
    含义：让 readout 暴露 group、summary、sources、roles，而不是 flat text。
    触发点：agent 使用 hierarchical summary 或 cluster expansion。
    落地：`TemplateReadout`、`views["summary_with_sources"]`、retrieval trace roles。

49. 增加 generic memory binding 边界
    类型：增加型
    含义：给外部 benchmark/agent 暴露统一 build/ingest/recall 接口。
    触发点：agent 想让不同 memory pipeline 直接接评测脚本。
    落地：`create_memory_binding()`、`build_system`、`ingest_event`、`recall`。

50. 增加 dedicated rerank runtime 配置
    类型：增加型
    含义：把 rerank 服务作为独立 runtime 能力接入。
    触发点：agent 需要真实 rerank，不想走 chat prompt 模拟。
    落地：`Runtime.rerank()`、`MEMPRIMITIVE_RERANK_BASE_URL`、`MEMPRIMITIVE_RERANK_MODEL`、`/rerank`。
