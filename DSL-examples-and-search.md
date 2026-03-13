# MemDSL：经典方法重表达与搜索支持

本文档包含 DSL 的经典方法重表达示例（第 5 节）与对搜索的支持分析（第 6 节），自 `DSLgrammar.md` 拆出。

---

## 5. 经典方法的 DSL 重表达

以下用本语言重新描述 8 个经典 agent memory 系统，以验证语言的表达力。

---

### 5.1 Generative Agents (Park et al., 2023)

```
system GenerativeAgents {
    stores {
        observation_stream : Flat(capacity=unlimited) [temporal]
        reflections        : Flat(capacity=unlimited) [temporal]
    }

    modules {
        unit_formation  : PassThrough()
        representation  : TextWithEmbedding(model="text-embedding-ada-002")
        write_trigger   : Always()
        organization    : TemporalAppend()
        update          : Append()
        compression     : LLMReflection(
                            prompt_template="higher_level_insights",
                            source_stores=["observation_stream"],
                            target_store="reflections",
                            batch_size=100
                          )
        retrieval       : Compose(
                            [Similarity(top_k=50), 
                             RecencyDecay(decay=0.995), 
                             ImportanceScore()],
                            merge=WeightedSum([0.5, 0.3, 0.2])
                          )
        readout         : FlatConcat(max_tokens=2000, format="text")
        maintenance     : None
    }

    triggers {
        compress_trigger : Periodic(every=100)
        maintain_trigger : Never
    }
}
```

**关键 motif**：全量写入 + 三信号融合检索 + 周期性反思。

---

### 5.2 MemGPT (Packer et al., 2023)

```
system MemGPT {
    stores {
        main_context : Flat(capacity=token_limited, token_budget=4000) []
        archival     : Indexed(capacity=unlimited) [vector]
        recall       : Flat(capacity=unlimited) [temporal]
    }

    modules {
        unit_formation  : TurnLevel()
        representation  : TextWithEmbedding()
        write_trigger   : AgentToolCall(
                            tool_names=["core_memory_save", "archival_memory_insert"]
                          )
        organization    : AgentExplicitTarget(
                            tool_to_store={
                              "core_memory_save": "main_context",
                              "archival_memory_insert": "archival"
                            }
                          )
        update          : Append()
        compression     : RecursiveSummarization(
                            source_store="main_context",
                            target_store="recall",
                            strategy="incremental"
                          )
        retrieval       : AgentToolCall(
                            tool_names=["archival_memory_search", "conversation_search"],
                            backend=Similarity(top_k=10)
                          )
        readout         : StructuredSections(
                            sections=["system_prompt", "core_memory", "recall_window", "archival_results"]
                          )
        maintenance     : ContextWindowManager(
                            strategy="summarize_and_evict",
                            budget_field="main_context.token_budget"
                          )
    }

    triggers {
        compress_trigger : OnEvent("context_overflow")
        maintain_trigger : OnEvent("context_overflow")
    }
}
```

**关键 motif**：agent 自主决策读写 + 显式多 store 路由 + context window 管理。

---

### 5.3 Reflexion (Shinn et al., 2023)

```
system Reflexion {
    stores {
        reflections : Flat(capacity=sliding_window, window_size=3) []
    }

    modules {
        unit_formation  : TaskReflectionExtractor(
                            input="trajectory + outcome",
                            output="natural_language_reflection"
                          )
        representation  : RawText()
        write_trigger   : OnEvent("task_failure")
        organization    : FlatAppend()
        update          : Append()
        compression     : None
        retrieval       : FullContext()
        readout         : FlatConcat(
                            prefix="Previous trial reflections:\n"
                          )
        maintenance     : SlidingWindow(window_size=3)
    }

    triggers {
        compress_trigger : Never
        maintain_trigger : AfterWrite
    }
}
```

**关键 motif**：极简结构——任务失败时写反思、全量注入、滑窗维护。

---

### 5.4 Voyager (Wang et al., 2023)

```
system Voyager {
    stores {
        skill_library : Indexed(capacity=unlimited) [vector, tag]
    }

    modules {
        unit_formation  : SkillExtractor(
                            fields=["code", "description", "dependencies"]
                          )
        representation  : CodeWithDescription(
                            embed_field="description",
                            model="text-embedding-ada-002"
                          )
        write_trigger   : OnEvent("task_success")
        organization    : TagBasedPlacement(tag_field="skill_type")
        update          : UpsertByKey(key="description_similarity", threshold=0.95)
        compression     : None
        retrieval       : Similarity(
                            query_source="task_description",
                            top_k=5
                          )
        readout         : CodeInjection(
                            format="available_functions",
                            include_docstring=true
                          )
        maintenance     : None
    }

    triggers {
        compress_trigger : Never
        maintain_trigger : Never
    }
}
```

**关键 motif**：技能即记忆——代码作为 unit、成功时写入、相似度检索。

---

### 5.5 MemoryBank (Zhong et al., 2024)

```
system MemoryBank {
    stores {
        short_term : Flat(capacity=token_limited, token_budget=2000) [temporal]
        long_term  : Indexed(capacity=unlimited) [vector, temporal]
    }

    modules {
        unit_formation  : Compose(
                            [TurnLevel(), FactExtractor()],
                            merge=Concat
                          )
        representation  : TextWithEmbedding()
        write_trigger   : Always()
        organization    : DualStoreRouter(
                            route_rules={
                              "turn": "short_term",
                              "fact": "long_term"
                            }
                          )
        update          : Dispatch(
                            on=unit.unit_type,
                            "turn": Append(),
                            "fact": MergeByEntity()
                          )
        compression     : Summarization(
                            source_store="short_term",
                            target_store="long_term",
                            method="llm_summary"
                          )
        retrieval       : Compose(
                            [Similarity(top_k=20), RecencyDecay(decay=0.99)],
                            merge=WeightedSum([0.6, 0.4])
                          )
        readout         : FlatConcat(format="text")
        maintenance     : EbbinghausForgetting(
                            decay_model="exponential",
                            base_retention=0.8,
                            rehearsal_boost=0.3
                          )
    }

    triggers {
        compress_trigger : OnEvent("session_end")
        maintain_trigger : Periodic(every=50)
    }
}
```

**关键 motif**：双 store 路由 + 遗忘曲线维护 + 会话结束时压缩。

---

### 5.6 SCM — Self-Controlled Memory (Wang et al., 2024)

```
system SCM {
    stores {
        short_term : Flat(capacity=token_limited, token_budget=2000) [temporal]
        long_term  : Indexed(capacity=unlimited) [entity, vector]
    }

    modules {
        unit_formation  : LLMStructuredExtractor(
                            output_schema=["entity", "attribute", "value", "source"]
                          )
        representation  : StructuredTriple(embed=true)
        write_trigger   : LLMJudge(
                            criteria=["relevance_to_task", "factual_content"],
                            threshold=0.6
                          )
        organization    : EntityLinked(link_type="entity_attribute")
        update          : UpsertByEntity(
                            conflict_resolution="llm_judge"
                          )
        compression     : EntityProfileUpdate(
                            method="attribute_aggregation"
                          )
        retrieval       : LLMControlled(
                            strategy="self_reflective_retrieval",
                            fallback=EntityLookup(max_results=10)
                          )
        readout         : StructuredSections(
                            sections=["user_profile", "relevant_facts", "conversation_context"]
                          )
        maintenance     : ConflictCleanup(method="recency_wins")
    }

    triggers {
        compress_trigger : Periodic(every=20)
        maintain_trigger : AfterWrite
    }
}
```

**关键 motif**：结构化事实抽取 + LLM 控制读写 + entity 级 upsert。

---

### 5.7 A-MEM — Agentic Memory (Xu et al., 2025)

```
system AMEM {
    stores {
        memory_graph : Graph(capacity=unlimited) [vector, entity, graph]
    }

    modules {
        unit_formation  : LLMStructuredExtractor(
                            output_schema=["key", "content", "tags", "connections"]
                          )
        representation  : HybridStructured(
                            embed=true,
                            entity_link=true,
                            tag_index=true
                          )
        write_trigger   : Always()
        organization    : GraphPlacement(
                            connection_method="llm_inferred",
                            edge_types=["related_to", "derived_from", "contradicts"]
                          )
        update          : AppendWithLinkUpdate()
        compression     : None
        retrieval       : Compose(
                            [Similarity(top_k=10),
                             GraphHop(max_hops=2, top_per_hop=5)],
                            merge=Union
                          )
        readout         : StructuredSections(
                            sections=["direct_matches", "connected_memories"]
                          )
        maintenance     : None
    }

    triggers {
        compress_trigger : Never
        maintain_trigger : Never
    }
}
```

**关键 motif**：图结构 + LLM 推断连接 + 多跳图检索。

---

### 5.8 TiM — Think-in-Memory (Liu et al., 2023)

```
system TiM {
    stores {
        thought_memory : Flat(capacity=token_limited, token_budget=4000) [temporal, vector]
    }

    modules {
        unit_formation  : ThoughtExtractor(
                            source="internal_reasoning_trace",
                            granularity="reasoning_step"
                          )
        representation  : TextWithEmbedding()
        write_trigger   : AfterReasoningStep()
        organization    : TemporalAppend()
        update          : Append()
        compression     : Summarization(
                            method="progressive",
                            source_store="thought_memory",
                            target_store="thought_memory"
                          )
        retrieval       : Compose(
                            [Recency(window=10), Similarity(top_k=5)],
                            merge=WeightedSum([0.6, 0.4])
                          )
        readout         : PrependToContext(format="thought_prefix")
        maintenance     : SlidingWindow(window_size=50)
    }

    triggers {
        compress_trigger : BudgetExceeded(threshold=0.8)
        maintain_trigger : BudgetExceeded(threshold=0.9)
    }
}
```

**关键 motif**：推理即记忆——思考痕迹作为 unit、预算驱动压缩。

---

## 6. 对搜索的支持分析

### 6.1 搜索空间的三个维度

本 DSL 的搜索空间是一个**三层笛卡尔积空间加约束剪枝**：

| 层 | 维度 | 说明 |
|----|------|------|
| **Structural** | Store 拓扑 | 几个 store、什么类型、哪些 index |
| **Modular** | 各 Slot 的实现选择 | 9 个 slot × 各自的候选实现集 |
| **Parametric** | 每个实现的超参 | 离散参数用枚举，连续参数用采样 |

一次搜索配置 = `(StoreTopo, UF, Rep, WT, Org, Upd, Comp, Ret, Read, Maint, Triggers)`

### 6.2 约束剪枝

搜索前根据 `ModuleSpec` 中的 `input_requirements`、`store_requirements` 等字段自动过滤不合法组合。约束来源：

1. **Store-Module 约束**：`Similarity` retrieval 要求 store 有 `vector` index
2. **Module-Module 约束**：`GraphHop` retrieval 要求 `organization` 产出 graph links
3. **Type 链约束**：下游模块的 `input_requirements` ⊆ 上游模块的 `output_guarantees`

### 6.3 配置向量化

为了让搜索算法（evolutionary / Bayesian / random）能操作，每个配置可编码为：

```
config_vector = [
    store_topo_id,                         // categorical
    unit_formation_id, uf_param_1, ...,    // categorical + numeric
    representation_id, rep_param_1, ...,
    write_trigger_id, wt_param_1, ...,
    organization_id, org_param_1, ...,
    update_id, upd_param_1, ...,
    compression_id, comp_param_1, ...,
    retrieval_id, ret_param_1, ...,
    readout_id, read_param_1, ...,
    maintenance_id, maint_param_1, ...,
    compress_trigger_id,
    maintain_trigger_id,
]
```

其中 categorical 变量用 one-hot 或整数编码，numeric 变量直接取值。

### 6.4 Motif 归纳

搜索产出的 top-K 配置可通过以下方式提取 motif：

1. **频率分析**：统计各 slot 中哪些实现出现频率最高
2. **共现分析**：统计哪些实现组合显著共现（χ² test / mutual information）
3. **消融对比**：固定其他 slot，切换单个 slot 实现，量化影响
4. **聚类分析**：对 config_vector 聚类，识别 configuration archetype
