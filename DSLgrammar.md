# MemDSL: Agent Memory 系统的组合式描述语言

## 1. 设计目标

本语言是一个**声明式、可组合、可搜索**的 memory system 描述框架，服务于四个核心目标：

| 目标 | 含义 |
|------|------|
| **重表达** | 用统一语法描述已有 agent memory 方法 |
| **可组合** | 每个 primitive slot 可独立替换实现 |
| **可搜索** | 配置空间可枚举、可采样、可约束剪枝 |
| **可归纳** | 从搜索结果中提取 recurring motifs |

### 1.1 语言层次

```
Level 3: System       — 完整的 memory system 声明（store 拓扑 + 所有 pipeline）
Level 2: Pipeline     — write / read / compress / maintain 四条数据通路
Level 1: Module       — 每个 primitive slot 的实现选择 + 组合算子
Level 0: Param        — 每个实现的离散/连续参数
```

### 1.2 核心抽象

整个系统可视为一个带副作用的数据流图：

```
                ┌──────────── Write Path ────────────┐
                │                                     │
Observation ──► UnitFormation ──► Representation ──► WriteTrigger ──► Organization ──► Update ──► Store
                                                                                         │
                ┌──────────── Read Path ─────────────┐                                   │
                │                                     │                                   │
Query ────────► Retrieval ──► Readout ──► Agent       │                                   │
                   ▲                                  │                                   │
                   └──────────────────────────────────┘                                   │
                                                                                         │
                ┌──────── Background Path ───────────┐                                   │
                │                                     │                                   │
                │   Compression ◄── Trigger           │◄──────────────────────────────────┘
                │   Maintenance ◄── Trigger           │
                └─────────────────────────────────────┘
```

---

## 2. 形式文法 (EBNF)

### 2.1 词法约定

```ebnf
IDENT       ::= [a-zA-Z_][a-zA-Z0-9_]*
INT         ::= [0-9]+
FLOAT       ::= [0-9]+ '.' [0-9]+
STRING      ::= '"' [^"]* '"'
BOOL        ::= 'true' | 'false'
COMMENT     ::= '//' [^\n]* | '/*' .* '*/'
```

### 2.2 顶层结构

```ebnf
Program         ::= SystemDecl+

SystemDecl      ::= 'system' IDENT '{' 
                       StoreBlock 
                       ModuleBlock 
                       TriggerBlock?
                       ConstraintBlock?
                     '}'
```

### 2.3 Store 声明

Store 拓扑是 memory system 的骨架结构。不同的 store 划分方式本身就是搜索维度之一。

```ebnf
StoreBlock      ::= 'stores' '{' StoreDecl+ '}'

StoreDecl       ::= IDENT ':' StoreType '(' StoreParams ')' IndexList?

StoreType       ::= 'Flat'             (* 线性列表 *)
                   | 'Indexed'          (* 支持多种索引 *)
                   | 'Graph'            (* 图结构 *)
                   | 'Hierarchical'     (* 层级结构 *)

StoreParams     ::= (Param (',' Param)*)?

Param           ::= IDENT '=' Value

IndexList       ::= '[' IndexType (',' IndexType)* ']'

IndexType       ::= 'vector'           (* 向量索引 *)
                   | 'entity'           (* 实体索引 *)
                   | 'temporal'         (* 时间索引 *)
                   | 'keyword'          (* 关键词/BM25 索引 *)
                   | 'graph'            (* 图邻接索引 *)
                   | 'tag'              (* 标签索引 *)
```

### 2.4 Module 声明

这是 DSL 的核心。每个 primitive slot 绑定一个 ModuleExpr。

```ebnf
ModuleBlock     ::= 'modules' '{' ModuleDecl+ '}'

ModuleDecl      ::= PrimitiveSlot ':' ModuleExpr

PrimitiveSlot   ::= 'unit_formation'   (* A: 记忆单元形成 *)
                   | 'representation'   (* B: 表示编码 *)
                   | 'write_trigger'    (* C: 写入决策 *)
                   | 'organization'     (* D: 关系/放置规划 *)
                   | 'update'           (* E: 写入执行 *)
                   | 'compression'      (* F: 抽象压缩 *)
                   | 'retrieval'        (* G: 检索策略 *)
                   | 'readout'          (* H: 输出格式化 *)
                   | 'maintenance'      (* I: 维护管理 *)
```

### 2.5 Module 表达式

支持四种组合算子，使得实现可以从简单到复杂任意嵌套。

```ebnf
ModuleExpr      ::= 'None'             (* 该 slot 不启用 *)
                   | ImplRef            (* 单一实现 *)
                   | ComposeExpr        (* 并行组合 *)
                   | CascadeExpr        (* 串行级联 *)
                   | DispatchExpr       (* 条件分发 *)

(* 单一实现引用 *)
ImplRef         ::= IDENT '(' ConfigArgs ')'

(* 并行组合：多个模块的结果按策略合并 *)
ComposeExpr     ::= 'Compose' '(' 
                       '[' ModuleExpr (',' ModuleExpr)* ']' ',' 
                       'merge' '=' MergeStrategy 
                     ')'

(* 串行级联：前一模块的输出作为后一模块的候选输入 *)
CascadeExpr     ::= 'Cascade' '(' ModuleExpr ',' ModuleExpr (',' ModuleExpr)* ')'

(* 条件分发：根据运行时条件选择不同实现 *)
DispatchExpr    ::= 'Dispatch' '(' 
                       'on' '=' FieldPath ',' 
                       DispatchCase (',' DispatchCase)* 
                       (',' 'default' '=' ModuleExpr)? 
                     ')'

DispatchCase    ::= STRING ':' ModuleExpr

MergeStrategy   ::= 'WeightedSum' '(' '[' FLOAT (',' FLOAT)* ']' ')'
                   | 'RRF' '(' ConfigArgs ')'        (* Reciprocal Rank Fusion *)
                   | 'Union'
                   | 'Intersection'
                   | 'Concat'
                   | IDENT '(' ConfigArgs ')'          (* 自定义合并 *)

FieldPath       ::= IDENT ('.' IDENT)*
```

### 2.6 Trigger 声明

控制 compression 和 maintenance 的触发时机。

```ebnf
TriggerBlock    ::= 'triggers' '{' TriggerDecl+ '}'

TriggerDecl     ::= TriggerSlot ':' TriggerExpr

TriggerSlot     ::= 'compress_trigger' | 'maintain_trigger'

TriggerExpr     ::= 'Periodic' '(' 'every' '=' INT ')'
                   | 'OnEvent' '(' EventType ')'
                   | 'BudgetExceeded' '(' 'threshold' '=' FLOAT ')'
                   | 'Conditional' '(' Predicate ')'
                   | 'AfterWrite'
                   | 'And' '(' TriggerExpr ',' TriggerExpr ')'
                   | 'Or' '(' TriggerExpr ',' TriggerExpr ')'
                   | 'Never'

EventType       ::= 'episode_end' | 'session_end' | 'task_complete' 
                   | 'task_failure' | 'context_overflow' | STRING

Predicate       ::= FieldPath COMP_OP Value
                   | Predicate 'and' Predicate
                   | Predicate 'or' Predicate

COMP_OP         ::= '>' | '<' | '>=' | '<=' | '==' | '!='
```

### 2.7 约束声明

显式声明模块间的兼容性约束，用于搜索时剪枝。

```ebnf
ConstraintBlock   ::= 'constraints' '{' ConstraintDecl+ '}'

ConstraintDecl    ::= RequiresDecl | IncompatibleDecl | CoRequiresDecl

(* A 的选择要求 store/其他模块满足某条件 *)
RequiresDecl      ::= 'requires' '(' PrimitiveSlot '.' IDENT ',' Predicate ')'

(* A 与 B 不能同时选 *)
IncompatibleDecl  ::= 'incompatible' '(' 
                         PrimitiveSlot '.' IDENT ',' 
                         PrimitiveSlot '.' IDENT 
                       ')'

(* A 必须与 B 同时选 *)
CoRequiresDecl    ::= 'co_requires' '(' 
                         PrimitiveSlot '.' IDENT ',' 
                         PrimitiveSlot '.' IDENT 
                       ')'
```

### 2.8 参数与值

```ebnf
ConfigArgs      ::= (ConfigArg (',' ConfigArg)*)?

ConfigArg       ::= IDENT '=' Value

Value           ::= INT | FLOAT | STRING | BOOL | 'None'
                   | '[' (Value (',' Value)*)? ']'        (* list *)
                   | '{' (ConfigArg (',' ConfigArg)*)? '}' (* dict *)
                   | FieldPath                              (* 引用 *)
                   | ModuleExpr                             (* 嵌套模块 *)
```

---

## 3. 统一模块签名与类型系统

### 3.1 所有模块共享的签名

每个 ModuleExpr 在运行时都遵守统一签名：

```
(Packet, MemoryStore, AgentContext, Config) → (Packet, MemoryStore, Aux)
```

其中 `Packet` 是通用数据载体，不同阶段填充不同字段：

```
Packet = {
    observation  : Observation?       // A 阶段读
    units        : list[MemoryUnit]?  // A 输出, B/C/D/E 读写
    decisions    : list[bool]?        // C 输出, filter 读
    placement    : list[PlacePlan]?   // D 输出, E 读
    query        : Query?             // G 阶段读
    retrieved    : RetrievedContext?   // G 输出, H 读
    readout      : ReadoutPacket?     // H 输出
}
```

### 3.2 模块的能力声明 (ModuleSpec)

每个具体实现必须声明自己的 IO 约束：

```
ModuleSpec = {
    name                : string
    slot                : PrimitiveSlot
    input_requirements  : list[FieldPath]    // 需要 Packet/Store 中哪些字段
    output_guarantees   : list[FieldPath]    // 保证产出哪些字段
    store_requirements  : list[IndexType]    // 要求 store 有哪些索引
    side_effects        : list[string]       // 对 store 的修改类型
    params              : list[ParamSpec]    // 可配参数
}
```

这使得搜索器可以在组合前验证兼容性：模块 B 的 `input_requirements` 必须被上游模块的 `output_guarantees` 覆盖，模块的 `store_requirements` 必须被 StoreBlock 中声明的 IndexList 覆盖。

---

## 4. 组合算子的语义

### 4.1 Compose — 并行组合

```
Compose([M1, M2, ...], merge=strategy)
```

所有子模块接收相同输入，各自独立执行，结果按 `merge` 策略合并。典型用于 retrieval（多信号融合）和 representation（多种编码并行）。

### 4.2 Cascade — 串行级联

```
Cascade(M1, M2, ...)
```

M1 的输出作为 M2 的输入。典型用于 retrieval 的 recall-then-rerank 模式。

### 4.3 Dispatch — 条件分发

```
Dispatch(on=field.path, "case1": M1, "case2": M2, default=M3)
```

根据运行时字段值选择不同实现。典型用于不同 observation type 使用不同 unit_formation。

### 4.4 None — 跳过

```
compression: None
```

该 slot 不执行任何操作，Packet 和 Store 透传。

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

---

## 7. 语法完备性检查

| 需求 | 是否覆盖 | 说明 |
|------|----------|------|
| 单一实现选择 | ✓ | `ImplRef` |
| 多策略融合 | ✓ | `Compose` + `MergeStrategy` |
| 分阶段处理 | ✓ | `Cascade` |
| 条件分发 | ✓ | `Dispatch` |
| 不启用某 slot | ✓ | `None` |
| 触发条件 | ✓ | `TriggerBlock` |
| 兼容性约束 | ✓ | `ConstraintBlock` |
| 多 store 路由 | ✓ | `Organization` 的 target_store 参数 |
| Agent 自主决策 | ✓ | `AgentToolCall` / `AgentDecision` 实现 |
| 参数化搜索 | ✓ | `ConfigArgs` 支持任意参数 |
| 嵌套组合 | ✓ | `ModuleExpr` 可递归嵌套 |
