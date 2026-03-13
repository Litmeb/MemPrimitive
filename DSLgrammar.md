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

Store 拓扑是 memory system 的骨架结构。这里不再把拓扑写成少数几个命名模板，而是显式建模为一个**多层结构**：每层单独声明自己的语义主题、存储形态和索引。

换句话说，hierarchical 在这里不是一种单独的 store type，而是“存在多层”的事实本身。

```ebnf
StoreBlock      ::= 'stores' '{' LayerDecl+ '}'

LayerDecl       ::= IDENT ':' 'Layer' '(' LayerArgs ')' IndexList?

LayerArgs       ::= ThemeArg ',' ShapeArg (',' ConfigArg)*

ThemeArg        ::= 'theme' '=' ThemeValue
ShapeArg        ::= 'shape' '=' ShapeValue

ThemeValue      ::= 'working'
                   | 'episodic'
                   | 'semantic'
                   | 'profile'
                   | 'skill'
                   | 'reflection'
                   | STRING              (* 自定义主题 *)

ShapeValue      ::= 'Flat'
                   | 'Graph'

Param           ::= IDENT '=' Value

IndexList       ::= '[' IndexType (',' IndexType)* ']'

IndexType       ::= 'vector'           (* 向量索引 *)
                   | 'entity'           (* 实体索引 *)
                   | 'temporal'         (* 时间索引 *)
                   | 'keyword'          (* 关键词/BM25 索引 *)
                   | 'graph'            (* 图邻接索引 *)
                   | 'tag'              (* 标签索引 *)
```

最小示例：

```text
stores {
    working_buffer : Layer(theme=working, shape=Flat, capacity=token_limited, token_budget=2000) [temporal]
    episodes       : Layer(theme=episodic, shape=Flat, capacity=unlimited) [vector, temporal]
    knowledge      : Layer(theme=semantic, shape=Graph, capacity=unlimited) [entity, graph, vector]
}
```

这使得搜索空间中的结构层被拆成三个更正交的维度：

- 层数
- 每层的 `shape`
- 每层的 `theme`

跨层流动策略，如写入路由、摘要迁移、提升/降级和 selective read，不放在 `StoreBlock` 中，而由 `organization`、`compression`、`retrieval`、`maintenance` 等模块负责。

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

### 2.5.1 Unit Formation 的规范写法（建议）

虽然 `ModuleExpr` 允许引用任意实现名，但对 `unit_formation`，推荐优先收敛到四类规范操作：`Segment`、`Extract`、`Abstract`、`Delegate`。

```text
Segment(
  mode="observation" | "message" | "turn" | "chunk",
  ...
)

Extract(
  target="event" | "fact" | "entity_state" | "triple" | "kv" | "skill" | "thought" | custom("..."),
  method="rule_based" | "schema_guided_llm" | "constrained_decoding" | "span_labeling"
        | "relation_extraction" | "ontology_guided" | "retrieval_assisted" | "freeform_llm",
  schema=...,
  granularity="message" | "turn" | "window",
  source_scope="observation" | "trajectory" | "session",
  ...
)

Abstract(
  target="summary" | "reflection" | custom("..."),
  method="llm_summary" | "extractive" | "abstractive" | "trajectory_reflection",
  source_scope="window" | "session" | "trajectory",
  ...
)

Delegate(
  model=...,
  instruction=...
)
```

推荐约束：

- `target` 优先使用受控词表，只在确有必要时使用 `custom("...")`
- `method` 应显式声明，因为它比“抽取什么”更直接决定行为差异
- 多粒度形成优先用 `Compose` 表达，而不是再单独引入一个 `MultiGranularity` primitive

示例：

```text
unit_formation : Segment(mode="turn", include_system=false)
```

```text
unit_formation : Extract(
    target="fact",
    method="schema_guided_llm",
    schema=["who", "what", "when", "where"],
    granularity="turn",
    source_scope="observation"
)
```

```text
unit_formation : Abstract(
    target="reflection",
    method="trajectory_reflection",
    source_scope="trajectory",
    include_outcome=true
)
```

```text
unit_formation : Compose(
    [
      Segment(mode="turn"),
      Extract(target="fact", method="schema_guided_llm", granularity="turn"),
      Abstract(target="summary", method="abstractive", source_scope="session")
    ],
    merge=Concat
)
```

### 2.5.2 Representation 的规范写法（建议）

对 `representation`，推荐不再使用一批命名模板，而是直接声明一个**表示元素集合**。  
表示编码的任务不是选 `A`、`B`、`A+B` 这类实现名，而是决定 unit 最终携带哪些 representation elements。

```text
Represent(
  elements={"text", "embedding", ...},
  ...
)
```

推荐元素词表：

```text
"text"
"embedding"
"triple"
"kv"
"frame"
"entities"
"tags"
"code"
"description"
"sparse_embedding"
```

推荐约束：

- 向量检索要求 `elements` 中包含 `"embedding"`
- BM25 / sparse retrieval 要求 `elements` 中包含 `"text"` 或 `"sparse_embedding"`
- entity-aware organization / update 通常要求 `"entities"`
- `code` 通常与 `description` 或 `embedding` 联合出现

示例：

```text
representation : Represent(elements={"text"})
```

```text
representation : Represent(
    elements={"text", "embedding"},
    embedding_model="text-embedding-3-large"
)
```

```text
representation : Represent(
    elements={"text", "embedding", "entities"},
    embedding_model="text-embedding-3-large",
    entity_linker="wikidata_linker"
)
```

```text
representation : Represent(
    elements={"code", "description", "embedding"},
    description_model="skill_describer",
    embedding_model="text-embedding-3-large"
)
```

### 2.5.3 Write Trigger 的规范写法（建议）

对 `write_trigger`，推荐不再使用很多命名 trigger，而是统一写成：

```text
WriteTrigger(
  signals={...},
  scorer=...,
  gates={...},
  policy=...,
  ...
)
```

推荐 signal 词表：

```text
"importance"
"novelty"
"surprise"
"duplicate_risk"
"task_relevance"
"user_relevance"
"entity_salience"
"event_match"
"llm_value_estimate"
```

推荐 scorer：

```text
identity(source=...)
weighted_sum(weights=[...])
max(sources=[...])
product(sources=[...])
rule_expr(expression=...)
llm_judge(model=..., prompt=..., criteria=[...])
```

推荐 gate：

```text
on_event("task_failure")
not_duplicate(threshold=0.9)
tool_called("memory_write")
predicate(...)
```

推荐 policy：

```text
always
never
threshold(value=0.7)
top_k_per_window(k=..., window=...)
sample_by_score(temperature=...)
boolean_gate(logic="all" | "any")
explicit_only(tool_names=[...])
```

示例：

```text
write_trigger : WriteTrigger(
    policy=always
)
```

```text
write_trigger : WriteTrigger(
    signals={"importance", "novelty"},
    scorer=weighted_sum(weights=[0.7, 0.3]),
    policy=threshold(value=0.6)
)
```

```text
write_trigger : WriteTrigger(
    gates={on_event("task_failure"), not_duplicate(threshold=0.95)},
    policy=boolean_gate(logic="all")
)
```

```text
write_trigger : WriteTrigger(
    signals={"llm_value_estimate"},
    scorer=llm_judge(
        model="gpt-4.1",
        criteria=["long_term_value", "task_relevance"]
    ),
    policy=threshold(value=0.6)
)
```

```text
write_trigger : WriteTrigger(
    gates={tool_called("memory_write")},
    policy=explicit_only(tool_names=["memory_write"])
)
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
    store_requirements  : list[IndexType]    // 要求某层具备哪些索引
    layer_requirements  : list[string]       // 要求某层具备哪些 theme/shape 特征
    side_effects        : list[string]       // 对 store 的修改类型
    params              : list[ParamSpec]    // 可配参数
}
```

这使得搜索器可以在组合前验证兼容性：模块 B 的 `input_requirements` 必须被上游模块的 `output_guarantees` 覆盖，模块的 `store_requirements` 必须被 `StoreBlock` 中某个 layer 的 `shape` 与 `IndexList` 覆盖。

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

## 5. 语法完备性检查

| 需求 | 是否覆盖 | 说明 |
|------|----------|------|
| 单一实现选择 | ✓ | `ImplRef` |
| 多策略融合 | ✓ | `Compose` + `MergeStrategy` |
| 分阶段处理 | ✓ | `Cascade` |
| 条件分发 | ✓ | `Dispatch` |
| 不启用某 slot | ✓ | `None` |
| 触发条件 | ✓ | `TriggerBlock` |
| 兼容性约束 | ✓ | `ConstraintBlock` |
| 多层路由 | ✓ | `Organization` 的 target_store / target_layer 参数 |
| Agent 显式触发 | ✓ | `tool_called(...)` + `explicit_only(...)` |
| 参数化搜索 | ✓ | `ConfigArgs` 支持任意参数 |
| 嵌套组合 | ✓ | `ModuleExpr` 可递归嵌套 |
