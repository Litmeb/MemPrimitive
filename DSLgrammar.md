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
Observation ──► UnitFormation ──► Representation ──► WriteTrigger ──► Organization ──► Store
                                                                            │
                                                                            ▼
                                                                   EvolutionTrigger
                                                                            │
                                                                            ▼
                                                                   MemoryEvolution
                                                                                         │
                ┌──────────── Read Path ─────────────┐                                   │
                │                                     │                                   │
Query ────────► Retrieval ──► Readout ──► Agent       │                                   │
                   ▲                                  │                                   │
                   └──────────────────────────────────┘                                   │
                                                                                         │
                ┌──────── Background Path ───────────┐                                   │
                │                                     │                                   │
                │   MemoryEvolution ◄── Trigger       │◄──────────────────────────────────┘
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

跨层流动策略，如写入路由、摘要迁移、提升/降级和 selective read，不放在 `StoreBlock` 中，而由 `organization`、`memory_evolution`、`retrieval` 等模块负责。
其中新的语义是：

- `organization` 负责 ingest-time 的组织与常规写入
- `memory_evolution` 负责额外触发的演化，而不是默认常规落库

### 2.4 Module 声明

这是 DSL 的核心。每个 primitive slot 绑定一个 ModuleExpr。

```ebnf
ModuleBlock     ::= 'modules' '{' ModuleDecl+ '}'

ModuleDecl      ::= PrimitiveSlot ':' ModuleExpr

PrimitiveSlot   ::= 'unit_formation'   (* A: 记忆单元形成 *)
                   | 'representation'   (* B: 表示编码 *)
                   | 'write_trigger'    (* C: 写入决策 *)
                   | 'organization'     (* D: 组织 + 常规写入 *)
                   | 'memory_evolution' (* E: 额外触发的记忆演化 *)
                   | 'retrieval'        (* G: 检索策略 *)
                   | 'readout'          (* H: 输出格式化 *)
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
- entity-aware organization / memory evolution 通常要求 `"entities"`
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

### 2.5.4 Organization 的规范写法（建议）

对 `organization`，推荐统一写成：

```text
Organization(
  routing=...,
  links={...},
  placement=...,
  ...
)
```

这里的 contract 需要特别强调：

```text
Organization handles ingest-time organization and normal write.
It is not only a placement-plan generator.
```

并且采用如下原则：

```text
Organization is topology-constrained.
StoreTopology defines the admissible routing targets, link types, and placement modes.
```

推荐 routing：

```text
default(target_layer=...)
by_unit_type(route_map={...})
by_tag(tag_field=..., route_map={...})
by_rule(rules=[...])
explicit(field=...)
agent_selected(tool_to_layer={...})
```

推荐 links：

```text
temporal(...)
entity(...)
similarity(...)
cluster_membership(...)
graph_edge(edge_types=[...], connection_method=...)
parent_child(...)
```

推荐 placement：

```text
append()
partition(partition_key=...)
cluster(similarity_threshold=...)
graph_node(node_policy=...)
hierarchical_slot(target_level=..., placement_policy=...)
```

推荐约束：

- `placement=graph_node(...)` 要求存在 `shape=Graph` 的 layer
- `links` 中包含 `graph_edge(...)` 时，通常要求目标 layer 有 `graph` index
- `placement=hierarchical_slot(...)` 或 `links={parent_child(...)}` 要求 `layer_count > 1`
- `routing=by_tag(...)` 要求 unit/representation 中可读出 `tags`
- `links={entity(...)}` 要求 unit/representation 中可读出 `entities`

补充语义约束：

- `organization` 默认完成常规写入，因此 `placement=append()` 不再意味着“只产出计划”，而意味着 append-style normal write
- 某些 `organization` 实现可以包含与 routing / placement 强耦合的 ingest-time merge / upsert / replace
- 这些 ingest-time update 不单独拆成新 slot，因为它们不构成独立搜索轴

示例：

```text
organization : Organization(
    routing=default(target_layer="episodic"),
    links={temporal()},
    placement=append()
)
```

```text
organization : Organization(
    routing=by_unit_type(route_map={"turn": "working", "fact": "semantic"}),
    links={},
    placement=append()
)
```

```text
organization : Organization(
    routing=default(target_layer="graph_memory"),
    links={graph_edge(edge_types=["related_to", "derived_from"], connection_method="llm_inferred")},
    placement=graph_node(node_policy="append")
)
```

```text
organization : Organization(
    routing=by_tag(tag_field="skill_type", route_map={"craft": "craft_skills", "nav": "nav_skills"}),
    links={},
    placement=partition(partition_key="skill_type")
)
```

### 2.5.5 Memory Evolution 的规范写法（建议）

对 `memory_evolution`，推荐统一写成：

```text
MemoryEvolution(
  selection=...,
  action=...,
  effect=...,
  trigger=...,
  ...
)
```

新的定义是：

```text
MemoryEvolution handles optional extra evolution over already-organized memory.
It is off by default and should not be used as the default normal write path.
```

推荐 selection：

```text
matched_by_key(key_field=...)
matched_by_entity(entity_field=...)
time_window(window_size=...)
layer_slice(target_layer=..., filter=...)
low_activity(activity_threshold=...)
all(scope=...)
```

推荐 action：

```text
replace()
merge()
upsert()
rewrite()
summarize()
reflect()
profile_update()
extract_concept()
prototype_form()
distill()
prune()
dedup()
move(target_layer=...)
consolidate()
review()
delta()
```

推荐 effect：

```text
add
modify
delete
move
merge
summarize
version
```

推荐 trigger：

```text
periodic(every=...)
on_event("session_end")
budget_exceeded(threshold=...)
count_exceeded(max_count=...)
conditional(...)
```

推荐约束：

- `selection=matched_by_entity(...)` 与 `action=merge/profile_update` 要求 `entities`
- `action=move(...)` 要求 `layer_count > 1`
- `action=summarize/reflect/extract_concept/prototype_form` 通常要求 selection 覆盖多个 unit
- `action=dedup/prune/review` 会影响后续 retrieval 可见内容

补充边界：

- `append`、普通 ingest-time `upsert`、普通 ingest-time `merge` 不应再作为默认 `memory_evolution` 基线
- 只有当这些动作被理解为额外触发的已有记忆重整时，才应保留在 `memory_evolution` 家族中
- `evolution_trigger` 默认语义应为 `Never`

示例：

```text
memory_evolution : MemoryEvolution(
    selection=matched_by_entity(entity_field="entities"),
    action=profile_update(),
    effect=modify,
    trigger=periodic(every=20)
)
```

```text
memory_evolution : MemoryEvolution(
    selection=time_window(window_size=100),
    action=reflect(),
    effect=add,
    trigger=periodic(every=100)
)
```

```text
memory_evolution : MemoryEvolution(
    selection=low_activity(activity_threshold=0.2),
    action=move(target_layer="cold_archive"),
    effect=move,
    trigger=budget_exceeded(threshold=0.8)
)
```

### 2.5.6 Retrieval 的规范写法（建议）

对 `retrieval`，推荐统一写成：

```text
Retrieval(
  signals={...},
  ranker=...,
  flow=...,
  constraints={...},
  ...
)
```

推荐 signal：

```text
similarity
keyword_match
recency
importance
entity_match
graph_proximity
hierarchy_match
diversity_penalty
```

推荐 ranker：

```text
identity(source=...)
weighted_sum(weights=[...])
rrf(k=...)
decay(decay_rate=..., half_life=...)
rerank_llm(model=..., rerank_prompt=...)
mmr(lambda=...)
```

推荐 flow：

```text
single_stage(top_k=...)
two_stage(recall_k=..., final_k=...)
top_down(levels=..., expand_top_k=...)
agent_invoked(tool_names=[...], backend=...)
```

推荐约束：

- `signals` 中包含 `similarity` 时，要求 `embedding`
- `signals` 中包含 `keyword_match` 时，要求 `text` 或 `sparse_embedding`
- `signals` 中包含 `entity_match` 时，要求 `entities`
- `signals` 中包含 `graph_proximity` 时，要求图边与 graph index
- `flow=top_down(...)` 时，要求层级摘要或父子层级存在

示例：

```text
retrieval : Retrieval(
    signals={similarity},
    ranker=identity(source="similarity"),
    flow=single_stage(top_k=10)
)
```

```text
retrieval : Retrieval(
    signals={similarity, recency, importance},
    ranker=weighted_sum(weights=[0.5, 0.3, 0.2]),
    flow=single_stage(top_k=50)
)
```

```text
retrieval : Retrieval(
    signals={entity_match},
    ranker=rerank_llm(model="gpt-4.1"),
    flow=two_stage(recall_k=20, final_k=5)
)
```

```text
retrieval : Retrieval(
    signals={hierarchy_match},
    ranker=identity(source="hierarchy_match"),
    flow=top_down(levels=3, expand_top_k=5)
)
```

### 2.5.7 Readout 的规范写法（建议）

对 `readout`，当前 design space 只保留少数真正影响 memory 使用方式的模式：

```text
FlatConcat(...)
SummarizedReadout(...)
TemplatedPrompt(...)
CodeInjection(...)
```

其中：

- `FlatConcat` 是默认 readout 基线
- `SummarizedReadout` 表示检索后再压缩
- `TemplatedPrompt` 表示用模板包装 readout
- `CodeInjection` 保留为特殊 mode，通常只用于 skill/tool memory，不参与默认搜索空间

示例：

```text
readout : FlatConcat(
    separator="\n\n",
    max_tokens=2000,
    format="text"
)
```

```text
readout : SummarizedReadout(
    model="gpt-4.1",
    max_length=400
)
```

```text
readout : TemplatedPrompt(
    template="Relevant memories:\n{{items}}\nUse them when answering."
)
```

```text
readout : CodeInjection(
    format="function_def"
)
```

### 2.6 Trigger 声明

控制 `memory_evolution` 的触发时机。

这里的 `evolution_trigger` 不再被理解为“常规写入的第二层写触发器”，而是：

- 是否要启动额外 memory evolution
- 默认应关闭，只有 reflective / maintenance / periodic 场景才打开

```ebnf
TriggerBlock    ::= 'triggers' '{' TriggerDecl+ '}'

TriggerDecl     ::= TriggerSlot ':' TriggerExpr

TriggerSlot     ::= 'evolution_trigger'

TriggerExpr     ::= 'Periodic' '(' 'every' '=' INT ')'
                   | 'OnEvent' '(' EventType ')'
                   | 'BudgetExceeded' '(' 'threshold' '=' FLOAT ')'
                   | 'Conditional' '(' Predicate ')'
                   | 'AfterWrite'      (* 显式要求写后立刻执行一次额外演化 *)
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
    decisions    : list[bool]?        // C 输出, D 读
    placement    : list[PlacePlan]?   // D 可写, 也可仅作为 trace/plan
    query        : Query?             // G 阶段读
    retrieved    : RetrievedContext?   // G 输出, H 读
    readout      : ReadoutPacket?     // H 输出
}
```

在新的语义下：

- D 槽 `organization` 可以直接修改 `store`
- E 槽 `memory_evolution` 是额外 pass，而不是默认落库点

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
memory_evolution: None
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
