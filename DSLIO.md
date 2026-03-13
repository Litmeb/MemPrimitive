目标是让 memory DSL 不只是 taxonomy，而是一个 **可组合、可搜索、可复用、可替换** 的系统，那么每个字段最好都定义成一个标准模块接口，而不是只写成配置项。

* **DSL 层**：描述系统由哪些模块组成
* **Interface 层**：规定每个模块吃什么、吐什么
* **Implementation 层**：具体算法去实现这些接口

---

# 1. 总体思路：把 memory system 看成一条数据流

可以把整个系统抽象成：

```text
raw input
  -> unit formation
  -> representation
  -> organization / storage
  -> write/update
  -> compression / abstraction
  -> retrieval
  -> maintenance
  -> readout
  -> downstream agent
```

所以每个模块都应该是：

```text
Module: InputState -> OutputArtifact (+ side effects on memory store)
```

但这里有一个关键点：

**memory 模块不只是纯函数**，很多模块会修改 store。
所以接口最好统一成下面这种形式：

```python
output, new_store, aux = module(input, store, context, config)
```

也就是说，每个模块都接收：

* `input`：当前要处理的对象
* `store`：当前 memory store
* `context`：运行上下文
* `config`：模块参数

返回：

* `output`：供下游使用的结果
* `new_store`：更新后的 store
* `aux`：中间分数、日志、trace、解释信息

这会非常适合模块化实验与搜索。

---

# 2. 先定义几个全局标准对象

在谈字段接口前，先定义全系统共享的数据类型。不然每个模块都各说各话。

---

## 2.1 Observation

表示外部输入。

```python
Observation = {
    "id": str,
    "type": str,           # dialogue / tool_result / environment_step / document / reflection
    "content": Any,        # raw text, json, image desc, tool output ...
    "timestamp": float,
    "source": str,
    "metadata": dict,
}
```

---

## 2.2 MemoryUnit

表示最小记忆单元。

```python
MemoryUnit = {
    "unit_id": str,
    "unit_type": str,      # turn / event / fact / entity_state / summary / latent
    "content": Any,        # text / structure / tensor ref
    "embedding": Any | None,
    "entities": list[str],
    "tags": list[str],
    "timestamp": float,
    "span": dict | None,   # source span or provenance
    "confidence": float | None,
    "importance": float | None,
    "novelty": float | None,
    "access_stats": {
        "count": int,
        "last_access": float | None
    },
    "relations": list[dict],
    "metadata": dict,
}
```

---

## 2.3 MemoryStore

表示整个 memory 状态。

```python
MemoryStore = {
    "stores": {
        "working": list[MemoryUnit],
        "episodic": list[MemoryUnit],
        "semantic": list[MemoryUnit],
        "profile": list[MemoryUnit],
    },
    "indices": {
        "vector_index": Any,
        "entity_index": dict,
        "time_index": Any,
        "graph_index": Any,
    },
    "budgets": {
        "max_units": int,
        "max_tokens": int,
    },
    "metadata": dict,
}
```

---

## 2.4 Query

表示 retrieval 的查询。

```python
Query = {
    "query_id": str,
    "text": str,
    "goal": str | None,
    "entities": list[str],
    "constraints": dict,
    "timestamp": float,
    "metadata": dict,
}
```

---

## 2.5 RetrievedContext

表示 retrieval 结果。

```python
RetrievedContext = {
    "items": list[MemoryUnit],
    "scores": list[dict],   # similarity / recency / utility / symbolic ...
    "summary": str | None,
    "trace": dict,
}
```

---

## 2.6 AgentContext

表示运行时上下文。

```python
AgentContext = {
    "task": str | None,
    "goal": str | None,
    "dialogue_history": list[dict],
    "current_state": dict,
    "tool_state": dict,
    "runtime": dict,
}
```

---

# 3. 每个字段如何定义标准接口

下面进入重点。

---

## A. Unit Formation Interface

它负责把原始 observation 切成 memory units。

### 接口定义

```python
def unit_formation(
    observation: Observation,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[MemoryUnit], dict]:
    ...
```

返回：

* `list[MemoryUnit]`：形成的记忆单元
* `dict`：aux，比如抽取证据、parser logits、触发原因

### 输入语义

* `observation`：一条新输入
* `store`：可选，用于 entity-aware parsing
* `context`：当前任务、对话历史等

### 输出要求

每个输出 unit 至少要包含：

* `unit_id`
* `unit_type`
* `content`
* `timestamp`
* `metadata.provenance`

### 常见实现

* `TurnSplitter`
* `EventExtractor`
* `FactExtractor`
* `EntityStateExtractor`
* `SessionSummarizer`

---

## B. Representation Interface

它负责把 unit 规范化成某种表示，并补全 embedding / schema。

### 接口定义

```python
def represent(
    units: list[MemoryUnit],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[MemoryUnit], dict]:
    ...
```

### 作用

* 给 unit 加 embedding
* 转成 kv / triple / frame
* 补 metadata
* 标准化字段名

### 输出要求

representation 模块不能改变 unit 的语义身份，只能改变或增强表示形式。

例如：

* 文本事实 -> triple
* 对话 turn -> hybrid(text + embedding + entities)

---

## C. Organization Interface

它负责把 units 放进某种结构中，但不一定真正写入。

这里建议区分：

* `organization`：产生 links / placement plan
* `write/update`：真正写入 store

### 接口定义

```python
def organize(
    units: list[MemoryUnit],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[dict], dict]:
    ...
```

输出可以是 `PlacementPlan`：

```python
PlacementPlan = {
    "unit_id": str,
    "target_store": str,       # working / episodic / semantic / profile
    "links_to_add": list[dict],
    "index_updates": list[dict],
}
```

### 常见功能

* 判定该放哪个子库
* 建立 temporal links
* 建立 entity links
* 建立 similarity / causal links

---

## D. Write Trigger Interface

它负责判断“要不要写”。

这是非常关键的独立模块，建议单列，不要和 update 混成一个字段。

### 接口定义

```python
def should_write(
    units: list[MemoryUnit],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[bool], list[dict]]:
    ...
```

返回：

* 每个 unit 是否写入
* 每个决策的解释与分数

例如输出：

```python
[
  True, False, True
],
[
  {"importance": 0.91, "novelty": 0.67, "reason": "new_user_preference"},
  {"importance": 0.21, "reason": "small_talk"},
  {"importance": 0.88, "reason": "task_outcome"}
]
```

### 这样设计的好处

* 能单独 ablate write policy
* 能做 learned write / heuristic write / llm-judge write 的统一比较

---

## E. Update Interface

它负责真正把 unit 写进 store，或者修改旧 memory。

### 接口定义

```python
def update_store(
    units: list[MemoryUnit],
    placement: list[dict],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[MemoryStore, dict]:
    ...
```

### 功能

* append
* replace
* merge
* delta update
* rewrite
* conflict resolution

### 输出中的 aux 应包含

* affected unit ids
* conflicts found
* merge trace
* overwritten records

这个 trace 很重要，因为你之后分析 memory 机制时会需要。

---

## F. Compression / Abstraction Interface

它负责从已有记忆中生成更高层表示。

### 接口定义

```python
def compress(
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[MemoryUnit], MemoryStore, dict]:
    ...
```

为什么返回两个东西？

* `list[MemoryUnit]`：新生成的摘要/概念记忆
* `MemoryStore`：更新后的 store

### 常见触发方式

* periodic
* episode_end
* budget_exceeded
* explicit_reflection

### 常见输出

* summary units
* concept units
* profile rewrite
* prototype units

---

## G. Retrieval Interface

它负责给定 query，从 store 找 relevant memory。

### 接口定义

```python
def retrieve(
    query: Query,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[RetrievedContext, dict]:
    ...
```

### 返回

* `RetrievedContext`
* 检索 trace

### trace 应该标准化

因为 retrieval 是核心实验变量之一。

例如：

```python
{
    "candidate_count": 200,
    "selected_count": 8,
    "scorers": {
        "similarity": [...],
        "recency": [...],
        "entity_match": [...],
        "utility": [...]
    },
    "final_ranking": [...]
}
```

这样你可以研究：

* 到底是 sim 起作用还是 recency 起作用
* 为什么某条记忆被选中

---

## H. Maintenance Interface

它负责容量控制、遗忘、去重、重排序等。

### 接口定义

```python
def maintain(
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[MemoryStore, dict]:
    ...
```

### 功能

* pruning
* decay update
* dedup
* conflict cleanup
* consolidation trigger
* archival movement

### 建议

把 `maintenance` 设计成可周期触发的后台步骤，但在 DSL 执行图里仍然是显式模块。

---

## I. Readout Interface

它负责把 retrieved memory 转成下游 agent 可消费的输入。

### 接口定义

```python
def readout(
    retrieved: RetrievedContext,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[dict, dict]:
    ...
```

输出可能是：

```python
ReadoutPacket = {
    "prompt_blocks": list[str],
    "structured_fields": dict,
    "citations": list[dict],
    "latent_state_update": Any | None,
}
```

### 为什么单独立接口

因为：

* retrieval 解决“找什么”
* readout 解决“怎么用”

这两个问题不能混。

同样的 retrieved items：

* 可以直接拼 prompt
* 可以先 summary 再拼
* 可以按 profile / facts / history 分区注入
* 可以只给 planner，不给 responder

---

# 4. 更进一步：统一成标准模块签名

为了让 DSL 真正模块化，你甚至可以强制所有模块都遵守统一签名：

```python
class MemoryModule:
    def __call__(self, packet, store, context, config):
        return output_packet, new_store, aux
```

其中 `packet` 是通用载体，按模块阶段不同装不同内容。

例如：

```python
Packet = {
    "observation": Observation | None,
    "units": list[MemoryUnit] | None,
    "query": Query | None,
    "retrieved": RetrievedContext | None,
    "readout": dict | None,
    "trace": dict,
}
```

然后不同模块只读写自己关心的字段。

这样你的 pipeline 可以写成：

```python
packet = {"observation": obs, "trace": {}}

packet, store, aux1 = unit_formation(packet, store, context, cfg1)
packet, store, aux2 = represent(packet, store, context, cfg2)
packet, store, aux3 = organize(packet, store, context, cfg3)
packet, store, aux4 = write(packet, store, context, cfg4)
...
```

这就是很典型的 **IR-style intermediate representation** 思路。
我觉得这对 DSL 很有帮助。

---

# 5. 还可以定义模块的“能力约束”

如果想做自动搜索，只定义 IO 还不够，还要定义 **type constraints**。
不然搜索会组合出很多非法配置。

比如：

* `graph_retrieval` 要求 store 里有 graph links
* `entity_merge_update` 要求 unit 里有 entity ids
* `similarity_retrieval` 要求 unit 有 embedding
* `profile_rewrite` 要求目标 store 是 profile / semantic
* `delta_update` 要求 unit 可对齐到旧记录

所以每个模块除了 IO 外，还应该声明：

```python
ModuleSpec = {
    "name": str,
    "input_requirements": list[str],
    "output_guarantees": list[str],
    "side_effects": list[str],
}
```

例如：

```python
{
    "name": "hybrid_retriever",
    "input_requirements": ["query.text", "store.indices.vector_index"],
    "output_guarantees": ["retrieved.items", "retrieved.scores.similarity"],
    "side_effects": []
}
```

再比如：

```python
{
    "name": "entity_merge_updater",
    "input_requirements": ["units.entities", "placement.target_store"],
    "output_guarantees": ["store.updated_units", "trace.merge_log"],
    "side_effects": ["modify_store"]
}
```

这对自动组合非常重要。

---

# 6. DSL 字段不仅能定义 IO，还能定义“操作语义”

你可以把每个字段分成三层：

### 1) Type signature

输入输出类型

### 2) Contract

这个模块承诺做什么，不承诺做什么

### 3) Side effects

它会如何修改 store 或 metadata

例如 `retrieve` 的 contract 可以是：

* 输入 query 和 store
* 输出一个有序的 memory list
* 必须提供 score trace
* 不允许直接修改原始 memory 内容

而 `update` 的 contract 是：

* 输入 units 与 placement
* 可以修改 store
* 必须返回 conflict / overwrite trace

这样整个系统会非常清晰。

---

# 7. 我建议的最小标准接口集合

如果你想尽快落地，不必一开始设计得太重。可以先定一个最小版本：

```python
unit_formation(observation, store, context, config) 
    -> units, aux

represent(units, store, context, config) 
    -> units, aux

should_write(units, store, context, config) 
    -> decisions, aux

organize(units, store, context, config) 
    -> placement, aux

update_store(units, placement, store, context, config) 
    -> store, aux

retrieve(query, store, context, config) 
    -> retrieved, aux

readout(retrieved, store, context, config) 
    -> readout_packet, aux

maintain(store, context, config) 
    -> store, aux
```

这是一个非常实用的起点。

---

# 8. 一个具体例子：如何模块化组合

假设你定义一个 memory config：

```python
MemoryConfig(
    unit_formation = EventExtractor(),
    representation = HybridRep(embed=True, entity_link=True),
    write_trigger = ImportanceNoveltyGate(),
    organization = EntityTemporalOrganizer(),
    update = MergeOrAppend(),
    retrieval = HybridRetriever(),
    readout = StructuredReadout(),
    maintenance = ImportanceDecayPruner()
)
```

那么运行时就是：

### 写入阶段

```python
units = EventExtractor(obs)
units = HybridRep(units)
decisions = ImportanceNoveltyGate(units, store)
selected_units = filter_by_decision(units, decisions)
placement = EntityTemporalOrganizer(selected_units, store)
store = MergeOrAppend(selected_units, placement, store)
store = ImportanceDecayPruner(store)
```

### 读取阶段

```python
retrieved = HybridRetriever(query, store)
readout_packet = StructuredReadout(retrieved, store)
```

这就已经是一个可执行的 memory DSL runtime 了。

---

# 9. 从研究角度看，标准 IO 接口的意义非常大

它不仅是工程问题，还直接决定你的研究能不能成立。

---

## 9.1 便于做组合搜索

如果模块共享接口，你才能：

* 固定 representation，搜索 retrieval
* 固定 retrieval，搜索 write policy
* 联合搜索 write + update + maintain

否则搜索空间只是纸上谈兵。

---

## 9.2 便于做公平比较

统一接口后，你可以真正控制变量：

* 同样的 unit formation，不同 retrieval
* 同样的 update，不同 write trigger
* 同样的 retrieval，不同 readout

这对论文实验很关键。

---

## 9.3 便于分析 recurring motifs

如果很多高性能系统都满足：

* event/fact unit
* hybrid representation
* selective write
* merge update
* hybrid retrieval
* periodic compression

那你就能从搜索结果归纳出 motif，而不是只得到“某个黑箱配置更好”。

---

# 10. 但要注意：不是所有字段都适合完全独立

这里有一个现实问题：
有些字段之间耦合很强，不能机械拆开。

比如：

### representation 和 retrieval 强耦合

* 没 embedding 就做不了向量检索
* 没 graph link 就做不了 graph hop retrieval

### unit formation 和 update 强耦合

* fact unit 更适合 replace/merge
* raw chunk 更适合 append-only

### compression 和 maintenance 强耦合

* summary 生成后可能反过来触发删 raw memory
* consolidation 会改变 retention policy

所以正确做法不是假装完全独立，而是：

* **接口独立**
* **语义上允许约束**
* **搜索时通过 compatibility constraints 过滤非法组合**

这比“完全自由组合”严谨得多。