# Graph 重写规划（真实 Node / Edge 结构）

## 背景

当前仓库里的 `Graph` layer 还不是一等公民图结构，而是：

- `MemoryStore.layers[layer]` 里依然存的是 `MemoryRecord` 列表。
- graph 语义主要挂在 `record.metadata["graph"]`。
- 边本质上是 `metadata["graph"]["links"]` 里的 `record_id` 列表。
- retrieval / evolution / tool call 都是“基于 record 的邻接关系”在工作，而不是“基于 node / edge 实体”在工作。

这套设计能支持 graph-like baseline，但它有几个根本问题：

- 节点、边没有独立身份，无法稳定表达异构图。
- relation edge 没有自己的属性、时间状态、来源和版本。
- node merge / edge invalidation / edge rewrite 都只能退化成改某条 record 的 metadata。
- 检索和演化只能围绕 record 展开，难以表达 node-level / edge-level 算法。
- `Mem0g`、`HippoRAG`、`AriGraph` 这类真正依赖图原语的系统很难做高保真重构。

## 目标

这次重构建议把目标定得非常清楚：不是“把 metadata 再整理一下”，而是让 graph 在 core 层成为真正的数据结构。

推荐目标：

- `Graph` layer 内部拥有真实的 `node`、`edge`、`adjacency`、`index`。
- `node` 与 `edge` 都有稳定 id，且都能携带 attributes / provenance / timestamp / status。
- 图结构操作通过显式 graph API 完成，而不是直接改 `record.metadata["graph"]`。
- 允许异构节点和异构边。
- 允许 node-level 检索、edge-level 维护、subgraph-level 读出。
- 尽量保持 `MemoryPipeline` 外层用法不被一次性打碎。

## 设计原则

- 先把“图存储模型”做对，再去迁移 baseline；不要继续围绕旧 `MemoryRecord + metadata` 打补丁。
- graph core 与通用 flat record store 分层，但不要完全分家到两套互不相干的系统。
- 兼容层必须显式存在，避免一次性重写所有 baselines。
- 先支持单机内存图，把接口抽象好；后续是否接 Neo4j / NetworkX / graph DB 再说。
- node / edge 的 schema 要足够通用，不能只为 `Mem0g` 定制。

## 推荐总体方案

我建议采用“同一个 `MemoryStore`，但 Graph layer 使用专门后端对象”的方案，而不是简单给 `MemoryRecord.metadata["graph"]` 升级字段。

推荐结构：

- `MemoryStore` 继续作为统一入口。
- 普通 `Flat` layer 继续存 `list[MemoryRecord]`。
- `Graph` layer 改为存 `GraphState`。
- `GraphState` 内部分别管理 `nodes`、`edges`、`adjacency`、`indices`。
- 如有需要，再提供 `GraphProjectionRecord` 把 node / edge 投影成 readout 友好的记录视图。

核心原因：

- 这样能保留现有 pipeline / topology / layer 语义。
- 改动集中在 `core` 和 graph baseline 边界，不必把整个仓库拆成两套 store。
- 后面如果要换成外部图库，也只需要替换 `GraphState` 后端。

## 建议的数据模型

### 1. GraphNode

建议新增类似结构：

```python
@dataclass(slots=True)
class GraphNode:
    node_id: str
    layer: str
    node_type: str
    text: str
    normalized_text: str
    embedding: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    source_record_ids: list[str] = field(default_factory=list)
    source_unit_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    status: str = "active"
```

`node_type` 至少要允许：

- `entity`
- `fact`
- `note`
- `passage`
- `event`
- `summary`

### 2. GraphEdge

建议新增类似结构：

```python
@dataclass(slots=True)
class GraphEdge:
    edge_id: str
    layer: str
    edge_type: str
    source_node_id: str
    target_node_id: str
    relation: str
    weight: float = 1.0
    directed: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    status: str = "active"
```

`edge_type` 至少要允许：

- `relation`
- `mention`
- `belongs_to`
- `neighbor`
- `synonymy`
- `temporal_next`
- `supports`

### 3. GraphState

建议 `GraphState` 负责：

- `nodes_by_id`
- `edges_by_id`
- `out_edges_by_node`
- `in_edges_by_node`
- `node_ids_by_type`
- `edge_ids_by_type`
- `entity_name_index`
- `embedding-ready node subset`
- 可选的 relation signature index，例如 `(source, relation, target)`

## Graph layer 应该提供的核心 API

这一层比数据结构本身更关键，因为后续 organization / evolution / retrieval 都会围绕它稳定下来。

建议至少提供：

- `add_node(...)`
- `update_node(...)`
- `merge_node(...)`
- `delete_node(...)`
- `add_edge(...)`
- `update_edge(...)`
- `upsert_edge(...)`
- `invalidate_edge(...)`
- `delete_edge(...)`
- `neighbors(node_id, edge_types=None, direction="out")`
- `find_nodes(...)`
- `find_edges(...)`
- `find_entity_nodes(entity_text)`
- `find_relation_edges(subject=None, relation=None, object=None)`
- `project_subgraph(...)`

这里要特别注意两件事：

- node API 和 edge API 必须分开，不要再退回“改某节点的 links 列表”。
- `invalidate_edge()` 应该是一等操作，因为 `Mem0g` / bi-temporal KG 类工作会非常需要它。

## 与现有 `MemoryRecord` 的关系

这里有三种可选路线：

1. 完全取消 graph layer 中的 `MemoryRecord`
2. node / edge 是主结构，但允许 projection 成 `MemoryRecord`
3. node 仍继承或复用 `MemoryRecord`

推荐选第 2 种。

原因：

- 第 1 种最纯粹，但会把现有 retrieval/readout 接口一次性打碎。
- 第 3 种会继续把图原语绑死在 record 语义上，长期又会退回旧问题。
- 第 2 种能把“内部真实图”与“外部兼容 record 视图”分层开。

建议做法：

- `GraphState` 内部只认 `GraphNode` / `GraphEdge`。
- graph retrieval/readout 需要兼容旧接口时，再把 node 或 subgraph 投影成 `MemoryRecord` 风格对象。
- 这层 projection 应明确标记来源，例如 `metadata["graph_projection"]`。

## 对 topology / core 的重构建议

### 1. `StoreLayerSpec`

`shape="Graph"` 不能再只代表“这个 layer 允许 metadata 里有 graph 字段”，而要代表“此层使用 graph backend”。

建议新增或强化：

- `settings["graph_backend"]`
- `settings["node_types"]`
- `settings["edge_types"]`
- `settings["default_node_type"]`
- `settings["default_edge_type"]`

### 2. `MemoryStore`

建议把内部存储从：

- `dict[str, list[MemoryRecord]]`

扩成：

- `dict[str, list[MemoryRecord] | GraphState]`

但对外不要直接暴露 union 细节，而是通过专门方法访问：

- `iter_records(layer)` 只服务 flat layer 或 graph projection。
- `graph(layer)` 返回 `GraphState`。
- `append(record)` 只允许 flat layer。
- `append_graph_node(...)` / `append_graph_edge(...)` 或直接 `graph(layer).add_node(...)`。

### 3. 新增 graph-aware id 体系

不要再让边借用 `record_id` 体系。

建议独立维护：

- `node-{sequence}`
- `edge-{sequence}`
- `rec-{sequence}`

## baseline 层应该怎么迁移

这里最重要的是不要“一口气重写全部 graph baseline”。推荐按依赖层次分三批迁移。

### 第一批：组织层

优先重写：

- `GraphAppendOrganization`
- `GraphEntityAppendOrganization`
- `GraphDeduplicationAppendOrganization`
- `GraphEntityDeduplicationAppendOrganization`

新的语义建议是：

- `GraphAppendOrganization` 写入 node / edge，而不是写 graph-shaped record。
- `GraphEntityAppendOrganization` 以 entity 为 node 主体，source observation 作为 provenance，不再一条实体一个伪 record。
- `GraphDeduplicationAppendOrganization` 做的是 node merge / node resolve，而不是 record replace。
- triple 不应只存在于 node metadata；应尽量转成 relation edge。

推荐映射：

- entity -> `GraphNode(node_type="entity")`
- 原始 observation / note -> `GraphNode(node_type="note")` 或保留在独立 flat layer
- `(subject, predicate, object)` -> 一条 `relation` edge
- source note 到 entity 的出现关系 -> `mention` 或 `supports` edge

### 第二批：检索层

优先重写：

- `GraphNeighborRetrieval`
- `ExpandRetrievedGraphNeighbors`
- `VectorGraphSeedAndExpandRetrieval`
- 与 graph 相关的 `EntityRetrieval` / `TripleMemoryRetrieval` 分支

建议语义：

- `GraphNeighborRetrieval` 输入输出基于 `node_id`，不再基于 `record_id`。
- seed 阶段支持 “按 node embedding 检索 node”。
- expand 阶段返回 node、edge 或 subgraph，而不是简单 record 列表。
- 为兼容现有 `RetrievedSet`，可暂时返回 node projection records。

### 第三批：演化层

优先重写：

- `GraphLinkEvolution`
- `GraphNeighborAppendEvolution`
- `GraphNeighborContextTraceEvolution`
- LLM graph tools 相关执行器

建议语义：

- `GraphLinkEvolution` 改为真正新增 / 更新 edge。
- `GraphNeighborContextTraceEvolution` 直接读取邻边和邻点，不再读 `links` 数组。
- `GRAPH_ADD_LINK / UPDATE / DELETE_LINK` 这组工具应该升级成 edge-level 工具，或者补一组新工具：
  - `GRAPH_ADD_EDGE`
  - `GRAPH_UPDATE_EDGE`
  - `GRAPH_INVALIDATE_EDGE`
  - `GRAPH_DELETE_EDGE`

## 推荐迁移顺序

我建议采用 6 个阶段，避免大爆炸式重写。

### 阶段 0：先定义边界，不动 baseline 语义

- 先把新的 `GraphNode` / `GraphEdge` / `GraphState` / `MemoryStore.graph(...)` API 设计定稿。
- 明确 graph layer 的“权威数据”以后只能来自 `GraphState`。
- 写一份兼容矩阵，说明哪些旧模块先继续可用，哪些会进入 deprecated 状态。

### 阶段 1：core 先落地真实图容器

- 在 `core.py` 引入 `GraphNode` / `GraphEdge` / `GraphState`。
- 让 `MemoryStore` 能持有 graph backend。
- 补基础单元测试，只验证 graph 容器本身。

### 阶段 2：做兼容投影层

- 提供 node/subgraph -> `MemoryRecord` projection。
- 提供旧式 `iter_graph_neighbors(...)` 的兼容实现，但内部改走 `GraphState`。
- 保证非 graph baseline 不受影响。

### 阶段 3：先迁 organization

- 让写入路径先变成真实 node / edge。
- 这是最关键的一步，因为只要写入仍是旧 record 形态，后面检索和演化都会被拖回去。

### 阶段 4：再迁 retrieval

- 先保留 `RetrievedSet` 外形。
- 内部改成 node-level seed + edge-level expand。
- 再考虑是否需要专门的 `RetrievedSubgraph` 类型。

### 阶段 5：最后迁 evolution 和 tool-calling

- 把 edge rewrite / invalidation / cleanup 全部迁到 edge-level。
- 让 LLM tools 不再直接接收 `links: [record_id]` 这种旧协议。

## 不建议直接做的事

- 不建议继续在 `metadata["graph"]` 上追加更多字段来“模拟” node/edge。
- 不建议把 node 和 edge 都硬塞进 `MemoryRecord`，只是加一个 `record_type`。
- 不建议一开始就绑定 Neo4j 或外部图库，否则 core 抽象还没稳定就会被后端牵着走。
- 不建议先改 retrieval 再改 organization，因为写入语义不正，检索层会很别扭。

## 兼容策略

这是这次重构成败的关键。

建议保留一层明确兼容期：

- 旧 graph baselines 在第一阶段仍可运行，但标记为 legacy。
- `store.iter_graph_neighbors(...)` 在兼容期继续存在。
- 旧的 `GRAPH_*_LINK` 工具可以先映射到 edge projection 层，但要标记 deprecated。
- 新的 graph baselines 统一走 `GraphState` API，不再直接读写 `metadata["graph"]`。

推荐增加一个兼容原则：

- 任何旧接口如果还能留，就只做“投影兼容”，不要让它继续成为 graph 的权威写路径。

## 测试策略

建议把测试重心从“metadata 有没有某字段”改成“图结构语义是否正确”。

至少补这几类测试：

- `GraphState` 基础操作测试：增删 node / edge、双向邻接、索引更新。
- dedup / merge 测试：entity merge 后 edge 是否重挂接。
- invalidation 测试：边失效后检索是否忽略 inactive edge。
- projection 测试：graph -> `MemoryRecord` 兼容视图是否稳定。
- organization 集成测试：从 unit/triple 到 node/edge 的完整写入。
- retrieval 集成测试：seed / expand 是否真按图遍历。
- evolution 集成测试：edge 更新、删除、失效是否只影响目标子图。

## 建议优先支持的三个 graph 用例

为了避免设计过度抽象，建议先用 3 个具体场景反推 API 是否够用：

1. `Mem0g`
   - entity node
   - relation edge
   - edge invalidation / relation update

2. `HippoRAG`
   - passage node
   - entity node
   - passage-entity incidence / support edge
   - synonymy edge

3. `A-MEM` / note graph
   - note node
   - similarity / strengthening edge
   - neighbor-context rewrite

如果这三个场景都能自然表达，图 core 基本就站稳了。

## 我最推荐的落地路线

如果要一句话总结，我的建议是：

- 先在 `core` 里引入真实 `GraphState`
- 再做 projection 兼容层
- 再优先重写 `organization`
- 最后迁 `retrieval` / `evolution` / tool-calling

而不是：

- 继续扩张 `MemoryRecord.metadata["graph"]`
- 或者直接全仓库同步大改

## 预期收益

完成后你会得到的不是“更像图一点的 record store”，而是：

- 可以表达真正异构图的底座
- 可以把 edge 当成一等对象维护
- 可以支持更 faithful 的 paper reconstruction
- 可以让 graph retrieval / graph evolution 的语义真正独立
- 可以为未来 DSL、搜索空间建模、外部 graph backend 适配打基础

## 一句话结论

这次重构最合适的方向，不是“把现有 graph metadata 规范化”，而是“把 `Graph` layer 升级成 `GraphState(node, edge, index)` 的真实图后端，再用兼容投影把旧 pipeline 平滑迁过去”。
