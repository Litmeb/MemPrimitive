# MemPrimitive 功能展示文档

这份文档不是 DSL 规范，也不是模块字典，而是给“想快速展示这个项目到底能做什么”的场景准备的。

它的重点只有一件事：

**展示 `MemPrimitive` 如何用统一的 `MemoryPipeline`，低成本地组装出不同风格的 memory module。**

如果你要给别人介绍这个项目，最适合讲的不是“我们定义了多少概念”，而是：

- 你可以先写一个最小 memory module
- 然后只替换少数 slot，就得到另一种行为
- 需要多层 memory、条件路由、触发式维护、graph memory 时，不用推倒重来
- 经典方法也可以被拆成显式可读的模块组合

建议配合 [`memprimitive/example/demonstration`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration) 下的脚本一起看。

## 1. 先用一句话理解项目

在 `MemPrimitive` 里，memory system 不是一个黑盒方法名，而是一条可组装的流水线：

```text
ingest:
  unit_formation
  -> representation
  -> write_trigger
  -> organization
  -> evolution_trigger
  -> memory_evolution

recall:
  retrieval
  -> readout
```

你真正要做的事，不是“选一个大而全的 memory 框架”，而是：

**把每个 memory 机制放到对应 slot，然后按任务需要替换、叠加和组合。**

这就是它灵活的来源。

## 2. 最小可用 memory module

展示项目时，最好的起点永远是 [`memprimitive/example/demonstration/minimal_pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/minimal_pipeline.py)。

核心代码非常短：

```python
pipeline = MemoryPipeline(
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(),
    retrieval=RecencyRetrieval(top_k=2),
    readout=ConcatenateReadout(),
)
```

这个例子很重要，因为它说明：

- 组一个 memory module 不需要先写一套复杂框架
- 每个 slot 都有可直接使用的 baseline 实现
- 先让系统跑通，再逐个替换 slot 就行

如果你只想演示“这个项目能很快搭一个 memory”，这一段就够了。

运行方式：

```text
python -m memprimitive.example.demonstration.minimal_pipeline
```

## 3. 灵活性到底体现在哪里

这个项目最值得展示的，不是单个模块有多复杂，而是下面这三种“低改动、高变化”的组合能力。

### 3.1 只换一个 slot，就能改掉系统行为

最小闭环里：

- `AlwaysWriteTrigger()` 表示所有内容都写入
- `AppendOrganization()` 表示按默认层直接追加
- `RecencyRetrieval(top_k=2)` 表示回忆最近内容

如果你把其中一个 slot 换掉，行为就会立刻变化：

- 换 `write_trigger`，系统从“全写”变成“选择性写入”
- 换 `retrieval`，系统从“最近回忆”变成“按 embedding / entity / graph 检索”
- 换 `organization`，系统从“平铺追加”变成“路由到多层或 graph 层”

这种改法的价值在于：

- 你可以做控制变量实验
- 你可以把复杂系统拆成局部设计决策
- 你可以先做最小版本，再逐步增强

### 3.2 一个 slot 可以串联多个模块

在 `MemPrimitive` 中，某些 slot 可以直接接收一个 iterable，这意味着“串行加工”，不是只能塞一个模块。

例如 A-MEM-like 演示中的表示层：

```python
representation=(
    SemanticFieldEnrichmentRepresentation(note_namespace="amem"),
    RetrievalOrientedEmbeddingRepresentation(note_namespace="amem"),
)
```

这表示：

1. 先把原始文本扩成结构化 note payload
2. 再把 note 变成适合检索的 embedding 表示

你不需要自己写一个大类把这两步硬编码在一起。

### 3.3 需要 fan-out 时，可以显式分发

如果你想让同一份输入同时进入多条 memory 路径，不是靠复制脚本，而是用 dispatch。

可以参考 [`memprimitive/example/demonstration/dispatch_organization_recall.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/dispatch_organization_recall.py)。

典型写法：

```python
DispatchOrganization(
    (
        AppendOrganization(target_layer="working"),
        GraphAppendOrganization(target_layer="knowledge_graph"),
    ),
    primary_index=0,
)
```

这很适合展示项目的“组合性”：

- 一次 ingest
- 同时落到普通层和 graph 层
- 下游仍然保持统一的 pipeline 语义

## 4. 方便性到底体现在哪里

“方便”不是指功能少，而是指项目把复杂度放在了明确的位置上。

### 4.1 先搭骨架，再加能力

推荐的组装方式是：

1. 先用最小 pipeline 跑通
2. 再反推自己需要什么组织和检索能力
3. 最后补 representation、trigger 和 evolution

这意味着你不用一开始就想清楚整套 architecture。

### 4.2 topology 是声明式的，不是散落在脚本里的

多层 store 不是额外系统，而是 `StoreTopology` 的一部分。

参考 [`memprimitive/example/demonstration/topology_store.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/topology_store.py)：

```python
topology = StoreTopology.from_layers(
    [
        StoreLayerSpec(name="working", theme="working", indices=("temporal", "keyword")),
        StoreLayerSpec(
            name="knowledge_graph",
            theme="knowledge_graph",
            shape="Graph",
            indices=("graph", "entity"),
        ),
    ]
)
```

这让你在展示项目时可以很清楚地说：

- memory 有哪些层
- 每层是什么 shape
- 每层支持哪些 index 能力

也就是说，结构不是隐含在模块内部，而是显式放在 topology 里。

### 4.3 trigger 可以“组”出来，不一定要新写类

参考 [`memprimitive/example/demonstration/composed_triggers.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/composed_triggers.py)。

```python
write_trigger=compose_write_trigger(
    name="demo_threshold_write_trigger",
    signal_providers=(ConstantSignal(signal_name="importance_hint", value=0.8),),
    scorer=WeightedSumScorer(weights={"importance_hint": 1.0}),
    gate=AlwaysOpenGate(),
    policy=ThresholdPolicy(threshold=0.5),
)
```

这个能力特别适合展示“方便性”：

- 不是每次都要新建一个 trigger 类
- 可以直接把 signal、scorer、gate、policy 拼起来
- 读代码的人能一眼看出触发逻辑

这比“隐藏在一个黑盒类里”更适合研究和演示。

## 5. 用几个演示脚本快速展示项目能力

如果你只有几分钟介绍项目，下面这条顺序最适合做 showcase。

### 5.1 第一步：最小闭环

脚本：

- [`memprimitive/example/demonstration/minimal_pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/minimal_pipeline.py)

要点：

- 先证明 memory module 可以很快组起来
- 让听众知道所有复杂能力都建立在同一条 pipeline 上

### 5.2 第二步：多层与路由

脚本：

- [`memprimitive/example/demonstration/topology_store.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/topology_store.py)
- [`memprimitive/example/demonstration/conditional_layer_routing.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/conditional_layer_routing.py)
- [`memprimitive/example/demonstration/layer_aware_semantic_working.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/layer_aware_semantic_working.py)

要点：

- memory 不只是一张表
- 你可以声明 working / semantic / graph 等不同层
- organization 和 retrieval 都能围绕这些层工作

### 5.3 第三步：组合式 trigger

脚本：

- [`memprimitive/example/demonstration/composed_triggers.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/composed_triggers.py)
- [`memprimitive/example/demonstration/trigger_family_infrastructure.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/trigger_family_infrastructure.py)

要点：

- 触发逻辑不是硬编码的单体模块
- 可以拆成 signal、scorer、gate、policy 四段
- 这让 memory 行为更容易比较、复用和搜索

### 5.4 第四步：graph memory 不是单模块，而是整条链协作

脚本：

- [`memprimitive/example/demonstration/graph_baseline_pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/graph_baseline_pipeline.py)
- [`memprimitive/example/demonstration/graph_dependent_pipeline.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/graph_dependent_pipeline.py)

要点：

- graph 不是“选一个 graph 模块”就结束
- 它需要 representation、organization、evolution、retrieval 共同配合
- 但这些部分仍然是同一套 slot 里的显式选择

这正好能体现项目“复杂，但不混乱”。

### 5.5 第五步：经典方法也能拆回 slot 组合

脚本：

- [`memprimitive/example/demonstration/reflexion_reflection_cycle.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/reflexion_reflection_cycle.py)
- [`memprimitive/example/demonstration/amem_like_graph_cycle.py`](/d:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/amem_like_graph_cycle.py)

这两个脚本最适合回答一个关键问题：

**这个框架是不是只能拼 baseline toy example？**

答案是否定的。它们展示的是：

- Reflexion-like 行为可以拆成显式的后半段模块组合
- A-MEM-like graph note 流程也可以拆成显式的 slot 组合

也就是说，项目不只是“能写几个简单 memory”，而是已经能把经典方法重写成模块级结构。

## 6. 三种特别适合展示的 memory module 组装方式

### 6.1 从 append-only 到 selective memory

起点：

```python
write_trigger=AlwaysWriteTrigger()
organization=AppendOrganization()
retrieval=RecencyRetrieval(top_k=3)
```

增强方式：

- 把 `AlwaysWriteTrigger()` 换成组合 trigger 或 LLM-judged trigger
- 保留 `AppendOrganization()` 不动
- 先只改变“写不写”，不改变“怎么存”

适合展示：

- selective write
- importance gate
- trigger family

### 6.2 从 flat memory 到 layered memory

起点：

- 默认 store 或单层 topology

增强方式：

- 显式声明 `working`、`semantic`、`knowledge_graph`
- 用 `ConditionalLayerOrganization` 或 `DispatchOrganization` 做写入分流
- 用 `LayerAwareRetrieval` 做分层召回

适合展示：

- memory 结构可以独立设计
- 同一输入可以进入多个 memory view
- 读取逻辑可以按层编排

### 6.3 从普通文本记忆到 graph note memory

起点：

- `BasicRepresentation()`
- `AppendOrganization()`

增强方式：

- 用结构化 representation 生成实体、三元组或 enriched note
- 写入 graph layer
- 加上 neighbor-triggered evolution
- 用 graph/vector retrieval 做读回

适合展示：

- representation 会决定后续能力
- graph 不是额外插件，而是 pipeline 内部的一种组合风格
- 更复杂的 classic family 也能落到相同接口上

## 7. 如果你要自己展示这个项目，建议这样讲

一个比较顺手的讲法是：

1. 先展示最小 `MemoryPipeline`，说明“memory module 可以像搭积木一样先拼起来”
2. 再展示 topology 和 layered retrieval，说明“结构层也是显式可设计的”
3. 然后展示 trigger composition，说明“行为策略也能被拆开重组”
4. 最后用 Reflexion-like 或 A-MEM-like 演示说明“经典方法可以被还原成 slot 组合”

这条叙事线比直接讲 DSL 愿景更容易让人感受到项目当前已经能做什么。

## 8. 一句话总结这个项目最值得展示的地方

`MemPrimitive` 最强的展示点，不是“它支持很多 memory 方法”，而是：

**它把 memory system 变成了一个可以从最小闭环开始、再通过替换 slot、叠加 topology、组合 trigger、扩展 graph 管线来逐步长出来的模块化系统。**

这就是它在“组建 memory module”这件事上的灵活性和方便性。
