# MemPrimitive 教程（简化版）

这份教程对应当前仓库的基础运行面。

## 先记住一件事

`trigger` 已经暂时简化，不再讲 `signal / scorer / gate / policy` 四段式，也不再讲 `ComposeTrigger`。

当前只保留四个基础 trigger：

- `AlwaysWriteTrigger`
- `ThresholdWriteTrigger`
- `NeverEvolutionTrigger`
- `ThresholdEvolutionTrigger`

## 1. 最小闭环

```python
from memprimitive import MemoryPipeline, Observation, Query
from memprimitive.baselines import (
    AlwaysWriteTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)

pipeline = MemoryPipeline(
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(),
    write_trigger=AlwaysWriteTrigger(),
    organization=AppendOrganization(),
    retrieval=RecencyRetrieval(top_k=2),
    readout=ConcatenateReadout(),
)

pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
readout = pipeline.recall(Query(text="Alice likes what?"))
print(readout.text)
```

这里的含义是：

- `PassThroughUnitFormation()`：把 observation 直接变成一个 unit
- `BasicRepresentation()`：补充基础表示
- `AlwaysWriteTrigger()`：所有 unit 都允许写入
- `AppendOrganization()`：真正落库
- `RecencyRetrieval()`：按时间取回
- `ConcatenateReadout()`：把结果拼成文本

## 2. 用阈值触发器

### 写入阈值

```python
from memprimitive.baselines import ThresholdWriteTrigger

write_trigger = ThresholdWriteTrigger(threshold=0.5, constant=1.0)
```

这表示：

- 如果 `constant >= threshold`，所有 unit 写入
- 否则所有 unit 都不写

### Evolution 阈值

```python
from memprimitive.baselines import ThresholdEvolutionTrigger

evolution_trigger = ThresholdEvolutionTrigger(threshold=0.5, constant=1.0)
```

这表示：

- 如果 `constant >= threshold`，所有 unit 的 `evolution_decisions=True`
- 否则为 `False`

## 3. Graph baseline 组合

```python
from memprimitive import MemoryPipeline
from memprimitive.baselines import (
    BasicRepresentation,
    GraphAppendOrganization,
    GraphNeighborAppendEvolution,
    GraphReadout,
    GraphSeedAndExpandRetrieval,
    ThresholdEvolutionTrigger,
    TripleRepresentation,
)

pipeline = MemoryPipeline(
    representation=(
        BasicRepresentation(elements=("text",)),
        TripleRepresentation(method="direct"),
        BasicRepresentation(elements=("tags", "keywords")),
    ),
    organization=GraphAppendOrganization(target_layer="knowledge_graph"),
    evolution_trigger=ThresholdEvolutionTrigger(threshold=0.5, constant=1.0),
    memory_evolution=GraphNeighborAppendEvolution(target_layer="knowledge_graph"),
    retrieval=GraphSeedAndExpandRetrieval(layer="knowledge_graph"),
    readout=GraphReadout(),
)
```

这里的重点不是复杂 trigger，而是：

- 用 graph organization 写入图层
- 用基础 evolution trigger 打开后续 graph evolution
- 用 graph retrieval + graph readout 做召回

## 4. 当前不再推荐的内容

以下内容已经不属于当前 baseline 教程：

- `compose_write_trigger(...)`
- `compose_evolution_trigger(...)`
- `signal / scorer / gate / policy`
- `OutcomeConditionedEvolutionTrigger`
- `NewWriteEvolutionTrigger`
- `NeighborExistsEvolutionTrigger`
- `LLMJudgedWriteTrigger`
- `MetadataGatedWriteTrigger`
- `KeyReadyWriteTrigger`

如果你在旧文档、旧 notebook、旧示例里看到这些名字，请把它们理解为历史设计或待清理遗留，而不是当前教程应继续使用的公开 API。
