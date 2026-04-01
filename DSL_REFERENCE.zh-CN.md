# MemPrimitive DSL 参考（当前实现面）

这份文档只描述仓库里当前仍然公开、可直接组合的 DSL-like 运行时表面。

## 当前结论

当前 `trigger` 层已经暂时收缩为基础 slot trigger，不再公开 `signal / scorer / gate / policy` 细分，也不再公开 `compose_*trigger(...)` 现场组装路径。

保留的 trigger API 只有：

- `AlwaysWriteTrigger()`
- `ThresholdWriteTrigger(threshold=..., constant=...)`
- `NeverEvolutionTrigger()`
- `ThresholdEvolutionTrigger(threshold=..., constant=...)`

## Pipeline Slots

标准顺序仍然是：

```text
unit_formation
-> representation
-> write_trigger
-> organization
-> evolution_trigger
-> memory_evolution
-> retrieval
-> readout
```

## Trigger Slots

### `write_trigger`

输出写到 `packet.decisions`。

#### `AlwaysWriteTrigger`

- 作用：所有 unit 都写入
- 输入：`packet.units`
- 输出：`packet.decisions`

```python
AlwaysWriteTrigger()
```

#### `ThresholdWriteTrigger`

- 作用：使用常量 `constant` 和阈值 `threshold` 产生统一写入决策
- 输入：`packet.units`
- 输出：`packet.decisions`

```python
ThresholdWriteTrigger(threshold=0.5, constant=1.0)
```

### `evolution_trigger`

输出写到 `packet.evolution_decisions`。

#### `NeverEvolutionTrigger`

- 作用：默认不触发额外 evolution
- 输入：`packet.units`、`packet.placements`
- 输出：`packet.evolution_decisions`

```python
NeverEvolutionTrigger()
```

#### `ThresholdEvolutionTrigger`

- 作用：使用常量 `constant` 和阈值 `threshold` 产生统一 evolution 决策
- 输入：`packet.units`、`packet.placements`
- 输出：`packet.evolution_decisions`

```python
ThresholdEvolutionTrigger(threshold=0.5, constant=1.0)
```

## Trace 说明

仍然保留：

- `trace["write_trigger"]`
- `trace["evolution_trigger"]`

但不再承诺旧的 trigger-family 字段，例如：

- `signals`
- `scorer`
- `gate`
- `policy`
- `family`

当前基础 trigger trace 只保证最小必要信息：

- `module`
- `decisions` / `evolution_decisions`
- `constant`
- `threshold`
- `per_unit`

## 示例

### 最小 pipeline

```python
from memprimitive import MemoryPipeline
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
```

### Graph baseline

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

## 说明

与 `Reflexion`、`MemGPT`、`A-MEM` 等经典工作流相关的更细 trigger 语义，目前不再由 baseline trigger API 公开表达；如果仓库中仍有旧名字出现，应视为历史残留或 classic 层封装，而不是当前 baseline DSL 的公开能力。
