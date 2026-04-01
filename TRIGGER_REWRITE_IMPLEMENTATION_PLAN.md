# Trigger 重写实现计划

更新时间：2026-04-01

## 1. 目标

基于 [TRIGGERS.md](./TRIGGERS.md) 与 [TRIGGER_SCORE_BOOLEAN_GATE_SURVEY.md](./TRIGGER_SCORE_BOOLEAN_GATE_SURVEY.md) 的结论，给 `MemPrimitive` 制定一份**与现有接口契合**的 trigger 重写计划。目标不是先做一套抽象很厚的新 trigger 语言，而是按当前仓库的真实结构，把高价值 trigger 家族落到可实现的代码接口上。

本计划覆盖：

- `OnInputTrigger`
- `BoundaryEventTrigger`
- `RuntimeEventTrigger`
- `ScalarRuleTrigger`
- `ModelJudgeTrigger`
- `ScheduledMaintenanceTrigger`
  - 拆分成 `PeriodicMaintenanceTrigger`
  - 拆分成 `IdleMaintenanceTrigger`

## 2. 现有接口约束

在开始设计前，必须先承认当前代码的几个硬约束：

### 2.1 Pipeline 只有两个 trigger slot

当前 `MemoryPipeline` 固定顺序是：

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

也就是说，trigger 目前只能落在：

- `write_trigger`
- `evolution_trigger`

对应接口在：

- [memprimitive/interfaces.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/interfaces.py)
- [memprimitive/pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py)

### 2.2 一个类不能同时直接兼容两个 slot

当前 `MemoryPipeline` 会同时检查：

- Python 抽象基类类型是否匹配
- `module.spec.slot` 是否严格等于目标 slot

因此，一个具体类不可能既是 `WriteTriggerModule`，又直接拿去放进 `evolution_trigger` 槽位。

这意味着：

- 文献语义上的 “`OnInputTrigger` / `BoundaryEventTrigger` / `RuntimeEventTrigger`” 可以作为**触发家族名**
- 但代码里建议实现为**slot-specific concrete class**
  - `OnInputWriteTrigger`
  - `OnInputEvolutionTrigger`
  - `BoundaryEventWriteTrigger`
  - `BoundaryEventEvolutionTrigger`
  - 以此类推

如果你坚持公开 API 必须就是精确类名 `OnInputTrigger` 这种单数名字，那么需要额外引入 builder/factory，而不是直接用一个类实例塞进 pipeline。按当前接口，我**不建议**走这条路。

### 2.3 下游已经绑定了两个决策字段

当前下游模块依赖：

- `organization` 读取 `packet.decisions`
- `memory_evolution` 读取 `packet.evolution_decisions`

对应代码主要在：

- [memprimitive/baselines/organization.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/organization.py)
- [memprimitive/baselines/memory_evolution.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/memory_evolution.py)

因此重写 trigger 时，**不要改动这两个核心输出字段**。新 trigger 的实现都应继续产出：

- `packet.decisions: list[bool]`
- `packet.evolution_decisions: list[bool]`

### 2.4 当前 Packet 里已有可复用入口，但还不够结构化

当前 `Packet` 已有这些潜在触发相关字段：

- `events`
- `target_layer_hint`
- `token_budget`
- `working_set`
- `trace`
- `observation.metadata`

定义在：

- [memprimitive/core.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/core.py)

但问题是：

- `events` 目前只是 `list[str] | None`
- 没有统一的 structured trigger payload
- 没有 scheduler 专用入口

所以本计划会优先采用：

1. 不破坏当前 `Packet` 主结构
2. 先统一使用 `observation.metadata["trigger"]`
3. 必要时再补一个轻量 structured context 字段

## 3. 重写总原则

### 3.1 保持 slot 结构不变

不重写 `MemoryPipeline` 的 slot 拓扑。新增 trigger 仍然是：

- `WriteTriggerModule` 子类
- `EvolutionTriggerModule` 子类

### 3.2 把“触发源”统一，而不是把语义名词继续堆厚

优先统一的是实现机制：

- 输入直通
- 边界事件
- 运行时事件
- 显式数值规则
- 模型判定
- 调度触发

而不是重新引入一整套 `signal / scorer / gate / policy` 厚分层。

### 3.3 “检测事件”与“响应事件”分离

这点非常重要：

- `BoundaryEventTrigger` 不负责从原始文本里“自己发明边界”
- `RuntimeEventTrigger` 不负责自己检测 task failure
- `ModelJudgeTrigger` 才是需要主动调用模型做判断的类

也就是说：

- 事件产生：上游 runtime / caller / representation / unit metadata
- trigger 决策：trigger slot

这样更贴近当前仓库接口，也更容易测试。

## 4. 推荐代码落点

### 4.1 新增共享 helper 文件

建议新增：

- [memprimitive/baselines/_trigger_common.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/_trigger_common.py)

职责：

- 统一读取 trigger 上下文
- 统一做 per-unit decision 广播
- 统一生成 trace
- 统一事件匹配 / 标量比较 / model judge 结果归一化

建议放的 helper：

- `extract_trigger_context(packet, store) -> dict[str, Any]`
- `resolve_trigger_events(packet) -> list[str]`
- `resolve_trigger_signals(packet, store) -> dict[str, Any]`
- `broadcast_decision(packet, decision: bool, *, field: str) -> list[bool]`
- `write_trigger_trace(...)`
- `write_evolution_trigger_trace(...)`
- `compare_scalar(...)`
- `normalize_model_judge_output(...)`

这样 `write_trigger.py` 和 `evolution_trigger.py` 不会重复造一份逻辑。

### 4.2 写入侧 trigger 放在原文件继续扩展

继续写到：

- [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)

新增类建议：

- `OnInputWriteTrigger`
- `BoundaryEventWriteTrigger`
- `RuntimeEventWriteTrigger`
- `ScalarRuleWriteTrigger`
- `ModelJudgeWriteTrigger`

### 4.3 演化侧 trigger 放在原文件继续扩展

继续写到：

- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

新增类建议：

- `OnInputEvolutionTrigger`
- `BoundaryEventEvolutionTrigger`
- `RuntimeEventEvolutionTrigger`
- `ScalarRuleEvolutionTrigger`
- `ModelJudgeEvolutionTrigger`
- `PeriodicMaintenanceTrigger`
- `IdleMaintenanceTrigger`

### 4.4 Pipeline 如需支持“真正后台维护”，再最小增补入口

如果要支持**不依赖新 observation 的真实后台 maintenance**，建议扩展：

- [memprimitive/pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py)

新增一个 maintenance 入口，例如：

```python
def run_maintenance(self, *, event: str, metadata: dict[str, Any] | None = None) -> Packet:
    ...
```

但这是第二阶段工作，不是第一步必须做的最小改动。

## 5. 统一输入协议设计

为了和当前接口兼容，第一阶段建议统一从 `Observation.metadata["trigger"]` 取 structured payload。

建议格式：

```python
{
    "events": ["session_end", "task_failed"],
    "signals": {
        "importance": 0.82,
        "memory_pressure": 0.91,
        "surprise": 1.37,
    },
    "schedule": {
        "kind": "periodic",
        "tick": 12,
        "idle_seconds": 45.0,
    },
    "judge_input": {
        "task_result": "failed",
        "goal": "...",
    },
}
```

同时保留兼容入口：

- `packet.events`
- `unit.metadata`
- `store.metadata`
- `packet.trace`

推荐读取优先级：

1. `observation.metadata["trigger"]`
2. `packet.events`
3. `unit.metadata`
4. `store.metadata`
5. `packet.trace`

## 6. 各 trigger 家族的具体计划

---

## 6.1 OnInputTrigger

### 功能定位

覆盖 survey 里的 `PassThroughHook`。

语义：

- 只要有新输入进入当前 ingest，就触发
- 它不是复杂判定器，而是默认直通 trigger

### 代码设计

#### 写入侧

类：

- `OnInputWriteTrigger(WriteTriggerModule)`

位置：

- [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)

接口建议：

```python
def __init__(
    self,
    *,
    allowed_sources: tuple[str, ...] | None = None,
    require_non_empty_units: bool = True,
) -> None:
```

实现逻辑：

1. 验证 `packet.units`
2. 如果 `allowed_sources is None`，则所有 observation source 都允许
3. 若 `require_non_empty_units=True` 且 `units` 为空则报错
4. 返回 `[True] * len(units)`

输出：

- `packet.decisions`
- `trace["write_trigger"]`

#### 演化侧

类：

- `OnInputEvolutionTrigger(EvolutionTriggerModule)`

位置：

- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

接口建议：

```python
def __init__(
    self,
    *,
    mode: str = "mirror_write",
) -> None:
```

`mode` 建议先只支持两个值：

- `"mirror_write"`：直接复用 `packet.decisions`
- `"all_true"`：对所有对齐 unit 直接给 `True`

推荐默认：

- `mirror_write`

因为当前 evolution 多数是“新写入后接一个后处理”的语义，更贴近 survey 中的 `new-write conditioned` 降级为 hook 的建议。

### trace 设计

建议保留现有字段并增加：

- `family: "on_input"`
- `source: "observation"`
- `mode`

### 备注

`AlwaysWriteTrigger` 仍可保留，`OnInputWriteTrigger` 是更语义化的替代，不必强制删老类。

---

## 6.2 BoundaryEventTrigger

### 功能定位

覆盖 `StructuralBoundaryHook`。

典型事件：

- `turn_end`
- `session_end`
- `chunk_end`
- `stage_end`
- `subgoal_done`
- `episode_boundary`

### 代码设计

#### 写入侧

类：

- `BoundaryEventWriteTrigger(WriteTriggerModule)`

#### 演化侧

类：

- `BoundaryEventEvolutionTrigger(EvolutionTriggerModule)`

位置：

- 写入侧：[memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)
- 演化侧：[memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

共享接口建议：

```python
def __init__(
    self,
    *,
    accepted_events: tuple[str, ...],
    match_mode: str = "any",
    invert: bool = False,
    default_decision: bool = False,
) -> None:
```

实现逻辑：

1. 从 `observation.metadata["trigger"]["events"]` 读取事件列表
2. 若没有，再读 `packet.events`
3. 根据 `accepted_events` + `match_mode` 判定是否命中
4. 命中则对齐广播成决策 mask

`match_mode` 第一阶段建议只支持：

- `"any"`
- `"all"`

### 与已有接口的契合方式

这种 trigger 不负责自己做边界检测。

边界检测来源建议：

- 上游 caller 显式传 `Observation(..., metadata={"trigger": {"events": [...]}})`
- 或上游 `unit_formation / representation` 提前把边界标签写进 metadata

### trace 设计

新增：

- `family: "boundary_event"`
- `accepted_events`
- `observed_events`
- `matched`

### 实现边界

如果你希望 trigger 自己从自然语言内容里判断 “这是不是 session end / subgoal done”，那其实已经滑向 `ModelJudgeTrigger`，不该继续放在 `BoundaryEventTrigger` 里。

---

## 6.3 RuntimeEventTrigger

### 功能定位

覆盖 `RuntimeCallback`。

典型事件：

- `task_failed`
- `task_succeeded`
- `memory_pressure`
- `buffer_full`
- `interrupt_received`
- `warning`

### 代码设计

#### 写入侧

- `RuntimeEventWriteTrigger`

#### 演化侧

- `RuntimeEventEvolutionTrigger`

位置：

- [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)
- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

共享接口建议：

```python
def __init__(
    self,
    *,
    accepted_events: tuple[str, ...],
    match_mode: str = "any",
    default_decision: bool = False,
) -> None:
```

实现上和 `BoundaryEventTrigger` 很像，但 trace 要明确区分来源：

- `family: "runtime_event"`
- `event_source: "runtime"`

### 推荐实现细节

建议把 `BoundaryEventTrigger` / `RuntimeEventTrigger` 共享为一个内部 helper：

- `_event_gate_decision(..., family="boundary_event" | "runtime_event")`

这样只在 trace 名称与文档语义上分开。

### 与当前仓库的契合点

`ReflectionGenerationEvolution` 已经通过 `observation.metadata` 读取 runtime 信息，这证明“把运行结果写在 observation metadata 里再进入 pipeline”是当前代码风格可接受的。

相关实现可参考：

- [memprimitive/baselines/memory_evolution.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/memory_evolution.py)

---

## 6.4 ScalarRuleTrigger

### 功能定位

覆盖 `ExplicitScalarRule`。

典型信号：

- `importance`
- `surprise`
- `utility`
- `memory_pressure`
- `retention`
- `consolidation_strength`

### 代码设计

#### 写入侧

- `ScalarRuleWriteTrigger`

#### 演化侧

- `ScalarRuleEvolutionTrigger`

位置：

- [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)
- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

共享接口建议：

```python
def __init__(
    self,
    *,
    signal_key: str,
    threshold: float,
    comparator: str = ">=",
    signal_source: str = "auto",
    aggregate: str = "broadcast",
    missing_value: float | None = None,
) -> None:
```

建议支持的 `comparator`：

- `">"`
- `">="`
- `"<"`
- `"<="`
- `"=="`

建议支持的 `aggregate`：

- `"broadcast"`：取单个标量，广播到全部 unit
- `"per_unit"`：从每个 unit 自己取值
- `"any_unit"`：只要有一个 unit 命中，全部触发
- `"all_units"`：所有 unit 都命中才触发

### 信号读取规则

第一阶段建议：

- `signal_source="auto"` 时，按顺序找：
  - `observation.metadata["trigger"]["signals"][signal_key]`
  - `packet.trace["signals"][signal_key]`
  - `store.metadata["trigger_signals"][signal_key]`
  - `unit.metadata[signal_key]`

如果是 `aggregate="per_unit"`，则优先读：

- `unit.metadata[signal_key]`

### 与当前接口的契合方式

这是最容易落地的一类，因为它只是在当前固定 trigger slot 里计算布尔 mask，不需要改 pipeline 结构。

### trace 设计

新增：

- `family: "scalar_rule"`
- `signal_key`
- `signal_source`
- `comparator`
- `threshold`
- `resolved_values`

---

## 6.5 ModelJudgeTrigger

### 功能定位

覆盖 `LLMJudge`。

适合：

- importance / worthiness 判断
- route / label 判断
- boundary 判定
- “该不该写” / “该不该演化” 判定

### 代码设计

#### 写入侧

- `ModelJudgeWriteTrigger`

#### 演化侧

- `ModelJudgeEvolutionTrigger`

位置：

- [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)
- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

模型调用统一复用：

- [memprimitive/utils/_runtime.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/utils/_runtime.py)

### 接口建议

```python
def __init__(
    self,
    *,
    system_prompt: str,
    decision_mode: str = "bool",
    threshold: float = 0.5,
    per_unit: bool = True,
    strict_json: bool = True,
) -> None:
```

建议先支持三种 `decision_mode`：

- `"bool"`：模型直接输出 `true/false`
- `"score"`：模型输出 `score`，再与 `threshold` 比较
- `"label"`：模型输出标签，再映射为布尔

### 输入构造

第一阶段不建议做成高度可编程模板引擎，先固定成统一 payload：

```python
{
    "observation": ...,
    "unit": ...,
    "placement": ...,
    "store_summary": ...,
    "trigger_context": ...,
}
```

这样能最大限度复用已有 runtime JSON 调用模式，也更容易测试。

### 推荐实现细节

在共享 helper 里做：

- `build_model_judge_payload(packet, store, unit=None, placement=None)`
- `normalize_model_judge_output(raw, mode=...)`

触发类里只负责：

1. 组织调用范围
2. 得到 bool mask
3. 写 trace

### 与现有接口的契合点

仓库里已有多处严格 JSON 输出 + runtime helper 的模式，直接沿用即可。

### 难点

这是第一阶段成本最高的 trigger：

- 运行成本高
- 测试需要 mock runtime 或注入 callable
- prompt 设计如果写太死，会很快不够用

建议加一个可选注入参数：

```python
judge_callable: Callable[[dict[str, Any]], dict[str, Any] | bool | float] | None = None
```

用于测试和受控实验。

---

## 6.6 ScheduledMaintenanceTrigger -> Period / Idle

### 功能定位

覆盖 `BackgroundScheduler`，但必须分开实现：

- `PeriodicMaintenanceTrigger`
- `IdleMaintenanceTrigger`

我不建议再保留一个泛泛的 `ScheduledMaintenanceTrigger` 具体类；保留它作为文档层总称即可。

### 代码设计

位置：

- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)

注意：

- 这两个 trigger **只建议实现为 evolution trigger**
- 不建议实现为 write trigger

因为它们本质上是 maintenance / consolidation / rewrite / move，而不是“是否写入当前 observation”。

### A. PeriodicMaintenanceTrigger

接口建议：

```python
def __init__(
    self,
    *,
    every_n: int,
    counter_key: str = "ingest_count",
    decision_mode: str = "broadcast",
) -> None:
```

实现方法分两层：

#### 第一阶段：兼容现有 ingest 入口

从以下位置取计数器：

- `store.metadata[counter_key]`
- 或 `observation.metadata["trigger"]["schedule"]["tick"]`

判定：

- `current_tick % every_n == 0` 时触发

输出：

- `packet.evolution_decisions`

#### 第二阶段：支持真实 maintenance tick

需要给 `MemoryPipeline` 新增 maintenance 入口。

没有这个入口时，`PeriodicMaintenanceTrigger` 只能在“某次 ingest 顺便触发 maintenance”，不能表达真正独立后台周期任务。

### B. IdleMaintenanceTrigger

接口建议：

```python
def __init__(
    self,
    *,
    min_idle_seconds: float | None = None,
    accepted_events: tuple[str, ...] = ("idle", "sleep", "background_idle"),
    decision_mode: str = "broadcast",
) -> None:
```

第一阶段实现：

命中任一条件即触发：

1. `observation.metadata["trigger"]["events"]` 中出现 idle 事件
2. `observation.metadata["trigger"]["schedule"]["idle_seconds"] >= min_idle_seconds`
3. `packet.events` 包含 idle 类事件

### 与当前接口的真正冲突

这是本次计划里最需要明确向你汇报的难点：

#### 难点 1：当前 pipeline 没有“无新输入的后台运行入口”

`MemoryPipeline.ingest()` 必须从一个 `Observation` 出发。

因此：

- 现在可以实现“带调度条件的 evolution trigger”
- 但不能完整实现“真正脱离新 observation 的后台维护作业”

#### 难点 2：当前 evolution_trigger 仍绑定当轮 `units` / `placements`

当前 `evolution_trigger` 的典型输入是：

- `packet.units`
- `packet.placements`

它更像“对本轮参与 ingest 的 unit 再做一层额外演化判断”，而不是“扫描整个 store 选老记录做后台维护”。

所以真正的 idle / periodic maintenance，如果要作用于**历史 store 中的既有 records**，至少还需要二选一：

1. 新增 pipeline maintenance 入口
2. 引入专门的“maintenance unit synthesis”机制，把既有 records 映射成临时 units

在当前接口下，我建议优先选：

1. `MemoryPipeline.run_maintenance(...)`

而不是把它硬塞进普通 ingest。

## 7. 对现有文件的具体修改清单

### 必改

- [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py)
  - 新增 5 个写入侧 trigger
  - 更新 `BASELINE_CLASSES`

- [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py)
  - 新增 7 个演化侧 trigger
  - 更新 `BASELINE_CLASSES`

- [memprimitive/baselines/_trigger_common.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/_trigger_common.py)
  - 新增共享 helper

- [tests/test_baselines.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/tests/test_baselines.py)
  - 为每个新增 trigger 加最小单测

- [tests/test_pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/tests/test_pipeline.py)
  - 覆盖 slot 校验、默认行为不破坏、组合运行

### 第二阶段再改

- [memprimitive/pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py)
  - 如需要真实 maintenance 入口，再加 `run_maintenance`

- [memprimitive/core.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/core.py)
  - 如需要更强结构化 trigger payload，再补轻量字段

- [DSL_REFERENCE.zh-CN.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/DSL_REFERENCE.zh-CN.md)
  - 更新公开 trigger surface

- [TUTORIAL.zh-CN.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/TUTORIAL.zh-CN.md)
  - 增加事件触发、标量触发、模型判定触发示例

## 8. 测试计划

### 8.1 单元测试

每个 trigger 都至少覆盖：

- 输入缺失时报错
- 命中时输出全 `True`
- 不命中时输出全 `False`
- trace 字段正确
- 与 units 对齐

### 8.2 兼容性测试

必须保证以下不破坏：

- `MemoryPipeline()` 默认仍然使用老默认模块
- `AlwaysWriteTrigger` / `NeverEvolutionTrigger` 仍工作
- `create_baseline_pipeline()` 行为不变

### 8.3 集成测试

建议新增组合场景：

1. `BoundaryEventEvolutionTrigger + SummaryRewriteEvolution`
2. `RuntimeEventEvolutionTrigger + ReflectionGenerationEvolution`
3. `ScalarRuleWriteTrigger + AppendOrganization`
4. `ModelJudgeWriteTrigger + AppendOrganization`

## 9. 实施顺序

推荐按下面顺序做，风险最低：

1. `OnInputTrigger`
2. `BoundaryEventTrigger`
3. `RuntimeEventTrigger`
4. `ScalarRuleTrigger`
5. `ModelJudgeTrigger`
6. `PeriodicMaintenanceTrigger`
7. `IdleMaintenanceTrigger`
8. 如有需要，再补 `run_maintenance(...)`

原因：

- 前四类完全能贴合当前 pipeline 结构
- `ModelJudgeTrigger` 依赖 runtime，但接口仍清晰
- `Periodic / Idle` 是唯一会逼近 pipeline 入口设计的问题

## 10. 我认为现在做不到或需要你确认的点

### 10.1 “精确同名单类”版本做不到

如果你的要求是：**一个类名就叫 `OnInputTrigger`，并且它既能放 `write_trigger` 又能放 `evolution_trigger`**，那在当前 `MemoryPipeline` 类型校验规则下做不到。

可行替代：

- 使用 `OnInputWriteTrigger` / `OnInputEvolutionTrigger`
- 或用 builder/factory

### 10.2 真正的后台 ScheduledMaintenance 目前做不完整

如果你的目标是：

- 没有新 observation
- 仅因为时间到了 / 空闲了
- 就扫描历史 store 并做 consolidation

那当前接口不完整，必须补一个 maintenance entrypoint。

### 10.3 Boundary / Runtime 事件的“检测器”不应塞进对应 trigger

如果你希望：

- `BoundaryEventTrigger` 自己从文本里判断 subgoal done
- `RuntimeEventTrigger` 自己判断某次任务算不算失败

那这两类会迅速和 `ModelJudgeTrigger` 混在一起，接口会重新膨胀。

我建议：

- 事件检测与事件响应分层

## 11. 最终建议

按当前仓库形状，最稳妥的 trigger 重写路线是：

1. 保留 `write_trigger` / `evolution_trigger` 两 slot 结构
2. 用 slot-specific concrete class 落地六大 trigger 家族
3. 用共享 helper 统一事件读取、标量比较、模型判定和 trace
4. 把 `ScheduledMaintenanceTrigger` 明确拆成 `PeriodicMaintenanceTrigger` 与 `IdleMaintenanceTrigger`
5. 先做“兼容当前 ingest 入口”的版本
6. 如果你确认需要真实后台 maintenance，再单独补 `MemoryPipeline.run_maintenance(...)`

这条路线和当前代码最契合，也最不容易引入一轮新的 trigger 过度设计。
