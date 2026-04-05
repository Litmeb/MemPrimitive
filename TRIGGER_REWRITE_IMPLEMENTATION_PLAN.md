# Trigger 重写实现计划

更新时间：2026-04-01

## 1. 目标

基于 [TRIGGERS.md](./TRIGGERS.md) 与 [TRIGGER_SCORE_BOOLEAN_GATE_SURVEY.md](./TRIGGER_SCORE_BOOLEAN_GATE_SURVEY.md) 的结论，给 `MemPrimitive` 制定一份**和当前仓库实现一致**的 trigger 重写计划。

这份计划的前提已经和旧版本不同：

- `write_trigger` / `evolution_trigger` 在代码里都统一为 `TriggerModule`
- baseline trigger 已经统一收敛到单文件：
  - [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)
- 当前公开 trigger API 已经是“**同一 trigger 类 + `slot` 参数切换写入/演化槽位**”的形状

所以，本计划不再建议拆成：

- `OnInputWriteTrigger`
- `OnInputEvolutionTrigger`

而是曾建议统一为：

- `AlwaysTrigger(slot="write_trigger")`
- `NeverTrigger(slot="evolution_trigger")`
- 其余 richer trigger 继续统一采用 `同一类 + slot` 的模式

同理适用于：

- `BoundaryEventTrigger`
- `RuntimeEventTrigger`
- `ScalarRuleTrigger`
- `LLMJudgeTrigger`
- `PeriodicMaintenanceTrigger`
- `IdleMaintenanceTrigger`

## 2. repo 现状约束

### 2.1 Pipeline 仍然只有两个 trigger slot

当前 `MemoryPipeline` ingest 顺序仍是：

```text
unit_formation
-> representation
-> write_trigger
-> organization
-> evolution_trigger
-> memory_evolution
```

对应实现：

- [memprimitive/pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py)

因此 trigger 重写的目标不是增加新 slot，而是在现有两个 slot 上扩展更多 trigger 家族。

### 2.2 但两个 slot 现在都只要求 `TriggerModule`

当前 `MemoryPipeline` 校验规则是：

- `write_trigger` 必须是 `TriggerModule` 且 `module.spec.slot == "write_trigger"`
- `evolution_trigger` 必须是 `TriggerModule` 且 `module.spec.slot == "evolution_trigger"`

这意味着：

- 一个 trigger 基类/实现体系完全可以统一
- 只需要在实例化时设置正确的 `slot`

也就是说，**“一个类不能同时兼容两个 slot”这条旧约束已经不成立了**。现在真正的约束只有：

- 同一个实例只能有一个 `spec.slot`
- 但同一个类可以通过 `slot=` 参数生成 write/evolution 两种实例

### 2.3 baseline trigger 已经统一成单文件模式

当前 baseline trigger 实现位于：

- [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)

已存在模式：

- `_BaseTrigger`
- `AlwaysTrigger`
- `NeverTrigger`

其核心设计已经说明了 repo 的真实方向：

1. 一个共享 trigger 基类
2. 通过 `slot` 参数切换写入侧 / 演化侧
3. 统一写入：
   - `packet.decisions`
   - `trace["write_trigger"]` 或 `trace["evolution_trigger"]`

所以后续扩展应沿用这条路径，而不是重新拆回 `write_trigger.py` / `evolution_trigger.py` 双文件。

### 2.4 下游仍然绑定 `packet.decisions`

和旧计划相比，这里也要改口径。

当前 trigger 层稳定输出是：

- `packet.decisions: list[bool]`

写入侧 decision 会保留在：

- `trace["write_trigger"]`

之后 `evolution_trigger` 会覆盖 `packet.decisions`，供 `memory_evolution` 使用，并把本轮演化侧 decision 写到：

- `trace["evolution_trigger"]`

这和 [memprimitive/core.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/core.py) 里的注释一致：

- `write_trigger` 先写 `packet.decisions`
- `evolution_trigger` 之后可能覆写它
- 写入侧决策通过 `trace["write_trigger"]` 保留

因此，旧计划里提到的：

- `packet.evolution_decisions`

不再符合 repo 现状。本次重写计划必须统一按：

- 单一 `packet.decisions`
- 双 trace 键保留写入/演化两次结果

### 2.5 公开注册和默认工厂也已经统一

当前 registry 逻辑在：

- [memprimitive/baselines/registry.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/registry.py)

现状是：

- `write_trigger` 默认工厂会对 trigger 类注入 `slot="write_trigger"`
- `evolution_trigger` 默认工厂会对 trigger 类注入 `slot="evolution_trigger"`

这意味着只要新 trigger 类遵守现有构造模式，baseline registry 几乎不需要额外抽象层。

## 3. 重写总原则

### 3.1 保持单文件 trigger surface

新 trigger 家族继续收敛到：

- [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)

不再拆成：

- `write_trigger.py`
- `evolution_trigger.py`

### 3.2 保持“同类 + slot 参数”模式

推荐新增 trigger 一律采用：

```python
SomeTrigger(slot="write_trigger")
SomeTrigger(slot="evolution_trigger")
```

而不是维护两套 slot-specific concrete class。

### 3.3 保持下游 contract 不变

不改下面这些稳定约定：

- `packet.decisions`
- `trace["write_trigger"]`
- `trace["evolution_trigger"]`

也就是说，新 trigger 只扩展决策来源，不改 trigger 到 organization / memory_evolution 的耦合面。

### 3.4 保持 trigger 轻量，不回到厚 trigger framework

不恢复已移除的那套：

- `signal / scorer / gate / policy`
- compose-style trigger builders

新 trigger 仍应是**代码形状直接、以当前 packet/store 接口为中心**的 baseline 实现。

### 3.5 事件检测与事件响应分离

仍然坚持：

- `BoundaryEventTrigger` 处理“事件是否命中”
- `RuntimeEventTrigger` 处理“运行时事件是否命中”
- `LLMJudgeTrigger` 才负责主动调用模型判定

不要把自然语言边界识别、task failure 识别等复杂检测逻辑，偷偷塞进 event trigger 本体。

## 4. 推荐代码落点

### 4.1 主实现仍在 `trigger.py`

主要改动文件：

- [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)

建议在现有 `_BaseTrigger` 基础上继续扩展，而不是另起一套新基类。

### 4.2 如需共享辅助逻辑，再新增轻量 helper

如果 `trigger.py` 变得过长，可以新增：

- [memprimitive/baselines/_trigger_common.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/_trigger_common.py)

但这里的定位要比旧计划更收敛，只放共享函数，例如：

- `extract_trigger_metadata(packet, store) -> dict[str, Any]`
- `resolve_trigger_events(packet) -> list[str]`
- `resolve_trigger_signals(packet, store) -> dict[str, Any]`
- `broadcast_decisions(packet, decision: bool) -> list[bool]`
- `write_trigger_trace(...)`
- `compare_scalar(...)`
- `normalize_model_judge_output(...)`

如果逻辑不多，也可以先全部留在 `trigger.py`，不强制拆 helper 文件。

### 4.3 registry 需要同步扩充

相关文件：

- [memprimitive/baselines/registry.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/registry.py)

当新增 trigger 类进入 baseline surface 后，需要同步更新：

- `write_trigger` 候选类列表
- `evolution_trigger` 候选类列表

前提是这些类都支持 `slot=...` 构造。

## 5. 统一输入协议设计

为了与当前 runtime 契合，第一阶段建议优先从：

- `packet.observation.metadata["trigger"]`

读取结构化 trigger payload。

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

兼容读取入口可以保留：

- `packet.events`
- `unit.metadata`
- `store.metadata`
- `packet.trace`

推荐优先级：

1. `packet.observation.metadata["trigger"]`
2. `packet.events`
3. `unit.metadata`
4. `store.metadata`
5. `packet.trace`

## 6. 各 trigger 家族的更新计划

## 6.1 OnInputTrigger（历史方案，已移除）

### 功能定位

原本用于覆盖 survey 中的 `PassThroughHook`。

语义：

- 只要当前 ingest 有输入并且通过当前 slot 的输入校验，就触发
- 但该默认语义与当前 baseline 中的 `AlwaysTrigger` 重复，因此已从代码中删除

### 实现建议

历史类名：

- `OnInputTrigger`

位置：

- [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)

历史接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "write_trigger",
    allowed_sources: tuple[str, ...] | None = None,
    require_non_empty_units: bool = True,
) -> None:
```

行为：

1. 复用 `_BaseTrigger` 的 slot 校验
2. 复用现有 unit / placement 校验
3. 若 source 命中且输入有效，则广播 `True`
4. 写对应 slot 的 trace

备注：

- `AlwaysTrigger` 继续保留为最小 baseline
- `OnInputTrigger` 已被证实主要是语义别名，因此不再保留为当前 API

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

### 实现建议

类名：

- `BoundaryEventTrigger`

接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "write_trigger",
    accepted_events: tuple[str, ...],
    match_mode: str = "any",
    invert: bool = False,
    default_decision: bool = False,
) -> None:
```

行为：

1. 从 `observation.metadata["trigger"]["events"]` 读取事件
2. 缺失时再读 `packet.events`
3. 根据 `accepted_events` 和 `match_mode` 判断命中
4. 将结果广播为当前 slot 的决策 mask

`match_mode` 第一阶段建议只支持：

- `"any"`
- `"all"`

### 设计边界

它只处理“外部已经给出的边界事件”，不负责从原始文本里自动识别 session end / subgoal done。

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

### 实现建议

类名：

- `RuntimeEventTrigger`

接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "evolution_trigger",
    accepted_events: tuple[str, ...],
    match_mode: str = "any",
    default_decision: bool = False,
) -> None:
```

实现上可以与 `BoundaryEventTrigger` 共用一套内部 event gate helper，只在：

- trace family
- 文档语义

上做区分。

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

### 实现建议

类名：

- `ScalarRuleTrigger`

接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "write_trigger",
    signal_key: str,
    threshold: float,
    comparator: str = ">=",
    signal_source: str = "auto",
    aggregate: str = "broadcast",
    missing_value: float | None = None,
) -> None:
```

建议支持：

- `comparator`: `>`, `>=`, `<`, `<=`, `==`
- `aggregate`: `broadcast`, `per_unit`, `any_unit`, `all_units`

信号解析建议：

- `observation.metadata["trigger"]["signals"][signal_key]`
- `packet.trace["signals"][signal_key]`
- `store.metadata["trigger_signals"][signal_key]`
- `unit.metadata[signal_key]`

这类 trigger 和当前 repo 最契合，因为它天然就是在当前 slot 上输出布尔 mask。

## 6.5 LLMJudgeTrigger

### 功能定位

覆盖 `LLMJudge`。

适用于：

- importance / worthiness 判定
- type routing
- boundary 判定
- “该不该写”
- “该不该演化”

### 实现建议

类名：

- `LLMJudgeTrigger`

位置仍是：

- [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)

模型调用统一复用：

- [memprimitive/utils/_runtime.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/utils/_runtime.py)

接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "write_trigger",
    system_prompt: str,
    decision_mode: str = "bool",
    threshold: float = 0.5,
    per_unit: bool = True,
    strict_json: bool = True,
    judge_callable: Callable[[dict[str, Any]], dict[str, Any] | bool | float] | None = None,
) -> None:
```

建议先支持：

- `"bool"`
- `"score"`
- `"label"`

输入 payload 建议固定，不做复杂模板 DSL：

```python
{
    "observation": ...,
    "unit": ...,
    "placement": ...,
    "store_summary": ...,
    "trigger_context": ...,
}
```

### 备注

这是第一阶段成本最高的 trigger，建议在 `judge_callable` 注入点上优先做好可测性，禁止为了“稳定”退回规则模拟。

## 6.6 PeriodicMaintenanceTrigger / IdleMaintenanceTrigger

### 功能定位

覆盖旧 survey 中的 `BackgroundScheduler`，但实现上直接拆成两个具体类：

- `PeriodicMaintenanceTrigger`
- `IdleMaintenanceTrigger`

不建议再做一个泛化的 `ScheduledMaintenanceTrigger` 具体实现。

### 设计建议

这两个类都继续沿用统一 trigger 模式：

- 仍然是 `TriggerModule`
- 仍然在 `trigger.py`
- 仍然通过 `slot="evolution_trigger"` 使用

不建议把它们作为 write-side 默认候选。

### A. PeriodicMaintenanceTrigger

接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "evolution_trigger",
    every_n: int,
    counter_key: str = "ingest_count",
    decision_mode: str = "broadcast",
) -> None:
```

第一阶段判定来源：

- `store.metadata[counter_key]`
- 或 `observation.metadata["trigger"]["schedule"]["tick"]`

命中规则：

- `current_tick % every_n == 0`

### B. IdleMaintenanceTrigger

接口建议：

```python
def __init__(
    self,
    *,
    slot: str = "evolution_trigger",
    min_idle_seconds: float | None = None,
    accepted_events: tuple[str, ...] = ("idle", "sleep", "background_idle"),
    decision_mode: str = "broadcast",
) -> None:
```

第一阶段命中任一条件即可：

1. `observation.metadata["trigger"]["events"]` 中出现 idle 事件
2. `observation.metadata["trigger"]["schedule"]["idle_seconds"] >= min_idle_seconds`
3. `packet.events` 包含 idle 类事件

### 当前 repo 下的真实限制

这一点和旧计划一致，但要用新接口表述：

- 当前 pipeline 仍然没有“无 observation 的独立 maintenance 入口”
- 因此现在能实现的是“在普通 ingest 中附带调度条件的 evolution trigger”
- 还不能完整表达真正后台异步 maintenance job

所以这两类 trigger 的第一阶段目标应是：

- 先支持“条件触发的 evolution gate”

而不是假装已经支持真正后台维护。

## 7. 对现有文件的具体修改清单

### 必改

- [memprimitive/baselines/trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/trigger.py)
  - 在现有 `_BaseTrigger` 之上新增 richer trigger 家族
  - 保持 `slot=` 驱动 write/evolution 两侧
  - 保持 trace 结构与当前 baseline 兼容

- [memprimitive/baselines/registry.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/registry.py)
  - 将新 trigger 类纳入 `write_trigger` / `evolution_trigger` baseline 列表
  - 继续使用现有 `slot=` 注入模式

- [tests/test_baselines_triggers.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/tests/test_baselines_triggers.py)（以及同目录下其他 `tests/test_baselines_*.py` 拆分文件）
  - 为每个新增 trigger 增加最小单测
  - 覆盖 write/evolution 两个 slot 实例化分支

- [tests/test_pipeline_core.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/tests/test_pipeline_core.py)、[tests/test_pipeline_triggers_dispatch.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/tests/test_pipeline_triggers_dispatch.py)
  - 覆盖统一 trigger 类在两个 slot 的组合行为
  - 保证默认 pipeline 行为不变

### 可选

- [memprimitive/baselines/_trigger_common.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/_trigger_common.py)
  - 当 `trigger.py` 共享逻辑过多时再抽出

### 第二阶段再改

- [memprimitive/pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py)
  - 如确实需要真实后台 maintenance，再考虑新增 `run_maintenance(...)`

- [memprimitive/core.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/core.py)
  - 如确实需要更强结构化 trigger payload，再增加轻量字段

- [DSL_REFERENCE.zh-CN.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/DSL_REFERENCE.zh-CN.md)
  - 更新公开 trigger surface

- [TUTORIAL.zh-CN.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/TUTORIAL.zh-CN.md)
  - 增加事件触发、标量触发、模型判定触发示例

## 8. 测试计划

### 8.1 单元测试

每个新增 trigger 至少覆盖：

- `slot="write_trigger"` 可运行
- `slot="evolution_trigger"` 可运行
- 输入缺失时报错
- 命中时输出正确 decisions
- 不命中时输出正确 decisions
- 对应 trace key 正确

### 8.2 兼容性测试

必须保证以下不破坏：

- `MemoryPipeline()` 默认仍然使用 `AlwaysTrigger()` + `NeverTrigger()`
- `create_baseline_pipeline()` 行为不变

### 8.3 集成测试

建议优先补这几组：

1. `BoundaryEventTrigger(slot="evolution_trigger") + SummaryRewriteEvolution`
2. `RuntimeEventTrigger(slot="evolution_trigger") + ReflectionGenerationEvolution`
3. `ScalarRuleTrigger(slot="write_trigger") + AppendOrganization`
4. `LLMJudgeTrigger(slot="write_trigger") + AppendOrganization`
