# Trigger 重写实现计划的补充

## 10. 当前最需要明确的结论

### 10.1 旧计划里的“双文件 + 双类名”假设应废弃

不再推荐：

- `OnInputWriteTrigger`
- `OnInputEvolutionTrigger`

现在更符合 repo 现状的是：

- `OnInputTrigger(slot="write_trigger")`
- `OnInputTrigger(slot="evolution_trigger")`

同理适用于其他 trigger 家族。

### 10.2 `packet.evolution_decisions` 不是当前方向

当前稳定 contract 是：

- 单一 `packet.decisions`
- `trace["write_trigger"]` 保留写入侧
- `trace["evolution_trigger"]` 保留演化侧

后续 trigger 重写不应再引入额外的平行 decision 字段。

### 10.3 真正后台 maintenance 仍然做不完整

如果目标是：

- 没有新 observation
- 只因时间到了/空闲了
- 就扫描历史 store 并执行 maintenance

那么当前接口仍不完整，需要后续新增专门入口。

## 11. 最终建议

按 repo 当前状态，最稳妥的 trigger 重写路线是：

1. 保留 `write_trigger` / `evolution_trigger` 两个 slot
2. 保留统一 `TriggerModule` 和单文件 `trigger.py` 设计
3. 继续使用“同一 trigger 类 + `slot` 参数”的公开 API
4. 在现有 `_BaseTrigger` 上逐步新增 richer trigger 家族
5. 保持 `packet.decisions` 与双 trace 键不变
6. 先做“兼容普通 ingest 的调度/事件/规则触发”
7. 只有在你确认需要真正后台 maintenance 时，再单独扩 pipeline 入口

这条路线比旧版计划更符合 repo 现状，也能避免把 trigger 设计又推回到一轮新的过度抽象。
