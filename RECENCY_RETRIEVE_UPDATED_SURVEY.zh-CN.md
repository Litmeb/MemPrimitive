# 检索时更新的 `recency` / 记忆强化机制调研

更新时间：2026-04-03

## 这份文档关注什么

这里讨论的不是“写入得越晚越新”的普通时间戳，而是更严格的一类量：

- 它参与 retrieval / recall 排序或保留概率计算；
- 它会在一次 successful retrieve / recall 之后被刷新、重置或强化；
- 因而它本质上更接近 `last-accessed`、`memory strength`、`consolidation state`，而不只是 `created_at`。

下面先精读你指定的三篇工作，再补充其他严格属于这一家族的代表。

---

## 1. 你指定的三篇论文

## 1.1 Generative Agents

- 论文：Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023
- 链接：
  - https://arxiv.org/abs/2304.03442
  - https://dl.acm.org/doi/10.1145/3586183.3606763

### 它的 `recency` 究竟是什么

Generative Agents 的 `recency` 不是“离写入时间有多近”，而是：

- 一个 memory 距离 **上一次被 retrieve** 过去了多久；
- 论文原文写的是 “the number of sandbox game hours since the memory was last retrieved”。

所以它是典型的 `last-retrieved` 型 recency。

### 具体怎么算

论文给出两条关键信息：

- `recency` 用 exponential decay；
- decay factor 是 `0.995`。

论文没有把这一项单独写成完整显式等式，但按文中的表述，最自然的还原是：

```text
recency_raw(i, t) = 0.995 ^ Δh_i
```

其中：

- `Δh_i` = 当前时刻到 memory `i` 上一次被 retrieve 的 sandbox hour 差值

这是根据论文描述做的等式化还原；论文明确给了“指数衰减 + 因子 0.995 + 自上次 retrieve 以来的小时数”，但没有把这一行直接排版成公式。

### 它如何进入最终 retrieval score

论文明确写出最终检索分数：

```text
score = α_recency * recency + α_importance * importance + α_relevance * relevance
```

并且：

- 三项先各自做 min-max normalization 到 `[0, 1]`
- 实现里 `α_recency = α_importance = α_relevance = 1`

### retrieve 之后怎么更新

因为 `recency` 依赖的是 “since last retrieved”：

- 一次 memory 被 retrieve 之后，它的 `last_retrieved_time` 会被刷新到当前时刻；
- 于是下一轮计算时，这条 memory 的 `Δh` 接近 `0`；
- 因而它的 `recency_raw` 会回到接近最大值。

### 这篇工作的使用场景

它的 `recency` 是一个 **attention bias**：

- 刚刚被想起过的事，短期内更容易再次被想起；
- 适合模拟 agent 当前 attentional sphere 里“脑中还热着”的记忆。

### 一句话总结

Generative Agents 用的是：

- `retrieve -> 刷新 last_retrieved -> recency 重新变大`

这是一种典型的 **访问时间 recency**。

---

## 1.2 MemoryBank

- 论文：Zhong et al., *MemoryBank: Enhancing Large Language Models with Long-Term Memory*, 2023
- 链接：
  - https://arxiv.org/abs/2305.10250

### 它的 `recency` 更准确地说是什么

MemoryBank 其实不把这个量命名为 `recency score`，而是把它建模成：

- `t`：距离上次学习/回忆过去的时间
- `S`：memory strength

所以它更像：

- `elapsed-time since last recall`
- 加上 `recall` 驱动的 strength reinforcement

这比 Generative Agents 的 `last_retrieved` 更“认知化”。

### 具体怎么算

论文直接采用 Ebbinghaus forgetting curve 的简化版：

```text
R = e^(-t / S)
```

其中：

- `R`：memory retention
- `t`：自学习该记忆以来经过的时间
- `S`：memory strength

### retrieve / recall 之后怎么更新

这是这篇论文最核心的点。论文原文非常明确：

```text
初次提及时：S = 1
每次该 memory 在对话中被 recalled 时：
S <- S + 1
t <- 0
```

这意味着：

- recall 会让强度 `S` 增加
- 同时把 elapsed time `t` 归零

因此同一条记忆被反复 retrieve 之后，会：

- retention 衰减更慢；
- 更不容易被忘掉。

### 它如何用于系统

MemoryBank 的 retrieval 本身主要是 dense retrieval / vector retrieval；
而这个“检索时更新”的量主要用于 **memory updating / forgetting**：

- 老、弱、且长期未被 recall 的 memory 更容易衰减；
- 被多次 recall 的 memory 会被“加固”。

### 这篇工作的使用场景

它适合：

- 长时对话；
- AI companion；
- persona / relationship memory；
- 希望 memory 会“自然遗忘，但被反复提起的事更牢固”的系统。

### 一句话总结

MemoryBank 用的是：

- `retrieve -> S + 1, t = 0 -> 遗忘曲线变平`

这是一种典型的 **retrieve-driven strength update**。

---

## 1.3 “My agent understands me better”

- 论文：Bae et al., *“My agent understands me better”: Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents*, CHI EA 2024
- 链接：
  - https://arxiv.org/abs/2404.00573
  - https://dl.acm.org/doi/10.1145/3613905.3650903

### 它的 `recency` 更准确地说是什么

这篇工作也不直接把核心变量命名为 `recency`，但如果按你给的定义，它确实属于同一家族。

它的核心不是单纯 “last retrieved”，而是：

- `t`：距离上次 recall 的 elapsed time
- `n`：被 recall 的次数
- `g_n` / `a`：随 recall 次数变化的 consolidation state / decay state

所以它可以理解成：

- 一个由 “时间 + 相关性 + recall history” 共同决定的动态 recency / consolidation 量。

### 基础 recall probability

论文先从指数衰减模型出发，给出：

```text
r(t) = μ e^(-a t)
```

并在 `b = 1` 的场景下写成 recall probability：

```text
p(t) = 1 - exp(-μ e^(-a t))
```

在实际 agent 模型里，它再把 `μ` 替换成和 query 相关的 relevance `r`：

```text
p(t) = 1 - exp(-r e^(-a t))
```

其中 relevance 用 cosine similarity：

```text
r = (a · b) / (||a|| ||b||)
```

### retrieve / recall 之后怎么更新

这篇论文真正关键的是：`a` 不是常数，而是由 recall history 更新。

论文定义：

```text
a_n = 1 / g_n,   g_0 = 1
g_n = g_(n-1) + S(t)
S(t) = (1 - e^(-t)) / (1 + e^(-t))
```

并给出归一化后的最终公式：

```text
p_n(t) = [1 - exp(-r e^(-t / g_n))] / (1 - e^(-1))
```

### 这套更新直觉上是什么意思

每次 recall 之后：

- recall 次数 `n` 增加；
- `g_n` 增大；
- 所以 `a_n = 1 / g_n` 变小；
- 衰减变慢；
- 该记忆之后的 recall probability 降得没那么快。

并且它不是简单做 `t <- 0, S <- S + 1`，而是把：

- recall 的次数
- recall 之间的间隔长度

都编码进了 `g_n`。

因此它比 MemoryBank 更细：

- 间隔太短的重复提取，强化有限；
- 经过一段时间再被 recall，会带来更明显的 consolidation 增益。

### 它如何用于系统

这篇工作用它来做：

- dialogue event recall；
- conversation history 中事件的优先回忆；
- 通过 repeated recall 形成更稳定的 user pattern / preference memory。

### 一句话总结

这篇工作用的是：

- `retrieve -> 更新 recall count 与 g_n -> 未来衰减更慢`

这是典型的 **retrieve-driven consolidation recency**。

---

## 1.4 三篇工作的并排比较

| 工作 | 被更新的量 | retrieve 后怎么变 | 它最终影响什么 | 更像哪一类 recency |
|---|---|---|---|---|
| Generative Agents | `last_retrieved_time` | 刷新为当前时刻 | retrieval score 中的 `recency` 项 | access recency |
| MemoryBank | `S` 与 `t` | `S <- S + 1`, `t <- 0` | retention / forgetting probability | strength-updated recency |
| My agent understands me better | `g_n` / `a_n` 与 elapsed time 基准 | recall 后 `n` 增加，`g_n` 增大，衰减减缓 | recall probability / consolidation | consolidation-aware recency |

一个很重要的结论是：

- **Generative Agents** 的 recency 最“工程化”，就是 last-accessed decay；
- **MemoryBank** 开始把 retrieve 视为强化；
- **My agent understands me better** 则把 retrieve history 本身做成了 consolidation dynamics。

也就是说，这三篇其实构成了一个很清楚的演化链：

```text
last-access timestamp
-> strength reinforced by recall
-> consolidation state updated by recall frequency + recall interval
```

---

## 2. 其他使用这类 “retrieve 时更新” 机制的工作

## 2.1 一个先说清楚的结论

我额外查了一轮后，结论是：

- 在 LLM agent / agent memory 论文里，
- **严格显式** 把某个 recency-like 量写成“每次 retrieve / recall 时会刷新或强化”的工作并不多；
- 真正最典型、最清楚的，确实就是你点名的这三篇。

很多其他工作会用：

- timestamp
- time decay
- recency feature
- session boundary

但它们往往只是：

- 按写入时间衰减；
- 或把时间当 metadata filter；
- 并不会在 retrieve 时显式更新该量。

所以如果严格按你的定义筛，额外最值得补的其实是 **认知建模祖先工作**，因为后面的 agent 论文很多就是在沿着这条线改写。

---

## 2.2 ACT-R 的 base-level learning

- 论文：Anderson et al., *An Integrated Theory of the Mind*, 2004
- 链接：
  - https://www.cs.utexas.edu/~dana/ACT-R.pdf
  - ACT-R 官方站点：https://act-r.psy.cmu.edu/

### 为什么它符合你的定义

ACT-R 里的 declarative memory chunk 有一个 `base-level activation`：

- 它由这个 chunk 过去被使用/练习的全部历史共同决定；
- 每一次新的 retrieval / practice 都会新增一项贡献；
- 因而 memory 一旦被再次取用，它的可得性会立刻上升。

这和你说的“retrieve 到时更新的值”是同一个家族，而且是非常经典的祖先。

### 具体怎么算

论文给出的 base-level learning equation 是：

```text
B_i = ln( Σ_j t_j^(-d) )
```

其中：

- `t_j`：距离第 `j` 次 practice / use / retrieval 已经过了多久
- `d`：decay 参数，ACT-R 社区常用默认值约为 `0.5`

### retrieve 之后怎么更新

一旦某条 memory 再次被 retrieve / practice：

- 历史里会新增一个新的使用时刻；
- 对应于求和里多出一个新的、非常“新鲜”的项；
- 于是 `B_i` 立刻变大；
- retrieval probability 升高，latency 降低。

论文还给出：

```text
P_i = 1 / (1 + e^(-(A_i - τ)/s))
T_i = F e^(-A_i)
```

所以 activation 上升会同时提升可取回概率、降低检索延迟。

### 它和前三篇的关系

如果从机制上看：

- Generative Agents 很像只保留了“最近一次访问”这一维的简化版；
- MemoryBank 很像把“retrieve 会强化记忆”压缩成 `S <- S + 1, t <- 0`；
- My agent understands me better 则更接近 ACT-R 这类“由完整 retrieval history 决定可回忆性”的思路。

### 一句话总结

ACT-R 用的是：

- `retrieve -> 在历史上增加一次 use -> activation 变大`

这是这条技术路线最经典、最系统的前身。

---

## 2.3 Rational Analysis of Memory / Anderson-Schooler 系谱

- 来源：
  - Anderson and Schooler 1991 被 ACT-R 明确列为 base-level equation 的理论来源
  - 可参见上面 ACT-R 论文对该来源的说明
  - 综述入口：https://docslib.org/doc/209653/the-adaptive-nature-of-memory

### 为什么它也值得单列

虽然它不是 LLM agent 论文，但它提出的核心观点和你关心的 recency 非常贴近：

- 一个 item 未来被需要的概率，与它过去出现/被使用的历史有关；
- 这种历史不是单纯 frequency，而是 **frequency + recency + context**；
- 因而“最近刚被用过的记忆更可能再次被需要”不是拍脑袋规则，而是对环境统计的适应。

### 它的算法形态

这一系谱最有操作性的落地就是上面的 ACT-R `base-level learning equation`。
所以如果你是为了 `MemPrimitive` 里的 primitive 设计，最实用的理解方式是：

```text
retrieve history
-> 估计未来再次需要它的概率
-> 把这个概率映射为 activation / recall score
```

### 为什么它重要

它解释了为什么这类机制会不断在后来的 agent memory 里反复出现：

- 因为它不是简单 heuristic；
- 它背后是“过去被用过的时间模式，可以预测未来还会不会用到”的统计假设。

### 一句话总结

这条系谱提供的是：

- **为什么要让 retrieve 更新 recency/activation**

而 ACT-R 提供的是：

- **怎样把它写成可计算公式**

---

## 3. 对 `MemPrimitive` 建模最有用的抽象

如果只看机制，而不看论文名字，这类 “retrieve 时更新” 的 recency 大致可以抽成三种 primitive：

## 3.1 Access-recency

形式：

```text
state = last_access_time
score = decay(now - last_access_time)
on_retrieve: last_access_time <- now
```

代表：

- Generative Agents

适合：

- 短期 attentional bias
- 最近刚提过的事更容易再次被提起

---

## 3.2 Strength-recency

形式：

```text
state = (strength S, elapsed_time t)
score = retention(t, S)
on_retrieve:
  S <- S + 1
  t <- 0
```

代表：

- MemoryBank

适合：

- 自然遗忘
- 长时 persona / relationship memory

---

## 3.3 Consolidation-recency

形式：

```text
state = recall_history or consolidation_state
score = f(relevance, elapsed_time, recall_count, recall_interval)
on_retrieve:
  update consolidation_state using recall interval/history
```

代表：

- My agent understands me better
- ACT-R family

适合：

- 想把 “隔一段时间再想起一次，会更牢” 也显式建模进去的系统

---

## 4. 对你当前问题的直接结论

### 结论 1

这三篇论文虽然都可被你归到 “retrieve 时更新的 recency” 家族，但它们并不是同一个算法：

- Generative Agents：`last retrieved time`
- MemoryBank：`strength + elapsed time`
- My agent understands me better：`consolidation state driven by recall frequency and interval`

### 结论 2

如果你后面要在 `MemPrimitive` 里做 primitive 设计，不建议只放一个叫 `Recency` 的统一模块，因为它至少分成三类：

- `AccessRecency`
- `StrengthRecency`
- `ConsolidationRecency`

### 结论 3

在 agent memory 文献里，**严格** 符合“retrieve 后更新”的显式 work 并不多。
如果要继续扩展这条线，最值得追的是：

- ACT-R / rational analysis of memory 这条认知建模祖先线；
- 以及后续是否有 personalized dialogue / companion agent 论文继续沿 `MemoryBank -> consolidation-aware recall` 这条路线发展。

---

## 5. 本文档使用的主要来源

- Park et al. 2023, *Generative Agents*:
  - https://arxiv.org/abs/2304.03442
  - https://dl.acm.org/doi/10.1145/3586183.3606763
- Zhong et al. 2023, *MemoryBank*:
  - https://arxiv.org/abs/2305.10250
- Bae et al. 2024, *My agent understands me better*:
  - https://arxiv.org/abs/2404.00573
  - https://dl.acm.org/doi/10.1145/3613905.3650903
- Anderson et al. 2004, *An Integrated Theory of the Mind*:
  - https://www.cs.utexas.edu/~dana/ACT-R.pdf
  - https://act-r.psy.cmu.edu/
- Schooler and Anderson 系谱综述入口：
  - https://docslib.org/doc/209653/the-adaptive-nature-of-memory
