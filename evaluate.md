# MemDSL coverage

前提：忽略agent loop和training和多模态/具身的数据处理部分。只看memory。

几乎所有的latent memory, parameter memory的工作都无法表达。不列在内。

## 完全覆盖

- mem0
- COMEDY
- A-MEM
- Reflexion（有复杂agent loop）
- Ego-LLaVA
- Mem-$\alpha$（有训练）
  
## 基本完全覆盖，缺一个薄的wrapper，或者有不影响功能的差异

- RET-LLM（有训练）
- Think-in-Memory
- mem0g
- Memory Sharing
- RecurrentGPT
- MemGuide（paper里抽取slot操作不好做，但repo里直接删掉了这个操作。repo可以复现，paper不行）
- SeCom
- MemSkill（有训练）
- ComoRAG（有复杂agent loop）
- MOOM（repo里有个复杂的启发式规则。paper可以正常复现，repo不行）
- MIRIX（routing agent）
- MEMENTO（有训练）
- Agent Workflow Memory
  
## 缺一个复杂的wrapper，或者缺一个模块

- "My agent understands me better"（recency）
- AriGraph（根据来源找更多内容的算法）
- HiAgent（根据上层找下层）
- H-MEM（根据上层找下层）
- SGMem（根据上层找下层）
- LightMem（按时间顺序重新ingest）
- Hierarchical Memory Organization for Wikipedia Generation（按embedding similarity聚类）
- RMM（对 query 和 memory 各做一次线性变换+残差连接，再用点积算相关性）

## 缺大于一个模块

- ExpeL（LLM打分tool+按分数forget）
- Memory Bank（recency+按分数forget）
- Generative agents（recency+retrieve分数相加）
- Zep（社群发现算法+constructor retrieval+temporal KG），这个复杂到极致了，不可能复现的
- GraphRAG（社群发现算法+根据来源找更多内容的算法）
- MemTree（tree存储+tree插入）
- HippoRAG（根据来源找更多内容的算法（PPR）+node specificity）
## 无法通过补模块复现

- Nemori（带agent loop的memory）
- MemGPT（带agent loop的memory）
  