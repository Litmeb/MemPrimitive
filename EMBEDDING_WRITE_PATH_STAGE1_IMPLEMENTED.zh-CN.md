# Embedding 第一阶段下沉说明

## 这次做了什么

- 在 `StoreLayerSpec.settings["embedding"]` 中引入了第一阶段最小 policy：
  - `enabled`
  - `mode="text"`
  - `refresh_on_update="semantic_text_change"`
- `MemoryStore.append()` 现在会在目标 layer 声明该 policy 时，自动为普通 text record 生成 `record.embedding`
- `MemoryStore.replace_record()` 现在会在语义文本变化时刷新 embedding；如果只是 metadata / graph links / 其他非语义字段变化，则沿用旧 embedding
- Mem0 fixed profile tools 的普通 add/update 不再手工调用 embedding，改为交给 store 写入路径维护

## 这次没有做什么

- 没有替掉 `TripleRepresentation(embed_extracted=True)` 的 graph/triple embedding
- 没有替掉 `embed_entities=True` 的 entity embedding map
- 没有替掉 note payload / A-MEM 专用 embedding 逻辑
- 没有替掉 query embedding 逻辑
- 没有重写 `UNIT_EMBEDDING_CONTRACT` 及整套 contract 体系

## 为什么只做最小版本

- 当前仓库里真正适合统一下沉的是“普通 record-level text embedding”
- entity、note payload、query 这些路径都不是简单的 `embed(record.text)`
- 先把 `append()` / `replace_record()` 的职责收口，可以先减少分散的普通 embedding 逻辑，同时不破坏 graph / note / retrieval 特例

## 当前明确保留的边界

- 显式提供的 `record.embedding` 仍然优先，store 不会强行覆盖
- `BasicRepresentation(elements=("embedding",))` 仍保留显式产出 `unit.embedding` 的能力
- graph link 更新仍视为结构更新，不触发 embedding 重算

## 第二阶段可能继续做什么

- 让更多 graph dedup / organization 路径统一复用 store-side embedding policy，而不是局部 fallback
- 评估是否引入 `note_payload` 等额外 mode
- 逐步把 contract 表达从 unit-side embedding 迁移到更贴近 store/layer 能力的表述
