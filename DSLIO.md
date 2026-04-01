閻╊喗鐖ｉ弰顖濐唨 memory DSL 娑撳秴褰ч弰?taxonomy閿涘矁鈧本妲告稉鈧稉?**閸欘垳绮嶉崥鍫涒偓浣稿讲閹兼粎鍌ㄩ妴浣稿讲婢跺秶鏁ら妴浣稿讲閺囨寧宕?* 閻ㄥ嫮閮寸紒鐕傜礉闁絼绠炲В蹇庨嚋鐎涙顔岄張鈧總浠嬪厴鐎规矮绠熼幋鎰娑擃亝鐖ｉ崙鍡樐侀崸妤佸复閸欙綇绱濋懓灞肩瑝閺勵垰褰ч崘娆愬灇闁板秶鐤嗘い骞库偓?

* **DSL 鐏?*閿涙碍寮挎潻鎵兇缂佺喓鏁遍崫顏冪昂濡€虫健缂佸嫭鍨?
* **Interface 鐏?*閿涙俺顫夌€规碍鐦℃稉顏吥侀崸妤€鎮嗘禒鈧稊鍫涒偓浣告倷娴犫偓娑?
* **Implementation 鐏?*閿涙艾鍙挎担鎾剁暬濞夋洖骞撶€圭偟骞囨潻娆庣昂閹恒儱褰?

---

# 1. 閹缍嬮幀婵婄熅閿涙碍濡?memory system 閻鍨氭稉鈧弶鈩冩殶閹诡喗绁?

閸欘垯浜掗幎濠冩殻娑擃亞閮寸紒鐔稿▕鐠炩剝鍨氶敍?

```text
raw input
  -> unit formation
  -> representation
  -> write trigger
  -> organization / normal write
  -> optional memory evolution
  -> retrieval
  -> readout
  -> downstream agent
```

閹碘偓娴犮儲鐦℃稉顏吥侀崸妤呭厴鎼存棁顕氶弰顖ょ窗

```text
Module: InputState -> OutputArtifact (+ side effects on memory store)
```

娴ｅ棜绻栭柌灞炬箒娑撯偓娑擃亜鍙ч柨顔惧仯閿?

**memory 濡€虫健娑撳秴褰ч弰顖滃嚱閸戣姤鏆?*閿涘苯绶㈡径姘侀崸妞剧窗娣囶喗鏁?store閵?
閹碘偓娴犮儲甯撮崣锝嗘付婵傜晫绮烘稉鈧幋鎰瑓闂堛垼绻栫粔宥呰埌瀵骏绱?

```python
output, new_store, aux = module(input, store, context, config)
```

娑旂喎姘ㄩ弰顖濐嚛閿涘本鐦℃稉顏吥侀崸妤呭厴閹恒儲鏁归敍?

* `input`閿涙艾缍嬮崜宥堫洣婢跺嫮鎮婇惃鍕嚠鐠?
* `store`閿涙艾缍嬮崜?memory store
* `context`閿涙俺绻嶇悰灞肩瑐娑撳鏋?
* `config`閿涙碍膩閸ф寮弫?

鏉╂柨娲栭敍?

* `output`閿涙矮绶垫稉瀣埗娴ｈ法鏁ら惃鍕波閺?
* `new_store`閿涙碍娲块弬鏉挎倵閻?store
* `aux`閿涙矮鑵戦梻鏉戝瀻閺佽埇鈧焦妫╄箛妞尖偓涔紃ace閵嗕浇袙闁插﹣淇婇幁?

鏉╂瑤绱伴棃鐐茬埗闁倸鎮庡Ο鈥虫健閸栨牕鐤勬灞肩瑢閹兼粎鍌ㄩ妴?

---

# 2. 閸忓牆鐣炬稊澶婂殤娑擃亜鍙忕仦鈧弽鍥у櫙鐎电钖?

閸︺劏鐨ョ€涙顔岄幒銉ュ經閸撳稄绱濋崗鍫濈暰娑斿鍙忕化鑽ょ埠閸忓彉闊╅惃鍕殶閹诡喚琚崹瀣ㄢ偓鍌欑瑝閻掕埖鐦℃稉顏吥侀崸妤呭厴閸氬嫯顕╅崥鍕樈閵?

---

## 2.1 Observation

鐞涖劎銇氭径鏍劥鏉堟挸鍙嗛妴?

```python
Observation = {
    "id": str,
    "type": str,           # dialogue / tool_result / environment_step / document / reflection
    "content": Any,        # raw text, json, image desc, tool output ...
    "timestamp": float,
    "source": str,
    "metadata": dict,
}
```

---

## 2.2 MemoryUnit

鐞涖劎銇氶張鈧亸蹇氼唶韫囧棗宕熼崗鍐︹偓?

```python
MemoryUnit = {
    "unit_id": str,
    "unit_type": str,      # turn / event / fact / entity_state / summary / latent
    "content": Any,        # text / structure / tensor ref
    "embedding": Any | None,
    "entities": list[str],
    "tags": list[str],
    "timestamp": float,
    "span": dict | None,   # source span or provenance
    "confidence": float | None,
    "importance": float | None,
    "novelty": float | None,
    "access_stats": {
        "count": int,
        "last_access": float | None
    },
    "relations": list[dict],
    "metadata": dict,
}
```

---

## 2.3 MemoryStore

鐞涖劎銇氶弫缈犻嚋 memory 閻樿埖鈧降鈧?

```python
MemoryStore = {
    "stores": {
        "working": list[MemoryUnit],
        "episodic": list[MemoryUnit],
        "semantic": list[MemoryUnit],
        "profile": list[MemoryUnit],
    },
    "indices": {
        "vector_index": Any,
        "entity_index": dict,
        "time_index": Any,
        "graph_index": Any,
    },
    "budgets": {
        "max_units": int,
        "max_tokens": int,
    },
    "metadata": dict,
}
```

---

## 2.4 Query

鐞涖劎銇?retrieval 閻ㄥ嫭鐓＄拠顫偓?

```python
Query = {
    "query_id": str,
    "text": str,
    "goal": str | None,
    "entities": list[str],
    "constraints": dict,
    "timestamp": float,
    "metadata": dict,
}
```

---

## 2.5 RetrievedContext

鐞涖劎銇?retrieval 缂佹挻鐏夐妴?

```python
RetrievedContext = {
    "items": list[MemoryUnit],
    "scores": list[dict],   # similarity / recency / utility / symbolic ...
    "summary": str | None,
    "trace": dict,
}
```

---

## 2.6 AgentContext

鐞涖劎銇氭潻鎰攽閺冩湹绗傛稉瀣瀮閵?

```python
AgentContext = {
    "task": str | None,
    "goal": str | None,
    "dialogue_history": list[dict],
    "current_state": dict,
    "tool_state": dict,
    "runtime": dict,
}
```

---

# 3. 濮ｅ繋閲滅€涙顔屾俊鍌欑秿鐎规矮绠熼弽鍥у櫙閹恒儱褰?

娑撳娼版潻娑樺弳闁插秶鍋ｉ妴?

---

## A. Unit Formation Interface

鐎瑰啳绀嬬拹锝嗗Ω閸樼喎顫?observation 閸掑洦鍨?memory units閵?

### 閹恒儱褰涚€规矮绠?

```python
def unit_formation(
    observation: Observation,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[MemoryUnit], dict]:
    ...
```

鏉╂柨娲栭敍?

* `list[MemoryUnit]`閿涙艾鑸伴幋鎰畱鐠佹澘绻傞崡鏇炲帗
* `dict`閿涙瓫ux閿涘本鐦俊鍌涘▕閸欐牞鐦夐幑顔衡偓涔竌rser logits閵嗕浇袝閸欐垵甯崶?

### 鏉堟挸鍙嗙拠顓濈疅

* `observation`閿涙矮绔撮弶鈩冩煀鏉堟挸鍙?
* `store`閿涙艾褰查柅澶涚礉閻劋绨?entity-aware parsing
* `context`閿涙艾缍嬮崜宥勬崲閸斅扳偓浣割嚠鐠囨繂宸婚崣鑼搼

### 鏉堟挸鍤憰浣圭湴

濮ｅ繋閲滄潏鎾冲毉 unit 閼峰啿鐨憰浣稿瘶閸氼偓绱?

* `unit_id`
* `unit_type`
* `content`
* `timestamp`
* `metadata.provenance`

### 鐢瓕顫嗙€圭偟骞?

* `TurnSplitter`
* `EventExtractor`
* `FactExtractor`
* `EntityStateExtractor`
* `SessionSummarizer`

---

## B. Representation Interface

鐎瑰啳绀嬬拹锝嗗Ω unit 鐟欏嫯瀵栭崠鏍ㄥ灇閺屾劗顫掔悰銊с仛閿涘苯鑻熺悰銉ュ弿 embedding / schema閵?

### 閹恒儱褰涚€规矮绠?

```python
def represent(
    units: list[MemoryUnit],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[MemoryUnit], dict]:
    ...
```

### 娴ｆ粎鏁?

* 缂?unit 閸?embedding
* 鏉烆剚鍨?kv / triple / frame
* 鐞?metadata
* 閺嶅洤鍣崠鏍х摟濞堥潧鎮?

### 鏉堟挸鍤憰浣圭湴

representation 濡€虫健娑撳秷鍏橀弨鐟板綁 unit 閻ㄥ嫯顕㈡稊澶庨煩娴犳枻绱濋崣顏囧厴閺€鐟板綁閹存牕顤冨楦裤€冪粈鍝勮埌瀵繈鈧?

娓氬顩ч敍?

* 閺傚洦婀版禍瀣杽 -> triple
* 鐎电鐦?turn -> hybrid(text + embedding + entities)

---

## C. Organization Interface

鐎瑰啳绀嬬拹?**ingest-time 閻ㄥ嫮绮嶇紒鍥︾瑢鐢瓕顫夐崘娆忓弳閹绘劒姘?*閵?

閸︺劍鏌婇惃鍕嚔娑斿绗呴敍灞肩瑝閸愬秵濡?`organization` 閸欘亞鎮婄憴锝勮礋 閳ユ笡lacement plan generator閳ユ縿鈧?
鐎瑰啯妫﹂崘鍐茬暰閿?

* unit 閸樿鎽㈡稉鈧仦?/ 閸濐亙閲滈崚鍡楀隘
* 娑撳骸鍑￠張?memory 瀵よ櫣鐝涙禒鈧稊?links
* 閸︺劌鐖剁憴鍕晸閸忋儴鐭惧鍕瑐婵″倷缍嶅锝呯础閽€钘夊弳 store

娑旂喎姘ㄩ弰顖濐嚛閿?

* `organization` = organize + commit normal write
* `memory_evolution` 娑撳秴鍟€姒涙顓婚幍鎸庡閺咁噣鈧?append / 鐢瓕顫夐拃钘夌氨

### 閹恒儱褰涚€规矮绠?

```python
def organize(
    units: list[MemoryUnit],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[dict], dict]:
    ...
```

鏉堟挸鍤崣顖欎簰閺?`PlacementPlan`閿?

```python
PlacementPlan = {
    "unit_id": str,
    "target_store": str,       # working / episodic / semantic / profile
    "links_to_add": list[dict],
    "index_updates": list[dict],
}
```

### 鏉堟挸鍤拠顓濈疅

* `PlacementPlan` 娴犲秴褰叉担婊€璐熼弰鎯х础娑擃參妫跨悰銊с仛
* 娴?`organization` 閸忎浇顔忛惄瀛樺复娣囶喗鏁?`store`
* trace 鎼存棁顔囪ぐ鏇窗
  * target store / layer
  * links / index updates
  * affected unit ids / record ids
  * 鐢瓕顫夐崘娆忓弳闁插洤褰囬惃?ingest-time update 鐞涘奔璐?

### 鐢瓕顫嗛崝鐔诲厴

* 閸掋倕鐣剧拠銉︽杹閸濐亙閲滅€涙劕绨?
* 瀵よ櫣鐝?temporal / entity / similarity / causal links
* append 閸掓壆娲伴弽?layer
* 閸ユ崘濡悙鐟板晸閸?
* 閸掑棗灏崘娆忓弳
* 娑撳骸缍嬮崜?organization strategy 缁毖嗏偓锕€鎮庨惃?ingest-time merge / upsert / replace

---

## D. Trigger Interface

鐎瑰啳绀嬬拹锝呭灲閺傤厸鈧粍鐓囨稉顏勬倵缂侇厼濮╂担婊嗩洣娑撳秷顩︾憴锕€褰傞垾婵勨偓?

閸︺劌缍嬮崜?DSL 闁插矉绱漷rigger 閺勵垯绔寸猾璇插讲婢跺秶鏁ら張鍝勫煑閿涘矁鈧奔绗夐弰顖氬涧閼宠姤婀囬崝鈥茬艾閸楁洑绔?slot 閻ㄥ嫪绗撶仦鐐茨侀崸妞尖偓?
閸氬奔绔?trigger family 閸欘垯浜掗崚鍡楀焼鐎圭偘绶ラ崠鏍﹁礋閿?

* `write_trigger`閿涙艾锝為崘?`decisions`閿涘本甯堕崚鑸垫Ц閸氾箒绻橀崗銉ョ埗鐟欏嫬鍟撻崗銉ㄧ熅瀵?
* `evolution_trigger`閿涙艾锝為崘?`decisions`閿涘本甯堕崚鑸垫Ц閸氾箑鎯庨崝銊╊杺婢舵牜娈?memory evolution

### 閹恒儱褰涚€规矮绠?

```python
def should_trigger(
    units: list[MemoryUnit],
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[bool], list[dict]]:
    ...
```

鏉╂柨娲栭敍?

* 濮ｅ繋閲?unit 閺勵垰鎯佺憴锕€褰傝ぐ鎾冲 slot 鐎电懓绨查惃鍕З娴?
* 濮ｅ繋閲滈崘宕囩摜閻ㄥ嫯袙闁插﹣绗岄崚鍡樻殶

娓氬顩ф潏鎾冲毉閿?

```python
[
  True, False, True
],
[
  {"importance": 0.91, "novelty": 0.67, "reason": "new_user_preference"},
  {"importance": 0.21, "reason": "small_talk"},
  {"importance": 0.88, "reason": "task_outcome"}
]
```

### Slot 閸栨牜瀹崇€?

閸?stage-1 runtime 娑擃叏绱漷rigger 閺堝搫鍩楃悮顐ｆ杹閸︺劋琚辨稉顏冪瑝閸氬瞼娈?ingest slot閿?

```text
unit_formation -> representation -> write_trigger -> organization -> evolution_trigger -> memory_evolution
```

鐎电懓绨查惃鍕殶閹诡噣娼扮痪锕€鐣鹃弰顖ょ窗

* `write_trigger` 閸?`Packet.decisions`
* `organization` 鐠囪褰?`Packet.decisions` 楠炶泛鐣幋鎰埗鐟欏嫬鍟撻崗?
* `evolution_trigger` 閸?`Packet.decisions`
* `memory_evolution` 鐠囪褰?`decisions`

婵″倹鐏夐弻鎰嚋 runtime 閺嗗倹妞傛禒宥呭帒鐠?`memory_evolution` 閸ョ偤鈧偓娴ｈ法鏁?`decisions`閿涘苯绨茬憴鍡曡礋閸氭垵鎮楅崗鐓庮啇鐞涘奔璐熼敍宀冣偓灞肩瑝閺勵垳娲伴弽鍥嚔娑斿鈧?

### 鏉╂瑦鐗辩拋鎹愵吀閻ㄥ嫬銈芥径?

* 閼宠棄宕熼悪?ablate write policy
* 閼宠姤濡?閳ユ粌鐖剁憴鍕晸閸忋儮鈧?娑?閳ユ粓顤傛径鏍ㄧ川閸栨牑鈧?濞撳懏娅氶崠鍝勫瀻
* 閼宠棄浠?learned write / heuristic write / llm-judge write 閻ㄥ嫮绮烘稉鈧В鏃囩窛

---

## E. Memory Evolution Interface

鐎瑰啳绀嬬拹?**姒涙顓绘稉宥呮儙閸斻劊鈧線顤傛径鏍曢崣?* 閻?memory evolution閵?

鏉╂瑩鍣烽惃鍕嚠鐠炩€茬瑝閺勵垪鈧粍婀板▎?observation 閻ㄥ嫬鐖剁憴鍕儰鎼存挴鈧繐绱濋懓灞炬Ц閿?

* 鐎电懓鍑￠張?memory 閻ㄥ嫭鏆ｉ悶?
* 妤傛ê鐪伴幎鍊熻杽
* 缂佸瓨濮?
* 闁绻?
* 閸氬骸褰撮柌宥呭晸 / 闁插秶绮?

### 閹恒儱褰涚€规矮绠?

```python
def memory_evolution(
    packet: dict,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[MemoryStore, dict]:
    ...
```

### 閸旂喕鍏?

* summarize
* reflect
* prune / forget
* dedup
* consolidate
* move / archive
* 鐎电懓鍑￠張?memory 閻?rewrite / review
* 閸︺劍妯夊蹇毿曢崣鎴炴鏉╂稖顢?profile / concept abstraction

### 鏉堟挸鍤稉顓犳畱 aux 鎼存柨瀵橀崥?

* affected unit ids
* affected record ids
* evolution trigger reason
* rewrite / summarize / prune trace
* overwritten / removed / archived records

鏉╂瑤閲?trace 瀵板牓鍣哥憰渚婄礉閸ョ姳璐熸担鐘辩閸氬骸鍨庨弸?memory 閺堝搫鍩楅弮鏈电窗闂団偓鐟曚降鈧?

---

## F. Compression / Abstraction Family

鏉╂瑤绗夐弰顖滃缁?slot閿涘矁鈧本妲?`memory_evolution` 閻ㄥ嫪绔存稉顏堝櫢鐟曚礁鐡欑€硅埖妫岄妴?

娑旂喎姘ㄩ弰顖濐嚛閿?

* `compress` / `abstract` 娴犲秶鍔ч弰顖炲櫢鐟曚焦婧€閸?
* 娴ｅ棗婀ぐ鎾冲 primitive 閸掓帒鍨庢稉瀣剁礉鐎瑰啩婊戠仦鐐扮艾 `memory_evolution` 閻ㄥ嫬鐤勯悳鏉垮綁娴?

### 閹恒儱褰涚€规矮绠?

```python
def compress(
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[list[MemoryUnit], MemoryStore, dict]:
    ...
```

娑撹桨绮堟稊鍫ｇ箲閸ョ偘琚辨稉顏冪鐟楀尅绱?

* `list[MemoryUnit]`閿涙碍鏌婇悽鐔稿灇閻ㄥ嫭鎲崇憰?濮掑倸搴风拋鏉跨箓
* `MemoryStore`閿涙碍娲块弬鏉挎倵閻?store

### 鐢瓕顫嗙憴锕€褰傞弬鐟扮础

* periodic
* episode_end
* budget_exceeded
* explicit_reflection

### 鐢瓕顫嗘潏鎾冲毉

* summary units
* concept units
* profile rewrite
* prototype units

---

## G. Retrieval Interface

鐎瑰啳绀嬬拹锝囩舶鐎?query閿涘奔绮?store 閹?relevant memory閵?

### 閹恒儱褰涚€规矮绠?

```python
def retrieve(
    query: Query,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[RetrievedContext, dict]:
    ...
```

### 鏉╂柨娲?

* `RetrievedContext`
* 濡偓缁?trace

### trace 鎼存棁顕氶弽鍥у櫙閸?

閸ョ姳璐?retrieval 閺勵垱鐗宠箛鍐ㄧ杽妤犲苯褰夐柌蹇庣娑撯偓閵?

娓氬顩ч敍?

```python
{
    "candidate_count": 200,
    "selected_count": 8,
    "scorers": {
        "similarity": [...],
        "recency": [...],
        "entity_match": [...],
        "utility": [...]
    },
    "final_ranking": [...]
}
```

鏉╂瑦鐗辨担鐘插讲娴犮儳鐖虹粚璁圭窗

* 閸掓澘绨抽弰?sim 鐠ц渹缍旈悽銊ㄧ箷閺?recency 鐠ц渹缍旈悽?
* 娑撹桨绮堟稊鍫熺厙閺壜ゎ唶韫囧棜顫﹂柅澶夎厬

---

## H. Maintenance Family

鏉╂瑥鎮撻弽铚傜瑝閺勵垳瀚粩?slot閿涘矁鈧本妲?`memory_evolution` 閻ㄥ嫬褰熸稉鈧稉顏勭摍鐎硅埖妫岄妴?

鐎瑰啳绀嬬拹锝咁啇闁插繑甯堕崚韬测偓渚€浠愯箛妯糕偓浣稿箵闁插秲鈧線鍣搁幒鎺戠碍缁涘顤傛径鏍ㄧ川閸栨牓鈧?

### 閹恒儱褰涚€规矮绠?

```python
def maintain(
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[MemoryStore, dict]:
    ...
```

### 閸旂喕鍏?

* pruning
* decay update
* dedup
* conflict cleanup
* consolidation trigger
* archival movement

### 瀵ら缚顔?

閹?`maintenance` 鐠佹崘顓搁幋鎰讲閸涖劍婀＄憴锕€褰傞惃鍕倵閸欑増顒炴銈忕礉娴ｅ棗婀ぐ鎾冲 DSL 娑擃厼鐨㈤崗鎯邦潒娑?`memory_evolution` 閻ㄥ嫪绔寸粔宥呯杽閻滅増膩瀵骏绱濋懓灞肩瑝閺勵垱鏌婃晶鐐恒€婄仦?slot閵?

---

## I. Readout Interface

鐎瑰啳绀嬬拹锝嗗Ω retrieved memory 鏉烆剚鍨氭稉瀣埗 agent 閸欘垱绉风拹鍦畱鏉堟挸鍙嗛妴?

### 閹恒儱褰涚€规矮绠?

```python
def readout(
    retrieved: RetrievedContext,
    store: MemoryStore,
    context: AgentContext,
    config: dict
) -> tuple[dict, dict]:
    ...
```

鏉堟挸鍤崣顖濆厴閺勵垽绱?

```python
ReadoutPacket = {
    "prompt_blocks": list[str],
    "structured_fields": dict,
    "citations": list[dict],
    "latent_state_update": Any | None,
}
```

### 娑撹桨绮堟稊鍫濆礋閻欘剛鐝涢幒銉ュ經

閸ョ姳璐熼敍?

* retrieval 鐟欙絽鍠呴垾婊勫娴犫偓娑斿牃鈧?
* readout 鐟欙絽鍠呴垾婊勨偓搴濈疄閻劉鈧?

鏉╂瑤琚辨稉顏堟６妫版ü绗夐懗鑺ヨ穿閵?

閸氬本鐗遍惃?retrieved items閿?

* 閸欘垯浜掗惄瀛樺复閹?prompt
* 閸欘垯浜掗崗?summary 閸愬秵瀚?
* 閸欘垯浜掗幐?profile / facts / history 閸掑棗灏▔銊ュ弳
* 閸欘垯浜掗崣顏嗙舶 planner閿涘奔绗夌紒?responder

---

# 4. 閺囩绻樻稉鈧銉窗缂佺喍绔撮幋鎰垼閸戝棙膩閸ф顒烽崥?

娑撹桨绨＄拋?DSL 閻喐顒滃Ο鈥虫健閸栨牭绱濇担鐘垫晪閼峰啿褰叉禒銉ュ繁閸掕埖澧嶉張澶嬆侀崸妤呭厴闁潧鐣х紒鐔剁缁涙儳鎮曢敍?

```python
class MemoryModule:
    def __call__(self, packet, store, context, config):
        return output_packet, new_store, aux
```

閸忔湹鑵?`packet` 閺勵垶鈧氨鏁ゆ潪鎴掔秼閿涘本瀵滃Ο鈥虫健闂冭埖顔屾稉宥呮倱鐟佸懍绗夐崥灞藉敶鐎瑰箍鈧?

娓氬顩ч敍?

```python
Packet = {
    "observation": Observation | None,
    "units": list[MemoryUnit] | None,
    "decisions": list[bool] | None,
    "placements": list[PlacementPlan] | None,
    "decisions": list[bool] | None,
    "query": Query | None,
    "retrieved": RetrievedContext | None,
    "readout": dict | None,
    "trace": dict,
}
```

閻掕泛鎮楁稉宥呮倱濡€虫健閸欘亣顕伴崘娆掑殰瀹稿崬鍙ц箛鍐畱鐎涙顔岄妴?

鏉╂瑦鐗辨担鐘垫畱 pipeline 閸欘垯浜掗崘娆愬灇閿?

```python
packet = {"observation": obs, "trace": {}}

packet, store, aux1 = unit_formation(packet, store, context, cfg1)
packet, store, aux2 = represent(packet, store, context, cfg2)
packet, store, aux3 = write_trigger(packet, store, context, cfg3)
packet, store, aux4 = organize(packet, store, context, cfg4)
packet, store, aux5 = evolution_trigger(packet, store, context, cfg5)
packet, store, aux6 = memory_evolution(packet, store, context, cfg6)
...
```

閸忔湹鑵戦弬鎵畱鐠囶厺绠熼弰顖ょ窗

* `organize(...)` 鐎瑰本鍨?ingest-time normal write
* `memory_evolution(...)` 閸欘亜婀?`evolution_trigger` 閹垫挸绱戦弮璺轰粵妫版繂顦诲鏂垮

鏉╂瑥姘ㄩ弰顖氱发閸忕鐎烽惃?**IR-style intermediate representation** 閹繆鐭鹃妴?
閹存垼顫庡妤勭箹鐎?DSL 瀵板牊婀佺敮顔煎И閵?

---

# 5. 鏉╂ê褰叉禒銉ョ暰娑斿膩閸ф娈戦垾婊嗗厴閸旀稓瀹抽弶鐔测偓?

婵″倹鐏夐幆鍐蹭粵閼奉亜濮╅幖婊呭偍閿涘苯褰х€规矮绠?IO 鏉╂ü绗夋径鐕傜礉鏉╂顩︾€规矮绠?**type constraints**閵?
娑撳秶鍔ч幖婊呭偍娴兼氨绮嶉崥鍫濆毉瀵板牆顦块棃鐐寸《闁板秶鐤嗛妴?

濮ｆ柨顩ч敍?

* `graph_retrieval` 鐟曚焦鐪?store 闁插本婀?graph links
* `entity_merge_update` 鐟曚焦鐪?unit 闁插本婀?entity ids
* `similarity_retrieval` 鐟曚焦鐪?unit 閺?embedding
* `profile_rewrite` 鐟曚焦鐪伴惄顔界垼 store 閺?profile / semantic
* `delta_update` 鐟曚焦鐪?unit 閸欘垰顕鎰煂閺冄嗩唶瑜?

閹碘偓娴犮儲鐦℃稉顏吥侀崸妤呮珟娴?IO 婢舵牭绱濇潻妯虹安鐠囥儱锛愰弰搴窗

```python
ModuleSpec = {
    "name": str,
    "input_requirements": list[str],
    "output_guarantees": list[str],
    "side_effects": list[str],
}
```

娓氬顩ч敍?

```python
{
    "name": "hybrid_retriever",
    "input_requirements": ["query.text", "store.indices.vector_index"],
    "output_guarantees": ["retrieved.items", "retrieved.scores.similarity"],
    "side_effects": []
}
```

閸愬秵鐦俊鍌︾窗

```python
{
    "name": "entity_merge_updater",
    "input_requirements": ["units.entities", "placement.target_store"],
    "output_guarantees": ["store.updated_units", "trace.merge_log"],
    "side_effects": ["modify_store"]
}
```

鏉╂瑥顕懛顏勫З缂佸嫬鎮庨棃鐐茬埗闁插秷顩﹂妴?

---

# 6. DSL 鐎涙顔屾稉宥勭矌閼宠棄鐣炬稊?IO閿涘矁绻曢懗钘夌暰娑斿鈧粍鎼锋担婊嗩嚔娑斿鈧?

娴ｇ姴褰叉禒銉﹀Ω濮ｅ繋閲滅€涙顔岄崚鍡樺灇娑撳鐪伴敍?

### 1) Type signature

鏉堟挸鍙嗘潏鎾冲毉缁鐎?

### 2) Contract

鏉╂瑤閲滃Ο鈥虫健閹佃儻顕崑姘矆娑斿牞绱濇稉宥嗗鐠囧搫浠涙禒鈧稊?

### 3) Side effects

鐎瑰啩绱版俊鍌欑秿娣囶喗鏁?store 閹?metadata

娓氬顩?`retrieve` 閻?contract 閸欘垯浜掗弰顖ょ窗

* 鏉堟挸鍙?query 閸?store
* 鏉堟挸鍤稉鈧稉顏呮箒鎼村繒娈?memory list
* 韫囧懘銆忛幓鎰返 score trace
* 娑撳秴鍘戠拋鍝ユ纯閹恒儰鎱ㄩ弨鐟板斧婵?memory 閸愬懎顔?

閼?`memory_evolution` 閻?contract 閺勵垽绱?

* 鏉堟挸鍙?packet / store / context
* 閸欘垯浜掓穱顔芥暭 store
* 韫囧懘銆忔潻鏂挎礀 evolution / overwrite / summarize / prune trace

鏉╂瑦鐗遍弫缈犻嚋缁崵绮烘导姘舵姜鐢憡绔婚弲鑸偓?

---

# 7. 閹存垵缂撶拋顔炬畱閺堚偓鐏忓繑鐖ｉ崙鍡樺复閸欙綁娉﹂崥?

婵″倹鐏夋担鐘冲厒鐏忚棄鎻╅拃钘夋勾閿涘奔绗夎箛鍛瀵偓婵顔曠拋鈥崇繁婢额亪鍣搁妴鍌氬讲娴犮儱鍘涚€规矮绔存稉顏呮付鐏忓繒澧楅張顒婄窗

```python
unit_formation(observation, store, context, config) 
    -> units, aux

represent(units, store, context, config) 
    -> units, aux

should_write(units, store, context, config) 
    -> decisions, aux

organize(units, store, context, config) 
    -> store, aux

memory_evolution(packet, store, context, config) 
    -> store, aux

retrieve(query, store, context, config) 
    -> retrieved, aux

readout(retrieved, store, context, config) 
    -> readout_packet, aux
```

鏉╂瑦妲告稉鈧稉顏堟姜鐢鐤勯悽銊ф畱鐠ч鍋ｉ妴?

---

# 8. 娑撯偓娑擃亜鍙挎担鎾茬伐鐎涙劧绱版俊鍌欑秿濡€虫健閸栨牜绮嶉崥?

閸嬪洩顔曟担鐘茬暰娑斿绔存稉?memory config閿?

```python
MemoryConfig(
    unit_formation = EventExtractor(),
    representation = HybridRep(embed=True, entity_link=True),
    write_trigger = ImportanceNoveltyGate(),
    organization = EntityTemporalOrganizer(),
    evolution_trigger = Never(),
    retrieval = HybridRetriever(),
    readout = StructuredReadout(),
    memory_evolution = ImportanceDecayPruner()
)
```

闁絼绠炴潻鎰攽閺冭泛姘ㄩ弰顖ょ窗

### 閸愭瑥鍙嗛梼鑸殿唽

```python
units = EventExtractor(obs)
units = HybridRep(units)
decisions = ImportanceNoveltyGate(units, store)
selected_units = filter_by_decision(units, decisions)
store, placement = EntityTemporalOrganizer(selected_units, store)
decisions = Never()(selected_units, store)
if any(decisions):
    store = ImportanceDecayPruner(store)
```

### 鐠囪褰囬梼鑸殿唽

```python
retrieved = HybridRetriever(query, store)
readout_packet = StructuredReadout(retrieved, store)
```

鏉╂瑥姘ㄥ鑼病閺勵垯绔存稉顏勫讲閹笛嗩攽閻?memory DSL runtime 娴滃棎鈧?

---

# 9. 娴犲海鐖虹粚鎯邦潡鎼达妇婀呴敍灞剧垼閸?IO 閹恒儱褰涢惃鍕壈娑斿娼敮绋裤亣

鐎瑰啩绗夋禒鍛Ц瀹搞儳鈻奸梻顕€顣介敍宀冪箷閻╁瓨甯撮崘鍐茬暰娴ｇ姷娈戦惍鏃傗敀閼虫垝绗夐懗鑺ュ灇缁斿鈧?

---

## 9.1 娓氬じ绨崑姘辩矋閸氬牊鎮崇槐?

婵″倹鐏夊Ο鈥虫健閸忓彉闊╅幒銉ュ經閿涘奔缍橀幍宥堝厴閿?

* 閸ュ搫鐣?representation閿涘本鎮崇槐?retrieval
* 閸ュ搫鐣?retrieval閿涘本鎮崇槐?write policy
* 閼辨柨鎮庨幖婊呭偍 write + organization + memory evolution

閸氾箑鍨幖婊呭偍缁屾椽妫块崣顏呮Ц缁鹃晲绗傜拫鍫濆徍閵?

---

## 9.2 娓氬じ绨崑姘彆楠炶櫕鐦潏?

缂佺喍绔撮幒銉ュ經閸氬函绱濇担鐘插讲娴犮儳婀″锝嗗付閸掕泛褰夐柌蹇ョ窗

* 閸氬本鐗遍惃?unit formation閿涘奔绗夐崥?retrieval
* 閸氬本鐗遍惃?organization閿涘奔绗夐崥?write trigger
* 閸氬本鐗遍惃?retrieval閿涘奔绗夐崥?readout

鏉╂瑥顕拋鐑樻瀮鐎圭偤鐛欏鍫濆彠闁款喓鈧?

---

## 9.3 娓氬じ绨崚鍡樼€?recurring motifs

婵″倹鐏夊鍫濐樋妤傛ɑ鈧嗗厴缁崵绮洪柈鑺ュ姬鐡掔绱?

* event/fact unit
* hybrid representation
* selective write
* organization-driven normal write
* hybrid retrieval
* periodic evolution

闁絼缍樼亸杈厴娴犲孩鎮崇槐銏㈢波閺嬫粌缍婄痪鍐插毉 motif閿涘矁鈧奔绗夐弰顖氬涧瀵版鍩岄垾婊勭厙娑擃亪绮︾粻閬嶅帳缂冾喗娲挎總瑙ｂ偓婵勨偓?

---

# 10. 娴ｅ棜顩﹀▔銊﹀壈閿涙矮绗夐弰顖涘閺堝鐡у▓鐢稿厴闁倸鎮庣€瑰苯鍙忛悪顒傜彌

鏉╂瑩鍣烽張澶夌娑擃亞骞囩€圭偤妫舵０姗堢窗
閺堝绨虹€涙顔屾稊瀣？閼帮箑鎮庡鍫濆繁閿涘奔绗夐懗鑺ユ簚濮婄増濯跺鈧妴?

濮ｆ柨顩ч敍?

### representation 閸?retrieval 瀵缚鈧箑鎮?

* 濞?embedding 鐏忓崬浠涙稉宥勭啊閸氭垿鍣哄Λ鈧槐?
* 濞?graph link 鐏忓崬浠涙稉宥勭啊 graph hop retrieval

### unit formation 閸?organization / evolution 瀵缚鈧箑鎮?

* fact unit 閺囨挳鈧倸鎮?entity/profile-oriented organization 閹存牕鎮楃紒?profile evolution
* raw chunk 閺囨挳鈧倸鎮?append-style organization

### compression 閸?maintenance 瀵缚鈧箑鎮?

* summary 閻㈢喐鍨氶崥搴″讲閼宠棄寮芥潻鍥ㄦ降鐟欙箑褰傞崚?raw memory
* consolidation 娴兼碍鏁奸崣?retention policy

閹碘偓娴犮儲顒滅涵顔间粵濞夋洑绗夐弰顖氫海鐟佸懎鐣崗銊у缁斿绱濋懓灞炬Ц閿?

* **閹恒儱褰涢悪顒傜彌**
* **鐠囶厺绠熸稉濠傚帒鐠佸摜瀹抽弶?*
* **閹兼粎鍌ㄩ弮鍫曗偓姘崇箖 compatibility constraints 鏉╁洦鎶ら棃鐐寸《缂佸嫬鎮?*

鏉╂瑦鐦垾婊冪暚閸忋劏鍤滈悽杈╃矋閸氬牃鈧繀寮楃拫銊ョ繁婢舵哎鈧?
