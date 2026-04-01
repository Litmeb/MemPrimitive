# Memory Primitive 閹兼粎鍌ㄧ粚娲？

閺堫剚鏋冨锝囬兇缂佺喐鐏囨稉鐐槨娑?primitive slot 閻ㄥ嫭澧嶉張澶婄杽閻滄澘褰查懗鑺モ偓褝绱濈€规矮绠熼幖婊呭偍缁屾椽妫块惃鍕珶閻ｅ矉绱濋獮璺哄瀻閺嬫劖膩閸ф妫块惃鍕悑鐎硅鈧呭閺夌喆鈧?

---

## 0. 閹兼粎鍌ㄧ粚娲？閹槒顫?

### 缂佹潙瀹崇紒鎾寸€?

```
閹兼粎鍌ㄧ粚娲？ = LayeredStoreTopo 鑴?UnitFormation 鑴?Representation 鑴?WriteTrigger 
         鑴?Organization 鑴?MemoryEvolution 鑴?Retrieval 鑴?Readout 
         鑴?EvolutionTrigger
```

濮ｅ繋閲滅紒鏉戝閺勵垯绔存稉?*缁傜粯鏆庨柅澶嬪闂?*閿涘牆鐤勯悳鏉垮綁娴ｆ搫绱氶敍灞剧槨娑擃亜褰夋担鎾冲敶闁劌褰查懗鑺ユ箒**缁傜粯鏆?鏉╃偟鐢荤搾鍛棘**閵?

### Store 閹锋挻澧ら柅澶嬪閿涘牏绮ㄩ弸鍕樊鎼达讣绱?

Store 閹锋挻澧ら崘鍐茬暰娴滃棙鏆ｆ稉顏嗛兇缂佺喓娈戞銊︾仸閿涘本妲搁張鈧妯虹湴閻ㄥ嫮绮ㄩ弸鍕偓褔鈧瀚ㄩ妴鍌濈箹闁插奔绗夐崘宥嗗Ω閹锋挻澧ら弸姘娑?`Single-Flat / Dual-Store / Graph-Centric` 鏉╂瑧琚崨钘夋倳濡剝婢橀敍宀冣偓灞炬Ц閺€瑙勫灇**婢舵艾鐪伴柊宥囩枂**閿?


| 缂佹挻鐎€涙顔?            | 閸氼偂绠?                | 閸忕鐎烽崣鏍р偓?                                                                         |
| ---------------- | ------------------ | ----------------------------------------------------------------------------- |
| `layer_count`    | 缁崵绮洪張澶婂殤鐏?memory layer | 1, 2, 3, 4                                                                    |
| `layer.theme`    | 鐠囥儱鐪伴惃鍕嚔娑斿顫楅懝?           | `working`, `episodic`, `semantic`, `profile`, `skill`, `reflection`, `custom` |
| `layer.shape`    | 鐠囥儱鐪伴惃鍕摠閸屻劌鑸伴幀?           | `Flat`, `Graph`                                                               |
| `layer.indices`  | 鐠囥儱鐪伴弨顖涘瘮閻ㄥ嫮鍌ㄥ鏇″厴閸?         | `vector`, `entity`, `temporal`, `keyword`, `graph`, `tag`                     |
| `layer.capacity` | 鐠囥儱鐪伴惃鍕暕缁?鐎瑰綊鍣虹粵鏍殣         | `token_limited`, `sliding_window`, `unlimited`                                |


鏉╂瑧顫掔悰銊с仛閹跺﹨绻冮崢璇层亣闁插繘鍣告径宥堫啎鐠佲剝濯堕幋鎰啊閺囧瓨顒滄禍銈囨畱缂佸嫬鎮庨敍?

- `Single-Flat` = 1 鐏?+ `shape=Flat`
- `Dual-Store` = 2 鐏?+ 娑撳秴鎮?`theme`
- `Graph-Centric` = 1 鐏?+ `shape=Graph`
- `Hybrid-Graph` = 婢舵艾鐪伴敍灞藉従娑擃厺绔寸仦?`Flat`閵嗕椒绔寸仦?`Graph`

閸ョ姵顒濋敍灞炬偝缁鳖澀绗夐崘宥嗘Ц閸︺劌鍤戞稉顏勬嚒閸氬秵瀚囬幍鎴滅闂傛潙鍨忛幑顫礉閼板本妲搁崷銊⑩偓婊冪湴閺?鑴?濮ｅ繐鐪版稉濠氼暯 鑴?濮ｅ繐鐪拌ぐ銏♀偓?鑴?濮ｅ繐鐪扮槐銏犵穿閳ユ繀绗傛潻娑滎攽缂佹挻鐎幖婊呭偍閵?

---

## A. Unit Formation 閳?鐠佹澘绻傞崡鏇炲帗瑜般垺鍨?

**閺嶇绺鹃梻顕€顣?*閿涙艾甯慨瀣翻閸忋儱顩ф担鏇炲瀼閸?鏉烆剙瀵叉稉?memory units閿?


| ID    | 鐎圭偟骞?                       | 閹诲繗鍫?                                 | 鏉堟挸鍤?unit_type | 閸忔娊鏁崣鍌涙殶                                                             |
| ----- | ------------------------- | ----------------------------------- | ------------ | ---------------------------------------------------------------- |
| UF-1  | **Segment(observation)**  | 閸樼喎顫?observation 閻╁瓨甯磋ぐ銏″灇娑撯偓娑?unit          | raw          | mode=`observation`                                               |
| UF-2  | **Segment(message)**      | 濮ｅ繑娼☉鍫熶紖閻欘剛鐝涢幋?unit                        | message      | mode=`message`                                                   |
| UF-3  | **Segment(turn)**         | 濮ｅ繗鐤嗙€电鐦?(user+assistant) 娴ｆ粈璐熸稉鈧稉?unit     | turn         | mode=`turn`, include_system=bool                                 |
| UF-4  | **Segment(chunk)**        | 閹稿娴愮€规艾銇囩亸蹇斿灗鐠囶厺绠熸潏鍦櫕閸掑棗娼?                       | chunk        | mode=`chunk`, chunk_size, overlap, strategy={fixed, semantic}    |
| UF-5  | **Extract(event)**        | 娴?observation 娑擃厽濞婇崣鏍瀲閺侊絼绨ㄦ禒?              | event        | target=`event`, method, schema, granularity, source_scope        |
| UF-6  | **Extract(fact)**         | 閹惰棄褰囨禍瀣杽閹冩嚒妫?(who/what/when/where)       | fact         | target=`fact`, method, schema, granularity, source_scope         |
| UF-7  | **Extract(entity_state)** | 閹惰棄褰囩€圭偘缍嬮崣濠傚従瑜版挸澧犻悩鑸碘偓?                         | entity_state | target=`entity_state`, method, entity_types, attribute_schema    |
| UF-8  | **Extract(triple)**       | 閹惰棄褰?(subject, predicate, object) 娑撳鍘撶紒?| triple       | target=`triple`, method, ontology, max_triples                   |
| UF-9  | **Extract(kv)**           | 閹惰棄褰?key-value 鐎电櫢绱欓崑蹇撱偨閵嗕礁鐫橀幀褏鐡戦敍?             | kv_pair      | target=`kv`, method, key_schema                                  |
| UF-10 | **Extract(skill)**        | 閹绘劕褰囬崣顖氼槻閻劋鍞惍?閹垛偓閼?                         | skill        | target=`skill`, method, include_code, include_description        |
| UF-11 | **Extract(thought)**      | 閹绘劕褰囬崘鍛村劥閹恒劎鎮婂銉╊€?                           | thought      | target=`thought`, method, granularity={step, chain}              |
| UF-12 | **Abstract(summary)**     | 閹跺﹣绔存稉?session / 缁愭褰涢崢瀣級娑?summary unit    | summary      | target=`summary`, method, source_scope, max_length               |
| UF-13 | **Abstract(reflection)**  | 娴犲簼鎹㈤崝陇寤烘潻閫涜厬瑜般垺鍨氶崣宥嗏偓?                         | reflection   | target=`reflection`, method, include_trajectory, include_outcome |
| UF-14 | **Delegate(freeform)**    | 鐠?LLM 閼奉亞鏁遍崘鍐茬暰閸掑洤鍨庨弬鐟扮础閵嗕胶鐭戞惔锕€鎷版潏鎾冲毉缁鐎?             | mixed        | model, instruction                                               |


### 閸欐ê绱撴潪?

- **閹垮秳缍旂猾璇茬€?*閿涙瓔egment / Extract / Abstract / Delegate
- **閻╊喗鐖ｇ猾璇茬€?*閿涙瓱vent / fact / entity_state / triple / kv / skill / thought / summary / reflection
- **缁帒瀹?*閿涙bservation 閳?message 閳?turn 閳?chunk 閳?event/fact 閳?triple/kv
- **缂佹挻鐎崠鏍柤鎼?*閿涙俺鍤滈悽杈ㄦ瀮閺?閳?娑撱儲鐗?schema
- **瑜般垺鍨氶弬瑙勭《**閿涙俺顫夐崚?/ parser / encoder / constrained LLM / free-form LLM

### 瀵ょ儤膩鐠囧瓨妲?

- `Segment` 鐠愮喕鐭楅崚鍥у瀻閵嗕焦澧﹂崠鍛嫲闁插秴鍨庨崸妤嬬幢鐎瑰啩绗夐弨鐟板綁閸愬懎顔愮拠顓濈疅閿涘苯褰ч弨鐟板綁 unit 鏉堝湱鏅妴?
- `Extract` 鐠愮喕鐭楁禒搴ょ翻閸忋儰鑵戦幎钘夊毉缂佹挻鐎崠鏍ㄥ灗閸楀﹦绮ㄩ弸鍕閸楁洖鍘撻敍娑氭埂濮濓綁鍣哥憰浣烘畱閹兼粎鍌ㄧ紒鏉戝閺?`target` 娑?`method`閿涘矁鈧奔绗夐弰顖欒礋濮ｅ繒顫?target 閸楁洜瀚崣鎴炴娑撯偓娑?primitive 閸氬秴鐡ч妴?
- `Abstract` 鐠愮喕鐭楅幎濠呯窛闂€鑳瘱閸ュ娈戞潏鎾冲弳閸樺缂夋稉娲彯鐏炲倸宕熼崗鍐跨礉婵?`summary` 閸?`reflection`閵?
- `Delegate` 閻劋绨拋鈺偰侀崹瀣躬瀵偓閺€鍓у箚婢у啩绗呴懛顏囶攽閸愬啿鐣捐ぐ銏″灇缁涙牜鏆愰敍宀勨偓鍌氭値娴ｆ粈璐熸妯垮殰閻㈠崬瀹抽崺铏瑰殠閵?
- 婢舵氨鐭戞惔锕€鑸伴幋鎰瑝閸愬秴宕熼悪顒€缂撳Ο鈥茶礋 `UF-14 MultiGranularity`閿涘矁鈧本妲搁柅姘崇箖 DSL 闁插瞼娈?`Compose` / `Cascade` 閺勫海鈥樼悰銊ㄦ彧閵?

### 鐢瓕顫?`method` 缁鐎?


| method                 | 閸氼偂绠?                                                | 鐢瓕顫嗛柅鍌滄暏 target                |
| ---------------------- | -------------------------------------------------- | -------------------------- |
| `rule_based`           | regex閵嗕焦膩閺夎￥鈧礁鍙ч柨顔跨槤閵嗕浇顕㈠▔鏇☆潐閸掓瑦濞婇崣?                               | kv, event                  |
| `schema_guided_llm`    | 閹稿绮扮€?schema/slot 婵夘偄鍘栫紒鎾寸€崠鏍х摟濞?                           | fact, entity_state, skill  |
| `constrained_decoding` | JSON schema / function calling / CFG 缁撅附娼潏鎾冲毉          | fact, entity_state, triple |
| `span_labeling`        | NER / trigger-argument / slot tagging              | event, entity_state        |
| `relation_extraction`  | 閸忓磭閮撮崚鍡欒閵嗕副penIE閵嗕椒绗侀崗鍐矋閹惰棄褰?                                 | triple, fact               |
| `ontology_guided`      | 閸︺劎绮扮€?ontology 娑擃厼浠涚€圭偘缍?閸忓磭閮?鐏炵偞鈧勫▕閸?                         | entity_state, triple       |
| `retrieval_assisted`   | 閸婄喎濮鍙夋箒 store 閸?entity disambiguation 閹?state linking | entity_state, fact         |
| `freeform_llm`         | 閸欘亞绮伴懛顏嗗姧鐠囶叀鈻堥幐鍥︽姢閿涘瞼鏁卞Ο鈥崇€烽懛顏嗘暠閸愬啿鐣炬潏鎾冲毉                                 | event, thought, reflection |


---

## B. Representation 閳?鐞涖劎銇氱紓鏍垳

**閺嶇绺鹃梻顕€顣?*閿涙nit 娴犮儰绮堟稊鍫濊埌瀵繐鐡ㄩ崒銊ユ嫲缁便垹绱╅敍?

閹跺€熻杽娑撳﹣绮涢弰?**representation element set**閿涘牊鐦℃稉?unit 闁瀚ㄩ幖鍝勭敨閸濐亙绨虹悰銊с仛閸忓啰绀岄敍澶堚偓鍌欑瑓闂?**B.1** 娑撳骸缍嬮崜?`memprimitive.baselines.representation` 鐎圭偟骞囨稉鈧稉鈧€电懓绨查敍?*B.2** 娣囨繄鏆€閹兼粎鍌ㄧ粚娲？闁插苯鐨婚張顏勬躬閸╄櫣鍤庢稉顓℃儰閸︽壆娈戦崗鍐閿涘奔绌舵禍搴℃嫲 DSL 鐎靛湱鍙庨妴?

### B.1 Stage-1 閸╄櫣鍤庨敍姝欱asicRepresentation` / `KeywordRepresentation`

- **濡€虫健**閿涙瓪BasicRepresentation`閵嗕梗KeywordRepresentation`閿涘牆鎮楅懓鍛Ц閽栧嫬鐨濈憗鍜冪礉姒涙顓?`elements=("text", "keywords", "tags")`閿涘苯鍙炬担娆愮€柅鐘插毐閺佹澘寮弫棰佺瑢閸撳秷鈧懐娴夐崥宀嬬礆閵?
- **鏉堟挸鍙?*閿涙瓪run` 鐟曚焦鐪?`packet.units` 瀹告彃鐡ㄩ崷顭掔幢**娑撳秳鎱ㄩ弨?* `MemoryStore`閵?
- **闁板秶鐤?*閿涙碍鐎柅鐘插毐閺?`elements: tuple[str, ...]`閿涘奔绱伴崢濠氬櫢楠炴湹绻氶幐渚€銆庢惔蹇ョ幢闂堢偞纭堕崥宥呯摟閸?`__init__` 閺?`ValueError`閵?
- **閸氬牊纭堕崗鍐閸?*閿涙矮绗屽┃鎰垳 `_VALID_ELEMENTS` 娑撯偓閼疯揪绱濋崗?12 娑擃亷绱?*濞屸剝婀?* `frame` / `code` / `sparse_embedding` 缁涘濞婄挒陇銆冩稉顓犳畱妞ょ櫢绱氶妴?

娑撳鍨悰銊︾壐鐠囧瓨妲戦敍?*閸忓啰绀岄崥?*閵?*閸愭瑥鍙嗘担宥囩枂**閿涘潉MemoryUnit` 鐎涙顔岄幋?`metadata["representation"]`閿涘鈧?*閺勵垰鎯佹潻娑樺弳** `representation_elements`閵?*鐎圭偟骞囬柅鏄忕帆閹芥顩?*閵?

| 閸忓啰绀岄崥?| 娑撴槒顩﹂崘娆忓弳娴ｅ秶鐤?| 鏉╂稑鍙?`representation_elements` | 鐞涘奔璐熼幗妯款洣 |
| ------ | ------------ | -------------------------------- | -------- |
| `text` | `unit.text`閿涘澃trip 閸氬函绱?| 閺勵垽绱欐慨瀣矒閸欘垱鐖ｉ敍?| 鐟欏嫯瀵栭崠鏍敄閻ц姤鏋冮張顒婄幢`normalized_text` 娑?casefold 閸忋劍鏋冮妴?|
| `embedding` | `unit.embedding: list[float]` | 閺勵垽绱欓幋鎰缂傛牜鐖滈崥搴礆 | `sentence_transformers.SentenceTransformer`閿涘牓绮拋?`MEMPRIMITIVE_EMBEDDING_MODEL` 閹?`all-MiniLM-L6-v2`閿涘绱漙normalize_embeddings=True`閿涙稒膩閸ㄥ瀵滈崥宥囩处鐎涙ǜ鈧?|
| `triple` | `unit.triples` | 娴犲懎缍嬮崚妤勩€冮棃鐐碘敄 | 娴兼ê鍘?`unit.metadata["triples"]` 娑擃厼鑸版俊?`[s,p,o]` 閻ㄥ嫰銆嶉敍娑樻儊閸掓瑧鏁ゅ锝呭灟娴犲孩顒滈弬鍥ㄥ▕ 閳ユ反 likes/prefers/閳?Y閳ユ績鈧反 is Y閳?濡€崇础閵?|
| `kv` | `unit.kv` | 娴犲懎缍?dict 闂堢偟鈹?| 娴兼ê鍘?`unit.metadata["kv"]`閿涙稑鎯侀崚?`Key: value` 鐞涘本膩瀵?+ likes/is 濡€崇础閻㈢喐鍨氶柨顔尖偓绗衡偓?|
| `entities` | `unit.entities` | 娴犲懎缍嬮崚妤勩€冮棃鐐碘敄 | 娴兼ê鍘?`unit.metadata["entities"]`閿涙稑鎯侀崚娆忋亣閸愭瑨鎹ｆ径瀵告畱鏉╃偟鐢荤拠宥呮健閿涘牆鎯庨崣鎴濈础 NER閿涘绱濇潻鍥ㄦ姢 the/a/an閵?|
| `tags` | `unit.tags` | 娴犲懎缍嬮崚妤勩€冮棃鐐碘敄 | 娴兼ê鍘?`unit.metadata["tags"]`閿涙稑鎯侀崚?`unit_type` + 閸忔娊鏁拠宥堛€冮敍鍧搑aph/memory/code/閳ワ讣绱? 閼汇儱鍑￠張?entities/kv/triples 閸掓瑥濮?`entity_rich` / `structured_kv` / `structured_triple`閵?|
| `keywords` | `metadata["representation"]["keywords"]` | 娴犲懎缍嬮崚妤勩€冮棃鐐碘敄 | 娴兼ê鍘?`unit.metadata["keywords"]`閿涙稑鎯侀崚娆愵劀閺傚洩鐦濇０鎴礄閸樿浠犻悽銊ㄧ槤閿涘本娓舵径?6 娑擃亷绱氶獮璺鸿嫙閸?entities/tags 閻ㄥ嫯藟閸忓懓鐦濋妴?|
| `summary` | `metadata["representation"]["summary"]` | 娴犲懎缍嬮悽鐔稿灇闂堢偟鈹?| 閸氼垰褰傚蹇庣閸欍儲鎲崇憰渚婄窗娴兼ê鍘涙＃鏍ㄦ蒋 triple閿涘苯鎯侀崚娆擃浕閺?kv閿涘苯鎯侀崚娆忓娑撱倓閲滅€圭偘缍?+ 閺傚洦婀伴崜宥囩磻閿涘苯鎯侀崚娆愬焻閺傤厽顒滈弬鍥风礄閳?6 鐎涙顑侀敍澶堚偓?|
| `time_anchor` | `metadata["representation"]["time_anchor"]` | 娴犲懎缍?dict 闂堢偟鈹?| 娴兼ê鍘?`unit.metadata["time_anchor"]`閿涘潐ict閿涘绱遍崥锕€鍨悽?`unit.timestamp` 閹峰棗鍤?`timestamp` / `date`閵?|
| `relation_tags` | `metadata["representation"]["relation_tags"]` | 娴犲懎缍嬮崚妤勩€冮棃鐐碘敄 | 娴兼ê鍘?`unit.metadata["relation_tags"]`閿涙稑鎯侀崚娆戞暠 triple 閻?predicate 閻㈢喐鍨?`predicate:...`閿涘苯鐤勬担鎾存殶閳? 閺冭泛濮?`multi_entity`閵?|
| `source_type` | `metadata["representation"]["source_type"]` | 娴犲懎缍嬮棃鐐碘敄 | 閺夈儴鍤?`unit.metadata["source"]` 鐎涙顑佹稉?strip閵?|
| `description` | `unit.description` | 娴犲懎缍嬮張鈧紒鍫ユ姜缁?| 閼?`unit.description` 瀹稿弶婀侀崚娆庣箽閻ｆ瑱绱遍崥锕€鍨懟銉╁帳缂冾喕绨?`api_key` + `base_url` + `model` 閸掓瑨鐨?OpenAI Chat 閻㈢喐鍨氭稉鈧崣銉﹀伎鏉╁府绱遍崥锕€鍨柅鈧崶鐐扮瑢 `summary` 閻╃鎮撻惃鍕儙閸欐垵绱￠幋鏍у斧閺傚洢鈧?|

濮濄倕顦诲В蹇旑偧 `run` 闁垝绱伴幎濠傛値楠炶泛鎮楅惃鍕喅鐟曚礁鍟撻崗?`unit.metadata["representation"]`閿涘牆鎯?`_representation_summary_from_unit` 娑撳簼绗傜悰銊よ厬閻ㄥ嫭澧跨仦鏇㈡暛閿涘鈧繖representation` slot 閻?trace 閸?`packet.trace["representation"]`閿涘牊膩閸ф鎮曢妴涔lements`閵嗕線鈧?unit 閻?`representation_elements`閿涘鈧?

### B.2 閹跺€熻杽閹兼粎鍌ㄧ粚娲？娑擃厾娈戦崗璺虹暊鐞涖劎銇氶崗鍐閿涘牆鐔€缁炬寧婀€圭偟骞囬敍?

娑撳鍨崷銊︻洤韫囧灚鏋冨?濡偓缁便垼顔曠拋鈥茶厬鐢瓕顫嗛敍灞肩稻 **瑜版挸澧?`_VALID_ELEMENTS` 娑撳秴瀵橀崥?*閿涘矂娓堕懛顏勭暰娑?`RepresentationModule` 閹存牕鍙剧€瑰啴妯佸▓浣兯夋鎰剁窗

| 濮掑倸搴烽崗鍐 | 閸氼偂绠?|
| -------- | ---- |
| `frame` | slot / frame 婵夘偄鍘栫紒鎾寸€?|
| `code` | 閸欘垱澧界悰灞煎敩閻焦鍨ㄩ悧鍥唽 |
| `sparse_embedding` | TF-IDF / BM25 缁涘鈻堥悿蹇撴倻闁?|

### B.3 閸欐ê绱撴潪缈犵瑢缁撅附娼敍鍫滅瑢鐎圭偟骞囩€靛綊缍堥柈銊ュ瀻閿?

- **閸忓啰绀岄柅澶嬪**閿涙艾婀?B.1 閻?12 娑擃亜鎮曠€涙ぞ鑵戦崑姘摍闂嗗棝鈧瀚ㄩ敍娌桲eywordRepresentation` 姒涙顓婚崑蹇旑梾缁鳖澁绱癭text + keywords + tags`閵?
- **閸氭垿鍣哄Λ鈧槐?*閿涙岸娓剁憰?`embedding` 閸忓啰绀岄敍鍫濆挤 `EmbeddingSimilarityRetrieval` 缁涘绱氶敍娑楃贩鐠ф牗婀伴崷?SentenceTransformer閵?
- **閸忔娊鏁拠?閺嶅洨顒风猾缁橆梾缁?*閿涙瓪keywords` / `tags` / `entities` 娑?`KeywordCountRetrieval`閵嗕梗TagRetrieval`閵嗕梗EntityRetrieval` 缁涘鍘ら崥鍫幢闁劌鍨?signal閿涘牆顩?`KeywordMatchSignal`閿涘顕?`metadata["representation"]["keywords"]`閵?
- **缂佹挻鐎崠?*閿涙瓪triple` / `kv` 閸欘垳鏁?UF 閻?metadata 妫板嫬锝為敍灞肩瘍閸欘垰鍙忛棃?representation 閸愬懓顫夐崚娆庣矤閺傚洦婀伴幎濮愨偓?

---

## C. Trigger 閳?鐟欙箑褰傞崘宕囩摜

**閺嶇绺鹃梻顕€顣?*閿涙矮绮堟稊鍫熸閸婃瑨顕氱憴锕€褰傞弻鎰嚋 memory 閻㈢喎鎳￠崨銊︽埂閸斻劋缍旈敍?

Stage-1 閹跺﹤褰叉径宥囨暏闁槒绶弨璺烘躬 **`memprimitive.baselines._trigger_family`**閿涘瞼鏁遍崶娑楅嚋**缂佸嫪娆㈢憴鎺曞**閹峰吋鍨氭稉鈧弶鈩冪槨娑?unit 閻ㄥ嫬鍠呯粵鏍懠閿涘牅绗夐弰?pipeline 閻?DSL slot 閸氬稄绱濋懓灞炬Ц family 閸愬懘鍎寸€涙劗绮ㄩ弸鍕剁礆閿?

```text
TriggerFamily = signal_providers閿涘牆顦跨捄?signal閿?+ scorer + gate + policy
```

鏉╂劘顢戦弮?**`TriggerFamilyRunner`** 鐎佃鐦℃稉?`packet.units[i]` 妞ゅ搫绨幍褑顢戦敍姘値楠炶埖澧嶉張?`SignalProvider.provide` 閳?`ScoreAggregator.score` 閳?`Gate.evaluate` 閳?`DecisionPolicy.decide`閿涘苯绶遍崚?`bool` 閸愭瑥鍙?`Packet.decisions`閿涘澋rite閿涘鍨?`Packet.decisions`閿涘潒volution閿涘鈧?

* **`write_trigger`**閿涙岸绮拋銈呯唨缁?`AlwaysTrigger`閵嗕梗ThresholdTrigger`閿涙稑褰查悽?`compose_write_trigger(...)` 閼奉亜鐣炬稊?family閵?
* **`evolution_trigger`**閿涙岸绮拋?`NeverTrigger`閵嗕梗ThresholdTrigger`閿涙稑褰查悽?`compose_evolution_trigger(...)`閵嗗倿鈧倿鍘ら崳銊╃帛鐠併倛顩﹀Ч?`packet.placements` 闂堢偟鈹栭敍鍧剅equired_fields`閿涘鈧?

閸忓彉闊╂稉濠佺瑓閺?**`TriggerContext`**閿涘潚rozen dataclass閿涘绱癭packet`閵嗕梗store`閵嗕梗output_field`閵嗕梗trace_key`閿涘牅绶?gate/signal 鐠?packet 閻樿埖鈧緤绱盽store` 瑜版挸澧?runner 閸愬懏婀鍝勫煑娴ｈ法鏁ら敍灞肩稻娣囨繄鏆€閹碘晛鐫嶉悙鐧哥礆閵?

### C.1 閸ユ稐閲滅紒鍕閻ㄥ嫭濞婄挒鈩冨复閸欙綇绱欒箛鍛淬€忕€圭偟骞囬惃鍕煙濞夋洩绱?

| 鐟欐帟澹?| ABC | 鐎圭偘绶ョ仦鐐粹偓?| 閹跺€熻杽閺傝纭?|
| ---- | --- | -------- | -------- |
| Signal | `SignalProvider` | `name: str`閿涘牆鐤勯悳鎵閻?`@property`閿?| `provide(self, context: TriggerContext, unit_index: int) -> SignalMap`閿涘畭SignalMap = dict[str, float \| bool]`閿涙稑褰茬€电懓鎮撴稉鈧?unit 鏉╂柨娲栨径姘嚋闁款喓鈧?|
| Scorer | `ScoreAggregator` | `name: str` | `score(self, signals: SignalMap) -> float` |
| Gate | `Gate` | `name: str` | `evaluate(self, context: TriggerContext, unit_index: int, *, signals: SignalMap, score: float) -> bool`閿涘潉True` 鐞涖劎銇氶柅姘崇箖绾剟妫幒褝绱?|
| Policy | `DecisionPolicy` | `name: str` | `decide(self, *, score: float, gate_open: bool) -> bool`閿涘牊娓剁紒鍫熸Ц閸氾箒袝閸欐埊绱?|

Runner 閸愬懎顕径姘嚋 signal provider 閻ㄥ嫯绻戦崶鐐测偓鐓庝粵 **`signals.update(provided)`**閿涘本鏅犻崥搴も偓鍛倱閸氬秹鏁导姘愁洬閻╂牕澧犻懓鍛偓?

### C.2 Signal 缂佸嫪娆㈤敍鍧凷ignalProvider` 鐎圭偟骞囩猾浼欑礆

| 缁鎮?| 鐎圭偟骞囬惃鍕▕鐠炩剝鏌熷▔?| 姒涙顓绘潏鎾冲毉娣団€冲娇闁?| 鐞涘奔璐熸稉搴″缂冾喗娼禒?|
| ---- | -------------- | -------------- | -------------- |
| `ConstantSignal` | `provide` | `signal_name`閿涘牓绮拋?`"constant"`閿?| 鐢憡鏆?`float(value)`閵?|
| `UnitLengthSignal` | `provide` | `signal_name`閿涘牓绮拋?`"unit_length"`閿?| 瑜版挸澧?unit 閺傚洦婀?strip 閸氬酣鏆辨惔锔肩幢閼?`normalize_by>0` 閸掓瑩娅庢禒銉嚉閸婄鈧?|
| `KeywordMatchSignal` | `provide` | `signal_name`閿涘牓绮拋?`"keyword_match"`閿?| **`packet.query` 韫囧懘銆忛棃鐐碘敄**閿涙硛uery 鐠囧秳绗?`representation["keywords"]` 閸?unit 閺傚洦婀扮拠宥囨畱娴溿倝娉﹂弫浼村櫤閿涘潚loat閿涘鈧?|
| `HasEntitySignal` | `provide` | `signal_name`閿涘牓绮拋?`"has_entity"`閿?| 閺堝鐤勬担鎾冲灟 `1.0` 閸氾箑鍨?`0.0`閵?|
| `HasTripleSignal` | `provide` | `signal_name`閿涘牓绮拋?`"has_triple"`閿?| 閺?triple 閸?`1.0` 閸氾箑鍨?`0.0`閵?|
| `HasKVSignal` | `provide` | `signal_name`閿涘牓绮拋?`"has_kv"`閿?| 閺?kv 閸?`1.0` 閸氾箑鍨?`0.0`閵?|
| `TagMatchSignal` | `provide` | `signal_name`閿涘牓绮拋?`"tag_match"`閿?| **`packet.query` 韫囧懘銆忛棃鐐碘敄**閿涙硛uery 鐠囧秳绗?`unit.tags` 娴溿倝娉﹂弫浼村櫤閵?|
| `LayerTargetSignal` | `provide` | `signal_name`閿涘牓绮拋?`"layer_target"`閿?| **`packet.placements` 韫囧懘銆忛棃鐐碘敄**閿涙矖placements[unit_index].target_layer` 閺勵垰鎯侀崷?`allowed_layers` 閸愬拑绱濋弰顖氬灟 `1.0` 閸氾箑鍨?`0.0`閵?|
| `QueryOverlapSignal` | `provide` | `signal_name`閿涘牓绮拋?`"query_overlap"`閿?| **`packet.query` 韫囧懘銆忛棃鐐碘敄**閿涙硛uery 娑?unit 閺傚洦婀伴惃?token 娴溿倝娉﹂弫浼村櫤閵?|

### C.3 Scorer 缂佸嫪娆㈤敍鍧凷coreAggregator` 鐎圭偟骞囩猾浼欑礆

| 缁鎮?| 鐎圭偟骞囬惃鍕▕鐠炩剝鏌熷▔?| 鐞涘奔璐?|
| ---- | -------------- | ---- |
| `IdentityScorer` | `score` | `signals[source]` 鏉烆兛璐?`float`閿涙稓宸遍柨顔煎灟 `ValueError`閵?|
| `WeightedSumScorer` | `score` | `sum(signals[k] * weights[k])`閿涙稓宸辨禒璁崇闁款喖鍨?`ValueError`閵?|
| `MaxScorer` | `score` | `sources` 娑擃厺淇婇崣椋庢畱閺堚偓婢堆冣偓绗衡偓?|
| `MinScorer` | `score` | `sources` 娑擃厺淇婇崣椋庢畱閺堚偓鐏忓繐鈧鈧?|
| `AverageScorer` | `score` | `sources` 娑擃厺淇婇崣椋庢畱缁犳婀抽獮鍐叉綆閵?|
| `ClippedWeightedSumScorer` | `score` | 閸忓牊瀵?`WeightedSumScorer` 濮瑰倸鎷伴敍灞藉晙 clip 閸?`[min_score, max_score]`閵?|

### C.4 Gate 缂佸嫪娆㈤敍鍧凣ate` 鐎圭偟骞囩猾浼欑礆

| 缁鎮?| 鐎圭偟骞囬惃鍕▕鐠炩剝鏌熷▔?| 鐞涘奔璐熸稉搴″缂冾喗娼禒?|
| ---- | -------------- | -------------- |
| `AlwaysOpenGate` | `evaluate` | 閹帊璐?`True`閵?|
| `RequireEntityGate` | `evaluate` | 瑜版挸澧?unit 閺?`entities` 閹?`True`閵?|
| `RequireTripleGate` | `evaluate` | 瑜版挸澧?unit 閺?`triples` 閹?`True`閵?|
| `RequireTagGate` | `evaluate` | `unit.tags` 娑?`required_tags`閿涘潏asefold閿涘婀佹禍銈夋肠閹?`True`閵?|
| `LayerAllowedGate` | `evaluate` | **`packet.placements` 韫囧懘銆忛棃鐐碘敄**閿涙矖target_layer` 韫囧懘銆忛崷?`allowed_layers` 娑擃厹鈧?|
| `QueryPresentGate` | `evaluate` | `packet.query is not None` 閺?`True`閵?|

### C.5 Policy 缂佸嫪娆㈤敍鍧凞ecisionPolicy` 鐎圭偟骞囩猾浼欑礆

| 缁鎮?| 鐎圭偟骞囬惃鍕▕鐠炩剝鏌熷▔?| 鐞涘奔璐?|
| ---- | -------------- | ---- |
| `AlwaysPolicy` | `decide` | 閹?`True`閿涘牆鎷烽悾?score/gate閿涘鈧?|
| `NeverPolicy` | `decide` | 閹?`False`閵?|
| `ThresholdPolicy` | `decide` | `gate_open and score >= threshold`閵?|
| `BooleanGatePolicy` | `decide` | 閻╁瓨甯存潻鏂挎礀 `gate_open`閵?|
| `BandPassThresholdPolicy` | `decide` | `gate_open and lower <= score <= upper`閵?|
| `ThresholdOrGatePolicy` | `decide` | `gate_open or score >= threshold`閵?|

### C.6 `TriggerFamilyRunner.run` 娑?trace

- **閸撳秶鐤?*閿涙瓪packet.units` 闂堢偟鈹栭敍娑樿嫙閹稿鈧倿鍘ら崳銊ょ炊閸忋儳娈?`required_fields` 濡偓閺?`query` / `placements` 缁涘鐡у▓鐢告姜 `None`閵?
- **鏉堟挸鍤?*閿涙艾婀?`packet` 娑撳﹨顔曠純?`output_field`閿涘潉decisions` 閹?`decisions`閿涘璐?`list[bool]`閿涙矖trace[trace_key]` 閸?`family`閵嗕梗policy`/`scorer`/`gate` 閻?`name`閵嗕梗per_unit`閿涘牊鐦￠崡鏇炲帗閻?signals/score/gate/decision閿涘鈧?

### C.7 姒涙顓婚崺铏瑰殠娑撳海绮嶉崥?API

| Pipeline slot | 濞夈劌鍞界猾?| 閸愬懐鐤?family閿涘牊鎲崇憰渚婄礆 |
| ------------- | ------ | ------------------- |
| `write_trigger` | `AlwaysTrigger` | `ConstantSignal(1.0)` + `IdentityScorer("constant")` + `AlwaysOpenGate` + `AlwaysPolicy` |
| `write_trigger` | `ThresholdTrigger` | 閸氬奔绗?signal/scorer/gate + `ThresholdPolicy(threshold)`閿涘澃core 鐎圭偘璐?`constant` 閸旂姵娼?1.0閿?|
| `evolution_trigger` | `NeverTrigger` | 閸氬奔绗?+ `NeverPolicy` |
| `evolution_trigger` | `ThresholdTrigger` | 閸氬奔绗?+ `ThresholdPolicy(threshold)` |

閼奉亜鐣炬稊澶涚窗 **`compose_write_trigger`** / **`compose_evolution_trigger`**閿涘潉write_trigger.py` / `evolution_trigger.py`閿涘绱濇导鐘插弳 `signal_providers`閵嗕梗scorer`閵嗕梗gate`閵嗕梗policy` 閸欏﹤褰查柅?`input_requirements`閿涘牓娼?`units` 閻ㄥ嫰銆嶆导姘崇箻閸?`required_fields`閿涘苯婀?runner 闁插苯宸遍崚?packet 鐎涙顔岀€涙ê婀敍澶堚偓?

### C.8 濮掑倸搴烽幍鈺佺潔閿涘牆缍嬮崜宥勫敩閻椒鑵戦張顏呭絹娓氭稓娈戠紒鍕瑜般垺鈧緤绱?

娑撳銆冩笟澶哥艾娑撳孩娲跨€圭晫娈戦幖婊呭偍缁屾椽妫跨€靛湱鍙庨敍?*閺堫亜婀?`_trigger_family.py` 娑擃厼鐤勯悳?*閿?

- Signal閿涙瓪importance` / `novelty` / `surprise` / `duplicate_risk`閵嗕俯LM 娴兼媽顓哥粵澶堚偓?
- Scorer閿涙瓪product`閵嗕梗rule_expr`閵嗕胶鍑?`llm_judge`閵?
- Policy閿涙瓪top_k_per_window`閵嗕梗sample_by_score`閵嗕梗explicit_only`閿涘牆浼愰崗鐤殶閻劑妫幒褝绱氱粵澶堚偓?
- Gate閿涙瓪on_event`閵嗕梗not_duplicate`閵嗕梗tool_called` 缁涘绨ㄦ禒?瀹搞儱鍙跨拫鎾圭槤閵?

### C.9 鐢瓕顫嗙粵鏍殣閸掔増顒濆鍡樼仸閻ㄥ嫭妲х亸鍕剁礄濮掑倸搴烽敍?


| 閸樼喎鍟撳▔?                   | 濮濄倖顢嬮弸璺哄晸濞?                                                                 |
| ---------------------- | ---------------------------------------------------------------------- |
| `Always`               | `policy=always`                                                        |
| `Never`                | `policy=never`                                                         |
| `ImportanceThreshold`  | `signals={importance}, scorer=identity, policy=threshold`              |
| `NoveltyThreshold`     | `signals={novelty}, scorer=identity, policy=threshold`                 |
| `ImportanceAndNovelty` | `signals={importance, novelty}, scorer=weighted_sum, policy=threshold` |
| `LLMJudge`             | `signals={unit, context, store}, scorer=llm_judge, policy=threshold`   |
| `AgentToolCall`        | `gates={tool_called(...)}, policy=explicit_only`                       |
| `RuleBased`            | `gates={predicate(...)}, scorer=rule_expr, policy=boolean_gate`        |
| `OnEvent`              | `gates={on_event(...)}, policy=boolean_gate`                           |
| `SurpriseGated`        | `signals={surprise}, scorer=identity, policy=threshold`                |
| `SampledWrite`         | `signals={...}, scorer=..., policy=sample_by_score`                    |
| `DuplicateAwareWrite`  | `gates={not_duplicate(...)}, policy=boolean_gate`                      |

### Stage-1 Runtime Mapping

閸︺劌缍嬮崜?`memprimitive` 鐎圭偟骞囨稉顓ㄧ礉ingest 妞ゅ搫绨稉鐚寸窗

```text
unit_formation -> representation -> write_trigger -> organization -> evolution_trigger -> memory_evolution
```

鐎电懓绨查惃鍕殶閹诡噣娼扮痪锕€鐣鹃弰顖ょ窗

* `write_trigger` 娴溠冨毉 `Packet.decisions`
* `organization` 鐠囪褰?`decisions` 楠炶泛鐣幋鎰埗鐟欏嫬鍟撻崗?
* `evolution_trigger` 娴溠冨毉 `Packet.decisions`
* `memory_evolution` 鐠囪褰?`decisions` 楠炶埖澧界悰宀勵杺婢舵牗绱ㄩ崠?

婵″倹鐏夐弻鎰嚋 runtime 娴犲秴鍘戠拋?`memory_evolution` 閸?`decisions` 娑撹櫣鈹栭弮璺烘礀闁偓閸?`decisions`閿涘苯绨茬憴鍡曡礋閸氭垵鎮楅崗鐓庮啇鐞涘奔璐熼敍宀冣偓灞肩瑝閺勵垳娲伴弽鍥嚔娑斿鈧?


---

## D. Organization 閳?閸忓磭閮磋ぐ鎺旂矋閵嗕焦鏂佺純顔荤瑢鐢瓕顫夐崘娆忓弳

**閺嶇绺鹃梻顕€顣?*閿涙nit 鎼存棁顕氶弨鎯ф躬閸濐亪鍣烽敍鐔剁瑢瀹稿弶婀?memory 瀵よ櫣鐝涙禒鈧稊鍫濆彠缁紮绱甸崷?ingest-time 鐢瓕顫夐崘娆忓弳鐠侯垰绶炴稉濠傤洤娴ｆ洘顒滃蹇氭儰閸?store閿?

鏉╂瑩鍣锋稉宥呭晙閹?organization 瀵ょ儤膩閹存劕绶㈡径姘嚒閸氬秶鐡ラ悾銉礉閼板本妲哥紒鐔剁閸愭瑦鍨氶敍?

```text
Organization = placement + links
```

閸忔湹鑵?**`placement` 鐞涖劎銇?unit 鐞氼偊鈧礁鍩岄崫顏堝櫡**閿涘牏娲伴弽?layer / 閸掑棗灏?/ 閺勬儳绱￠拃鐣屽仯閿涘绱濇稉宥呭晙閸楁洜瀚担璺ㄦ暏 `routing` 娑撯偓鐠囧稄绱遍崚鎷屾彧閻╊喗鐖ｇ仦鍌欑閸?*婵″倷缍嶉崘娆忓弳閵嗕礁顩ф担鏇氱瑢缂佹挻鐎紒鎾虫値**閿涘潊ppend閵嗕礁鍨庡韬测偓浣戒粵缁眹鈧礁缂撻崶鎹愬Ν閻愬湱鐡戦敍澶婃躬閺傚洦銆傞柌宀€袨娑?*鐏炲倸鍞撮崘娆忓弳瑜般垺鈧?*閿涘奔绗?`placement` 閻ㄥ嫮娲伴弽鍥偓澶嬪濮濓絼姘﹂敍宀冾潌娑撳濡粭顑跨癌瀵姾銆冮妴?

娴ｅ棗鐣犻惃?contract 闂団偓鐟曚焦鏁奸幋鎰剁窗

```text
Organization handles ingest-time organization and normal write.
```

楠炴湹绗栭弰鎯х础閹佃儻顓婚敍?

```text
Organization is topology-constrained.
StoreTopology defines the admissible placement targets, link types, and within-layer write shapes.
```

### 缂佸嫮绮愮紒鍕


| 缂佸嫪娆?         | 閸氼偂绠?                  | 閸忕鐎烽崣鏍р偓?                                                                                  |
| ----------- | -------------------- | -------------------------------------------------------------------------------------- |
| `placement` | unit **闁礁鍩岄崫顏冪鐏?/ 閸濐亙绔撮崚鍡楀隘**閿涘牏娲伴弽鍥儰閻愮櫢绱?| `default`, `by_unit_type`, `by_tag`, `by_rule`, `explicit`, `agent_selected`           |
| `links`     | 閸愭瑥鍙嗛弮鏈电瑢瀹稿弶婀?memory 瀵よ櫣鐝涢崫顏冪昂閸忓磭閮?| `temporal`, `entity`, `similarity`, `cluster_membership`, `graph_edge`, `parent_child` |


### 鐢瓕顫?placement閿涘牏娲伴弽鍥х湴 / 閸掑棗灏敍?


| placement        | 閸氼偂绠?               | 閸忔娊鏁崣鍌涙殶                 |
| ---------------- | ----------------- | -------------------- |
| `default`        | 閹粯妲搁崘娆忓弳姒涙顓?layer      | target_layer         |
| `by_unit_type`   | 閹?unit_type 闁瀚ㄩ惄顔界垼鐏?  | route_map            |
| `by_tag`         | 閹?tag 閹?topic 闁瀚ㄩ惄顔界垼鐏?| tag_field, route_map |
| `by_rule`        | 閹稿顫夐崚娆愭蒋娴犲爼鈧瀚ㄩ惄顔界垼鐏?      | rules                |
| `explicit`       | 娑撳﹥鐖剁€涙顔屾稉顓炲嚒鐢附婀侀惄顔界垼娴ｅ秶鐤?     | field                |
| `agent_selected` | agent 闁俺绻冨銉ュ徔閹存牕鐡у▓鍨瘹鐎规氨娲伴弽?| tool_to_layer        |


### 鐏炲倸鍞撮崘娆忓弳瑜般垺鈧緤绱欓崚鎷屾彧閻╊喗鐖ｇ仦鍌欑閸氬函绱?

娑撳簺鈧矂鈧礁鍩岄崫顏傗偓宥嗩劀娴溿倧绱伴崗鍫㈡暠 `placement` 闁鐣鹃惄顔界垼鐏?閸掑棗灏敍灞藉晙閸愬啿鐣鹃崷銊嚉鐏炲倸鍞存俊鍌欑秿閽€钘夋勾閵?


| 閸愭瑥鍙嗚ぐ銏♀偓?             | 閸氼偂绠?             | 閸忔娊鏁崣鍌涙殶                               |
| ------------------- | --------------- | ---------------------------------- |
| `append`            | 閻╁瓨甯存潻钘夊閸掓壆娲伴弽?layer   | 閳?                                 |
| `partition`         | 閸愭瑥鍙嗛弻鎰嚋閸掑棗灏?/ bucket | partition_key                      |
| `cluster`           | 閺€鎯у弳閺堚偓鏉╂垹娈?cluster   | similarity_threshold, max_clusters |
| `graph_node`        | 閸︺劌娴樼仦鍌欒厬閸掓稑缂撻懞鍌滃仯楠炶埖甯存潏?    | node_policy                        |
| `hierarchical_slot` | 閺€鎯у弳婢舵艾鐪扮紒鎾寸€稉顓犳畱閺屾劒绔寸仦?濡叉垝缍? | target_level, slot_policy          |


### 鐢瓕顫?links


| link                 | 閸氼偂绠?                      | 閸忔娊鏁崣鍌涙殶                          |
| -------------------- | ------------------------ | ----------------------------- |
| `temporal`           | 瀵よ櫣鐝涢弮鍫曟？闁粯甯撮崗宕囬兇                 | window, direction             |
| `entity`             | 閹稿鍙℃禍顐㈢杽娴ｆ挸缂撶粩瀣彠閼?               | entity_field, link_type       |
| `similarity`         | 瀵よ櫣鐝涢惄闀愭妧鎼达箒绻庨柇璇插彠缁?               | metric, threshold             |
| `cluster_membership` | 鏉╃偞甯撮崚鐗堢厙娑?cluster / centroid | cluster_key                   |
| `graph_edge`         | 閸︺劌娴樼仦鍌欒厬閸掓稑缂撻張澶岃閸ㄥ绔?              | edge_types, connection_method |
| `parent_child`       | 瀵よ櫣鐝涚仦鍌滈獓閻栬泛鐡欓崗宕囬兇                 | level_policy                  |


### 閸欐ê绱撴潪?

- **閻╊喗鐖ｉ拃鐣屽仯閿涘潷lacement閿?*閿涙艾骞撻崫顏冪鐏?/ 閸濐亙閲滈崚鍡楀隘
- **閸忓磭閮寸紒鎾寸€敍鍧檌nks閿?*閿涙艾缂撶粩瀣憿娴?link
- **鐏炲倸鍞撮崘娆忓弳瑜般垺鈧?*閿涙瓫ppend / partition / cluster / graph / hierarchical
- **鐢瓕顫夐崘娆忓弳閺傜懓绱?*閿涙氨鏁?organization strategy 闂呮劕鎯堥崘鍐茬暰閻?ingest-time update
- **缂佹挻鐎痪锔芥将閺夈儲绨?*閿涙氨鏁?StoreTopology 閸愬啿鐣鹃崣顖炩偓澶庡厴閸旀稖绔熼悾?

### Topology-Constrained Organization

- `StoreTopology` 閸愬啿鐣?admissible placement targets閿涙碍鐥呴張澶屾畱 layer 娑撳秷鍏樻担婊€璐熼拃鐣屽仯閵?
- `StoreTopology` 閸愬啿鐣?admissible link types閿涙矮绶ユ俊鍌涚梾閺?graph layer 閹?`graph` index閿涘苯姘ㄦ稉宥堫嚉閸忎浇顔?`graph_edge`閵?
- `StoreTopology` 缁撅附娼仦鍌氬敶閸愭瑥鍙嗚ぐ銏♀偓渚婄窗娓氬顩ч崡鏇炵湴 flat store 娑撳秴绨茬拠銉ュ帒鐠?`hierarchical_slot`閵?
- `Organization` 鐠愮喕鐭楅崷銊ㄧ箹娴滄稖鍏橀崝娑滅珶閻ｅ苯鍞撮敍宀勨偓澶嬪閸忚渹缍嬮惃?placement / links / 鐏炲倸鍞撮崘娆忓弳瑜般垺鈧緤绱濋獮璺虹暚閹存劕鐖剁憴鍕晸閸忋儯鈧?
- **鐏炲倸鍞?`append`** 娑撳秴鍟€閹板繐鎳楅惈鈧垾婊冨涧閻㈢喐鍨氱拋鈥冲灊閳ユ繐绱濋懓灞惧壈閸涘磭娼?append-style normal write閿涘牅绗岃ぐ鎾冲 stage-1 閸╄櫣鍤庢稉?`MemoryStore.append` 娑撯偓閼疯揪绱氶妴?
- 閺屾劒绨?organization 閸欐ü缍嬮崣顖欎簰閸栧懎鎯堟稉?placement閿涘牏娲伴弽鍥ㄥ灗鐏炲倸鍞磋ぐ銏♀偓渚婄礆瀵缚鈧箑鎮庨惃?ingest-time merge / upsert / replace閵?
- 鏉╂瑤绨?ingest-time update 娑撳秴宕熼悪顒佸閹存劖鏌?slot閿涘苯娲滄稉鍝勭暊娴狀兛绗夐弸鍕灇閻欘剛鐝涢幖婊呭偍鏉炴番鈧?

### 閸忔娊鏁痪锔芥将

- `links contains entity` 鐟曚焦鐪?unit 閺?`entities` 鐎涙顔?閳?闂団偓鐟?UF 娴溠冨毉 entity 閹?Rep 鐞涖儱鍘?entity
- `placement=by_tag` 鐟曚焦鐪?unit/representation 娑擃厽婀?`tags`
- 鐏炲倸鍞磋ぐ銏♀偓?`cluster` 閹?`links contains similarity` 闁艾鐖剁憰浣圭湴 `embedding`
- 鐏炲倸鍞磋ぐ銏♀偓?`graph_node` 閹?`links contains graph_edge` 鐟曚焦鐪扮€涙ê婀?`shape=Graph` 閻?layer閿涘奔绗栭柅姘埗鐟曚焦鐪?`graph` index
- 鐏炲倸鍞磋ぐ銏♀偓?`hierarchical_slot` 閹?`links contains parent_child` 鐟曚焦鐪?layer_count > 1
- `placement=by_unit_type`閵嗕梗by_rule`閵嗕梗agent_selected` 缁涘鍏樼憰浣圭湴鐎电懓绨查惄顔界垼 layer 閸︺劌缍嬮崜宥嗗珖閹垫垳鑵戠€涙ê婀?

### 閺冄冩嚒閸氬秶鐡ラ悾銉ュ煂閺傜増顢嬮弸鍓佹畱閺勭姴鐨?

娑撳鏋?**`placement=`** 娴犲懓銆冪粈?*閻╊喗鐖ｉ拃鐣屽仯**閿涘牓鈧礁鍩岄崫顏冪鐏?/ 閸濐亙绔撮崚鍡楀隘閿涘绱?*`write=`** 鐞涖劎銇?*鐏炲倸鍞撮崘娆忓弳瑜般垺鈧?*閿涘牅绗屾稉濠冩瀮閵嗗苯鐪伴崘鍛晸閸忋儱鑸伴幀浣碘偓宥堛€冩稉鈧懛杈剧礆閿涘矂浼╅崗宥勭瑢 `placement` 鐠囶厺绠熷ǎ閿嬬┋閵?

| 閺冄冨晸濞?                    | 閺傛澘鍟撳▔?                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `FlatAppend`            | `placement=default, links={}, write=append`                        |
| `TemporalAppend`        | `placement=default, links={temporal}, write=append`                |
| `EntityLinked`          | `placement=default, links={entity}, write=append`                  |
| `TemporalAndEntity`     | `placement=default, links={temporal, entity}, write=append`        |
| `GraphPlacement`        | `placement=default, links={graph_edge(...)}, write=graph_node`     |
| `HierarchicalPlacement` | `placement=default, links={parent_child}, write=hierarchical_slot` |
| `DualStoreRouter`       | `placement=by_unit_type(...), links={}, write=append`              |
| `AgentExplicitTarget`   | `placement=agent_selected(...), links={}, write=append`            |
| `SimilarityCluster`     | `placement=default, links={cluster_membership}, write=cluster`     |
| `TagBasedPlacement`     | `placement=by_tag(...), links={}, write=partition`                 |


---

## E. Memory Evolution 閳?鐠佹澘绻傚鏂垮

**閺嶇绺鹃梻顕€顣?*閿涙艾婀敮姝岊潐閸愭瑥鍙嗗鑼病鐎瑰本鍨氭稊瀣倵閿涘tore 閺勵垰鎯佹潻姗€娓剁憰渚€顤傛径鏍畱濠曟柨瀵查敍?*

鏉╂瑩鍣风紒鐔剁閸氬憡鏁归崢鐔告降閻?`Compression`閵嗕梗Maintenance`閿涘奔浜掗崣濠囧亝娴?*姒涙顓绘稉宥呮儙閸斻劊鈧線顤傛径鏍曢崣?* 閻ㄥ嫰鍣搁崘?闁插秵鏆ｉ幙宥勭稊閵? 
鐎瑰啩绗夐崘宥嗗閹峰懏娅橀柅?append 瀵繗鎯ゆ惔鎾扁偓?

```text
MemoryEvolution = selection + action + effect + trigger
```

### 濠曟柨瀵茬紒鍕


| 缂佸嫪娆?         | 閸氼偂绠?                | 閸忕鐎烽崣鏍р偓?                                                                                                                                                                               |
| ----------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `selection` | 闁鑵戦崫顏冪昂瀹稿弶婀?memory 閸欏倷绗屽鏂垮 | `matched_by_key`, `matched_by_entity`, `time_window`, `layer_slice`, `low_activity`, `all`                                                                                           |
| `action`    | 鐎靛綊鈧鑵戠€电钖勯弬钘夊娴犫偓娑斿牊绱ㄩ崠鏍ㄦ惙娴?     | `replace`, `merge`, `upsert`, `rewrite`, `summarize`, `reflect`, `profile_update`, `extract_concept`, `prototype_form`, `prune`, `dedup`, `move`, `consolidate`, `review`         |
| `effect`    | 鐎?store 闁姵鍨氭担鏇狀潚閻樿埖鈧礁褰夐崠?  | `add`, `modify`, `delete`, `move`, `summarize`, `merge`, `version`                                                                                                                  |
| `trigger`   | 娴ｆ洘妞傞幍褑顢戠拠銉︾川閸栨牗鎼锋担?         | `periodic`, `on_event`, `budget_exceeded`, `count_exceeded`, `conditional`                                                                                                          |


### 鐢瓕顫?selection


| selection           | 閸氼偂绠?             | 閸忔娊鏁崣鍌涙殶                      |
| ------------------- | --------------- | ------------------------- |
| `matched_by_key`    | 閹?key 閸栧綊鍘ゅ鍙夋箒 unit | key_field, match_strategy |
| `matched_by_entity` | 閹?entity 閸栧綊鍘ゅ鍙夋箒鐠佹澘绻?| entity_field              |
| `time_window`       | 婢跺嫮鎮婇弻鎰嚋閺冨爼妫跨粣妤€褰涢崘鍛畱鐠佹澘绻?   | start, end / window_size  |
| `layer_slice`       | 婢跺嫮鎮婇弻鎰湴閹存牗鐓囬崚鍡楀隘娑擃厾娈戠€电钖?   | target_layer, filter      |
| `low_activity`      | 婢跺嫮鎮婃担搴㈡た鐠哄啨鈧椒缍嗘禒宄扳偓鐓庮嚠鐠?    | activity_threshold        |
| `all`               | 婢跺嫮鎮婇弫缈犻嚋閻╊喗鐖ｉ梿鍡楁値        | scope                     |


### 鐢瓕顫?action


| action             | 閸氼偂绠?             | 鐎电懓绨查弮褍鎮?                  |
| ------------------ | --------------- | ---------------------- |
| `replace`          | 閺囨寧宕查崠褰掑帳鐎电钖?         | Upd-2                  |
| `merge`            | 閸氬牆鑻熼弬鐗堟＋鐎电钖?         | Upd-3, Upd-7           |
| `upsert`           | 閺堝鍨弴瀛樻煀閿涘本妫ら崚娆愬絻閸?      | Upd-4                  |
| `rewrite`          | LLM 閹存牞顫夐崚娆撳櫢閸愭瑥鍑￠張澶婂敶鐎?  | Upd-6                  |
| `delta`            | 閸欘亣顔囪ぐ鏇炲綁閸栨牠鍣?         | Upd-5                  |
| `summarize`        | 瑜般垺鍨氶幗妯款洣            | Comp-2, Comp-8, Comp-9 |
| `reflect`          | 瑜般垺鍨氶崣宥嗏偓?insight    | Comp-4                 |
| `profile_update`   | 閼辨艾鎮庢稉鍝勭杽娴ｆ挾鏁鹃崓?profile | Comp-5                 |
| `extract_concept`  | 瑜般垺鍨氬鍌氬悍閸楁洖鍘?         | Comp-6                 |
| `prototype_form`   | 瑜般垺鍨氶崢鐔风€?schema     | Comp-7                 |
| `distill`          | 妤傛ê鐦戞惔锕佹崁妫?          | Comp-10                |
| `prune`            | 閸掔娀娅庢担搴濈幆閸婄厧顕挒?        | Maint-2/3/4/5/6/7      |
| `dedup`            | 閸樺鍣搁獮璺烘値楠?          | Maint-8                |
| `move`             | 鐏炲倿妫挎潻浣盒?瑜版帗銆?        | Maint-10               |
| `consolidate`      | 鐎规碍婀￠弫鏉戞値绾板海澧?         | Maint-12               |
| `review`           | 鐎光剝鐓￠惄绋垮彠閹勫灗閸愯尙鐛?       | Maint-9, Maint-13      |


### 鐢瓕顫?effect


| effect      | 閸氼偂绠?            |
| ----------- | -------------- |
| `add`       | 閺傛澘顤?unit        |
| `modify`    | 娣囶喗鏁煎鍙夋箒 unit      |
| `delete`    | 閸掔娀娅庡鍙夋箒 unit      |
| `move`      | 娴犲簼绔寸仦鍌濈讣缁夎鍩岄崣锔跨鐏?     |
| `merge`     | 閸氬牆鑻熸径姘嚋 unit      |
| `summarize` | 閻㈢喐鍨氭妯虹湴閸楁洖鍘撻獮璺哄讲閼宠姤娴涢幑銏犲斧鐎电钖?|
| `version`   | 娣囨繄鏆€閻楀牊婀伴柧?         |


### 閸欐ê绱撴潪?

- **濠曟柨瀵查惄顔界垼**閿涙岸顤傛径鏍櫢閺?/ 妤傛ê鐪伴幎鍊熻杽 / 閻㈢喎鎳￠崨銊︽埂缁狅紕鎮?
- **娴ｆ粎鏁ら懠鍐ㄦ纯**閿涙瓬y-entity / by-window / by-layer / global
- **閻樿埖鈧礁褰夐崠?*閿涙瓫dd / modify / delete / move / summarize / version
- **鐟欙箑褰傚Ο鈥崇础**閿涙艾鎳嗛張?/ 娴滃娆?/ 妫板嫮鐣?/ 閺夆€叉

### 瀵ょ儤膩鐠囧瓨妲?

- 鐢瓕顫夐崘娆忓弳瀹歌尙绮￠悽?D 濡?`organization` 鐎瑰本鍨氶妴?
- E 濡茶棄褰х悰銊с仛妫版繂顦荤憴锕€褰傞惃?memory evolution閵?
- 閸?`Compression` 閺勵垯绔寸粔宥夌彯鐏炲倹濞婄挒鈩冪川閸栨牭绱遍崢?`Maintenance` 閺勵垯绔寸粔宥囨晸閸涜棄鎳嗛張鐔哥川閸栨牓鈧?
- `AppendWithLinkUpdate` 娑撳秴鍟€閸楁洖鍨敍娌磇nk 閺囧瓨鏌婇崢鐔峰灟娑撳﹤绨查崶鐐插煂 D 濡叉枻绱滶 閸欘亣绀嬬拹?extra evolution over existing memory閵?
- `summarize`閵嗕梗prune`閵嗕梗move`閵嗕梗consolidate` 娑?`merge` 娑撯偓閺嶅嚖绱濋柈鑺ユЦ鐎?store state 閻ㄥ嫪绗夐崥宀勵杺婢舵牗绱ㄩ崠鏍ㄦ煙瀵繈鈧?
- `replace/merge/upsert/rewrite` 閸欘亝婀侀崷銊潶閻炲棜袙娑撴椽顤傛径鏍曢崣鎴犳畱瀹稿弶婀佺拋鏉跨箓闁插秵鏆ｉ弮璁圭礉閹靛秴鐫樻禍?E 濡插鈧?

### 閸忔娊鏁痪锔芥将

- `selection=matched_by_entity`閵嗕梗action=merge/profile_update` 鐟曚焦鐪?unit 閹?representation 閹绘劒绶?`entities`
- `action=delta` 鐟曚焦鐪伴懗钘夘嚠姒绘劕鍩屽鍙夋箒 unit 閳?闂団偓鐟?key_field 閹?entity
- `action=summarize/reflect/extract_concept/prototype_form` 闁艾鐖剁憰浣圭湴 selection 鐟曞棛娲婃径姘嚋 unit
- `action=move` 鐟曚焦鐪?layer_count > 1
- `action=rewrite/review` 闁艾鐖剁敮锔芥降妫版繂顦?LLM 閹存劖婀伴敍灞肩瑝闁倸鎮庢姗€顣剁憴锕€褰?
- `action=prune`閵嗕梗dedup`閵嗕梗review` 娑?retrieval 鐞涘奔璐熷楦库偓锕€鎮庨敍灞藉讲閼宠姤鏁奸崣妯烘倵缂侇厼褰茬拠璁崇瑐娑撳鏋?

### 閺冄勀侀崸妤€鍩岄弬鐗堫攱閺嬪墎娈戦弰鐘茬殸


| 閺冄冨晸濞?                             | 閺傛澘鍟撳▔?                                                                    |
| -------------------------------- | ----------------------------------------------------------------------- |
| `Upd-3 MergeByEntity`            | `selection=matched_by_entity, action=merge, effect=modify`              |
| `Upd-4 UpsertByKey`              | `selection=matched_by_key, action=upsert, effect=add+modify`            |
| `Upd-6 LLMRewrite`               | `selection=matched_by_key/entity, action=rewrite, effect=modify`        |
| `Comp-2 Summarization`           | `selection=time_window/layer_slice, action=summarize, effect=summarize` |
| `Comp-4 LLMReflection`           | `selection=time_window/all, action=reflect, effect=add`                 |
| `Comp-5 EntityProfileUpdate`     | `selection=matched_by_entity, action=profile_update, effect=modify`     |
| `Maint-8 Deduplication`          | `selection=layer_slice/all, action=dedup, effect=delete+modify`         |
| `Maint-10 ArchivalMovement`      | `selection=low_activity, action=move, effect=move`                      |
| `Maint-12 PeriodicConsolidation` | `selection=layer_slice, action=consolidate, effect=merge+delete`        |


---

## G. Retrieval 閳?濡偓缁便垻鐡ラ悾?

**閺嶇绺鹃梻顕€顣?*閿涙氨绮扮€?query閿涘苯顩ф担鏇氱矤 store 娑擃厽澹橀崚鎵祲閸?memory閿?

鏉╂瑩鍣锋稉宥呭晙閹?retrieval 瀵ょ儤膩閹存劕绶㈡径姘嚒閸氬秵鏌熷▔鏇礉閼板本妲哥紒鐔剁閸愭瑦鍨氶敍?

```text
Retrieval = signals + ranker + flow + constraints
```

### 濡偓缁便垻绮嶆禒?


| 缂佸嫪娆?           | 閸氼偂绠?                   | 閸忕鐎烽崣鏍р偓?                                                                                                                             |
| ------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `signals`     | 濡偓缁便垺妞傛担璺ㄦ暏閸濐亙绨虹拠浣瑰祦             | `similarity`, `keyword_match`, `recency`, `importance`, `entity_match`, `graph_proximity`, `hierarchy_match`, `diversity_penalty` |
| `ranker`      | 婵″倷缍嶉幎濠傤樋娑?signals 閸氬牊鍨氭稉鐑樻付缂佸牊甯撴惔?| `identity`, `weighted_sum`, `rrf`, `decay`, `rerank_llm`, `mmr`                                                                   |
| `flow`        | 濡偓缁便垺绁︾粙瀣波閺?               | `single_stage`, `two_stage`, `top_down`, `agent_invoked`                                                                          |
| `constraints` | 鐎电绻戦崶鐐电波閺嬫粍鏌﹂崝鐘垫畱缂佹挻鐎幋鏍暕缁犳瀹抽弶?      | `top_k`, `window`, `edge_filter`, `expand_depth`, `final_k`                                                                       |


### 鐢瓕顫?signal


| signal              | 閸氼偂绠?             | 鐢瓕顫嗛弶銉︾爱                        |
| ------------------- | --------------- | --------------------------- |
| `similarity`        | 閸氭垿鍣洪惄闀愭妧鎼?          | embedding search            |
| `keyword_match`     | 閸忔娊鏁拠?缁嬧偓閻ゅ繐灏柊?       | BM25 / sparse retrieval     |
| `recency`           | 閺冨爼妫块弬鎷岀箮閹?          | time index                  |
| `importance`        | 闁插秷顩﹂幀褍鍨庨弫?          | memory metadata             |
| `entity_match`      | 鐎圭偘缍嬬划鍓р€橀崠褰掑帳閹?overlap | entity index                |
| `graph_proximity`   | 閸ラ箖鍋︽潻鎴欌偓浣割樋鐠哄啿鍙ч懕?       | graph edges                 |
| `hierarchy_match`   | 鐏炲倻楠囬幗妯款洣/閻栬泛鐡欑仦鍌滈獓閻╃鍙ч幀?   | hierarchy links / summaries |
| `diversity_penalty` | 婢舵碍鐗遍幀褏瀹抽弶鐔稿灗閹晝缍掓い?      | MMR / DPP-like logic        |


### 鐢瓕顫?ranker


| ranker         | 閸氼偂绠?                    | 閸忔娊鏁崣鍌涙殶                  |
| -------------- | ---------------------- | --------------------- |
| `identity`     | 閸楁洑绔?signal 閻╁瓨甯撮幒鎺戠碍         | source                |
| `weighted_sum` | 婢舵矮閲?signal 閸旂姵娼堝Ч鍌氭嫲         | weights               |
| `rrf`          | Reciprocal Rank Fusion | k                     |
| `decay`        | 鐎佃妞傞梻?閸掑棙鏆熼崑姘斥€滈崙蹇撳綁閹?           | decay_rate, half_life |
| `rerank_llm`   | 閻?LLM 鐎电懓鈧瑩鈧绻樼悰宀勫櫢閹?         | model, rerank_prompt  |
| `mmr`          | 閻╃鍙ч幀褍鎷版径姘壉閹嗕粓閸氬牊甯撴惔?           | lambda                |


### 鐢瓕顫?flow


| flow            | 閸氼偂绠?              | 閸忔娊鏁崣鍌涙殶                 |
| --------------- | ---------------- | -------------------- |
| `single_stage`  | 閸楁洟妯佸▓鐢垫纯閹恒儱褰崶鐐叉嫲閹烘帒绨?      | top_k                |
| `two_stage`     | 閸忓牏鐭栭崣顒€娲栭敍灞藉晙缁偓甯?        | recall_k, final_k    |
| `top_down`      | 閸忓牊顥呯槐銏ょ彯鐏炲倹鎲崇憰渚婄礉閸愬秴鐫嶅鈧紒鍡氬Ν    | levels, expand_top_k |
| `agent_invoked` | 閻?agent 閺勬儳绱￠崣鎴ｆ崳濡偓缁便垹浼愰崗?| tool_names, backend  |


### 閸欐ê绱撴潪?

- **娣団€冲娇闁瀚?*閿涙矮濞囬悽銊ユ憿娴?retrieval signals
- **閹烘帒绨弬鐟扮础**閿涙艾宕熸穱鈥冲娇 / 閾诲秴鎮?/ 鐞涙澘鍣?/ LLM 闁插秵甯?/ 婢舵碍鐗遍幀褏瀹抽弶?
- **濞翠胶鈻肩紒鎾寸€?*閿涙艾宕熼梼鑸殿唽 / 娑撱倝妯佸▓?/ top-down / agent 閺勬儳绱＄憴锕€褰?
- **缂佹挻鐏夌痪锔芥将**閿涙op_k / final_k / edge_filter / expand_depth

### 閸忔娊鏁痪锔芥将

- `signals contains similarity` 鐟曚焦鐪?store 閺?`vector` index 娑?unit 閺?`embedding`
- `signals contains keyword_match` 鐟曚焦鐪?store 閺?`keyword` index 閹?unit 閺堝鏋冮張顒€鍞寸€?
- `signals contains entity_match` 鐟曚焦鐪?store 閺?`entity` index 娑?unit 閺?`entities`
- `signals contains graph_proximity` 鐟曚焦鐪?store 閺?`graph` index 娑?`Org` 娴溠冨毉娴?graph edges
- `signals contains hierarchy_match` 閹?`flow=top_down` 鐟曚焦鐪扮€涙ê婀仦鍌滈獓閹芥顩?閻栬泛鐡欑仦鍌滈獓
- `ranker=rerank_llm` 娴兼艾顤冮崝鐘活杺婢?LLM 閹存劖婀?
- `flow=agent_invoked` 娑撳秴鍟€鐟欏棔璐熸稉鈧粔宥囧缁?retrieval primitive閿涘矁鈧苯褰ч弰顖涱梾缁便垻娈戠憴锕€褰傞弬鐟扮础

### 瀵ょ儤膩鐠囧瓨妲?

- `similarity / recency / importance / entity_match / graph_proximity` 娑?C 濡蹭粙鍣烽惃?signal 閹繄娣穱婵囧瘮娑撯偓閼疯揪绱濋柈鑺ユЦ閸欘垰顦查悽銊ф畱閺堝搫鍩楁穱鈥冲娇閵?
- `TwoStageRecallRerank` 娑撳秴鍟€閸楁洖鍨敍宀冣偓宀冾潶 `flow=two_stage` + `ranker=...` 鐟曞棛娲婇妴?
- `HierarchicalRetrieval` 娑撳秴鍟€閸楁洖鍨敍宀冣偓宀冾潶 `signals={hierarchy_match}` + `flow=top_down` 鐟曞棛娲婇妴?
- `DiversityAware` 闂勫秳璐?`ranker=mmr` 閹?`signals` 娑擃厾娈?`diversity_penalty`閵?
- `FullContext`閵嗕梗LLMControlled`閵嗕梗ContextualBandit` 閺嗗倷绗夋担婊€璐熼崺铏诡攨娑撴槒銆冮惃鍕缁涘鍨氶崨妯圭箽閻ｆ瑱绱辨俊鍌涚亯闂団偓鐟曚緤绱濋崣顖欑稊娑撳搫鎮楃紒顓熷⒖鐏炴洏鈧?

### 閺冄勵梾缁便垺鏌熷▔鏇炲煂閺傜増顢嬮弸鍓佹畱閺勭姴鐨?


| 閺冄冨晸濞?                    | 閺傛澘鍟撳▔?                                                            |
| ----------------------- | --------------------------------------------------------------- |
| `Similarity`            | `signals={similarity}, ranker=identity, flow=single_stage`      |
| `BM25`                  | `signals={keyword_match}, ranker=identity, flow=single_stage`   |
| `Recency`               | `signals={recency}, ranker=identity, flow=single_stage`         |
| `RecencyDecay`          | `signals={recency}, ranker=decay, flow=single_stage`            |
| `ImportanceRanked`      | `signals={importance}, ranker=identity, flow=single_stage`      |
| `EntityLookup`          | `signals={entity_match}, ranker=identity, flow=single_stage`    |
| `GraphHop`              | `signals={graph_proximity}, ranker=identity, flow=single_stage` |
| `WeightedMultiSignal`   | `signals={...}, ranker=weighted_sum, flow=single_stage`         |
| `TwoStageRecallRerank`  | `flow=two_stage, ranker=...`                                    |
| `LLMRerank`             | `ranker=rerank_llm`                                             |
| `HierarchicalRetrieval` | `signals={hierarchy_match}, flow=top_down`                      |
| `AgentToolCall`         | `flow=agent_invoked`                                            |


---

## H. Readout 閳?鏉堟挸鍤弽鐓庣础閸?

**閺嶇绺鹃梻顕€顣?*閿涙碍顥呯槐銏犲煂閻?memory 婵″倷缍嶆潪顒€瀵叉稉?agent 閸欘垱绉风拹鍦畱鏉堟挸鍙嗛敍?


鏉╂瑩鍣锋稉宥呭晙閹?readout 鐟欏棔璐熸稉鈧稉顏勩亣閻ㄥ嫯鍤滈悽杈ㄦ偝缁便垺蝎閵嗗倸顕禍搴＄秼閸?design space閿涘苯褰ф穱婵堟殌鐏忔垶鏆熼惇鐔割劀娴兼碍鏁奸崣?memory 娴ｈ法鏁ら弬鐟扮础閻?readout閿涙稑鍙炬担娆忣樋閺佺増娲块崓蹇庢崲閸斺剝甯撮崣锝嗗灗鐎瑰じ瀵屽鍡樼仸鐏炲倻娈戦弽鐓庣础闁瀚ㄩ妴?

| ID | 鐎圭偟骞?| 閹诲繗鍫?| 鏉堟挸鍤弽鐓庣础 | 閸忔娊鏁崣鍌涙殶 |
|----|------|------|---------|----------|
| Read-1 | **FlatConcat** | 閹稿绨幏鍏煎复濡偓缁便垻绮ㄩ弸婊€璐熼弬鍥ㄦ拱 | text block | separator, max_tokens, format |
| Read-3 | **SummarizedReadout** | 閸忓牊鎲崇憰浣诡梾缁便垻绮ㄩ弸婊冨晙鏉堟挸鍤?| summary text | model, max_length |
| Read-4 | **TemplatedPrompt** | 婵夘偄鍙嗘０鍕暰娑?prompt 濡剝婢?| filled template | template |
| Read-6 | **CodeInjection** | 娴犮儱鍤遍弫?瀹搞儱鍙跨€规矮绠熻ぐ銏犵础濞夈劌鍙?| code/tool defs | format={function_def, docstring} |


### 閸欐ê绱撴潪?

- **鏉堟挸鍤ぐ銏犵础**閿涙氨娲块幒銉﹀閹?/ 閹芥顩﹂崥搴ょ翻閸?/ 濡剝婢橀崠鍛邦棅
- **濞夈劌鍙嗘担宥囩枂**閿涙rompt 閺傚洦婀?/ tool/code 鐎规矮绠?
- **閸樺缂夋惔?*閿涙艾鍙忛柌?/ 閹芥顩﹂崥搴㈡暈閸?

### 瀵ょ儤膩鐠囧瓨妲?

- `FlatConcat` 閺勵垱娓堕崺铏诡攨閵嗕焦娓堕柅姘辨暏閻?readout 閸╄櫣鍤庨妴?
- `SummarizedReadout` 閺勵垰鐨弫鎵埂濮濓絼绱伴弨鐟板綁 memory 娴ｈ法鏁ょ捄顖氱窞閻?readout閿涘苯娲滃銈勭箽閻ｆ瑣鈧?
- `TemplatedPrompt` 閸欘垯浜掗惇瀣╃稊鐎?text readout 閻ㄥ嫬瀵樼憗鍛湴閿涘奔绻氶悾娆忕暊閺勵垯璐熸禍鍡氼洬閻╂牜婀＄€圭偟閮寸紒鐔惰厬閻?prompt 閹恒儱褰涘顔肩磽閵?
- `CodeInjection` 娣囨繄鏆€閿涘奔绲炬惔鏃囶潒娑?*閻楄鐣?readout mode**閿涙岸鈧艾鐖舵禒鍛躬 skill/tool memory 娑擃厺濞囬悽顭掔礉娑撳秴寮稉搴ㄧ帛鐠併倖鎮崇槐銏⑩敄闂傛番鈧?
- `StructuredSections`閵嗕梗PrependToContext`閵嗕梗SelectiveByRole`閵嗕梗RankedList`閵嗕梗JSONStructured`閵嗕梗LatentInjection` 閺嗗倿妾锋稉杞版崲閸?濡楀棙鐏︾仦鍌涚壐瀵繘鈧瀚ㄩ敍灞肩瑝閸愬秳缍旀稉杞板瘜閹兼粎鍌ㄦい骞库偓?

---

## J. Evolution Trigger 閳?鐟欙箑褰傞弶鈥叉閿涘湣emory Evolution 閸忓彉闊╅敍?


| ID     | 鐎圭偟骞?                | 閹诲繗鍫?            | 閸忔娊鏁崣鍌涙殶             |
| ------ | ------------------ | -------------- | ---------------- |
| Trig-1 | **Never**          | 娴犲簼绗夌憴锕€褰?          | 閳?               |
| Trig-2 | **Periodic**       | 濮?N 濮濄儴袝閸欐垳绔村▎?     | every: int       |
| Trig-3 | **AfterWrite**     | 濮ｅ繑顐奸崘娆忓弳閸氬氦袝閸?       | 閳?               |
| Trig-4 | **OnEvent**        | 閻楃懓鐣炬禍瀣╂閸欐垹鏁撻弮鎯靶曢崣?     | event_type       |
| Trig-5 | **BudgetExceeded** | 妫板嫮鐣绘担璺ㄦ暏鐡掑懓绻冮梼鍫濃偓鍏兼鐟欙箑褰?   | threshold: float |
| Trig-6 | **CountExceeded**  | unit 閺佷即鍣虹搾鍛扮箖闂冨牆鈧吋妞傜憴锕€褰?| max_count: int   |
| Trig-7 | **Conditional**    | 閼奉亜鐣炬稊澶嬫蒋娴犳儼銆冩潏鎯х础       | predicate        |


---

## 1. 閸忕厧顔愰幀褏瀹抽弶鐔虹叐闂?

娴犮儰绗呯痪锔芥将閸︺劍鎮崇槐銏℃閻劋绨崜顏呯亰闂堢偞纭剁紒鍕値閵?

### 1.1 Store 閳?Module 缁撅附娼敍鍧皌ore 閹锋挻澧ら梽鎰煑濡€虫健闁瀚ㄩ敍?


| 缁撅附娼?                                                                            | 閸氼偂绠?                   |
| ------------------------------------------------------------------------------ | --------------------- |
| `requires(Ret.signals contains similarity, store.index contains vector)`       | similarity 濡偓缁便垼顩﹀Ч鍌氭倻闁插繒鍌ㄥ?  |
| `requires(Ret.signals contains keyword_match, store.index contains keyword)`   | keyword 濡偓缁便垼顩﹀Ч鍌氬彠闁款喛鐦濈槐銏犵穿     |
| `requires(Ret.signals contains graph_proximity, store.index contains graph)`   | graph proximity 鐟曚焦鐪伴崶鍓у偍瀵?|
| `requires(Ret.signals contains entity_match, store.index contains entity)`     | entity match 鐟曚焦鐪扮€圭偘缍嬬槐銏犵穿   |
| `requires(Org.write=graph_node, exists layer.shape == Graph)`              | 閸ユ崘濡悙鐟扮湴閸愬懎鍟撻崗銉洣濮瑰倸鐡ㄩ崷?graph layer |
| `requires(Org.links contains graph_edge, exists layer.indices contains graph)` | 閸ユ崘绔熺憰浣圭湴 graph index      |
| `requires(Org.write=hierarchical_slot, layer_count > 1)`                   | 鐏炲倻楠囧Σ鎴掔秴閸愭瑥鍙嗙憰浣圭湴婢舵艾鐪伴幏鎾村ⅳ            |
| `requires(Org.placement=by_unit_type, layer_count > 1)`                          | 閹?unit 缁鐎烽崚鍡氭儰閻愬綊鈧艾鐖剁憰浣圭湴婢舵矮閲滈惄顔界垼 layer   |
| `requires(Evo.action=move, layer_count > 1)`                                   | 鐏炲倿妫挎潻浣盒╃憰浣圭湴婢舵艾鐪伴幏鎾村ⅳ            |


### 1.2 Module 閳?Module 缁撅附娼敍鍫滅瑐濞撴瓕绶崙娲閸掓湹绗呭〒鎼佲偓澶嬪閿?


| 缁撅附娼?                                                                                                  | 閸氼偂绠?                              |
| ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| `requires(Ret.signals contains similarity, Rep-* produces embedding)`                                | 閸氭垿鍣哄Λ鈧槐銏ｎ洣濮瑰倹婀?embedding                |
| `requires(Org.links contains entity, UF/Rep produces entities)`                                      | entity link 鐟曚焦鐪伴張澶婄杽娴?               |
| `requires(Org.placement=by_tag, Rep/UF produces tags)`                                                 | 閹?tag 闁鎯ら悙纭咁洣濮?tags              |
| `requires(Org.write=cluster, Rep produces embedding)`                                            | cluster 鐏炲倸鍞撮崘娆忓弳鐟曚焦鐪?embedding           |
| `requires(Evo.action in {merge, profile_update}, UF/Rep produces entities)`                          | entity 閸氬牆鑻?閻㈣鍎氶弴瀛樻煀鐟曚焦鐪伴張澶婄杽娴?             |
| `requires(Evo.action=delta, UF produces alignable units)`                                            | delta 濠曟柨瀵茬憰浣圭湴閸欘垰顕?                   |
| `requires(Ret.signals contains graph_proximity, Org.links contains graph_edge)`                      | graph proximity 鐟曚焦鐪伴張澶婃禈鏉?           |
| `requires(Read-6, UF-10 produces code)`                                                              | CodeInjection 鐟曚焦鐪版禒锝囩垳 unit          |
| `requires(Ret.flow=top_down or Ret.signals contains hierarchy_match, Evo.action produces hierarchy)` | top-down/hierarchy 濡偓缁便垼顩﹀Ч鍌氱湴缁狙勬喅鐟?閻栬泛鐡欑仦鍌滈獓 |


### 1.3 鐠囶厺绠熼懓锕€鎮庣痪锔芥将閿涘牆宸遍悜鍫熷腹閼芥劗娈戠紒鍕値濡€崇础閿?


| 缁撅附娼?                                                                                | 閸氼偂绠?                           |
| ---------------------------------------------------------------------------------- | ----------------------------- |
| `co_requires(UF-6/7/8/9, Evo.action in {merge, upsert, profile_update})`           | 缂佹挻鐎崠?unit 娑撳酣顤傛径鏍х杽娴?闁款喗甯跺鏂垮瀵搫鍙ч懕?       |
| `co_requires(UF-2/3, Org.write=append)`                                        | 鐎电鐦界痪?unit 娑?append-style organization 瀵搫鍙ч懕?|
| `co_requires(Evo.action=profile_update, UF-7)`                                     | profile 閺囧瓨鏌婃稉?EntityState 閹惰棄褰囧鍝勫彠閼?|
| `co_requires(Org.links contains graph_edge, Ret.signals contains graph_proximity)` | 閸ユ崘绔熺紒鍕矏娑撳骸娴橀柇鏄忕箮濡偓缁便垹宸遍崗瀹犱粓                 |
| `co_requires(Rep contains code, Read-6)`                                           | 娴狅絿鐖滅悰銊с仛娑?CodeInjection 瀵搫鍙ч懕?      |


### 1.4 娴滄帗鏋肩痪锔芥将


| 缁撅附娼?                                                                             | 閸氼偂绠?         |
| ------------------------------------------------------------------------------- | ----------- |
| `incompatible(WT.policy=never, Org.write=append)`                           | 娴犲簼绗夐崘娆忓弳娑?append-style organization 娴滄帗鏋?|


---

## 2. 閹兼粎鍌ㄧ粚娲？鐟欏嫭膩娴兼壆鐣?

### 2.1 閸?slot 閸婃瑩鈧鏆?


| Slot               | 閸婃瑩鈧鏆?       | 閸氼偆绮嶉崥鍫㈢暬鐎涙劕鎮?                       |
| ------------------ | ---------- | ----------------------------- |
| Layered Store Topo | `S_struct` | `S_struct`                    |
| UnitFormation      | 14         | 14 + Compose/Cascade 缂佸嫬鎮?閳?120 |
| Representation     | `S_rep`    | `S_rep`                       |
| WriteTrigger       | `S_wt`     | `S_wt`                        |
| Organization       | `S_org`    | `S_org`                       |
| MemoryEvolution    | `S_evo`    | `S_evo`                       |
| Retrieval          | `S_ret`    | `S_ret`                       |
| Readout            | 3 + `Read-6*` | 3 + `Read-6*`                |
| EvolutionTrigger   | 7          | 7                             |

`Read-6*` 鐞涖劎銇?`CodeInjection` 娴ｆ粈璐熼悧瑙勭暕 readout mode 娣囨繄鏆€閿涘奔绲炬妯款吇娑撳秶鎾奸崗銉ョ埗鐟欏嫭鎮崇槐銏⑩敄闂傛番鈧?


### 2.2 缁屾椽妫挎稉濠勬櫕

娑撳秴鎯堢紒鍕値缁犳鐡欓敍姝歋_struct 鑴?14 鑴?S_rep 鑴?S_wt 鑴?S_org 鑴?S_evo 鑴?S_ret 鑴?3 鑴?7`

閸氼偆绮嶉崥鍫㈢暬鐎涙劒绲炬稉宥呮儓閸欏倹鏆熼敍姝歋_struct 鑴?120 鑴?S_rep 鑴?S_wt 鑴?S_org 鑴?S_evo 鑴?S_ret 鑴?3 鑴?7`

閸忔湹鑵?`S_struct` 娑撳秴鍟€閺勵垯绔存稉顏勬祼鐎规艾鐖堕弫甯礉閼板本妲哥紒鎾寸€悽鐔稿灇閸ｃ劌鍘戠拋鍝ユ畱 layer 闁板秶鐤嗛弫鑸偓鍌欑伐婵″偊绱濊ぐ鎾绘閸掓湹璐熼敍?

- 鐏炲倹鏆?`1..4`
- `theme` 娴?6 娑擃亜褰堥幒褌瀵屾０妯硅厬闁?
- `shape 閳?{Flat, Graph}`
- 濮ｅ繐鐪?index mask 娴犲骸鎮庡▔鏇㈡肠閸氬牅鑵戦柅?

閸?`S_struct` 娴兼岸娈㈢紒鎾寸€痪锔芥将閻ㄥ嫭鏁圭槐褎鍨ㄩ弨鐐緱閼板苯褰夐崠鏍モ偓?

閸氬瞼鎮婇敍瀹峉_rep` 娑旂喍绗夐崘宥嗘Ц閸ュ搫鐣剧敮鍛婃殶閵嗗倸顩ч弸婊冪唨绾偓閸忓啰绀岀拠宥堛€冩稉?10 娑擃亷绱濋崚娆愭￥缁撅附娼弮鍫曟姜缁屽搫鐡欓梿鍡曠瑐閻ｅ奔璐?`2^10 - 1 = 1023`閿涙稐绲鹃崷銊ョ杽鐠哄吀鑵戞导姘垛偓姘崇箖閼宠棄濮忕痪锔芥将鏉╁洦鎶ら幒澶娿亣闁插繑妫ら幇蹇庣疅缂佸嫬鎮庨妴?

`S_wt` 娑旂喍绗夐崘宥嗘Ц閸ュ搫鐣剧敮鍛婃殶閿涘矁鈧苯褰囬崘鍏呯艾閿?

- 閸忎浇顔忔担璺ㄦ暏閸濐亙绨?`signals`
- `scorer` 閸忎浇顔忛崫顏冪昂缂佸嫬鎮庨弬鐟扮础
- 閺勵垰鎯侀崥顖滄暏 `gates`
- `policy` 閻ㄥ嫬鈧瑩鈧娉﹂崥鍫礉娴犮儱寮烽弰顖氭儊閸氼垳鏁?gates / scorer

`S_org` 閸氬本鐗辨稉宥呭晙閺勵垰娴愮€规艾鐖堕弫甯礉閼板苯褰囬崘鍏呯艾閿?

- 閸欘垶鈧娈?**placement**閿涘牏娲伴弽鍥儰閻愮櫢绱氱粵鏍殣鐎硅埖妫?
- 閸欘垶鈧娈?**links** 鐎涙劙娉?
- 閸欘垶鈧娈?*鐏炲倸鍞撮崘娆忓弳瑜般垺鈧?*閿涘潉append` / `partition` / 閳ワ讣绱?
- 娴犮儱寮?`StoreTopology` 閹绘劒绶甸惃?capability 鏉堝湱鏅?

`S_evo` 娑旂喍绗夐崘宥嗘Ц閸ュ搫鐣剧敮鍛婃殶閿涘矁鈧苯褰囬崘鍏呯艾閿?

- 閸欘垶鈧娈?`selection`
- 閸欘垶鈧娈?`action`
- 閸欘垶鈧娈?`effect`
- 娴犮儱寮?`evolution_trigger` 閻ㄥ嫯袝閸欐垶娼禒?

`S_ret` 娑旂喍绗夐崘宥嗘Ц閸ュ搫鐣剧敮鍛婃殶閿涘矁鈧苯褰囬崘鍏呯艾閿?

- 閸欘垶鈧娈?`signals`
- 閸欘垶鈧娈?`ranker`
- 閸欘垶鈧娈?`flow`
- 娴犮儱寮?`constraints` 閸欏倹鏆熺粚娲？

### 2.3 缁撅附娼崜顏呯亰閸氬海娈戦張澶嬫櫏缁屾椽妫?

閺嶈宓侀崗鐓庮啇閹呭閺夌喍鍙婄粻妤冨 5-10% 閻ㄥ嫮绮嶉崥鍫熸Ц閸氬牊纭堕惃鍕剁窗

- **娑撳秴鎯堢紒鍕値缁犳鐡?*閿涙畧10^8 閺堝鏅ラ柊宥囩枂
- **閸氼偆绮嶉崥鍫㈢暬鐎?*閿涙畧10^11 閺堝鏅ラ柊宥囩枂

### 2.4 閸欘垵顢戦惃鍕偝缁便垻鐡ラ悾?


| 缁屾椽妫跨憴鍕?       | 閹恒劏宕樼粵鏍殣                                         |
| ----------- | -------------------------------------------- |
| 10^4 娴犮儰绗?    | 缁岃渹濡?(grid search)                             |
| 10^4 - 10^6 | 闂呭繑婧€閹兼粎鍌?+ 缁撅附娼崜顏呯亰                                  |
| 10^6 - 10^9 | 鏉╂稑瀵茬粻妤佺《 / Bayesian optimization                 |
| 10^9 娴犮儰绗?    | 閸掑棗鐪伴幖婊呭偍閿涙艾鍘涢幖婊呭偍 structural level閿涘苯娴愮€规艾鎮楅幖婊呭偍 modular閿涘本娓堕崥搴ょ殶閸?|


閹恒劏宕樼粵鏍殣閿?*閸掑棗鐪伴幖婊呭偍**

```
Phase 1: 閸ュ搫鐣?LayeredStoreTopo + Trigger閿涘本鎮崇槐?9 娑?modular slot閿涘牆鐔€绾偓鐎圭偟骞囬敍灞肩瑝閸氼偆绮嶉崥鍫㈢暬鐎涙劧绱?
          缁屾椽妫?閳?14 鑴?S_rep 鑴?S_wt 鑴?S_org 鑴?S_evo 鑴?S_ret 鑴?3
          閳?鏉╂稑瀵茬粻妤佺《閿涘opulation=100, generations=200

Phase 2: 闁褰?top-10 闁板秶鐤嗛敍灞筋嚠 Retrieval 閸?UnitFormation 鐏炴洖绱戠紒鍕値缁犳鐡欓幖婊呭偍
          缁屾椽妫?閳?10 鑴?200 鑴?120 閳?2.4 鑴?10^5 per config
          閳?闂呭繑婧€閹兼粎鍌?+ 缁墽绮忕拫鍐ㄥ棘

Phase 3: 閸欏倹鏆熺拫鍐х喘閿涘牊鐦℃稉顏堝帳缂冾喖鍞撮柈銊ф畱鏉╃偟鐢婚崣鍌涙殶閿?
          閳?Bayesian optimization / grid search
```

---

## 3. 瀹稿弶婀佸銉ょ稊閻?Primitive 閸掑棜袙

鐏?DSLgrammar.md 娑?8 娑擃亞绮￠崗鍝ラ兇缂佺喎鍨庣憴锝勮礋 Primitive ID閿?


| System            | UF                  | Rep                                         | WT                                                                  | Org                                                                                                        | Evo                                                                                                                             | Ret                                                                                 | Read   |
| ----------------- | ------------------- | ------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------ |
| MemGPT            | UF-3                | `{text, embedding}`                         | `gates={tool_called(...)}, policy=explicit_only`                    | `placement=agent_selected(...), links={}, write=append`                                                  | `selection=layer_slice(main_context), action=summarize+move, effect=summarize+move, trigger=budget_exceeded`                    | `signals={similarity}, ranker=identity, flow=agent_invoked`                         | Read-4 |
| Reflexion         | UF-13               | `{text}`                                    | `gates={on_event("task_failure")}, policy=boolean_gate`             | `placement=default(reflection_layer), links={}, write=append`                                            | `selection=layer_slice(reflection_layer), action=prune, effect=delete, trigger=on_event("task_failure")`                        | `signals={}, ranker=identity, flow=single_stage`                                    | Read-1 |
| A-MEM             | UF-14               | `{text, embedding, triple, tags, entities}` | `policy=always`                                                     | `placement=default(graph_layer), links={graph_edge(...)}, write=graph_node`                              | `Trig-1 Never`                                                                                                                  | `signals={similarity, graph_proximity}, ranker=identity, flow=single_stage`         | Read-4 |


### 娴犲簼鑵戦崣顖濐潎鐎电喎鍩岄惃鍕灥濮?Motif

**Motif 1: Embed-and-Retrieve閿涘牆鍤悳?7/8 濞嗏槄绱?*

- 鐞涖劎銇?set 閸栧懎鎯?`embedding` + retrieval signals 娑擃厼瀵橀崥?`similarity`
- 閸戠姳绠幍鈧張澶岄兇缂佺喖鍏橀棁鈧憰?embedding 娴ｆ粈璐熼崺铏诡攨濡偓缁便垺澧滃▓?

**Motif 2: Organization-Driven Normal Write閿涘牆鍤悳?6/8 濞嗏槄绱?*

- append-style normal write 瀵扳偓瀵扳偓閻?`organization` 鐎瑰本鍨?
- 閸欘亝婀佸☉澶婂挤妫版繂顦?entity/profile 閺囧瓨鏌婇幋鏍彯鐏炲倹鏆ｉ悶鍡欐畱缁崵绮洪幍宥嗘▔瀵繋濞囬悽?`memory_evolution`

**Motif 3: Selective-Write 閸掑棗瀵?*

- 缁犫偓閸楁洜閮寸紒鐔奉樋閻?`policy=always` 閹?`gates={on_event(...)}`
- 婢跺秵娼呯化鑽ょ埠閺囨潙鈧儳鎮滄禍?`scorer=llm_judge` 閹?`policy=explicit_only`
- 鏉╂瑦妲搁崠鍝勫瀻 passive vs active memory 閻ㄥ嫬鍙ч柨顔炬樊鎼?

**Motif 4: High-Level Evolution 閸欘垶鈧?*

- 楠炲爼娼幍鈧張澶岄兇缂佺喖鍏橀崑姘剁彯鐏炲倹绱ㄩ崠鏍电幢瀵板牆顦跨化鑽ょ埠閸欘亜浠?organization-driven normal write閿涘矂顤傛径鏍ㄧ川閸栨牔绻氶幐浣稿彠闂傤厽鍨ㄩ弸浣戒氦闁?
- 闂団偓鐟曚線鐝仦鍌涚川閸栨牜娈戠化鑽ょ埠閸婃儳鎮滄禍搴濈瑝閸氬苯鐪板▎鈽呯窗`summarize` / `reflect` / `profile_update`

**Motif 5: Lifecycle Control 娑撱倖鐎崠?*

- 鐟曚椒绠為崣顏勪粵闂堢偛鐖舵潪鑽ゆ畱濠曟柨瀵查敍宀冾洣娑斿牊妲戠涵顔间粵 `prune / move / summarize`
- 婢跺秵娼呴惃鍕粣韫囨ɑ膩閸ㄥ婀€圭偤妾化鑽ょ埠娑擃厺绮涢惄绋款嚠鐏忔垼顫?

---

## 4. 濞兼粌婀惃鍕弓閹恒垻鍌ㄩ崠鍝勭厵閿涘牊鎮崇槐銏㈡畱娴犲嘲鈧吋澧嶉崷顭掔礆

闁俺绻冪€电懓鍑￠張澶婁紣娴ｆ粎娈戦崚鍡毿掗敍灞藉讲娴犮儴鐦戦崚顐㈠毉閹兼粎鍌ㄧ粚娲？娑?*鐏忔碍婀悮顐ｅ赴缁便垻娈戠紒鍕値**閿?


| 閺堫亝甯扮槐銏㈢矋閸?                                                                                                                                                                                               | 閸嬪洩顔?                | 妫板嫭婀℃禒宄扳偓?       |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----------- |
| `UF-8` (Extract(triple)) + `Retrieval(signals={graph_proximity}, ranker=identity, flow=single_stage)` + `Evo(action=prune, selection=low_activity, trigger=periodic)`                                | 缂佹挻鐎崠鏍ф禈鐠佹澘绻?+ 鐠併倗鐓￠柆妤€绻?     | 闂€鎸庢埂鐎电鐦芥稉顓犳畱閻儴鐦戝鏂垮  |
| `signals={surprise} + policy=threshold` + `Evo(action=reflect, selection=time_window, trigger=periodic)`                                                                                             | 閸欘亣顔囬幇蹇擃樆 + 閸欏秵鈧?         | 妤傛ɑ鏅ョ€涳缚绡勯崹?agent |
| `Retrieval(signals={similarity, recency, diversity_penalty}, ranker=mmr, flow=two_stage)` + `Read-4` (TemplatedPrompt)                                                                               | 婢舵矮淇婇崣鐑藉櫢閹?+ 濡剝婢樺▔銊ュ弳       | 閼奉亪鈧倸绨?memory  |
| `Compose(UF-3, UF-6, UF-12)` + `Organization(placement=default(...), links={parent_child}, write=hierarchical_slot(...))` + `Retrieval(signals={hierarchy_match}, ranker=identity, flow=top_down)` | 婢舵氨鐭戞惔?+ 鐏炲倻楠囩€涙ê鍋?+ 鐏炲倻楠囧Λ鈧槐? | 缁姹夌拋鏉跨箓鐏炲倻楠?     |
| `Evo(action=prototype_form, selection=layer_slice, trigger=periodic)` + `Evo(action=consolidate, selection=layer_slice, trigger=periodic)`                                                           | 閸樼喎鐎疯ぐ銏″灇 + 绾板海澧栭弫鏉戞値        | 濮掑倸搴风€涳缚绡?       |
| `signals={importance, novelty} + scorer=weighted_sum + policy=threshold` + `Evo(action=rewrite+summarize, selection=matched_by_key+time_window, trigger=after_write)`                                | 闁瀚ㄩ幀褍鍟?+ 閸斻劍鈧焦鏁奸崘?+ 濞撴劘绻橀幗妯款洣 | 濞茬粯鈧嗩唶韫囧棛閮寸紒?     |
