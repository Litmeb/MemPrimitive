# Trigger Survey for MemPrimitive

> Note
> This file is a literature-survey / design-space note, not a description of the current public baseline API.
> The baseline runtime has been simplified and now only publicly exposes `AlwaysTrigger`, `ThresholdTrigger`, `NeverTrigger`, and `ThresholdTrigger`.

鏇存柊鏃堕棿锛?026-03-31

## 杩欎唤鏂囨。鏄仛浠€涔堢殑

杩欎唤鏂囨。鍙洖绛斾竴涓灦鏋勯棶棰橈細

鍦ㄥ綋鍓?40 绡囧€欓€夋枃鐚噷锛宍trigger` 鍒板簳鏄笉鏄竴涓€煎緱鍦?`MemPrimitive` 閲屽崟鐙缓鎴愬帤閲嶅瓙绯荤粺鐨勬牳蹇冪淮搴︼紵

杩欐鏇存柊涓嶅啀鎶婂垽鏂仠鐣欏湪 `core / secondary / generic` 涓夋。锛岃€屾槸杩涗竴姝ユ妸姣忕瘒璁烘枃钀藉埌鏇村叿浣撶殑锛?
- `write trigger` 鍘熷瀷
- `evolution / maintenance trigger` 鍘熷瀷
- 瀵瑰簲鐨?`signal`
- 瑙﹀彂鏈哄埗鍦ㄦ暣绡囪鏂囦腑鐨勭湡瀹炰綔鐢?
杩欓噷璁ㄨ鐨?trigger 鍖呮嫭锛?
- `write_trigger`
- `evolution_trigger`
- 鏇村箍涔夌殑鈥滀綍鏃惰矾鐢便€佸帇缂┿€佸弽鎬濄€佸珐鍥恒€佹暣鐞嗐€佽縼绉汇€佺绾跨淮鎶も€?
鏈疆宸ヤ綔鏂瑰紡锛?
- 閫愮瘒鎼滅储骞堕槄璇昏鏂囧師鏂囬〉闈紝浼樺厛浣跨敤 arXiv / ACL Anthology / OpenReview / 椤圭洰涓婚〉绛変竴鎵嬫潵婧?- 鐩爣涓嶆槸閲嶅缓瀹屾暣鏂规硶缁嗚妭锛岃€屾槸鎶藉彇鈥滀綍鏃惰Е鍙?memory 鍔ㄤ綔鈥濊繖涓€灞?- 瀵瑰皯鏁版柟娉曠粏鑺備粛涓嶅畬鍏ㄥ叕寮€鐨勮鏂囷紝淇濇寔淇濆畧姒傛嫭锛屼笉鎿呰嚜琛ヤ笉瀛樺湪鐨勫鏉?trigger

## 鏈枃浣跨敤鐨?trigger 鍘熷瀷璇嶈〃

涓轰簡閬垮厤鎶婃瘡绡囬兘鍐欐垚鑷畾涔夎瑷€锛岃繖閲岀粺涓€浣跨敤涓€缁勮緝钖勭殑 trigger 鍘熷瀷锛?
1. `always`
  - 姣忎釜 observation / step / turn 榛樿閮借繘鍏ュ啓鍏ユ垨缁存姢娴佺▼
2. `threshold(score)`
  - 鐢辨煇涓樉寮忓垎鏁拌秴杩囬槇鍊艰Е鍙?3. `boolean signal gate`
  - 鏌愪釜甯冨皵浜嬩欢鍑虹幇鏃惰Е鍙?4. `failure/outcome-conditioned`
  - 澶辫触銆佽礋鍙嶉銆佺粨鏋滀笉杈炬爣鏃惰Е鍙?5. `capacity/budget-conditioned`
  - 缂撳啿鍖恒€乼oken銆佹Ы浣嶃€佸瓨鍌ㄩ绠楀埌闃堝€兼椂瑙﹀彂
6. `PeriodicTrigger`
  - 浠ュ浐瀹氬懆鏈熴€佸浐瀹氳疆鏁般€佸浐瀹氳皟搴﹁妭鎷嶈Е鍙?7. `SessionEndTrigger`
  - 鍦?session / chunk / stage / turn 鏀舵潫鏃惰Е鍙?8. `IdleTrigger`
  - 鍦?sleep-time銆佺┖闂茬獥鍙ｃ€佸悗鍙颁綆璐熻浇鏃惰Е鍙?9. `new-write conditioned`
  - 鍙湁鍑虹幇鏂板啓鍏ュ悗锛屽悗缁暣鐞?鍘嬬缉/閾炬帴鎵嶈Е鍙?10. `type-routing / multi-label routing`
  - 鍏堝垽鏂簲鍐欏叆鍝竴绫?memory锛屽啀璺敱鍒颁笉鍚?store
11. `subgoal-completion conditioned`
  - 瀛愮洰鏍囧畬鎴愩€侀樁娈甸棴鍚堛€乪pisode 杈圭晫鍒版潵鏃惰Е鍙?
## 鎬讳綋缁撹

鎶?40 绡囪鏂囬€愮瘒钀藉埌鍏蜂綋 trigger 鍘熷瀷涔嬪悗锛岀粨璁烘瘮鏃х増鏇存槑纭細

- 鐪熸寮轰緷璧栫嫭鐗?trigger 鐨勮鏂囷紝浠嶇劧鏄皯鏁?- 澶氭暟璁烘枃鐨?trigger 閮借兘鍘嬬缉鍒?`always`銆乣threshold(score)`銆乣boolean gate`銆乣capacity`銆佸皯閲?`Periodic/SessionEnd/Idle` 杩欏嚑绫?- 鏃х殑 `scheduled/offline` 杩囦簬瀹芥硾锛氬叾涓彧鏈変竴閮ㄥ垎鐪熻兘绋冲畾钀藉埌鏂扮殑涓夌被锛屽墿涓嬩笉灏戝叾瀹炲彧鏄€滃瓨鍦ㄥ悗鍙伴樁娈碘€濓紝涓嶅疁纭紪鐮佹垚 trigger
- 鐪熸鎷夊紑寮傝川鎬х殑锛屽ぇ澶氫粛鐒舵槸锛?  - `representation`
  - `organization`
  - `memory_evolution`
  - `retrieval`

鍥犳锛屽 `MemPrimitive` 鏇村悎鐞嗙殑鏂瑰悜浠嶇劧鏄細

- 淇濈暀灏戞暟楂樹环鍊?trigger 鍘熷瀷
- 涓嶆妸 trigger 寤烘垚姣?organization / evolution 鏇村鏉傜殑涓€灞?- 鎶婂鏉傛€ф洿澶氭斁鍦ㄢ€滆Е鍙戝悗鍋氫粈涔堚€濊€屼笉鏄€滆Е鍙戣瑷€鏈韩澶氬鏉傗€?
## 40 绡囬€愰」鏄犲皠

璇存槑锛?
- `鍒ゆ柇` 浠嶄繚鐣欙紝鏂逛究鍜屾棫鐗堣鎺?- `鍐欏叆鍘熷瀷` / `婕斿寲鍘熷瀷` 鏄缓璁槧灏勫埌 `MemPrimitive` 鐨?trigger prototype
- `signal` 鍙啓璁烘枃閲岀湡姝ｈ兘瀵瑰簲鍔ㄤ綔鏃舵満鐨勫叧閿俊鍙?- `trigger 鎽樿` 鑱氱劍鈥滆Е鍙戠偣鍒板簳鎵紨浠€涔堣鑹测€?

| #   | 璁烘枃                                                                            | 鍒ゆ柇        | 鍐欏叆鍘熷瀷                                                      | 婕斿寲鍘熷瀷                                                                      | 鍏抽敭 signal                                                        | 璇︾粏 trigger 鎽樿                                                                                                                                                                           |
| --- | ----------------------------------------------------------------------------- | --------- | --------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Generative Agents                                                             | secondary | `threshold(score)`                                        | `threshold(score)`                                                        | `importance score`銆佹渶杩戞€с€佺浉鍏虫€с€佺疮璁?reflection 鍒嗘暟                      | 閲嶆柊鏌ュ師鏂囧悗锛屾洿绋冲Ε鐨勮娉曟槸锛欸enerative Agents 鐨勬牳蹇?maintenance trigger 浠嶆槸绱 `importance` 瓒呴槇鍊煎悗鐨?reflection銆傛枃涓櫧鏈夋棩绋嬭鍒掍笌鈥滄柊涓€澶┾€濊妭濂忥紝浣嗛偅鏇村儚 agent loop 鐨勬椂闂寸粨鏋勶紝涓嶅疁鐩存帴璁版垚 `PeriodicTrigger` 鎴?`SessionEndTrigger`銆?     |
| 2   | Reflexion                                                                     | core      | `failure/outcome-conditioned`                             | `failure/outcome-conditioned`                                             | 浠诲姟澶辫触銆佺幆澧冨弽棣堛€佸崟鍏冩祴璇曞け璐ャ€佸鍔?缁撴灉涓嶈揪鏍?                                       | 杩欐槸鏈€鍏稿瀷鐨?`failure-triggered reflection`銆俶emory 鍐欏叆涓庡悗缁?verbal reinforcement 閮界洿鎺ョ敱澶辫触缁撴灉瑙﹀彂锛涙病鏈夊け璐ュ氨涓嶄細鐢熸垚閭ｆ潯鈥滀笅娆¤鎬庝箞鍋氣€濈殑鍙嶆€濊蹇嗐€倀rigger 鏈韩灏辨槸璁烘枃杈ㄨ瘑搴︾殑涓€閮ㄥ垎銆?                                                     |
| 3   | Voyager                                                                       | generic   | `boolean signal gate`                                     | `failure/outcome-conditioned` + `new-write conditioned`                   | 鏂版妧鑳芥垚鍔熷彂鐜般€佹墽琛屽け璐ャ€佷换鍔″畬鎴愬弽棣?                                             | Voyager 鐨勮蹇嗕富杞存槸 skill library銆傜湡姝ｅ啓鍏ュ父鍙戠敓鍦ㄥ彂鐜板彲澶嶇敤鎶€鑳芥垨鎴愬姛绋嬪簭鍚庯紝鍚庣画杩唬鍒欑敱鎵ц缁撴灉椹卞姩淇銆傚畠鏇村儚鈥滄妧鑳芥娊鍙栦笌绋嬪簭鏀瑰啓鈥濈殑 workflow锛屼笉鏄笓闂ㄨ璁?trigger 鏈哄埗鐨勮鏂囷紱鍙繎浼兼垚鈥滄柊鎶€鑳藉嚭鐜版墠鍐欏叆 + 澶辫触鏃朵慨璁⑩€濄€?                                                |
| 4   | MemGPT                                                                        | core      | `type-routing / multi-label routing`                      | `capacity/budget-conditioned` + `boolean signal gate`                     | memory pressure銆佷笂涓嬫枃绐楀彛鍘嬪姏銆佸伐鍏疯皟鐢ㄦ剰鍥俱€佹樉寮忎腑鏂?                             | MemGPT 鐨勫叧閿笉鏄崟涓€鍐欏叆闃堝€硷紝鑰屾槸 agent 鍦ㄨ繍琛屾椂鍐冲畾璇ユ妸淇℃伅鐣欏湪 main context銆佸啓鍒?recall memory 杩樻槸 archival memory锛屽苟鍦?memory pressure / interrupt 涓嬪仛杩佺Щ鎴栧垎椤点€倀rigger 涓?store 绠＄悊銆佽矾鐢卞拰鎺у埗娴佺揣鑰﹀悎锛屾槸灏戞暟鐪熸闇€瑕佹樉寮?trigger 鍘熷瀷鏃忕殑绯荤粺銆?|
| 5   | MemoryBank                                                                    | secondary | `threshold(score)`                                        | `threshold(score)` / `鏃犳硶琛ㄧず锛堣繛缁椂闂磋“鍑忥級`                                       | 閬楀繕鏇茬嚎寮哄害銆侀噸瑕佸害銆佹椂闂磋“鍑忋€侀噸澶嶆彁鍙?                                            | 鍘熸枃鏇存帴杩戔€滃甫鏃堕棿琛板噺鐨?retention function鈥濓紝鑰屼笉鏄浐瀹氬懆鏈熸暣鐞嗐€傚啓鎴?`scheduled/offline` 鍋忛噸浜嗭紱鏇寸ǔ濡ョ殑鏄繚鐣?`importance/retention threshold`锛屽苟娉ㄦ槑涓€閮ㄥ垎婕斿寲鐢辫繛缁椂闂村彉閲忛┍鍔紝涓嶈兘琚繖濂楃鏁?trigger 骞插噣琛ㄧず銆?                                 |
| 6   | RET-LLM                                                                       | generic   | `always`                                                  | `new-write conditioned`                                                   | 鏂版娊鍙?triplet銆佹枃妗ｈ繘鍏ョ郴缁?                                              | RET-LLM 鏇村儚鈥滆鍐欏瀷鐭ヨ瘑瀛樺偍鎺ュ彛鈥濄€傛柊鏂囨湰杩涘叆鏃堕粯璁ゆ娊鍙?triplet 骞跺啓鍏?memory锛屼箣鍚庡洿缁曡繖浜涘啓鍏ュ仛璇诲彇鍜屾椂闂村瀷 QA锛涜Е鍙戝眰鍑犱箮娌℃湁鐙壒璁捐锛屾帴杩?`always write`銆?                                                                                  |
| 7   | My Agent Understands Me Better                                                | secondary | `boolean signal gate`                                     | `threshold(score)` / `鏃犳硶琛ㄧず锛堟椂闂存祦閫濋┍鍔ㄥ珐鍥猴級`                                     | `cue recall` 鍛戒腑銆佷笂涓嬫枃鐩稿叧鎬с€佹椂闂存祦閫濄€佸洖蹇嗛娆°€佸珐鍥哄害                             | 閲嶆柊鏍稿鍚庯紝鏂囩珷鏇村儚 cue-based recall 鍔?consolidation score 婕斿寲锛岃€岄潪鍥哄畾鍚庡彴璋冨害銆傚師鏂囨。閲岀殑 `scheduled/offline` 鏈変簺杩囧害姒傛嫭锛涜繖閲屾敼涓衡€滃彲閮ㄥ垎钀藉埌闃堝€硷紝鍓╀綑鐢辨椂闂存祦閫濋┍鍔紝涓嶅己琛岃〃绀衡€濄€?                                                          |
| 8   | Compress to Impress                                                           | secondary | `SessionEndTrigger`                                       | `SessionEndTrigger`                                                       | session 杈圭晫銆佸璇濊疆娈电粨鏉熴€佸帇缂╅樁娈靛惎鍔?                                        | 澶嶆牳鍚庢洿閫傚悎绾补鍐欐垚 `SessionEndTrigger`銆侰OMEDY 寮鸿皟鐨勬槸 session-specific summary 涓庡璇濇钀芥敹鏉熷悗鐨勫帇缂╋紝涓嶆槸鈥滃瓙鐩爣瀹屾垚鍚庤Е鍙戔€濈殑 agentic subgoal 璇箟銆?                                                        |
| 9   | Toward Conversational Agents with Context and Time Sensitive Long-term Memory | generic   | `always`                                                  | `always` / `new-write conditioned`                                        | 鏂板璇?turn銆佹椂闂存埑銆佷簨浠堕『搴忕害鏉?                                             | 璁烘枃涓昏酱鏄?temporal/context-sensitive retrieval锛屼笉鏄啓鍏ヨЕ鍙戙€傚璇濆唴瀹瑰熀鏈寜 turn 鎸佺画杩涘叆闀挎湡璁板繂锛屽叧閿樊寮傚湪 retrieval 渚у浣曟寜鏃堕棿銆佷簨浠堕『搴忓拰灞€閮ㄤ笂涓嬫枃閲嶆瀯鏌ヨ銆倀rigger 鍙繎浼间负 `always write`銆?                                             |
| 10  | Agent Workflow Memory                                                         | secondary | `threshold(score)` / `boolean signal gate`                | `new-write conditioned` + `threshold(score)`                              | workflow 閲嶇敤棰戠巼銆佹垚鍔熺巼銆佷换鍔＄浉浼兼€?                                         | AWM 鍏冲績鐨勬槸鈥滀粈涔堟椂鍊欐妸閲嶅鍑虹幇鐨?routine 鎶借薄鎴?workflow memory锛屽苟鍦ㄥ悗缁换鍔′腑閫夋嫨鎬ф彁渚涚粰 agent鈥濄€傝繖涓嶆槸绾?`always`锛屼絾涔熶笉鏄鏉?trigger 浣撶郴锛涙洿閫傚悎鐪嬫垚 `workflow utility / reuse threshold`锛岃揪鍒伴槇鍊煎悗鍥哄寲涓?procedural memory銆?            |
| 11  | HiAgent                                                                       | core      | `always`锛坰tep 绾?working memory 鏇存柊锛?                       | `subgoal-completion conditioned`                                          | 瀛愮洰鏍囧畬鎴愩€佷换鍔￠樁娈甸棴鍚堛€乼rajectory chunk 瀹屾垚                                 | HiAgent 鏈€鍏抽敭鐨?trigger 涓嶅湪閫愭鍐欏叆锛岃€屽湪鈥滀粈涔堟椂鍊欐妸搴曞眰鎵ц杞ㄨ抗鎬荤粨鎴愪笂灞?working memory chunk鈥濄€傚畠鏄庣‘浠?subgoal 涓?memory chunk锛屽洜姝?summarization / abstraction 鏄?`subgoal-completion conditioned`锛岃繖鏄鏂囨牳蹇冧箣涓€銆?          |
| 12  | Memory OS of AI Agent                                                         | secondary | `type-routing / multi-label routing`                      | `capacity/budget-conditioned`                                             | 鐭?涓?闀挎湡灞傜骇鍒ゆ柇銆佽闂鐜囥€佺敓鍛藉懆鏈熼樁娈?                                          | 瑙﹀彂鐐瑰叾瀹為兘鏄€滃瓨婊?瓒呴檺鍐嶈縼绉绘垨娣樻卑鈥?                                                                                                                                                                   |
| 13  | Mem0                                                                          | generic   | `threshold(score)`                                        | `new-write conditioned`                                                   | 璁板繂鎶藉彇缃俊銆佺浉鍏虫€с€佹洿鏂板啿绐佹娴?                                               | Mem0 寮鸿皟 selective memory formation锛屼絾 selectivity 涓昏钀藉湪鈥滄槸鍚︽娊鍙栧嚭鍊煎緱瀛樼殑 memory item銆佹槸鍚︿笌鏃ц蹇?merge/update鈥濄€傝繖鏇村儚 extraction/update 閫昏緫锛屼笉鏄嫭绔?trigger 瀹舵棌锛涘彲鐢?`memory-worthiness threshold` 姒傛嫭銆?         |
| 14  | A-MEM                                                                         | secondary | `threshold(score)` + `type-routing / multi-label routing` | `new-write conditioned` + `boolean signal gate`                           | 閲嶈搴︺€佸叧鑱旈偦灞呫€佸彲閾炬帴鎬с€乤gent 鍐崇瓥                                           | A-MEM 鐨?agent 浼氬垽鏂柊淇℃伅鍊间笉鍊煎緱鎴愪负 note銆佸簲濡備綍閾炬帴鍒扮幇鏈?note 缃戠粶锛涘悗缁浘缁存姢甯哥敱鏂板啓鍏ユ垨閭诲眳鍏宠仈瑙﹀彂銆傚叾澶嶆潅搴︿富瑕佸湪 graph-note organization锛屼笉鍦?trigger 璇█鏈韩銆?                                                                    |
| 15  | HippoRAG                                                                      | generic   | `always`                                                  | `new-write conditioned`                                                   | 鏂版钀借繘鍏ャ€乼riplet 鎶藉彇瀹屾垚銆乹uery association                             | HippoRAG 鐨勭绾块樁娈靛熀鏈槸鈥滄枃鏈潵浜嗗氨鎶?triple / build graph鈥濓紝鍦ㄧ嚎闃舵鏄浘妫€绱笌鑱旀兂寮?recall銆傝Е鍙戣繎浼?`always write`锛屽叧閿紓璐ㄦ€у湪 graph retrieval 鍜?propagation銆?                                                           |
| 16  | From RAG to Memory / HippoRAG 2                                               | generic   | `always`                                                  | `new-write conditioned` / `鏃犳硶琛ㄧず锛堢绾垮缓鍥鹃樁娈碉級`                                  | 绂荤嚎寤哄浘闃舵銆乸aragraph integration銆乹uery-time association              | HippoRAG 2 鐨勨€渙ffline鈥濇洿鍍忕郴缁熺绾块樁娈碉紝鑰屼笉鏄懆鏈?浼氳瘽缁撴潫/绌洪棽瑙﹀彂銆傚師鏂囨。鎶婂畠璁颁负 `scheduled/offline` 涓嶇畻閿欙紝浣嗗湪鏂板垎绫讳笅鏇撮€傚悎鏄庣‘鍐欐垚鈥滅绾垮缓鍥鹃樁娈垫棤娉曠敱鐜版湁 trigger 鍘熷瀷绮剧‘琛ㄧず鈥濄€?                                                               |
| 17  | AriGraph                                                                      | generic   | `always`                                                  | `new-write conditioned`                                                   | 鏂?observation銆佸浘鏇存柊闇€姹傘€乬raph pruning                                | AriGraph 鍦ㄦ帰绱㈢幆澧冩椂鎸佺画鎶?observation 鍐欏叆 episodic + semantic graph锛屽苟鍦ㄦ瘡姝ュ悗鍋?world-model 鏇存柊銆傛洿璐磋繎 `always write` + `new-write conditioned graph update`锛涚湡姝ｉ毦鐐瑰湪鍥剧粨鏋勫浣曞缓妯″拰瑁佸壀銆?                           |
| 18  | Zep / Graphiti                                                                | secondary | `always`                                                  | `new-write conditioned`                                                   | 鏂板璇?涓氬姟浜嬩欢銆佹椂闂存埑銆佸巻鍙插叧绯诲彉鍖?                                             | 澶嶆煡鍚庢洿绋冲Ε鐨勮娉曟槸锛欸raphiti 涓昏鐢辨柊浜嬩欢鍒拌揪椹卞姩鏃堕棿鐭ヨ瘑鍥炬洿鏂帮紝鍘熸枃骞舵病鏈夋妸鍚庡彴鏁寸悊鏄庣‘鍐欐垚鍥哄畾鍛ㄦ湡鎴?idle 缁存姢銆傚洜姝ゅ幓鎺夋棫鐗堥噷鍋忓鐨?`scheduled/offline`锛屽彧淇濈暀 `new-write conditioned`銆?                                                          |
| 19  | Bridging Intuitive Associations and Deliberate Recall                         | generic   | `always`                                                  | `always` / `new-write conditioned`                                        | 鏂?utterance銆佷簨浠舵娊鍙栥€佹煡璇㈡剰鍥?                                           | Associa 鐨勫叧閿槸 event-centric graph 涓庝袱闃舵 retrieval銆俶emory graph 鐨勫舰鎴愬鏂板璇濆熀鏈槸鎸佺画鐨勶紝涓昏澶嶆潅搴﹀湪鈥滅洿瑙夎仈鎯?+ 娣辨€濇绱⑩€濈殑璇诲彇杩囩▼锛岃€屼笉鏄綍鏃跺啓鍏ャ€?                                                                           |
| 20  | From Experience to Strategy                                                   | secondary | `always`                                                  | `threshold(score)` / `鏃犳硶琛ㄧず锛堣缁冨紡绛栫暐钂搁闃舵锛塦                                    | reward feedback銆佺粡楠?utility銆乻trategy weight                       | 杩欑瘒鐨勯珮灞?strategy abstraction 鏇村儚璁粌鎴栬捀棣忛樁娈碉紝鑰屼笉鏄槑纭殑鍛ㄦ湡銆乻ession-end 鎴?idle 瑙﹀彂銆傚洜姝ゆ棫鐗堢殑 `scheduled/offline` 闇€瑕佹敹鍥烇紝淇濈暀 `utility threshold` 鏇寸ǔ濡ャ€?                                                         |
| 21  | Hierarchical Memory Organization for Wikipedia Generation                     | generic   | `always`                                                  | `鏃犳硶琛ㄧず锛堢绾垮眰绾х粍缁囩绾匡級`                                                          | 鏂囨。 chunk 杩涘叆銆佸眰绾х粍缁囨祦绋嬪惎鍔?                                            | MOG 鐨勫眰绾х粍缁囨槸绂荤嚎 pipeline锛屾湰璐ㄤ笂涓嶆槸 trigger 璁烘枃銆傛棫鐗堝啓 `scheduled/offline` 瀹规槗璁╁畠鐪嬭捣鏉ュ儚瀛樺湪鏄庣‘ maintenance 瑙﹀彂锛涜繖閲屾敼鎴愮洿鎺ユ壙璁も€滃綋鍓?trigger 璇嶈〃鏃犳硶绮剧‘琛ㄧず鈥濄€?                                                              |
| 22  | Optimizing the Interface Between KGs and LLMs                                 | generic   | `always`                                                  | `always`                                                                  | ingestion銆佹绱㈣秴鍙傛暟銆佹彁绀哄弬鏁?                                            | 杩欑瘒鏈川鏄郴缁熻皟鍙備笌鎺ュ彛浼樺寲锛屼笉鏄?memory trigger 璁烘枃銆傚彲浠ヨ涓哄浘鏋勫缓涓庢绱㈤粯璁ゆ墽琛岋紝娌℃湁鍊煎緱鍗曞垪鐨?trigger 鍘熷瀷銆?                                                                                                                 |
| 23  | AI-native Memory 2.0: Second Me                                               | secondary | `type-routing / multi-label routing`                      | `new-write conditioned` / `鏃犳硶琛ㄧず锛堝悗鍙板弬鏁板寲涓庤縼绉伙級`                                | 鐢ㄦ埛淇℃伅绫诲瀷銆佽蹇嗗眰绾с€佸弬鏁板寲鏃舵満                                                | Second Me 纭湁鍚庡彴鍙傛暟鍖栦笌灞傞棿杩佺Щ锛屼絾鍘熸枃娌℃湁缁欏嚭瓒冲鏄庣‘鐨勫懆鏈熴€乻ession-end 鎴?idle 鏉′欢銆傛棫鐗?`scheduled/offline` 鍋忔硾鍖栵紱杩欓噷淇濈暀璺敱涓庡啓鍚庢紨鍖栵紝鎶婂墿浣欓儴鍒嗚涓烘棤娉曡〃绀烘洿绋冲Ε銆?                                                                     |
| 24  | O-Mem                                                                         | secondary | `type-routing / multi-label routing`                      | `new-write conditioned` / `鏃犳硶琛ㄧず锛堝悗鍙板眰绾ф暣鐞嗭級`                                  | persona 灞炴€ф娊鍙栥€乪vent record 鎶藉彇銆乼opic/context 灞傜骇                    | O-Mem 鐨勫悗缁暣鐞嗘洿鍍忕郴缁熺粍缁囬樁娈碉紝鑰屼笉鏄柊涓夌被閲屾煇涓ǔ瀹?trigger銆傝繖閲岄『鎵嬩慨姝ｆ棫鐗堬紝鎶?`scheduled/offline` 鍘绘帀锛岄伩鍏嶆妸鈥滄湁鍚庡彴鏁寸悊鈥濊鍐欐垚鈥滄湁鏄庣‘ schedule鈥濄€?                                                                                 |
| 25  | In Prospect and Retrospect                                                    | secondary | `SessionEndTrigger`                                       | `boolean signal gate` + `failure/outcome-conditioned`                     | session/turn/utterance 绮掑害杈圭晫銆丩LM cited evidence銆佹绱㈣宸?             | 澶嶆牳鍚庡簲鍘绘帀 `subgoal-completion conditioned`銆侾rospective Reflection 鏄寜 utterance / turn / session 澶氱矑搴﹁竟鐣岀敓鎴愭€荤粨锛屼笉鏄换鍔″瓙鐩爣瀹屾垚瑙﹀彂銆?                                           |
| 26  | Nemori                                                                        | core-ish  | `boolean signal gate`                                     | `failure/outcome-conditioned` + `boolean signal gate`                     | event boundary銆乪pisode 瀵归綈淇″彿銆乸rediction gap銆乧alibration error     | 澶嶆牳鍚庝笉搴斿啀鍐?`subgoal-completion conditioned`銆侼emori 鐨勫啓鍏ヨ竟鐣屾潵鑷?event segmentation / episode alignment锛岃€屼笉鏄樉寮忓瓙鐩爣瀹屾垚锛涘叾婕斿寲鍒欑敱 prediction gap 绛変俊鍙烽┍鍔ㄣ€?                                 |
| 27  | MemOS                                                                         | generic   | `type-routing / multi-label routing`                      | `PeriodicTrigger` + `capacity/budget-conditioned`                         | memory form銆佺増鏈?鏉ユ簮鍏冩暟鎹€佽縼绉绘垚鏈?                                       | MemOS 纭疄瀛樺湪闈㈠悜 memory scheduling 鐨勫悗鍙拌皟搴︼紱鍦ㄦ柊鍒嗙被涓嬶紝鎶婂畠鏀剁獎涓?`PeriodicTrigger` 姣?`scheduled/offline` 鏇磋创鍒囷紝鍥犱负璁烘枃璁ㄨ鐨勬槸鎸佺画璋冨害涓庤祫婧愮紪鎺掞紝鑰屼笉鏄?session-end 鎴?sleep-time 璇箟銆?                                        |
| 28  | MIRIX                                                                         | core      | `type-routing / multi-label routing`                      | `capacity/budget-conditioned` + `PeriodicTrigger` + `boolean signal gate` | memory type classifier銆佹ā鎬佹潵婧愩€佹椿璺冧娇鐢ㄥ害銆佸瓨鍌ㄩ绠椼€乵ulti-agent controller 鍐崇瓥 | MIRIX 鏄庣‘鐢卞浠ｇ悊鍗忓悓鎺у埗鍏被 memory 鐨勬洿鏂颁笌妫€绱€傛妸鏃х殑 `scheduled/offline` 鎷嗗紑鍚庯紝杩欓噷鏇存帴杩?controller 椹卞姩鐨勫懆鏈熸€х淮鎶わ紝鑰屼笉鏄?session-end 鎴?idle-only 璇箟锛屽洜姝よ惤鍒?`PeriodicTrigger` 鏇寸ǔ濡ャ€?                                      |
| 29  | SEDM                                                                          | core-ish  | `boolean signal gate` + `threshold(score)`                | `PeriodicTrigger` + `threshold(score)`                                    | verifiable replay 閫氳繃涓庡惁銆佺粡楠?utility 鎺掑悕銆佽法鍩熸墿鏁ｄ环鍊?                     | SEDM 鏄庣‘鍐欏埌 self-scheduling controller锛涘湪鏂板垎绫讳笅锛岃繖鏇村儚 `PeriodicTrigger` 椹卞姩鐨?consolidation锛岃€屼笉鏄硾娉涚殑 offline銆傚師鏂囨。鐨勫ぇ鏂瑰悜鏄鐨勶紝浣嗛渶瑕佹媶缁嗐€?                                                                    |
| 30  | LightMem                                                                      | core      | `capacity/budget-conditioned`                             | `IdleTrigger` + `new-write conditioned`                                   | buffer 瀹归噺銆乼opic grouping 瀹屾垚銆乻leep-time 绐楀彛                        | LightMem 鏄渶鏄庣‘鐨?trigger-rich 璁烘枃涔嬩竴銆傛妸鏃х被鎷嗗紑鍚庯紝杩欓噷鐨勭绾垮珐鍥哄簲鐩存帴鍐欐垚 `IdleTrigger`锛屽洜涓鸿鏂囨槑纭娇鐢?sleep-time window锛涜繖涓€鐐规瘮鏃х殑 `scheduled/offline` 鏇村噯纭€?                                                          |
| 31  | Towards LifeSpan Cognitive Systems                                            | generic   | `always`                                                  | `鏃犳硶琛ㄧず锛堥暱鏈熷惛鏀堕樁娈碉級`                                                            | 鏂?experience銆侀暱鏈熷惛鏀堕樁娈?                                             | 杩欐洿鍍忎笂浣嶈摑鍥俱€傝櫧鐒惰璁?experience absorbing 涓?response generation锛屼絾娌℃湁缁欏嚭瓒冲鍏蜂綋鐨勫懆鏈熴€乻ession-end 鎴?idle 瑙﹀彂锛屽洜姝よ繖閲屼笉鍐嶅己琛屾斁杩涙柊涓夌被銆?                                                                              |
| 32  | Human-inspired Episodic Memory for Infinite Context LLMs / EM-LLM             | generic   | `boolean signal gate`                                     | `boolean signal gate` + `new-write conditioned`                           | `surprise`銆乪vent boundary銆佷袱闃舵 retrieval 鍛戒腑                       | EM-LLM 鐨勯噸瑕佹帶鍒剁偣鍦?event segmentation锛氫笂涓嬫枃涓嶆槸浠绘剰瀹氶暱鍒囧潡锛岃€屾槸鐢?`surprise` 淇″彿鍜岃竟鐣岀簿鐐兼満鍒跺垏鎴?episodic units銆傚畠纭疄鏈?signal锛屼絾鏇村儚 unit formation signal锛岃€屼笉鏄鏉?memory-action trigger 浣撶郴銆?                          |
| 33  | MemVerse                                                                      | secondary | `always`                                                  | `PeriodicTrigger` + `threshold(score)`                                    | multimodal distillation 鍛ㄦ湡銆侀噸瑕佹牱鏈€夋嫨                                | MemVerse 鐨勫叧閿洿鍍?multimodal memory + periodic distillation銆傝繖閲屽彲浠ユ槑纭惤鍒?`PeriodicTrigger`锛屾棫鐗?`scheduled/offline` 鐨勬剰鎬濆熀鏈纭紝浣嗘柊鍐欐硶鏇寸簿纭€?                                                            |
| 34  | MGA                                                                           | generic   | `always`                                                  | `new-write conditioned`                                                   | GUI observation銆乼rajectory step                                  | MGA 闈㈠悜 observation-centric GUI memory锛岄€氬父姣忎釜 step / state 閮芥寔缁舰鎴愮粨鏋勫寲澶栭儴璁板繂銆傚悗缁洿鏂板彧鏄殢鐫€鏂拌建杩瑰啓鍏ヨ嚜鐒跺彂鐢燂紝trigger 涓嶆瀯鎴愬崟鐙垱鏂扮偣銆?                                                                               |
| 35  | KARMA                                                                         | secondary | `type-routing / multi-label routing`                      | `capacity/budget-conditioned`                                             | short-term vs long-term scene 淇℃伅銆佸璞＄姸鎬佸彉鍖栥€佺煭鏈熸Ы浣嶆浛鎹㈠帇鍔?                | KARMA 鍖哄垎闀挎湡 3D scene graph 鍜岃杞藉眬閮ㄥ彉鍖栫殑鐭湡璁板繂锛屽苟鏄庣‘鏈?short-term replacement 绛栫暐銆傚啓鍏ラ渶瑕佸厛鍒ゆ柇杩涘摢绫?store锛岀煭鏈熸洿鏂板張鍙楀閲忓拰閲嶈鍙樺寲褰卞搷锛屽洜姝ゅ彲鏄犲皠鎴?`type-routing` + `capacity-conditioned replacement`銆?                     |
| 36  | VideoAgent                                                                    | generic   | `boolean signal gate`                                     | `failure/outcome-conditioned` / `boolean signal gate`                     | query-driven frame request銆佸綋鍓嶈瘉鎹笉瓒?                               | VideoAgent 涓嶆槸鍏稿瀷闀挎湡 memory 璁烘枃锛屼絾濡傛灉纭鏄犲皠 trigger锛屽叾鎺у埗鐐瑰湪 agent 浣曟椂缁х画璋冪敤瑙嗚宸ュ叿銆佷綍鏃舵绱㈡洿澶氬抚銆傝繖涓?trigger 鏄?query-driven evidence gathering锛屼笉鏄?memory 鍐欏叆 trigger銆?                                          |
| 37  | WorldMM                                                                       | secondary | `type-routing / multi-label routing`                      | `boolean signal gate` / `鏃犳硶琛ㄧず锛堣渚ф椂搴忔帶鍒讹級`                                    | query 绫诲瀷銆佹墍闇€妯℃€併€佹椂闂寸矑搴﹂€夋嫨銆佽瘉鎹厖鍒嗘€?                                      | WorldMM 鐨勯潪骞冲嚒鎺у埗涓昏鍦?retrieval routing 涓?temporal granularity 閫夋嫨銆傛棫鐗堟妸瀹冨苟鍏?`scheduled/offline` 鍋忓锛涙洿鍑嗙‘鐨勬槸鎵胯杩欓噷涓昏鏄渚ф帶鍒讹紝涓嶉€傚悎纭杩涙柊鐨?maintenance trigger 涓夊垎娉曘€?                                         |
| 38  | Seeing, Listening, Remembering, and Reasoning / M3-Agent                      | generic   | `type-routing / multi-label routing`                      | `new-write conditioned`                                                   | 瀹炰綋鎶藉彇銆佹ā鎬佹潵婧愩€佸疄浣撶骇鏇存柊                                                  | M3-Agent 浼氬皢瑙嗚銆佸惉瑙夎緭鍏ョ粍缁囦负 entity-centric multimodal memory锛屽苟鎸佺画鏇存柊 episodic / semantic memory銆傚畠纭疄鏈夆€滄寜瀹炰綋/妯℃€佽惤搴撯€濈殑璺敱淇″彿锛屼絾杩欐洿澶氭槸琛ㄧず灞傚喅瀹氾紝涓嶆槸澶嶆潅 trigger 璁捐銆?                                                |
| 39  | HippoMM                                                                       | generic   | `boolean signal gate`                                     | `IdleTrigger` + `new-write conditioned`                                   | adaptive temporal segmentation銆佽法妯℃€佽仈鎯抽渶姹?                          | HippoMM 鐢?adaptive temporal segmentation 褰㈡垚 episodic units锛屽啀鍋氱煭闀挎椂宸╁浐涓庤法妯℃€佹绱€傝嫢蹇呴』钀藉埌鏂颁笁绫伙紝杈冪ǔ濡ョ殑鏄妸绫讳技 consolidation 鐨勭绾垮珐鍥哄啓鎴?`IdleTrigger`锛涗絾杩欎粛鐒朵笉鏄鏂囦富杞淬€?                                             |
| 40  | Episodic Memory Representation for Long-form Video Understanding              | generic   | `boolean signal gate`                                     | `boolean signal gate`                                                     | episodic event boundary銆丆oT 閫夋嫨鐨勫叧閿?episodic subset                | Video-EM 鐨勫叧閿槸鎶婂叧閿抚瑙嗕负 temporally ordered episodic events锛屽苟鐢?CoT 杩唬鎸戝嚭鏈€灏忎絾淇℃伅閲忔渶楂樼殑 episodic subset銆傝繖閲岀殑鈥滆Е鍙戔€濇洿澶氭槸浜嬩欢鍒嗘鍜屾绱㈣凯浠ｆ潯浠讹紝涓嶆槸闀挎湡 memory action trigger銆?                                          |


## 鏇翠弗鏍肩殑缁熻

濡傛灉鍙妸鈥渢rigger 鏄庣‘鏋勬垚绯荤粺楠ㄦ灦鈥濈殑璁烘枃绠椾綔涓ユ牸 `core`锛?
- Reflexion
- MemGPT
- HiAgent
- MIRIX
- LightMem

杩欐槸淇濆畧鐨?`5 / 40`銆?
濡傛灉鎶?trigger 璁捐涔熼潪甯哥獊鍑恒€佷絾鏇村亸鎺у埗鍘熷垯鑰岄潪瀹屾暣宸ョ▼璇箟鐨勬潯鐩畻杩涙潵锛?
- Nemori
- SEDM

鍒欐槸 `7 / 40`銆?
濡傛灉鍐嶆妸鈥滄湁鏄庣‘鑰屽繀瑕佺殑闈炲钩鍑?trigger锛屼絾涓诲垱鏂颁粛鍦ㄥ埆澶勨€濈殑 `secondary` 涔熺撼鍏ワ細

- Generative Agents
- MemoryBank
- My Agent Understands Me Better
- Compress to Impress
- Agent Workflow Memory
- Memory OS of AI Agent
- A-MEM
- Zep
- From Experience to Strategy
- Second Me
- O-Mem
- In Prospect and Retrospect
- MemVerse
- KARMA
- WorldMM

鍒欏ぇ鑷存槸 `22 / 40` 鑷冲皯鍏锋湁鈥滃彲鍛藉悕鐨?trigger 鍘熷瀷鈥濓紝浣嗗叾涓粷澶у鏁颁粛鑳借钖?trigger 璇嶈〃瑕嗙洊銆?
閲嶆柊閫愮瘒鏍稿鍚庯紝鍏充簬鏃х殑 `scheduled/offline` 杩橀渶瑕佽ˉ涓€灞傝鏄庯細

- 鐪熸鑳芥槑纭惤鍒?`PeriodicTrigger` 鐨勶紝鍙崰灏戞暟锛屼唬琛ㄦ€ф潯鐩槸 MemOS銆丮IRIX銆丼EDM銆丮emVerse
- 鐪熸鑳芥槑纭惤鍒?`SessionEndTrigger` 鐨勪篃涓嶅锛屼唬琛ㄦ€ф潯鐩槸 Compress to Impress銆両n Prospect and Retrospect
- 鐪熸鑳芥槑纭惤鍒?`IdleTrigger` 鐨勬洿灏戯紝鏈€鍏稿瀷鐨勬槸 LightMem锛汬ippoMM 鍙兘绠椾繚瀹堣繎浼?- 鍏朵綑涓嶅皯鏃ф潯鐩櫧鐒垛€滄湁绂荤嚎闃舵/鍚庡彴缁存姢鈥濓紝浣嗗師鏂囧苟娌℃湁缁欏嚭瓒冲鏄庣‘鐨勫懆鏈熴€佷細璇濈粨鏉熸垨绌洪棽绐楀彛璇箟锛屽洜姝ゆ洿璇氬疄鐨勫啓娉曟槸 `鏃犳硶琛ㄧず`

## 瀵?MemPrimitive 鐨勭洿鎺ュ惎鍙?
### 1. 鏈€鍊煎緱淇濈暀鐨勯珮浠峰€?trigger 鍘熷瀷

濡傛灉鐩爣鏄€滃紑濮嬮噸璁捐鏁翠釜 trigger 绫烩€濓紝鎴戝缓璁妸鈥滈珮棰戝嚭鐜扳€濅笌鈥滃€煎緱鍋氭垚涓€绛夊疄鐜扳€濆垎寮€鐪嬨€?
鍏堣缁撹锛?
- **寤鸿鍋氭垚涓€绛?trigger 绫荤殑**锛?  - `threshold(score)`
  - `boolean signal gate`
  - `failure/outcome-conditioned`
  - `capacity/budget-conditioned`
  - `PeriodicTrigger`
  - `SessionEndTrigger`
  - `IdleTrigger`
  - `type-routing / multi-label routing`
  - `subgoal-completion conditioned`
- **寤鸿淇濈暀锛屼絾涓嶈鍋氭垚鍘氶噸绫诲眰娆＄殑**锛?  - `always`
  - `new-write conditioned`

鍘熷洜鏄細

- `always` 铏界劧鍑虹幇澶氾紝浣嗗畠鏇村儚榛樿绌鸿Е鍙戝櫒锛岄€氬父涓嶅€煎緱浣滀负澶嶆潅绫诲疄鐜?- `new-write conditioned` 涔熷嚭鐜板緢澶氾紝浣嗘湰璐ㄤ笂甯稿父鍙槸鈥滄煇娆?write 涔嬪悗鑷姩鎸傛帴鐨勫悗澶勭悊鈥濓紝鏇撮€傚悎鍋氫簨浠堕挬瀛愭垨 pipeline edge锛岃€屼笉鏄嫭绔嬬瓥鐣ュ璞?- 鐪熸鍊煎緱鍗曠嫭寤烘ā鐨勶紝鏄偅浜涗細鏄捐憲鏀瑰彉绯荤粺鎺у埗娴併€乵emory routing 鎴?maintenance scheduling 鐨?trigger

涓嬮潰鎸夊嚭鐜版鏁扮粺璁★紝骞朵繚鐣欏嚭鐜板湪鍝簺璁烘枃涓殑淇℃伅銆?

| trigger 鍘熷瀷                           | 鍑虹幇绡囨暟 | 鏄惁寤鸿浣滀负涓€绛夊疄鐜?| 璇存槑                                                                                          | 鍑虹幇浜?                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------ | ---- | ---------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PeriodicTrigger`                    | 4    | 鏄?         | 鐪熸鏄庣‘鍐欐垚鍛ㄦ湡璋冨害鐨勫苟涓嶅锛屼絾涓€鏃﹀嚭鐜伴€氬父灏辨槸绯荤粺缁存姢楠ㄦ灦銆?                                                            | MemOS锛汳IRIX锛汼EDM锛汳emVerse                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `SessionEndTrigger`                  | 2    | 鏄?         | 棰戞涓嶉珮锛屼絾寰堢ǔ瀹氬湴瀵瑰簲鈥渃hunk / turn / session 鏀舵潫鍚庡啀鍘嬬缉鎴栨€荤粨鈥濄€?                                            | Compress to Impress锛汭n Prospect and Retrospect                                                                                                                                                                                                                                                                                                                                                                                              |
| `IdleTrigger`                        | 2    | 鏄?         | 鏈€鍏稿瀷鐨勬槸 sleep-time / idle-window consolidation銆傛槑纭嚭鐜扮殑璁烘枃寰堝皯锛屼絾璇箟闈炲父娓呮銆?                            | LightMem锛汬ippoMM锛堜繚瀹堣繎浼硷級                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `new-write conditioned`              | 18   | 鍚︼紝寤鸿闄嶄负浜嬩欢閽╁瓙 | 杩欑被瑙﹀彂楂橀鍑虹幇锛屼絾澶氭暟涓嶆槸璁烘枃涓诲垱鏂帮紝鑰屾槸鈥滄柊璁板繂鍐欏叆鍚庤嚜鍔ㄥ仛 link / merge / consolidate / organize鈥濄€傛洿閫傚悎鍋?`on_write` 閽╁瓙銆?| Voyager锛汻ET-LLM锛汿oward Conversational Agents with Context and Time Sensitive Long-term Memory锛汚gent Workflow Memory锛汳em0锛汚-MEM锛汬ippoRAG锛汧rom RAG to Memory / HippoRAG 2锛汚riGraph锛沍ep / Graphiti锛汢ridging Intuitive Associations and Deliberate Recall锛汚I-native Memory 2.0: Second Me锛汷-Mem锛汱ightMem锛汬uman-inspired Episodic Memory for Infinite Context LLMs / EM-LLM锛汳GA锛汼eeing, Listening, Remembering, and Reasoning / M3-Agent锛汬ippoMM |
| `always`                             | 14   | 鍚︼紝寤鸿浣滀负榛樿鍊? | 鍑虹幇棰戠巼楂樿鏄庡緢澶氳鏂囨牴鏈笉渚濊禆澶嶆潅瑙﹀彂锛涗絾杩欐伆鎭拌鏄庡畠涓嶈鏄鏉傜被锛岃€屽簲鏄粯璁?no-op / passthrough trigger銆?                      | RET-LLM锛汿oward Conversational Agents with Context and Time Sensitive Long-term Memory锛汬iAgent锛汬ippoRAG锛汧rom RAG to Memory / HippoRAG 2锛汚riGraph锛沍ep / Graphiti锛汢ridging Intuitive Associations and Deliberate Recall锛汧rom Experience to Strategy锛汬ierarchical Memory Organization for Wikipedia Generation锛汷ptimizing the Interface Between KGs and LLMs锛汿owards LifeSpan Cognitive Systems锛汳emVerse锛汳GA                                    |
| `boolean signal gate`                | 14   | 鏄?         | 杩欐槸鏈€閫氱敤銆佹渶璇ヤ繚鐣欑殑鍩虹鍘熷瀷銆傚緢澶氳鏂囧叾瀹炰笉闇€瑕?scorer/gate/policy 鍏ㄥ锛屽彧闇€瑕佲€滄煇涓俊鍙峰嚭鐜板氨瑙﹀彂鈥濄€?                            | Voyager锛汳emGPT锛汳y Agent Understands Me Better锛汚gent Workflow Memory锛汚-MEM锛汭n Prospect and Retrospect锛汵emori锛汳IRIX锛汼EDM锛汬uman-inspired Episodic Memory for Infinite Context LLMs / EM-LLM锛沄ideoAgent锛沇orldMM锛汬ippoMM锛汦pisodic Memory Representation for Long-form Video Understanding                                                                                                                                                        |
| `type-routing / multi-label routing` | 10   | 鏄?         | 杩欐槸灏戞暟鐪熸浼氭敼鍙?memory topology 鐨?trigger 鍘熷瀷銆傚浜庡 store / 澶?memory type 绯荤粺闈炲父鍏抽敭銆?                   | MemGPT锛汳emory OS of AI Agent锛汚-MEM锛汚I-native Memory 2.0: Second Me锛汷-Mem锛汳emOS锛汳IRIX锛汯ARMA锛沇orldMM锛汼eeing, Listening, Remembering, and Reasoning / M3-Agent                                                                                                                                                                                                                                                                                 |
| `threshold(score)`                   | 9    | 鏄?         | 杩欐槸鏈€甯歌鐨勨€滆交閲忔櫤鑳介€夋嫨鈥濆師鍨嬨€傛瘮璧峰鏉?policy锛屽緢澶氳鏂囧彧闇€瑕?importance / utility / retention / worthiness 杩囬槇鍊笺€?    | Generative Agents锛汳emoryBank锛汳y Agent Understands Me Better锛汚gent Workflow Memory锛汳em0锛汚-MEM锛汧rom Experience to Strategy锛汼EDM锛汳emVerse                                                                                                                                                                                                                                                                                                      |
| `capacity/budget-conditioned`        | 6    | 鏄?         | 鍑虹幇棰戠巼涓嶇畻鏈€楂橈紝浣嗗伐绋嬩环鍊兼瀬楂橈紝鍥犱负瀹冩濂藉搴?buffer銆亀indow銆乻lot銆乻torage budget 杩欎簺鐪熷疄绯荤粺鍘嬪姏銆?                       | MemGPT锛汳emory OS of AI Agent锛汳emOS锛汳IRIX锛汱ightMem锛汯ARMA                                                                                                                                                                                                                                                                                                                                                                                     |
| `failure/outcome-conditioned`        | 5    | 鏄?         | 铏界劧鍙湪灏戞暟璁烘枃涓嚭鐜帮紝浣嗚繖浜涜鏂囪鲸璇嗗害寰堥珮锛屽挨鍏舵槸 Reflexion 绯汇€傝繖涓師鍨嬪繀椤讳繚鐣欍€?                                           | Reflexion锛沄oyager锛汭n Prospect and Retrospect锛汵emori锛沄ideoAgent                                                                                                                                                                                                                                                                                                                                                                              |
| `subgoal-completion conditioned`     | 1    | 鏄?         | 棰戞寰堜綆锛屼絾涓€鏃﹀嚭鐜伴€氬父灏辨槸绯荤粺鏍稿績銆備弗鏍煎鏍稿悗锛屽綋鍓?40 绡囬噷鏈€绋冲Ε鐨勫吀鍨嬩富瑕佸氨鏄?HiAgent銆?                                        | HiAgent                                                                                                                                                                                                                                                                                                                                                                                                                                      |


鍥犳锛屽鏋滃彧淇濈暀涓€涓?*鏈€灏忎絾澶熺敤**鐨?trigger 绫婚泦鍚堬紝鎴戝缓璁槸锛?
1. `AlwaysTrigger`
  - 浣滀负榛樿瀹炵幇瀛樺湪锛屼絾灏介噺钖勶紝涓嶈缁х画鍫嗗鏉傚瓙绫?2. `ThresholdTrigger`
  - 瑕嗙洊 importance / utility / retention / worthiness
3. `BooleanSignalTrigger`
  - 瑕嗙洊 cue銆乪vent boundary銆乸rediction gap銆乪vidence sufficiency 绛夊竷灏斾俊鍙?4. `FailureTrigger`
  - 涓撻棬淇濈暀缁?Reflexion / outcome-driven systems
5. `CapacityTrigger`
  - 瑕嗙洊 buffer / token / slot / budget pressure
6. `PeriodicTrigger`
  - 瑕嗙洊鍥哄畾鍛ㄦ湡銆佸浐瀹氳疆鏁般€佽皟搴﹁妭鎷嶈Е鍙?7. `SessionEndTrigger`
  - 瑕嗙洊 session / turn / chunk / stage 鏀舵潫
8. `IdleTrigger`
  - 瑕嗙洊 sleep-time / idle-window / 鍚庡彴绌洪棽缁存姢
9. `RoutingTrigger`
  - 瑕嗙洊 type-routing / multi-label routing / store selection
10. `SubgoalTrigger`
  - 瑕嗙洊闃舵闂悎銆佸瓙鐩爣瀹屾垚銆乪pisode close

鍏朵腑锛?
- `NewWriteConditioned` 涓嶅缓璁仛鎴愮嫭绔?heavyweight trigger class
- 鏇村缓璁妸瀹冮檷鎴愶細
  - `on_write` hook
  - `after_write` maintenance edge
  - 鎴栬€?`PeriodicTrigger` / `SessionEndTrigger` / `IdleTrigger` / `RoutingTrigger` 鍐呴儴鍙粍鍚堟潯浠?
濡傛灉杩樿鍐嶇爫涓€鍒€锛屽彧淇濈暀**鐪熸蹇呰**鐨勪竴缁勶紝鎴戜細鍘绘帀 `AlwaysTrigger` 鐨勨€滅被鍦颁綅鈥濓紝鎶婂畠褰撻粯璁ら厤缃€硷紝鍙繚鐣欙細

1. `ThresholdTrigger`
2. `BooleanSignalTrigger`
3. `FailureTrigger`
4. `CapacityTrigger`
5. `PeriodicTrigger`
6. `SessionEndTrigger`
7. `IdleTrigger`
8. `RoutingTrigger`
9. `SubgoalTrigger`

杩?9 绫诲凡缁忚冻澶熻鐩栧綋鍓?40 绡囬噷缁濆ぇ澶氭暟闈炲钩鍑?trigger 璁捐锛屽悓鏃朵笉浼氭妸 trigger 瀛愮郴缁熼噸鏂板仛閲嶏紱鑰岄偅浜涘彧鏈夆€滃悗鍙伴樁娈碘€濅絾娌℃湁鏄庣‘璋冨害璇箟鐨勭郴缁燂紝鍒欏簲鐩存帴鍏佽鍐欐垚 `鏃犳硶琛ㄧず`銆?
### 2. 鍝簺 signal 鍊煎緱鍦ㄦ鏋堕噷浣滀负涓€绛夎緭鍏?
濡傛灉 trigger 灞傝钖勶紝浣嗕粛淇濈暀璁烘枃瀵归綈鑳藉姏锛屾渶鍊煎緱鏄惧紡鏀寔鐨?signal 鏄細

- `importance`
- `salience`
- `relevance`
- `novelty`
- `retention / forgetting strength`
- `consolidation strength`
- `memory pressure / capacity usage`
- `subgoal_done`
- `failure_detected`
- `prediction_gap`
- `event_boundary`
- `cue_match`
- `type_label`
- `utility / reward`
- `evidence_sufficiency`

### 3. 鏇村悎鐞嗙殑绯荤粺杈圭晫

杩欎竴杞€愮瘒鏄犲皠涔嬪悗锛屾洿娓呮鐨勪竴鐐规槸锛?
- 澶氭暟璁烘枃骞朵笉闇€瑕?`signal provider -> scorer -> gate -> policy` 杩欏鍘?trigger 鍫嗘爤鍏ㄥ紑
- 寰堝鎵€璋?trigger 宸紓锛屾渶缁堥兘鍙槸锛?  - 涓€涓畝鍗?signal
  - 涓€涓槇鍊?  - 涓€涓竷灏旀潯浠?  - 涓€涓?schedule
  - 涓€涓矾鐢辨爣绛?
鍥犳姣旇緝鑷劧鐨勬柟鍚戞槸锛?
- trigger 鍙礋璐ｂ€滀綍鏃跺彂鐢熲€?- representation / organization / evolution 璐熻矗鈥滃彂鐢熷悗鎬庝箞鍋氣€?- 涓嶆妸澶ч噺璁烘枃鐨勭粍缁囬€昏緫璇杩?trigger 灞?
## 缁撹

浠?40 绡囪鏂囩殑鍘熸枃绾ч€愮瘒绛涚湅锛宍MemPrimitive` 娌℃湁蹇呰闀挎湡缁存姢涓€濂楅珮搴﹀睍寮€鐨?trigger 瀛愯瑷€銆?
鏇村ソ鐨勫仛娉曟槸锛?
- 鐢ㄥ皯鏁扮ǔ瀹?trigger 鍘熷瀷瑕嗙洊澶у鏁拌鏂?- 鎶婂鏉傛€ч泦涓湪 memory 鍗曞厓褰㈡垚銆佺粍缁囥€佹紨鍖栧拰妫€绱?- 鍙鏋佸皯鏁?trigger-heavy 绯荤粺淇濈暀鏇存槑纭殑 failure / capacity / routing / subgoal / offline 璇箟

涔熷氨鏄锛宼rigger 渚濈劧閲嶈锛屼絾鏇村儚鈥滆杽鎺у埗灞傗€濓紝鑰屼笉鏄暣涓?memory ontology 閲屾渶閲嶇殑涓绘垬鍦恒€?