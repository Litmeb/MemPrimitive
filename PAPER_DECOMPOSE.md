> Note
> This file is a paper-to-primitive analysis note.
> Richer trigger names mentioned here are literature-mapping labels, not the current public baseline trigger API.

## HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

璁烘枃閾炬帴: <https://arxiv.org/abs/2405.14831>

瀹樻柟 repo: <https://github.com/OSU-NLP-Group/HippoRAG>

### 璁烘枃渚?memory 鏈哄埗閫熷啓

HippoRAG 鎶婂閮ㄦ枃妗ｈ蹇嗗缓妯℃垚涓€涓彈 hippocampal indexing theory 鍚彂鐨勯暱鏈熻蹇嗙郴缁熴€傚叾 memory 鏍稿績涓嶆槸鈥滄妸 passage 鍚戦噺鍖栧悗鐩存帴鏌ュ簱鈥濓紝鑰屾槸鍏堟妸 passage 澶勭悊鎴愬紑鏀惧紡鐭ヨ瘑鍥捐氨锛屽啀鍦ㄦ煡璇㈡椂鎶?query 涓殑鍏抽敭瀹炰綋閾炬帴鍒板浘鑺傜偣锛岄殢鍚庣敤 Personalized PageRank 鍦ㄥ浘涓婂仛涓€娆″彈鏌ヨ鍋忕疆鐨勬墿鏁ｏ紝鏈€鍚庢妸鑺傜偣婵€娲诲垎鏁拌仛鍚堝洖 passage 鎺掑簭銆?
浠?MemPrimitive slot 瑙掑害鐪嬶紝瀹冪殑涓婚摼璺彲浠ユ鎷负锛?
- `unit_formation`: 浠?passage 涓哄熀鏈啓鍏ュ崟鍏冦€?- `representation`: 瀵规瘡涓?passage 鍏堝仛 named entity extraction锛屽啀鍋?NER-conditioned OpenIE triple extraction锛屽苟涓?noun phrase / entity 鑺傜偣寤虹珛 embedding銆?- `write_trigger`: 榛樿绂荤嚎鍏ㄩ噺鍐欏叆銆?- `organization`: 鎶?passage銆乶oun phrase / entity銆乼riple 鍏崇郴缁勭粐鎴愬彲妫€绱㈢殑鍥惧紡绱㈠紩锛屽苟淇濈暀 node-to-passage 褰掑睘鍏崇郴銆?- `evolution_trigger`: 璁烘枃涓讳綋娌℃湁鍗曠嫭寮鸿皟鍦ㄧ嚎婕斿寲瑙﹀彂锛涚储寮曟瀯寤洪樁娈电殑鍥惧寮烘洿鍍忓啓鍏ユ椂/鍐欏悗澶勭悊銆?- `memory_evolution`: 澧炶ˉ similarity/synonymy edges锛屼娇鐩歌繎浣嗕笉瀹屽叏鐩稿悓鐨?noun phrases 浜掕仈銆?- `retrieval`: query 鍏堟娊 named entities锛屽啀閾炬帴鍒?KG 鑺傜偣锛屽仛 node specificity 鍔犳潈鍜?PPR 鎵╂暎锛屾渶鍚庢妸鑺傜偣鍒嗘暟鑱氬悎鎴?passage 鍒嗘暟銆?- `readout`: 杈撳嚭 top-k passages 缁欎笅娓?reader/QA銆?
### 鎸?MemPrimitive slot 鐨勬媶瑙?
#### unit_formation

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍩烘湰浠?retrieval corpus 涓殑鍗曚釜 passage 涓虹储寮曞崟浣嶏紝鍥捐氨鏋勫缓鏄€?passage 鎵ц OpenIE銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `PassThroughUnitFormation`锛氬鏋滀笂娓稿凡缁忔妸杈撳叆鍑嗗涓哄崟 passage observation锛屽彲鐩存帴琛ㄨ揪鈥滀竴鏉¤緭鍏ュ搴斾竴涓啓鍏ュ崟鍏冣€濄€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SentenceSplitUnitFormation`銆乣LineSplitUnitFormation`銆乣WindowedUnitFormation`锛氬畠浠兘鍋氬垏鍒嗭紝浣?HippoRAG 璁烘枃鏍稿績骞朵笉渚濊禆杩欎簺鍒囧垎绛栫暐銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€傝 slot 涓嶆槸 HippoRAG 鐨勬満鍒剁摱棰堛€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歰ffline indexing 鏄 retrieval corpus 閫?passage 澶勭悊骞舵娊鍙?triples銆?  - repo 瀹炵幇鍙‘璁わ細`HippoRAG.index(docs)` 鎺ユ敹 `List[str]` 鏂囨。/娈佃惤骞堕€愭潯鍋?OpenIE銆?  - 鎺ㄦ柇锛氬湪 MemPrimitive 涓妸姣忎釜 passage 瑙嗕负涓€涓?`Observation`/`MemoryUnit` 宸茶冻澶熸壙杞借 slot銆?
#### representation

- 璁烘枃閲屽仛浜嗕粈涔?  - 瀵规瘡涓?passage 鍏堟娊 named entities锛屽啀鎶?named entities 鏀惧洖 prompt 閲屽仛 NER-conditioned triple extraction銆?  - 鍚屾椂涓?entity / noun phrase 寤?embedding锛岀敤浜庡悗缁?query linking 鍜?synonymy edge 鏋勯€犮€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆傚綋鍓嶆病鏈変竴涓幇鏈?representation 鑳戒竴娆℃€т骇鍑衡€渜uery-independent 鐨?passage OpenIE 鍥捐妭鐐?+ 鍥捐竟 + 涓撶敤浜庤妭鐐归摼鎺ョ殑 embedding 杞戒綋鈥濄€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BasicRepresentation(elements=("text", "embedding", "entities", "triple"))`
    - 鑳借鐩栤€滄娊瀹炰綋銆佹娊涓夊厓缁勩€佸仛 embedding鈥濊繖涓€澶栬锛屼絾褰撳墠璇箟鏇村儚鎶婄粨鏋滈檮鐫€鍦ㄥ崟涓?`MemoryUnit` 涓婏紝涓嶇瓑浜?HippoRAG 鎵€闇€鐨?NER-conditioned OpenIE 绱㈠紩琛ㄧず銆?  - `KeywordRepresentation`
    - 鍙兘瑕嗙洊 lexical side information锛屼笌璁烘枃鏍稿績琛ㄧず涓嶈冻澶熷尮閰嶃€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥滃厛 NER銆佸啀鐢?NER 绾︽潫 triple extraction鈥濈殑涓ら樁娈佃〃绀烘ā鍧椼€?  - 缂哄皯鎶?noun phrase / entity 浣滀负鍥捐妭鐐圭骇瀵硅薄绋冲畾杈撳嚭鐨勮〃绀鸿竟鐣屻€?  - 缂哄皯鈥減assage-node incidence 淇℃伅鈥濆湪琛ㄧず闃舵鐨勬槑纭骇鐗┿€?  - 缂哄皯 query 渚?named entity extraction 鐨勫绉版満鍒讹紱褰撳墠 recall 璺緞娌℃湁鍗曠嫭鐨?query representation slot锛屽彧鑳借惤鍒?retrieval 鍐呴儴瀹炵幇銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氬厛 extract named entities锛屽啀鎶?named entities 鍔犲叆 OpenIE prompt 浠ユ娊 triples銆?  - repo 瀹炵幇鍙‘璁わ細`prompts/templates/ner.py` 涓?`prompts/templates/triple_extraction.py` 鏄剧ず纭湁 NER 涓?NER-conditioned triple extraction 涓ら樁娈?prompt锛沗openie_vllm_offline.py` 涔熸寜璇ラ『搴忔壒閲忔墽琛屻€?  - 鎺ㄦ柇锛歁emPrimitive 褰撳墠 `BasicRepresentation` 铏界劧鏈?`entities` 鍜?`triple`锛屼絾娌℃湁绋冲畾鏆撮湶鈥滃浘鑺傜偣瀵硅薄鍖?+ 鍚庣画閾炬帴涓撶敤琛ㄧず鈥濈殑鑳藉姏杈圭晫锛屽洜姝ゅ彧鑳介儴鍒嗗鐢ㄣ€?
#### write_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 璁烘枃涓讳綋鎶?indexing 瑙嗕负绂荤嚎鍏ㄩ噺鏋勫缓杩囩▼锛屾病鏈夊鏉傜殑 selective write 鏈哄埗锛涙柊 passage 閫氬父鐩存帴绾冲叆绱㈠紩銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `AlwaysTrigger`锛氬彲鐩存帴琛ㄨ揪鈥滄墍鏈?passage 鍧囪繘鍏ョ储寮曗€濄€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`锛氭妧鏈笂鑳界敤锛屼絾涓嶆槸璁烘枃鏍稿績鏈哄埗銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€侶ippoRAG 鐨勫垱鏂扮偣涓嶅湪 write gating銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歰ffline indexing 澶勭悊鏁翠釜 retrieval corpus銆?  - repo 瀹炵幇鍙‘璁わ細`index(docs)` 瀵硅緭鍏?docs 鎵归噺鎵ц OpenIE 鍜屽浘鏋勫缓锛屾病鏈夊崟鐙殑 selective write 鍒ゅ畾闃舵銆?  - 鎺ㄦ柇锛氳 slot 鍙洿鎺ョ敤鏈€绠€鍗曠殑鍏ㄥ啓鍏ヨ〃杈俱€?
#### organization

- 璁烘枃閲屽仛浜嗕粈涔?  - 缁勭粐鎴愪竴涓紑鏀惧紡 KG / hippocampal index銆?  - 鍥鹃噷鑷冲皯鍖呭惈 noun phrase / entity 灞傞潰鐨勮妭鐐广€乼riple 璇卞鐨勫叧绯昏竟锛屼互鍙?node-to-passage 鐨勫綊灞炵粺璁★紱璁烘枃杩樻樉寮忔彁鍒颁竴涓褰?noun phrase 鍦ㄥ悇 passage 涓嚭鐜版鏁扮殑鐭╅樀锛岀敤浜庢妸鑺傜偣婵€娲昏仛鍚堝洖 passage銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphAppendOrganization`
    - 鑳芥妸璁板綍鏀捐繘 graph layer锛屽苟鍐欎竴浜?graph metadata銆?    - 浣嗗畠鐨勨€滃浘鈥濅粛鐒舵槸 record-centric metadata graph锛屼笉鏄?HippoRAG 閭ｇ鏄惧紡 node/edge/index 缁撴瀯銆?  - `GraphAppendLinkReadyOrganization`
    - 鑳藉噯澶囧浘灞?+ link-ready metadata锛屼絾璁捐鐩爣鏄?note graph/A-MEM 椋庢牸锛屼笉鏄?passage-entity-triple 寮傛瀯绱㈠紩銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯寮傛瀯鍥剧粍缁囪兘鍔涳細闇€瑕佸悓鏃惰〃绀?passage 鑺傜偣銆乪ntity/noun phrase 鑺傜偣銆乺elation/fact 杈癸紝涓旇繖浜涘璞′笉鏄悓涓€绉嶆櫘閫?`MemoryRecord` 灏辫兘鑷劧琛ㄨ揪銆?  - 缂哄皯 node-to-passage incidence matrix 鎴栫瓑浠风粺璁＄粨鏋勩€?  - 缂哄皯鈥滃浘鑺傜偣绾?embedding store鈥濅笌 record store 鐨勬槑纭垎绂汇€?  - 缂哄皯 synonymy edge 灏嗗姞鍏ヤ綍澶勩€佸浣曚笌鍘熷 triple edges 骞跺瓨鐨勬寮忕粍缁囪竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欻ippoRAG builds an open KG and defines a matrix containing the number of times each noun phrase appears in each original passage銆?  - repo 瀹炵幇鍙‘璁わ細`EmbeddingStore` 鍒嗗埆缁存姢 chunk/entity/fact 涓夊瀛樺偍锛沗HippoRAG.index()` 鏄庣‘鏋勫缓鍥惧苟鍗曠嫭缁存姢瀹炰綋銆佷簨瀹炪€乸assage 鐨?embedding 涓庡浘杩炴帴銆?  - 鎺ㄦ柇锛歁emPrimitive 鐜版湁 graph organization 鍙琛ㄨ揪鈥滆褰曚箣闂存湁鍥鹃摼鎺モ€濓紝涓嶈冻浠ョ洿鎺ヨ〃杈?HippoRAG 鐨勫紓鏋勫浘绱㈠紩銆?
#### evolution_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 璁烘枃娌℃湁鎶娾€滀綍鏃跺仛棰濆婕斿寲鈥濆崟鐙綋鎴愪竴涓湪绾挎帶鍒堕棶棰樻潵璁诧紱鍚屼箟/鐩镐技鑺傜偣杩炶竟鏇村儚绱㈠紩鏋勫缓娴佹按绾夸腑鐨勫浐瀹氭楠ゃ€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `NeverTrigger`锛氬鏋滄妸鍥惧寮鸿涓?organization 鍐呴儴瀹屾垚锛屽垯鍙互鐩存帴涓嶄娇鐢ㄩ澶?evolution trigger銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `NeighborExistsEvolutionTrigger`
    - 璇箟涓婂儚鈥滃凡鏈夊浘閭诲眳鏃跺啀鍋氶澶栧浘缁存姢鈥濓紝浣?HippoRAG 鐨?synonymy edge augmentation 涓嶆槸鐢辩幇鎴愰偦灞呭瓨鍦ㄨЕ鍙戯紝鑰屾槸鐢?embedding similarity 瑙勫垯瑙﹀彂銆?  - `ThresholdTrigger`
    - 鍙兘鍏呭綋鍗犱綅鎺у埗锛屼笉鏄鏂囪涔夈€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鑻ュ潥鎸佹妸 synonymy augmentation 鏀惧叆 `memory_evolution` slot锛屽垯缂哄皯涓€涓€滄柊 passage / 鏂拌妭鐐瑰啓鍏ュ悗锛屽鍊欓€?entity 鑺傜偣鍋?similarity linking鈥濈殑涓撶敤瑙﹀彂鍣ㄣ€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歴ynonymy relations 鏄?indexing process 鐨勭粍鎴愰儴鍒嗐€?  - repo 瀹炵幇鍙‘璁わ細`index()` 鍦ㄥ浘鏋勫缓鍚庤皟鐢?`add_synonymy_edges()`锛屽苟闈炲崟鐙殑鍦ㄧ嚎妫€绱㈡湡瑙﹀彂銆?  - 鎺ㄦ柇锛氳 slot 鍦?HippoRAG 涓緝寮憋紝浣嗚嫢瑕佸湪 MemPrimitive 鍐呭共鍑€钀戒綅锛屾渶濂借ˉ涓€涓?graph augmentation trigger銆?
#### memory_evolution

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍦ㄥ熀纭€ OpenIE 鍥句箣涓婏紝鐢?retrieval encoder 涓虹浉浼间絾涓嶅畬鍏ㄧ浉鍚岀殑 noun phrases/entity 澧炶ˉ synonymy/similarity edges锛屽府鍔?pattern completion銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphLinkEvolution`
    - 鑳藉湪 graph layer 閲岃ˉ link锛屼絾褰撳墠鍋囪鏄?record-to-record 閭绘帴锛屼笉鏄?entity-node embedding 鐩镐技椹卞姩鐨勮妭鐐圭骇 augmentation銆?  - `GraphNeighborAppendEvolution`
    - 鍙槸 `GraphLinkEvolution` 鐨勫吋瀹瑰寘瑁咃紝鑳藉姏杈圭晫鐩稿悓銆?  - `LinkStrengtheningEvolution`
    - 鑳藉仛鍥鹃摼鎺ュ己鍖栵紝浣嗚璁″墠鎻愭槸 note graph + LLM judge锛屼笉鏄?HippoRAG 鐨?retrieval-encoder synonymy edge銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥滃熀浜庤妭鐐?embedding 鐩镐技搴﹂槇鍊兼壒閲忓缓绔?synonymy edges鈥濈殑鏄惧紡婕斿寲妯″潡銆?  - 缂哄皯瀵瑰紓鏋勫浘鑺傜偣绾у埆鑰岄潪 record 绾у埆鐨勮竟鍐欏洖銆?  - 缂哄皯鎶?node specificity 鎵€闇€鐨?local support statistics 浣滀负鍥炬紨鍖栧壇浜х墿鎸佷箙鍖栥€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氶澶栬竟鏉ヨ嚜 retrieval encoders锛屽湪 entity representations cosine similarity 瓒呰繃闃堝€兼椂鍔犲叆銆?  - repo 瀹炵幇鍙‘璁わ細`index()` 鍦?`add_fact_edges` / `add_passage_edges` 涔嬪悗璋冪敤 `add_synonymy_edges()`銆?  - 鎺ㄦ柇锛氬綋鍓?MemPrimitive 鍥炬紨鍖栧鏃忔洿鎺ヨ繎 record graph 鎴?note graph锛屼笉瓒充互鏃犳柊澧炴ā鍧楄〃杈?HippoRAG 鐨?synonymy augmentation銆?
#### retrieval

- 璁烘枃閲屽仛浜嗕粈涔?  - query 鍏堟娊 named entities銆?  - 鎶?query named entities 閾炬帴鍒?KG 鑺傜偣锛屽舰鎴?query nodes銆?  - 鐢?query nodes 浣滀负 Personalized PageRank 绉嶅瓙锛屽苟鍋?node specificity 鍔犳潈銆?  - 灏?PPR 鍚庣殑鑺傜偣姒傜巼鑱氬悎鍒?passage 鍒嗘暟锛岃緭鍑?top passages銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphSeedAndExpandRetrieval`
    - 瑕嗙洊浜嗏€滀粠 query 鎵?graph seed 鍐嶅仛涓€璺虫墿鏁ｂ€濈殑绮楄疆寤擄紝浣嗗綋鍓嶆槸 token/entity overlap + one-hop expansion锛屼笉鏄?query entity linking + PPR銆?  - `VectorGraphSeedAndExpandRetrieval`
    - 瑕嗙洊浜嗏€渧ector seed + graph expansion鈥濈殑绮楄疆寤擄紝浣嗘湇鍔″璞℃槸 enriched note graph锛屼笉鏄?HippoRAG 鐨?noun phrase/entity KG銆?  - `EmbeddingSimilarityRetrieval`
    - 鍙鐢ㄥ叾涓殑 embedding 鐩镐技搴﹁绠楃洿瑙夛紝浣嗗畠鍙仛鍚戦噺妫€绱紝涓嶅仛鍥句笂浼犳挱鍜?passage 鑱氬悎銆?  - `LayerAwareRetrieval`
    - 鍙槸鍦ㄥ眰闂村垎鍙戠粨鏋滐紝涓嶈Е鍙?HippoRAG 鐨勬牳蹇冩绱㈢畻娉曘€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 query named entity extraction + entity-to-node linking 鐨?retrieval 鍓嶅崐娈点€?  - 缂哄皯 Personalized PageRank 妫€绱㈠櫒銆?  - 缂哄皯 node specificity / local support statistics 鍔犳潈銆?  - 缂哄皯 node-to-passage incidence 鑱氬悎锛屾妸鑺傜偣婵€娲诲垎鏁版槧灏勫洖 passage 鎺掑簭銆?  - 缂哄皯涓庡師璁烘枃涓€鑷寸殑鈥滃崟姝ュ璺斥€濆浘鎵╂暎妫€绱㈣涔夈€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歲uery named entities -> query nodes -> PPR -> multiply by passage incidence matrix 寰?passage ranking銆?  - repo 瀹炵幇鍙‘璁わ細`HippoRAG.retrieve()` 涓?`run_ppr()` 琛ㄦ槑 repo 纭疄鏈?PPR 妫€绱富骞诧紱鍚屾椂褰撳墠涓诲垎鏀繕寮曞叆浜?fact scoring銆丏SPy rerank銆乸assage node weight 绛夋洿鍋?HippoRAG 2 鐨勬帶鍒堕€昏緫銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍗充娇涓嶇収鎼?repo 鐨勬柊缁嗚妭锛孧emPrimitive 浠嶇己灏?original HippoRAG 鎵€闇€鐨?PPR retrieval primitive銆?
#### readout

- 璁烘枃閲屽仛浜嗕粈涔?  - 鎶?top-ranked passages 杩斿洖缁?reader/QA 妯″潡锛況eadout 鏈韩涓嶆槸璁烘枃鍒涙柊鐐广€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
  - 杩欎簺閮借冻浠ユ妸妫€绱㈠嚭鐨?passages 绾挎€ф暣鐞嗙粰涓嬫父銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphReadout`
    - 閫傚悎璋冭瘯鍥剧粨鏋勶紝浣嗕笉鏄?HippoRAG 璁烘枃榛樿鐨?passage readout銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€傚彧瑕?retrieval 宸茬粡浜у嚭姝ｇ‘鐨?passage ranking锛宺eadout 鐢ㄧ幇鏈夋ā鍧楀嵆鍙€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歅PR 鏈€缁堢敤浜?rank passages for retrieval銆?  - repo 瀹炵幇鍙‘璁わ細`retrieve()` 杩斿洖 top_k docs 鏂囨湰鍒楄〃銆?  - 鎺ㄦ柇锛欻ippoRAG 鐨?readout 鍙互鐢?MemPrimitive 鐜版湁閫氱敤鏂囨湰 readout 鐩存帴鎵挎媴銆?
### 涓?MemPrimitive 鐜版湁缁勪欢鐨勫鐓х粨璁?
| slot | 缁撹 | 璇存槑 |
| --- | --- | --- |
| `unit_formation` | 鐩存帴澶嶇敤 | passage 绾у啓鍏ュ崟鍏冨彲鐢?`PassThroughUnitFormation` 鐩存帴琛ㄨ揪 |
| `representation` | 閮ㄥ垎澶嶇敤 | 鏈?entities/triples/embedding锛屼絾缂?NER-conditioned OpenIE 涓庤妭鐐圭骇琛ㄧず杈圭晫 |
| `write_trigger` | 鐩存帴澶嶇敤 | `AlwaysTrigger` 瓒冲 |
| `organization` | 閮ㄥ垎澶嶇敤 | 鏈?graph layer锛屼絾涓嶆槸 HippoRAG 鎵€闇€寮傛瀯鍥剧储寮?|
| `evolution_trigger` | 閮ㄥ垎澶嶇敤 | 璁烘枃寮卞寲璇?slot锛屼絾鑻ュ崟鍒?synonymy augmentation 瑙﹀彂锛岀幇鏈夎Е鍙戝櫒涓嶈创鍚?|
| `memory_evolution` | 閮ㄥ垎澶嶇敤 | 鏈?graph link evolution锛屼絾娌℃湁鑺傜偣 embedding 鐩镐技椹卞姩鐨?synonymy augmentation |
| `retrieval` | 閮ㄥ垎澶嶇敤 | 鏈?seed-and-expand 闆忓舰锛屼絾缂?query linking銆丳PR銆乶ode specificity銆乸assage 鑱氬悎 |
| `readout` | 鐩存帴澶嶇敤 | 閫氱敤 passage readout 鍗冲彲 |

### 閲嶈〃杈惧垽鏂?
鍙兘閮ㄥ垎鏄犲皠銆?
鍘熷洜涓嶆槸 slot 鏁伴噺涓嶅锛岃€屾槸褰撳墠缂虹殑姝ｅソ鏄?HippoRAG 鏈€鏍稿績鐨勫嚑娈垫満鍒讹細

- 寮傛瀯鍥剧储寮曡€岄潪鏅€?record graph
- query 瀹炰綋鎶藉彇涓庤妭鐐归摼鎺?- Personalized PageRank 妫€绱?- 鑺傜偣婵€娲诲埌 passage 鎺掑悕鐨勮仛鍚堢煩闃?缁熻缁撴瀯

濡傛灉鍙敤鐜版湁妯″潡寮鸿鎷艰锛屾渶澶氳兘鍋氬嚭鈥滄湁 triples銆佹湁 graph銆佹湁 seed-expand鈥濈殑杩戜技鐗堬紝浣嗗緢闅炬妸 HippoRAG 鏈€鍏抽敭鐨勫崟姝ュ璺?pattern completion 妫€绱㈣涔夎〃杈惧噯纭€?
### 澶囨敞涓庤瘉鎹竟鐣?
- 璁烘枃鏄庤
  - 绂荤嚎闃舵: passage -> named entities -> NER-conditioned OpenIE triples -> open KG銆?  - 棰濆鐢?retrieval encoders 娣诲姞 synonymy relations銆?  - 鍦ㄧ嚎闃舵: query named entities -> query nodes -> Personalized PageRank -> passage ranking銆?  - node specificity 鐢ㄤ簬璋冭妭 query node 姒傜巼銆?- repo 瀹炵幇鍙‘璁?  - repo 涓诲垎鏀繚鐣欎簡 OpenIE銆佺嫭绔嬬殑 entity/fact/passage embedding stores銆佸浘鏋勫缓涓?`run_ppr()`銆?  - `prompts/templates/ner.py` 涓?`prompts/templates/triple_extraction.py` 璇佸疄浜嗕袱闃舵 NER + triple extraction銆?  - `HippoRAG.retrieve()` 璇佸疄褰撳墠瀹炵幇浠嶄互鍥炬绱㈠拰 PPR 涓洪鏋躲€?- 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂?  - 闇€瑕佸湪 MemPrimitive 涓柊澧炵殑锛屼笉鍙槸涓€涓?`PPRRetrieval`锛岃繕鍖呮嫭涓庝箣閰嶅鐨勫紓鏋勫浘缁勭粐涓?node-to-passage 鑱氬悎缁撴瀯銆?  - query entity extraction 鍦ㄥ綋鍓?slot 浣撶郴涓嬫洿閫傚悎钀藉叆 retrieval module 鍐呴儴锛岃€屼笉鏄己琛屽杩?ingest-side representation銆?- 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
  - 瀹樻柟 repo 褰撳墠涓诲垎鏀凡鏄庢樉甯︽湁 HippoRAG 2 婕斿寲鐥曡抗锛屽寘鍚?fact scoring銆丏SPy rerank銆乸assage node weight 绛夐€昏緫锛涜繖浜涗笉搴旂洿鎺ュ綋鎴?2024 鍘熻鏂囩殑涓ユ牸鏈哄埗銆?  - 璁烘枃娌℃湁鎶婃墍鏈夊伐绋嬫€у浘鏁版嵁缁撴瀯瀹屽叏褰㈠紡鍖栧埌鍙竴涓€鏄犲皠 MemPrimitive contract 鐨勭矑搴︼紝鍥犳鏌愪簺鈥滅煩闃垫槸 organization 浜х墿杩樻槸 retrieval 杈呭姪绱㈠紩鈥濆彧鑳藉仛鍚堢悊钀戒綅锛屼笉鑳藉０绉板敮涓€姝ｇ‘銆?
## AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents

璁烘枃閾炬帴: <https://arxiv.org/abs/2407.04363>

瀹樻柟 repo: <https://github.com/AIRI-Institute/AriGraph>

### 璁烘枃渚?memory 鏈哄埗閫熷啓

AriGraph 涓嶆槸绂荤嚎鏂囨。绱㈠紩鍨?memory锛岃€屾槸闈㈠悜浜や簰寮忕幆澧冪殑 world model memory銆傚叾鏍稿績鏄妸姣忎竴姝?observation 鍚屾椂鍐欒繘涓ゅ鐩镐簰杩炴帴鐨勮蹇嗭細

- semantic memory锛氭妸 observation 涓娊鍙栧嚭鐨?triplet `(object1, relation, object2)` 骞跺叆璇箟鍥俱€?- episodic memory锛氭妸璇ユ observation 鏈韩浣滀负 episodic vertex 淇濆瓨锛屽苟鐢?episodic edge 鎶娾€滆繖涓€姝ュ悓鏃跺嚭鐜扮殑璇箟 triplets鈥濅笌璇?observation 杩炴帴璧锋潵銆?
鍦ㄥ啓鍏ユ椂锛孉riGraph 涓嶅彧鏄?append 鏂?triplets锛岃繕浼氬厛瀹氫綅涓庡綋鍓?observation 鎻愬埌瀵硅薄鐩稿叧鐨勬棦鏈?semantic edges锛岃瘑鍒叾涓凡杩囨椂鐨勪簨瀹炲苟鍒犻櫎锛屽啀鎶婃柊 triplets 鍐欏叆銆? 
鍦ㄦ绱㈡椂锛岃鏂囬噰鐢ㄤ袱闃舵杩囩▼锛?
1. semantic search锛氬厛鎸?query 鎵剧浉鍏?triplets锛屽啀娌?semantic graph 鍋氬彈娣卞害/瀹藉害鎺у埗鐨勬墿灞曘€?2. episodic search锛氬啀鏍规嵁杩欎簺 triplets 鍙嶆煡鐩稿叧 episodic observations锛岃緭鍑烘渶鐩稿叧鐨?past experiences銆?
浠?MemPrimitive 鐨?slot 瑙嗚鐪嬶紝瀹冩洿鍍忥細

- `unit_formation`锛氭瘡姝?observation 浣滀负鍐欏叆鍗曞厓銆?- `representation`锛氭妸 observation 瑙ｆ瀽鎴?semantic triplets銆?- `write_trigger`锛氭瘡姝?observation 榛樿閮藉啓鍏ャ€?- `organization`锛氬悓鏃剁淮鎶?semantic graph 涓?episodic observation linkage銆?- `evolution_trigger`锛氭瘡姝?observation 閮借Е鍙戝宸叉湁 semantic memory 鐨勬洿鏂般€?- `memory_evolution`锛氬垹闄や笌鏂?observation 鍐茬獊鎴栧凡杩囨椂鐨勬棫 triplets銆?- `retrieval`锛氬厛 semantic graph retrieval锛屽啀 episodic memory retrieval銆?- `readout`锛氭妸妫€绱㈠嚭鐨勮涔?facts 涓?episodic observations 鏁寸悊缁欒鍒?鍐崇瓥妯″潡銆?
### 鎸?MemPrimitive slot 鐨勬媶瑙?
#### unit_formation

- 璁烘枃閲屽仛浜嗕粈涔?  - 姣忎釜鏃堕棿姝ヨ緭鍏ユ槸涓€鏉℃柊鐨勭幆澧?observation锛涜 observation 鏃細鎴愪负涓€涓柊鐨?episodic memory entry锛屼篃浼氳Е鍙?semantic triplet 鎶藉彇銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `PassThroughUnitFormation`
    - 濡傛灉鎶婃瘡姝?observation 浣滀负涓€涓?`Observation` 杈撳叆锛屽垯鍙洿鎺ヨ〃杈锯€滀竴姝?observation -> 涓€涓?memory unit鈥濄€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SentenceSplitUnitFormation`
  - `LineSplitUnitFormation`
  - `WindowedUnitFormation`
    - 杩欎簺妯″潡鑳藉仛鍒囧垎锛屼絾 AriGraph 鐨勬牳蹇冨啓鍏ョ矑搴︿笉鏄?observation 鍐呭垎鍙ワ紝鑰屾槸 step-level observation銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€傝 slot 涓嶆槸 AriGraph 鐨勬満鍒剁摱棰堛€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氭瘡涓€姝?agent 鎺ユ敹 observation `o_t`锛屽苟鍩轰簬瀹冩洿鏂?semantic 涓?episodic memory銆?  - repo 瀹炵幇鍙‘璁わ細`pipeline_arigraph.py` 涓瘡涓€姝ラ兘璋冪敤 `graph.update(observation, ...)`銆?  - 鎺ㄦ柇锛氭妸 step-level observation 浣滀负鍗曚釜 `MemoryUnit` 瓒充互鎵胯浇璇?slot銆?
#### representation

- 璁烘枃閲屽仛浜嗕粈涔?  - 浠?observation 涓娊鍙?semantic triplets `(object1, relation, object2)`锛岃繖浜?triplets 浼氭垚涓?semantic memory 鐨勫閲忋€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BasicRepresentation(elements=("text", "triple"))`
    - 澶栬涓婃渶鎺ヨ繎鈥滄妸 observation 鍙樻垚 triples鈥濄€?    - 浣嗗綋鍓?MemPrimitive 涓殑 `triple` 琛ㄧず鑳藉姏骞朵笉绛変环浜?AriGraph 璁烘枃閲岀殑 observation-conditioned world-fact extraction锛岃€屼笖椤圭洰杩涘睍鏂囨。宸叉槑纭妸 triple extraction 浠嶅亸 heuristic 瑙嗕负寰呮敼杩涢」銆?  - `KeywordRepresentation`
    - 鍙兘鎻愪緵 lexical side information锛屼笉瑕嗙洊 AriGraph 鐨勬牳蹇?triplet representation銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯涓€涓潰鍚?observation 鐨勩€佹樉寮忚緭鍑?world-fact triplets 鐨勮〃绀烘ā鍧椼€?  - 缂哄皯鎶?triplets 浣滀负鍚庣画 semantic graph 鍐欏叆濂戠害绋冲畾鏆撮湶鍑烘潵鐨勮〃绀鸿竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欰riGraph continuously learns world model by extracting semantic triplets from textual observations銆?  - repo 瀹炵幇鍙‘璁わ細`prompts/prompts.py` 涓?`prompt_extraction_current` 鏄庣‘瑕佹眰浠?observation 鎶藉彇 `"subject, relation, object"` 搴忓垪锛沗graphs/contriever_graph.py` 鐨?`update()` 璋冪敤璇?prompt 骞惰В鏋?triplets銆?  - 鎺ㄦ柇锛歁emPrimitive 褰撳墠铏界劧鏈?`triple` 鍏冪礌锛屼絾杩樹笉瓒充互瑙嗕负宸茶惤鍦颁簡涓?AriGraph 鍚岃涔夌殑 observation-triplet extraction primitive銆?
#### write_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 姣忎竴姝?observation 閮借Е鍙?memory learning锛涜鏂囨病鏈夎璁￠澶栫殑 selective write gate銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `AlwaysTrigger`
- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`
    - 鍙互閫氳繃甯搁噺闃堝€兼ā鎷熲€滄€绘槸鍐欏叆鈥濓紝浣嗚繖鍙槸缁曞啓锛屼笉鏄渶鐩存帴琛ㄨ揪銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欵very observation triggers learning that updates agent鈥檚 world model銆?  - repo 瀹炵幇鍙‘璁わ細`pipeline_arigraph.py` 鍦ㄦ瘡涓?step 閮芥棤鏉′欢璋冪敤 `graph.update(...)`銆?  - 鎺ㄦ柇锛氳 slot 鍙敱鐜版湁鈥滃叏鍐欏叆鈥濇ā鍧楃洿鎺ヨ〃杈俱€?
#### organization

- 璁烘枃閲屽仛浜嗕粈涔?  - 缁勭粐涓€涓贩鍚?memory graph `G = (V_s, E_s, V_e, E_e)`锛?    - semantic vertices / edges 鎵胯浇 object-level facts锛?    - episodic vertices 瀛?observation 鏂囨湰锛?    - episodic edges 鎶娾€滃悓涓€姝?observation 鎻愬彇鍑虹殑 semantic triplets鈥濅笌璇?observation 杩炴帴璧锋潵銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphAppendOrganization`
    - 鑳芥妸璁板綍鏀惧叆 graph layer锛屽苟缁存姢涓€浜涘浘鍏冩暟鎹€?    - 浣嗗畠琛ㄨ揪鐨勬槸 record-centric graph append锛屼笉鏄?AriGraph 鎵€闇€鐨勨€渟emantic triplets + episodic observation + 璺ㄤ袱绫昏蹇嗙殑杩炴帴鈥濄€?  - `GraphAppendLinkReadyOrganization`
    - 鎻愪緵 graph layer + link-ready 鍐欏叆澹冲瓙锛屼絾瀹冩湇鍔＄殑鏄?note graph/A-MEM 椋庢牸锛屼笉鏄?semantic/episodic 鍙岃蹇嗙粨鏋勩€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鍚屾椂缁勭粐 semantic graph 涓?episodic observation memory 鐨勭粺涓€ organization primitive銆?  - 缂哄皯鈥渙bservation -> 褰撴 triplets鈥濊繖绉嶈法涓ょ被璁板繂瀵硅薄鐨勮繛鎺ョ粨鏋勩€?  - 缂哄皯瀵?episodic edge 杩欑鈥滆繛鎺?observation 涓庝竴缁?semantic edges鈥濈殑姝ｅ紡鎵胯浇杈圭晫銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欰riGraph world model 鐢?semantic vertices/edges 涓?episodic vertices/edges 鍏卞悓缁勬垚銆?  - repo 瀹炵幇鍙‘璁わ細
    - `graphs/parent_graph.py` / `graphs/contriever_graph.py` 缁存姢 semantic triplets 鍥撅紱
    - `graphs/contriever_graph.py` 杩樼淮鎶?`obs_episodic`锛屾妸 observation 涓庡叾瀵瑰簲 triplets/embedding 鍏宠仈淇濆瓨銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細
    - repo 涓?episodic memory 鏇村儚鈥渙bservation -> associated triplets鈥濈殑瀛楀吀锛岃€屼笉鏄鏂囧浘绀洪噷鐨勬樉寮?episodic hyperedge锛屼絾涓よ€呰〃杈剧殑 memory linkage 鏄悓涓€绫绘満鍒躲€?
#### evolution_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 姣忔鏂?observation 鍒版潵鏃讹紝閮戒細瑙﹀彂瀵圭浉鍏虫棦鏈?semantic facts 鐨勬鏌ヤ笌鏇存柊銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0)`
    - 閫氳繃甯搁噺瑙﹀彂鍙互琛ㄨ揪鈥滄瘡娆″啓鍏ュ悗閮芥墽琛屾紨鍖栤€濄€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `NewWriteEvolutionTrigger`
    - 鏈夆€滄柊鍐欏叆鍚庤Е鍙戝眬閮ㄧ淮鎶も€濈殑杞粨锛屼絾瀹冨綋鍓嶉潰鍚?keyed/local-maintenance 瀹舵棌锛屼笉鏄?AriGraph 鐨?observation-conditioned fact revision 璇箟銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲繀椤绘柊澧炵殑 trigger 缂哄彛锛涜 slot 鐨勫叧閿笉鍦?trigger锛岃€屽湪鍚庣画 evolution 鏈綋銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欸iven new observation `o_t`锛岀郴缁熶細鍏堟壘鐩稿叧宸叉湁鐭ヨ瘑锛屽啀绉婚櫎杩囨椂杈瑰苟鎵╁睍 semantic memory銆?  - repo 瀹炵幇鍙‘璁わ細`graph.update(...)` 鍦ㄦ瘡娆?observation 鍒版潵鏃堕兘鎵ц鈥滄娊鍙?-> 鎵剧浉鍏冲瓙鍥?-> 璇嗗埆 outdated -> 鍒犻櫎 -> 鍐欏叆鈥濄€?  - 鎺ㄦ柇锛氳繖涓?slot 鍙互閫氳繃鐜版湁 always-like evolution trigger 缁勫悎琛ㄨ揪銆?
#### memory_evolution

- 璁烘枃閲屽仛浜嗕粈涔?  - 瀵?observation 娑夊強鍒扮殑瀵硅薄锛屽厛鎵惧埌宸叉湁鐩稿叧 semantic edges锛?  - 鍐嶈瘑鍒摢浜涙棫浜嬪疄宸茶繃鏃讹紝涓庡綋鍓?observation 鐨勬柊 triplets 鍐茬獊锛?  - 鍒犻櫎杩欎簺 outdated edges锛岀劧鍚庡啀骞跺叆鏂扮煡璇嗐€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphLinkEvolution`
    - 鑳藉 graph layer 鍋氶澶栧浘鍐欏洖锛屼絾鍋忓悜琛?link锛屼笉鏀寔鈥滄牴鎹柊 observation 鍒犻櫎/鏇挎崲杩囨椂 semantic facts鈥濄€?  - `GraphNeighborAppendEvolution`
    - 鍙槸 `GraphLinkEvolution` 鐨勫吋瀹瑰寘瑁咃紝鑳藉姏杈圭晫鐩稿悓銆?  - `NeighborContextUpdateEvolution`
    - 鑳介噸鍐欓偦灞呬笂涓嬫枃锛屼絾鍏剁洰鏍囨槸 note graph 閭诲眳鏀瑰啓锛屼笉鏄?AriGraph 鐨勪簨瀹炵骇鍐茬獊娑堣В銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥滃熀浜庢柊 observation triplets锛屽鐩稿叧鏃?semantic edges 鍋氬啿绐佹娴嬩笌鍒犻櫎鈥濈殑 memory evolution primitive銆?  - 缂哄皯鍒犻櫎寮?graph evolution锛岃€屼笉浠呮槸 append/link-strengthening銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歰utdated edges in related semantic edges are detected by comparing them with new triplets and removed from the graph銆?  - repo 瀹炵幇鍙‘璁わ細
    - `prompts/prompts.py` 涓?`prompt_refining_items` 涓撻棬瑕佹眰鍒ゆ柇 existing triplets 涓摢浜涘簲琚柊 triplets 鏇挎崲锛?    - `graphs/contriever_graph.py` 鐨?`update()` 浼氳皟鐢?`parse_triplets_removing(...)` 鍜?`delete_triplets(...)`銆?  - 鎺ㄦ柇锛欰riGraph 鐨勫叧閿?evolution 涓嶆槸鈥滃琛ョ浉浼艰竟鈥濓紝鑰屾槸鈥滃眬閮ㄤ簨瀹炰慨璁⑩€濓紱杩欏湪 MemPrimitive 褰撳墠妯″潡闆嗛噷鏄槑纭己鍙ｃ€?
#### retrieval

- 璁烘枃閲屽仛浜嗕粈涔?  - 妫€绱㈠垎涓ゆ锛?    - semantic search锛氬厛鎸?query 鎵炬渶鐩稿叧 triplets锛屽啀娌?semantic graph 閫掑綊鎵╁睍锛?    - episodic search锛氬啀鎶婅繖浜?triplets 鍥炶繛鍒?past episodic observations锛岀粰 observation-level experiences 鎵撳垎骞惰繑鍥?top-k銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphSeedAndExpandRetrieval`
    - 瑕嗙洊浜嗏€滃厛鎵?seed锛屽啀娌垮浘鎵╁睍鈥濈殑绮楄疆寤撱€?    - 浣嗗畠褰撳墠浠嶆槸 MemPrimitive baseline graph 鐨勪竴璺?seed-expand锛屼笉鍖呭惈 AriGraph 鐨勨€渟emantic triplet retrieval -> episodic retrieval鈥濅袱娈靛紡杈撳嚭璇箟銆?  - `EmbeddingSimilarityRetrieval`
    - 鍙被姣?AriGraph semantic search 涓殑 embedding-based relevance锛屼絾瀹冨彧杩斿洖鐩镐技璁板綍锛屼笉浼氱户缁浘鎵╁睍锛屼篃涓嶄細鍥炶繛 episodic memory銆?  - `GraphNeighborRetrieval`
    - 鍙€傚悎宸茬煡 seed id 鐨勯偦灞呮绱紝涓嶉€傚悎 AriGraph 杩欑浠?query 鏂囨湰鍑哄彂鐨?triplet-level semantic retrieval銆?  - `LayerAwareRetrieval`
    - 鍙兘鍋氬眰闂村垎鍙戯紝涓嶈兘琛ㄨ揪 AriGraph 鐨?semantic/episodic 鑱斿姩妫€绱㈡祦绋嬨€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥渟emantic graph retrieval + episodic memory retrieval鈥濅竴浣撳寲妫€绱?primitive銆?  - 缂哄皯鍩轰簬 triplet 鍖归厤缁撴灉缁?episodic observations 鎵撳垎骞惰繑鍥炵殑鏈哄埗銆?  - 缂哄皯瀵光€渜uery set / 澶氫釜鏌ヨ瀹炰綋鈥濊緭鍏ョ殑鍘熺敓鏀寔杈圭晫銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欰lgorithm 1 鍏堝仛 `SemanticSearch(q, V_s, E_s, d, w)`锛屽啀鍋?`EpisodicSearch(E_s^Q, V_e, E_e, k)`銆?  - repo 瀹炵幇鍙‘璁わ細
    - `utils/retriever_search_drafts.py` 鐨?`graph_retr_search(...)` 閫氳繃 embedding search over triplet strings + BFS 鎵╁睍杩斿洖 associated subgraph锛?    - `utils/utils.py` 鐨?`find_top_episodic_emb(...)` 鏍规嵁 retrieved triplets 涓?observation 鍏宠仈 triplets 鐨勯噸鍚堝害锛屽苟缁撳悎 observation/plan embedding similarity锛岃繑鍥?top episodic memories锛?    - `agents/parent_agent.py` 杩樹細鍏堜粠 observation/plan 涓彁鍙栤€渃rucial items鈥濅綔涓?retrieval queries銆?  - 澶囨敞
    - repo 鐨?episodic scoring 姣旇鏂囨洿宸ョ▼鍖栵紝闄や簡 triplet overlap 杩樻贩鍏ヤ簡 observation-plan embedding similarity锛涜繖搴旇涓?repo 缁嗗寲锛岃€屼笉鏄鏂囦富鏈哄埗鏈韩銆?
#### readout

- 璁烘枃閲屽仛浜嗕粈涔?  - 鎶?relevant semantic memories 涓?relevant episodic memories 鏀惧叆 working memory锛屼緵 planning 鍜?decision making 浣跨敤銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
    - 杩欎簺妯″潡閮藉彲浠ユ壙鎷呪€滄妸妫€绱㈠嚭鐨勬枃鏈寲璁板繂鎷兼帴缁欎笅娓糕€濈殑鑱岃矗銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphReadout`
    - 閫傚悎璋冭瘯鍥剧粨鏋勶紝浣?AriGraph 榛樿 readout 闈㈠悜 working memory 鐨勬枃鏈笂涓嬫枃锛屼笉鏄浘璋冭瘯杈撳嚭銆?  - `PromptContextReadout`
    - 鍏峰鈥滀负涓嬫父 prompt 缁勪笂涓嬫枃鈥濈殑鏂瑰悜锛屼絾褰撳墠鏄?Reflexion 椋庢牸涓婁笅鏂囨ā鏉匡紝涓嶆槸 AriGraph 鐨?semantic+episodic 缁勫悎鏍煎紡銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鑻ュ彧瑕佹眰鈥滄妸妫€绱㈢粨鏋滀氦缁欎笅娓糕€濓紝鏃犲叧閿己鍙ｃ€?  - 鑻ヨ姹傛洿璐磋繎璁烘枃涓殑 working-memory 缁勭粐鏍煎紡锛屽垯缂哄皯涓€涓兘鏄惧紡鍖哄垎 semantic memories 涓?episodic memories 鐨?readout 妯℃澘銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歸orking memory 鍖呭惈 current observation銆乺ecent history銆乺elevant semantic memories銆乺elevant episodic memories銆乬oal 涓?current plan銆?  - repo 瀹炵幇鍙‘璁わ細`pipeline_arigraph.py` 鍦?planning / action prompt 涓垎鍒敞鍏?`subgraph` 鍜?`top_episodic`銆?  - 鎺ㄦ柇锛歳eadout 涓嶆槸 AriGraph 鐨勪富瑕佹満鍒剁摱棰堬紝涓昏缂哄彛浠嶅湪 retrieval 涔嬪墠銆?
### 涓?MemPrimitive 鐜版湁缁勪欢鐨勫鐓х粨璁?
| slot | 缁撹 | 璇存槑 |
| --- | --- | --- |
| `unit_formation` | 鐩存帴澶嶇敤 | `PassThroughUnitFormation` 瓒充互琛ㄨ揪 step-level observation 鍐欏叆 |
| `representation` | 閮ㄥ垎澶嶇敤 | 鏈?`triple` 琛ㄧず澹冲瓙锛屼絾缂?observation-level銆侀潪 heuristic 鐨?triplet extraction primitive |
| `write_trigger` | 鐩存帴澶嶇敤 | `AlwaysTrigger` 瓒冲 |
| `organization` | 閮ㄥ垎澶嶇敤 | 鏈?graph append锛屼絾缂?semantic memory + episodic memory 鑱斿悎缁勭粐 |
| `evolution_trigger` | 鐩存帴澶嶇敤 | 鍙敤鐜版湁 always-like trigger 缁勫悎琛ㄨ揪鈥滄瘡姝ラ兘婕斿寲鈥?|
| `memory_evolution` | 閮ㄥ垎澶嶇敤 | 鏈?graph evolution 澶栧３锛屼絾缂?AriGraph 鏍稿績鐨?outdated fact pruning |
| `retrieval` | 閮ㄥ垎澶嶇敤 | 鏈?graph seed-expand 闆忓舰锛屼絾缂?semantic retrieval 涓?episodic retrieval 鐨勪袱娈佃仈鍔?|
| `readout` | 鐩存帴澶嶇敤 | 閫氱敤鏂囨湰 readout 鍙壙杞藉熀纭€ working-memory 娉ㄥ叆 |

### 閲嶈〃杈惧垽鏂?
鍙兘閮ㄥ垎鏄犲皠銆?
涓昏鍘熷洜涓嶆槸 slot 浣撶郴涓嶉€傞厤锛岃€屾槸褰撳墠缂哄け鐨勬濂芥槸 AriGraph 鐨勪笁绫绘牳蹇冩満鍒讹細

- semantic graph 涓?episodic observation memory 鐨勮仈鍚堢粍缁囩粨鏋?- 鍩轰簬鏂?observation 鐨?outdated fact pruning
- semantic retrieval 涔嬪悗鍥炶繛 episodic memory 鐨勪袱娈靛紡妫€绱?
濡傛灉鍙敤鐜版湁妯″潡寮鸿鎷艰锛屽彲浠ュ緱鍒颁竴涓€滄湁 triplets銆佹湁 graph銆佽兘鍋氫竴浜?graph retrieval鈥濈殑杩戜技鐗堬紝浣嗚繕涓嶈冻浠ュ繝瀹為噸琛ㄨ揪 AriGraph 浣滀负 world-model memory 鐨勫叧閿棴鐜€?
### 澶囨敞涓庤瘉鎹竟鐣?
- 璁烘枃鏄庤
  - AriGraph 鐢?semantic vertices/edges 涓?episodic vertices/edges 鍏卞悓缁勬垚銆?  - 鏂?observation 鍒版潵鍚庯紝浼氭彁鍙?triplets銆佽瘑鍒浉鍏虫棫鐭ヨ瘑銆佺Щ闄よ繃鏃惰竟锛屽啀鎵╁睍 semantic memory銆?  - 妫€绱㈠垎涓?semantic search 涓?episodic search 涓ゆ銆?  - episodic relevance 鐢卞尮閰?triplet 鏁伴噺涓?observation 淇℃伅閲忓叡鍚屽喅瀹氥€?- repo 瀹炵幇鍙‘璁?  - `graphs/contriever_graph.py` 鐨?`update()` 瀹炵幇浜?triplet 鎶藉彇銆佺浉鍏冲瓙鍥炬绱€乷utdated triplet 鍒犻櫎涓庢柊 triplet 鍐欏叆銆?  - `utils/retriever_search_drafts.py` 瀹炵幇浜嗗熀浜?embedding search + BFS 鐨?semantic subgraph retrieval銆?  - `utils/utils.py` 鐨?`find_top_episodic_emb(...)` 瀹炵幇浜?episodic memory 鎺掑簭銆?  - `pipeline_arigraph.py` 灏?retrieved semantic memories 涓?episodic memories 涓€璧锋敞鍏?planning / action prompt銆?- 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂?  - 鍦?MemPrimitive 涓紝AriGraph 鏈€鑷劧鐨勭己澶辨ā鍧椾笉鏄崟涓€鐨?graph retriever锛岃€屾槸鈥渟emantic/episodic graph organization + outdated fact pruning + two-stage retrieval鈥濊繖涓€鏁村 primitive 杈圭晫銆?  - `readout` 鍙互缁х画澶嶇敤閫氱敤鏂囨湰 readout锛屼笉闇€瑕佷负 AriGraph 鍗曠嫭鏂板閲嶅瀷 readout family銆?- 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
  - 璁烘枃涓殑 episodic edge 鏇村儚鏄惧紡鍥剧粨鏋勶紱repo 褰撳墠鏇村亸宸ョ▼瀹炵幇锛屼娇鐢?observation 鍒?triplet 鍒楄〃/embedding 鐨勫叧鑱斿瓨鍌ㄣ€備袱鑰呰涔夋帴杩戯紝浣嗘暟鎹粨鏋勫苟涓嶄竴涓€鍚屾瀯銆?  - repo 鐨?episodic scoring 娣峰叆浜?observation/plan embedding similarity锛岃繖姣旇鏂囦富鏂囦腑鐨勫叕寮忔洿寮猴紱涓嶈兘鎶婅繖閮ㄥ垎鐩存帴褰撴垚璁烘枃鏄庣‘涓诲紶銆?
## HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model

璁烘枃閾炬帴: <https://aclanthology.org/2025.acl-long.1575/>

瀹樻柟 repo: <https://github.com/HiAgent2024/HiAgent>

### 璁烘枃渚?memory 鏈哄埗閫熷啓

HiAgent 鍏虫敞鐨勪笉鏄法 trial 闀挎湡璁板繂锛岃€屾槸鍗曟浠诲姟灏濊瘯涓殑 working memory 绠＄悊銆傚畠鎶?subgoal 浣滀负 working-memory chunk锛氬綋鍓?subgoal 淇濈暀瀹屾暣鐨?action-observation 缁嗚妭锛屽凡瀹屾垚 subgoal 鍒欒鍘嬬缉鎴愪竴涓?summarized observation锛屼笌 subgoal 涓€璧风暀鍦ㄤ笂涓嬫枃閲屻€傝繖鏍凤紝鍘嗗彶杞ㄨ抗涓嶄細琚暣娈靛師鏍峰杩?prompt锛岃€屾槸浠モ€渀(subgoal, summary)` 涓轰富锛屽綋鍓?subgoal 淇濈暀缁嗚妭鈥濈殑灞傜骇缁撴瀯鏆撮湶缁欐ā鍨嬨€?
濡傛灉妯″瀷鍒ゆ柇鏌愪釜鏃?subgoal 鐨勮缁嗚建杩瑰褰撳墠鍐崇瓥浠嶇劧鍏抽敭锛孒iAgent 鍏佽妯″瀷鏄惧紡鐢熸垚 `retrieve(subgoal_id)`锛屾妸瀵瑰簲 subgoal 鐨勫畬鏁?action-observation pairs 鍐嶆媺鍥炰笂涓嬫枃銆傛寜 MemPrimitive slot 鐪嬶紝瀹冩洿鍍忥細

- `unit_formation`: 姣忎竴姝ヤ氦浜掑舰鎴愭柊鐨?working-memory 澧為噺锛?- `representation`: 涓昏淇濇寔鑷劧璇█ action / observation / subgoal 鏂囨湰琛ㄧず锛?- `write_trigger`: 姣忎竴姝ラ粯璁ゅ啓鍏ュ綋鍓?working-memory 杞ㄨ抗锛?- `organization`: 鎸?subgoal 鍒?chunk锛屽尯鍒嗗綋鍓?detailed chunk 涓庡巻鍙?summarized chunks锛?- `evolution_trigger`: 褰撴ā鍨嬪垽鏂綋鍓?subgoal 宸插畬鎴愭椂锛岃Е鍙戝帇缂╁綊妗ｏ紱
- `memory_evolution`: 鎶婂凡瀹屾垚 subgoal 鐨勮缁嗚建杩规€荤粨鎴?summarized observation锛屽苟浠?summary 鍙栦唬榛樿鏆撮湶鐨勬棫缁嗚妭锛?- `retrieval`: 鎸夐渶鎸?subgoal id 鍙栧洖鏌愪釜鏃?chunk 鐨勫畬鏁磋缁嗚建杩癸紱
- `readout`: 鎶娾€滃巻鍙?summary + 褰撳墠缁嗚妭 + 鎸夐渶鎭㈠鐨勬棫缁嗚妭鈥濇嫾鎴愬伐浣滆蹇嗕笂涓嬫枃銆?
### 鎸?MemPrimitive slot 鐨勬媶瑙?
#### unit_formation

- 璁烘枃閲屽仛浜嗕粈涔?  - 姣忎釜鏃堕棿姝ラ兘浼氭妸鏂扮殑浜や簰缁撴灉骞跺叆褰撳墠 trial 鐨?working memory锛況epo 涓疄闄呬繚瀛樼殑鏄寜鏃堕棿杩藉姞鐨?action-observation 瀵癸紝浠ュ強蹇呰鏃舵彃鍏ョ殑 `Subgoal` 鏍囪銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `PassThroughUnitFormation`
    - 濡傛灉鎶婃瘡涓€姝?observation 瑙嗕负涓€娆?ingest锛屾垨鎶?action 涓€骞剁紪鐮佽繘 observation 鏂囨湰/metadata锛屽垯瓒充互鎵胯浇鈥滄瘡姝ヤ骇鐢熶竴涓柊鐨?working-memory 澧為噺鈥濊繖涓€鏈€灏忓啓鍏ュ崟浣嶃€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SentenceSplitUnitFormation`
  - `LineSplitUnitFormation`
  - `WindowedUnitFormation`
    - 瀹冧滑鑳藉仛鍒囧垎锛屼絾 HiAgent 鐨勫叧閿矑搴︿笉鏄彞瀛愭垨绐楀彛锛岃€屾槸浜や簰 step 涓?subgoal chunk銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€侶iAgent 鐨勪富瑕佸垱鏂颁笉鍦?unit formation銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歛gent 鍦ㄦ瘡涓椂闂存缁存姢 working memory锛屽苟鍥寸粫 subgoal 缁勭粐 action-observation history銆?  - repo 瀹炵幇鍙‘璁わ細`agentboard/agents/cme_final.py` 涓?`self.memory` 鎸夋杩藉姞 `[("Action", action), ("Observation", state)]`锛屽苟鍦ㄩ渶瑕佹椂鎻掑叆 `[("Subgoal", subgoal)]`銆?  - 鎺ㄦ柇锛歁emPrimitive 鐜版湁 step-level unit 褰㈡垚鑳藉姏瓒充互鎵胯浇璇?slot銆?
#### representation

- 璁烘枃閲屽仛浜嗕粈涔?  - HiAgent 鐨?memory 琛ㄧず鏍稿績浠嶆槸鑷劧璇█鏂囨湰锛岃€屼笉鏄?embedding / graph / triples锛氬寘鎷?subgoal 鏂囨湰銆乤ction-observation 瀵规枃鏈紝浠ュ強宸插畬鎴?subgoal 鐨?summarized observation銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `BasicRepresentation(elements=("text",))`
    - 瓒充互琛ㄨ揪 HiAgent 瀵?working-memory 鍐呭鐨勬渶鍩烘湰鏂囨湰琛ㄧず銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BasicRepresentation(elements=("text", "summary"))`
    - 琛ㄩ潰涓婃湁 summary 鑳藉姏锛屼絾璇?summary 鏄鍗曚釜 unit 鏂囨湰鐨勮〃绀哄寮猴紝涓嶆槸 HiAgent 杩欑鈥滃涓€涓畬鏁?subgoal trajectory 鍋?subgoal-conditioned summarization鈥濄€?  - `SemanticFieldEnrichmentRepresentation`
    - 鍙负 unit 闄勫姞缁撴瀯鍖?note payload锛屼絾瀹冩湇鍔＄殑鏄?note-like enrichment锛屼笉鏄?HiAgent 鐨?subgoal / summarized observation 璇箟銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲繀椤诲崟鐙柊澧炲埌 `representation` slot 鐨勫叧閿兘鍔涖€侶iAgent 鐨勫叧閿?summary 鏇撮€傚悎钀藉湪 `memory_evolution`锛岃€屼笉鏄?ingest-side representation銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歴ummarized observation `s_i = S(g_i, o_0, a_0, ..., o_t)` 鏄洿缁?subgoal 瀵瑰巻鍙茶建杩圭殑鍘嬬缉琛ㄧず銆?  - repo 瀹炵幇鍙‘璁わ細`cme_final.py` 涓?memory 涓昏鐢?`"Subgoal"`銆乣"Action"`銆乣"Observation"` 杩欑被鏂囨湰鏍囩鍙婂叾瀛楃涓插唴瀹规瀯鎴愩€?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細HiAgent 娌℃湁渚濊禆澶嶆潅鐨勫簳灞傝〃绀哄涔狅紱鍏跺叧閿笉鍦ㄨ〃绀虹绫伙紝鑰屽湪鍚庣画濡備綍鎸?subgoal 绠＄悊杩欎簺鏂囨湰鍧椼€?
#### write_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 榛樿姣忎竴姝ヤ氦浜掗兘浼氬啓鍏ュ綋鍓?working-memory 杞ㄨ抗锛屾病鏈変笓闂ㄧ殑 selective write 鏈哄埗銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `AlwaysTrigger`
- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`
    - 鎶€鏈笂鍙互妯℃嫙鍏ㄥ啓鍏ワ紝浣嗕笉鏄渶鐩存帴琛ㄨ揪銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欻iAgent 鐨勫尯鍒富瑕佸湪 working-memory 鐨勫眰绾х鐞嗭紝鑰屼笉鏄€滃摢浜?step 鍊煎緱鍐欏叆鈥濄€?  - repo 瀹炵幇鍙‘璁わ細`cme_final.py` 鐨?`update()` 鍦ㄦ瘡涓?step 閮芥妸 action-observation 瀵硅拷鍔犺繘 `self.memory`銆?  - 鎺ㄦ柇锛氳 slot 鍙敱鐜版湁鍏ㄥ啓鍏ユā鍧楃洿鎺ヨ〃杈俱€?
#### organization

- 璁烘枃閲屽仛浜嗕粈涔?  - 浠?subgoal 涓?chunk 缁勭粐 working memory锛?    - 褰撳墠 subgoal 淇濈暀瀹屾暣 action-observation pairs锛?    - 宸插畬鎴?subgoal 鍙繚鐣?`(subgoal, summarized observation)`锛?    - 闇€瑕佹椂鍙啀鎶婃煇涓棫 subgoal 鐨勮缁嗚建杩规仮澶嶅埌涓婁笅鏂囦腑銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `AppendOrganization`
    - 鑳介『搴忓啓鍏ヨ褰曪紝浣嗕笉鑳借〃杈锯€滃綋鍓?chunk 淇濈暀缁嗚妭銆佸巻鍙?chunk 榛樿鍙毚闇?summary鈥濈殑灞傜骇 chunk 缁勭粐銆?  - `ConditionalLayerOrganization`
    - 鍙互鎸夎鍒欏垎灞傦紝浣嗙己灏?subgoal chunk 杈圭晫銆乻ubgoal id銆佸綋鍓?鍘嗗彶鐘舵€佸垏鎹㈢瓑鎺у埗閫昏緫銆?  - `PlacementWithoutAppendOrganization`
    - 鑳藉彂 placement锛屼絾涓嶈兘鐙珛瀹屾垚 HiAgent 鐨?chunk 缁勭粐銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鏄惧紡鐨?subgoal-chunk organization primitive銆?  - 缂哄皯鈥滃綋鍓?detailed chunk / 鍘嗗彶 summarized chunk / 鎸夐渶鎭㈠鏃?detailed chunk鈥濈殑缁熶竴缁勭粐杈圭晫銆?  - 缂哄皯绋冲畾鐨?subgoal id 涓?chunk 鐢熷懡鍛ㄦ湡鎵胯浇鏂瑰紡銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歸orking memory 褰㈠紡浠庣洿鎺ヤ繚鐣欏畬鏁村巻鍙诧紝鍙樻垚浠?subgoal 涓?chunk 鐨勫眰绾ц〃绀猴紝杩囧幓 subgoal 鍙繚鐣?summary锛屽綋鍓?subgoal 淇濈暀缁嗚妭銆?  - repo 瀹炵幇鍙‘璁わ細`cme_final.py` 涓?`serialize_history()` 浼氭壂鎻?`Subgoal` 杈圭晫锛屽鏃?subgoal 鏀瑰啓鎴愮紪鍙峰悗鐨?`Subgoal + Observation(summary)`锛岃€屾渶鍚庝竴涓?subgoal 涔嬪悗鐨勮缁嗚建杩逛繚鎸佸睍寮€銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細HiAgent 鐨勬牳蹇冪粍缁囨満鍒朵笉鏄畝鍗?append锛岃€屾槸鈥渃hunked working-memory exposure policy鈥濄€?
#### evolution_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 褰撴ā鍨嬪垽鏂綋鍓?subgoal 宸插畬鎴愭椂锛岃Е鍙戞妸璇?subgoal 鐨勮缁嗚建杩瑰帇缂╂垚 summarized observation锛屽苟杩涘叆涓嬩竴涓?subgoal銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`
    - 鍙互妯℃嫙鈥滄瘡娆￠兘鍋氭紨鍖栤€濓紝浣嗕笉鑳借〃杈?HiAgent 闇€瑕佺殑鈥滃彧鍦?subgoal 瀹屾垚鏃惰Е鍙戝帇缂┾€濄€?  - `OutcomeConditionedEvolutionTrigger`
    - 杞粨涓婃湁鈥滃熀浜庣粨鏋滀俊鍙疯Е鍙戔€濈殑鏂瑰悜锛屼絾瀹冮潰鍚?Reflexion 寮?trial 缁撴灉锛屼笉鏄?HiAgent 鐨?subgoal-completion 鍒ゆ柇銆?  - `NewWriteEvolutionTrigger`
    - 鏈夆€滄柊鍐欏叆鍚庡仛缁存姢鈥濈殑杞粨锛屼絾涓嶆槸 HiAgent 鐨?chunk 鍘嬬缉鏃舵満銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯闈㈠悜鈥渟ubgoal 瀹屾垚鈥濊繖涓€浜嬩欢鐨?evolution trigger銆?  - 缂哄皯鎶?LLM 鐨?subgoal-completion 鍒ゆ柇鏄惧紡杞垚 `decisions` 鐨勯€氱敤杈圭晫銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歀LM 鍙互鍦ㄦ瘡姝ヨ涔堢户缁綋鍓?subgoal锛岃涔堝湪鍒ゆ柇褰撳墠 subgoal 宸插畬鎴愬悗鐢熸垚鏂?subgoal銆?  - repo 瀹炵幇鍙‘璁わ細`cme_final.py` 涓綋妯″瀷杈撳嚭鍖呭惈 `Subgoal:` 鏃讹紝浼氭妸瀹冭涓鸿繘鍏ユ柊 subgoal 鐨勫垏鎹㈢偣锛涙棫 subgoal 闅忓悗浼氬湪搴忓垪鍖栭樁娈佃鍘嬬缉銆?  - 鎺ㄦ柇锛氱幇鏈?evolution trigger 瀹舵棌杩樻病鏈夌洿鎺ヨ鐩栬繖绉?subgoal-level switch 璇箟銆?
#### memory_evolution

- 璁烘枃閲屽仛浜嗕粈涔?  - 瀵瑰凡瀹屾垚 subgoal 鐨勫畬鏁?action-observation trajectory 鍋?summarization锛屽緱鍒?summarized observation锛?  - 榛樿涓婁笅鏂囦笉鍐嶅睍寮€鏃?detailed trajectory锛岃€屾槸鐢?`(subgoal, summary)` 鍙栦唬锛?  - 褰撳墠 subgoal 鐨勮缁嗚建杩圭户缁繚鎸佸睍寮€銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SummaryRewriteEvolution`
    - 鏈夆€滆拷鍔?summary record鈥濈殑杞粨锛屼絾瀹冨榻愮殑鏄崟涓?unit 鐨?summary rewrite锛岃€屼笉鏄?HiAgent 鐨勫姝?subgoal trajectory summarization銆?  - `LayerMoveEvolution`
    - 鍙互鎶婂唴瀹瑰鍒跺埌鍙︿竴灞傦紝浣嗗彧鏄?copy-append锛屼笉鍏峰鈥滄棫缁嗚妭榛樿闅愯棌銆乻ummary 鍙栦唬鍏朵笂涓嬫枃鏆撮湶鈥濈殑鏇挎崲璇箟銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯瀵瑰畬鏁?subgoal chunk 鍋?summarization 鐨?evolution primitive銆?  - 缂哄皯鈥渟ummary 鍙栦唬鏃х粏鑺備负榛樿涓婁笅鏂囨毚闇插舰寮忥紝浣嗘棫缁嗚妭浠嶅彲鎸夐渶鎭㈠鈥濈殑 replacement/archive 鏈哄埗銆?  - 缂哄皯瀵?chunk 绾ц€岄潪 unit 绾?memory evolution 鐨勬槑纭壙杞借竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氬凡瀹屾垚 subgoal 鐨?action-observation pairs 浼氳 summarized observation 鏇挎崲锛涘綋鍓?subgoal 淇濈暀缁嗚妭銆?  - repo 瀹炵幇鍙‘璁わ細
    - `cme_final.py` 鍦?`serialize_history()` 涓闄ゆ渶鍚庝竴涓箣澶栫殑鏃?subgoal 鍋氬帇缂╋紱
    - 瀹冭皟鐢?`TrajectorySummarizer.generate_summary([trajectory], [subgoal])` 鐢熸垚 summary锛?    - 鏃?subgoal 鍦ㄤ笂涓嬫枃涓鏀瑰啓鎴?`Subgoal + Observation(summary)`銆?  - 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
    - 鍏紑 repo 涓?`TrajectorySummarizer` 鐨勬簮鐮佸綋鍓嶆湭鎵惧埌锛屽洜姝?summary 妯″瀷鍐呴儴瀹炵幇缁嗚妭鏃犳硶纭銆?
#### retrieval

- 璁烘枃閲屽仛浜嗕粈涔?  - 褰撴ā鍨嬭寰楁煇涓棫 subgoal 鐨勮缁嗚建杩逛粛鐒跺叧閿椂锛屾樉寮忕敓鎴?retrieval function锛屾寜 subgoal id 鍙栧洖瀵瑰簲鐨勫畬鏁?action-observation pairs銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BufferRetrieval`
    - 鍙兘鍙栨渶杩戠獥鍙ｏ紝涓嶈兘鎸?subgoal id 绮剧‘鍥炲彇鏌愪釜鏃?chunk銆?  - `RecencyRetrieval`
    - 鍙兘鎸夋椂闂撮『搴忚鏈€杩戣褰曪紝涓嶇悊瑙?subgoal 杈圭晫銆?  - `LayerAwareRetrieval`
    - 鍙兘鍦ㄥ涓?retriever 闂村垎鍙戯紝涓嶈兘鍗曠嫭鎵挎媴鈥滄寜 subgoal 鏍囪瘑绗︽仮澶嶈缁嗚建杩光€濄€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鎸?subgoal id 妫€绱㈡棫 detailed trajectory 鐨?retrieval primitive銆?  - 缂哄皯鈥渟ummary chunk -> detailed chunk鈥濈殑鎸夐渶 rehydrate/reveal 璇箟銆?  - 缂哄皯涓庢ā鍨嬫樉寮?`retrieve(subgoal_id)` 璋冪敤瀵规帴鐨?retrieval 杈圭晫銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歵rajectory retrieval module 鍏佽妯″瀷鍦ㄩ渶瑕佹椂鍙栧洖杩囧幓鏌愪釜 subgoal 鐨勫畬鏁?action-observation pairs銆?  - repo 瀹炵幇鍙‘璁わ細
    - `cme_final.py` 閫氳繃姝ｅ垯瑙ｆ瀽 `retrieve(3)` 杩欑被鍔ㄤ綔锛?    - 鍛戒腑鍚庢妸 subgoal id 璁板叆 `self.subgoal_idx`锛屽啀閲嶆柊鏋勯€?prompt锛?    - `serialize_history()` 瀵硅鐐瑰悕鐨勬棫 subgoal 涓嶅啀鍘嬬缉锛岃€屾槸鎶婂師濮嬭缁嗚建杩归噸鏂版嫾鍥炰笂涓嬫枃銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細HiAgent 鐨?retrieval 涓嶆槸璇箟鐩镐技妫€绱紝鑰屾槸 chunk id-addressed trajectory recall銆?
#### readout

- 璁烘枃閲屽仛浜嗕粈涔?  - 鎶?working memory 娓叉煋鎴?prompt 涓婁笅鏂囷細
    - 鍘嗗彶 subgoal 浠?`subgoal + summary` 褰㈠紡鍑虹幇锛?    - 褰撳墠 subgoal 浠ュ畬鏁磋缁嗚建杩瑰嚭鐜帮紱
    - 鑻ヨЕ鍙?retrieval锛屽垯鏌愪簺鏃?subgoal 涔熶細鎭㈠涓哄畬鏁磋缁嗚建杩广€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ConcatenateReadout`
    - 鑳藉仛绾枃鏈嫾鎺ワ紝浣嗕笉鑳芥樉寮忎繚鐣欏眰绾?working-memory 缁撴瀯銆?  - `GroupedByLayerReadout`
    - 杞粨涓婃渶鎺ヨ繎鈥滄寜灞傛樉绀衡€濓紝浣嗕粛涓嶈兘琛ㄨ揪 subgoal 缂栧彿銆乻ummary-vs-detail 鍒囨崲銆佹寜闇€鎭㈠鏃?chunk 绛夋牸寮忋€?  - `PromptContextReadout`
    - 鏈夆€滄嫾 prompt context鈥濈殑鏂瑰悜锛屼絾褰撳墠鏇村亸 Reflexion/readout 璇箟锛屼笉鏄?HiAgent 鐨勫眰绾у伐浣滆蹇嗗憟鐜般€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯涓€涓樉寮忛潰鍚?hierarchical working memory 鐨?readout 妯℃澘銆?  - 缂哄皯绋冲畾鍛堢幇鈥滃巻鍙?summary + 褰撳墠缁嗚妭 + 鎭㈠鏃х粏鑺傗€濈殑璇诲嚭鏍煎紡銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛欻iAgent 鐨?working memory 鏈川涓婂氨鏄竴绉嶅眰绾т笂涓嬫枃缁勭粐鏂瑰紡銆?  - repo 瀹炵幇鍙‘璁わ細`cme_final.py` 鐨?`make_prompt()` 鐩存帴閫氳繃 `serialize_history()` 鎶婂帇缂╁悗鐨勬棫 subgoal銆佸綋鍓嶈缁嗚建杩广€佷互鍙婅 retrieval 鎸囧畾鐨勬棫璇︾粏杞ㄨ抗鎷兼垚鏈€缁?prompt銆?  - 鎺ㄦ柇锛氳 slot 涓嶆槸 HiAgent 鐨勬渶闅剧偣锛屼絾褰撳墠閫氱敤 readout 杩樹笉瓒充互蹇犲疄閲嶈〃杈惧叾灞傜骇涓婁笅鏂囨牸寮忋€?
### 涓?MemPrimitive 鐜版湁缁勪欢鐨勫鐓х粨璁?
| slot | 缁撹 | 璇存槑 |
| --- | --- | --- |
| `unit_formation` | 鐩存帴澶嶇敤 | `PassThroughUnitFormation` 瓒充互鎵胯浇 step-level working-memory 澧為噺 |
| `representation` | 鐩存帴澶嶇敤 | HiAgent 涓昏渚濊禆鑷劧璇█鏂囨湰琛ㄧず锛屽鏉傛満鍒朵笉鍦ㄨ slot |
| `write_trigger` | 鐩存帴澶嶇敤 | `AlwaysTrigger` 瓒冲 |
| `organization` | 閮ㄥ垎澶嶇敤 | 鏈?append / routing 澶栧３锛屼絾缂?subgoal-chunk 灞傜骇缁勭粐 |
| `evolution_trigger` | 閮ㄥ垎澶嶇敤 | 鏈夐€氱敤 trigger 澶栧３锛屼絾缂?subgoal-completion 瑙﹀彂 |
| `memory_evolution` | 閮ㄥ垎澶嶇敤 | 鏈?`SummaryRewriteEvolution` / `LayerMoveEvolution` 杞粨锛屼絾缂?chunk-level summarization 涓?replacement/archive 璇箟 |
| `retrieval` | 閮ㄥ垎澶嶇敤 | 鏈?recency/buffer/layer-aware 杞粨锛屼絾缂烘寜 subgoal id 鎭㈠鏃?detailed trajectory |
| `readout` | 閮ㄥ垎澶嶇敤 | 鏈夐€氱敤鏂囨湰/Prompt readout锛屼絾缂?hierarchical working-memory 鍛堢幇鏍煎紡 |

### 閲嶈〃杈惧垽鏂?
鍙兘閮ㄥ垎鏄犲皠銆?
涓昏鍘熷洜涓嶆槸 HiAgent 闇€瑕佹柊鐨勯《灞?slot锛岃€屾槸褰撳墠缂哄け鐨勬濂芥槸瀹冪殑鏍稿績 working-memory 鏈哄埗杈圭晫锛?
- subgoal-chunked hierarchical organization
- subgoal-completion-triggered chunk summarization
- summary 鏇夸唬鏃х粏鑺傜殑榛樿涓婁笅鏂囨毚闇茶涔?- 鎸?subgoal id 鎭㈠鏃?detailed trajectory 鐨?retrieval

濡傛灉鍙敤鐜版湁妯″潡寮鸿鎷艰锛屾渶澶氳兘鍋氬嚭鈥滄湁鏂囨湰 summary銆佹湁 append銆佹湁绠€鍗?readout鈥濈殑杩戜技鐗堬紝浣嗚繕涓嶈冻浠ュ繝瀹為噸琛ㄨ揪 HiAgent 浣滀负灞傜骇 working-memory 绠＄悊鏂规硶鐨勫叧閿満鍒躲€?
### 澶囨敞涓庤瘉鎹竟鐣?
- 璁烘枃鏄庤
  - HiAgent 鎶?subgoal 浣滀负 working-memory chunk銆?  - 褰撳墠 subgoal 淇濈暀璇︾粏 action-observation pairs锛岃繃鍘?subgoal 淇濈暀 summarized observation銆?  - summarized observation 鐢?`S(g_i, o_0, a_0, ..., o_t)` 鐢熸垚锛屽苟闇€瑕佽瘎浼?subgoal 鏄惁瀹屾垚銆?  - 褰撻渶瑕佹棫缁嗚妭鏃讹紝妯″瀷鍙敓鎴?retrieval function 鍙栧洖鏌愪釜 subgoal 鐨勫畬鏁?trajectory銆?- repo 瀹炵幇鍙‘璁?  - `agentboard/agents/cme_final.py` 缁存姢 `self.memory`锛屽叾涓贩鍚堜繚瀛?`Subgoal`銆乣Action`銆乣Observation` 鏂囨湰鏉＄洰銆?  - `serialize_history()` 浼氬帇缂╂渶鍚庝竴涓?subgoal 涔嬪墠鐨勬棫 subgoal锛屽彧淇濈暀 `Subgoal + Observation(summary)`銆?  - `retrieve(subgoal_id)` 浼氳鏄惧紡瑙ｆ瀽锛屽懡涓殑鏃?subgoal 涓嶅啀鍘嬬缉锛岃€屾槸鎶婂師濮?detailed trajectory 閲嶆柊鎷煎洖 prompt銆?- 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂?  - HiAgent 鐨勫叧閿?primitive 杈圭晫鏇存帴杩戔€渉ierarchical working-memory chunk lifecycle鈥濓紝鑰屼笉鏄紶缁熺殑 semantic retrieval 鎴?long-term memory indexing銆?  - 鍦?MemPrimitive 閲岋紝鏈€鑷劧鐨勬柊澧炴ā鍧楀簲闆嗕腑鍦?`organization`銆乣evolution_trigger`銆乣memory_evolution`銆乣retrieval`銆乣readout`锛岃€屼笉鏄?`representation`銆?- 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
  - 鍏紑 repo 涓?`TrajectorySummarizer` 琚皟鐢紝浣嗗搴旀簮鐮佸綋鍓嶆湭鎵惧埌锛屽洜姝ゆ憳瑕佹ā鍨嬬殑鍏蜂綋瀹炵幇鏂瑰紡鏃犳硶纭銆?  - 鍏紑 eval config 浣跨敤鐨?agent 鍚嶇О鏄?`ContextEfficientAgent`锛岃€屼唬鐮侀噷娉ㄥ唽鐨勬槸 `ContextEfficientAgentV2`锛況epo 褰撳墠瀛樺湪鍛藉悕涓嶄竴鑷达紝璇存槑鍏紑瀹炵幇鍙兘鏈夌己澶辨垨鏁寸悊涓嶅畬鏁翠箣澶勩€?
## Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

璁烘枃閾炬帴: <https://arxiv.org/abs/2504.19413>

瀹樻柟 repo: <https://github.com/mem0ai/mem0>

### 璁烘枃渚?memory 鏈哄埗閫熷啓

Mem0 璁烘枃涓讳綋鐨勬牳蹇冧笉鏄€滄妸鏁存瀵硅瘽鐩存帴濉炶繘鍚戦噺搴撯€濓紝鑰屾槸涓€涓袱闃舵 memory pipeline锛?
1. extraction phase  
   - 浠モ€滃綋鍓嶆秷鎭鈥濅綔涓哄鐞嗗崟浣嶏紱
   - 缁撳悎浼氳瘽鎽樿涓庢渶杩戣嫢骞叉潯娑堟伅浣滀负涓婁笅鏂囷紱
   - 鐢?LLM 鎶藉彇 candidate facts / candidate memories銆?2. update phase  
   - 瀵规瘡涓?candidate fact 妫€绱?top-k 鐩镐技鏃ц蹇嗭紱
   - 鍐嶇敱 LLM 鍐崇瓥瀵圭煡璇嗗簱鎵ц `ADD / UPDATE / DELETE / NONE`锛?   - 鏈€缁堟妸缁撴灉鍐欏洖鍚戦噺璁板繂搴擄紝骞剁淮鎶ゆ椂闂翠竴鑷存€т笌鍘诲啑浣欍€?
璁烘枃杩樻彁鍑轰簡 graph-memory 澧炲己鐗堬細鎶?memory 琛ㄧず鎴愬甫瀹炰綋绫诲瀷銆佸疄浣?embedding 鍜屽叧绯昏竟鐨勬湁鍚戞爣娉ㄥ浘锛涙柊淇℃伅杩涘叆鏃朵細鍋氬啿绐佹娴嬶紝鎶婅繃鏃跺叧绯绘爣涓烘棤鏁堬紱妫€绱㈡椂鍚屾椂鏀寔 query 瀹炰綋閿氬畾鎵╁睍涓?triplet-level 璇箟鍖归厤銆?
濡傛灉鎸?MemPrimitive 褰撳墠 slot 浣撶郴寮鸿钀戒綅锛屾渶鑷劧鐨勬媶娉曟槸锛?
- `unit_formation`: 浠ヤ竴瀵规柊娑堟伅涓轰氦浜掑崟鍏冦€?- `representation`: 浠庢柊浜や簰涓娊 salient facts锛沢raph 鐗堝垯鎶藉疄浣撶被鍨嬩笌鍏崇郴 triplets銆?- `write_trigger`: 瀵瑰凡鎶藉嚭鐨?candidate facts 榛樿閮借繘鍏ュ悗缁瘎浼般€?- `organization`: 鍩虹鐗堟槸骞冲潶鐨?scoped vector memory锛沢raph 鐗堟槸 typed entity-relation graph銆?- `evolution_trigger`: 姣忎釜 candidate fact 閮借Е鍙戜竴娆♀€滀笌鏃ц蹇嗘瘮杈冨苟鍐冲畾濡備綍缁存姢鈥濈殑娴佺▼銆?- `memory_evolution`: 鐪熸鎵ц `ADD / UPDATE / DELETE / NONE`锛沢raph 鐗堣繕瑕佸仛鍐茬獊鍏崇郴澶辨晥鍖栥€?- `retrieval`: 鍩虹鐗堟槸 embedding similarity recall锛沢raph 鐗堟槸 entity-centric + triplet similarity 鐨勫浘妫€绱€?- `readout`: 杩斿洖鐩稿叧 memory 鏂囨湰锛実raph 鐗堥檮甯﹀叧绯讳笂涓嬫枃銆?
### 鎸?MemPrimitive slot 鐨勬媶瑙?
#### unit_formation

- 璁烘枃閲屽仛浜嗕粈涔?  - 璁烘枃鎶娾€滃綋鍓嶆秷鎭笌鍏跺墠涓€鏉℃秷鎭€濈粍鎴愪竴涓?interaction unit锛涙娊鍙栨椂杩樹細寮曞叆 conversation summary 涓?recent messages 浣滀负杈呭姪涓婁笅鏂囥€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲畬鍏ㄧ洿鎺ュ鐢ㄦā鍧椼€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `PassThroughUnitFormation`
    - 濡傛灉涓婃父宸茬粡鎶娾€滄秷鎭 + 蹇呰涓婁笅鏂団€濋鍏堟墦鍖呮垚涓€涓?`Observation`锛屽彲浠ユ壙杞?Mem0 鐨勬渶灏忓鐞嗗崟鍏冦€?  - `MetadataHintUnitFormation`
    - 鍙互浠?metadata hints 涓瀯閫?unit锛屼絾瀹冨苟涓嶆槸涓衡€滃璇濆弻娑堟伅鍗曞厓鈥濊璁＄殑銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯闈㈠悜 conversation turn-pair 鐨勫師鐢?unit formation銆?  - 缂哄皯鎶?recent-message window / conversation summary 绋冲畾骞跺叆褰撳墠 unit 鐨勫舰鎴愯竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歁em0 浠ユ柊鐨?message pair 涓哄鐞嗗崟浣嶏紝骞剁粨鍚?conversation summary 涓?recent messages 鍋?extraction銆?  - repo 瀹炵幇鍙‘璁わ細褰撳墠 OSS `Memory.add(...)` 鐩存帴鎺ユ敹 message 鍒楄〃锛宍parse_messages(...)` 鎶婂璇濅覆鎴愭枃鏈紝浣嗘病鏈変竴涓樉寮忕殑鈥渢urn-pair unit formation鈥濇ā鍧椼€?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍦?MemPrimitive 涓紝杩欎竴璇箟鍙互鍊?`PassThroughUnitFormation` 鎵胯浇锛屼絾闇€瑕佷笂娓告墜鍔ㄦ墦鍖咃紝鍥犳鍙兘閮ㄥ垎澶嶇敤銆?
#### representation

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍩虹 Mem0锛氫粠鏂颁氦浜掍腑鎶藉彇 salient facts / memories锛屼笖鎶藉彇瑕佹劅鐭ヤ細璇濇憳瑕佷笌鏈€杩戜笂涓嬫枃銆?  - graph 鐗堬細鍏堟娊瀹炰綋鍙婄被鍨嬶紝鍐嶆娊瀹炰綋闂村叧绯?triplets銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BasicRepresentation(elements=("text", "summary"))`
    - 鏈夆€滄枃鏈?+ 鎽樿鈥濆瑙傦紝浣嗗畠鏄鍗曚釜 unit 鍋氬眬閮ㄨ〃绀哄寮猴紝涓嶆槸 Mem0 閭ｇ鈥滃熀浜庢柊浜や簰 + 浼氳瘽涓婁笅鏂囨娊 candidate facts鈥濈殑 extraction銆?  - `BasicRepresentation(elements=("text", "triple", "entities"))`
    - 瀵?graph 鐗堝彧瑕嗙洊浜嗏€滅湅璧锋潵鍍忔湁 entities/triples鈥濈殑琛ㄩ潰缁撴瀯锛屼笉绛変簬甯﹀疄浣撶被鍨嬬殑鍏崇郴鎶藉彇銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 conversation-context-aware 鐨?salient fact extraction primitive銆?  - 缂哄皯鈥滀粠娑堟伅瀵逛腑鐩存帴浜у嚭 candidate facts 鍒楄〃鈥濈殑琛ㄧず妯″潡銆?  - 缂哄皯 graph 鐗堟墍闇€鐨?typed entity extraction + relation triplet extraction primitive銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氬熀纭€ Mem0 鐨?extraction phase 缁撳悎 `S_t` 涓?recent messages锛屼粠鏂?message pair 涓娊 memories锛沢raph 鐗堟槑纭垎涓?entity extraction 涓?relationship generation 涓ゆ銆?  - repo 瀹炵幇鍙‘璁わ細`mem0/memory/main.py` 閲?`get_fact_retrieval_messages(...)` 椹卞姩 LLM 鎶藉彇 `facts`锛沗mem0/memory/graph_memory.py` 涓?`_retrieve_nodes_from_data(...)` 涓?`_establish_nodes_relations_from_data(...)` 鍒嗗埆鍋氬疄浣撲笌鍏崇郴鎶藉彇銆?  - repo 瀹炵幇鍙‘璁わ細褰撳墠 OSS 璺緞娌℃湁鐪嬪埌璁烘枃鎵€璇寸殑鐙珛 async summary refresh 妯″潡鏆撮湶鍦ㄥ紑婧愪富閾捐矾涓€?  - 鎺ㄦ柇锛歁emPrimitive 鐜版湁 representation 瀹舵棌涓昏鍋?unit-local 琛ㄧず澧炲己锛屼笉瓒充互鐩存帴琛ㄨ揪 Mem0 鐨?extraction 鏍稿績銆?
#### write_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 璁烘枃娌℃湁鍗曠嫭寮鸿皟涓€涓鏉?write gate锛涗竴鏃?extraction phase 浜у嚭 candidate facts锛岃繖浜涘€欓€夊氨浼氳繘鍏?update evaluation銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `AlwaysTrigger`
- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`
    - 鍙互鍏呭綋 always-like 瑙﹀彂锛屼絾涓嶆槸鏈€鑷劧琛ㄨ揪銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歝andidate facts 浼氳繘鍏?update phase锛屼笌鐩镐技鏃ц蹇嗘瘮杈冨悗鍐冲畾鏈€缁堟搷浣溿€?  - repo 瀹炵幇鍙‘璁わ細`_add_to_vector_store(...)` 涓彧瑕佹娊鍑轰簡 `new_retrieved_facts`锛屽氨浼氱户缁蛋鐩镐技妫€绱㈠拰 memory action 鍐崇瓥銆?  - 鎺ㄦ柇锛歁em0 鐨?selectivity 涓昏鍙戠敓鍦?extraction 涓?update-resolution锛屼笉鍦ㄧ嫭绔?write gate銆?
#### organization

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍩虹 Mem0锛氭妸 memory 缁存姢鍦ㄤ竴涓?scoped 鐨勫钩鍧﹁蹇嗗簱閲岋紝渚濊禆 embedding 妫€绱€佹椂闂存埑鍜岀ǔ瀹?ID銆?  - graph 鐗堬細鎶?memory 缁勭粐鎴愭湁鍚戞爣娉ㄥ浘锛岃妭鐐瑰惈瀹炰綋绫诲瀷銆乪mbedding 涓庡厓鏁版嵁锛岃竟涓哄叧绯?triplets銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `AppendOrganization`
    - 瀵瑰熀纭€鐗?ADD 璺緞鏄帴杩戠殑锛岃兘琛ㄨ揪鈥滄妸鏂?fact 浣滀负 record 鍐欏叆鏌愬眰鈥濄€?    - 浣嗗畠涓嶅唴寤?Mem0 鐨?scoped mutable memory 璇箟锛屼篃涓嶅鐞嗕笌鍚庣画 update/delete 鐢熷懡鍛ㄦ湡鐨勮€﹀悎銆?  - `GraphAppendOrganization`
    - 瀵?graph 鐗堝彧鎻愪緵浜?record-centric graph append 澶栧３锛屼笉绛変簬 typed entity-relation graph銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯闈㈠悜鍩虹 Mem0 鐨勨€滃钩鍧︿絾鍙悗缁師浣嶇淮鎶も€濈殑 memory organization 璇箟杈圭晫銆?  - 缂哄皯 graph 鐗堢殑 typed entity-relation graph organization銆?  - 缂哄皯瀵瑰疄浣撹妭鐐?embedding銆佸叧绯昏竟鐘舵€併€佷綔鐢ㄥ煙 metadata 鐨勭粺涓€缁勭粐 contract銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氬熀纭€鐗堜娇鐢ㄥ悜閲?memory database锛沢raph 鐗堜娇鐢?directed labeled graph锛岃妭鐐瑰惈 type/embedding/metadata銆?  - repo 瀹炵幇鍙‘璁わ細`mem0/memory/main.py` 鎶?memory 浣滀负鍚戦噺搴撹褰曠鐞嗭紱`mem0/memory/graph_memory.py` 鐢ㄥ浘鏁版嵁搴撶淮鎶ゅ疄浣撹妭鐐逛笌鍏崇郴杈广€?  - 鎺ㄦ柇锛氬熀纭€鐗堢殑闈欐€佺粍缁囦笉澶嶆潅锛屼絾濡傛灉鎸?MemPrimitive 鐨勭粍缁?slot 涓ユ牸鍒掔晫锛屽綋鍓嶆ā鍧椾粛鍙鐩栦簡 append 澶栧３銆?
#### evolution_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 姣忎釜 candidate fact 閮戒細瑙﹀彂涓€娆?update evaluation锛沢raph 鐗堟柊鍏崇郴杩涘叆鏃朵篃浼氳Е鍙?conflict detection銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鍙洿鎺ョ敤鐜版湁 always-like 缁勫悎琛ㄨ揪銆?  - `ThresholdTrigger`
    - 璁惧畾鎴愭亽鐪熸椂锛屽彲琛ㄨ揪鈥滄瘡涓?candidate fact 閮借繘鍏ュ悗缁?evolution鈥濄€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `NewWriteEvolutionTrigger`
    - 鏈夆€滄柊鍐欏叆鍚庡仛缁存姢鈥濈殑杞粨锛屼絾涓嶇瓑浜?Mem0 鐨勬瘡-fact update evaluation銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲繀椤绘柊澧炵殑瑙﹀彂 primitive锛涚幇鏈夌粍鍚堣冻澶熸壙杞借 slot銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歶pdate phase 鏄?extraction 鍚庣殑鍥哄畾鍚庣画闃舵锛岃€屼笉鏄潯浠跺緢澶嶆潅鐨勭█鐤忚Е鍙戙€?  - repo 瀹炵幇鍙‘璁わ細`_add_to_vector_store(...)` 瀵规瘡涓柊 fact 閮芥墽琛岀浉浼兼绱笌 memory action 鍐崇瓥銆?  - 鎺ㄦ柇锛氳 slot 鍦?Mem0 涓笉鏄富瑕佺摱棰堛€?
#### memory_evolution

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍩虹 Mem0 鐨勭湡姝ｆ牳蹇冨湪杩欓噷锛氬姣忎釜 candidate fact 鍏堟壘 top-k 鐩镐技鏃ц蹇嗭紝鍐嶇敱 LLM 鍐冲畾鎵ц `ADD / UPDATE / DELETE / NONE`銆?  - graph 鐗堣繕瑕佸鍐茬獊鍏崇郴鍋?invalidation锛岃€屼笉鏄彧浼氳拷鍔犮€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SummaryRewriteEvolution`
    - 鏈夆€滃熀浜庡凡鏈夊唴瀹瑰仛鏀瑰啓鈥濈殑杞粨锛屼絾涓?Mem0 鐨?action-resolution 瀹屽叏涓嶅悓銆?  - `LayerMoveEvolution`
    - 鏈?record rewrite / move 鐨勫懗閬擄紝浣嗕笉璐熻矗鐩镐技妫€绱笌 `ADD/UPDATE/DELETE/NONE` 鍐崇瓥銆?  - `GraphLinkEvolution`
    - graph 鐗堝彲鍊熷叾鈥滃浘涓婂仛棰濆缁存姢鈥濈殑澶栧３锛屼絾涓嶆敮鎸?conflict invalidation銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 `ADD / UPDATE / DELETE / NONE` 鍥涜矾 memory maintenance primitive銆?  - 缂哄皯鈥渃andidate fact -> top-k similar old memories -> LLM decision -> store mutation鈥濈殑瀹屾暣 evolution module銆?  - 缂哄皯 graph 鐗堚€滄妸鍐茬獊鍏崇郴鏍囪澶辨晥鑰岄潪鐩存帴鍒犻櫎鈥濈殑 evolution 璇箟銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氬熀纭€ Mem0 鐨?update phase 鏄庣‘鏈夊洓绉嶆搷浣滐紱graph 鐗堝鍐茬獊鍏崇郴鍋?obsolete/invalidation銆?  - repo 瀹炵幇鍙‘璁わ細`mem0/memory/main.py` 涓?`_add_to_vector_store(...)` 鍏堝姣忎釜鏂?fact 鍋?vector search锛屽啀鐢?`get_update_memory_messages(...)` 璁?LLM 杈撳嚭 `ADD / UPDATE / DELETE / NONE`锛屽苟鍒嗗埆璋冪敤 `_create_memory` / `_update_memory` / `_delete_memory`銆?  - repo 瀹炵幇鍙‘璁わ細`mem0/memory/graph_memory.py` 涓?`_get_delete_entities_from_search_output(...)` + `_delete_entities(...)` 浼氭妸鍏崇郴鏍囨垚 `valid = false`銆?  - 鎺ㄦ柇锛氳繖鏄綋鍓?MemPrimitive 涓?Mem0 涔嬮棿鏈€鍏抽敭銆佷篃鏈€闆嗕腑鐨勭己鍙ｃ€?
#### retrieval

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍩虹 Mem0锛氫互 embedding similarity 鍋?memory recall銆?  - graph 鐗堬細鍚屾椂鏈?entity-centric graph traversal 涓?triplet semantic matching 涓ゆ潯 retrieval 璺€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `EmbeddingSimilarityRetrieval`
    - 瀵瑰熀纭€鐗?user-facing recall 瓒冲鎺ヨ繎銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphSeedAndExpandRetrieval`
    - 鑳借〃杈锯€滃浘涓壘 seed 鍐嶆墿灞曗€濈殑绮楄疆寤擄紝浣嗕笉绛変簬 Mem0 graph 鐗堢殑 entity-anchor traversal銆?  - `BM25Retrieval`
    - 瀵?graph repo 鐨?BM25 relation rerank only 鏄緢寮辩殑灞€閮ㄨ繎浼硷紝涓嶆槸璁烘枃涓诲共銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 濡傛灉瑕佽鐩?graph 鐗堬紝闇€瑕佷竴涓€渆ntity-centric + triplet similarity鈥濈殑娣峰悎鍥炬绱?primitive銆?  - 褰撳墠 retrieval slot 涔熸病鏈夌洿鎺ヨ〃杈锯€渋ngest-time top-k similar memories for update evaluation鈥濈殑浣嶇疆锛涜鑳藉姏鏇撮€傚悎涓嬫矇鍒?`memory_evolution` 鍐呴儴銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛氬熀纭€鐗堢敤鍚戦噺 memory database 鍋氱浉浼兼绱紱graph 鐗堝悓鏃舵湁 entity-centric 涓?semantic triplet retrieval銆?  - repo 瀹炵幇鍙‘璁わ細`search(...)` 涓昏矾寰勮皟鐢?`_search_vector_store(...)`锛屽浘妫€绱㈠苟琛岃繑鍥?`relations`锛沗graph_memory.py` 涓?`search(...)` 鍏堟娊 query entities锛屽啀鍦ㄥ浘涓婃壘鐩歌繎鑺傜偣涓庡叧绯伙紝鏈€鍚庣敤 BM25 鍋氬叧绯诲簭鍒楅噸鎺掋€?  - 鎺ㄦ柇锛氬熀纭€鐗?retrieval 鍙洿鎺ュ鐢紝graph 鐗堝彧鑳介儴鍒嗘槧灏勩€?
#### readout

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍩虹鐗堟妸鐩稿叧 memories 浜ょ粰涓嬫父瀵硅瘽妯″瀷浣滀负琛ュ厖涓婁笅鏂囥€?  - graph 鐗堜細棰濆杩斿洖鍏崇郴涓婁笅鏂囥€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GraphReadout`
    - 瀵?graph 鐗堣皟璇曞彲鐢紝浣嗕笉鏄?Mem0 榛樿鐨勭粨鏋滃舰鎬併€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鍩虹鐗堟棤鍏抽敭缂哄彛銆?  - 濡傛灉甯屾湜蹇犲疄瀵归綈 graph 鐗堚€滃悜閲忓懡涓?+ relations 骞惰杩斿洖鈥濈殑鎺ュ彛褰㈠紡锛屽彲鑳介渶瑕佷竴涓洿璐磋繎 Mem0 API 鐨?readout 閫傞厤灞傦紝浣嗚繖涓嶆槸涓绘満鍒剁己鍙ｃ€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔?  - 璁烘枃鏄庤锛歳etrieved memories 琚敞鍏ュ悗缁洖绛旇繃绋嬶紱graph 鐗堟彁渚涢澶?relational context銆?  - repo 瀹炵幇鍙‘璁わ細`search(...)` 杩斿洖 `{"results": original_memories, "relations": graph_entities}`銆?  - 鎺ㄦ柇锛歳eadout 涓嶆瀯鎴?Mem0 鐨勪富瑕侀噸琛ㄨ揪闅滅銆?
### 涓?MemPrimitive 鐜版湁缁勪欢鐨勫鐓х粨璁?
| slot | 缁撹 | 璇存槑 |
| --- | --- | --- |
| `unit_formation` | 閮ㄥ垎澶嶇敤 | `PassThroughUnitFormation` 鍙壙杞介鎵撳寘鐨?turn-pair锛屼絾缂哄師鐢?conversation pair / context bundling |
| `representation` | 鍙兘閮ㄥ垎澶嶇敤 | 缂轰笂涓嬫枃鎰熺煡 fact extraction锛沢raph 鐗堢己 typed entity + relation triplet extraction |
| `write_trigger` | 鐩存帴澶嶇敤 | `AlwaysTrigger` 瓒冲琛ㄨ揪鈥滃€欓€夎繘鍏?update evaluation鈥?|
| `organization` | 閮ㄥ垎澶嶇敤 | 鍩虹 append 澶栧３鍙敤锛屼絾 graph 鐗?typed entity-relation graph 浠嶇己鏄庣‘鎵胯浇 |
| `evolution_trigger` | 鐩存帴澶嶇敤 | 鍙敤 always-like 瑙﹀彂琛ㄨ揪鈥滄瘡涓?candidate fact 閮借繘鍏ョ淮鎶も€?|
| `memory_evolution` | 褰撳墠缂哄け鍏抽敭鑳藉姏 | 缂?Mem0 鏍稿績鐨?similarity-resolved `ADD/UPDATE/DELETE/NONE` 缁存姢 primitive |
| `retrieval` | 閮ㄥ垎澶嶇敤 | 鍩虹鍚戦噺 recall 鍙鐢紱graph 鐗?retrieval 鍙兘灞€閮ㄦ槧灏?|
| `readout` | 鐩存帴澶嶇敤 | 閫氱敤 memory 鏂囨湰 readout 瓒充互鎵胯浇鍩虹鐗堣緭鍑?|

### 閲嶈〃杈惧垽鏂?
鍙ぇ閮ㄥ垎閲嶈〃杈俱€?
鏇村噯纭湴璇达細

- **鍩虹 Mem0** 宸茬粡寰堟帴杩?MemPrimitive 褰撳墠鑳藉姏杈圭晫锛?- 鐪熸闃荤鈥滃畬鏁撮噸琛ㄨ揪鈥濈殑锛屾槸瀹冩渶鏍稿績鐨?update-resolution 娌℃湁鐜版垚 primitive锛?- **graph-memory 澧炲己鐗?* 鍒欒繘涓€姝ユ毚闇插嚭 typed graph organization銆佸叧绯诲け鏁堝寲銆佸浘妫€绱㈡贩鍚堢瓥鐣ヨ繖浜涙柊澧炵己鍙ｃ€?
鎵€浠ュ鏋滃彧鐪嬭鏂囩殑鍩虹 Mem0 涓讳綋锛岀瀹屾暣閲嶈〃杈惧彧宸皯閲忎絾鍏抽敭鐨勬ā鍧楋紱濡傛灉鎶?graph-memory 鍙樹綋涔熺畻杩涘悓涓€鏉＄洰锛屽垯鏁翠綋浠嶅簲淇濆畧鍐欐垚鈥滃彲澶ч儴鍒嗛噸琛ㄨ揪鈥濓紝鑰屼笉鏄€滃彲瀹屾暣閲嶈〃杈锯€濄€?
### 澶囨敞涓庤瘉鎹竟鐣?
- 璁烘枃鏄庤
  - Mem0 涓讳綋鍒?extraction 涓?update 涓ら樁娈点€?  - extraction 浣跨敤褰撳墠 message pair銆乧onversation summary銆乺ecent messages銆?  - update 瀵规瘡涓?candidate fact 鍏堟壘 top-k 鐩镐技鏃ц蹇嗭紝鍐嶅仛 `ADD / UPDATE / DELETE / NOOP`銆?  - graph-memory 鍙樹綋鎶婅蹇嗚〃绀烘垚 directed labeled graph锛屽苟瀵瑰啿绐佸叧绯诲仛澶辨晥鍖栥€?  - graph 妫€绱㈠寘鍚?entity-centric retrieval 涓?semantic triplet retrieval銆?- repo 瀹炵幇鍙‘璁?  - `mem0/memory/main.py` 涓?`_add_to_vector_store(...)` 鍏堢敤 LLM 鎶?`facts`锛屽啀瀵规瘡涓?fact 鍋氬悜閲忔绱紝鍐嶈 LLM 杈撳嚭 `ADD / UPDATE / DELETE / NONE`锛屾渶鍚庡垎鍒墽琛?create/update/delete銆?  - `mem0/memory/main.py` 鐨?`search(...)` 浠?vector search 涓轰富锛実raph 鎼滅储骞惰杩斿洖鍦?`relations`銆?  - `mem0/memory/graph_memory.py` 涓疄浣撴娊鍙栦笌鍏崇郴鎶藉彇鏄袱姝ワ紱鍥炬洿鏂版敮鎸佹妸鍐茬獊杈规爣璁颁负 `valid = false`銆?- 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂?  - 鍦?MemPrimitive 涓紝Mem0 鐨勬牳蹇冩柊澧?primitive 鏈€鑷劧搴旇惤鍦?`representation` 涓?`memory_evolution`锛岃€屼笉鏄?`write_trigger`銆?  - 璁烘枃閲岀殑鈥渋ngest-time similarity retrieval鈥濇洿鍍?`memory_evolution` 鍐呴儴瀛愯繃绋嬶紝鑰屼笉鏄綋鍓?recall-side `retrieval` slot 鐨勭洿鎺ュ搴旂墿銆?- 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
  - 璁烘枃瀹ｇО鐨?async conversation-summary module锛屽湪褰撳墠 OSS 涓昏矾寰勯噷娌℃湁鐪嬪埌鍚屾牱娓呮櫚鐨勭嫭绔嬪疄鐜拌竟鐣岋紱瀹冨彲鑳芥槸璁烘枃绯荤粺鎴栧钩鍙颁晶鑳藉姏锛岃€屼笉鏄綋鍓嶅紑婧愪唬鐮佷腑鐨勫悓绾фā鍧椼€?  - 褰撳墠寮€婧?repo 宸叉贩鍏ュ钩鍙板寲鑳藉姏锛屽 `user_id/agent_id/run_id` 浣滅敤鍩熴€乺eranker銆乸rocedural memory銆乬raph 骞惰杩斿洖绛夛紱杩欎簺涓嶅簲鍏ㄩ儴鍙嶆帹涓鸿鏂囦富鏂囩殑鏍稿績 memory 鏈哄埗銆?## LightMem: Lightweight and Efficient Memory-Augmented Generation

璁烘枃閾炬帴: <https://openreview.net/forum?id=8QOO8Ufq2M>

瀹樻柟 repo: <https://github.com/zjunlp/LightMem>

鏈 repo 璇佹嵁涓昏鏍稿鐗堟湰: `zjunlp/LightMem` `main` 鍒嗘敮锛屼复鏃舵鏌ュ埌鐨勬彁浜や负 `ca39c30`銆?
### 璁烘枃渚?memory 鏈哄埗閫熷啓

LightMem 鐨?memory 涓绘満鍒朵笉鏄€滄洿澶嶆潅鐨勫彫鍥炲櫒鈥濓紝鑰屾槸鎶婇暱鏈熻蹇嗘瀯寤烘媶鎴愪笁娈碉細

- `Light1 / sensory memory`锛氭柊瀵硅瘽 turn 鍏堢粡杩囬鍘嬬缉锛屽彧淇濈暀淇℃伅瀵嗗害鏇撮珮鐨?token锛屽啀杩涘叆涓€涓湁 token 瀹归噺涓婇檺鐨勬劅瀹樼紦鍐插尯銆?- `Light2 / topic-aware STM`锛氬綋鎰熷畼缂撳啿鍖鸿揪鍒板閲忛槇鍊煎悗锛岀郴缁熷熀浜?attention + 鐩搁偦 turn 璇箟鐩镐技搴﹀仛 topic segmentation锛屾妸鑻ュ共 turn 缁勭粐鎴?topic segment锛涢殢鍚庢妸杩欎簺 segment 鏆傚瓨杩?STM锛孲TM 鍒拌揪闃堝€煎悗鍐嶆寜 topic 绮掑害鍋氭€荤粨锛屽舰鎴愯繘鍏?LTM 鐨勮蹇嗘潯鐩€?- `Light3 / LTM with sleep-time update`锛氬湪绾块樁娈靛鏂?entry 鍏堝仛 soft insert锛屼笉闃诲浜や簰锛涙洿鏂拌鎺ㄨ繜鍒扳€渟leep time鈥濓紝鍏堜负姣忎釜 entry 鏋勯€犱竴涓甫鏃堕棿绾︽潫鐨勭浉浼煎€欓€夋洿鏂伴槦鍒楋紝鍐嶇绾垮苟琛屾墽琛?update/delete/ignore銆?
濡傛灉鎸?MemPrimitive 褰撳墠 slot 浣撶郴纭槧灏勶紝鏈€鑷劧鐨勪富閾捐矾鏄細

- `unit_formation`锛氫互澧為噺瀵硅瘽 turn锛堥€氬父鏄竴杞?user/assistant 浜ゆ崲锛変綔涓鸿繘鍏?sensory memory 鐨勫熀鏈緭鍏ュ崟鍏冦€?- `representation`锛氬厛瀵?turn 鍋氶鍘嬬缉锛屽啀鍦?topic 绮掑害涓婂舰鎴愬彲鎬荤粨銆佸彲绱㈠紩鐨勬枃鏈〃绀猴紱杩涘叆 LTM 鐨?entry 鑷冲皯甯?topic-aware summary 涓?embedding銆?- `write_trigger`锛氫笉鏄瘡鏉?turn 绔嬪嵆鍐欏叆闀挎湡璁板繂锛岃€屾槸鐢?buffer capacity 瑙﹀彂鍒嗘涓庢€荤粨銆?- `organization`锛氭牳蹇冩槸 sensory buffer -> topic-aware STM buffer -> LTM 涓夊眰缁勭粐锛岃€屼笉鏄崟灞?append銆?- `evolution_trigger`锛氶暱鏈熻蹇嗘洿鏂扮敱 sleep-time / offline trigger 鍚姩锛岃€岄潪姣忔鍦ㄧ嚎鍐欏叆鏃剁珛鍗冲畬鎴愩€?- `memory_evolution`锛氬厛鎸夌浉浼煎害涓庢椂闂存埑涓烘瘡涓?entry 鏋勯€?update queue锛屽啀骞惰鍐冲畾 update/delete/ignore銆?- `retrieval`锛氶潰鍚戠敤鎴锋煡璇㈡椂涓昏鏄?embedding retrieval锛涜鏂囧苟涓嶆妸 retrieval 璁捐褰撲綔涓昏鍒涙柊鐐广€?- `readout`锛氭妸妫€绱㈠埌鐨?memory 鏂囨湰鎷兼帴杩斿洖缁欎笅娓搁棶绛斿嵆鍙€?
### 鎸?MemPrimitive slot 鐨勬媶瑙?
#### unit_formation

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍦ㄥ疄楠岃瀹氶噷閲囩敤 incremental dialogue turn feeding锛涙瘡娆¤緭鍏ユ槸涓€杞竴杞埌鏉ョ殑瀵硅瘽 turn銆?  - 浠庢満鍒朵笂鐪嬶紝杩涘叆 sensory memory 鐨勬渶灏忓鐞嗗崟鍏冩槸 turn 绾у唴瀹癸紝鑰屼笉鏄暣娈典細璇濅竴娆℃€у啓鍏ャ€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `PassThroughUnitFormation`
    - 濡傛灉涓婃父宸茬粡鎶婁竴娆?user/assistant 浜や簰棰勫厛鎵撳寘鎴愪竴涓?`Observation`锛屽彲浠ユ壙杞解€渢urn 绾ц緭鍏ュ崟鍏冣€濊繖涓€鏈€灏忓瑙傘€?  - `MetadataHintUnitFormation`
    - 涔熷彲浠ラ潬涓婃父 hints 浜哄伐鏋勯€犲鍗曞厓锛屼絾杩欎笉鏄?LightMem 鍘熺敓鐨?turn ingestion 璇箟銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鏄惧紡鐨勨€滃璇?turn / turn-pair 褰㈡垚鍗曞厓鈥濊兘鍔涜竟鐣屻€?  - 缂哄皯涓庡悗缁?sensory buffer 鍗忎綔鐨?unit formation 绾﹀畾锛涘綋鍓?unit_formation 涓嶇煡閬?buffer token 棰勭畻銆乻peaker 瀵归綈鎴?turn 绮掑害銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛氶噰鐢?turn-level incremental feeding銆?  - repo 瀹炵幇鍙‘璁わ細`src/lightmem/memory/lightmem.py` 鐨?`add_memory(...)` 浠ユ柊娑堟伅鎵硅繘鍏ワ紱`sensory_memory.py` 閲屾寜 user/assistant 鎴愬 turn 缁勭粐鍒囨銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍦?MemPrimitive 閲岋紝杩欎竴 slot 鑻ヨ蹇犲疄琛ㄨ揪 LightMem锛岃嚦灏戦渶瑕佹妸鈥渢urn 绾ц緭鍏ュ崟鍏冣€濇樉寮忓寲锛岃€屼笉鍙槸涓€鑸枃鏈?passthrough銆?
#### representation

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍏堝仛 iterative pre-compression锛屼繚鐣欐洿楂樹俊鎭瘑搴?token銆?  - 鍦?STM/LTM 渚э紝topic segment 浼氳鎬荤粨鎴愭洿绱у噾鐨?summary 琛ㄧず锛屽苟鐢熸垚 embedding 渚涢暱鏈熺储寮曞拰妫€绱€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BasicRepresentation(elements=("summary", "embedding"))`
    - 鑳借鐩栤€滅敓鎴愭憳瑕?+ 鐢熸垚鍚戦噺鈥濊繖涓€绮楄疆寤撱€?  - `BasicRepresentation(elements=("keywords", "entities", "tags"))`
    - 鍙兘鎻愪緵杈呭姪鍏冧俊鎭紝涓嶈兘琛ㄨ揪 LightMem 鐨?token-level pre-compression銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥滈鍘嬬缉鈥濊繖涓€ representation primitive锛涚幇鏈夎〃绀哄寮烘病鏈?token retention / entropy-style compression 璇箟銆?  - 缂哄皯鈥渢opic-aware segment summary鈥濊繖涓€绋冲畾琛ㄥ緛杈圭晫锛涘綋鍓?summary 鏄?unit-local 鐨勶紝涓嶆槸 segment-local 鐨勩€?  - 缂哄皯鎶娾€滃師濮?turn 鍐呭 + 鍘嬬缉鍐呭 + topic summary + summary embedding鈥濇墦鎴愬悓涓€绫婚暱鏈熻蹇?entry 琛ㄧず鐨勬満鍒躲€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歀ight1 鍏堝仛 pre-compression锛汱ight2 鎶?topic 缁撴瀯鎬荤粨鍚庡舰鎴愯繘鍏?LTM 鐨勬潯鐩紝鏉＄洰鍖呭惈 summary embedding銆?  - repo 瀹炵幇鍙‘璁わ細`lightmem.py` 閲?`pre_compress=True` 鏃跺厛璋冪敤 `compress(...)`锛泃opic segment 杩涘叆 `meta_text_extract(...)` 鍚庤杞垚 `MemoryEntry`锛屽苟鍦ㄦ彃鍏ュ悜閲忓簱鏃朵緷璧?embedding銆?  - 褰撳墠璇佹嵁杈圭晫锛歳epo 鐨?extraction prompt 鏇村亸鈥滀簨瀹炴娊鍙栤€濓紝鑰岃鏂囨鏂囧湪鏈哄埗灞傛洿寮鸿皟 topic-aware summarization锛涗袱鑰呬笉鑳藉畬鍏ㄧ敾绛夊彿銆?
#### write_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍐欏叆涓嶆槸鈥滄瘡涓?turn 閮界洿鎺ユ€荤粨骞跺叆 LTM鈥濄€?  - 鎰熷畼缂撳啿鍖鸿揪鍒板閲忛槇鍊兼椂瑙﹀彂 topic segmentation锛汼TM 缂撳啿鍖哄啀娆¤揪鍒伴槇鍊兼椂瑙﹀彂 topic-level extraction / summarization 骞跺舰鎴愰暱鏈熻蹇嗘潯鐩€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`
    - 鍙湁鈥滈槇鍊艰Е鍙戔€濈殑澶栧３鐩镐技锛屼絾褰撳墠鍒嗘暟鏉ユ簮鏄父閲?淇″彿锛屼笉鏄?buffer occupancy 鎴?token budget銆?  - `AlwaysTrigger`
    - 鍙€傚悎琛ㄨ揪鈥渆ntry 涓€鏃﹀舰鎴愬氨鍏ュ簱鈥濓紝涓嶉€傚悎琛ㄨ揪 LightMem 鐨勬壒閲忕紦鍐茶Е鍙戙€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 buffer-capacity / token-budget 椹卞姩鐨?write trigger銆?  - 缂哄皯澶氱骇瑙﹀彂锛歴ensory -> STM 鐨勫垎娈佃Е鍙戯紝浠ュ強 STM -> LTM 鐨勬€荤粨瑙﹀彂銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歴ensory buffer 杈惧埌瀹归噺鍚庤Е鍙?segmentation锛汼TM 杈惧埌闃堝€煎悗鍐嶅仛 summarization銆?  - repo 瀹炵幇鍙‘璁わ細`SenMemBufferManager.add_messages(...)` 涓?`ShortMemBufferManager.add_segments(...)` 閮芥槸鎸?token 涓婇檺鍐冲畾鏄惁瑙﹀彂鍚庣画闃舵銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細LightMem 鐨勫啓瑙﹀彂鏈川鏄€滅紦瀛橀槇鍊艰Е鍙戔€濓紝涓嶆槸鐜版湁 MemPrimitive 瑙﹀彂鏃忔墍瑕嗙洊鐨?metadata/key/甯搁噺闃堝€笺€?
#### organization

- 璁烘枃閲屽仛浜嗕粈涔?  - 缁勭粐缁撴瀯鏄笁灞傜殑锛歴ensory memory buffer銆乼opic-aware STM銆丩TM銆?  - STM 涓棿鎬佷笉鏄畝鍗?append锛岃€屾槸 `{topic, message turns}` 杩欐牱鐨?topic-segment 绱㈠紩缁撴瀯锛汱TM entry 鍒欒繘涓€姝ュ彉鎴?topic summary + raw turn support 鐨勯暱鏈熸潯鐩€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `AppendOrganization`
    - 鍙兘琛ㄨ揪鈥滄妸鏉＄洰鍐欏埌鏌愬眰鈥濓紝涓嶈兘琛ㄨ揪灞傞棿缂撳啿涓?topic segment 缁撴瀯銆?  - `ConditionalLayerOrganization`
    - 鍙仛绠€鍗曞垎灞傝矾鐢憋紝浣嗕笉鑳借〃杈锯€滃厛鍦?sensory/STM 鏆傚瓨锛屽悗鎵归噺鎻愬崌鍒?LTM鈥濈殑鐢熷懡鍛ㄦ湡銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鏄惧紡鐨勫垎灞?buffer organization primitive銆?  - 缂哄皯 topic-segment 绾ф暟鎹粨鏋勫強鍏朵笌 LTM entry 鐨勭粦瀹氬叧绯汇€?  - 缂哄皯鈥滃湪绾垮厛 soft insert锛屽悗绂荤嚎鏇存柊鈥濈殑 LTM 鐢熷懡鍛ㄦ湡缁勭粐杈圭晫銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歀ightMem 鐢?sensory銆丼TM銆丩TM 涓夋ā鍧楃粍鎴愶紱STM 缁存姢 `{topic, message turns}`锛孡TM 瀛?`{topic, {sum_i, user_i, model_i}}`銆?  - repo 瀹炵幇鍙‘璁わ細`SenMemBufferManager`銆乣ShortMemBufferManager`銆乣LightMemory` 鐨勫悜閲?LTM 鎻掑叆璺緞鍒嗗埆瀵瑰簲涓夊眰銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細褰撳墠 MemPrimitive 鐨?layer 姒傚康鑳借〃杈锯€滃灞傗€濓紝浣嗕笉鑳界洿鎺ヨ〃杈锯€滃灞傜紦鍐?+ topic segment 鐢熷懡鍛ㄦ湡鈥濄€?
#### evolution_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - 闀挎湡璁板繂鏇存柊琚欢鍚庡埌 sleep time銆?  - 褰撴墍鏈?entries 鎻掑叆瀹屾垚鎴栨敹鍒?update trigger 鏃讹紝绯荤粺涓?LTM entries 鏋勯€?update queue锛屽苟鍚姩绂荤嚎鏇存柊銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `NewWriteEvolutionTrigger`
    - 鏈夆€滄柊鍐欏叆鍚庤Е鍙戠淮鎶も€濈殑杞粨锛屼絾瀹冩槸灞€閮ㄥ湪绾跨淮鎶わ紝涓嶆槸 LightMem 鐨勭绾挎壒澶勭悊瑙﹀彂銆?  - `ThresholdTrigger`
    - 鍙兘鍋氭娊璞￠槇鍊艰Е鍙戯紝缂哄皯 sleep-time / batch-update 璇箟銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鏄惧紡鐨?sleep-time / offline batch evolution trigger銆?  - 缂哄皯鈥滄瀯閫?update queue鈥濆拰鈥滄墽琛岀绾挎洿鏂扳€濅袱涓樁娈典箣闂寸殑瑙﹀彂杈圭晫銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛氭洿鏂颁笌鍦ㄧ嚎鎺ㄧ悊瑙ｈ€︼紝绛夋墍鏈?entries 鎻掑叆瀹屾垚鎴栨洿鏂拌Е鍙戝埌鏉ュ悗锛屽啀璁＄畻 update queue銆?  - repo 瀹炵幇鍙‘璁わ細`construct_update_queue_all_entries(...)` 涓?`offline_update_all_entries(...)` 鏄嫭绔嬬殑绂荤嚎姝ラ銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍦?MemPrimitive 閲岋紝鎶婅繖涓€璺緞寮鸿濉炶繘鐜版湁 online evolution trigger 浼氫涪澶?LightMem 鐨勫叧閿晥鐜囪涔夈€?
#### memory_evolution

- 璁烘枃閲屽仛浜嗕粈涔?  - 鍦ㄧ嚎闃舵鍙仛 soft update锛屽嵆鏂?entry 鐩存帴鎻掑叆 LTM銆?  - 绂荤嚎闃舵鍏堜负姣忎釜 entry 寤虹珛 top-k 鐩镐技鍊欓€夐槦鍒楋紝骞跺姞鍏ユ椂闂寸害鏉燂紝鍙厑璁歌緝鏂扮殑 entry 鏇存柊杈冩棫鐨?entry銆?  - 闅忓悗姣忎釜鐩爣 entry 鐙珛鎵ц update/delete/ignore锛屽洜姝ゅ彲浠ュ苟琛屻€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SummaryRewriteEvolution`
    - 鏈夆€滃熀浜庡凡鏈夊唴瀹圭敓鎴愭柊鐗堟湰鈥濈殑琛ㄩ潰鐩镐技鎬э紝浣嗕笉鍏峰 update queue 涓庢椂闂寸害鏉熴€?  - `LayerMoveEvolution`
    - 鑳藉仛璁板綍鏀瑰啓/杩佺Щ锛屼絾娌℃湁鈥滅浉浼煎€欓€?+ delete/update/ignore鈥濆喅绛栥€?  - `TraceOnlyEvolution`
    - 鍙兘琛ㄨ揪 no-op/璁板綍鐥曡抗锛屾棤娉曟壙杞界湡瀹炵绾挎洿鏂般€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 timestamp-constrained similarity update queue 鏋勯€犳ā鍧椼€?  - 缂哄皯鈥渢arget entry + candidate source entries -> update/delete/ignore鈥濈殑绂荤嚎缁存姢 primitive銆?  - 缂哄皯骞惰 batch evolution 鐨勮兘鍔涜竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歚Q(e_i)` 鐢辩浉浼煎害 top-k 涓庢椂闂寸害鏉熺粍鎴愶紝涓旀洿鏂板彲骞惰鎵ц銆?  - repo 瀹炵幇鍙‘璁わ細`construct_update_queue_all_entries(...)` 鍏堜负姣忎釜 entry 鍐欏叆 `update_queue`锛沗offline_update_all_entries(...)` 鍐嶅苟琛岃皟鐢?LLM 鍐冲畾 `update` 鎴?`delete`锛屽惁鍒欒烦杩囥€?  - 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣锛歳epo 褰撳墠 `UPDATE_PROMPT` 鍙樉寮忔毚闇?`update/delete/ignore` 涓夌被鍔ㄤ綔锛涜鏂囨鏂囪嫢鎶娾€滅洿鎺ユ彃鍏ユ柊 entry鈥濅篃瑙嗕綔 update 浣撶郴鐨勪竴閮ㄥ垎锛屽垯 repo 鐨勫姩浣滄爣绛句笌璁烘枃鎶借薄灞傜骇骞朵笉瀹屽叏鍚屾瀯銆?
#### retrieval

- 璁烘枃閲屽仛浜嗕粈涔?  - 璁烘枃娌℃湁鎶?retrieval 璁捐褰撲綔涓昏鍒涙柊鐐癸紱鏂规硶璐＄尞涓昏闆嗕腑鍦?memory construction 涓?sleep-time update銆?  - 浠庣郴缁熶娇鐢ㄨ搴︾湅锛孡TM entry 閫氳繃 embedding semantic search 琚彫鍥炪€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `EmbeddingSimilarityRetrieval`
    - 瓒充互琛ㄨ揪鈥滃闀挎湡璁板繂鏉＄洰鍋氬悜閲忕浉浼兼绱⑩€濊繖涓€鏍稿績 recall 鏈哄埗銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `LayerAwareRetrieval`
    - 濡傛灉浠ュ悗鎯虫樉寮忓尯鍒?LTM 涓?summary store锛屽彲浣滀负鍒嗗眰璺敱澶栧３锛屼絾涓嶆槸璁烘枃蹇呰鏉′欢銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 瀵?LightMem 璁烘枃涓讳綋鑰岃█锛屾棤鍏抽敭缂哄彛銆?  - 鑻ユ妸 repo 鍚庣画鐨?summary store / StructMem 鎵╁睍涔熺畻杩涙潵锛屽垯闇€瑕侀澶栫殑灞傞棿妫€绱㈣璁★紱浣嗛偅涓嶆槸杩欑瘒璁烘枃鐨勬牳蹇冭蹇嗘満鍒躲€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛氭晥鐜囧垎鏋愪富瑕佸姣?Summary/Update 鎴愭湰锛屽苟鏄庣‘璇存槑 retrieval stage 涓嶆槸鍏跺叧娉ㄩ噸鐐广€?  - repo 瀹炵幇鍙‘璁わ細`retrieve(...)` 鐩存帴瀵?query 鍋?embedding锛岀劧鍚庢煡鍚戦噺搴撹繑鍥?memory 鏂囨湰銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍦?MemPrimitive 涓紝LightMem 鐨?retrieval 鍙鐜版湁 embedding retrieval 鐩存帴鎵胯浇銆?
#### readout

- 璁烘枃閲屽仛浜嗕粈涔?  - 鎶婃绱㈠嚭鐨?memory 鏉＄洰鏂囨湰浜ょ粰涓嬫父闂瓟鎴栦唬鐞嗕娇鐢ㄣ€?  - readout 涓嶆槸璁烘枃鍒涙柊鐐广€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `ConcatenateReadout`
  - `BulletListReadout`
  - `GroupedByLayerReadout`
- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `JSONReadout`
    - 鍙槸鍦ㄩ渶瑕佺粨鏋勫寲涓嬫父鎺ュ彛鏃跺彲閫夛紝涓嶆槸璁烘枃蹇呰銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 鏃犲叧閿己鍙ｃ€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歀ightMem 鐨勯噸鐐瑰湪 memory bank construction 涓?update efficiency锛屼笉鍦ㄧ壒娈?readout 鏍煎紡銆?  - repo 瀹炵幇鍙‘璁わ細`retrieve(...)` 鏈€缁堟妸缁撴灉鏍煎紡鍖栨垚瀛楃涓插垪琛?鎷兼帴鏂囨湰銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍙 retrieval 浜у嚭姝ｇ‘锛岀幇鏈夐€氱敤 readout 鍗冲彲鎵胯浇 LightMem銆?
### 涓?MemPrimitive 鐜版湁缁勪欢鐨勫鐓х粨璁?
| slot | 缁撹 | 璇存槑 |
| --- | --- | --- |
| `unit_formation` | 閮ㄥ垎澶嶇敤 | `PassThroughUnitFormation` 鍙壙杞介鎵撳寘 turn锛屼絾缂哄師鐢?turn / turn-pair 褰㈡垚璇箟 |
| `representation` | 閮ㄥ垎澶嶇敤 | 鏈?summary/embedding 澶栧３锛屼絾缂?pre-compression 涓?topic-segment summary 琛ㄥ緛 |
| `write_trigger` | 缂哄け | 褰撳墠娌℃湁 buffer-capacity / token-budget 椹卞姩瑙﹀彂鍣?|
| `organization` | 閮ㄥ垎澶嶇敤 | 鏈夊灞?append 澶栧３锛屼絾缂?sensory -> STM -> LTM 鐨勫垎灞傜紦鍐茬粍缁?|
| `evolution_trigger` | 閮ㄥ垎澶嶇敤 | 鍙媺寮鸿〃杈锯€滄柊鍐欏叆鍚庣淮鎶も€濓紝浣嗙己 sleep-time batch trigger |
| `memory_evolution` | 缂哄け | 褰撳墠娌℃湁 timestamp-constrained update queue + offline parallel update primitive |
| `retrieval` | 鐩存帴澶嶇敤 | `EmbeddingSimilarityRetrieval` 瓒充互琛ㄨ揪璁烘枃涓讳綋 recall |
| `readout` | 鐩存帴澶嶇敤 | 閫氱敤鏂囨湰 readout 瓒冲 |

### 閲嶈〃杈惧垽鏂?
鍙兘閮ㄥ垎鏄犲皠銆?
鍘熷洜涓嶅湪浜?slot 鏁伴噺涓嶅锛岃€屽湪浜?LightMem 鏈€鍏抽敭鐨勪袱娈垫満鍒跺綋鍓嶉兘娌℃湁鐪熷疄钀藉湴鎵胯浇鐗╋細

- 缂撳啿闃堝€奸┍鍔ㄧ殑涓夊眰 memory organization
- sleep-time 鐨?update queue 鏋勯€犱笌绂荤嚎骞惰鏇存柊

濡傛灉鍙敤鐜版湁妯″潡寮鸿鎷艰锛屾渶澶氳兘鍋氬嚭鈥滃璇濆啓鍏?+ 鎽樿 + 鍚戦噺妫€绱⑩€濈殑杩戜技鐗堬紝浣嗗緢闅炬妸 LightMem 鐨勬晥鐜囧鍚戞牳蹇冩満鍒跺畬鏁磋〃杈惧嚭鏉ャ€?
### 澶囨敞涓庤瘉鎹竟鐣?
- 璁烘枃鏄庤
  - LightMem 鐢?sensory memory銆乼opic-aware STM銆乻leep-time updated LTM 涓夐儴鍒嗙粍鎴愩€?  - 鏂拌緭鍏ュ厛缁?pre-compression锛屽啀杩涘叆 sensory buffer銆?  - sensory buffer 杈惧埌闃堝€煎悗瑙﹀彂鍩轰簬 attention + similarity 鐨?topic segmentation銆?  - STM 杈鹃槇鍊煎悗鎸?topic 绮掑害鎬荤粨锛屽舰鎴愯繘鍏?LTM 鐨勬潯鐩€?  - LTM 鏇存柊鍏?soft insert锛屽啀鍦?sleep time 鏋勯€犲甫鏃堕棿绾︽潫鐨勭浉浼?update queue锛屽苟绂荤嚎骞惰鏇存柊銆?  - retrieval stage 涓嶆槸璁烘枃涓昏浼樺寲瀵硅薄銆?- repo 瀹炵幇鍙‘璁?  - `src/lightmem/factory/memory_buffer/sensory_memory.py` 瀹炵幇浜嗘湁 token 涓婇檺鐨?sensory buffer锛屼互鍙?topic segmentation 鍚庡垏娈点€?  - `src/lightmem/factory/topic_segmenter/llmlingua_2.py` 瀹炵幇浜?attention-based boundary proposal銆?  - `src/lightmem/factory/memory_buffer/short_term_memory.py` 瀹炵幇浜?STM token 闃堝€艰Е鍙戙€?  - `src/lightmem/memory/lightmem.py` 涓?`construct_update_queue_all_entries(...)` 浼氫负鏉＄洰鍐欏叆 `update_queue`锛沗offline_update_all_entries(...)` 浼氬苟琛屾墽琛岀绾?update銆?  - `retrieve(...)` 鏄洿鎺ョ殑 embedding retrieval銆?- 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂?  - 鍦?MemPrimitive 涓紝pre-compression 鏇撮€傚悎浣滀负 `representation`锛岃€?topic-segment 鐢熷懡鍛ㄦ湡鏇撮€傚悎浣滀负 `organization`銆?  - sleep-time update 鍦?slot 涓婂簲鎷嗘垚 `evolution_trigger` + `memory_evolution` 涓ら儴鍒嗭紝鑰屼笉鏄杩涘崟涓€ organization side effect銆?- 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
  - repo 褰撳墠 extraction prompt 鏄庢樉姣旇鏂囨鏂囨洿鍋忊€滀簨瀹炴娊鍙栤€濓紱杩欐槸鍚︾瓑鍚屼簬璁烘枃涓殑 topic summary 鍏蜂綋瀹炵幇锛岃瘉鎹笉瓒炽€?  - repo 褰撳墠 `UPDATE_PROMPT` 鏆撮湶鐨勫姩浣滄槸 `update/delete/ignore`锛涜鏂囨娊璞″眰璇寸殑鏄?soft insert + offline update 妗嗘灦銆備袱鑰呭湪鍔ㄤ綔鏍囩涓婂苟闈炲畬鍏ㄤ竴涓€瀵瑰簲锛屼笉鑳借繃搴︾瓑鍚屻€?  - repo 2026 骞翠富鍒嗘敮宸茬粡鍖呭惈 StructMem銆乥aseline toolkit 绛夊悗缁墿灞曪紱杩欎簺涓嶅簲鍙嶆帹涓?LightMem 鍘熻鏂囩殑鏍稿績 memory 鏈哄埗銆?
## MIRIX: Multi-Agent Memory System for LLM-Based Agents

璁烘枃閾炬帴: <https://arxiv.org/abs/2507.07957>

瀹樻柟 repo: <https://github.com/MIRIX-AI/MIRIX>

鏈 repo 璇佹嵁涓昏鏍稿鐗堟湰: `MIRIX-AI/MIRIX` 榛樿鍒嗘敮锛屼复鏃舵鏌ユ椂閲嶇偣璇诲彇 `docs/ARCHITECTURE.md`銆乣mirix/functions/function_sets/memory_tools.py`銆佸悇 memory schema 涓?manager 瀹炵幇銆?
### 璁烘枃渚?memory 鏈哄埗閫熷啓

MIRIX 鐨?memory 鏈哄埗閲嶇偣锛屼笉鏄€滃崟涓€璁板繂搴撲笂鍐嶅彔涓€灞傛绱⑩€濓紝鑰屾槸鎶婇暱鏃惰蹇嗘樉寮忔媶鎴愬绫汇€佸啀浜ょ粰澶?agent 鍒嗗伐缁存姢銆傝鏂囦笌瀹樻柟鏂囨。涓€鑷村己璋冨叚绫?memory锛?
- `Core Memory`锛氶潰鍚?persona / human profile 鐨勯珮浼樺厛绾у潡鐘惰蹇?- `Episodic Memory`锛氫簨浠跺寲缁忓巻
- `Semantic Memory`锛氬叧浜庡疄浣撱€佸亸濂姐€佷簨瀹炪€佸叧绯荤殑姒傚康鎬х煡璇?- `Procedural Memory`锛氬彲澶嶇敤鐨勬搷浣滄楠や笌宸ヤ綔娴?- `Resource Memory`锛氭枃妗ｃ€佺綉椤点€佹枃浠剁瓑璧勬簮鍐呭
- `Knowledge Vault`锛氬甫鏁忔劅绾у埆鐨勬満瀵嗘垨鍙楅檺淇℃伅

瀵瑰簲鐨勫啓鍏ラ摼璺槸锛?
- 鍏堢敱 `Meta Memory Manager Agent` 璇诲彇绱Н瀵硅瘽锛屽垽鏂繖娆′氦浜掑簲鏇存柊鍝簺 memory types
- 鍐嶇敱鍚?memory-type 涓撶敤 agent 浜у嚭瀵瑰簲 schema 鐨?memory item锛屽苟鍐冲畾鏄彃鍏ャ€佹洿鏂般€佸悎骞惰繕鏄烦杩?- 鏈€鍚庣敱鍚?memory manager 鎸佷箙鍖栵紝骞跺湪闇€瑕佹椂閲嶇畻 embedding

瀵瑰簲鐨勭淮鎶や笌璇诲彇閾捐矾鏄細

- `Reflexion Agent` 璐熻矗璺?memory types 鐨勫幓閲嶃€佹暣鐞嗐€佷粠 episode 涓彁鐐兼洿楂樺眰 pattern
- 鏌ヨ鏃舵寜 memory type 鍒嗗埆妫€绱紝鍐嶆妸涓嶅悓绫诲瀷鐨勭粨鏋滄寜绫诲瀷鍒嗙粍娉ㄥ叆 system prompt
- 妫€绱㈠苟涓嶆槸缁熶竴鐨勫崟搴?recall锛岃€屾槸澶?store銆佸瓧娈垫劅鐭ャ€佺被鍨嬫劅鐭ョ殑 BM25 / embedding / fuzzy match 缁勫悎

濡傛灉纭槧灏勫埌 MemPrimitive 褰撳墠 slot 浣撶郴锛孧IRIX 鐨勬牳蹇冩柊鎰忎富瑕佸帇鍦ㄥ洓澶勶細

- `write_trigger`锛氫笉鏄€滃啓涓嶅啓鈥濅簩鍊硷紝鑰屾槸鈥滀竴娆′氦浜掕璺敱鍒板摢浜?memory types鈥?- `organization`锛氫笉鏄崟涓€ layer append锛岃€屾槸寮傛瀯澶氳蹇嗗簱骞跺瓨
- `memory_evolution`锛氫笉鏄畝鍗曡拷鍔狅紝鑰屾槸绫诲瀷鍖?update / merge / rewrite / dedupe
- `retrieval/readout`锛氫笉鏄崟涓€鍙洖鍣紝鑰屾槸澶?store 妫€绱㈠悗鍐嶆寜 memory type 缁勭粐璇诲嚭

### 鎸?MemPrimitive slot 鐨勬媶瑙?
#### unit_formation

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鐨勪笂娓歌緭鍏ュ苟涓嶆槸鍗曞彞浜嬪疄锛岃€屾槸鈥滅疮绉埌褰撳墠鏃跺埢鐨勪竴娈典氦浜掍笂涓嬫枃鈥濄€俙Meta Memory Manager Agent` 浼氳鍙?accumulated messages / recent interaction锛屽啀鍐冲畾鍝簺 memory types 闇€瑕佹洿鏂般€?  - 杩涘叆涓撶敤 memory agent 涔嬪墠锛屾渶灏忓彲鎰熺煡鍗曞厓鏇村儚鈥渃onversation bundle / interaction chunk鈥濓紝鑰屼笉鏄函鏂囨湰鍙ュ瓙鍒囩墖銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - `PassThroughUnitFormation`
    - 濡傛灉涓婃父宸茬粡鎶婁竴娈靛璇濅笂涓嬫枃鎵撳寘鎴愬崟涓?`Observation`锛屽彲浠ュ媺寮烘壙杞?MIRIX 鐨勮緭鍏ュ瑙傘€?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `MetadataHintUnitFormation`
    - 鍙€熷姪 hints 浜哄伐琛ヤ笂浼氳瘽杈圭晫鎴栨潵婧愭爣璁帮紝浣嗗畠涓嶅師鐢熻〃杈锯€滅疮绉氦浜掑寘鈥濊繖涓€杈撳叆璇箟銆?  - `WindowedUnitFormation`
    - 鍙兘鍋氬眬閮ㄧ獥鍙ｅ垏鐗囷紝涓嶈兘绋冲畾琛ㄨ揪 MIRIX 閭ｇ闈㈠悜 memory-router 鐨勪氦浜掓墦鍖呰竟鐣屻€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鏄惧紡鐨勨€渃onversation bundle / interaction chunk鈥濆舰鎴?primitive銆?  - 缂哄皯鎶婂杞枃鏈€侀檮浠跺紩鐢ㄣ€佹椂闂磋寖鍥淬€佸弬涓庤€呬俊鎭竴璧峰皝瑁呬负鍚庣画 typed-memory extraction 杈撳叆鐨勬爣鍑?contract銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歁IRIX 鐢?MetaAgent 鏍规嵁浜や簰鍐呭鍐冲畾鏇存柊鍝簺 memory types銆?  - repo 瀹炵幇鍙‘璁わ細`meta_memory_agent.txt` 鐩存帴瑕佹眰 agent 浠?accumulated messages 鍒ゆ柇搴旀洿鏂扮殑涓€缁?memory types銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細瀵?MemPrimitive 鏉ヨ锛岃繖閲岀殑鑷劧杈撳叆鍗曞厓涓嶆槸鍙ュ瓙锛岃€屾槸鈥滃緟璺敱鐨勪竴娈典氦浜掑寘鈥濄€?
#### representation

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鐨勮〃绀轰笉鏄粺涓€鐨?`summary + embedding`锛岃€屾槸鍏堢粡鐢变笓鐢?agent 鎶芥垚涓嶅悓 memory family 鐨?typed payload銆?  - 渚嬪 episodic memory 浼氬舰鎴愪簨浠舵憳瑕併€佺粏鑺傘€佸弬涓庤€呫€佹椂闂达紱semantic memory 浼氬舰鎴愬疄浣撳悕銆佹憳瑕併€佺粏鑺傘€佹潵婧愶紱procedural memory 浼氬舰鎴愭楠ゅ寲娴佺▼锛況esource memory 浼氬舰鎴愭爣棰樸€佹憳瑕併€佹鏂囷紱knowledge vault 浼氬舰鎴愭晱鎰熷害涓庡瘑閽ュ瀷鍐呭銆?  - 澶氭暟 memory types 鍦ㄨ惤搴撴椂杩樹細琛?embedding锛屼絾 embedding 鍙槸 typed representation 鐨勯檮灞烇紝涓嶆槸涓讳綋銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BasicRepresentation`
    - 鍙彁渚?`summary`銆乣embedding`銆乣keywords` 绛夐€氱敤澧炲己锛屼絾涓嶈兘浜у嚭 MIRIX 鎵€闇€鐨勫 schema typed memory payload銆?  - `RetrievalOrientedEmbeddingRepresentation`
    - 鍙鐩栤€滆〃绀哄悗闄?embedding 渚涙绱⑩€濈殑灞€閮ㄩ渶姹傦紝浣嗕笉璐熻矗鎶婁氦浜掓娊鎴?episodic / semantic / procedural / resource / vault 杩欎簺涓嶅悓缁撴瀯銆?  - `SemanticFieldEnrichmentRepresentation`
    - 鍙槸鍦ㄧ粺涓€璁板綍涓婅ˉ璇箟瀛楁锛屼笉鏄寜 memory family 鍒嗗寲 schema銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥滀竴娆′氦浜?-> 澶氱 memory schema 鍊欓€夆€濈殑 typed extraction primitive銆?  - 缂哄皯瀵?block memory銆乪vent memory銆乧oncept memory銆乸rocedure memory銆乺esource memory銆乻ensitive secret memory 鐨勭粺涓€琛ㄧず杈圭晫銆?  - 缂哄皯鈥滀笉鍚?memory type 浣跨敤涓嶅悓瀛楁闆嗕笌鍚庣画绱㈠紩瀛楁鈥濈殑 representation contract銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歁IRIX 浣跨敤澶氱被 memory锛屽苟鐢变笉鍚?memory agents 绠＄悊銆?  - repo 瀹炵幇鍙‘璁わ細鍚?schema 鏂囦欢鍒嗗埆瀹氫箟浜?episodic / semantic / procedural / resource / knowledge vault 鐨勫瓧娈电粨鏋勶紝涓斿伐鍏峰眰瀛樺湪鍒?memory type 鐨?insert/update 鎺ュ彛銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細MIRIX 鍦?slot 涓婃洿鎺ヨ繎鈥渢yped schema extraction representation鈥濓紝鑰屼笉鏄崟涓€閫氱敤琛ㄧず澧炲己銆?
#### write_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鐨勫啓瑙﹀彂鏍稿績涓嶆槸绠€鍗曞垽鏂€滃啓涓嶅啓鈥濓紝鑰屾槸鐢?MetaAgent 瀵瑰綋鍓嶄氦浜掑仛澶氭爣绛捐矾鐢憋紝鍐冲畾鏈搴旀洿鏂板摢浜?memory types銆?  - 褰撴煇涓?memory type 琚€変腑鍚庯紝杩樹細杩涘叆璇ョ被鍨嬩笓鐢?agent 鐨勫悗缁垽鏂笌鏇存柊銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `LLMJudgedWriteTrigger`
    - 鍙互鍊?LLM 鍋氣€滄槸鍚﹀€煎緱鍐欌€濆垽鏂紝浣嗗畠娌℃湁 MIRIX 鎵€闇€鐨?one-to-many memory-type routing 杈撳嚭銆?  - `MetadataGatedWriteTrigger`
    - 濡傛灉涓婃父宸插啓濂界洰鏍?memory types锛屽彲鎸?metadata 杩囨护锛涗絾 MIRIX 鑷韩鍏抽敭鍦ㄤ簬瑙﹀彂鍣ㄥ唴閮ㄤ骇鍑鸿矾鐢辩粨鏋滐紝鑰屼笉鏄秷璐规棦鏈夋爣绛俱€?  - `AlwaysTrigger`
    - 鍙兘琛ㄨ揪鈥滃叏閮ㄩ€佸叆鍚庣画娴佺▼鈥濓紝鏃犳硶琛ㄨ揪鈥滃彧鏇存柊 episodic + semantic锛屼笉鏇存柊 resource / vault鈥濄€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 multi-label銆乼ype-selective 鐨?write trigger銆?  - 缂哄皯鈥滃綋鍓嶄氦浜?-> 鐩爣 memory type 闆嗗悎鈥濈殑瑙﹀彂缁撴灉鏍煎紡銆?  - 缂哄皯鎶婅矾鐢卞喅绛栫户缁紶缁欏悇 store / agent 鐨勬樉寮忔帶鍒惰竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歁eta Memory Manager Agent 鍐冲畾 memory update 鐨勭被鍨嬪垎鍙戙€?  - repo 瀹炵幇鍙‘璁わ細`trigger_memory_update` 浼氬厛鏀堕泦 memory types锛屽啀骞惰瑙﹀彂瀵瑰簲瀛?agent銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細鍦?MemPrimitive slot 閲岋紝杩欏簲钀藉湪 `write_trigger`锛屼笖鐜版湁瑙﹀彂鍣ㄥ彧瑕嗙洊鈥滄槸鍚﹀啓鈥濓紝鏈鐩栤€滃啓鍒板摢浜?type鈥濄€?
#### organization

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 涓嶆槸鍗曚竴 memory table锛岃€屾槸鍏被寮傛瀯 memory store 骞跺瓨銆?  - 鍏朵腑 `Core Memory` 杩樻槸鍧楃姸 memory block 绠＄悊锛涘叾浠栧嚑绫绘洿鍍?typed record store锛屼絾瀛楁銆佺储寮曞煙銆佸彲鏇存柊鏂瑰紡閮戒笉鍚屻€?  - 涓嶅悓 memory 杩樺甫鏈?scope / owner / sensitivity 绛夎繃婊ょ淮搴︼紝灏ゅ叾 knowledge vault 鏈夐澶栬闂竟鐣屻€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ConditionalLayerOrganization`
    - 鍙繎浼艰〃杈锯€滄寜绫诲埆鍒嗗眰鈥濓紝浣嗘棤娉曡〃杈?MIRIX 杩欑 heterogeneous stores with distinct schemas 鐨勭粍缁囨柟寮忋€?  - `AppendOrganization`
    - 鍙兘鎵胯浇鍗曚竴 record append锛屼笉瓒充互琛ㄧず block memory 涓?typed stores 骞跺瓨銆?  - `PlacementWithoutAppendOrganization`
    - 鍙〃杈捐矾鐢?钀戒綅澶栬锛屼絾娌℃湁 MIRIX 鎵€闇€鐨勫疄闄呭紓鏋勫瓨鍌ㄨ涔夈€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯鈥滃紓鏋勫璁板繂搴撶粍缁団€?primitive銆?  - 缂哄皯 block-style core memory 涓?record-style typed memories 骞跺瓨鐨勭粍缁囨娊璞°€?  - 缂哄皯 per-type schema銆乻cope銆乻ensitivity銆乷wner constraints 鐨勭粍缁囪竟鐣屻€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歁IRIX 閲囩敤澶氱 memory types銆?  - repo 瀹炵幇鍙‘璁わ細`ARCHITECTURE.md`銆乻chemas銆乵emory managers 閮芥樉绀哄叚绉?memory store 鍒嗙锛沜ore memory 閫氳繃 blocks 绠＄悊锛宬nowledge vault 甯?sensitivity銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細MemPrimitive 褰撳墠 layer 姒傚康鍙兘閮ㄥ垎杩戜技 MIRIX 鐨?store 鍒嗗寲锛屼笉鑳藉繝瀹炴壙杞藉叾寮傛瀯缁勭粐銆?
#### evolution_trigger

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鐨勭淮鎶よЕ鍙戜笉鏄崟涓€妯″紡锛岃€屾槸鑷冲皯鏈変笁绫伙細
  - `Core Memory` 鍦ㄦ帴杩戝閲忎笂闄愭椂瑙﹀彂 rewrite / condense銆?  - 鍚?typed memories 鍦ㄦ娴嬪埌閲嶅銆侀噸鍙犳垨鍚屼竴瀵硅薄鏃惰Е鍙?update / merge / replace銆?  - `Reflexion Agent` 杩樹細鍦ㄩ澶栨椂鏈哄鎵€鏈?memory types 鍋氬幓閲嶃€佹暣鐞嗕笌楂樺眰 pattern 鎻愮偧銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `ThresholdTrigger`
    - 鍙繎浼艰〃杈锯€滆揪鍒伴槇鍊煎悗瑙﹀彂鈥濓紝浣嗙幇鏈夎Е鍙戜俊鍙蜂笉鏄?core block fullness銆?  - `NewWriteEvolutionTrigger`
    - 鍙〃杈锯€滄柊鍐欏叆鍚庡仛缁存姢鈥濓紝浣?MIRIX 涓緢澶氱淮鎶ゆ槸 type-aware update / merge锛岃€岄潪缁熶竴鍚庡鐞嗐€?  - `OutcomeConditionedEvolutionTrigger`
    - 瀵?Reflexion 绫昏Е鍙戞湁涓€鐐瑰褰㈢浉浼硷紝浣?MIRIX Reflexion 涓嶆槸璇曢敊鍙嶉瑙﹀彂锛岃€屾槸闈㈠悜 memory hygiene 鐨勬暣鐞嗚Е鍙戙€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯瀹归噺椹卞姩鐨?core-block rewrite trigger銆?  - 缂哄皯绫诲瀷鍖?memory update / merge trigger銆?  - 缂哄皯 scheduled / per-query reflexion-style maintenance trigger銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛氱郴缁熷寘鍚?Reflexion agent锛屽苟寮鸿皟澶?memory maintenance銆?  - repo 瀹炵幇鍙‘璁わ細core tools 閲屽瓨鍦?rewrite block锛涘悇 memory tools 瀛樺湪 insert/update/merge锛沗reflexion_agent.txt` 鏄庣‘鍐欎簡 cleanup / dedup / pattern extraction銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細MIRIX 鐨?evolution trigger 鏄庢樉鏄婧愯Е鍙戞棌锛岃€屼笉鏄綋鍓嶅崟涓€闃堝€兼垨鍗曚竴鏂板啓鍏ヨЕ鍙戝彲瑕嗙洊銆?
#### memory_evolution

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鐨?memory evolution 鍖呮嫭澶氱被 type-specific 缁存姢鍔ㄤ綔锛?  - core memory 浼氬仛 block rewrite / condense銆?  - episodic memory 浼?merge / replace 浜嬩欢鏉＄洰銆?  - semantic / procedural / resource / knowledge vault 浼氬宸叉湁鏉＄洰鍋?update锛屼笖浼氳烦杩囬噸澶嶉」銆?  - Reflexion 杩樹細鍋氳法 memory types 鐨?dedupe銆佹暣鐞嗭紝浠ュ強浠?episodic 涓彁鐐?lifestyle / behavior pattern 鍐欏洖鏇撮珮灞?memory銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `SummaryRewriteEvolution`
    - 涓?core block rewrite 鏈夎〃闈㈢浉浼硷紝浣嗗畠涓嶆槸 block-capacity-conditioned rewrite锛屼篃涓嶉潰鍚?persona/human blocks銆?  - `ReflectionGenerationEvolution`
    - 瀵?Reflexion 寮忊€滀粠缁忓巻鐢熸垚鏇撮珮灞傛礊瑙佲€濇湁灞€閮ㄧ浉浼硷紝浣嗕笉鏀寔璺?store dedupe 涓?typed write-back銆?  - `AppendOnlyEvolution`
    - 鍙兘琛ㄨ揪鏂板锛屾棤娉曡〃杈?MIRIX 鐨?update / merge / replace / dedupe 涓讳綋銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 typed memory update / merge / replace primitive銆?  - 缂哄皯璺?memory stores 鐨?deduplication / cleanup primitive銆?  - 缂哄皯鈥滀粠 episodic pattern 鎻愮偧 semantic insight 骞跺啓鍥炩€濈殑璺ㄧ被鍨嬫紨鍖栬兘鍔涖€?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歁IRIX 閫氳繃澶氱被 memory 鍗忓悓鑾峰緱鏇村己闀挎湡璁板繂绠＄悊銆?  - repo 瀹炵幇鍙‘璁わ細memory tools 鎻愪緵鍒嗙被鍨?insert/update/merge/check锛汻eflexion prompt 鏄庣‘瑕佹眰 dedup銆乧leanup銆乸attern extraction銆?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細MIRIX 鐨勬牳蹇冪淮鎶よ涔夎惤鍦?typed evolution锛岃€屼笉鏄綋鍓?append/rewrite/graph-link 瀹舵棌鑳藉畬鏁磋鐩栫殑鑼冨洿銆?
#### retrieval

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鏌ヨ鏃朵笉鏄崟涓€鍚戦噺鍙洖锛岃€屾槸鎸?memory type 鍒嗗埆妫€绱€?  - episodic memory 浼氬悓鏃跺尯鍒?recent events 涓?relevant events銆?  - 鍏朵粬 memory types 浣跨敤瀛楁鎰熺煡鐨?BM25 / embedding / fuzzy matching锛沰nowledge vault 杩樺甫鏁忔劅绾у埆杩囨护锛岄伩鍏嶄笉璇ユ毚闇茬粰褰撳墠 agent 鐨勭瀵嗚璇诲嚭銆?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `BM25Retrieval`
    - 鍙鐩?MIRIX 妫€绱腑鐨勪竴鏉′富璺緞锛屼絾鍙潰鍚戠粺涓€璁板綍璇箟銆?  - `EmbeddingSimilarityRetrieval`
    - 鍙鐩栧悜閲忔绱㈤儴鍒嗭紝浣嗕笉琛ㄨ揪 per-type fields銆乺ecent-vs-relevant dual view 涓?sensitivity constraints銆?  - `LayerAwareRetrieval`
    - 鍙仛澶氬眰璺敱澶栧３锛屼絾绂?MIRIX 鐨勫 store銆佸瓧娈垫劅鐭ユ绱?orchestration 浠嶆湁鏄庢樉璺濈銆?  - `RecencyRetrieval`
    - 浠呰兘瑕嗙洊 episodic recent branch 鐨勪竴閮ㄥ垎銆?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯璺ㄥ紓鏋?stores 鐨勮仈鍚堟绱?orchestrator銆?  - 缂哄皯 per-type searchable fields銆乭ybrid search policy 涓?sensitivity filtering銆?  - 缂哄皯鈥渞ecent episodic + relevant episodic + other typed memories鈥濆苟琛屽彫鍥炵殑缁熶竴鎺ュ彛銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛歁IRIX 鐢ㄥ绫?memory 鎻愪緵鏇村畬鏁翠笂涓嬫枃銆?  - repo 瀹炵幇鍙‘璁わ細鍚?manager 鏀寔 BM25 / embedding / fuzzy 绛変笉鍚屾绱紱agent 鏋?prompt 鏃跺垎鍒媺鍙?episodic recent / relevant 涓庡叾浠?memory types锛沰nowledge vault 浼氶檺鍒舵晱鎰熷唴瀹规毚闇层€?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細褰撳墠 MemPrimitive 鐨?retrieval family 鍙鐢ㄩ儴鍒嗗眬閮ㄦ绱㈠櫒锛屼絾缂哄皯 MIRIX 椋庢牸鐨勫 store orchestrator銆?
#### readout

- 璁烘枃閲屽仛浜嗕粈涔?  - MIRIX 鐨勮鍑轰笉鏄畝鍗曟嫾鎺ュ叏閮ㄥ懡涓枃鏈紝鑰屾槸鎸?memory type 鍒嗙粍瑁呴厤鍒?system prompt銆?  - episodic memory 浼氬垎 recent 涓?relevant 涓ゅ潡鍛堢幇锛涘叾浠?memory 鍒欐寜绫诲瀷鍒楀嚭锛涢儴鍒嗗満鏅繕浼氫繚鐣?item id 渚涘悗缁洿鏂般€?  - knowledge vault 鐨勮鍑哄彈鍙鎬т笌鏁忔劅搴︾害鏉熴€?- MemPrimitive 鐜版湁鍝簺妯″潡鍙洿鎺ュ鐢?  - 鏃犲彲鐩存帴瀹屾暣澶嶇敤妯″潡銆?- 鍝簺妯″潡鍙兘閮ㄥ垎澶嶇敤
  - `GroupedByLayerReadout`
    - 鍙繎浼艰〃杈锯€滃垎缁勫憟鐜扳€濓紝浣嗙粍鐨勮涔夊彧鏄?layer锛屼笉鏄?MIRIX 鐨?typed memory families銆?  - `PromptContextReadout`
    - 鍙壙杞芥渶缁?prompt 娉ㄥ叆澶栬锛屼絾涓嶈嚜甯?MIRIX 閭ｇ recent/relevant split 涓庢晱鎰熶俊鎭姂鍒躲€?  - `ConcatenateReadout`
    - 鍙兘鍋氭渶绮楃矑搴︽嫾鎺ャ€?- 褰撳墠缂哄け浠€涔堣兘鍔?  - 缂哄皯 typed memory prompt assembly primitive銆?  - 缂哄皯 episodic recent/relevant 鍙屽尯鍧?readout銆?  - 缂哄皯涓?sensitivity / owner filtering 鍗忓悓鐨?readout contract銆?- 浣犵殑鍒ゆ柇渚濇嵁鏄粈涔堬紙璁烘枃 / repo / 鎺ㄦ柇锛?  - 璁烘枃鏄庤锛氱郴缁熷湪鎺ㄧ悊鏃惰皟鐢ㄤ笉鍚?memory 浠ュ寮轰笂涓嬫枃銆?  - repo 瀹炵幇鍙‘璁わ細`build_system_prompt_with_memories` 浼氭寜 memory type 鏋勯€犱笉鍚岀墖娈碉紝骞跺 knowledge vault 鍋氶檺鍒躲€?  - 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂細瀵?MemPrimitive 鑰岃█锛宺eadout 涓嶆槸闆堕毦鐐癸紝浣嗗畠鏇村儚 retrieval 涔嬪悗鐨?typed formatting 缂哄彛銆?
### 涓?MemPrimitive 鐜版湁缁勪欢鐨勫鐓х粨璁?
| slot | 缁撹 | 璇存槑 |
| --- | --- | --- |
| `unit_formation` | 閮ㄥ垎澶嶇敤 | `PassThroughUnitFormation` 鍙媺寮烘壙杞介鎵撳寘浜や簰锛屼絾缂?conversation bundle contract |
| `representation` | 缂哄け | 褰撳墠娌℃湁鈥滀竴娆′氦浜?-> 澶?memory schema typed payload鈥濇ā鍧?|
| `write_trigger` | 缂哄け | 褰撳墠娌℃湁澶氭爣绛?memory-type routing trigger |
| `organization` | 閮ㄥ垎澶嶇敤 | 鍙敤 layer/router 澶栧３杩戜技锛屼絾缂哄紓鏋勫 store 涓?core-block 缁勭粐 |
| `evolution_trigger` | 閮ㄥ垎澶嶇敤 | 鏈夐槇鍊?鏂板啓鍏?缁撴灉椹卞姩瑙﹀彂澶栧３锛屼絾缂?capacity-triggered rewrite 涓?reflexion scheduling |
| `memory_evolution` | 缂哄け | 褰撳墠娌℃湁 typed update/merge/replace/dedupe 璺?store 缁存姢 primitive |
| `retrieval` | 閮ㄥ垎澶嶇敤 | BM25/embedding/recency 鍙眬閮ㄥ鐢紝浣嗙己澶?store銆佸瓧娈垫劅鐭ャ€佹晱鎰熷害绾︽潫 orchestrator |
| `readout` | 閮ㄥ垎澶嶇敤 | 鐜版湁 prompt/group readout 鍙仛杩戜技锛屼絾缂?typed prompt assembly 涓?episodic dual-view readout |

### 閲嶈〃杈惧垽鏂?
鍙兘閮ㄥ垎鏄犲皠銆?
鍘熷洜涓嶅湪浜?MIRIX 鏃犳硶鏀捐繘鍏?slot 妗嗘灦锛岃€屽湪浜庡畠鐨勫叧閿満鍒跺苟涓嶆槸鍗曠偣缂哄彛锛岃€屾槸鏁存潯鈥滃绫诲瀷璺敱 -> 寮傛瀯缁勭粐 -> 绫诲瀷鍖栫淮鎶?-> 澶?store 妫€绱?璇诲嚭鈥濋摼鏉￠兘姣斿綋鍓嶆ā鍧楁棌鏇村己锛?
- 鍓嶅崐娈电己 multi-memory-type routing trigger
- 涓缂?heterogeneous memory-store organization
- 鍚庡崐娈电己 typed update / dedupe / cross-store reflexion evolution
- recall 渚ц櫧鑳藉鐢ㄩ儴鍒?BM25 / embedding / recency primitive锛屼絾绂?MIRIX 鐨?typed orchestration 浠嶆湁鏄庢樉宸窛

鍥犳锛屽綋鍓?MemPrimitive 鍙互琛ㄨ揪 MIRIX 鐨勪竴浜涘眬閮ㄦ€濇兂锛屼緥濡傦細

- 鐢?layer-aware 璺敱杩戜技澶?memory categories
- 鐢?embedding / BM25 recall 杩戜技鑻ュ共鍗?store 妫€绱?- 鐢?summary/rewrite/reflection 灞€閮ㄨ繎浼煎皯鏁扮淮鎶ゅ姩浣?
浣嗚繕涓嶈兘鎶?MIRIX 蹇犲疄閲嶈〃杈炬垚涓€涓€滃彧闈犵幇鏈夋ā鍧楃粍鍚堝嵆鍙垚绔嬧€濈殑瀹屾暣绯荤粺銆?
### 澶囨敞涓庤瘉鎹竟鐣?
- 璁烘枃鏄庤
  - MIRIX 閲囩敤 multi-agent memory architecture銆?  - 绯荤粺缁存姢鍏被 memory锛屽苟鐢ㄤ笓闂?agent 鍗忓悓绠＄悊銆?  - 绯荤粺杩樺寘鍚?Reflexion 鏈哄埗浠ユ彁鍗囬暱鏈熻蹇嗙鐞嗚川閲忋€?- repo 瀹炵幇鍙‘璁?  - `ARCHITECTURE.md` 鏄庣‘鍒楀嚭鍏被 memory 涓?MetaAgent / Reflexion Agent銆?  - 鍚?schema 涓?manager 鏄庣‘鏄剧ず episodic / semantic / procedural / resource / knowledge vault 鐨勭湡瀹炲瓧娈点€佺储寮曚笌妫€绱㈡柟娉曘€?  - `memory_tools.py` 鏄庣‘瀛樺湪鎸?memory type 鐨?insert / update / merge / rewrite / dedupe 鐩稿叧宸ュ叿鍏ュ彛銆?  - `agent.py` 涓殑 prompt 鏋勯€犻€昏緫纭疄鎸?memory type 璇诲彇锛屽苟鎶?episodic recent / relevant 鍒嗗紑鍛堢幇銆?- 渚濇嵁璁烘枃涓?repo 鍋氬嚭鐨勫悎鐞嗘帹鏂?  - 鍦?MemPrimitive slot 鏄犲皠涓紝MetaAgent 鐨勨€滃喅瀹氬啓鍏ュ摢浜?memory types鈥濇渶鑷劧钀藉湪 `write_trigger` 鑰屼笉鏄?`organization`銆?  - MIRIX 鐨勪富瑕佺己鍙ｄ笉鍦ㄥ崟涓?retriever锛岃€屽湪 typed routing 涓?heterogeneous organization銆?  - Reflexion 鍦?slot 涓婃洿閫傚悎浣滀负 `evolution_trigger + memory_evolution` 鐨勭粍鍚堬紝鑰屼笉鏄崟鐙柊澧為《灞?slot銆?- 褰撳墠璇佹嵁涓嶈冻銆佷笉鑳戒笅缁撹鐨勭偣
  - 璁烘枃姝ｆ枃涓?repo 鏂囨。閮芥病鏈夋妸鈥滀氦浜掑寘鐨勭簿纭畾涔夆€濆舰寮忓寲鍒板彲鐩存帴绛変环涓烘煇涓?unit schema锛屾晠 `unit_formation` 鐨勭粏绮掑害杈圭晫浠嶆湁瑙ｉ噴绌洪棿銆?  - repo 鏌愪簺妫€绱㈣矾寰勫綋鍓嶉粯璁ゆ洿鍋?BM25锛岃€岃鏂囧眰闈㈢殑 memory 浣跨敤鎻忚堪鏇撮珮灞傦紱涓嶈兘鎹鍙嶆帹鈥滆鏂囧彧涓诲紶 BM25鈥濄€?  - Reflexion 鐨勭湡瀹炵敓浜цЕ鍙戦鐜囥€佽皟搴︾瓥鐣ャ€佷笌鍦ㄧ嚎鎺ㄧ悊鐨勪弗鏍兼椂搴忚竟鐣岋紝鍦ㄥ綋鍓嶅叕寮€鏉愭枡涓瘉鎹笉澶熷厖鍒嗭紝涓嶈兘鍐欐涓哄敮涓€鏈哄埗銆?