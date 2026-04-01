# 璁″垝鏂板MODULE

## unit_formation

### `DialogueTurnPairFormation`

- 灞炰簬鍝釜 slot
  - `unit_formation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶婂閲忓埌鏉ョ殑 user/assistant 涓€杞氦浜掔ǔ瀹氱粍缁囨垚涓€涓?turn-level memory unit锛屽苟淇濈暀 speaker 瀵归綈銆乼urn 杈圭晫銆佹椂闂存埑涓庡悗缁?buffer 璁℃暟鎵€闇€瀛楁銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `PassThroughUnitFormation` 鍙兘鎵胯浇涓婃父宸茬粡鎵撳寘濂界殑鏂囨湰銆?  - `MetadataHintUnitFormation` 鍙互闈?hints 浜哄伐鏋勯€犲崟鍏冿紝浣嗘病鏈?LightMem 鎵€闇€鐨勫璇?turn 璇箟锛屼篃涓嶅拰 buffer 瀹归噺鏈哄埗鑷劧瀵规帴銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?conversational memory 鏂规硶閮介渶瑕佹妸鈥滃崟鏉℃秷鎭€濇彁鍗囦负鈥滃崟杞氦浜掆€濅綔涓虹湡瀹炲啓鍏ュ崟鍏冦€?
### `ConversationBundleFormation`

- 灞炰簬鍝釜 slot
  - `unit_formation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶婁竴娈靛緟澶勭悊鐨勪氦浜掍笂涓嬫枃绋冲畾灏佽鎴?`conversation bundle / interaction chunk`锛屾樉寮忎繚鐣欐秷鎭垪琛ㄣ€佸弬涓庤€呫€佹椂闂磋寖鍥淬€侀檮浠?璧勬簮寮曠敤銆佹潵婧?agent 绛変俊鎭紝渚涘悗缁?memory-type 璺敱涓?typed extraction 浣跨敤銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `PassThroughUnitFormation` 鍙兘鎵胯浇涓婃父宸叉墦鍖呭ソ鐨勬枃鏈€?  - `WindowedUnitFormation` 鍙兘鍋氱獥鍙ｅ垏鍒嗐€?  - 褰撳墠娌℃湁妯″潡鎶娾€滃杞氦浜掑寘鈥濅綔涓轰竴绛?unit contract 鏆撮湶鍑烘潵锛屽洜姝ゆ棤娉曞繝瀹炴壙杞?MIRIX 杩欑被鍏堝仛 memory-type routing 鍐嶅仛 typed extraction 鐨勫啓鍏ラ摼璺€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄?conversational / multi-agent memory 绯荤粺锛岄兘鍙兘鍏堝涓€娈典氦浜掑寘鍋氳矾鐢卞拰鎶藉彇锛岃€屼笉鏄€愬彞鐩存帴鍐欏簱銆?
## representation

### `NERConditionedOpenIERepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵?passage 鍏堝仛 named entity extraction锛屽啀鎶?named entities 浣滀负绾︽潫鏉′欢鍋氱浜岄樁娈?triple extraction锛岀ǔ瀹氫骇鍑?entity/noun-phrase銆乼riples銆佷互鍙婂悗缁浘缁勭粐鎵€闇€鐨勪腑闂磋〃绀恒€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation` 铏界劧鑳戒骇鍑?`entities`/`triples`/`embedding`锛屼絾娌℃湁鈥滃厛 NER 鍐嶅彈 NER 绾︽潫鐨?OpenIE鈥濊繖涓€涓ら樁娈垫満鍒讹紝涔熸病鏈夋樉寮忔妸鍥捐妭鐐圭骇琛ㄧず杈圭晫绋冲畾鏆撮湶鍑烘潵銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?graph-memory / OpenIE-memory 绯荤粺閮介渶瑕佲€滃彈绾︽潫鐨勭粨鏋勬娊鍙栤€濊€屼笉鏄竴娆℃€ф澗鏁ｈ〃绀恒€?
### `ObservationTripletExtractionRepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶?step-level observation 绋冲畾瑙ｆ瀽涓?world-fact triplets锛屼綔涓哄悗缁?semantic graph 鍐欏叆涓庢绱㈢殑鏍囧噯涓棿琛ㄧず銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation` 铏界劧鍙互澹版槑 `triple` 鍏冪礌锛屼絾褰撳墠浠撳簱閲岀殑 triplet extraction 浠嶄笉搴旇瑙嗕负涓?AriGraph 鍚岃涔夌殑 observation-conditioned triplet parser锛涘畠涔熸病鏈夋妸鈥滀緵 semantic graph 缁勭粐/淇浣跨敤鐨?triplet payload鈥濅綔涓烘槑纭?contract 鏆撮湶鍑烘潵銆?- 杩欐槸涓?AriGraph 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄€滀氦浜?observation -> 缁撴瀯鍖栦簨瀹炲浘鈥濈殑 memory 绯荤粺閮藉彲鑳藉鐢ㄣ€?
### `ConversationContextSalientFactRepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 浠モ€滃綋鍓嶆秷鎭 + recent-message window + 鍙€?conversation summary鈥濅负杈撳叆锛屾娊鍙?candidate facts / candidate memories锛屾湇鍔?Mem0 杩欑被涓婁笅鏂囨劅鐭ョ殑浜嬪疄鎶藉彇銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation` 鍙仛 unit-local 鐨?text/embedding/entities/triples/summary 澧炲己锛屼笉浼氭妸澶氳疆瀵硅瘽涓婁笅鏂囪瀺鍚堟垚涓€缁勫緟缁存姢鐨?candidate facts銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?conversational memory 绯荤粺閮介渶瑕佲€滀粠鏂颁氦浜掍腑鎶藉€煎緱璁颁綇鐨勪簨瀹炩€濓紝鑰屼笉鏄彧瀵瑰崟鏉℃枃鏈仛琛ㄧず澧炲己銆?
### `TypedEntityRelationExtractionRepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 浠庡璇濇枃鏈腑鍏堟娊瀹炰綋鍙婂疄浣撶被鍨嬶紝鍐嶆娊鍏崇郴 triplets锛屼骇鍑哄彲鐩存帴閫佸叆鍥剧粍缁囦笌鍥炬洿鏂扮殑 typed graph payload銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation(elements=("entities", "triple"))` 鍙粰鍑烘澗鏁ｅ疄浣?涓夊厓缁勫瑙傦紝娌℃湁 graph-memory 鎵€闇€鐨勫疄浣撶被鍨嬭竟鐣岋紝涔熸病鏈夌ǔ瀹氱殑 graph payload contract銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄浘璁板繂銆佺煡璇嗗浘璋卞紡 memory銆佸叧绯绘绱㈠紡 memory 閮藉彲鑳藉鐢ㄣ€?
### `IterativePreCompressionRepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦ㄥ啓鍏ュ墠瀵?turn 鏂囨湰鍋?token-level 棰勫帇缂╋紝鍙繚鐣欓珮淇℃伅瀵嗗害鍐呭锛屽悓鏃朵繚鐣欏師鏂囦笌鍘嬬缉鍚庢枃鏈殑瀵瑰簲鍏崇郴锛屼緵鍚庣画鍒嗘鍜屾€荤粨浣跨敤銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation` 鍙互鍋?summary/embedding/keywords 绛夊寮猴紝浣嗘病鏈夆€滀繚鐣欏摢浜?token銆佷涪寮冨摢浜?token鈥濈殑棰勫帇缂╄涔夈€?  - LightMem 鐨勬晥鐜囨敹鐩婇鍏堟潵鑷繖涓€姝ワ紝鑰屼笉鏄櫘閫氭憳瑕併€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆備换浣曞笇鏈涘湪 memory construction 鍓嶅仛杞婚噺鍘嬬缉闄嶆湰鐨勭郴缁熼兘鍙兘闇€瑕併€?
### `TopicSegmentSummaryRepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵?topic segment 绾у埆鐨勫 turn 鍐呭鐢熸垚 topic-aware summary锛屽苟鍚屾椂浜у嚭闀挎湡绱㈠紩鐢ㄧ殑 embedding-ready 琛ㄧず銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation(elements=("summary", "embedding"))` 鏄?unit-local 鐨勶紱瀹冧笉浼氭妸鈥滀竴涓?topic segment鈥濅綔涓虹ǔ瀹氳〃绀哄璞°€?  - LightMem 鐨勯暱鏈熻蹇?entry 鏉ヨ嚜 segment-level 鎬荤粨锛屼笉鏄崟 turn 灞€閮ㄥ寮恒€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶氬垎娈靛紡瀵硅瘽璁板繂鏂规硶閮戒細鎶?segment summary 褰撲綔闀挎湡璁板繂涓昏〃绀恒€?
### `TypedMemorySchemaExtractionRepresentation`

- 灞炰簬鍝釜 slot
  - `representation`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 浠ヤ氦浜掑寘涓鸿緭鍏ワ紝鎶藉彇涓€缁勫甫 `memory_type` 鏍囩鐨?typed memory payloads锛屼緥濡?core block update銆乪pisodic event銆乻emantic fact/profile銆乸rocedural workflow銆乺esource note銆乲nowledge-vault secret锛屽苟涓烘瘡绫?payload 鐢熸垚鍚庣画缁勭粐/妫€绱㈡墍闇€瀛楁銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BasicRepresentation` 涓庣幇鏈夊寮鸿〃绀烘ā鍧楅兘榛樿杈撳嚭缁熶竴璁板綍涓婄殑閫氱敤瀛楁銆?  - MIRIX 闇€瑕佺殑鏄€滀竴娆¤緭鍏ワ紝浜у嚭澶氱 schema 鍊欓€夆€濈殑琛ㄧず鑳藉姏锛岃€屼笉鏄湪鍗曚竴 record 涓婅ˉ `summary` 鎴?`embedding`銆?  - 褰撳墠娌℃湁妯″潡鏄惧紡鎵胯浇 `memory_type -> schema fields -> downstream storage contract` 杩欎竴灞傘€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄?typed-memory / multi-store memory 绯荤粺閮藉彲鑳介渶瑕佸畠銆?
## write_trigger

> Note
> This document discusses possible future trigger/module additions from the literature side.
> It does not reflect the current public baseline trigger API, which has been simplified to basic slot triggers only.

### `BufferCapacityWriteTrigger`

- 灞炰簬鍝釜 slot
  - `write_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 褰撴寚瀹氱紦鍐插眰鐨?token 鏁版垨鏉＄洰鏁拌揪鍒伴槇鍊兼椂瑙﹀彂鍚庣画鍐欏叆闃舵锛屽彲鐢ㄤ簬琛ㄨ揪 sensory buffer 婊¤Е鍙?segmentation銆丼TM 婊¤Е鍙?summary-to-LTM銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `ThresholdTrigger` 鍙兘娑堣垂宸叉湁 signal 鍒嗗€硷紝娌℃湁鐜版垚鐨?buffer occupancy / token budget 淇″彿銆?  - `AlwaysTrigger` 鏃犳硶琛ㄨ揪 LightMem 鐨勬壒閲忕紦鍐茶Е鍙戣涔夈€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄€滅紦瀛樻弧浜嗗啀澶勭悊鈥濈殑 memory pipeline 閮戒細闇€瑕併€?
### `MultiMemoryTypeRoutingWriteTrigger`

- 灞炰簬鍝釜 slot
  - `write_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵瑰綋鍓嶄氦浜掑寘鍋?one-to-many 璺敱锛岃緭鍑烘湰娆″簲鏇存柊鐨?memory type 闆嗗悎锛屼緥濡?`episodic + semantic`銆乣resource only`銆乣core + knowledge_vault`锛屼緵鍚庣画缁勭粐涓庣淮鎶ら摼璺娇鐢ㄣ€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 鐜版湁 `write_trigger` 瀹舵棌鏈€澶氳〃杈锯€滃啓 / 涓嶅啓鈥濇垨娑堣垂宸叉湁 metadata銆?  - MIRIX 鐨勫叧閿笉鏄槸鍚﹀€煎緱鍐欙紝鑰屾槸鈥滃簲璇ユ妸杩欐浜や簰鍒嗗彂鍒板摢浜?memory stores鈥濄€?  - 褰撳墠娌℃湁浠讳綍 trigger 鑳界ǔ瀹氳緭鍑哄鏍囩璺敱缁撴灉銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚绫诲瀷璁板繂绯荤粺銆佸伐鍏疯蹇嗙郴缁熴€佹潈闄愬垎鍖鸿蹇嗙郴缁熼兘鍙兘闇€瑕佽繖绉嶈矾鐢辫Е鍙戝櫒銆?
## organization

### `EntityFactPassageGraphOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶?passage銆乪ntity/noun phrase銆乫act/triple 缁勭粐鎴愬紓鏋勫浘绱㈠紩锛屽苟淇濈暀 node-to-passage incidence 淇℃伅锛屼緵鍚庣画鍥炬绱㈠拰 passage 鑱氬悎浣跨敤銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphAppendOrganization` 涓?`GraphAppendLinkReadyOrganization` 閮芥槸 record-centric graph 鍐欏叆锛涘畠浠病鏈?HippoRAG 鎵€闇€鐨勫紓鏋勮妭鐐圭被鍨嬨€乫act 杈硅涔夈€佷互鍙婅妭鐐瑰埌 passage 鐨勮仛鍚堢粺璁＄粨鏋勩€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄€済raph over extracted concepts, readout over source documents鈥濈殑 memory 绯荤粺閮藉彲鑳藉鐢ㄣ€?
### `SemanticEpisodicGraphOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍚屾椂缁存姢 semantic memory 涓?episodic memory锛?    - 鎶?observation 鎶藉嚭鐨?triplets 鍐欏叆 semantic graph锛?    - 鎶?observation 鏈韩鍐欐垚 episodic entry锛?    - 寤虹珛鈥滆 observation 瀵瑰簲杩欐壒 triplets鈥濈殑鍏宠仈缁撴瀯銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphAppendOrganization` 鍙兘琛ㄨ揪鏅€?graph-layer append锛?  - `GraphAppendLinkReadyOrganization` 闈㈠悜 note graph锛?  - 涓よ€呴兘涓嶈兘鑷劧琛ㄨ揪 AriGraph 鐨?semantic / episodic 鍙岃蹇嗚仈鍚堢粍缁囥€?- 杩欐槸涓?AriGraph 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?world-model memory銆佷簨浠跺浘 memory銆乤gent episode memory 閮介渶瑕佽繖绉嶅弻灞傜粍缁囪兘鍔涖€?
### `HierarchicalSubgoalChunkOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶?working memory 缁勭粐鎴愨€滃綋鍓?subgoal 鐨勮缁?action-observation chunk + 宸插畬鎴?subgoal 鐨勫綊妗?chunk鈥濅袱灞傜粨鏋勶紝骞剁ǔ瀹氱淮鎶?subgoal id銆乧hunk 杈圭晫銆佸綋鍓?鍘嗗彶鐘舵€併€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `AppendOrganization` 鍙兘椤哄簭杩藉姞锛?  - `ConditionalLayerOrganization` 鍙兘鎸夎鍒欏垎灞傦紱
  - `PlacementWithoutAppendOrganization` 鍙兘鍙?placement锛?  - 瀹冧滑閮戒笉鑳借嚜鐒惰〃杈?HiAgent 鎵€闇€鐨勨€滄寜 subgoal 鍒?chunk锛屽苟璁╁綋鍓?chunk 涓庡巻鍙?chunk 浣跨敤涓嶅悓淇濈湡搴︹€濈殑灞傜骇 working-memory 缁勭粐銆?- 杩欐槸涓?HiAgent 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄€滆鍒?鎵ц-鍘嬬缉-蹇呰鏃跺洖鐪嬧€濈殑 agent working-memory 绯荤粺閮藉彲鑳藉鐢ㄣ€?
### `TypedEntityRelationGraphOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶?typed entities銆乪ntity embeddings銆乺elation triplets 缁勭粐鎴?graph-memory 鎵€闇€鐨勮妭鐐?杈圭粨鏋勶紝骞剁ǔ瀹氫繚鐣欏疄浣撶被鍨嬨€佹椂闂存埑銆佷綔鐢ㄥ煙 metadata銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphAppendOrganization` 鍙敮鎸?record-centric graph append锛?  - 瀹冧笉鑳借嚜鐒惰〃杈?Mem0 graph 鐗堟墍闇€鐨勨€滃疄浣撹妭鐐?+ 鍏崇郴杈?+ 鑺傜偣 embedding + 杈圭姸鎬佲€濊繖濂?typed graph organization銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶氬浘璁板繂鏂规硶閮介渶瑕?typed node/edge organization锛岃€屼笉鍙槸鏅€?record graph銆?
### `HierarchicalBufferMemoryOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鏄惧紡缁勭粐 sensory buffer銆乼opic-aware STM銆丩TM 涓夊眰缁撴瀯锛屾敮鎸佹潯鐩厛杩涘叆鐭湡缂撳啿锛屽啀浠ユ壒澶勭悊鏂瑰紡鎻愬崌鍒伴暱鏈熻蹇嗐€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `AppendOrganization` 鍙兘鐩存帴钀藉眰銆?  - `ConditionalLayerOrganization` 鍙兘鍋氶潤鎬佽矾鐢憋紝涓嶈兘琛ㄨ揪鈥滃厛缂撳啿銆佸悗鎻愬崌鈥濈殑鐢熷懡鍛ㄦ湡銆?  - 褰撳墠娌℃湁涓€涓ā鍧楄兘鎶?LightMem 鐨勪笁灞傜粨鏋勮涓轰竴涓暣浣撶粍缁囧師璇€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?memory 绯荤粺閮芥湁 cache/working/LTM 鍒嗗眰锛屽彧鏄?LightMem 鎶婅繖涓€鐐瑰仛寰楁洿鏄庣‘銆?
### `TopicSegmentIndexOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶婁竴鎵?turn 缁勭粐鎴?`{topic, message turns}` 褰㈠紡鐨?topic segment 绱㈠紩缁撴瀯锛屽苟缁存姢 segment 鍒板悗缁?LTM entry 鐨勬潵婧愮粦瀹氥€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 organization 瀹舵棌娌℃湁鈥渟egment-level 绱㈠紩瀵硅薄鈥濇蹇碉紝鍙湁 record append 鎴?graph append銆?  - LightMem 鐨?STM 涓嶆槸鏅€?layer锛岃€屾槸 topic-aware segment store銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傛墍鏈?topic/session/chunk 绾ц蹇嗙郴缁熼兘鍙兘闇€瑕佺被浼肩殑涓棿缁勭粐灞傘€?
### `HeterogeneousMemoryStoreOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶婂甫 `memory_type` 鐨?typed payload 鍒嗗彂鍒板紓鏋?memory stores锛屽厑璁镐笉鍚?store 鎷ユ湁涓嶅悓 schema銆佷笉鍚屼富閿?鏇存柊绛栫暐銆佷笉鍚岃繃婊ょ淮搴︿笌涓嶅悓妫€绱㈠瓧娈点€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `AppendOrganization` 涓?`ConditionalLayerOrganization` 閮介粯璁ゅ湪鍚屾瀯璁板綍绌洪棿閲屽仛杩藉姞鎴栬矾鐢便€?  - MIRIX 涓嶆槸鈥滃悓涓€ record 鎹釜 layer鈥濓紝鑰屾槸 block memory銆乪vent memory銆乧oncept memory銆乺esource memory銆乻ensitive vault 绛夊苟瀛樸€?  - 褰撳墠娌℃湁妯″潡鎶娾€滃紓鏋?store family鈥濊涓虹粍缁囧眰鐨勪竴绛夊璞°€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?production memory systems 閮戒細鎶婁笉鍚岀被鍨嬬殑璁板繂鍒嗗埌涓嶅悓 store锛岃€屼笉鏄叡浜崟涓€ schema銆?
### `CoreBlockMemoryOrganization`

- 灞炰簬鍝釜 slot
  - `organization`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鏄惧紡鎵胯浇 bounded block-style memory锛屼緥濡?persona block銆乭uman-profile block銆乻ystem-preference block锛屾敮鎸佸潡绾ч噸鍐欍€佸閲忓害閲忎笌楂樹紭鍏堢骇鎻愮ず娉ㄥ叆銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 organization 瀹舵棌涓昏闈㈠悜 record append銆乬raph append 鎴?layer placement銆?  - 瀹冧滑閮戒笉鎶娾€滃浐瀹氬皯閲忛珮浠峰€煎潡鐘惰蹇嗏€濆綋浣滀竴绉嶇嫭绔嬬粍缁囧舰鎬侊紝鍥犳鏃犳硶蹇犲疄琛ㄨ揪 MIRIX 鐨?core memory銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?agent memory 绯荤粺閮戒細淇濈暀涓€灏忕粍楂樹紭鍏堢骇 profile / persona blocks銆?
## evolution_trigger

### `EmbeddingSimilarityAugmentationTrigger`

- 灞炰簬鍝釜 slot
  - `evolution_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦ㄦ柊 passage/entity 鑺傜偣鍐欏叆鍚庯紝鍒ゆ柇鏄惁闇€瑕佹墽琛屽熀浜?embedding 鐩镐技搴︾殑鑺傜偣杩炶竟澧炲己銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `NeighborExistsEvolutionTrigger` 鍋囪宸叉湁鍥鹃偦灞呮垨 record-level graph candidate锛汬ippoRAG 鏇村儚鈥滄柊鑺傜偣鍐欏叆鍚庯紝瀵硅妭鐐?embedding 绌洪棿鍋氬€欓€夋绱㈠拰闃堝€煎垽鏂€濄€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive锛屼絾鏈€鍒濋渶姹傛潵鑷?HippoRAG銆?
### `SubgoalCompletionEvolutionTrigger`

- 灞炰簬鍝釜 slot
  - `evolution_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 褰撴ā鍨嬫垨涓婃父鎺у埗閫昏緫鍒ゆ柇鈥滃綋鍓?subgoal 宸插畬鎴愶紝搴斿帇缂╁苟褰掓。璇?chunk鈥濇椂锛岃Е鍙戝悗缁?memory evolution銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `ThresholdTrigger` 鍙兘鍋氬父閲?闃堝€艰Е鍙戯紱
  - `OutcomeConditionedEvolutionTrigger` 闈㈠悜 Reflexion 寮忚瘯娆＄粨鏋滐紱
  - `NewWriteEvolutionTrigger` 闈㈠悜 keyed/local-maintenance锛?  - 褰撳墠娌℃湁涓€涓樉寮忛潰鍚戔€渟ubgoal 瀹屾垚鈥濊繖涓€ working-memory 鍒囨崲浜嬩欢鐨勮Е鍙?primitive銆?- 杩欐槸涓?HiAgent 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶氬垎灞傝鍒掑瀷 agent 閮介渶瑕佲€滈樁娈靛畬鎴愬悗鍐嶅帇缂╄蹇嗏€濈殑瑙﹀彂杈圭晫銆?
### `AlwaysUpdateEvaluationTrigger`

- 灞炰簬鍝釜 slot
  - `evolution_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵规瘡涓凡鎶藉嚭鐨?candidate fact 缁熶竴瑙﹀彂涓€娆♀€滅浉浼兼棫璁板繂姣旇緝 + 鍐崇瓥缁存姢鈥濈殑鍚庣画娴佺▼锛屾槑纭〃杈?Mem0 杩欑被鍥哄畾 update phase銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 鐜版湁 `ThresholdTrigger` 鍙互鍕夊己鍏呭綋鎭掔湡瑙﹀彂锛屼絾璇箟涓婁笉澶熸竻鏅帮紱Mem0 鐨勭壒鐐逛笉鏄槇鍊艰Е鍙戯紝鑰屾槸 extraction 涔嬪悗鍥哄畾杩涘叆 update evaluation銆?- 杩欐槸涓?Mem0 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?memory-maintenance 娴佹按绾块兘瀛樺湪鈥滃€欓€変竴鏃︾敓鎴愶紝灏卞繀椤昏繃涓€娆＄淮鎶ゅ喅绛栤€濈殑鍥哄畾鍚庡鐞嗛樁娈点€?
### `SleepTimeBatchEvolutionTrigger`

- 灞炰簬鍝釜 slot
  - `evolution_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鏄惧紡琛ㄨ揪鈥滃湪绾垮啓鍏ヤ箣鍚庝笉绔嬪嵆缁存姢锛岀瓑鍒?sleep-time / offline window / 澶栭儴 update signal 鍒版潵鏃讹紝鍐嶇粺涓€瑙﹀彂 evolution鈥濄€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `NewWriteEvolutionTrigger` 鏄湪绾裤€佸眬閮ㄧ殑鏂板啓鍏ヨЕ鍙戙€?  - `ThresholdTrigger` 娌℃湁绂荤嚎鎵瑰鐞嗘垨 sleep-time 璇箟銆?  - LightMem 鐨勫叧閿偣鎭版伆鏄妸 update 浠庡湪绾胯矾寰勬媶鍑哄幓銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆備换浣曟妸缁存姢寤跺悗鍒板悗鍙版垨绌洪棽鏃舵鐨?memory 绯荤粺閮藉彲鑳介渶瑕併€?
### `CapacityConditionedBlockRewriteTrigger`

- 灞炰簬鍝釜 slot
  - `evolution_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 褰?block-style core memory 鎺ヨ繎瀹归噺涓婇檺鏃讹紝瑙﹀彂 block rewrite / condense / consolidation锛岃€屼笉鏄户缁棤绾︽潫杩藉姞銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `ThresholdTrigger` 鍙兘娑堣垂涓€鑸€у垎鏁伴槇鍊硷紝娌℃湁 block fullness / token occupancy 璇箟銆?  - MIRIX 鐨?core memory rewrite 鏄?block-aware 缁存姢锛岃€屼笉鏄櫘閫氳褰曠殑鍚庡鐞嗐€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄噰鐢?bounded profile blocks 鎴?memory blocks 鐨勭郴缁熼兘鍙兘闇€瑕併€?
### `ScheduledReflexionEvolutionTrigger`

- 灞炰簬鍝釜 slot
  - `evolution_trigger`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦ㄦ寚瀹氭椂鏈鸿Е鍙戣法 memory stores 鐨?cleanup / dedupe / pattern extraction锛屼緥濡?per-query銆乸eriodic銆乮dle-time 鎴?explicit maintenance window銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `OutcomeConditionedEvolutionTrigger` 闈㈠悜璇曢敊鍙嶉銆?  - `NewWriteEvolutionTrigger` 闈㈠悜鍐欏悗灞€閮ㄧ淮鎶ゃ€?  - 褰撳墠娌℃湁涓€涓樉寮忚〃杈锯€滀负浜?memory hygiene 涓?higher-order synthesis 鑰屽畾鏈熸暣鐞嗏€濈殑瑙﹀彂鍣ㄣ€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶氶暱鏈熻繍琛?agent 閮介渶瑕佺嫭绔嬬殑缁存姢绐楀彛鏉ュ仛鍙嶆€濄€佸幓閲嶅拰鎻愮偧銆?
## memory_evolution

### `SynonymyEdgeAugmentationEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍩轰簬 entity/noun phrase embedding 鐨勭浉浼煎害闃堝€硷紝涓哄浘鑺傜偣鎵归噺澧炲姞 synonymy/similarity edges銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphLinkEvolution` 瀹舵棌褰撳墠闈㈠悜 record graph/link metadata锛屼笉鏄妭鐐圭骇 embedding augmentation锛涗篃涓嶇淮鎶?HippoRAG 闇€瑕佺殑鍚屼箟杈硅涔夈€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠閫傜敤浜庡緢澶氣€滄蹇靛浘 + 鐩镐技鑺傜偣妗ユ帴鈥濈殑闀挎湡璁板繂鏂规銆?
### `OutdatedSemanticEdgePruningEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍩轰簬褰撳墠 observation 鎻愬彇鍑虹殑鏂?triplets锛屽鐩稿叧鏃?semantic edges 鍋氬啿绐佹娴嬨€佽繃鏃朵簨瀹炶瘑鍒笌鍒犻櫎锛岀劧鍚庢妸 semantic memory 淇鍒版渶鏂扮姸鎬併€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 `GraphLinkEvolution` / `GraphNeighborAppendEvolution` / `NeighborContextUpdateEvolution` 閮藉亸鍚戞柊澧炴垨鏀瑰啓涓婁笅鏂囷紝涓嶈鐩?AriGraph 鍏抽敭鐨勨€滃垹闄ゅ紡浜嬪疄淇鈥濄€?- 杩欐槸涓?AriGraph 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆備换浣曟寔缁氦浜掋€佷笘鐣岀姸鎬佷細鍙樺寲鐨?graph memory 閮藉彲鑳介渶瑕佸眬閮ㄤ簨瀹炰慨璁紝鑰屼笉鍙槸 append銆?
### `SubgoalTrajectorySummarizationEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵光€滃凡瀹屾垚 subgoal鈥濆搴旂殑鏁存 action-observation trajectory 鍋?subgoal-conditioned summarization锛屼骇鍑哄彲鏇夸唬鏃х粏鑺傜殑 summarized observation銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `SummaryRewriteEvolution` 铏界劧鑳借拷鍔?summary record锛屼絾瀹冩槸鍩轰簬鍗曚釜 unit 宸叉湁 summary/description 瀛楁鍋?append-only 閲嶅啓锛?  - HiAgent 闇€瑕佺殑鏄€滃涓€涓?subgoal chunk 鐨勫姝ヨ建杩瑰仛鎽樿鈥濓紝鑰屼笉鏄鍗曟潯 unit 鏂囨湰鍋氳交閲忔敼鍐欍€?- 杩欐槸涓?HiAgent 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?working-memory 鍘嬬缉鏂规閮介渶瑕?chunk-level 鑰岄潪 unit-level 鐨勬憳瑕佹紨鍖栥€?
### `HierarchicalChunkReplacementEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦ㄥ畬鎴?subgoal 鍚庯紝鎶婃棫 detailed chunk 鍒囨崲鎴愨€滈粯璁ら殣钘?褰掓。銆佷絾浠嶅彲鎸夐渶鎭㈠鈥濈殑鐘舵€侊紝骞惰 summarized observation 鎴愪负榛樿鏆撮湶缁欏伐浣滆蹇嗙殑鐗堟湰銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `LayerMoveEvolution` 鍜?`SummaryRewriteEvolution` 閮芥槸 copy-append 椋庢牸锛?  - 瀹冧滑涓嶄細鐪熸琛ㄨ揪 HiAgent 闇€瑕佺殑鈥滄棫缁嗚妭琚?summary 鍙栦唬涓洪粯璁や笂涓嬫枃锛屼絾璇︾粏杞ㄨ抗浠嶅彲鍥炲彇鈥濈殑鏇挎崲/褰掓。璇箟銆?- 杩欐槸涓?HiAgent 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠瀵瑰簲鐨勬槸涓€绫烩€滃帇缂╁悗榛樿闅愯棌銆佷絾淇濇寔鍙仮澶嶁€濈殑 working-memory 鐢熷懡鍛ㄦ湡绠＄悊銆?
### `SimilarityResolvedMemoryUpdateEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵规瘡涓?candidate fact 鍏堟绱?top-k 鐩镐技鏃ц蹇嗭紝鍐嶇敱 LLM 鎴栫瓑浠峰喅绛栧櫒杈撳嚭 `ADD / UPDATE / DELETE / NONE`锛屽苟鎵ц瀹為檯鐨?create/update/delete/no-op銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 `SummaryRewriteEvolution`銆乣LayerMoveEvolution`銆乣GraphLinkEvolution` 閮戒笉鍏峰鈥滅浉浼艰蹇嗘瘮杈?+ 鍥涜矾缁存姢鍔ㄤ綔鍐崇瓥 + 瀹為檯 store mutation鈥濊繖涓€缁勫悎鑳藉姏锛?  - 杩欐鏄?Mem0 涓讳綋鏈€鏍稿績鐨勬満鍒躲€?- 杩欐槸涓?Mem0 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠瀵瑰簲鐨勬槸涓€绫烩€渃andidate fact 缁忚繃 similarity-based maintenance policy 鍚庡啀鍐冲畾钀藉簱鈥濈殑闀挎湡璁板繂缁存姢妯″紡銆?
### `GraphConflictInvalidationEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵规柊杩涘叆鐨勫叧绯?triplets 妫€娴嬩笌鏃㈡湁鍥惧叧绯荤殑鍐茬獊锛屽苟鎶婅繃鏃跺叧绯绘爣璁颁负澶辨晥鑰屼笉鏄畝鍗曡拷鍔犳垨纭垹闄ゃ€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphLinkEvolution` 鍋忓悜鏂板閾炬帴锛?  - `NeighborContextUpdateEvolution` 鍋忓悜閲嶅啓閭诲眳涓婁笅鏂囷紱
  - 褰撳墠娌℃湁涓€涓槑纭〃杈锯€滃啿绐佸叧绯诲け鏁堝寲 / temporal invalidation鈥濈殑鍥剧淮鎶?primitive銆?- 杩欐槸涓?Mem0 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄渶瑕佷繚鐣欐椂闂翠竴鑷存€ф垨鍘嗗彶鐘舵€佺殑鍥捐蹇嗙郴缁熼兘鍙兘闇€瑕佽繖绉?invalidation 鏈哄埗銆?
### `TimestampConstrainedUpdateQueueEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 涓烘瘡涓暱鏈熻蹇嗘潯鐩瀯閫犱竴涓熀浜?embedding 鐩镐技搴︾殑 top-k update queue锛屽苟鍔犲叆鏃堕棿绾︽潫锛屽彧鍏佽杈冩柊鐨勬潯鐩垚涓鸿緝鏃ф潯鐩殑鍊欓€夋洿鏂版潵婧愩€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 `memory_evolution` 瀹舵棌娌℃湁鈥滃厛鐢熸垚鍏ㄥ眬 update queue 鍐嶆洿鏂扳€濈殑闃舵鍖栨満鍒躲€?  - `SummaryRewriteEvolution`銆乣LayerMoveEvolution` 閮戒笉浼氫骇鍑哄彲澶嶇敤鐨勫€欓€夋洿鏂伴槦鍒椼€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄噰鐢ㄢ€滃厛鎵惧€欓€夛紝鍐嶅仛缁存姢鍐崇瓥鈥濈殑绂荤嚎鏇存柊绯荤粺閮藉彲鑳藉鐢ㄣ€?
### `ParallelQueueResolvedUpdateEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 浠?`target entry + update_queue candidate sources` 涓鸿緭鍏ワ紝鎵ц绂荤嚎骞惰鐨?`update / delete / ignore` 鍐崇瓥涓?store mutation銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠娌℃湁浠讳綍 evolution 妯″潡鑳芥妸鈥滃€欓€夋潵婧愰泦鍚堚€濅綔涓烘洿鏂颁笂涓嬫枃锛屽苟鏀寔鎵归噺骞惰鎵ц銆?  - 杩欐鏄?LightMem 涓庡湪绾块€愭潯缁存姢绯荤粺鐨勫叧閿樊寮傘€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶氫綆寤惰繜 memory 绯荤粺閮戒細鍙楃泭浜庢妸閲嶇淮鎶ゆ惉鍒板悗鍙板苟琛屽鐞嗐€?
### `TypedMemoryUpdateEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 閽堝涓嶅悓 memory type 鎵ц `update / merge / replace / skip` 缁存姢鍔ㄤ綔锛屽苟鍏佽姣忕被 memory 浣跨敤鑷繁鐨勫瓧娈靛榻愩€侀噸澶嶆娴嬩笌 embedding 鍒锋柊绛栫暐銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `SummaryRewriteEvolution`銆乣AppendOnlyEvolution`銆乣LayerMoveEvolution` 閮介粯璁ら潰瀵硅緝缁熶竴鐨勮褰曟搷浣溿€?  - MIRIX 闇€瑕佺殑鏄?typed maintenance family锛岃€屼笉鏄崟涓€ rewrite 鎴?append 妯″紡銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚 schema memory 绯荤粺鏅亶闇€瑕佹寜绫诲瀷鍖哄垎缁存姢璇箟銆?
### `CrossStoreDeduplicationEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵瑰涓?memory stores 杩涜璺ㄥ簱鍘婚噸銆佹竻鐞嗕笌涓€鑷存€ф暣鐞嗭紝閬垮厤鍚屼竴淇℃伅鍦?episodic / semantic / resource 绛?store 涓互澶辨帶鏂瑰紡閲嶅鍫嗙Н銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 鐜版湁 evolution 妯″潡閮戒富瑕佸湪鍗?layer 鎴栧崟鍥剧粨鏋勫唴閮ㄥ伐浣溿€?  - 褰撳墠娌℃湁妯″潡鏄惧紡鎶娾€滃涓?store 涔嬮棿鐨勯噸澶嶄笌鍐椾綑鈥濆綋浣滅淮鎶ゅ璞°€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚彧瑕佺郴缁熷悓鏃剁淮鎶ゅ绫昏蹇嗗簱锛屽氨鍙兘闇€瑕佽繖绉?hygiene primitive銆?
### `BehaviorPatternSynthesisEvolution`

- 灞炰簬鍝釜 slot
  - `memory_evolution`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 浠?episodic / interaction histories 涓彁鐐兼洿楂樺眰琛屼负妯″紡銆佸亸濂芥垨鐢熸椿瑙勫緥锛屽苟鎶婄粨鏋滃洖鍐欏埌 semantic 鎴栧叾浠栨洿绋冲畾鐨?memory store銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `ReflectionGenerationEvolution` 鍙兘灞€閮ㄨ鐩栤€滅敓鎴愬弽鎬濇枃鏈€濈殑澶栬銆?  - 瀹冧笉璐熻矗浠庝竴涓?store 璇汇€佹彁鐐?pattern銆佸啀鍐欏洖鍙︿竴涓?typed store銆?  - MIRIX Reflexion 鐨勫叧閿箣涓€姝ｆ槸杩欑璺ㄧ被鍨嬬殑 higher-order synthesis銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?agent memory 绯荤粺閮藉笇鏈涙妸缁忓巻閫愭娌夋穩鎴愭洿绋冲畾鐨勫亸濂姐€佽鍒欐垨 profile銆?
## retrieval

### `QueryEntityLinkingPPRRetrieval`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦?retrieval 鍐呭畬鎴?query named entity extraction銆乪ntity-to-node linking銆乹uery seed weighting銆丳ersonalized PageRank 鎵╂暎锛屼互鍙?passage 鎺掑悕杈撳嚭銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphSeedAndExpandRetrieval` 鍜?`VectorGraphSeedAndExpandRetrieval` 鍙湁 seed + expand 鐨勮繎浼奸鏋讹紝娌℃湁 query linking銆丳PR銆乶ode specificity 涓?passage incidence 鑱氬悎銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠瀵瑰簲鐨勬槸涓€绫烩€済raph propagation retrieval鈥濊€屼笉鍙槸涓€绡囪鏂囥€?
### `NodeToPassageAggregationRetrievalAdapter`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶婂浘鑺傜偣鍒嗘暟鎸?node-to-passage incidence 缁熻鑱氬悎鎴?source passage 鍒嗘暟锛屽舰鎴愮湡姝ｇ殑鏂囨。绾ф绱㈣緭鍑恒€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 retrieval 妯″潡榛樿鐩存帴瀵?record 鎵撳垎鎴栧仛 record 閭绘帴鎵╁睍锛屾病鏈夆€滃厛鑺傜偣鎵撳垎銆佸啀鍥炴姇鍒?passage鈥濈殑涓ょ骇妫€绱㈡帴鍙ｃ€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠鏄?graph-memory 鏂囨。妫€绱腑鐨勫父瑙佹ˉ鎺ュ眰銆?
### `SemanticEpisodicGraphRetrieval`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍏堜粠 query 涓绱㈢浉鍏?semantic triplets 骞跺仛鍙楁帶鍥炬墿灞曪紝鍐嶅熀浜庤繖浜?triplets 鍥炶繛 episodic memories锛屽 past observations 鎵撳垎骞惰繑鍥?top-k銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GraphSeedAndExpandRetrieval` 鍙湁 graph seed-expand 鐨勮疆寤擄紝娌℃湁 AriGraph 鎵€闇€鐨勨€渟emantic retrieval -> episodic retrieval鈥濅袱娈靛紡妫€绱紱
  - `EmbeddingSimilarityRetrieval` 涔熶笉鑳芥妸 semantic 妫€绱㈢粨鏋滆繘涓€姝ヨ浆鎴?episodic recall銆?- 杩欐槸涓?AriGraph 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠瀵瑰簲鐨勬槸涓€绫烩€滃浘浜嬪疄妫€绱㈠悗鍥炶繛浜嬩欢/杞ㄨ抗璁板繂鈥濈殑 agent-memory 妯″紡銆?
### `SubgoalTrajectoryRetrieval`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎸?subgoal id 妫€绱㈡煇涓凡褰掓。 subgoal 鐨勫畬鏁?action-observation trajectory锛屽苟鎶婂畠閲嶆柊鏆撮湶鍒板綋鍓嶅伐浣滆蹇嗕笂涓嬫枃銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BufferRetrieval` 鍙細璇绘渶杩戠獥鍙ｏ紱
  - `RecencyRetrieval` 涔熸槸鏃堕棿椤哄簭鍙栧洖锛?  - `LayerAwareRetrieval` 鍙兘鍒嗗彂宸叉湁 retriever锛?  - 褰撳墠娌℃湁涓€涓€滄寜 chunk 鏍囪瘑绗︾簿纭洖鍙栨棫 detailed trajectory鈥濈殑 retrieval primitive銆?- 杩欐槸涓?HiAgent 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄噰鐢?chunk/archive 璁板繂缁勭粐鐨?agent 閮藉彲鑳介渶瑕佹寜 chunk id 绮剧‘鎭㈠缁嗚妭銆?
### `EntityCentricTripletHybridRetrieval`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍚屾椂鏀寔锛?    - 浠?query 涓娊鍏抽敭瀹炰綋骞堕敋瀹氬埌鍥捐妭鐐癸紱
    - 娌垮浘鍋氬疄浣撲腑蹇冪殑鍏崇郴鎵╁睍锛?    - 鍐嶅 triplet 鏂囨湰琛ㄧず鍋氳涔夊尮閰嶄笌閲嶆帓锛?  - 浣滀负 Mem0 graph 鐗堢殑 retrieval 涓诲共銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `EmbeddingSimilarityRetrieval` 鍙鐩栧熀纭€鍚戦噺妫€绱紱
  - `GraphSeedAndExpandRetrieval` 鍙湁閫氱敤 seed-expand 楠ㄦ灦锛屾病鏈?Mem0 graph 鐗堢殑 entity-centric + triplet semantic retrieval 鍙岃矾寰勩€?- 杩欐槸涓?Mem0 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠瀵瑰簲鐨勬槸涓€绫烩€滃浘閿氬畾妫€绱?+ triplet 璇箟鍖归厤鈥濈殑鍥捐蹇嗘绱㈡ā寮忋€?
### `MultiStoreFieldAwareRetrieval`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦ㄥ涓紓鏋?memory stores 涓婄粺涓€缂栨帓妫€绱紝鍏佽姣忎釜 store 浣跨敤涓嶅悓 searchable fields銆佷笉鍚?search method 缁勫悎銆佷笉鍚岃繃婊ゆ潯浠讹紝骞舵妸缁撴灉鏁村悎鎴?typed retrieval result銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `BM25Retrieval`銆乣EmbeddingSimilarityRetrieval` 鍙鐩栧崟涓€妫€绱㈢瓥鐣ャ€?  - `LayerAwareRetrieval` 鍙兘鍋氳緝娴呭眰鐨勫灞傚垎鍙戙€?  - MIRIX 闇€瑕佺殑鏄€滄寜 memory type 閫夋嫨瀛楁銆佺瓥鐣ヤ笌杩囨护鍣ㄢ€濈殑 retrieval orchestrator銆?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚嚒鏄?multi-store memory 鏋舵瀯閮戒細闇€瑕佺被浼艰兘鍔涖€?
### `DualViewEpisodicRetrieval`

- 灞炰簬鍝釜 slot
  - `retrieval`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 瀵?episodic memory 鍚屾椂浜у嚭 `recent` 涓?`relevant` 涓よ矾缁撴灉锛屽垎鍒湇鍔℃椂闂磋繎鍥犱笌涓婚鐩稿叧鎬э紝骞舵妸涓よ€呬綔涓轰笉鍚岃鍥句氦缁?readout銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `RecencyRetrieval` 涓?`EmbeddingSimilarityRetrieval` 鍚勮嚜鍙兘鍋氫竴鏉¤矾銆?  - 褰撳墠娌℃湁妯″潡鎶娾€滃悓涓€ episodic store 涓婄殑鍙岃鍥惧彫鍥炩€濅綔涓轰竴涓ǔ瀹氭帴鍙ｈ緭鍑恒€?- 杩欐槸涓?HippoRAG 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚緢澶?episodic-memory 绯荤粺閮藉悓鏃堕渶瑕佲€滄渶杩戝彂鐢熶簡浠€涔堚€濆拰鈥滄渶鐩稿叧鐨勬棫缁忓巻鈥濄€?
## readout

### `HierarchicalWorkingMemoryReadout`

- 灞炰簬鍝釜 slot
  - `readout`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶娾€滃巻鍙?subgoal 鐨?summary銆佸綋鍓?subgoal 鐨勮缁嗚建杩广€佹寜闇€鎭㈠鐨勬棫 subgoal 缁嗚妭鈥濈粍瑁呮垚绋冲畾 prompt 涓婁笅鏂囷紝骞舵樉寮忎繚鐣?subgoal 缂栧彿涓庡眰绾у叧绯汇€?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `ConcatenateReadout`銆乣BulletListReadout`銆乣GroupedByLayerReadout` 鍙兘鍋氶€氱敤鎷兼帴锛?  - `PromptContextReadout` 褰撳墠鍋忓悜 Reflexion 鍦烘櫙锛?  - 瀹冧滑閮戒笉鑳界洿鎺ヨ〃杈?HiAgent 鐨勫眰绾?working-memory 鍛堢幇鏍煎紡銆?- 杩欐槸涓?HiAgent 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚畠鏈嶅姟鐨勬槸涓€绫烩€滃帇缂╁巻鍙?+ 淇濈暀褰撳墠缁嗚妭 + 鎸夐渶鍐嶅睍寮€鈥濈殑 prompt-memory 璇诲嚭妯″紡銆?
### `TypedMemoryPromptReadout`

- 灞炰簬鍝釜 slot
  - `readout`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鎶婂 store retrieval results 鎸?memory type 缁勮涓虹ǔ瀹?prompt 鐗囨锛屾敮鎸?typed section headers銆佸瓧娈佃鍓€佸彲閫?item ids 涓庢瘡绫昏蹇嗙殑灞曠ず妯℃澘銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - `GroupedByLayerReadout` 鍙細鎸?layer 鍒嗙粍銆?  - `PromptContextReadout` 鍙彁渚涢€氱敤 prompt 娉ㄥ叆澶栬銆?  - MIRIX 闇€瑕佺殑鏄樉寮忕殑 typed-memory prompt assembly锛岃€屼笉鏄櫘閫氭枃鏈嫾鎺ャ€?- 杩欐槸涓?MIRIX 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚绫诲瀷璁板繂绯荤粺鍑犱箮閮介渶瑕佹寜绫诲瀷缁勭粐缁欎笅娓告ā鍨嬬湅鐨勪笂涓嬫枃銆?
### `SensitivityAwareMemoryReadout`

- 灞炰簬鍝釜 slot
  - `readout`
- 瑕佽ˉ鐨勮兘鍔涙槸浠€涔?  - 鍦ㄨ鍑洪樁娈电粨鍚?sensitivity銆乷wner銆乻cope 绛夋潈闄愪俊鎭紝瀵?retrieval results 鍋氳繃婊ゃ€侀檷绾у睍绀烘垨灞忚斀锛岄伩鍏嶆妸涓嶈鏆撮湶鐨勪俊鎭洿鎺ユ敞鍏?prompt銆?- 涓轰粈涔堢幇鏈夋ā鍧椾笉澶?  - 褰撳墠 readout 瀹舵棌榛樿鈥滄绱㈠埌浜嗗氨璇诲嚭鈥濄€?  - MIRIX 鐨?knowledge vault 鏄庢樉瀛樺湪棰濆鐨勫彲瑙佹€ц竟鐣岋紝杩欎笉鏄櫘閫?`ConcatenateReadout` 鎴?`PromptContextReadout` 鑳借〃杈剧殑銆?- 杩欐槸涓?MIRIX 鐗瑰寲锛岃繕鏄彲鎶借薄涓洪€氱敤 primitive
  - 鍙娊璞′负閫氱敤 primitive銆傚彧瑕佽蹇嗗簱閲屾贩鏈夋晱鎰熶俊鎭€佺鏈変俊鎭垨涓嶅悓 agent 鐨勪綔鐢ㄥ煙淇℃伅锛屽氨浼氶渶瑕併€?