# MemPrimitive

## 椤圭洰绠€浠?

`MemPrimitive` 鏄竴涓潰鍚?agent memory research 鐨勭郴缁熷寲鐮旂┒妗嗘灦銆傝繖涓」鐩殑鏍稿績鐩爣锛屼笉鏄啀鎻愬嚭涓€涓崟鐙殑 memory 鏂规硶锛岃€屾槸璇曞浘鍥炵瓟涓€涓洿鍩虹鐨勯棶棰橈細

**鎴戜滑鑳藉惁鐢ㄧ粺涓€銆佹満鍒剁骇銆佸彲缁勫悎鐨勬柟寮忔潵鎻忚堪 agent memory 鐨勮璁＄┖闂达紝浠庤€屾妸鐜版湁鏂规硶鏀惧埌鍚屼竴涓鏋朵笅姣旇緝銆侀噸缁勪笌鎼滅储锛?*

褰撳墠鍏充簬 agent memory 鐨勭爺绌跺凡缁忓嚭鐜颁簡澶ч噺鏂规硶锛屽畠浠湪琛ㄩ潰褰㈠紡涓婂樊寮傚緢澶э紝鏈夌殑寮鸿皟 episodic memory锛屾湁鐨勫己璋?semantic memory锛屾湁鐨勪緷璧栧悜閲忔绱紝鏈夌殑寮鸿皟鍙嶆€濄€佹憳瑕併€佸浘缁撴瀯銆佹妧鑳藉簱鎴?agent 涓诲姩鎺у埗鐨勮蹇嗚鍐欍€備絾濡傛灉浠庡簳灞傛満鍒舵潵鐪嬶紝杩欎簺鏂规硶寰€寰€閮藉湪閲嶅鑻ュ共鐩镐技鐨勮璁″姩浣滐紝渚嬪锛?

- 鎶婅緭鍏ュ垏鎴愭煇绉嶈蹇嗗崟鍏?
- 鍐冲畾鍝簺鍐呭鍊煎緱鍐欏叆
- 鍐冲畾璁板繂濡備綍缁勭粐涓庢洿鏂?
- 鍦ㄩ渶瑕佹椂浠庤蹇嗕腑妫€绱㈢浉鍏冲唴瀹?
- 瀵硅蹇嗗仛鍘嬬缉銆佹娊璞°€佺淮鎶ゅ拰閬楀繕

鐜版湁宸ヤ綔鐨勯棶棰樺湪浜庯紝杩欎簺鍔ㄤ綔閫氬父琚皝瑁呭湪鍗曠瘒璁烘枃鎻愬嚭鐨勫叿浣撶粨鏋勪腑锛屽鑷?memory design space 闅句互琚樉寮忚〃杈俱€傜粨鏋滄槸锛?

- 涓嶅悓鏂规硶涔嬮棿闅句互鍏钩姣旇緝
- 鐩镐技鏈哄埗寰€寰€琚笉鍚屾湳璇噸澶嶅懡鍚?
- 妯″潡鏃犳硶鏂逛究鍦版媶瑙ｃ€佹浛鎹㈠拰澶嶇敤
- 鏂版柟娉曠殑璁捐鏇村渚濊禆缁忛獙锛岃€岄潪绯荤粺鎼滅储
- 寰堥毦浠庡涓垚鍔熸柟娉曚腑褰掔撼 recurring motifs

`MemPrimitive` 鐨勫嚭鍙戠偣锛屽氨鏄妸 memory system 浠庘€滃崟涓€鏂规硶鈥濋噸鍐欎负鈥滃彲缁勫悎绯荤粺鈥濄€?

---

## 鏍稿績涓诲紶

鏈」鐩彁鍑轰竴涓?**compositional memory DSL**锛屽皢 agent memory system 鍒嗚В涓轰竴缁勫彲缁勫悎鐨?primitive銆傛瘡涓?primitive 瀵瑰簲 memory 娴佺▼涓殑涓€绉嶅熀纭€鏈哄埗锛岃€屼笉鏄煇绡囪鏂囦腑鐨勬暣浣撴灦鏋勩€?

杩欎簺 primitive 鍖呮嫭浣嗕笉闄愪簬锛?

- `unit formation`锛氬浣曟妸鍘熷 observation 褰㈡垚璁板繂鍗曞厓
- `representation`锛氳蹇嗗崟鍏冨浣曠紪鐮併€佺粨鏋勫寲鍜岀储寮曞寲
- `write trigger`锛氱郴缁熶綍鏃跺喅瀹氬啓鍏ヨ蹇?
- `organization`锛歩ngest-time 濡備綍鍐冲畾鏀剧疆浣嶇疆銆佸叧绯荤粨鏋勫苟瀹屾垚甯歌鍐欏叆
- `memory evolution`锛氶粯璁や笉鍚姩鏃讹紝宸叉湁璁板繂濡備綍琚澶栧湴閲嶅啓銆佸帇缂┿€佽縼绉汇€佹竻鐞嗗拰鏁寸悊
- `retrieval`锛氱粰瀹?query 濡備綍閫夊彇鐩稿叧璁板繂
- `readout`锛氭绱㈢粨鏋滃浣曡浆鍖栦负 agent 鍙娇鐢ㄧ殑涓婁笅鏂?

杩欓噷鏈€閲嶈鐨勬€濇兂鏄細

**memory system 涓嶆槸涓€涓暣浣撴ā鍧楋紝鑰屾槸涓€鏉＄敱澶氫釜 primitive 缁勬垚鐨勬満鍒堕摼銆?*

涓€鏃﹀皢绯荤粺鎷嗘垚杩欎簺 primitive锛屽氨鍙互鎶娾€滄柟娉曡璁♀€濊浆鍖栦负鈥滄ā鍧楃粍鍚堥棶棰樷€濓紝鎶娾€滄柊鏂规硶鍙戠幇鈥濊浆鍖栦负鈥滈厤缃悳绱㈤棶棰樷€濓紝鎶娾€滅粡楠岃瀵熲€濊浆鍖栦负鈥渕otif 褰掔撼闂鈥濄€?

---

## 杩欎釜椤圭洰鎯宠В鍐充粈涔堥棶棰?

### 1. 缁熶竴鎻忚堪闂

鏈」鐩鍏堟兂瑙ｅ喅鐨勬槸涓€涓〃绀哄眰闂锛氬浣曠敤缁熶竴璇█閲嶈〃杈惧凡鏈?agent memory 鏂规硶銆?

濡傛灉娌℃湁缁熶竴鎻忚堪妗嗘灦锛岄偅涔堜笉鍚屽伐浣滀箣闂村線寰€鍙兘鍋滅暀鍦ㄨ嚜鐒惰瑷€灞傞潰鐨勭矖鐣ユ瘮杈冿紝渚嬪鈥滆繖涓柟娉曠敤浜嗛暱鏈熻蹇嗭紝閭ｄ釜鏂规硶鐢ㄤ簡鍙嶆€濃€濄€備絾杩欐牱鐨勬瘮杈冨苟涓嶈兘鎻ず鏈哄埗宸紓銆?

鎴戜滑甯屾湜灏嗘瘡涓?memory system 鏄庣‘鍒嗚В涓猴細

- 瀹冨浣曞舰鎴愯蹇嗗崟鍏?
- 瀹冨浣曡〃绀鸿蹇?
- 瀹冧綍鏃跺啓鍏?
- 瀹冨浣曠粍缁囧苟甯歌鍐欏叆璁板繂
- 瀹冨浣曟绱笌浣跨敤璁板繂
- 瀹冩槸鍚﹁繘琛岄澶栫殑鍘嬬缉銆佸弽鎬濇垨缁存姢璁板繂

杩欐牱锛屽儚 MemGPT銆丷eflexion銆丄-MEM 绛夌粡鍏哥郴缁燂紝灏卞彲浠ヨ鐪嬩綔鍚屼竴璇█涓殑涓嶅悓閰嶇疆锛岃€屼笉鏄郊姝ゅ绔嬬殑鏂规硶鍚嶃€?

### 2. 鍙瘮杈冮棶棰?

濡傛灉涓嶅悓绯荤粺鑳借閲嶅啓鎴愬悓涓€绉嶇粨鏋勫寲琛ㄧず锛屽氨鍙互鍋氱湡姝ｇ殑鎺у埗鍙橀噺姣旇緝銆?

渚嬪锛?

- 鍥哄畾 `unit formation`锛屽彧姣旇緝涓嶅悓 `retrieval`
- 鍥哄畾 `retrieval`锛屽彧姣旇緝涓嶅悓 `write trigger`
- 鍥哄畾 `representation`锛屾瘮杈冧笉鍚岄澶?`memory evolution`

杩欎娇寰?memory 鐮旂┒浠庘€滄柟娉曞鏂规硶鈥濈殑姣旇緝锛岃浆鍚戔€滄満鍒跺鏈哄埗鈥濈殑姣旇緝銆?

### 3. 鍙悳绱㈤棶棰?

鏈」鐩殑绗簩涓牳蹇冮棶棰樻槸锛?*memory architecture 鑳藉惁鍍?program space 涓€鏍疯绯荤粺鎼滅储锛?*

涓€鏃︽瘡涓?primitive 閮借鍐欐垚鏍囧噯妯″潡鎺ュ彛锛屽苟涓旀瘡涓?slot 閮芥湁鏄庣‘鐨勫€欓€夊疄鐜帮紝閭ｄ箞涓€涓?memory system 灏卞彲浠ヨ鐪嬩綔锛?

`LayeredStoreTopology 脳 PrimitiveChoices 脳 Hyperparameters 脳 Constraints`

杩欐剰鍛崇潃鎴戜滑涓嶅啀鍙槸鎵嬪伐璁捐涓€涓?memory 鏂规硶锛岃€屾槸鍙互锛?

- 鏋氫妇鏌愪竴绫?memory configuration
- 鍦ㄧ害鏉熶笅鑷姩缁勫悎涓嶅悓妯″潡
- 鎼滅储姣旂幇鏈夋枃鐚洿浼樻垨鏇寸ǔ鍋ョ殑閰嶇疆
- 瑙傚療鍝簺 primitive 缁勫悎鍙嶅鍦ㄤ笉鍚屼换鍔′笂鍑虹幇

### 4. 鏈哄埗褰掔撼闂

濡傛灉鎼滅储鑳藉湪澶ц妯￠厤缃┖闂翠腑鎵惧埌澶氫釜楂樻€ц兘 memory systems锛岄偅涔堟洿楂樺眰鐨勯棶棰樺氨鍙樻垚锛?

**楂樻€ц兘 agent memory 鏄惁瀛樺湪 recurring motifs锛?*

渚嬪锛屾湭鏉ュ彲鑳借瀵熷埌锛?

- 寰堝鏈夋晥绯荤粺閮介噰鐢ㄢ€滅粨鏋勫寲鎶藉彇 + selective write + hybrid retrieval鈥?
- 鏌愪簺浠诲姟鏇村亸濂解€渁ppend-only + periodic summarization鈥?
- 鏌愪簺闀挎湡浜や簰鍦烘櫙鏇撮渶瑕佲€渆ntity merge + profile update + extra memory evolution鈥?

杩欑被缁撹涓嶆槸浠庡崟绡囨柟娉曚腑寰楀埌鐨勶紝鑰屾槸浠庣郴缁熸悳绱㈠拰鏈哄埗姣旇緝涓綊绾冲嚭鏉ョ殑銆?

---

## 椤圭洰鐨勭爺绌惰瑙?

`MemPrimitive` 鍏虫敞鐨勬槸 **mechanism-level memory design**锛岃€屼笉鏄叿浣撳伐绋嬪疄鐜般€?

涔熷氨鏄锛岃繖涓」鐩叧蹇冪殑闂涓昏鏄細

- memory system 鐨勬瀯鎴愮淮搴︽湁鍝簺
- 鍚勭淮搴︿箣闂寸殑杈圭晫搴斿浣曞垝鍒?
- 鍝簺妯″潡搴旇鐙珛鍑烘潵浣滀负 primitive
- 妯″潡涔嬮棿鏈夊摢浜涘吋瀹规€х害鏉?
- 鍝簺缁勫悎鏋勬垚鐜版湁缁忓吀宸ヤ綔
- 鍝簺鍖哄煙灏氭湭琚帰绱?

鍥犳锛屾湰椤圭洰涓嶆槸涓€涓崟绾殑鈥渕emory 搴撯€濇垨鈥渁gent 妗嗘灦鈥濓紝鑰屾洿鎺ヨ繎涓€涓爺绌跺熀纭€璁炬柦銆傚畠鐨勪环鍊煎湪浜庢彁渚涳細

- 涓€涓粺涓€鐨勬弿杩拌瑷€
- 涓€涓竻鏅扮殑妯″潡鎺ュ彛灞?
- 涓€涓彲鍒嗘瀽銆佸彲鎼滅储鐨勮璁＄┖闂?
- 涓€涓府鍔╁綊绾?memory 鏈哄埗瑙勫緥鐨勭爺绌惰瑙?

---

## 鏁翠綋绯荤粺瑙?

鏈」鐩皢 agent memory system 鐞嗚В涓轰竴涓甫鏈夊壇浣滅敤鐨勬暟鎹祦绯荤粺銆?

浠庡閮ㄨ緭鍏ュ埌鏈€缁堣 agent 浣跨敤锛宮emory 鐨勭敓鍛藉懆鏈熷ぇ鑷村寘鎷互涓嬮樁娈碉細

1. 澶栭儴 observation 鍒拌揪绯荤粺
2. observation 琚垏鍒嗘垨鎶藉彇涓?memory units
3. memory units 琚紪鐮佷负鍙瓨鍌ㄣ€佸彲绱㈠紩銆佸彲姣旇緝鐨勮〃绀?
4. 绯荤粺鍐冲畾鍝簺 unit 鍊煎緱鍐欏叆
5. unit 琚斁缃埌鍚堥€傜殑 store锛屽苟涓庡凡鏈夎蹇嗗缓绔嬪叧绯?
6. store 琚洿鏂帮紝鍙兘鍙戠敓杩藉姞銆佹浛鎹€佸悎骞舵垨閲嶅啓
7. 鍦ㄥ悗鍙版垨棰濆瑙﹀彂璺緞涓墽琛屽帇缂┿€佹娊璞°€佹暣鐞嗐€侀仐蹇樻垨杩佺Щ
8. 褰?agent 鏈?query 鎴栧綋鍓嶄换鍔￠渶瑕佹椂锛岀郴缁熸绱㈢浉鍏宠蹇?
9. 妫€绱㈢粨鏋滆缁勭粐鎴?agent 鍙秷璐圭殑 readout
10. agent 灏?readout 绾冲叆鑷繁鐨勬帹鐞嗐€佽鍒掑拰鍝嶅簲杩囩▼

杩欎釜瑙嗚寮鸿皟涓ょ偣锛?

- memory 涓嶆槸闈欐€佹暟鎹簱锛岃€屾槸鍔ㄦ€佹紨鍖栫郴缁?
- memory 鐮旂┒涓嶅簲鍙湅 retrieval锛岃€屽簲瑕嗙洊鍐欏叆銆佺粍缁囥€佹娊璞″拰缁存姢鐨勫畬鏁撮棴鐜?

---

## 涓轰粈涔堥渶瑕?DSL

鏈」鐩娇鐢?DSL锛屼笉鏄负浜嗚拷姹傝娉曞舰寮忔湰韬紝鑰屾槸鍥犱负鑷劧璇█鎻忚堪涓嶈冻浠ユ敮鎾戠郴缁熺爺绌躲€?

涓€涓悎鏍肩殑 memory DSL 鑷冲皯搴旇鏀寔鍥涗欢浜嬶細

### 1. 閲嶈〃杈惧凡鏈夋柟娉?

DSL 搴旇鑳芥妸宸叉湁 memory 宸ヤ綔閲嶆柊鍐欐垚缁熶竴閰嶇疆銆傚褰撳墠闃舵鑰岃█锛岄噸鐐逛笉鍐嶅彧鏄寫閫夊皯鏁扮粡鍏稿伐浣滃仛灞曠ず锛岃€屾槸灏藉彲鑳芥妸璋冪爺鑼冨洿浠庣粡鍏镐唬琛ㄦ柟娉曟墿灞曞埌鏇村箍娉涖€佹洿澶氭牱鐨?agent memory 鏂囩尞锛屽苟鐢ㄧ粺涓€璇█鎸佺画鍚哥撼杩欎簺鏂规硶銆?

褰撳墠鏇村叿浣撶殑鐩爣鏄細

- 灏嗚皟鐮旇鐩栭潰鎵╁睍鍒扮害 40 绡囦笉鍚岄鏍肩殑 memory 宸ヤ綔锛岃€屼笉鍙仠鐣欏湪灏戞暟缁忓吀妗堜緥
- 灏介噺璁╁叾涓害 1/4 鐨勬柟娉曞彲浠ヨ **瀹屽叏閲嶈〃杈?* 涓虹粺涓€鐨?primitive 缁勫悎涓庨厤缃?
- 瀵规殏鏃惰繕鏃犳硶瀹屽叏閲嶈〃杈剧殑鏂规硶锛屼篃鑷冲皯鍋氬埌缁撴瀯鍖栨媶瑙ｃ€佸畾浣嶇己澶辨満鍒讹紝骞跺弽杩囨潵鎺ㄥ姩 module 杈圭晫缁х画鎵╁睍

鍙湁杩欐牱锛岃繖濂楄瑷€鎵嶄笉鏄┖娲?taxonomy锛岃€屾槸鐪熸鍏锋湁瑙ｉ噴鍔涖€佹墿灞曟€у拰鏂囩尞鎵胯浇鑳藉姏鐨勭爺绌跺伐鍏枫€?

### 2. 琛ㄨ揪妯″潡缁勫悎

寰堝 memory 鏂规硶涓嶆槸鍗曟ā鍧楀喅绛栵紝鑰屾槸澶氭満鍒剁粍鍚堬紝渚嬪锛?

- 妫€绱腑鍚屾椂浣跨敤 similarity銆乺ecency銆乮mportance
- 涓嶅悓 observation type 浣跨敤涓嶅悓 unit formation
- recall 鍜?rerank 鍒嗕袱闃舵鎵ц

鍥犳 DSL 涓嶈兘鍙敮鎸佲€滃崟瀛楁閰嶇疆鈥濓紝杩樺簲鏀寔缁勫悎銆佺骇鑱斿拰鏉′欢鍒嗗彂銆?

### 3. 鏀寔绾︽潫

memory primitive 涔嬮棿骞堕潪瀹屽叏鑷敱缁勫悎銆?

渚嬪锛?

- graph retrieval 闇€瑕佸浘缁撴瀯鎴栧浘閾炬帴
- entity-based memory evolution 闇€瑕?entity-aware units
- similarity retrieval 闇€瑕?embedding
- hierarchical retrieval 寰€寰€闇€瑕佸眰绾у帇缂╂垨灞傜骇缁勭粐

鎵€浠?DSL 蹇呴』涓嶄粎琛ㄨ揪鈥滈€変粈涔堚€濓紝杩樿琛ㄨ揪鈥滃摢浜涚粍鍚堟槸鍚堟硶鐨勨€濄€?

### 4. 闈㈠悜鎼滅储

濡傛灉鏈潵瑕佸仛鑷姩鎼滅储锛岄偅涔?DSL 閲岀殑姣忎竴涓€夋嫨閮藉簲璇ユ槸鍙灇涓俱€佸彲楠岃瘉銆佸彲閲囨牱鐨勩€備篃灏辨槸璇达紝DSL 搴旇澶╃劧瀵瑰簲涓€涓彲鎿嶄綔鐨?configuration space锛岃€屼笉浠呬粎鏄汉绫诲彲璇荤殑璇存槑鏂囨。銆?

---

## 鏈」鐩腑鐨?design space

鍦?`MemPrimitive` 涓紝memory design space 鐢变笁灞傚叡鍚屾瀯鎴愩€?

### 1. 缁撴瀯灞?

缁撴瀯灞傛弿杩?memory store 鐨勯鏋讹紝鍖呮嫭锛?

- 鏈夊嚑灞?memory layer
- 姣忓眰鐨勪富棰樻槸浠€涔?
- 姣忓眰鏄?`Flat` 杩樻槸 `Graph`
- 鍚勫眰鏈夊摢浜?index

杩欐槸鏁翠釜绯荤粺鐨?structural prior銆傝繖閲岀殑 hierarchical 涓嶅啀琚涓轰竴绉嶅崟鐙殑 store type锛岃€屾槸鈥滃瓨鍦ㄥ灞?layer鈥濊繖涓€缁撴瀯浜嬪疄鏈韩銆?

### 2. 妯″潡灞?

妯″潡灞傛弿杩版瘡涓?primitive slot 閫夋嫨鍝瀹炵幇銆?

渚嬪锛?

- `unit formation` 閫夋嫨 turn-level銆乫act extraction 杩樻槸 multi-granularity
- `write trigger` 閫夋嫨 always write銆乮mportance gate銆丩LM judge 杩樻槸 agent-controlled write
- `retrieval` 閫夋嫨 similarity銆乬raph hop銆丩LM-controlled retrieval 杩樻槸 hybrid retrieval

妯″潡灞傛槸 design space 鐨勪富浣撱€?

### 3. 鍙傛暟灞?

鍙傛暟灞傛弿杩版瘡涓?primitive 鍐呴儴鐨勮秴鍙傛暟銆侀槇鍊煎拰绛栫暐閫夐」銆?

渚嬪锛?

- top-k
- similarity threshold
- decay rate
- summary batch size
- merge policy

鍙傛暟灞傚喅瀹氱殑鏄悓涓€鏈哄埗瀹舵棌鍐呴儴鐨勮涓哄樊寮傘€?

---

## 鎼滅储涓嶆槸绌蜂妇锛岃€屾槸鍙楃害鏉熺殑缁勫悎鎺㈢储

鏈」鐩墍璁炬兂鐨勬悳绱紝涓嶆槸瀵规墍鏈夋ā鍧楄繘琛屾棤宸埆绌蜂妇锛岃€屾槸 **constraint-aware search**銆?

鍘熷洜寰堢畝鍗曪細primitive 铏界劧琚帴鍙ｅ寲浜嗭紝浣嗗畠浠箣闂翠粛鐒跺瓨鍦ㄨ涔夎€﹀悎銆?

渚嬪锛?

- `representation` 鍜?`retrieval` 寰€寰€寮鸿€﹀悎
- `unit formation` 涓?`organization` / `memory evolution` 甯稿父寮鸿€﹀悎
- `organization` 鍐冲畾浜嗗悗缁兘鍚﹁繘琛?graph 鎴?hierarchical retrieval锛屼篃鎵挎媴甯歌鍐欏叆

鍥犳锛屼竴涓湁鏁堢殑鎼滅储绌洪棿蹇呴』鍚屾椂鍖呭惈锛?

- 鍙粍鍚堟€?
- 绫诲瀷涓€鑷存€?
- 鑳藉姏绾︽潫
- 妯″潡闂村吋瀹规€?

鎹㈣█涔嬶紝`MemPrimitive` 杩芥眰鐨勪笉鏄€滃畬鍏ㄨ嚜鐢辩粍鍚堚€濓紝鑰屾槸鈥滃湪褰㈠紡鍖栫害鏉熶笅鐨勫彲鎺х粍鍚堚€濄€?

褰撳墠瀹炵幇澶勪簬涓€娆＄害鏉熸満鍒堕噸鏋勮繃娓℃湡锛?

- `MemoryPipeline` 浠嶇劧浼氬湪鏋勯€犳湡妫€鏌?slot 鎶借薄绫诲瀷涓?`ModuleSpec.slot` 鏄惁瀵归綈
- 鏃х殑 baseline 渚?store/topology eager compatibility check 宸茬Щ闄?
- baseline/runtime 渚х幇宸叉敼涓轰互 `MemoryStore.check()` 涓轰腑蹇冪殑缁勫悎鍚堟硶鎬ф楠岋細`MemoryPipeline` 璐熻矗鎶婃ā鍧楀０鏄庣殑 `requires_contracts` / `produces_contracts` 娉ㄥ唽鍒板叡浜?store锛岄殢鍚庣敱璋冪敤鏂瑰湪闇€瑕佹椂鏄惧紡鎵ц `store.check()`

---

## 杩欎釜椤圭洰甯屾湜瑕嗙洊鍝簺宸叉湁 memory 鐮旂┒

杩欎釜椤圭洰鐨勭洰鏍囦箣涓€锛屾槸璁╀笉鍚岄鏍肩殑 agent memory 鏂规硶閮借兘琚斁鍒颁竴涓粺涓€妗嗘灦涓悊瑙ｃ€傚綋鍓嶉樁娈典細涓诲姩鎶婃枃鐚睜浠庣粡鍏稿伐浣滄墿灞曞埌灏藉彲鑳藉鐨勪唬琛ㄦ€т笌寮傝川鎬у伐浣滐紝鐩爣瑙勬ā绾︿负 40 绡囷紱鍏朵腑浼氶噸鐐规寫閫夌害 1/4 浣滀负瀹屽叏閲嶈〃杈惧璞★紝鍏朵綑宸ヤ綔涔熶細绾冲叆缁熶竴鎷嗚В涓庡鐓у垎鏋愩€傝鐩栬寖鍥村寘鎷絾涓嶉檺浜庯細

- 浠?observation stream 涓烘牳蹇冪殑 episodic memory
- 浠ヤ簨瀹炴娊鍙栥€佸睘鎬ф洿鏂颁负鏍稿績鐨?semantic/profile memory
- 浠ュ弽鎬濄€佹€荤粨鍜屾娊璞′负鏍稿績鐨?reflective memory
- 浠ユ妧鑳姐€佷唬鐮佺墖娈典负鏍稿績鐨?skill memory
- 浠ュ浘鑺傜偣涓庡叧绯婚摼鎺ヤ负鏍稿績鐨?graph memory
- 浠?agent 涓诲姩宸ュ叿璋冪敤涓烘牳蹇冪殑 self-managed memory
- 浠ュ伐浣滆蹇嗐€侀暱鏈熻蹇嗐€佸綊妗ｈ蹇嗗垎灞備负鏍稿績鐨?multi-store memory

杩欎簺绯荤粺铏界劧琛ㄩ潰褰㈠紡涓嶅悓锛屼絾鐞嗘兂鐘舵€佷笅閮藉簲琚繕鍘熶负涓€缁?primitive 鐨勪笉鍚屽彇鍊煎拰涓嶅悓缁勫悎鏂瑰紡銆傝嫢鐜版湁妯″潡杩樹笉瓒充互鎵胯浇鏌愪竴绫绘柟娉曪紝閭ｄ箞鎵╁睍 module 鏈韩灏辨槸褰撳墠闃舵鐨勯噸瑕佺爺绌跺伐浣滐紝鑰屼笉鏄緥澶栨儏鍐点€?

---

## 杩欎釜椤圭洰鏈€缁堟兂浜у嚭浠€涔?

浠庣爺绌剁洰鏍囦笂鐪嬶紝`MemPrimitive` 甯屾湜鏈€缁堟敮鎸佷互涓嬪嚑绫讳骇鍑恒€傚氨褰撳墠闃舵鑰岃█锛屾渶浼樺厛鐨勪腑鏈熺洰鏍囨槸锛氭妸鏂囩尞瑕嗙洊闈㈡墿灞曞埌绾?40 绡囷紝骞跺敖閲忎娇鍏朵腑绾?1/4 鑳藉琚瘮杈冩湁璇存湇鍔涘湴澹扮О涓衡€滃畬鍏ㄩ噸琛ㄨ揪鈥濓紱涓烘锛岄」鐩細缁х画鎵╁睍 module families 涓庣粍鍚堣竟鐣岋紝璁╂洿澶氳繃鍘绘柟娉曞彲浠ヨ惤鍒扮粺涓€妗嗘灦涓€?

### 1. 涓€濂楃粺涓€鐨?memory 鎻忚堪璇█

鐢ㄤ簬琛ㄨ揪銆佹瘮杈冨拰閲嶆瀯宸叉湁 agent memory 绯荤粺銆?

### 2. 涓€濂楁満鍒剁骇 primitive ontology

鏄庣‘ memory system 鐢卞摢浜涘熀纭€鏈哄埗缁勬垚锛屾瘡绉嶆満鍒舵湁鍝簺瀹炵幇鍙樹綋锛屽畠浠箣闂村浣曞吋瀹规垨鍐茬獊銆?

### 3. 涓€涓彲鎼滅储鐨?memory configuration space

浣?memory design 浠庢墜宸ュ惎鍙戝紡璁捐锛岃浆鍚戝彲绾︽潫鐨勭郴缁熸帰绱€?

### 4. 涓€缁?recurring motifs

閫氳繃瀵圭粡鍏告柟娉曞拰鎼滅储缁撴灉鐨勫垎鏋愶紝褰掔撼鍑洪珮鎬ц兘 memory systems 涓弽澶嶅嚭鐜扮殑璁捐妯″紡銆?

### 5. 涓€涓柊鐨勭爺绌堕棶棰樻鏋?

鍗虫妸 agent memory 鐮旂┒浠庘€滄彁鍑轰竴涓柊鏋舵瀯鈥濓紝鎺ㄨ繘鍒扳€滅爺绌?memory design space 鐨勭粍缁囪寰嬨€佺粨鏋勫亸缃笌鎼滅储鏂规硶鈥濄€?

---

## 鏈粨搴撳綋鍓嶆枃妗ｇ殑鍒嗗伐

- `memprimitive/baselines/README.md`
  璇存槑闃舵涓€ baseline 浠ｇ爜濡備綍鎸?primitive slot 鎷嗗垎鍒板涓?`.py` 鏂囦欢銆乣__init__.py` 涓?`simple.py` 鐨勫鍑哄叧绯伙紝浠ュ強鎵╁睍鏂板疄鐜版椂鐨勭害瀹氥€?  褰撳墠 `RecencyRetrieval` 璇箟宸插浐瀹氫负鈥滅洿鎺ヨ繑鍥炴渶鏂扮殑 top-k 璁板綍鈥濓紝涓嶅啀鏍规嵁 query token 鍏堝仛鏂囨湰杩囨护銆?  褰撳墠 trigger-family baseline 宸茶鐩?metadata-gated write銆乲ey-ready write銆乷utcome-conditioned evolution trigger 涓?new-write local-maintenance trigger 杩欑被闈?graph motif锛屽彲鐩存帴鐢ㄧǔ瀹氱被鍚嶅拰 builder 琛ㄨ揪 TiM銆丷eflexion銆丮emGPT 椋庢牸瑙﹀彂璇箟銆?  鍚屾椂锛孯eflexion-like back-half motif 鐜板湪涔熷凡鏈夐€氱敤 baseline锛歚PlacementWithoutAppendOrganization`銆乣ReflectionGenerationEvolution`銆乣BufferRetrieval`銆乣PromptContextReadout`锛宑lassic Reflexion wrapper 鐩存帴澶嶇敤杩欎簺 slot-level 瀹炵幇銆?
- `memprimitive/example/demonstration/README.md`
  姹囨€诲彲杩愯 demonstration锛屽寘鎷け璐?/ 鎴愬姛 trial 鐨?Reflexion 椋庢牸瑙﹀彂銆佸畬鏁寸殑 failed-trial -> reflection -> next-recall context Reflexion-like 闂幆锛屼互鍙?partition-ready local maintenance 鐨?TiM 椋庢牸瑙﹀彂婕旂ず銆?
  杩欎簺 demonstration 榛樿搴斾互鏈€绠€娲佺殑 `MemoryPipeline + module composition` 褰㈠紡灞曠ず DSL 鐢ㄦ硶锛岃€屼笉鏄緷璧栨洿楂樺眰鐨?workflow 灏佽銆?

- `DSLIO.md`
  璁ㄨ memory system 鍚勬ā鍧楃殑鏍囧噯杈撳叆杈撳嚭鎺ュ彛锛屾槑纭郴缁熶腑鐨勫叡浜璞°€佹ā鍧楃鍚嶃€佸壇浣滅敤涓庤兘鍔涚害鏉熴€?

- `DSLgrammar.md`
  灏嗘帴鍙ｅ眰鎶借薄鎴愬０鏄庡紡璇█锛屽畾涔?memory system 鐨勫舰寮忚娉曘€佺粍鍚堢畻瀛愩€佺害鏉熻〃杈炬柟寮忥紝浠ュ強濡備綍鐢ㄨ璇█閲嶈〃杈剧粡鍏告柟娉曘€?

- `Primitives.md`
  鏋氫妇姣忎釜 primitive slot 鐨勫彲鑳藉疄鐜帮紝鍒荤敾鎼滅储绌洪棿銆佸吋瀹规€х害鏉熴€佸凡鏈夊伐浣滃垎瑙ｅ拰娼滃湪鏈帰绱㈢粍鍚堛€?

杩欎笁涓枃妗ｅ叡鍚屾瀯鎴愪簡椤圭洰鐨勭爺绌跺熀纭€锛?

- `DSLIO.md` 鍥炵瓟鈥滄ā鍧楁帴鍙ｆ槸浠€涔堚€?
- `DSLgrammar.md` 鍥炵瓟鈥滅郴缁熷浣曡璇█鍖栨弿杩扳€?
- `Primitives.md` 鍥炵瓟鈥滄悳绱㈢┖闂撮噷鍒板簳鏈変粈涔堚€?

---

## 閫傜敤鍦烘櫙

`MemPrimitive` 閫傚悎浠ヤ笅鍑犵被鐮旂┒宸ヤ綔锛?

- 鎯崇郴缁熸暣鐞?agent memory 鏂囩尞鐨勪汉
- 鎯虫瘮杈冧笉鍚?memory 鏈哄埗鑰屼笉鏄粎姣旇緝瀹屾暣鏂规硶鐨勪汉
- 鎯虫妸宸叉湁 memory 鏋舵瀯閲嶅啓鎴愮粺涓€褰㈠紡鐨勪汉
- 鎯冲紑灞?memory architecture search 鐨勪汉
- 鎯充粠鎼滅储缁撴灉涓綊绾?recurring motifs 鐨勪汉
- 鎯崇爺绌?long-term memory銆乻emantic memory銆乺eflective memory銆乻kill memory 涔嬮棿鍏崇郴鐨勪汉

---

## 涓嶅湪褰撳墠鑼冨洿鍐呯殑鍐呭

鏈」鐩綋鍓嶅叧娉ㄧ殑鏄郴缁熻璁′笌鐮旂┒鎶借薄锛屽洜姝ゆ殏涓嶆妸閲嶇偣鏀惧湪浠ヤ笅鏂归潰锛?

- 鍏蜂綋宸ョ▼瀹炵幇缁嗚妭
- 鍏蜂綋妯″瀷璋冪敤涓庝唬鐮佺粍缁?
- 鍏蜂綋 benchmark 鐨勮繍琛岃剼鏈?
- 鐗瑰畾妗嗘灦涓嬬殑閮ㄧ讲鏂瑰紡
- 闈㈠悜鐢熶骇鐜鐨勪紭鍖栫粏鑺?

杩欎簺鍐呭鏈潵閮藉彲浠ュ缓绔嬪湪褰撳墠鐮旂┒鎶借薄涔嬩笂锛屼絾涓嶆槸褰撳墠闃舵鐨勬牳蹇冪洰鏍囥€?

---

## 涓€鍙ヨ瘽姒傛嫭

`MemPrimitive` 鎯冲仛鐨勪簨鏄細

**鎶?agent memory 浠庘€滄柟娉曢泦鍚堚€濋噸鏋勪负鈥滃彲缁勫悎銆佸彲鎼滅储銆佸彲褰掔撼鐨勬満鍒剁┖闂粹€濓紱鑰屽綋鍓嶉樁娈垫渶鏍稿績鐨勭洰鏍囷紝鏄妸灏藉彲鑳藉鐨勬棦鏈夋柟娉曠撼鍏ョ粺涓€閲嶈〃杈炬鏋朵腑锛屾墿灞曞埌绾?40 绡囨枃鐚紝骞朵簤鍙栧叾涓害 1/4 杈惧埌瀹屽叏閲嶈〃杈俱€?*
