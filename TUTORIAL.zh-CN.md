# MemPrimitive 闂佽桨鐒﹂悷褔鍩㈡總鍛婃櫖闁割偁鍨婚弳鍡涙煕閺嶃劎澧卞褎顨婇弫?
闁哄鏅滈悷銈夊船閵堝鏋佹繛鍡樺灍閺屻倝鎮楅悽鍨殌缂併劏灏妵鎰板箻閸愬樊鏋€婵炲濮甸幐鍝ヨ姳闁秵鍎嶉柛鏇ㄥ亞閸炪劎鐥娑樹壕闁哄鏅滈崝姗€銆侀幋锔筋棃妞ゎ偒鍏橀崑?
## 闂佺绻愰悧鎰邦敊閸ャ劍濯撮煫鍥ф捣椤忓崬霉閻樿櫕鏋勭紒?
`trigger` 閻庤鐡曠亸娆戝垝閿熺姴姹查柛灞剧⊕椤ρ呯磼閻橆偄浜鹃梺鍛婄墬閻楊厾妲愬┑鍥┾枖鐎广儱鎳庨弲娆撴偣?`signal / scorer / gate / policy` 闂佹悶鍎茬粙鎺楊敊鐏炵瓔鍤曢煫鍥ュ劤缁€澶娾槈閺冨倸鏋嶇紒妤€顦靛畷妯侯吋閸偄鏁?`ComposeTrigger`闂?
閻熸粎澧楅幐鍛婃櫠閻樿鐭楁い蹇撳暟缁犱粙鏌ｉ敐鍡欐噧婵炲弶鎸荤粙澶愵敂閸曨厼鏁ょ紒缁㈠幐閸?trigger闂?
- `AlwaysTrigger`
- `ThresholdTrigger`
- `NeverTrigger`
- `ThresholdTrigger`

## 1. 闂佸搫鐗冮崑鎾绘倶韫囨挾绠绘俊顐ｆ尦閹?
```python
from memprimitive import MemoryPipeline, Observation, Query
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)

pipeline = MemoryPipeline(
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(),
    write_trigger=AlwaysTrigger(),
    organization=AppendOrganization(),
    retrieval=RecencyRetrieval(top_k=2),
    readout=ConcatenateReadout(),
)

pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="notes"))
readout = pipeline.recall(Query(text="Alice likes what?"))
print(readout.text)
```

闁哄鏅滈悷鈺呭闯閻戣姤鍎嶉柛鏇ㄥ亝閸庢挸鈽夐弬娆炬Ч婵″弶鎮傞弫?
- `PassThroughUnitFormation()`闂佹寧绋掔喊宥嗕繆?observation 闂佺儵鏅涢悺銊ф暜閹绢喖鐭楁俊顖濐嚙閻忓洤鈽夐幘顖氫壕婵?unit
- `BasicRepresentation()`闂佹寧绋掓穱楦挎＂闂佺绻愰幊搴ㄦ偄閳ь剛鐥娑樹壕闁荤偞绋忛崝搴ㄥΦ?- `AlwaysTrigger()`闂佹寧绋掔喊宥嗘櫠瀹ュ瀚?unit 闂備緡鍠涘Λ鍕储閹寸姵濯肩紒瀣仢閺呮悂鏌?- `AppendOrganization()`闂佹寧绋掑銊ワ耿閳ユ剚娼伴柨婵嗙墛閸庝即骞?- `RecencyRetrieval()`闂佹寧绋掔喊宥団偓鍨矒瀵噣宕奸弴鐕傜吹闂佸憡鐟﹂悧鏇灻?- `ConcatenateReadout()`闂佹寧绋掔喊宥嗕繆閸濄儳纾奸柟鎯ь嚟娴滎垶鏌熷畡鏉挎倯闁搞劍宀稿顒勫炊閵婏附瀚?
## 2. 闂佹椿娼块崝鎴澪ｉ崶顒€纾归柤褰掓緩閺囥垹鐭楅柟瀛樼箖閻?
### 闂佸憡鍔栭悷銉╁矗閸℃稒鈷撻柛顐ｇ妇閸?
```python
from memprimitive.baselines import ThresholdTrigger

write_trigger = ThresholdTrigger(threshold=0.5, constant=1.0)
```

闁哄鏅滈悷銊╁Υ閸愵亞鐭嗛柣姘嚟缁?
- 婵犵鈧啿鈧綊鎮?`constant >= threshold`闂佹寧绋戦張顒佹櫠瀹ュ瀚?unit 闂佸憡鍔栭悷銉╁矗?- 闂佸憡鐔粻鎴﹀垂椤栫偛绠ラ柍褜鍓熷?unit 闂備緡鍠楅崹婵堢箔婢舵劕绀?
### Evolution 闂傚倸鍟悧鍡涘焵?
```python
from memprimitive.baselines import ThresholdTrigger

evolution_trigger = ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0)
```

闁哄鏅滈悷銊╁Υ閸愵亞鐭嗛柣姘嚟缁?
- 婵犵鈧啿鈧綊鎮?`constant >= threshold`闂佹寧绋戦張顒佹櫠瀹ュ瀚?unit 闂?`decisions=True`
- 闂佸憡鐔粻鎴﹀垂椤栨稓鈻?`False`

## 3. Graph baseline 缂傚倷绀佺€氼剟骞?
```python
from memprimitive import MemoryPipeline
from memprimitive.baselines import (
    BasicRepresentation,
    GraphAppendOrganization,
    GraphNeighborAppendEvolution,
    GraphReadout,
    GraphSeedAndExpandRetrieval,
    ThresholdTrigger,
    TripleRepresentation,
)

pipeline = MemoryPipeline(
    representation=(
        BasicRepresentation(elements=("text",)),
        TripleRepresentation(method="direct"),
        BasicRepresentation(elements=("tags", "keywords")),
    ),
    organization=GraphAppendOrganization(target_layer="knowledge_graph"),
    evolution_trigger=ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0),
    memory_evolution=GraphNeighborAppendEvolution(target_layer="knowledge_graph"),
    retrieval=GraphSeedAndExpandRetrieval(layer="knowledge_graph"),
    readout=GraphReadout(),
)
```

闁哄鏅滈悷鈺呭闯閻戣姤鍎嶉柛鏇ㄥ灠濞呫垽鏌ｉ幇鎵冲亾濞戞氨鎲归梺鍝勫閸ㄤ即藝閺屻儱绾?trigger闂佹寧绋戦惌渚€鍩€椤掆偓閺堫剙危閹间焦鏅?
- 闂?graph organization 闂佸憡鍔栭悷銉╁矗閸℃稑鐐婇柟顖嗗懏鍕?- 闂佹椿娼块崝宀勬偄閳ь剛鐥娑樹壕 evolution trigger 闂佺懓鐏氶幐鍝ユ閹达箑瑙﹂幖杈剧悼閺?graph evolution
- 闂?graph retrieval + graph readout 闂佺顑嗛懝鎹愩亹椤愶箑鐐?
## 4. 閻熸粎澧楅幐鍛婃櫠閻樺磭鈻旂€广儱鎳庨弲娆撴煙閹帒濮€鐎规洘锕㈤幆鍐礋椤愩垺鏆ラ柣?
婵炲濮伴崕鎵箔閸涙潙绀冮柛娑卞弾閸熷洨鈧鐡曠亸娆戝垝閳╁啰鈻旂€广儱鎳愬锝吤瑰鍐€楃紓宥咁儔瀹?baseline 闂佽桨鐒﹂悷褔鍩㈡總鍛婃櫖?
- `compose_write_trigger(...)`
- `compose_evolution_trigger(...)`
- `signal / scorer / gate / policy`
- `OutcomeConditionedEvolutionTrigger`
- `NewWriteEvolutionTrigger`
- `NeighborExistsEvolutionTrigger`
- `LLMJudgedWriteTrigger`
- `MetadataGatedWriteTrigger`
- `KeyReadyWriteTrigger`

婵犵鈧啿鈧綊鎮樻径瀣闁绘ê寮堕煬顒勬煛閸愬嫬瀚悗顔戒繆濡ゅ绁烽柍褜鍏涢悞锕€螞?notebook闂侀潧妫旈悞锕€螞椤愩倗鐭嗛弶鐐村娴兼劙姊洪幓鎺旂伇婵犫偓閸涙潙绀嗛柟宄扮灱缁犵懓霉濠婂嫮鈽夐柟顔芥礈閳ь剚绋掗〃鎰濠靛牊瀚氶梺鍨儏鎯熼柣搴ｆ嚀閸熲晛顭ㄩ幋锔藉仩闁糕剝顕撮幒鎾垛枖闁告繂瀚崸濠囨煕濞嗘帒钄兼い鏃€娲滈幏瀣焺閸愩劎浠愰悗鍨緲閹冲繒绮╂繝姘仩闁糕剝绋愮划锝夋煟閿濆棛鎳曠紒杈ㄧ箞閹虫挾浠﹂懖鈺冩喒闂佸搫瀚烽崹鎵礊鐎ｎ喖绀堢€广儱妫欏▓宀€绱掔€ｎ亶鍎忕紒銊ㄥ皺缁辨帟顦撮柣銏㈢帛閹峰懐鎹勯妸锔芥闂佹眹鍔岀€氼剟宕ｉ弴鐑嗗殨闁?API闂?
