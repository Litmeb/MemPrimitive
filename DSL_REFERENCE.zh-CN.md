# MemPrimitive DSL 闂佸憡鐟ラ崐浠嬪焵椤掆偓閸犳稓妲愬▎鎺嬩汗闁规儳鍟块·鍛存倵閸︻厼浠ф鐐叉喘濡啴锝為鍓ь槴

闁哄鏅滈悷銈夊船閵堝妫橀柛銉ｅ妸閳ь剙鍊垮畷锝夘敂閸涱厺绱熼柡澶嗘櫊椤ｏ妇鍒掗妸锔藉劅闁规崘顕у▍锛勬喐閻楀牊灏褏濮电粋鎺戭吋閸パ冃侀梺绋挎禋閸撴瑧妲愰幋锕€违濞达絿顭堢拋鏌ユ煟閳轰胶鎽犻悽顖氼嚟缁辨帡宕熼锝呪偓銈夋煟?DSL-like 闁哄鏅滈崝姗€銆侀幋锕€绫嶉柟顖濆焽閳ь剙鍟村Λ鍐綖椤戣棄浜?
## 閻熸粎澧楅幐鍛婃櫠閻樼數纾奸柟鎹愵嚃閸?
閻熸粎澧楅幐鍛婃櫠?`trigger` 闁诲繒鍋涢崐鎼佸礄閿涘嫮纾奸煫鍥ㄦ⒐閻ｎ垶鏌￠崘顓炵厫闁轰礁婀辩槐鎾诲煛閳ь剛鎷归悢鐓庢槬闁惧繗顕栭弨?slot trigger闂佹寧绋戞總鏃傜箔婢舵劕绀冪€广儱鎳庤ぐ鍡欌偓娈垮枓閸?`signal / scorer / gate / policy` 缂傚倷绀佸Λ妤呭垂鎼淬劍鏅悘鐐跺亹閻﹀秴鈽夐幘宕囆㈤柛鐔插亾闂佺娴氶崜娆戞?`compose_*trigger(...)` 闂佺粯绮嶅妯衡攦閳ь剛绱撴担绋款伂妞ゃ儱锕﹂幑鍕敍濮樿京鐛梺?
婵烇絽娲︾换鍕汲閳ь剟鏌?trigger API 闂佸憡鐟禍婵嗭耿娓氣偓閺?
- `AlwaysTrigger()`
- `ThresholdTrigger(threshold=..., constant=...)`
- `NeverTrigger()`
- `ThresholdTrigger(slot="evolution_trigger", threshold=..., constant=...)`

## Pipeline Slots

闂佸搫绉村ú銈夊闯椤栨稏浜滈柛婵嗗绾板秴霉閻樿尙肖闁告枮鍥у強妞ゆ牓鍊楃粣?
```text
unit_formation
-> representation
-> write_trigger
-> organization
-> evolution_trigger
-> memory_evolution
-> retrieval
-> readout
```

## Trigger Slots

### `write_trigger`

闁哄鐗婇幐鎼佸吹椤撱垹绀冩繛鍡楃箰閻?`packet.decisions`闂?
#### `AlwaysTrigger`

- 婵炶揪绲剧划搴ㄥ极閵堝鏅慨姗嗗幖椤ｆ煡鏌?unit 闂備緡鍠涘Λ鍕疮閹捐绀?- 闁哄鐗婇幐鎼佸矗閸℃稒鏅慨婵囶劒acket.units`
- 闁哄鐗婇幐鎼佸吹椤撱垺鏅慨婵囶劒acket.decisions`

```python
AlwaysTrigger()
```

#### `ThresholdTrigger`

- 婵炶揪绲剧划搴ㄥ极閵堝鏅慨姗嗗亜閳诲繘鏌ｉ～顒€濡奸柣鏍х埣閺?`constant` 闂佸憡绮岄惌鍌毼ｉ崶顒€纾?`threshold` 婵炲瓨绫傞崨顔芥缂傚倷鑳堕崰宥囩博閹绢喖绀冩繛鍡楃箰瀵娊鏌涢幇顒傦紞闁?- 闁哄鐗婇幐鎼佸矗閸℃稒鏅慨婵囶劒acket.units`
- 闁哄鐗婇幐鎼佸吹椤撱垺鏅慨婵囶劒acket.decisions`

```python
ThresholdTrigger(slot="evolution_trigger", threshold=0.5, constant=1.0)
```

### `evolution_trigger`

闁哄鐗婇幐鎼佸吹椤撱垹绀冩繛鍡楃箰閻?`packet.decisions`闂?
#### `NeverTrigger`

- 婵炶揪绲剧划搴ㄥ极閵堝鏅慨妯哄鐢盯鎮规担闈涒偓鎾剁箔婢跺本鍠嗛柨鏇楀亾鐟滄澘鍊归敍鎰攽閸♀晜鈻?evolution
- 闁哄鐗婇幐鎼佸矗閸℃稒鏅慨婵囶劒acket.units`闂侀潧妫斿姊cket.placements`
- 闁哄鐗婇幐鎼佸吹椤撱垺鏅慨婵囶劒acket.decisions`

```python
NeverTrigger()
```

#### `ThresholdTrigger`

- 婵炶揪绲剧划搴ㄥ极閵堝鏅慨姗嗗亜閳诲繘鏌ｉ～顒€濡奸柣鏍х埣閺?`constant` 闂佸憡绮岄惌鍌毼ｉ崶顒€纾?`threshold` 婵炲瓨绫傞崨顔芥缂傚倷鑳堕崰宥囩博?evolution 闂佸憡鍔曠壕顓㈡偤?- 闁哄鐗婇幐鎼佸矗閸℃稒鏅慨婵囶劒acket.units`闂侀潧妫斿姊cket.placements`
- 闁哄鐗婇幐鎼佸吹椤撱垺鏅慨婵囶劒acket.decisions`

```python
ThresholdTrigger(threshold=0.5, constant=1.0)
```

## Trace 闁荤姴娲ら悺銊ノ?
婵炲濮寸粔鍫曞礉瑜庣粚鍗炩攽閸喐鐣梺?
- `trace["write_trigger"]`
- `trace["evolution_trigger"]`

婵炶揪绲藉Λ鏃傜箔婢舵劕绀冪€广儱妫楅ˉ鐐烘偣閸モ晛鍔嬫俊顐亰閹?trigger-family 闁诲孩绋掗〃鍡涱敊瀹€鍕櫖閻忕偠鍋愭导鎰攽閳ュ啿浜扮紒?
- `signals`
- `scorer`
- `gate`
- `policy`
- `family`

閻熸粎澧楅幐鍛婃櫠閻樿鏄ラ柧蹇氼嚃閺€?trigger trace 闂佸憡鐟禍娆戞崲濮樿鲸瀚氬ù锝堫潐娴犳﹢鎮樿箛鎾剁缂佺儵鍋撻柣鐔告磻濡炴帗绌辨繝鍥х畳妞ゆ牓鍊楃粣?
- `module`
- `decisions`
- `constant`
- `threshold`
- `per_unit`

## 缂備讲鍋撻弶鐐村娴?
### 闂佸搫鐗冮崑鎾绘倶?pipeline

```python
from memprimitive import MemoryPipeline
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
```

### Graph baseline

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

## 闁荤姴娲ら悺銊ノ?
婵?`Reflexion`闂侀潧妫斿妗礶mGPT`闂侀潧妫斿妗?MEM` 缂備焦绋戦ˇ杈╁垝閿熺姴绀傜紒瀣仒缁憋絽霉閿濆棛鐭嬬紒渚婄畵閹嫮绮欓崹顔肩稑闂佹眹鍔岀€氼厼煤鐠恒劎纾?trigger 闁荤姴娴傞崢铏圭不閻斿吋鏅€光偓閳ь剙煤娴兼潙绀堢€广儱瀚悷婵嬫煕閹邦剛肖闁?baseline trigger API 闂佺娴氶崜娆戞閹寸姵鍋橀柕濞垮妽瑜把囨煥濞戞瑧鈽夋い鈹洤鍑犳繝濞惧亾缂侇喓鍔嶉幆鏃堝箻閼艰泛骞€婵炲濮寸粔闈涳耿娓氣偓瀵喛顦查柟顔芥礈閳ь剚绋掗〃鍛村吹椤撱垺鍋濋悽顖ｅ枤缁€澶愬箹鐎涙ɑ顥嗘い顐㈩儐缁嬪宕崟顐㈡綉闂佸憡鐟ュ鍫曟偩椤愶附鍋╂繛鍡樺姇閻?classic 闁诲繒鍋涢崐鎼佹儍濠靛牊鍟戦柛婊冨暟缁€澶愭煠閺夊灝顨欑紒妤€顦靛浼搭敍濮樿京歇闂?baseline DSL 闂佹眹鍔岀€氼剟宕ｉ弴鐑嗗殨闁逞屽墴閹虫鎸婃径濠庢綕闂?
