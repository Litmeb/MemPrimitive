# `memprimitive.example.demonstration`

鏉╂瑤閲滈惄顔肩秿閺€鍓ф畱閺勵垰褰查惄瀛樺复鏉╂劘顢戦惃?demonstration 閼存碍婀伴敍宀€鏁ら弶銉ョ潔缁€?`MemoryPipeline`閵嗕梗MemoryStore`閵嗕攻aseline primitives閿涘奔浜掗崣?layered store / graph family 缁涘婧€閸掕泛顩ф担鏇氫簰閺堚偓鐏忓繘妫撮悳顖涙煙瀵繒绮嶉崥鍫ｆ崳閺夈儯鈧?
瀵ら缚顔呴崷銊ょ波鎼存挻鐗撮惄顔肩秿鏉╂劘顢戦敍?
```text
python -m memprimitive.example.demonstration.<濡€虫健閸?
```

娓氬顩ч敍?
```text
python -m memprimitive.example.demonstration.minimal_pipeline
```

## 瑜版挸澧犵拠瀛樻

trigger 鐎涙劗閮寸紒鐔峰嚒缂佸繑娈忛弮鑸垫暪缂傗晙璐熼崺铏诡攨 slot trigger閿?
- `AlwaysTrigger`
- `NeverTrigger`

濮濄倕澧犻崺杞扮艾 `signal / scorer / gate / policy` 閻?trigger-family 缂佸嫬鎮庣粈杞扮伐閿涘奔浜掗崣濠佺贩鐠ф牞绻栨禍娑氱矎閸?trigger 閻?Reflexion / TiM / A-MEM 妞嬪孩鐗?demonstration閿涘苯鍑＄紒蹇庣矤閺堫剛娲拌ぐ鏇犘╅梽銈冣偓?
## 閸╄櫣顢?pipeline / store 濠曟梻銇?
| 濡€虫健閺傚洣娆?| 鐠囧瓨妲?|
| --- | --- |
| `minimal_pipeline.py` | 閺堚偓鐏?ingest -> recall 闂傤厾骞嗛妴?|
| `topology_store.py` | 鐏炴洜銇?`StoreTopology` / `MemoryStore` 閻ㄥ嫬顦跨仦鍌涘珖閹垫垵锛愰弰搴涒偓浣瑰瘻鐏炲倸鍟撻崗銉ユ嫲閹稿鐪伴弻銉嚄閵?|
| `embedding_similarity_retrieval.py` | 鐏炴洜銇氱敮?`embedding` 閻ㄥ嫯銆冪粈鍝勭湴閿涘奔浜掗崣?`EmbeddingSimilarityRetrieval` 閻ㄥ嫭顥呯槐銏ｎ攽娑撴亽鈧?|

## Layered / routing / dispatch 濠曟梻銇?
| 濡€虫健閺傚洣娆?| 鐠囧瓨妲?|
| --- | --- |
| `layer_aware_semantic_working.py` | 閸愭瑥鍙?`working` / `semantic` 娑撱倕鐪伴敍灞借嫙閻?`LayerAwareRetrieval` 閹稿鐪扮紒鍕値娑撳秴鎮撳Λ鈧槐銏犳珤閵?|
| `layer_aware_working_graph.py` | 閸掑棛顬?working layer 閸?graph layer閿涘苯鍟€闁俺绻?layer-aware retrieval 缂佺喍绔撮崣顒€娲栭妴?|
| `conditional_layer_routing.py` | `ConditionalLayerOrganization` 閹稿顫夐崚娆愬Ω娑撳秴鎮?unit 鐠侯垳鏁遍崚棰佺瑝閸?layer閵?|
| `dispatch_organization_recall.py` | `DispatchOrganization` 閸氬本妞傛す鍗炲З婢舵矮閲?organization child閵?|
| `dispatch_organization_trace.py` | 娑撳簼绗傛笟瀣娴肩》绱濇担鍡樻纯瀵缚鐨?trace 鏉堟挸鍤妴?|

## Graph baseline 濠曟梻銇?
| 濡€虫健閺傚洣娆?| 鐠囧瓨妲?|
| --- | --- |
| `graph_append_entity_retrieval.py` | `GraphAppendOrganization` + `EntityRetrieval` 閻ㄥ嫯浜ら柌?graph 缁€杞扮伐閵?|
| `graph_baseline_pipeline.py` | graph baseline 闂傤厾骞嗛敍姝欸raphAppendOrganization -> AlwaysTrigger -> GraphNeighborAppendEvolution -> GraphSeedAndExpandRetrieval -> GraphReadout`閵?|

## 閹恒劏宕橀梼鍛邦嚢妞ゅ搫绨?
1. `minimal_pipeline.py`
2. `topology_store.py`
3. `layer_aware_semantic_working.py`
4. `dispatch_organization_recall.py`
5. `graph_baseline_pipeline.py`
