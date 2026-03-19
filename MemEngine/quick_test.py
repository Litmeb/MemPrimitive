import json
from pathlib import Path

from memengine import MemoryConfig, FUMemory, STMemory, LTMemory, MBMemory


def make_common_config(*, usable_gpu: str = "", display_method: str = "ScreenDisplay") -> dict:
    return {
        "global_config": {"usable_gpu": usable_gpu},
        "storage": {},
        "display": {
            "method": display_method,
            "prefix": "----- Current Memory Start (%s) -----",
            "suffix": "----- Current Memory End -----",
            "key_format": "(%s)",
            "key_value_sep": "\n",
            "item_sep": "\n",
            "output_path": "logs/sample.log",
        },
        "recall": {
            "truncation": {"method": "LMTruncation", "mode": "word", "number": 256, "path": ""},
            "utilization": {
                "method": "ConcateUtilization",
                "prefix": "[Memory Start]",
                "suffix": "[Memory End]",
                "list_config": {"index": True, "sep": "\n"},
                "dict_config": {"key_format": "(%s)", "key_value_sep": "\n", "item_sep": "\n"},
            },
            "empty_memory": "None",
        },
        "store": {},
    }


def make_text_retrieval_config(*, topk: int = 5, st_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict:
    return {
        "method": "TextRetrieval",
        "encoder": {"method": "STEncoder", "name": st_model, "dimension": 384, "path": st_model},
        "mode": "cosine",
        "topk": topk,
    }


def run_memory(memory, obs_list, query_text, *, with_time: bool = False, fixed_time: int | None = None):
    memory.reset()
    for i, obs in enumerate(obs_list):
        if with_time:
            t = fixed_time if fixed_time is not None else i
            memory.store({"text": obs, "time": t})
        else:
            memory.store(obs)
    return memory.recall(query_text)


def main():
    data_path = Path(__file__).with_name("locomo10.json")
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    qa = raw[0]["qa"]
    qa_valid = [x for x in qa if "answer" in x]
    observations = [f"Q: {x['question']}\nA: {x['answer']}" for x in qa_valid]
    query = qa_valid[0]["question"]

    # FUMemory
    fu_cfg = make_common_config()
    fu_cfg["name"] = "FUMemory"
    fu_cfg["store"] = {"method": "FUMemoryStore"}
    fu_cfg["recall"]["method"] = "FUMemoryRecall"
    fu = FUMemory(MemoryConfig(fu_cfg))
    print("FUMemory:", run_memory(fu, observations[:30], query))

    # STMemory
    st_cfg = make_common_config()
    st_cfg["name"] = "STMemory"
    st_cfg["store"] = {"method": "LTMemoryStore"}
    st_cfg["recall"].update({"method": "STMemoryRecall", "time_retrieval": {"method": "TimeRetrieval", "mode": "raw", "topk": 5}})
    st = STMemory(MemoryConfig(st_cfg))
    print("STMemory:", run_memory(st, observations[:30], query))

    # LTMemory
    lt_cfg = make_common_config()
    lt_cfg["name"] = "LTMemory"
    lt_cfg["store"] = {"method": "LTMemoryStore"}
    lt_cfg["recall"].update({"method": "LTMemoryRecall", "text_retrieval": make_text_retrieval_config(topk=5)})
    lt = LTMemory(MemoryConfig(lt_cfg))
    print("LTMemory:", run_memory(lt, observations[:80], query))

    # MBMemory (fixed time to avoid summarizer trigger)
    mb_cfg = make_common_config()
    mb_cfg["name"] = "MBMemory"
    mb_cfg["store"] = {
        "method": "MBMemoryStore",
        "summarizer": {
            "method": "LLMSummarizer",
            "LLM_config": {"method": "APILLM", "name": "gpt-4o-mini", "api_key": "DUMMY", "base_url": "https://api.openai.com/v1", "temperature": 0.0},
            "prompt": {"template": "Content: {content}\nSummarize the above content concisely.", "input_variables": ["content"]},
        },
    }
    mb_cfg["recall"].update({"method": "MBMemoryRecall", "text_retrieval": make_text_retrieval_config(topk=5)})
    mb = MBMemory(MemoryConfig(mb_cfg))
    print("MBMemory:", run_memory(mb, observations[:80], query, with_time=True, fixed_time=0))


if __name__ == "__main__":
    main()

