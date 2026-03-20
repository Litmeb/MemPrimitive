import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import os
import random
import re
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


MEMORY_MODULES = [
    "FUMemory",
    "STMemory",
    "LTMemory",
    "MBMemory",
    "GAMemory",
    "SCMemory",
    "MGMemory",
]

DEFAULT_SEARCH_SPACE = {
    "arch": ["single", "dual"],
    "memory": MEMORY_MODULES,
    "merge_policy": ["fallback", "merge"],
    "topk": [1, 3, 5, 10],
    "time_bucket": [4, 8, 16],
}

DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the answer is not in the context, reply with 'Unknown'. "
    "Keep the answer short and specific."
)
DEFAULT_JUDGE_SYSTEM_PROMPT = "You are a strict evaluator. Output JSON only."


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonicalize_config(config: dict) -> dict:
    out = {
        "arch": config["arch"],
        "memory1": config["memory1"],
        "topk": int(config["topk"]),
        "time_bucket": int(config["time_bucket"]),
    }
    if config["arch"] == "dual":
        out["memory2"] = config["memory2"]
        out["merge_policy"] = config["merge_policy"]
    return out


def config_to_key(config: dict) -> str:
    return json.dumps(canonicalize_config(config), sort_keys=True, ensure_ascii=False)


def iter_all_configs(search_space: dict):
    for arch in search_space["arch"]:
        for memory1 in search_space["memory"]:
            for topk in search_space["topk"]:
                for time_bucket in search_space["time_bucket"]:
                    if arch == "single":
                        yield canonicalize_config(
                            {
                                "arch": arch,
                                "memory1": memory1,
                                "topk": topk,
                                "time_bucket": time_bucket,
                            }
                        )
                    else:
                        for memory2 in search_space["memory"]:
                            if memory1 == memory2:
                                continue
                            for merge_policy in search_space["merge_policy"]:
                                yield canonicalize_config(
                                    {
                                        "arch": arch,
                                        "memory1": memory1,
                                        "memory2": memory2,
                                        "merge_policy": merge_policy,
                                        "topk": topk,
                                        "time_bucket": time_bucket,
                                    }
                                )


def sample_next_config(tried_configs: set, *, rng, search_space: dict) -> dict | None:
    for _ in range(64):
        arch = rng.choice(search_space["arch"])
        memory1 = rng.choice(search_space["memory"])
        topk = rng.choice(search_space["topk"])
        time_bucket = rng.choice(search_space["time_bucket"])
        if arch == "single":
            config = canonicalize_config(
                {
                    "arch": arch,
                    "memory1": memory1,
                    "topk": topk,
                    "time_bucket": time_bucket,
                }
            )
        else:
            memory2 = rng.choice(search_space["memory"])
            if memory1 == memory2:
                continue
            config = canonicalize_config(
                {
                    "arch": arch,
                    "memory1": memory1,
                    "memory2": memory2,
                    "merge_policy": rng.choice(search_space["merge_policy"]),
                    "topk": topk,
                    "time_bucket": time_bucket,
                }
            )
        if config_to_key(config) not in tried_configs:
            return config

    remaining = [cfg for cfg in iter_all_configs(search_space) if config_to_key(cfg) not in tried_configs]
    if not remaining:
        return None
    return rng.choice(remaining)


def iter_sessions(conv: dict):
    idx = 1
    while True:
        dt_key = f"session_{idx}_date_time"
        s_key = f"session_{idx}"
        if dt_key not in conv or s_key not in conv:
            break
        yield idx, conv[dt_key], conv[s_key]
        idx += 1


def flatten_conversation(sample: dict) -> list[dict]:
    conv = sample["conversation"]
    turns = []
    for session_idx, session_dt, sess in iter_sessions(conv):
        for turn in sess:
            if "text" not in turn:
                continue
            turns.append(
                {
                    "session_idx": session_idx,
                    "session_dt": session_dt,
                    "speaker": turn.get("speaker", ""),
                    "dia_id": turn.get("dia_id", ""),
                    "text": turn["text"],
                    "img_url": turn.get("img_url"),
                    "blip_caption": turn.get("blip_caption"),
                    "query": turn.get("query"),
                }
            )
    return turns


def format_turn_for_memory(turn: dict) -> str:
    parts = []
    if turn.get("dia_id"):
        parts.append(str(turn["dia_id"]))
    if turn.get("speaker"):
        parts.append(str(turn["speaker"]))
    header = " | ".join(parts) if parts else "Turn"
    body = str(turn.get("text", ""))
    if turn.get("blip_caption"):
        body += f"\n[image_caption] {turn['blip_caption']}"
    if turn.get("query"):
        body += f"\n[image_query] {turn['query']}"
    return f"{header}: {body}".strip()


def normalize_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def is_empty_context(value) -> bool:
    text = normalize_text(value)
    if not text:
        return True
    return text.lower() in {"none", "null", "empty", "no memory"}


def load_dataset(dataset_path: Path) -> list[dict]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def get_default_dataset_path() -> Path:
    return Path(__file__).with_name("locomo10.json")


def get_default_llm_config() -> dict:
    return {
        "name": os.environ.get("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free"),
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "temperature": safe_float(os.environ.get("OPENROUTER_TEMPERATURE", 0.0), 0.0),
    }


def check_python_dependencies() -> list[str]:
    missing = []
    for module_name in ["openai"]:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


def ensure_langchain_prompts_shim() -> None:
    try:
        importlib.import_module("langchain.prompts")
        return
    except Exception:
        pass

    try:
        core_prompts = importlib.import_module("langchain_core.prompts")
    except Exception:
        return

    shim = types.ModuleType("langchain.prompts")
    if hasattr(core_prompts, "PromptTemplate"):
        shim.PromptTemplate = core_prompts.PromptTemplate
    sys.modules["langchain.prompts"] = shim


def _patch_memengine_llm_trigger_no_execute_returns_list() -> None:
    """memengine LLMTrigger returns False for NO_EXECUTE; MGMemoryRecall expects an iterable."""
    try:
        trigger_mod = importlib.import_module("memengine.function.Trigger")
    except Exception:
        return
    LLMTrigger = getattr(trigger_mod, "LLMTrigger", None)
    if LLMTrigger is None:
        return
    current = LLMTrigger.__parse_excuate_function__
    if getattr(current, "_locomo_search_no_execute_patch", False):
        return

    def _wrapped(self, res):
        out = current(self, res)
        return [] if out is False else out

    _wrapped._locomo_search_no_execute_patch = True
    LLMTrigger.__parse_excuate_function__ = _wrapped


def check_memengine_dependencies() -> dict:
    info = {"ok": False, "errors": []}
    missing = check_python_dependencies()
    if missing:
        info["errors"].append(f"Missing Python packages: {', '.join(missing)}")

    try:
        ensure_langchain_prompts_shim()
        memengine = importlib.import_module("memengine")
        _patch_memengine_llm_trigger_no_execute_returns_list()
        info["memengine"] = memengine
        info["MemoryConfig"] = getattr(memengine, "MemoryConfig")
        info["exports"] = {name: getattr(memengine, name) for name in MEMORY_MODULES}
        info["ok"] = len(info["errors"]) == 0
        return info
    except Exception as exc:
        info["errors"].append(
            "Failed to import memengine. Current memengine versions often require a compatible "
            "`langchain.prompts` import path. Install or fix the compatible dependency set before running. "
            f"Original error: {exc}"
        )
        return info


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
        "encoder": {
            "method": "STEncoder",
            "name": st_model,
            "dimension": 384,
            "path": st_model,
        },
        "mode": "cosine",
        "topk": int(topk),
    }


def build_single_memory(memory_name: str, *, topk: int, time_bucket: int, llm_cfg: dict, st_model: str, memengine_info: dict) -> dict:
    MemoryConfig = memengine_info["MemoryConfig"]
    memory_cls = memengine_info["exports"][memory_name]

    if memory_name == "FUMemory":
        cfg = make_common_config()
        cfg["name"] = "FUMemory"
        cfg["store"] = {"method": "FUMemoryStore"}
        cfg["recall"]["method"] = "FUMemoryRecall"
        return {"name": memory_name, "memory": memory_cls(MemoryConfig(cfg)), "store_mode": "text", "time_bucket": int(time_bucket)}

    if memory_name == "STMemory":
        cfg = make_common_config()
        cfg["name"] = "STMemory"
        cfg["store"] = {"method": "STMemoryStore"}
        cfg["recall"].update(
            {
                "method": "STMemoryRecall",
                "time_retrieval": {"method": "TimeRetrieval", "mode": "raw", "topk": int(topk)},
            }
        )
        return {"name": memory_name, "memory": memory_cls(MemoryConfig(cfg)), "store_mode": "text", "time_bucket": int(time_bucket)}

    if memory_name == "LTMemory":
        cfg = make_common_config()
        cfg["name"] = "LTMemory"
        cfg["store"] = {"method": "LTMemoryStore"}
        cfg["recall"].update(
            {
                "method": "LTMemoryRecall",
                "text_retrieval": make_text_retrieval_config(topk=topk, st_model=st_model),
            }
        )
        return {"name": memory_name, "memory": memory_cls(MemoryConfig(cfg)), "store_mode": "text", "time_bucket": int(time_bucket)}

    if memory_name == "MBMemory":
        cfg = make_common_config()
        cfg["name"] = "MBMemory"
        cfg["store"] = {
            "method": "MBMemoryStore",
            "summarizer": {
                "method": "LLMSummarizer",
                "LLM_config": {
                    "method": "APILLM",
                    "name": llm_cfg["name"],
                    "api_key": llm_cfg["api_key"],
                    "base_url": llm_cfg["base_url"],
                    "temperature": safe_float(llm_cfg.get("temperature", 0.0), 0.0),
                },
                "prompt": {
                    "template": "Content: {content}\nSummarize the above content concisely, extracting the main themes and key information.",
                    "input_variables": ["content"],
                },
            },
        }
        cfg["recall"].update(
            {
                "method": "MBMemoryRecall",
                "text_retrieval": make_text_retrieval_config(topk=topk, st_model=st_model),
            }
        )
        memory = memory_cls(MemoryConfig(cfg))

        class SafeSummarizer:
            def __init__(self, inner):
                self.inner = inner

            def reset(self):
                reset_fn = getattr(self.inner, "reset", None)
                if callable(reset_fn):
                    reset_fn()

            def __call__(self, content):
                for _ in range(2):
                    try:
                        summary = self.inner(content)
                    except Exception:
                        summary = None
                    if isinstance(summary, str) and summary.strip():
                        return summary
                text = normalize_text(content)
                if not text:
                    return "Summary (fallback): <empty content>"
                return "Summary (fallback): " + text[:600] + ("..." if len(text) > 600 else "")

        if hasattr(memory, "store_op") and hasattr(memory.store_op, "summarizer"):
            memory.store_op.summarizer = SafeSummarizer(memory.store_op.summarizer)
        return {"name": memory_name, "memory": memory, "store_mode": "mb", "time_bucket": int(time_bucket)}

    if memory_name == "GAMemory":
        cfg = make_common_config()
        cfg["name"] = "GAMemory"
        cfg["store"] = {"method": "GAMemoryStore"}
        cfg["recall"].update(
            {
                "method": "GAMemoryRecall",
                "topk": 8,
                "text_retrieval": make_text_retrieval_config(topk=32, st_model=st_model),
                "time_retrieval": {"method": "TimeRetrieval", "mode": "exp", "coef": {"decay": 0.99}, "topk": 32},
                "importance_retrieval": {"method": "ValueRetrieval", "mode": "identical", "topk": 32},
                "importance_judge": {
                    "method": "LLMJudge",
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_cfg["name"],
                        "api_key": llm_cfg["api_key"],
                        "base_url": llm_cfg["base_url"],
                        "temperature": 0.0,
                    },
                    "post_scale": 10,
                    "prompt": {
                        "template": "You are scoring how important a memory is for future use.\nMemory: {message}\nReturn ONLY a number from 0 to 10.",
                        "input_variables": ["message"],
                    },
                },
            }
        )
        cfg["reflect"] = {
            "reflector": {
                "threshold": 1_000_000_000,
                "reflection_topk": 5,
                "question_number": 3,
                "insight_number": 3,
                "LLM_config": {
                    "method": "APILLM",
                    "name": llm_cfg["name"],
                    "api_key": llm_cfg["api_key"],
                    "base_url": llm_cfg["base_url"],
                    "temperature": 0.0,
                },
                "question_prompt": {
                    "template": "Given the information below, write exactly {question_number} questions (one per line).\n\nInformation:\n{information}",
                    "input_variables": ["information", "question_number"],
                },
                "insight_prompt": {
                    "template": "Given the statements below, write exactly {insight_number} high-level insights (one per line).\n\nStatements:\n{statements}",
                    "input_variables": ["statements", "insight_number"],
                },
            }
        }
        return {"name": memory_name, "memory": memory_cls(MemoryConfig(cfg)), "store_mode": "text", "time_bucket": int(time_bucket)}

    if memory_name == "SCMemory":
        cfg = make_common_config()
        cfg["name"] = "SCMemory"
        cfg["store"] = {"method": "SCMemoryStore"}
        cfg["recall"].update(
            {
                "method": "SCMemoryRecall",
                "flash_capacity": 5,
                "activation_topk": 8,
                "text_retrieval": make_text_retrieval_config(topk=32, st_model=st_model),
                "time_retrieval": {"method": "TimeRetrieval", "mode": "exp", "coef": {"decay": 0.99}, "topk": 32},
                "summarizer": {
                    "method": "LLMSummarizer",
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_cfg["name"],
                        "api_key": llm_cfg["api_key"],
                        "base_url": llm_cfg["base_url"],
                        "temperature": 0.0,
                    },
                    "prompt": {
                        "template": "Summarize in 1 sentence, keep key entities and dates.\nContent: {content}",
                        "input_variables": ["content"],
                    },
                },
                "activation_judge": {
                    "method": "LLMJudge",
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_cfg["name"],
                        "api_key": llm_cfg["api_key"],
                        "base_url": llm_cfg["base_url"],
                        "temperature": 0.0,
                    },
                    "prompt": {
                        "template": "Query: {query}\nFlash memory: {flash_memory}\nDo we need to retrieve more history beyond flash memory to answer? Return ONLY True or False.",
                        "input_variables": ["query", "flash_memory"],
                    },
                },
                "summary_judge": {
                    "method": "LLMJudge",
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_cfg["name"],
                        "api_key": llm_cfg["api_key"],
                        "base_url": llm_cfg["base_url"],
                        "temperature": 0.0,
                    },
                    "prompt": {
                        "template": "Query: {query}\nActivation summary: {activation_summary}\nFlash memory: {flash_memory}\nIs the activation summary sufficient (no need for full texts)? Return ONLY True or False.",
                        "input_variables": ["query", "activation_summary", "flash_memory"],
                    },
                },
            }
        )
        return {"name": memory_name, "memory": memory_cls(MemoryConfig(cfg)), "store_mode": "text", "time_bucket": int(time_bucket)}

    if memory_name == "MGMemory":
        cfg = make_common_config()
        cfg["name"] = "MGMemory"
        cfg["store"] = {
            "method": "MGMemoryStore",
            "summarizer": {
                "method": "LLMSummarizer",
                "LLM_config": {
                    "method": "APILLM",
                    "name": llm_cfg["name"],
                    "api_key": llm_cfg["api_key"],
                    "base_url": llm_cfg["base_url"],
                    "temperature": 0.0,
                },
                "prompt": {
                    "template": "Recursive summary (keep factual details).\nPrevious summary: {recursive_summary}\nNew content: {flush_context}\nReturn updated summary only.",
                    "input_variables": ["recursive_summary", "flush_context"],
                },
            },
            "flush_checker": {"method": "LMTruncation", "mode": "word", "number": 120, "path": ""},
        }
        cfg["recall"].update(
            {
                "method": "MGMemoryRecall",
                "warning_threshold": 0.8,
                "warning_content": "[Warning: memory capacity is near the limit]",
                "recall_retrieval": make_text_retrieval_config(topk=topk, st_model=st_model),
                "archival_retrieval": make_text_retrieval_config(topk=topk, st_model=st_model),
                "trigger": {
                    "method": "LLMTrigger",
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_cfg["name"],
                        "api_key": llm_cfg["api_key"],
                        "base_url": llm_cfg["base_url"],
                        "temperature": 0.0,
                    },
                    "func_list": [
                        {
                            "name": "memory_recall",
                            "args": ["query"],
                            "args_type": ["str"],
                            "func_description": "Retrieve related items from recall storage into FIFO memory.",
                            "args_description": ["query: retrieval query"],
                        },
                        {
                            "name": "memory_retrieval",
                            "args": ["query"],
                            "args_type": ["str"],
                            "func_description": "Retrieve related items from archival storage into working memory.",
                            "args_description": ["query: retrieval query"],
                        },
                        {
                            "name": "memory_transfer",
                            "args": ["memory_list"],
                            "args_type": ["list"],
                            "func_description": "Transfer items from FIFO memory to working memory.",
                            "args_description": ["memory_list: list of FIFO indexes"],
                        },
                        {
                            "name": "memory_archive",
                            "args": ["memory_list"],
                            "args_type": ["list"],
                            "func_description": "Archive items from FIFO memory into recall storage.",
                            "args_description": ["memory_list: list of FIFO indexes"],
                        },
                        {
                            "name": "memory_save",
                            "args": ["memory_list"],
                            "args_type": ["list"],
                            "func_description": "Save items from working memory into archival storage.",
                            "args_description": ["memory_list: list of working memory indexes"],
                        },
                    ],
                    "few_shot": "",
                    "prompt": {
                        "template": "You are managing a memory OS.\n{warning_content}{no_execute_prompt}\n{function_prompt}\n\nMemory state:\n{memory_prompt}\n\nUser text:\n{text}\n\nReturn ONE OR MORE function calls, one per line (e.g. memory_recall(\"Alice\")).\nIf no function is needed, return: NO_EXECUTE",
                        "input_variables": [
                            "warning_content",
                            "no_execute_prompt",
                            "function_prompt",
                            "few_shot",
                            "memory_prompt",
                            "text",
                        ],
                    },
                    "no_execuate": "NO_EXECUTE",
                },
            }
        )
        return {"name": memory_name, "memory": memory_cls(MemoryConfig(cfg)), "store_mode": "text", "time_bucket": int(time_bucket)}

    raise ValueError(f"Unsupported memory: {memory_name}")


def build_memory_system(config: dict, *, llm_cfg: dict, st_model: str):
    memengine_info = check_memengine_dependencies()
    if not memengine_info.get("ok"):
        raise RuntimeError("; ".join(memengine_info["errors"]))

    config = canonicalize_config(config)
    memory1_adapter = build_single_memory(
        config["memory1"],
        topk=config["topk"],
        time_bucket=config["time_bucket"],
        llm_cfg=llm_cfg,
        st_model=st_model,
        memengine_info=memengine_info,
    )
    memory2_adapter = None
    merge_policy = None
    if config["arch"] == "dual":
        memory2_adapter = build_single_memory(
            config["memory2"],
            topk=config["topk"],
            time_bucket=config["time_bucket"],
            llm_cfg=llm_cfg,
            st_model=st_model,
            memengine_info=memengine_info,
        )
        merge_policy = config["merge_policy"]

    return {
        "arch": config["arch"],
        "memory1_adapter": memory1_adapter,
        "memory2_adapter": memory2_adapter,
        "merge_policy": merge_policy,
    }


def reset_adapter(adapter: dict) -> None:
    adapter["memory"].reset()


def store_turns_in_adapter(adapter: dict, turns: list[dict]) -> None:
    for idx, turn in enumerate(turns):
        text = format_turn_for_memory(turn)
        if adapter["store_mode"] == "mb":
            bucket = idx // max(1, int(adapter["time_bucket"]))
            adapter["memory"].store({"text": text, "time": bucket})
        else:
            adapter["memory"].store(text)


def recall_with_adapter(adapter: dict, question: str) -> str:
    return normalize_text(adapter["memory"].recall(question))


def recall_context(memory_system: dict, question: str) -> tuple[str, dict]:
    details = {"arch": memory_system["arch"], "memory_calls": []}
    t0 = time.perf_counter()
    ctx1 = recall_with_adapter(memory_system["memory1_adapter"], question)
    details["memory_calls"].append(
        {
            "memory": memory_system["memory1_adapter"]["name"],
            "context": ctx1,
            "empty": is_empty_context(ctx1),
        }
    )

    if memory_system["arch"] == "single":
        details["latency_sec"] = time.perf_counter() - t0
        return ctx1, details

    if memory_system["merge_policy"] == "fallback":
        if is_empty_context(ctx1):
            ctx2 = recall_with_adapter(memory_system["memory2_adapter"], question)
            details["memory_calls"].append(
                {
                    "memory": memory_system["memory2_adapter"]["name"],
                    "context": ctx2,
                    "empty": is_empty_context(ctx2),
                }
            )
            final_context = ctx2
        else:
            final_context = ctx1
    elif memory_system["merge_policy"] == "merge":
        ctx2 = recall_with_adapter(memory_system["memory2_adapter"], question)
        details["memory_calls"].append(
            {
                "memory": memory_system["memory2_adapter"]["name"],
                "context": ctx2,
                "empty": is_empty_context(ctx2),
            }
        )
        final_context = "\n\n".join([item for item in [ctx1, ctx2] if not is_empty_context(item)])
    else:
        raise ValueError(f"Unsupported merge_policy: {memory_system['merge_policy']}")

    details["latency_sec"] = time.perf_counter() - t0
    return normalize_text(final_context), details


def openai_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def chat_completion(*, llm_cfg: dict, messages: list[dict], phase: str, trial_id: int, call_log: list[dict], max_tokens: int = 256) -> str:
    call_id = len(call_log) + 1
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    raw_response = ""
    error = None
    try:
        client = openai_client(llm_cfg["base_url"], llm_cfg["api_key"])
        resp = client.chat.completions.create(
            model=llm_cfg["name"],
            messages=messages,
            temperature=safe_float(llm_cfg.get("temperature", 0.0), 0.0),
            max_tokens=max_tokens,
        )
        raw_response = normalize_text(resp.choices[0].message.content)
        return raw_response
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        call_log.append(
            {
                "call_id": call_id,
                "trial_id": trial_id,
                "phase": phase,
                "model": llm_cfg.get("name"),
                "base_url": llm_cfg.get("base_url"),
                "messages": messages,
                "raw_response_text": raw_response,
                "started_at": started_at,
                "ended_at": utc_now_iso(),
                "latency_sec": time.perf_counter() - t0,
                "error": error,
            }
        )


def answer_question(*, question: str, context: str, llm_cfg: dict, trial_id: int, call_log: list[dict]) -> str:
    return chat_completion(
        llm_cfg=llm_cfg,
        messages=[
            {"role": "system", "content": DEFAULT_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ],
        phase="answer",
        trial_id=trial_id,
        call_log=call_log,
        max_tokens=128,
    )


def llm_judge(*, question: str, gold: str, pred: str, context: str, llm_cfg: dict, trial_id: int, call_log: list[dict]) -> dict:
    raw = chat_completion(
        llm_cfg=llm_cfg,
        messages=[
            {"role": "system", "content": DEFAULT_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Rate the prediction against the gold answer for the given question.\n\n"
                    "Return a JSON object with fields:\n"
                    "- score: number in [0,1]\n"
                    '- verdict: one of ["correct","partially_correct","incorrect","unknown"]\n'
                    "- rationale: short string\n\n"
                    f"Question: {question}\n\n"
                    f"Gold answer: {gold}\n\n"
                    f"Prediction: {pred}\n\n"
                    f"Retrieved context:\n{context}\n"
                ),
            },
        ],
        phase="judge",
        trial_id=trial_id,
        call_log=call_log,
        max_tokens=256,
    )
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"score": 0.0, "verdict": "incorrect", "rationale": f"non_json: {raw[:200]}"}
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return {"score": 0.0, "verdict": "incorrect", "rationale": f"bad_json: {raw[:200]}"}
    return {
        "score": min(1.0, max(0.0, safe_float(obj.get("score", 0.0), 0.0))),
        "verdict": obj.get("verdict", "incorrect"),
        "rationale": normalize_text(obj.get("rationale", "")),
    }


def evaluate_one_qa(*, question: str, gold: str, memory_system: dict, llm_cfg: dict, trial_id: int, call_log: list[dict]) -> dict:
    recall_t0 = time.perf_counter()
    context, recall_details = recall_context(memory_system, question)
    recall_latency = time.perf_counter() - recall_t0

    answer_start = len(call_log)
    pred = answer_question(question=question, context=context, llm_cfg=llm_cfg, trial_id=trial_id, call_log=call_log)
    answer_latency = sum(item["latency_sec"] for item in call_log[answer_start:])

    judge_start = len(call_log)
    judge = llm_judge(
        question=question,
        gold=gold,
        pred=pred,
        context=context,
        llm_cfg=llm_cfg,
        trial_id=trial_id,
        call_log=call_log,
    )
    judge_latency = sum(item["latency_sec"] for item in call_log[judge_start:])

    return {
        "question": question,
        "gold": gold,
        "retrieved_context": context,
        "pred": pred,
        "judge_score": judge["score"],
        "judge_verdict": judge["verdict"],
        "judge_rationale": judge["rationale"],
        "recall_latency_sec": recall_latency,
        "answer_latency_sec": answer_latency,
        "judge_latency_sec": judge_latency,
        "recall_details": recall_details,
    }


def evaluate_config(config: dict, *, dataset, llm_cfg: dict, latency_penalty: float, log_dir: Path, eval_opts: dict) -> dict:
    config = canonicalize_config(config)
    trial_id = int(eval_opts.get("trial_id", 0))
    st_model = eval_opts.get("st_model", "sentence-transformers/all-MiniLM-L6-v2")
    top_samples = eval_opts.get("top_samples")
    top_qas_per_sample = eval_opts.get("top_qas_per_sample")

    ensure_dir(log_dir)
    trial_json_path = log_dir / f"trial_{trial_id}.json"
    trial_jsonl_path = log_dir / f"trial_{trial_id}.jsonl"
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    qa_rows = []
    llm_calls = []

    try:
        memory_system = build_memory_system(config, llm_cfg=llm_cfg, st_model=st_model)
        samples = dataset[:top_samples] if top_samples else dataset

        for sample_idx, sample in enumerate(samples):
            turns = flatten_conversation(sample)
            reset_adapter(memory_system["memory1_adapter"])
            store_turns_in_adapter(memory_system["memory1_adapter"], turns)
            if memory_system["arch"] == "dual":
                reset_adapter(memory_system["memory2_adapter"])
                store_turns_in_adapter(memory_system["memory2_adapter"], turns)

            qa_items = [item for item in sample.get("qa", []) if "question" in item and "answer" in item]
            if top_qas_per_sample:
                qa_items = qa_items[:top_qas_per_sample]

            for qa_idx, qa_item in enumerate(qa_items):
                row = evaluate_one_qa(
                    question=normalize_text(qa_item["question"]),
                    gold=normalize_text(qa_item["answer"]),
                    memory_system=memory_system,
                    llm_cfg=llm_cfg,
                    trial_id=trial_id,
                    call_log=llm_calls,
                )
                row.update(
                    {
                        "trial_id": trial_id,
                        "sample_id": sample.get("sample_id"),
                        "sample_index": sample_idx,
                        "qa_index": qa_idx,
                        "config": config,
                    }
                )
                qa_rows.append(row)

        accuracy = sum(row["judge_score"] for row in qa_rows) / len(qa_rows) if qa_rows else None
        latency = time.perf_counter() - t0
        score = None if accuracy is None else float(accuracy) - float(latency_penalty) * float(latency)
        full_result = {
            "trial_id": trial_id,
            "status": "ok",
            "score": score,
            "accuracy": accuracy,
            "latency": latency,
            "config": config,
            "error": None,
            "start_time": started_at,
            "end_time": utc_now_iso(),
            "metrics_summary": {
                "num_samples": len(samples),
                "num_qas": len(qa_rows),
                "mean_judge_score": accuracy,
            },
            "log_paths": {"trial_json": str(trial_json_path), "trial_jsonl": str(trial_jsonl_path)},
            "qa_results": qa_rows,
            "llm_calls": llm_calls,
            "memory_runtime_notes": {
                "arch": config["arch"],
                "memory1": config["memory1"],
                "memory2": config.get("memory2"),
                "merge_policy": config.get("merge_policy"),
                "note": (
                    "This script records answer/judge LLM calls directly. "
                    "memengine-internal LLM calls are only indirectly observable via config, timing, and errors."
                ),
            },
        }
    except Exception as exc:
        full_result = {
            "trial_id": trial_id,
            "status": "error",
            "score": None,
            "accuracy": None,
            "latency": time.perf_counter() - t0,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "start_time": started_at,
            "end_time": utc_now_iso(),
            "metrics_summary": {"num_samples": 0, "num_qas": 0, "mean_judge_score": None},
            "log_paths": {"trial_json": str(trial_json_path), "trial_jsonl": str(trial_jsonl_path)},
            "qa_results": qa_rows,
            "llm_calls": llm_calls,
        }

    dump_json(trial_json_path, full_result)
    events = [{"event": "qa_result", **row} for row in full_result.get("qa_results", [])]
    events.extend({"event": "llm_call", **row} for row in full_result.get("llm_calls", []))
    if full_result["status"] == "error":
        events.append(
            {
                "event": "trial_error",
                "trial_id": full_result["trial_id"],
                "config": full_result["config"],
                "error": full_result["error"],
                "traceback": full_result.get("traceback"),
            }
        )
    append_jsonl(trial_jsonl_path, events)

    return {
        "trial_id": full_result["trial_id"],
        "status": full_result["status"],
        "score": full_result["score"],
        "accuracy": full_result["accuracy"],
        "latency": full_result["latency"],
        "config": full_result["config"],
        "error": full_result.get("error"),
        "metrics_summary": full_result["metrics_summary"],
        "log_paths": full_result["log_paths"],
    }


def run_search(
    *,
    num_trials: int,
    dataset,
    latency_penalty: float,
    seed: int,
    out_path: Path,
    log_dir: Path,
    eval_opts: dict,
    max_workers: int = 1,
) -> dict:
    rng = random.Random(seed)
    search_space = eval_opts.get("search_space", DEFAULT_SEARCH_SPACE)
    ensure_dir(out_path)
    ensure_dir(log_dir)

    trials_json_path = out_path / "trials.json"
    trials_jsonl_path = out_path / "trials.jsonl"
    best_result_path = out_path / "best_result.json"
    search_summary_path = out_path / "search_summary.json"

    tried_configs = set()
    scheduled_trials = []
    exhausted = False

    for trial_idx in range(1, int(num_trials) + 1):
        config = sample_next_config(tried_configs, rng=rng, search_space=search_space)
        if config is None:
            exhausted = True
            break
        tried_configs.add(config_to_key(config))
        scheduled_trials.append((trial_idx, config))

    trials_by_id = {}
    max_workers = max(1, int(max_workers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_trial_id = {
            executor.submit(
                evaluate_config,
                config,
                dataset=dataset,
                llm_cfg=eval_opts["llm_cfg"],
                latency_penalty=latency_penalty,
                log_dir=log_dir,
                eval_opts={**eval_opts, "trial_id": trial_idx},
            ): trial_idx
            for trial_idx, config in scheduled_trials
        }

        for future in as_completed(future_to_trial_id):
            trial = future.result()
            trials_by_id[trial["trial_id"]] = trial

            append_jsonl(trials_jsonl_path, [trial])
            ordered_trials = [trials_by_id[idx] for idx in sorted(trials_by_id)]
            dump_json(trials_json_path, ordered_trials)

            ok_trials = [item for item in ordered_trials if item["status"] == "ok" and item["score"] is not None]
            dump_json(best_result_path, max(ok_trials, key=lambda item: item["score"]) if ok_trials else None)

    trials = [trials_by_id[idx] for idx in sorted(trials_by_id)]
    ok_trials = [item for item in trials if item["status"] == "ok" and item["score"] is not None]
    best_trial = max(ok_trials, key=lambda item: item["score"]) if ok_trials else None
    summary = {
        "num_trials_requested": int(num_trials),
        "num_trials_completed": len(trials),
        "num_trials_failed": sum(1 for item in trials if item["status"] == "error"),
        "search_space_exhausted": exhausted,
        "latency_penalty": latency_penalty,
        "seed": seed,
        "max_workers": max_workers,
    }
    dump_json(search_summary_path, summary)
    dump_json(best_result_path, best_trial)
    return {
        "trials": trials,
        "best_trial": best_trial,
        "best_config": best_trial["config"] if best_trial else None,
        "search_summary": summary,
    }


def make_timestamped_output_dir(base_dir: Path | None = None) -> tuple[Path, Path]:
    if base_dir is None:
        base_dir = Path(__file__).with_name("logs") / "locomo_search"
    run_dir = ensure_dir(base_dir / datetime.now().strftime("%Y%m%d_%H%M%S"))
    return run_dir, ensure_dir(run_dir / "logs")


def smoke_test_sampling() -> None:
    rng = random.Random(123)
    tried = set()
    configs = []
    for _ in range(5):
        cfg = sample_next_config(tried, rng=rng, search_space=DEFAULT_SEARCH_SPACE)
        assert cfg is not None
        assert config_to_key(cfg) not in tried
        tried.add(config_to_key(cfg))
        configs.append(cfg)
    assert len({config_to_key(cfg) for cfg in configs}) == len(configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search memory configs on LOCOMO.")
    parser.add_argument("--dataset", type=str, default=str(get_default_dataset_path()))
    parser.add_argument("--num-trials", type=int, default=40)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--latency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-samples", type=int, default=3)
    parser.add_argument("--top-qas-per-sample", type=int, default=1000)
    parser.add_argument("--st-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    smoke_test_sampling()
    if args.smoke_test:
        print("sampling smoke test passed")
        return 0

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    dep_info = check_memengine_dependencies()
    if not dep_info.get("ok"):
        for err in dep_info["errors"]:
            print(err)
        return 1

    llm_cfg = get_default_llm_config()
    if not llm_cfg.get("api_key"):
        print("Missing env: OPENROUTER_API_KEY")
        return 1

    dataset = load_dataset(dataset_path)
    if args.output_dir:
        out_dir = ensure_dir(Path(args.output_dir))
        log_dir = ensure_dir(out_dir / "logs")
    else:
        out_dir, log_dir = make_timestamped_output_dir()

    result = run_search(
        num_trials=args.num_trials,
        dataset=dataset,
        latency_penalty=args.latency_penalty,
        seed=args.seed,
        out_path=out_dir,
        log_dir=log_dir,
        max_workers=args.max_workers,
        eval_opts={
            "top_samples": args.top_samples,
            "top_qas_per_sample": args.top_qas_per_sample,
            "st_model": args.st_model,
            "llm_cfg": llm_cfg,
            "search_space": DEFAULT_SEARCH_SPACE,
        },
    )
    print(json.dumps(result["search_summary"], ensure_ascii=False, indent=2))
    print(json.dumps({"best_config": result["best_config"], "best_trial": result["best_trial"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
