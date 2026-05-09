"""Mem0-style LoCoMo evaluation helpers for benchmark prediction files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from memprimitive.utils._runtime import Runtime

_TOKEN_PATTERN = re.compile(r"\w+")

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question,
    (2) a gold answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

Be generous with grading. If the generated answer touches on the same topic as the gold answer, count it as CORRECT.
For time-related questions, count different formats as CORRECT when they refer to the same date or time period.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return strict JSON only with this shape: {{"label": "CORRECT"}} or {{"label": "WRONG"}}.
"""


def simple_tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).casefold())


def calculate_metrics(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return {"exact_match": 0.0, "f1": 0.0}
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": float(str(prediction).strip().casefold() == str(reference).strip().casefold()),
        "f1": f1,
    }


def calculate_bleu_scores(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"bleu1": 0.0}
    ref_counts: dict[str, int] = defaultdict(int)
    for token in ref_tokens:
        ref_counts[token] += 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts[token] > 0:
            overlap += 1
            ref_counts[token] -= 1
    precision = overlap / len(pred_tokens)
    brevity_penalty = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return {"bleu1": brevity_penalty * precision}


def evaluate_llm_judge(question: str, gold_answer: str, generated_answer: str, *, runtime: Runtime | None = None) -> int:
    judge_runtime = runtime if runtime is not None else Runtime()
    result = judge_runtime.json(
        system="You are a strict JSON answer correctness judge.",
        user=ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        ),
    )
    label = str(result.get("label", "") if isinstance(result, dict) else "").strip().upper()
    return 1 if label == "CORRECT" else 0


def _read_input(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalize_items(data: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(data, dict):
        if all(isinstance(value, list) for value in data.values()):
            return {str(key): [dict(item) for item in value] for key, value in data.items()}
        data = [data]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw_item in enumerate(data if isinstance(data, list) else [], start=1):
        if not isinstance(raw_item, dict):
            continue
        metadata = raw_item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        memory_metadata = raw_item.get("memory_metadata", {})
        if not isinstance(memory_metadata, dict):
            memory_metadata = {}
        group_key = str(
            metadata.get("locomo_sample_id")
            or raw_item.get("locomo_sample_id")
            or raw_item.get("benchmark_name")
            or f"group-{index}"
        )
        category = (
            raw_item.get("category")
            or metadata.get("qa_category")
            or memory_metadata.get("qa_category")
            or ""
        )
        grouped[group_key].append(
            {
                "question": raw_item.get("question") or raw_item.get("query_text") or "",
                "answer": raw_item.get("answer") or raw_item.get("reference_answer") or "",
                "response": raw_item.get("response") or raw_item.get("predicted_answer") or "",
                "category": category,
                "evidence": raw_item.get("evidence") or metadata.get("evidence") or memory_metadata.get("evidence") or [],
                "adversarial_answer": (
                    raw_item.get("adversarial_answer")
                    or metadata.get("adversarial_answer")
                    or memory_metadata.get("adversarial_answer")
                    or ""
                ),
                "speaker_1_memories": (
                    raw_item.get("speaker_1_memories")
                    or metadata.get("speaker_1_memories")
                    or memory_metadata.get("speaker_1_memories")
                    or ""
                ),
                "speaker_2_memories": (
                    raw_item.get("speaker_2_memories")
                    or metadata.get("speaker_2_memories")
                    or memory_metadata.get("speaker_2_memories")
                    or ""
                ),
                "num_speaker_1_memories": (
                    raw_item.get("num_speaker_1_memories")
                    or metadata.get("num_speaker_1_memories")
                    or memory_metadata.get("num_speaker_1_memories")
                    or 0
                ),
                "num_speaker_2_memories": (
                    raw_item.get("num_speaker_2_memories")
                    or metadata.get("num_speaker_2_memories")
                    or memory_metadata.get("num_speaker_2_memories")
                    or 0
                ),
                "speaker_1_user_id": (
                    raw_item.get("speaker_1_user_id")
                    or metadata.get("speaker_1_user_id")
                    or memory_metadata.get("speaker_1_user_id")
                    or ""
                ),
                "speaker_2_user_id": (
                    raw_item.get("speaker_2_user_id")
                    or metadata.get("speaker_2_user_id")
                    or memory_metadata.get("speaker_2_user_id")
                    or ""
                ),
                "speaker_1_name": (
                    raw_item.get("speaker_1_name")
                    or metadata.get("speaker_1_name")
                    or memory_metadata.get("speaker_1_name")
                    or ""
                ),
                "speaker_2_name": (
                    raw_item.get("speaker_2_name")
                    or metadata.get("speaker_2_name")
                    or memory_metadata.get("speaker_2_name")
                    or ""
                ),
            }
        )
    return dict(grouped)


def process_item(item_data: tuple[str, list[dict[str, Any]]], *, use_llm_judge: bool) -> dict[str, list[dict[str, Any]]]:
    key, items = item_data
    local_results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer) if use_llm_judge else None

        result = {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
        }
        if llm_score is not None:
            result["llm_score"] = llm_score
        local_results[key].append(result)

    return dict(local_results)


def _mp_eval_shard(args: tuple[tuple[str, list[dict[str, Any]]], bool]) -> dict[str, list[dict[str, Any]]]:
    """Picklable entrypoint for ProcessPoolExecutor (Windows spawn-safe)."""
    item_data, use_llm_flag = args
    return process_item(item_data, use_llm_judge=use_llm_flag)


def evaluate_file(
    *,
    input_file: Path,
    output_file: Path,
    max_workers: int = 10,
    use_llm_judge: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    data = _normalize_items(_read_input(input_file))
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_tasks = list(data.items())
    pool_cap = max(1, len(item_tasks))
    effective_workers = min(max(1, max_workers), pool_cap)

    if item_tasks:
        executor_cls = concurrent.futures.ProcessPoolExecutor if use_llm_judge else concurrent.futures.ThreadPoolExecutor
        with executor_cls(max_workers=effective_workers) as executor:
            if use_llm_judge:
                futures = [
                    executor.submit(_mp_eval_shard, (item_data, use_llm_judge))
                    for item_data in item_tasks
                ]
            else:
                futures = [
                    executor.submit(process_item, item_data, use_llm_judge=use_llm_judge)
                    for item_data in item_tasks
                ]
            for future in concurrent.futures.as_completed(futures):
                local_results = future.result()
                for key, items in local_results.items():
                    results[key].extend(items)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(dict(results), ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(results)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MemPrimitive benchmark predictions with Mem0-style metrics.")
    parser.add_argument("--input_file", type=Path, required=True, help="JSON or JSONL prediction file.")
    parser.add_argument("--output_file", type=Path, default=Path("benchmarks/outputs/evaluation_metrics.json"))
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Parallelism across LoCoMo groups: process pool when LLM judge is enabled, thread pool when --skip_llm_judge.",
    )
    parser.add_argument("--skip_llm_judge", action="store_true", help="Only calculate local BLEU/F1 metrics.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    evaluate_file(
        input_file=args.input_file,
        output_file=args.output_file,
        max_workers=args.max_workers,
        use_llm_judge=not args.skip_llm_judge,
    )
    print(f"Results saved to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
