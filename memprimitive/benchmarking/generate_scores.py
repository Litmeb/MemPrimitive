"""Summarize Mem0-style evaluation metrics by LoCoMo category."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _load_items(input_file: Path) -> list[dict[str, Any]]:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    items: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for group_items in data.values():
            if isinstance(group_items, list):
                items.extend(dict(item) for item in group_items if isinstance(item, dict))
    return items


def summarize_scores(input_file: Path) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _load_items(input_file):
        category = str(item.get("category", "")).strip() or "unknown"
        by_category[category].append(item)

    category_scores: dict[str, dict[str, float | int]] = {}
    for category, items in sorted(by_category.items()):
        metrics = {
            "bleu_score": [float(item["bleu_score"]) for item in items if "bleu_score" in item],
            "f1_score": [float(item["f1_score"]) for item in items if "f1_score" in item],
            "llm_score": [float(item["llm_score"]) for item in items if "llm_score" in item],
        }
        category_scores[category] = {
            metric: round(mean(values), 4) for metric, values in metrics.items() if values
        }
        category_scores[category]["count"] = len(items)

    all_items = [item for items in by_category.values() for item in items]
    overall: dict[str, float | int] = {"count": len(all_items)}
    for metric in ("bleu_score", "f1_score", "llm_score"):
        values = [float(item[metric]) for item in all_items if metric in item]
        if values:
            overall[metric] = round(mean(values), 4)

    return {"by_category": category_scores, "overall": overall}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate category score summaries from evaluation metrics.")
    parser.add_argument("--input_file", type=Path, default=Path("benchmarks/outputs/evaluation_metrics.json"))
    parser.add_argument("--output_file", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = summarize_scores(args.input_file)
    print("Mean Scores Per Category:")
    for category, scores in summary["by_category"].items():
        print(f"  {category}: {scores}")
    print("\nOverall Mean Scores:")
    print(f"  {summary['overall']}")

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
