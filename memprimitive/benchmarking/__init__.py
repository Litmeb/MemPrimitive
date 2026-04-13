"""Benchmark data adapters and baseline runners for MemPrimitive."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BenchmarkPrediction",
    "BenchmarkSample",
    "SingleRecallLLMAnswerRunner",
    "create_minimal_benchmark_pipeline",
    "load_benchmark_samples",
    "run_minimal_baseline",
    "run_minimal_baseline_sample",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".minimal_baseline", __name__)
    return getattr(module, name)
