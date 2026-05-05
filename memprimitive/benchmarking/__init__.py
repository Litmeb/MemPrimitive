"""Benchmark adapters, memory adapters, and runners for MemPrimitive."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AnswerRunner",
    "BenchmarkAdapter",
    "BenchmarkPrediction",
    "BenchmarkRunResult",
    "BenchmarkSample",
    "ConversationTurn",
    "DEFAULT_BENCHMARK_ROOT",
    "DEFAULT_OUTPUT_PATH",
    "FunctionMemoryAdapter",
    "LoCoMoBenchmarkAdapter",
    "LongMemEvalBenchmarkAdapter",
    "MemoryAdapter",
    "MemoryIngestEvent",
    "MemoryRecall",
    "MemorySession",
    "MemorySystemBinding",
    "MemMachineLoCoMoAnswerRunner",
    "Mem0LoCoMoAnswerRunner",
    "PairwiseDialogueMemoryAdapter",
    "PipelineMemoryAdapter",
    "RecallContext",
    "SharedConversationLoCoMoMemoryAdapter",
    "SingleRecallLLMAnswerRunner",
    "VALID_BENCHMARKS",
    "VALID_LONGMEMEVAL_VARIANTS",
    "create_benchmark_adapter",
    "create_dual_speaker_locomo_memory_adapter",
    "create_amem_memory_adapter",
    "create_mem0_memory_adapter",
    "create_memmachine_memory_adapter",
    "create_minimal_benchmark_pipeline",
    "create_yaml_pipeline_memory_adapter",
    "default_turn_to_observation",
    "load_benchmark_samples",
    "run_benchmark",
    "run_minimal_baseline",
    "run_minimal_baseline_sample",
    "write_predictions_jsonl",
]

_MODULE_BY_NAME = {
    "AnswerRunner": "._types",
    "BenchmarkAdapter": "._types",
    "BenchmarkPrediction": "._types",
    "BenchmarkRunResult": "._types",
    "BenchmarkSample": "._types",
    "ConversationTurn": "._types",
    "MemoryAdapter": "._types",
    "MemoryIngestEvent": "._types",
    "MemoryRecall": "._types",
    "MemorySession": "._types",
    "MemorySystemBinding": "._types",
    "RecallContext": "._types",
    "default_turn_to_observation": "._types",
    "DEFAULT_BENCHMARK_ROOT": "._bench_adapters",
    "LoCoMoBenchmarkAdapter": "._bench_adapters",
    "LongMemEvalBenchmarkAdapter": "._bench_adapters",
    "VALID_LONGMEMEVAL_VARIANTS": "._bench_adapters",
    "create_benchmark_adapter": "._bench_adapters",
    "FunctionMemoryAdapter": "._memory_adapters",
    "PairwiseDialogueMemoryAdapter": "._memory_adapters",
    "PipelineMemoryAdapter": "._memory_adapters",
    "SharedConversationLoCoMoMemoryAdapter": "._memory_adapters",
    "create_dual_speaker_locomo_memory_adapter": "._memory_adapters",
    "create_amem_memory_adapter": "._memory_adapters",
    "create_mem0_memory_adapter": "._memory_adapters",
    "create_memmachine_memory_adapter": "._memory_adapters",
    "create_yaml_pipeline_memory_adapter": "._memory_adapters",
    "MemMachineLoCoMoAnswerRunner": "._runner",
    "Mem0LoCoMoAnswerRunner": "._runner",
    "SingleRecallLLMAnswerRunner": "._runner",
    "run_benchmark": "._runner",
    "write_predictions_jsonl": "._runner",
    "DEFAULT_OUTPUT_PATH": ".minimal_baseline",
    "VALID_BENCHMARKS": ".minimal_baseline",
    "create_minimal_benchmark_pipeline": ".minimal_baseline",
    "load_benchmark_samples": ".minimal_baseline",
    "run_minimal_baseline": ".minimal_baseline",
    "run_minimal_baseline_sample": ".minimal_baseline",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_BY_NAME[name], __name__)
    return getattr(module, name)
