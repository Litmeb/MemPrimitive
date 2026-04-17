"""Minimal benchmark baseline compatibility wrapper built on the new adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    PassThroughUnitFormation,
)

from ._bench_adapters import (
    DEFAULT_BENCHMARK_ROOT,
    VALID_LONGMEMEVAL_VARIANTS,
    _iter_json_array_file,
    create_benchmark_adapter,
)
from ._memory_adapters import PipelineMemoryAdapter
from ._runner import SingleRecallLLMAnswerRunner, run_benchmark, write_predictions_jsonl
from ._types import BenchmarkPrediction, BenchmarkSample

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "outputs" / "minimal_baseline_predictions.jsonl"
VALID_BENCHMARKS = frozenset({"locomo", "longmemeval", "dmr"})


class _SingleSampleBenchmarkAdapter:
    def __init__(self, sample: BenchmarkSample) -> None:
        self.name = sample.benchmark_name
        self.sample = sample

    def iter_samples(self, *, limit: int | None = None) -> Iterator[BenchmarkSample]:
        if limit == 0:
            return
        yield self.sample


class _LegacyDMRBenchmarkAdapter:
    name = "dmr"

    def __init__(self, *, benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> None:
        self.benchmark_root = Path(benchmark_root)

    def iter_samples(self, *, limit: int | None = None) -> Iterator[BenchmarkSample]:
        yielded = 0
        for sample in _iter_dmr_samples(self.benchmark_root):
            yield sample
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def create_minimal_benchmark_pipeline(*, top_k: int = 5) -> MemoryPipeline:
    """Build the simplest benchmark-ready memory pipeline."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="memory",
                    theme="semantic",
                    indices=("vector", "temporal"),
                )
            ]
        )
    )
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text", "embedding")),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="memory"),
        retrieval=EmbeddingSimilarityRetrieval(top_k=top_k, layer="memory"),
        readout=ConcatenateReadout(),
        store=store,
    )


def _minimal_memory_adapter(*, top_k: int) -> PipelineMemoryAdapter:
    return PipelineMemoryAdapter(
        name="minimal_pipeline",
        pipeline_factory=lambda: create_minimal_benchmark_pipeline(top_k=top_k),
    )


def _compat_benchmark_adapter(
    name: str,
    *,
    benchmark_root: Path | str,
    longmemeval_variant: str,
):
    benchmark_name = str(name).strip().casefold()
    if benchmark_name == "dmr":
        return _LegacyDMRBenchmarkAdapter(benchmark_root=benchmark_root)
    return create_benchmark_adapter(
        benchmark_name,
        benchmark_root=benchmark_root,
        longmemeval_variant=longmemeval_variant,
    )


def load_benchmark_samples(
    name: str,
    *,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
    longmemeval_variant: str = "s_cleaned",
    limit: int | None = None,
) -> Iterator[BenchmarkSample]:
    """Yield normalized samples for one supported benchmark."""

    benchmark_name = str(name).strip().casefold()
    if benchmark_name not in VALID_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark {name!r}. Choose from {sorted(VALID_BENCHMARKS)}.")
    adapter = _compat_benchmark_adapter(
        benchmark_name,
        benchmark_root=benchmark_root,
        longmemeval_variant=longmemeval_variant,
    )
    yield from adapter.iter_samples(limit=limit)


def run_minimal_baseline_sample(
    sample: BenchmarkSample,
    *,
    top_k: int = 5,
    answer_runner: SingleRecallLLMAnswerRunner | None = None,
) -> BenchmarkPrediction:
    """Run the one-recall baseline for a single normalized benchmark sample."""

    result = run_benchmark(
        _SingleSampleBenchmarkAdapter(sample),
        _minimal_memory_adapter(top_k=top_k),
        answer_runner=answer_runner,
        limit=1,
    )
    if not result.predictions:
        raise RuntimeError("Single-sample benchmark run produced no predictions.")
    prediction = result.predictions[0]
    prediction.metadata = {
        **prediction.metadata,
        "retrieved_item_count": len(prediction.retrieved_source_ids),
    }
    return prediction


def run_minimal_baseline(
    *,
    benchmark_name: str,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
    longmemeval_variant: str = "s_cleaned",
    limit: int | None = None,
    top_k: int = 5,
    answer_runner: SingleRecallLLMAnswerRunner | None = None,
) -> list[BenchmarkPrediction]:
    """Run the minimal baseline across one supported benchmark."""

    adapter = _compat_benchmark_adapter(
        benchmark_name,
        benchmark_root=benchmark_root,
        longmemeval_variant=longmemeval_variant,
    )
    result = run_benchmark(
        adapter,
        _minimal_memory_adapter(top_k=top_k),
        answer_runner=answer_runner,
        limit=limit,
    )
    for prediction in result.predictions:
        prediction.metadata = {
            **prediction.metadata,
            "retrieved_item_count": len(prediction.retrieved_source_ids),
        }
    return result.predictions


def _iter_dmr_samples(benchmark_root: Path) -> Iterator[BenchmarkSample]:
    path = benchmark_root / "DMR" / "msc_self_instruct.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            history_observations = _dmr_history_observations(payload)
            self_instruct = payload.get("self_instruct", {})
            for speaker_key in ("A", "B"):
                answer = str(self_instruct.get(speaker_key, "")).strip()
                if not answer:
                    continue
                yield BenchmarkSample(
                    sample_id=f"dmr-{row_index}-{speaker_key.casefold()}",
                    benchmark_name="dmr",
                    history_observations=list(history_observations),
                    query=Query(
                        text=_dmr_query_text(payload, speaker_key=speaker_key),
                        metadata={"task": "dialogue_continuation", "target_speaker": speaker_key},
                    ),
                    reference_answer=answer,
                    metadata={
                        "row_index": row_index,
                        "target_speaker": speaker_key,
                        "dialog_turn_count": len(payload.get("dialog", [])),
                        "previous_dialog_count": len(payload.get("previous_dialogs", [])),
                    },
                )


def _dmr_history_observations(payload: dict[str, Any]) -> list[Observation]:
    observations: list[Observation] = []
    previous_dialogs = list(payload.get("previous_dialogs", []))
    for dialogue_index, dialogue_payload in enumerate(previous_dialogs, start=1):
        dialog_turns = list(dialogue_payload.get("dialog", []))
        time_back = str(dialogue_payload.get("time_back", "")).strip()
        for turn_index, turn in enumerate(dialog_turns, start=1):
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            observations.append(
                Observation(
                    text=text,
                    source="dialogue",
                    metadata={
                        "benchmark": "dmr",
                        "history_scope": "previous_dialog",
                        "dialogue_index": dialogue_index,
                        "turn_index": turn_index,
                        "time_back": time_back,
                    },
                )
            )
    for turn_index, turn in enumerate(payload.get("dialog", []), start=1):
        speaker = str(turn.get("id", "")).strip()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        prefix = f"{speaker}: " if speaker else ""
        observations.append(
            Observation(
                text=f"{prefix}{text}",
                source="dialogue",
                metadata={
                    "benchmark": "dmr",
                    "history_scope": "current_dialog",
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "convai2_id": turn.get("convai2_id"),
                },
            )
        )
    return observations


def _dmr_query_text(payload: dict[str, Any], *, speaker_key: str) -> str:
    dialogue = list(payload.get("dialog", []))
    last_turn = str(dialogue[-1].get("text", "")).strip() if dialogue else ""
    speaker_label = f"Speaker {1 if speaker_key == 'A' else 2}"
    if last_turn:
        return (
            f"Write the next reply as {speaker_label}. "
            f"Keep it consistent with the established multi-session conversation history and personal facts. "
            f"The current dialogue most recently said: {last_turn}"
        )
    return (
        f"Write the next reply as {speaker_label}. "
        "Keep it consistent with the established multi-session conversation history and personal facts."
    )


def _write_predictions_jsonl(predictions, output_path: Path) -> int:
    return write_predictions_jsonl(predictions, output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(VALID_BENCHMARKS), required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--longmemeval-variant", choices=sorted(VALID_LONGMEMEVAL_VARIANTS), default="s_cleaned")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    predictions: list[BenchmarkPrediction] = []
    for index, sample in enumerate(
        load_benchmark_samples(
            args.benchmark,
            benchmark_root=args.benchmark_root,
            longmemeval_variant=args.longmemeval_variant,
            limit=args.limit,
        ),
        start=1,
    ):
        print(f"[{index}] running {sample.benchmark_name}:{sample.sample_id}")
        predictions.append(run_minimal_baseline_sample(sample, top_k=args.top_k))
    written = _write_predictions_jsonl(predictions, args.output)
    print(f"wrote {written} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
