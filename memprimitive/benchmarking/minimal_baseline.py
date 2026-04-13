"""Minimal benchmark baseline: ingest history, retrieve once, answer with a real LLM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    PassThroughUnitFormation,
)
from memprimitive.utils._runtime import Runtime

DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "outputs" / "minimal_baseline_predictions.jsonl"
VALID_BENCHMARKS = frozenset({"locomo", "longmemeval", "dmr"})
VALID_LONGMEMEVAL_VARIANTS = frozenset({"oracle", "s_cleaned", "m_cleaned"})


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    benchmark_name: str
    history_observations: list[Observation]
    query: Query
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkPrediction:
    sample_id: str
    benchmark_name: str
    query_text: str
    reference_answer: str
    predicted_answer: str
    retrieved_text: str
    retrieved_source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleRecallLLMAnswerRunner:
    """Answer from one retrieved memory block using the existing OpenAI-compatible runtime."""

    def __init__(
        self,
        *,
        runtime: Runtime | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.runtime = runtime if runtime is not None else Runtime()
        self.system_prompt = system_prompt or (
            "You answer only from the provided retrieved memory and the user request. "
            "If the retrieved memory is empty or insufficient, say that the memory does not contain enough information. "
            "Do not invent facts that are not grounded in the retrieved memory."
        )

    def answer(self, *, sample: BenchmarkSample, retrieved_text: str) -> str:
        retrieved_block = retrieved_text.strip() or "<no retrieved memory>"
        user_prompt = (
            f"Benchmark: {sample.benchmark_name}\n"
            f"Sample ID: {sample.sample_id}\n\n"
            f"User request:\n{sample.query.text}\n\n"
            f"Retrieved memory:\n{retrieved_block}\n"
        )
        return self.runtime.text(system=self.system_prompt, user=user_prompt, temperature=0.0)


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


def load_benchmark_samples(
    name: str,
    *,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
    longmemeval_variant: str = "s_cleaned",
    limit: int | None = None,
) -> Iterator[BenchmarkSample]:
    """Yield normalized samples for one supported benchmark."""

    benchmark_name = str(name).strip().casefold()
    root = Path(benchmark_root)
    if benchmark_name not in VALID_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark {name!r}. Choose from {sorted(VALID_BENCHMARKS)}.")

    if benchmark_name == "locomo":
        iterator = _iter_locomo_samples(root)
    elif benchmark_name == "longmemeval":
        iterator = _iter_longmemeval_samples(root, variant=longmemeval_variant)
    else:
        iterator = _iter_dmr_samples(root)

    yielded = 0
    for sample in iterator:
        yield sample
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def run_minimal_baseline_sample(
    sample: BenchmarkSample,
    *,
    top_k: int = 5,
    answer_runner: SingleRecallLLMAnswerRunner | None = None,
) -> BenchmarkPrediction:
    """Run the one-recall baseline for a single normalized benchmark sample."""

    pipeline = create_minimal_benchmark_pipeline(top_k=top_k)
    for observation in sample.history_observations:
        pipeline.ingest(observation)
    readout = pipeline.recall(sample.query)
    runner = answer_runner if answer_runner is not None else SingleRecallLLMAnswerRunner()
    predicted_answer = runner.answer(sample=sample, retrieved_text=readout.text)
    return BenchmarkPrediction(
        sample_id=sample.sample_id,
        benchmark_name=sample.benchmark_name,
        query_text=sample.query.text,
        reference_answer=sample.reference_answer,
        predicted_answer=predicted_answer,
        retrieved_text=readout.text,
        retrieved_source_ids=list(readout.source_ids),
        metadata={
            **sample.metadata,
            "history_observation_count": len(sample.history_observations),
            "retrieved_item_count": len(readout.source_ids),
            "readout_metadata": dict(readout.metadata),
        },
    )


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

    predictions: list[BenchmarkPrediction] = []
    for sample in load_benchmark_samples(
        benchmark_name,
        benchmark_root=benchmark_root,
        longmemeval_variant=longmemeval_variant,
        limit=limit,
    ):
        predictions.append(
            run_minimal_baseline_sample(
                sample,
                top_k=top_k,
                answer_runner=answer_runner,
            )
        )
    return predictions


def _iter_locomo_samples(benchmark_root: Path) -> Iterator[BenchmarkSample]:
    path = benchmark_root / "LoCoMo" / "data" / "locomo10.json"
    conversations = json.loads(path.read_text(encoding="utf-8"))
    for conversation_payload in conversations:
        sample_prefix = str(conversation_payload["sample_id"]).strip()
        observations = _locomo_history_observations(conversation_payload)
        for qa_index, qa_payload in enumerate(conversation_payload.get("qa", []), start=1):
            question = str(qa_payload.get("question", "")).strip()
            answer = str(qa_payload.get("answer", "")).strip()
            if not question or not answer:
                continue
            yield BenchmarkSample(
                sample_id=f"{sample_prefix}-qa-{qa_index}",
                benchmark_name="locomo",
                history_observations=list(observations),
                query=Query(text=question, metadata={"task": "question_answering"}),
                reference_answer=answer,
                metadata={
                    "locomo_sample_id": sample_prefix,
                    "qa_category": qa_payload.get("category"),
                    "evidence": list(qa_payload.get("evidence", [])),
                },
            )


def _locomo_history_observations(conversation_payload: dict[str, Any]) -> list[Observation]:
    conversation = conversation_payload.get("conversation", {})
    observations: list[Observation] = []
    session_numbers = sorted(
        int(key.split("_")[1])
        for key in conversation
        if key.startswith("session_") and key.count("_") == 1
    )
    for session_number in session_numbers:
        session_key = f"session_{session_number}"
        session_timestamp = str(conversation.get(f"{session_key}_date_time", "")).strip()
        for turn_index, turn in enumerate(conversation.get(session_key, []), start=1):
            speaker = str(turn.get("speaker", "")).strip() or "speaker"
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            observations.append(
                Observation(
                    text=f"{speaker}: {text}",
                    source="dialogue",
                    metadata={
                        "benchmark": "locomo",
                        "session_id": session_key,
                        "session_timestamp": session_timestamp,
                        "turn_index": turn_index,
                        "speaker": speaker,
                        "dialogue_id": turn.get("dia_id"),
                    },
                )
            )
    return observations


def _iter_longmemeval_samples(benchmark_root: Path, *, variant: str) -> Iterator[BenchmarkSample]:
    normalized_variant = str(variant).strip().casefold()
    if normalized_variant not in VALID_LONGMEMEVAL_VARIANTS:
        raise ValueError(
            f"Unsupported LongMemEval variant {variant!r}. Choose from {sorted(VALID_LONGMEMEVAL_VARIANTS)}."
        )
    path = benchmark_root / "LongMemEval" / _longmemeval_filename(normalized_variant)
    for payload in _iter_json_array_file(path):
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            continue
        yield BenchmarkSample(
            sample_id=str(payload.get("question_id", "")).strip() or f"longmemeval-{normalized_variant}",
            benchmark_name="longmemeval",
            history_observations=_longmemeval_history_observations(payload),
            query=Query(
                text=question,
                metadata={
                    "task": "question_answering",
                    "question_type": payload.get("question_type"),
                    "question_date": payload.get("question_date"),
                },
            ),
            reference_answer=answer,
            metadata={
                "variant": normalized_variant,
                "question_type": payload.get("question_type"),
                "question_date": payload.get("question_date"),
                "answer_session_ids": list(payload.get("answer_session_ids", [])),
                "haystack_session_ids": list(payload.get("haystack_session_ids", [])),
            },
        )


def _longmemeval_filename(variant: str) -> str:
    mapping = {
        "oracle": "longmemeval_oracle.json",
        "s_cleaned": "longmemeval_s_cleaned.json",
        "m_cleaned": "longmemeval_m_cleaned.json",
    }
    return mapping[variant]


def _longmemeval_history_observations(payload: dict[str, Any]) -> list[Observation]:
    observations: list[Observation] = []
    session_ids = list(payload.get("haystack_session_ids", []))
    session_dates = list(payload.get("haystack_dates", []))
    sessions = list(payload.get("haystack_sessions", []))
    for session_index, session in enumerate(sessions):
        session_id = session_ids[session_index] if session_index < len(session_ids) else session_index
        session_date = str(session_dates[session_index]).strip() if session_index < len(session_dates) else ""
        for turn_index, turn in enumerate(session, start=1):
            role = str(turn.get("role", "")).strip() or "speaker"
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            observations.append(
                Observation(
                    text=f"{role}: {content}",
                    source="dialogue",
                    metadata={
                        "benchmark": "longmemeval",
                        "session_id": session_id,
                        "session_date": session_date,
                        "turn_index": turn_index,
                        "speaker": role,
                    },
                )
            )
    return observations


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


def _iter_json_array_file(path: Path, *, chunk_size: int = 1 << 16) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            end_of_file = chunk == ""

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if not array_started:
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array file.")
                    array_started = True
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    payload, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if end_of_file:
                        raise
                    break
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSON array item.")
                yield payload
                buffer = buffer[offset:].lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    return

            if end_of_file:
                break
    raise ValueError(f"JSON array file {path} ended unexpectedly.")


def _write_predictions_jsonl(predictions: Iterable[BenchmarkPrediction], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_json_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


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
