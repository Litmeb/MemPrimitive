"""Minimal benchmark baseline compatibility wrapper built on the new adapters."""

from __future__ import annotations

import argparse
import json
import re
import threading
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    EmbeddingSimilarityRetrieval,
    PassThroughUnitFormation,
)
from memprimitive.example.classics import amem_memory, mem0_memory, memmachine_memory

from ._bench_adapters import (
    DEFAULT_BENCHMARK_ROOT,
    VALID_LONGMEMEVAL_VARIANTS,
    _iter_json_array_file,
    create_benchmark_adapter,
    parse_comma_separated_values,
)
from ._memory_adapters import (
    PipelineMemoryAdapter,
    SharedConversationLoCoMoMemoryAdapter,
    create_dual_speaker_locomo_memory_adapter,
    create_generic_memory_binding_adapter,
    create_amem_memory_adapter,
    create_mem0_memory_adapter,
    create_memmachine_memory_adapter,
)
from ._runner import (
    MemMachineLoCoMoAnswerRunner,
    Mem0LoCoMoAnswerRunner,
    SingleRecallLLMAnswerRunner,
    run_benchmark,
    write_predictions_jsonl,
)
from ._types import AnswerRunner, BenchmarkPrediction, BenchmarkSample

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "outputs"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "minimal_baseline_predictions.jsonl"
VALID_BENCHMARKS = frozenset({"locomo", "longmemeval", "dmr"})


class _TqdmBenchmarkProgress:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.bar = None
        self.memory_bar = None
        self.qa_bars: dict[str, Any] = {}
        self.memory_bars: dict[str, Any] = {}
        self.user_totals: dict[str, int] = {}
        self.user_done: dict[str, int] = {}
        self.user_labels: dict[str, str] = {}
        self.user_memory_totals: dict[str, int] = {}
        self.user_positions: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(
        self,
        *,
        phase: str,
        total: int | None = None,
        total_turns: int | None = None,
        memory_turn_total: int | None = None,
        turn_index: int | None = None,
        turn_id: str | None = None,
        turn_increment: int | None = None,
        samples: list[BenchmarkSample] | None = None,
        sample: BenchmarkSample | None = None,
        session_key: str | None = None,
        group_sample_index: int | None = None,
        **_: Any,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._update(
                phase=phase,
                total=total,
                total_turns=total_turns,
                memory_turn_total=memory_turn_total,
                turn_index=turn_index,
                turn_id=turn_id,
                turn_increment=turn_increment,
                samples=samples,
                sample=sample,
                session_key=session_key,
                group_sample_index=group_sample_index,
            )

    def _update(
        self,
        *,
        phase: str,
        total: int | None = None,
        total_turns: int | None = None,
        memory_turn_total: int | None = None,
        turn_index: int | None = None,
        turn_id: str | None = None,
        turn_increment: int | None = None,
        samples: list[BenchmarkSample] | None = None,
        sample: BenchmarkSample | None = None,
        session_key: str | None = None,
        group_sample_index: int | None = None,
    ) -> None:
        if phase == "init":
            self._initialize(
                total=0 if total is None else total,
                samples=[] if samples is None else samples,
                memory_turn_total=memory_turn_total,
            )
            return
        if not self.qa_bars:
            self._initialize(total=0 if total is None else total, samples=[], memory_turn_total=memory_turn_total)
        if sample is not None and phase in {"start", "done"}:
            self._set_sample_status(sample, done=(phase == "done"))
        if sample is not None and phase == "memory_load_start":
            self._set_memory_status(sample, status="loading", session_key=session_key)
        if sample is not None and phase == "memory_init":
            self._set_memory_status(
                sample,
                status=f"0/{0 if total_turns is None else total_turns} loading",
                session_key=session_key,
            )
        if sample is not None and phase == "memory_turn_done":
            increment = 1 if turn_increment is None else int(turn_increment)
            self._set_memory_status(
                sample,
                status=(
                    f"{0 if turn_index is None else turn_index}"
                    f"/{'' if total_turns is None else total_turns} {turn_id or 'turn'}"
                ),
                session_key=session_key,
            )
            memory_bar = self._memory_bar_for(sample)
            if memory_bar is not None:
                memory_bar.update(increment)
        if sample is not None and phase == "memory_finish":
            self._set_memory_status(sample, status="loaded", session_key=session_key)
        if sample is not None and phase == "memory_loaded":
            self._set_memory_status(sample, status="loaded", session_key=session_key)
        if sample is not None and phase == "memory_reuse":
            if group_sample_index is None:
                self._set_memory_status(sample, status="reused", session_key=session_key)
            else:
                self._set_memory_status(sample, status=f"reused qa {group_sample_index}", session_key=session_key)
        if phase == "done" and sample is not None:
            qa_bar = self._qa_bar_for(sample)
            if qa_bar is not None:
                qa_bar.update(1)
        if phase == "finish":
            for bar in list(self.memory_bars.values()):
                bar.close()
            for bar in list(self.qa_bars.values()):
                bar.close()
            self.memory_bars.clear()
            self.qa_bars.clear()
            self.memory_bar = None
            self.bar = None

    def _initialize(
        self,
        *,
        total: int,
        samples: list[BenchmarkSample],
        memory_turn_total: int | None = None,
    ) -> None:
        self.user_totals.clear()
        self.user_done.clear()
        self.user_labels.clear()
        self.user_memory_totals.clear()
        self.user_positions.clear()
        seen_grouped_memory_keys: set[str] = set()
        for sample in samples:
            key = self._user_key(sample)
            self.user_totals[key] = self.user_totals.get(key, 0) + 1
            self.user_labels.setdefault(key, self._user_label(sample))
            if key not in self.user_positions:
                self.user_positions[key] = len(self.user_positions) * 2
            if self._uses_user_memory_group(sample):
                if key in seen_grouped_memory_keys:
                    continue
                seen_grouped_memory_keys.add(key)
            self.user_memory_totals[key] = self.user_memory_totals.get(key, 0) + len(sample.history_turns)
        if not samples and total:
            key = "benchmark"
            self.user_totals[key] = total
            self.user_labels[key] = "benchmark"
            self.user_memory_totals[key] = 0 if memory_turn_total is None else memory_turn_total
            self.user_positions[key] = 0
        for key in self.user_positions:
            self._ensure_user_bars(key)

    def _set_sample_status(self, sample: BenchmarkSample, *, done: bool) -> None:
        key = self._user_key(sample)
        self.user_totals.setdefault(key, 1)
        self.user_labels.setdefault(key, self._user_label(sample))
        self._ensure_user_bars(key)
        if done:
            self.user_done[key] = self.user_done.get(key, 0) + 1
        current_done = self.user_done.get(key, 0)
        user_total = self.user_totals.get(key, 1)
        status = "done" if done else "running"
        self.qa_bars[key].set_postfix_str(
            f"{self.user_labels[key]} {current_done}/{user_total} {status} {sample.sample_id}",
            refresh=True,
        )

    def _set_memory_status(self, sample: BenchmarkSample, *, status: str, session_key: str | None = None) -> None:
        key = self._user_key(sample)
        self.user_labels.setdefault(key, self._user_label(sample))
        self._ensure_user_bars(key)
        suffix = f" {session_key}" if session_key else ""
        self.memory_bars[key].set_postfix_str(
            f"{self.user_labels[key]} {status}{suffix}",
            refresh=True,
        )

    def _ensure_user_bars(self, key: str) -> None:
        if key not in self.user_positions:
            self.user_positions[key] = len(self.user_positions) * 2
        label = self.user_labels.get(key, key)
        position = self.user_positions[key]
        if key not in self.memory_bars:
            self.memory_bars[key] = tqdm(
                total=self.user_memory_totals.get(key, 0),
                unit="turn",
                dynamic_ncols=True,
                desc=f"memory {label}",
                position=position,
            )
        if key not in self.qa_bars:
            self.qa_bars[key] = tqdm(
                total=self.user_totals.get(key, 0),
                unit="qa",
                dynamic_ncols=True,
                desc=f"qa {label}",
                position=position + 1,
            )
        self.memory_bar = self.memory_bars[key]
        self.bar = self.qa_bars[key]

    def _memory_bar_for(self, sample: BenchmarkSample):
        key = self._user_key(sample)
        self._ensure_user_bars(key)
        return self.memory_bars.get(key)

    def _qa_bar_for(self, sample: BenchmarkSample):
        key = self._user_key(sample)
        self._ensure_user_bars(key)
        return self.qa_bars.get(key)

    @staticmethod
    def _uses_user_memory_group(sample: BenchmarkSample) -> bool:
        metadata = sample.metadata
        return bool(
            str(metadata.get("locomo_user_index", "")).strip()
            or str(metadata.get("locomo_sample_id", "")).strip()
        )

    @staticmethod
    def _user_key(sample: BenchmarkSample) -> str:
        metadata = sample.metadata
        user_index = str(metadata.get("locomo_user_index", "")).strip()
        if user_index:
            return f"locomo-user-{user_index}"
        sample_group = str(metadata.get("locomo_sample_id", "")).strip()
        if sample_group:
            return sample_group
        return str(sample.benchmark_name).strip() or "benchmark"

    @staticmethod
    def _user_label(sample: BenchmarkSample) -> str:
        metadata = sample.metadata
        user_index = str(metadata.get("locomo_user_index", "")).strip()
        speaker_a = str(metadata.get("speaker_a", "")).strip()
        speaker_b = str(metadata.get("speaker_b", "")).strip()
        if user_index and (speaker_a or speaker_b):
            speakers = "/".join(part for part in (speaker_a, speaker_b) if part)
            return f"user {user_index} ({speakers})"
        if user_index:
            return f"user {user_index}"
        sample_group = str(metadata.get("locomo_sample_id", "")).strip()
        if sample_group:
            return sample_group
        return str(sample.benchmark_name).strip() or "benchmark"


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
        if limit == 0:
            return
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


def _create_cli_memory_adapter(
    name: str,
    *,
    benchmark_name: str = "locomo",
    top_k: int | None,
    similar_top_k: int = 5,
    mem0_speaker_workers: int = 2,
    memmachine_stm_record_budget: int = 20,
    memmachine_profile_max_turns: int = 6,
    memory_binding: str | None = None,
    memory_binding_kwargs: dict[str, Any] | None = None,
):
    adapter_name = str(name).strip().casefold()
    normalized_benchmark_name = str(benchmark_name).strip().casefold()
    if adapter_name == "minimal":
        return _minimal_memory_adapter(top_k=5 if top_k is None else top_k)
    if adapter_name == "mem0":
        if normalized_benchmark_name == "longmemeval":
            recall_top_k = 30 if top_k is None else top_k
            return create_generic_memory_binding_adapter(
                lambda: mem0_memory.create_memory_binding(
                    recent_top_k=6,
                    recall_top_k=recall_top_k,
                    similar_top_k=similar_top_k,
                ),
                name="mem0",
            )
        return create_mem0_memory_adapter(
            top_k=top_k,
            similar_top_k=similar_top_k,
            speaker_workers=mem0_speaker_workers,
        )
    if adapter_name == "amem":
        if normalized_benchmark_name == "longmemeval":
            recall_top_k = 30 if top_k is None else top_k
            return create_generic_memory_binding_adapter(
                lambda: amem_memory.create_memory_binding(
                    note_namespace="amem",
                    candidate_k=5,
                    recall_top_k=recall_top_k,
                ),
                name="amem",
            )
        return create_amem_memory_adapter(
            top_k=top_k,
            speaker_workers=mem0_speaker_workers,
        )
    if adapter_name == "memmachine":
        if normalized_benchmark_name == "longmemeval":
            limit = 30 if top_k is None else top_k
            return create_generic_memory_binding_adapter(
                lambda: memmachine_memory.create_memory_binding(
                    limit=limit,
                    expand_context=3,
                    profile_top_k=10 if top_k is None else top_k,
                    stm_record_budget=memmachine_stm_record_budget,
                    profile_max_turns=memmachine_profile_max_turns,
                ),
                name="memmachine",
            )
        return create_memmachine_memory_adapter(
            top_k=top_k,
            stm_record_budget=memmachine_stm_record_budget,
            profile_max_turns=memmachine_profile_max_turns,
            speaker_workers=mem0_speaker_workers,
        )
    if adapter_name == "binding":
        binding_factory = _load_memory_binding_factory(
            memory_binding,
            kwargs=dict(memory_binding_kwargs or {}),
        )
        if normalized_benchmark_name == "longmemeval":
            return create_generic_memory_binding_adapter(
                binding_factory,
                name=_binding_adapter_name(memory_binding),
            )
        if normalized_benchmark_name == "locomo" and _locomo_binding_is_memmachine_shared_conversation(memory_binding):
            return SharedConversationLoCoMoMemoryAdapter(
                binding_factory=binding_factory,
                name=_binding_adapter_name(memory_binding),
            )
        return create_dual_speaker_locomo_memory_adapter(
            binding_factory,
            name=_binding_adapter_name(memory_binding),
            speaker_workers=mem0_speaker_workers,
        )
    raise ValueError("Unsupported memory adapter. Choose from ['binding', 'mem0', 'amem', 'memmachine', 'minimal'].")


def _load_memory_binding_factory(spec: str | None, *, kwargs: dict[str, Any]):
    if not spec:
        raise ValueError("--memory-binding is required when --memory-adapter binding.")
    module_name, separator, attr_name = str(spec).strip().partition(":")
    if not separator:
        module_name, _, attr_name = str(spec).strip().rpartition(".")
    if not module_name or not attr_name:
        raise ValueError("--memory-binding must be in 'module:factory' or 'module.factory' form.")
    target = getattr(import_module(module_name), attr_name)
    if not callable(target):
        raise TypeError(f"Memory binding target {spec!r} is not callable.")
    return lambda: target(**kwargs)


def _locomo_binding_is_memmachine_shared_conversation(memory_binding: str | None) -> bool:
    """True when ``binding`` targets the classic MemMachine factory (LoCoMo shared-session path)."""

    spec = str(memory_binding or "").strip()
    if not spec:
        return False
    module_name, separator, attr_name = spec.partition(":")
    if not separator:
        module_name, _, attr_name = spec.rpartition(".")
    module_tail = module_name.strip().rsplit(".", 1)[-1].strip().casefold()
    return module_tail == "memmachine_memory" and attr_name.strip() == "create_memory_binding"


def _binding_adapter_name(spec: str | None) -> str:
    if not spec:
        return "binding"
    tail = str(spec).strip().replace(":", ".").rsplit(".", 1)[-1]
    return tail or "binding"


def _create_cli_answer_runner(
    *,
    benchmark_name: str,
    memory_adapter_name: str,
    memory_binding: str | None = None,
    llm_max_input_tokens: int | None = None,
):
    if benchmark_name == "locomo" and memory_adapter_name in {"amem", "memmachine"}:
        return MemMachineLoCoMoAnswerRunner(max_input_tokens=llm_max_input_tokens)
    if (
        benchmark_name == "locomo"
        and memory_adapter_name == "binding"
        and _locomo_binding_is_memmachine_shared_conversation(memory_binding)
    ):
        return MemMachineLoCoMoAnswerRunner(max_input_tokens=llm_max_input_tokens)
    if benchmark_name == "locomo" and memory_adapter_name in {"binding", "mem0"}:
        return Mem0LoCoMoAnswerRunner(max_input_tokens=llm_max_input_tokens)
    return SingleRecallLLMAnswerRunner(max_input_tokens=llm_max_input_tokens)


def _compat_benchmark_adapter(
    name: str,
    *,
    benchmark_root: Path | str,
    longmemeval_variant: str,
    locomo_users: str | list[str] | tuple[str, ...] | None = None,
):
    benchmark_name = str(name).strip().casefold()
    if benchmark_name == "dmr":
        return _LegacyDMRBenchmarkAdapter(benchmark_root=benchmark_root)
    return create_benchmark_adapter(
        benchmark_name,
        benchmark_root=benchmark_root,
        longmemeval_variant=longmemeval_variant,
        locomo_users=locomo_users,
    )


def load_benchmark_samples(
    name: str,
    *,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
    longmemeval_variant: str = "s_cleaned",
    locomo_users: str | list[str] | tuple[str, ...] | None = None,
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
        locomo_users=locomo_users,
    )
    yield from adapter.iter_samples(limit=limit)


def run_minimal_baseline_sample(
    sample: BenchmarkSample,
    *,
    top_k: int = 5,
    answer_runner: AnswerRunner | None = None,
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
    locomo_users: str | list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    top_k: int = 5,
    answer_runner: AnswerRunner | None = None,
) -> list[BenchmarkPrediction]:
    """Run the minimal baseline across one supported benchmark."""

    adapter = _compat_benchmark_adapter(
        benchmark_name,
        benchmark_root=benchmark_root,
        longmemeval_variant=longmemeval_variant,
        locomo_users=locomo_users,
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


def _safe_output_name_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._+-]+", "_", value.strip())
    component = re.sub(r"_+", "_", component).strip("._-")
    return component or "unknown"


def _default_output_dataset_name(*, benchmark_name: str, longmemeval_variant: str) -> str:
    if benchmark_name == "longmemeval":
        return f"{benchmark_name}_{longmemeval_variant}"
    return benchmark_name


def _default_output_user_name(locomo_users: str | list[str] | tuple[str, ...] | None) -> str:
    filters = parse_comma_separated_values(locomo_users)
    if not filters:
        return "all-users"
    return "+".join(_safe_output_name_component(item) for item in filters)


def _build_default_output_path(
    *,
    benchmark_name: str,
    memory_adapter_name: str,
    smoke_test: bool,
    locomo_users: str | list[str] | tuple[str, ...] | None,
    longmemeval_variant: str,
    timestamp: datetime | None = None,
) -> Path:
    timestamp = datetime.now() if timestamp is None else timestamp
    parts = [
        _default_output_dataset_name(benchmark_name=benchmark_name, longmemeval_variant=longmemeval_variant),
        memory_adapter_name,
        "smoke" if smoke_test else "full",
        _default_output_user_name(locomo_users),
        timestamp.strftime("%Y%m%d_%H%M%S"),
    ]
    filename = "_".join(_safe_output_name_component(part) for part in parts) + ".jsonl"
    return DEFAULT_OUTPUT_DIR / filename


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(VALID_BENCHMARKS), required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--longmemeval-variant", choices=sorted(VALID_LONGMEMEVAL_VARIANTS), default="s_cleaned")
    parser.add_argument(
        "--locomo-users",
        nargs="+",
        default=None,
        help=(
            "Comma-separated LoCoMo conversation filters. "
            "Each value may be a 1-based conversation index, locomo sample_id, or speaker name."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=None,
        help="Limit each sample to the first N history turns before loading memory.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a cheap smoke test: first 10 history turns per sample and first 10 questions.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--similar-top-k",
        type=int,
        default=5,
        help="Mem0 write-time similar-memory top-k. Only used with --memory-adapter mem0.",
    )
    parser.add_argument(
        "--mem0-speaker-workers",
        type=int,
        default=2,
        help="Parallel speaker workers for Mem0 LoCoMo memory load and recall. Ignored by A-MEM and MemMachine shared-conversation adapters.",
    )
    parser.add_argument(
        "--memmachine-stm-record-budget",
        type=int,
        default=20,
        help="MemMachine STM working-record budget before consolidation. Only used with --memory-adapter memmachine.",
    )
    parser.add_argument(
        "--memmachine-profile-max-turns",
        type=int,
        default=6,
        help="Max tool-call agent turns for MemMachine profile evolution. Only used with --memory-adapter memmachine.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel QA recall/answer workers. Increase for API-backed runs if rate limits allow.",
    )
    parser.add_argument(
        "--memory-adapter",
        choices=("minimal", "mem0", "amem", "memmachine", "binding"),
        default="minimal",
        help=(
            "Memory system to evaluate. 'minimal' preserves the legacy baseline; "
            "'mem0', 'amem', and 'memmachine' run classics reconstructions; "
            "'binding' loads a MemorySystemBinding factory from --memory-binding."
        ),
    )
    parser.add_argument(
        "--memory-binding",
        default=None,
        help="Import path for --memory-adapter binding, for example memprimitive.example.classics.mem0_memory:create_memory_binding.",
    )
    parser.add_argument(
        "--memory-binding-kwargs",
        default="{}",
        help="JSON object passed to the binding factory when --memory-adapter binding.",
    )
    parser.add_argument(
        "--llm-max-input-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only with --benchmark locomo: cap the answer LLM request size (system + user) to N tokens "
            "before the API call, using tiktoken for MEMPRIMITIVE_MODEL when installed (~chars/4 fallback)."
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm benchmark progress bar.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Prediction JSONL path. Defaults to "
            "benchmarks/outputs/<benchmark>_<memory-adapter>_<smoke-or-full>_<users>_<timestamp>.jsonl."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    benchmark_name = str(args.benchmark).strip().casefold()
    if args.llm_max_input_tokens is not None and benchmark_name != "locomo":
        parser.error("--llm-max-input-tokens is only supported when --benchmark locomo")
    memory_adapter_name = str(args.memory_adapter).strip().casefold()
    output_path = args.output
    if output_path is None:
        output_path = _build_default_output_path(
            benchmark_name=benchmark_name,
            memory_adapter_name=memory_adapter_name,
            smoke_test=bool(args.smoke_test),
            locomo_users=args.locomo_users,
            longmemeval_variant=args.longmemeval_variant,
        )
    limit = args.limit
    max_history_turns = args.max_history_turns
    if args.smoke_test:
        limit = 10 if limit is None else min(limit, 10)
        max_history_turns = 10 if max_history_turns is None else min(max_history_turns, 10)
    benchmark_adapter = _compat_benchmark_adapter(
        benchmark_name,
        benchmark_root=args.benchmark_root,
        longmemeval_variant=args.longmemeval_variant,
        locomo_users=args.locomo_users,
    )
    binding_kwargs = json.loads(args.memory_binding_kwargs)
    if not isinstance(binding_kwargs, dict):
        raise ValueError("--memory-binding-kwargs must be a JSON object.")
    memory_adapter = _create_cli_memory_adapter(
        memory_adapter_name,
        benchmark_name=benchmark_name,
        top_k=args.top_k,
        similar_top_k=args.similar_top_k,
        mem0_speaker_workers=args.mem0_speaker_workers,
        memmachine_stm_record_budget=args.memmachine_stm_record_budget,
        memmachine_profile_max_turns=args.memmachine_profile_max_turns,
        memory_binding=args.memory_binding,
        memory_binding_kwargs=binding_kwargs,
    )
    answer_runner = _create_cli_answer_runner(
        benchmark_name=benchmark_name,
        memory_adapter_name=memory_adapter_name,
        memory_binding=args.memory_binding,
        llm_max_input_tokens=args.llm_max_input_tokens,
    )
    result = run_benchmark(
        benchmark_adapter,
        memory_adapter,
        answer_runner=answer_runner,
        limit=limit,
        max_history_turns=max_history_turns,
        max_workers=args.max_workers,
        progress_callback=_TqdmBenchmarkProgress(enabled=not args.no_progress),
    )
    predictions = result.predictions
    for index, prediction in enumerate(predictions, start=1):
        print(f"[{index}] ran {prediction.benchmark_name}:{prediction.sample_id}")
    written = _write_predictions_jsonl(predictions, output_path)
    print(f"wrote {written} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
