from __future__ import annotations

import json
from pathlib import Path

from memprimitive.benchmarking.minimal_baseline import (
    BenchmarkSample,
    MemMachineLoCoMoAnswerRunner,
    Mem0LoCoMoAnswerRunner,
    SingleRecallLLMAnswerRunner,
    _TqdmBenchmarkProgress,
    _build_arg_parser,
    _create_cli_answer_runner,
    _create_cli_memory_adapter,
    _iter_json_array_file,
    create_minimal_benchmark_pipeline,
    load_benchmark_samples,
    run_minimal_baseline_sample,
)
from memprimitive.benchmarking._types import MemoryRecall
from memprimitive.benchmarking._memory_adapters import (
    DualSpeakerLoCoMoMemoryAdapter,
    GenericMemoryBindingAdapter,
    SharedConversationLoCoMoMemoryAdapter,
)
from memprimitive.benchmarking.prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH, ANSWER_PROMPT_ZEP
from memprimitive.benchmarking.evals import evaluate_file
from memprimitive.benchmarking.generate_scores import summarize_scores
from memprimitive.core import MemoryStore, Packet, StoreLayerSpec, StoreTopology
from memprimitive.utils._llm_function_tools import WriteToolCallContext
from memprimitive.utils._mem0_family import build_fixed_profile_tools
from memprimitive.utils._runtime import Runtime


class FakeRuntime(Runtime):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def text(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return f"ANSWER::{user.split('Retrieved memory:\\n', 1)[-1].strip()}"


class CapturingRuntime(Runtime):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def text(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return "captured-answer"


def test_create_minimal_benchmark_pipeline_ingests_and_recalls() -> None:
    pipeline = create_minimal_benchmark_pipeline(top_k=2)
    pipeline.ingest(__import__("memprimitive").Observation(text="Alice likes tea.", source="dialogue"))
    pipeline.ingest(__import__("memprimitive").Observation(text="Bob likes coffee.", source="dialogue"))

    readout = pipeline.recall(__import__("memprimitive").Query(text="What does Alice like?"))

    assert readout.text
    assert readout.source_ids


def test_iter_json_array_file_streams_objects(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text('[{"a": 1}, {"b": 2}]', encoding="utf-8")

    items = list(_iter_json_array_file(path, chunk_size=4))

    assert items == [{"a": 1}, {"b": 2}]


def test_load_locomo_samples_normalizes_dialogue_and_qa(tmp_path: Path) -> None:
    data_dir = tmp_path / "LoCoMo" / "data"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01T00:00:00",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "I like tea."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "I like coffee."},
                ],
            },
            "qa": [
                {
                    "question": "What does Alice like?",
                    "answer": "Tea",
                    "category": 1,
                    "evidence": ["D1:1"],
                    "adversarial_answer": "Coffee",
                },
            ],
        }
    ]
    (data_dir / "locomo10.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    samples = list(load_benchmark_samples("locomo", benchmark_root=tmp_path))

    assert len(samples) == 1
    assert samples[0].query.text == "What does Alice like?"
    assert samples[0].history_turns[0].turn_id == "D1:1"
    assert samples[0].history_turns[0].session_id == "session_1"
    assert [obs.text for obs in samples[0].history_observations] == ["Alice: I like tea.", "Bob: I like coffee."]
    assert samples[0].metadata["adversarial_answer"] == "Coffee"


def test_load_locomo_samples_filters_by_user_values(tmp_path: Path) -> None:
    data_dir = tmp_path / "LoCoMo" / "data"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "I like tea."}],
            },
            "qa": [{"question": "What does Alice like?", "answer": "Tea", "category": 1}],
        },
        {
            "sample_id": "conv-2",
            "conversation": {
                "speaker_a": "Carol",
                "speaker_b": "Dave",
                "session_1": [{"speaker": "Carol", "dia_id": "D2:1", "text": "I like jazz."}],
            },
            "qa": [{"question": "What does Carol like?", "answer": "Jazz", "category": 2}],
        },
    ]
    (data_dir / "locomo10.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    by_index_and_id = list(load_benchmark_samples("locomo", benchmark_root=tmp_path, locomo_users="1,conv-2"))
    by_speaker = list(load_benchmark_samples("locomo", benchmark_root=tmp_path, locomo_users="Carol"))

    assert [sample.metadata["locomo_sample_id"] for sample in by_index_and_id] == ["conv-1", "conv-2"]
    assert [sample.metadata["locomo_sample_id"] for sample in by_speaker] == ["conv-2"]
    assert by_speaker[0].metadata["locomo_user_index"] == 2


def test_load_longmemeval_samples_flattens_haystack_sessions(tmp_path: Path) -> None:
    data_dir = tmp_path / "LongMemEval"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "question_id": "q-1",
            "question_type": "single-hop",
            "question": "Who likes tea?",
            "question_date": "2024-01-01",
            "answer": "Alice",
            "answer_session_ids": [1],
            "haystack_dates": ["2024-01-01"],
            "haystack_session_ids": [1],
            "haystack_sessions": [[{"role": "user", "content": "Alice likes tea."}]],
        }
    ]
    (data_dir / "longmemeval_s_cleaned.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    samples = list(load_benchmark_samples("longmemeval", benchmark_root=tmp_path, longmemeval_variant="s_cleaned"))

    assert len(samples) == 1
    assert samples[0].history_turns[0].session_id == "1"
    assert samples[0].history_observations[0].text == "user: Alice likes tea."
    assert samples[0].reference_answer == "Alice"


def test_load_dmr_samples_emits_two_targets(tmp_path: Path) -> None:
    data_dir = tmp_path / "DMR"
    data_dir.mkdir(parents=True)
    row = {
        "dialog": [
            {"text": "Hello", "id": "Speaker 1"},
            {"text": "Hi", "id": "Speaker 2"},
        ],
        "previous_dialogs": [
            {"dialog": [{"text": "We talked about tea."}], "time_back": "2 days ago"},
        ],
        "self_instruct": {"A": "I still like tea.", "B": "I remember that too."},
    }
    (data_dir / "msc_self_instruct.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    samples = list(load_benchmark_samples("dmr", benchmark_root=tmp_path))

    assert len(samples) == 2
    assert samples[0].metadata["target_speaker"] == "A"
    assert "multi-session conversation history" in samples[0].query.text
    assert len(samples[0].history_observations) == 3


def test_run_minimal_baseline_sample_uses_answer_runner() -> None:
    sample = BenchmarkSample(
        sample_id="sample-1",
        benchmark_name="locomo",
        history_observations=[
            __import__("memprimitive").Observation(text="Alice likes jasmine tea.", source="dialogue"),
            __import__("memprimitive").Observation(text="Bob likes coffee.", source="dialogue"),
        ],
        query=__import__("memprimitive").Query(text="What does Alice like?"),
        reference_answer="jasmine tea",
    )
    runner = SingleRecallLLMAnswerRunner(runtime=FakeRuntime())

    prediction = run_minimal_baseline_sample(sample, top_k=1, answer_runner=runner)

    assert prediction.predicted_answer.startswith("ANSWER::")
    assert prediction.retrieved_source_ids
    assert prediction.memory_adapter_name == "minimal_pipeline"


def test_mem0_locomo_answer_prompt_matches_upstream_key_fragments() -> None:
    assert "4 May 2022" in ANSWER_PROMPT
    assert "Ignore the reference" in ANSWER_PROMPT
    assert "# APPROACH" in ANSWER_PROMPT
    assert "4 May 2022" in ANSWER_PROMPT_GRAPH
    assert "# APPROACH" in ANSWER_PROMPT_ZEP


def test_mem0_locomo_answer_runner_renders_prompt_and_uses_timestamped_memory() -> None:
    runtime = CapturingRuntime()
    runner = Mem0LoCoMoAnswerRunner(runtime=runtime)
    sample = BenchmarkSample(
        sample_id="locomo-1",
        benchmark_name="locomo",
        history_observations=[],
        query=__import__("memprimitive").Query(text="What did Alice do?"),
        reference_answer="unused",
        metadata={"speaker_a": "Alice", "speaker_b": "Bob"},
    )

    answer = runner.answer(
        sample=sample,
        memory_recall=MemoryRecall(
            text="Alice memory block\n\nBob memory block",
            metadata={
                "speaker_1_name": "Alice",
                "speaker_2_name": "Bob",
                "speaker_1_memories": "2022-05-04 Alice went to India last year.",
                "speaker_2_memories": "2022-05-04 Bob stayed home.",
            },
        ),
    )

    assert answer == "captured-answer"
    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["system"] == ""
    user_prompt = str(call["user"])
    assert "Memories for user Alice" in user_prompt
    assert "Memories for user Bob" in user_prompt
    assert "What did Alice do?" in user_prompt
    assert "2022-05-04 Alice went to India last year." in user_prompt
    assert "2022-05-04 Bob stayed home." in user_prompt
    assert "You answer only from the provided retrieved memory" not in user_prompt


def test_cli_answer_runner_switches_to_mem0_locomo_for_locomo_mem0() -> None:
    runner = _create_cli_answer_runner(benchmark_name="locomo", memory_adapter_name="mem0")
    assert isinstance(runner, Mem0LoCoMoAnswerRunner)

    amem_runner = _create_cli_answer_runner(benchmark_name="locomo", memory_adapter_name="amem")
    assert isinstance(amem_runner, MemMachineLoCoMoAnswerRunner)

    memmachine_runner = _create_cli_answer_runner(benchmark_name="locomo", memory_adapter_name="memmachine")
    assert isinstance(memmachine_runner, MemMachineLoCoMoAnswerRunner)

    binding_runner = _create_cli_answer_runner(benchmark_name="locomo", memory_adapter_name="binding")
    assert isinstance(binding_runner, Mem0LoCoMoAnswerRunner)

    other_runner = _create_cli_answer_runner(benchmark_name="longmemeval", memory_adapter_name="mem0")
    assert isinstance(other_runner, SingleRecallLLMAnswerRunner)


def test_cli_memory_adapter_can_load_binding_factory() -> None:
    adapter = _create_cli_memory_adapter(
        "binding",
        benchmark_name="locomo",
        top_k=None,
        memory_binding="memprimitive.example.classics.mem0_memory:create_memory_binding",
        memory_binding_kwargs={"recall_top_k": 2},
    )

    assert isinstance(adapter, DualSpeakerLoCoMoMemoryAdapter)
    assert adapter.name == "create_memory_binding"
    session = adapter.create_session()
    assert session.speaker_1_binding.recall_top_k == 2
    assert session.speaker_2_binding.recall_top_k == 2


def test_cli_memory_adapter_uses_generic_binding_for_longmemeval() -> None:
    adapter = _create_cli_memory_adapter(
        "binding",
        benchmark_name="longmemeval",
        top_k=None,
        memory_binding="memprimitive.example.classics.mem0_memory:create_memory_binding",
        memory_binding_kwargs={"recall_top_k": 2},
    )

    assert isinstance(adapter, GenericMemoryBindingAdapter)
    assert adapter.name == "create_memory_binding"
    session = adapter.create_session()
    assert session.binding.recall_top_k == 2


def test_cli_memory_adapter_can_build_amem_adapter() -> None:
    adapter = _create_cli_memory_adapter("amem", top_k=7)

    assert isinstance(adapter, SharedConversationLoCoMoMemoryAdapter)
    assert adapter.name == "amem"
    session = adapter.create_session()
    assert session.binding.recall_top_k == 7


def test_cli_classic_memory_adapters_use_generic_binding_for_longmemeval() -> None:
    mem0_adapter = _create_cli_memory_adapter("mem0", benchmark_name="longmemeval", top_k=7)
    amem_adapter = _create_cli_memory_adapter("amem", benchmark_name="longmemeval", top_k=8)
    memmachine_adapter = _create_cli_memory_adapter(
        "memmachine",
        benchmark_name="longmemeval",
        top_k=9,
        memmachine_stm_record_budget=3,
    )

    assert isinstance(mem0_adapter, GenericMemoryBindingAdapter)
    assert isinstance(amem_adapter, GenericMemoryBindingAdapter)
    assert isinstance(memmachine_adapter, GenericMemoryBindingAdapter)
    mem0_session = mem0_adapter.create_session()
    amem_session = amem_adapter.create_session()
    memmachine_session = memmachine_adapter.create_session()
    assert mem0_session.binding.recall_top_k == 7
    assert amem_session.binding.recall_top_k == 8
    assert memmachine_session.binding.limit == 9
    assert memmachine_session.binding.stm_record_budget == 3


def test_cli_memory_adapter_choices_include_amem() -> None:
    parser = _build_arg_parser()
    action = next(item for item in parser._actions if item.dest == "memory_adapter")

    assert "amem" in action.choices


def test_tqdm_progress_groups_locomo_samples_by_user() -> None:
    progress = _TqdmBenchmarkProgress(enabled=False)
    sample = BenchmarkSample(
        sample_id="locomo-1-qa-1",
        benchmark_name="locomo",
        history_observations=[],
        query=__import__("memprimitive").Query(text="Question?"),
        reference_answer="Answer",
        metadata={"locomo_user_index": 1, "speaker_a": "Alice", "speaker_b": "Bob"},
    )

    assert progress._user_key(sample) == "locomo-user-1"
    assert progress._user_label(sample) == "user 1 (Alice/Bob)"


def test_mem0_style_eval_reads_benchmark_jsonl(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    predictions_path.write_text(
        json.dumps(
            {
                "benchmark_name": "locomo",
                "query_text": "What does Alice like?",
                "reference_answer": "tea",
                "predicted_answer": "Alice likes tea.",
                "metadata": {"locomo_sample_id": "conv-1", "qa_category": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = evaluate_file(
        input_file=predictions_path,
        output_file=metrics_path,
        max_workers=1,
        use_llm_judge=False,
    )
    summary = summarize_scores(metrics_path)

    assert results["conv-1"][0]["f1_score"] > 0
    assert summary["overall"]["count"] == 1
    assert summary["by_category"]["1"]["count"] == 1


def test_mem0_style_eval_reads_legacy_and_prediction_jsonl(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "benchmark_name": "locomo",
                        "query_text": "What does Alice like?",
                        "reference_answer": "tea",
                        "predicted_answer": "Alice likes tea.",
                        "metadata": {
                            "locomo_sample_id": "conv-1",
                            "qa_category": 1,
                            "adversarial_answer": "coffee",
                            "speaker_1_memories": "Alice memory",
                            "num_speaker_1_memories": 1,
                        },
                        "memory_metadata": {
                            "speaker_1_memories": "Alice memory",
                            "speaker_2_memories": "Bob memory",
                            "num_speaker_1_memories": 1,
                            "num_speaker_2_memories": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "question": "What does Bob like?",
                        "answer": "coffee",
                        "response": "Bob likes coffee.",
                        "category": 2,
                        "locomo_sample_id": "conv-2",
                        "speaker_1_memories": "Carol memory",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results = evaluate_file(
        input_file=predictions_path,
        output_file=metrics_path,
        max_workers=1,
        use_llm_judge=False,
    )

    assert results["conv-1"][0]["category"] == "1"
    assert results["conv-2"][0]["question"] == "What does Bob like?"
    assert results["conv-1"][0]["response"] == "Alice likes tea."


def test_mem0_profile_tools_reject_invisible_update_without_mutation() -> None:
    store = MemoryStore(topology=StoreTopology.from_layers([StoreLayerSpec(name="profile")]))
    context = WriteToolCallContext(
        packet=Packet(),
        store=store,
        module_slot="memory_evolution",
        default_target_layer="profile",
        selected_records=[],
        visible_records=[],
    )
    update_tool = next(tool for tool in build_fixed_profile_tools(embed_on_add=False, embed_on_update=False) if tool.name == "UPDATE_PROFILE")

    result = update_tool.executor(context, {"record_id": "unit-not-a-record", "text": "bad update"})

    assert result.store.count("profile") == 0
    assert result.effects[0]["status"] == "rejected"
    assert result.effects[0]["record_id"] == "unit-not-a-record"
