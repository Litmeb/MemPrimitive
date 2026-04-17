from __future__ import annotations

import json
from pathlib import Path

from memprimitive.benchmarking.minimal_baseline import (
    BenchmarkSample,
    SingleRecallLLMAnswerRunner,
    _iter_json_array_file,
    create_minimal_benchmark_pipeline,
    load_benchmark_samples,
    run_minimal_baseline_sample,
)
from memprimitive.utils._runtime import Runtime


class FakeRuntime(Runtime):
    def __init__(self) -> None:
        pass

    def text(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        return f"ANSWER::{user.split('Retrieved memory:\\n', 1)[-1].strip()}"


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
                {"question": "What does Alice like?", "answer": "Tea", "category": 1, "evidence": ["D1:1"]},
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
