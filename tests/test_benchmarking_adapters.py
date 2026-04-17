from __future__ import annotations

from pathlib import Path

from memprimitive import FreeMemoryPipeline, MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AlwaysTrigger,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.benchmarking import (
    BenchmarkSample,
    ConversationTurn,
    FunctionMemoryAdapter,
    PairwiseDialogueMemoryAdapter,
    PipelineMemoryAdapter,
    create_mem0_memory_adapter,
    create_yaml_pipeline_memory_adapter,
    run_benchmark,
)


def _sample_with_turns(*texts: str, sample_id: str = "sample-1") -> BenchmarkSample:
    turns = [
        ConversationTurn(
            turn_id=f"turn-{index}",
            session_id="session-1",
            session_timestamp="2026-04-17T00:00:00Z",
            role="user" if index % 2 else "assistant",
            speaker="user" if index % 2 else "assistant",
            text=text,
        )
        for index, text in enumerate(texts, start=1)
    ]
    return BenchmarkSample(
        sample_id=sample_id,
        benchmark_name="locomo",
        history_observations=[],
        history_turns=turns,
        query=Query(text="What should be recalled?"),
        reference_answer="unused",
    )


def _build_recency_pipeline() -> MemoryPipeline:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory", theme="semantic", indices=("temporal",))]
        )
    )
    return MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=AlwaysTrigger(),
        organization=AppendOrganization(target_layer="memory"),
        retrieval=RecencyRetrieval(top_k=1, layer="memory"),
        readout=ConcatenateReadout(),
        store=store,
    )


def _build_free_recency_pipeline() -> FreeMemoryPipeline:
    store = MemoryStore(
        topology=StoreTopology.from_layers(
            [StoreLayerSpec(name="memory", theme="semantic", indices=("temporal",))]
        )
    )
    return FreeMemoryPipeline(
        modules=(
            PassThroughUnitFormation(),
            BasicRepresentation(elements=("text",)),
            AlwaysTrigger(),
            AppendOrganization(target_layer="memory"),
            RecencyRetrieval(top_k=1, layer="memory"),
            ConcatenateReadout(),
        ),
        store=store,
    )


def test_pipeline_memory_adapter_isolates_memory_pipeline_sessions() -> None:
    adapter = PipelineMemoryAdapter(name="recency", pipeline_factory=_build_recency_pipeline)

    first_session = adapter.create_session()
    first_session.load_case(_sample_with_turns("Alice likes tea.", sample_id="a"))
    first_recall = first_session.recall(Query(text="ignored"))

    second_session = adapter.create_session()
    second_session.load_case(_sample_with_turns("Bob likes coffee.", sample_id="b"))
    second_recall = second_session.recall(Query(text="ignored"))

    assert "Alice likes tea." in first_recall.text
    assert "Bob likes coffee." in second_recall.text
    assert "Alice likes tea." not in second_recall.text


def test_pipeline_memory_adapter_accepts_free_memory_pipeline() -> None:
    adapter = PipelineMemoryAdapter(name="free", pipeline_factory=_build_free_recency_pipeline)

    session = adapter.create_session()
    session.load_case(_sample_with_turns("Free pipeline memory."))
    recall = session.recall(Query(text="ignored"))

    assert "Free pipeline memory." in recall.text
    assert recall.metadata["session_kind"] == "pipeline"


def test_yaml_pipeline_memory_adapter_builds_fresh_pipeline_per_session(tmp_path: Path) -> None:
    config_path = tmp_path / "simple_pipeline.yml"
    config_path.write_text(
        (
            "version: 1\n"
            "root: pipeline\n"
            "objects:\n"
            "  shared_store:\n"
            "    $call: memprimitive.core.MemoryStore\n"
            "  pipeline:\n"
            "    $call: memprimitive.pipeline.MemoryPipeline\n"
            "    kwargs:\n"
            "      unit_formation: PassThroughUnitFormation\n"
            "      representation:\n"
            "        $call: BasicRepresentation\n"
            "        kwargs:\n"
            "          elements:\n"
            "            - text\n"
            "      write_trigger: AlwaysTrigger\n"
            "      organization:\n"
            "        $call: AppendOrganization\n"
            "        kwargs:\n"
            "          target_layer: default\n"
            "      evolution_trigger: NeverTrigger\n"
            "      memory_evolution: AppendOnlyEvolution\n"
            "      retrieval:\n"
            "        $call: RecencyRetrieval\n"
            "        kwargs:\n"
            "          top_k: 1\n"
            "          layer: default\n"
            "      readout: ConcatenateReadout\n"
            "      store:\n"
            "        $ref: shared_store\n"
        ),
        encoding="utf-8",
    )
    adapter = create_yaml_pipeline_memory_adapter(str(config_path), name="yaml-pipeline")

    first_session = adapter.create_session()
    first_session.load_case(_sample_with_turns("Alice from yaml.", sample_id="yaml-a"))
    first_recall = first_session.recall(Query(text="ignored"))

    second_session = adapter.create_session()
    second_session.load_case(_sample_with_turns("Bob from yaml.", sample_id="yaml-b"))
    second_recall = second_session.recall(Query(text="ignored"))

    assert "Alice from yaml." in first_recall.text
    assert "Bob from yaml." in second_recall.text
    assert "Alice from yaml." not in second_recall.text


def test_function_memory_adapter_normalizes_readout() -> None:
    def _system_factory() -> dict[str, object]:
        return {"pipeline": _build_recency_pipeline()}

    def _load_case(system: dict[str, object], sample: BenchmarkSample) -> dict[str, int]:
        pipeline = system["pipeline"]
        assert isinstance(pipeline, MemoryPipeline)
        for observation in sample.history_observations:
            pipeline.ingest(observation)
        return {"load_helper_calls": 1}

    def _recall(system: dict[str, object], query: Query):
        pipeline = system["pipeline"]
        assert isinstance(pipeline, MemoryPipeline)
        return pipeline.recall(query)

    adapter = FunctionMemoryAdapter(
        name="function",
        system_factory=_system_factory,
        load_case=_load_case,
        recall=_recall,
    )

    session = adapter.create_session()
    session.load_case(_sample_with_turns("Function adapter memory."))
    recall = session.recall(Query(text="ignored"))

    assert "Function adapter memory." in recall.text
    assert recall.metadata["load_helper_calls"] == 1


def test_pairwise_dialogue_memory_adapter_keeps_odd_last_turn() -> None:
    def _system_factory() -> dict[str, object]:
        return {"pairs": []}

    def _ingest_pair(system: dict[str, object], *, user_text: str, assistant_text: str, session_id: str, turn_id: str) -> None:
        system["pairs"].append(
            {
                "user_text": user_text,
                "assistant_text": assistant_text,
                "session_id": session_id,
                "turn_id": turn_id,
            }
        )

    def _recall(system: dict[str, object], query: Query):
        del query
        return " || ".join(
            f"{pair['user_text']} -> {pair['assistant_text']}" for pair in system["pairs"]
        )

    adapter = PairwiseDialogueMemoryAdapter(
        name="pairwise",
        system_factory=_system_factory,
        ingest_pair=_ingest_pair,
        recall=_recall,
    )

    session = adapter.create_session()
    session.load_case(
        _sample_with_turns(
            "User turn one.",
            "Assistant turn one.",
            "User turn two.",
        )
    )
    recall = session.recall(Query(text="ignored"))

    assert "User turn one. -> Assistant turn one." in recall.text
    assert "User turn two. -> " in recall.text
    assert recall.metadata["loaded_pair_count"] == 2


def test_create_mem0_memory_adapter_reuses_mem0_helpers(monkeypatch) -> None:
    from memprimitive.example.classics import mem0_memory

    def _build_system() -> dict[str, object]:
        return {"pairs": []}

    def _ingest_pair(system: dict[str, object], *, user_text: str, assistant_text: str, session_id: str, turn_id: str) -> None:
        system["pairs"].append((user_text, assistant_text, session_id, turn_id))

    def _recall_profile(system: dict[str, object], *, user_query: str) -> str:
        return f"{user_query} :: {len(system['pairs'])} pairs"

    monkeypatch.setattr(mem0_memory, "build_mem0_memory_system", _build_system)
    monkeypatch.setattr(mem0_memory, "ingest_message_pair", _ingest_pair)
    monkeypatch.setattr(mem0_memory, "recall_profile", _recall_profile)

    adapter = create_mem0_memory_adapter()
    session = adapter.create_session()
    session.load_case(
        _sample_with_turns(
            "Alice likes tea.",
            "Assistant confirms tea.",
            "Alice also likes jasmine.",
        )
    )
    recall = session.recall(Query(text="What should be remembered?"))

    assert recall.text == "What should be remembered? :: 2 pairs"
    assert recall.metadata["loaded_pair_count"] == 2
    assert recall.metadata["recall_helper"] == "recall_profile"


def test_run_benchmark_collects_prediction_and_scores() -> None:
    sample = _sample_with_turns("Scored memory.")

    class _BenchmarkAdapter:
        name = "fakebench"

        def iter_samples(self, *, limit: int | None = None):
            if limit == 0:
                return
            yield sample

        def score_prediction(self, *, prediction):
            assert prediction.memory_adapter_name == "recency"
            return {"exact_match": 1.0}

        def aggregate_scores(self, *, predictions):
            assert len(predictions) == 1
            return {"mean_exact_match": 1.0}

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return f"{sample.sample_id}::{memory_recall.text}"

    result = run_benchmark(
        _BenchmarkAdapter(),
        PipelineMemoryAdapter(name="recency", pipeline_factory=_build_recency_pipeline),
        answer_runner=_AnswerRunner(),
    )

    assert len(result.predictions) == 1
    assert result.predictions[0].scores == {"exact_match": 1.0}
    assert result.aggregate_scores == {"mean_exact_match": 1.0}
    assert result.predictions[0].memory_adapter_name == "recency"


def test_run_benchmark_without_scoring_hooks_still_runs() -> None:
    sample = _sample_with_turns("Unscored memory.")

    class _BenchmarkAdapter:
        name = "noscore"

        def iter_samples(self, *, limit: int | None = None):
            if limit == 0:
                return
            yield sample

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return f"{sample.sample_id}::{memory_recall.text}"

    result = run_benchmark(
        _BenchmarkAdapter(),
        PipelineMemoryAdapter(name="recency", pipeline_factory=_build_recency_pipeline),
        answer_runner=_AnswerRunner(),
    )

    assert len(result.predictions) == 1
    assert result.predictions[0].scores == {}
    assert result.aggregate_scores == {}
