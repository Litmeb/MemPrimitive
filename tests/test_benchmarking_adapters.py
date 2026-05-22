from __future__ import annotations

import threading
import time
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
    GenericMemoryBindingAdapter,
    PairwiseDialogueMemoryAdapter,
    PipelineMemoryAdapter,
    MemoryRecall,
    create_amem_memory_adapter,
    create_dual_speaker_locomo_memory_adapter,
    create_generic_memory_binding_adapter,
    create_mem0_memory_adapter,
    create_memmachine_memory_adapter,
    create_yaml_pipeline_memory_adapter,
    run_benchmark,
)
from memprimitive.benchmarking.minimal_baseline import _create_cli_memory_adapter
from memprimitive.benchmarking import _memory_adapters as memory_adapters_module
from memprimitive.benchmarking._memory_adapters import _locomo_message_event


def test_locomo_message_event_forwards_blip_caption() -> None:
    sample = BenchmarkSample(
        sample_id="conv-1-qa-1",
        benchmark_name="locomo",
        history_observations=[],
        history_turns=[],
        query=Query(text="What is in the photo?"),
        reference_answer="trail",
        metadata={
            "locomo_sample_id": "conv-1",
            "locomo_user_index": 1,
            "speaker_a": "Alice",
            "speaker_b": "Bob",
        },
    )
    turn = ConversationTurn(
        turn_id="D1:1",
        session_id="session_1",
        session_timestamp="2024-01-01T00:00:00",
        role="Alice",
        speaker="Alice",
        text="Check out this photo!",
        metadata={"blip_caption": "a person on a mountain trail"},
    )

    event = _locomo_message_event(sample, turn)

    assert event.text == "Check out this photo!"
    assert event.metadata["blip_caption"] == "a person on a mountain trail"


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


def test_create_mem0_memory_adapter_uses_per_speaker_systems(monkeypatch) -> None:
    from memprimitive.example.classics import mem0_memory

    build_calls: list[dict[str, object]] = []

    calls: list[dict[str, object]] = []

    def _build_system(*, recent_top_k: int, similar_top_k: int, recall_top_k: int) -> dict[str, object]:
        build_calls.append(
            {
                "recent_top_k": recent_top_k,
                "similar_top_k": similar_top_k,
                "recall_top_k": recall_top_k,
            }
        )
        return {"label": f"system-{len(build_calls)}", "pairs": []}

    def _ingest_pair(
        system: dict[str, object],
        *,
        user_text: str,
        assistant_text: str,
        session_id: str,
        turn_id: str,
        timestamp: str | None = None,
    ) -> None:
        calls.append(
            {
                "system_label": system["label"],
                "user_text": user_text,
                "assistant_text": assistant_text,
                "session_id": session_id,
                "turn_id": turn_id,
                "timestamp": timestamp,
            }
        )
        system["pairs"].append((user_text, assistant_text, session_id, turn_id))

    def _recall_profile(system: dict[str, object], *, user_query: str) -> str:
        return f"{system['label']} :: {user_query} :: {len(system['pairs'])} pairs"

    monkeypatch.setattr(mem0_memory, "build_mem0_memory_system", _build_system)
    monkeypatch.setattr(mem0_memory, "ingest_message_pair", _ingest_pair)
    monkeypatch.setattr(mem0_memory, "recall_profile", _recall_profile)

    adapter = create_mem0_memory_adapter(top_k=17)
    session = adapter.create_session()
    assert build_calls == [
        {
            "recent_top_k": 6,
            "similar_top_k": 5,
            "recall_top_k": 17,
        },
        {
            "recent_top_k": 6,
            "similar_top_k": 5,
            "recall_top_k": 17,
        },
    ]
    sample = BenchmarkSample(
        sample_id="mem0-timestamps",
        benchmark_name="locomo",
        history_observations=[],
        history_turns=[
            ConversationTurn(
                turn_id="turn-1",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="user",
                speaker="Alice",
                text="likes tea.",
            ),
            ConversationTurn(
                turn_id="turn-2",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="assistant",
                speaker="Bob",
                text="likes coffee.",
            ),
            ConversationTurn(
                turn_id="turn-3",
                session_id="session-1",
                session_timestamp="2026-04-17T08:10:00Z",
                role="user",
                speaker="Alice",
                text="also likes jasmine.",
            ),
        ],
        query=Query(text="What should be remembered?"),
        reference_answer="unused",
        metadata={
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "locomo_user_index": 1,
        },
    )
    session.load_case(
        sample
    )
    recall = session.recall(Query(text="What should be remembered?"), sample=sample)

    assert "system-1 :: What should be remembered? :: 2 pairs" in recall.text
    assert "system-2 :: What should be remembered? :: 1 pairs" in recall.text
    assert recall.metadata["loaded_pair_count"] == 2
    assert recall.metadata["recall_helper"] == "mem0.recall"
    assert recall.metadata["speaker_1_user_id"] == "Alice_1"
    assert recall.metadata["speaker_2_user_id"] == "Bob_1"
    assert recall.metadata["speaker_1_memories"] == "system-1 :: What should be remembered? :: 2 pairs"
    assert recall.metadata["speaker_2_memories"] == "system-2 :: What should be remembered? :: 1 pairs"
    assert recall.metadata["num_speaker_1_memories"] == 1
    assert recall.metadata["num_speaker_2_memories"] == 1
    assert [call["system_label"] for call in calls] == ["system-1", "system-2", "system-1"]
    assert calls[0]["user_text"] == "Alice: likes tea."
    assert calls[0]["assistant_text"] == "Bob: likes coffee."
    assert calls[1]["user_text"] == "Bob: likes coffee."
    assert calls[1]["assistant_text"] == "Alice: likes tea."
    assert calls[2]["user_text"] == "Alice: also likes jasmine."
    assert calls[2]["assistant_text"] == ""
    assert [call["timestamp"] for call in calls] == [
        "2026-04-17T08:00:00Z",
        "2026-04-17T08:00:00Z",
        "2026-04-17T08:10:00Z",
    ]


def test_create_memmachine_memory_adapter_uses_shared_conversation_system(monkeypatch) -> None:
    from memprimitive.example.classics import memmachine_memory

    build_calls: list[dict[str, object]] = []
    ingest_calls: list[dict[str, object]] = []

    def _build_system(
        *,
        stm_record_budget: int,
        profile_max_turns: int,
        limit: int,
        expand_context: int,
        sentence_top_k: int | None = None,
        episode_top_k: int | None = None,
        profile_top_k: int,
    ) -> dict[str, object]:
        build_calls.append(
            {
                "stm_record_budget": stm_record_budget,
                "profile_max_turns": profile_max_turns,
                "limit": limit,
                "expand_context": expand_context,
                "sentence_top_k": sentence_top_k,
                "episode_top_k": episode_top_k,
                "profile_top_k": profile_top_k,
            }
        )
        return {"label": f"memmachine-{len(build_calls)}", "episodes": []}

    def _ingest_episode(
        system: dict[str, object],
        *,
        text: str,
        session_id: str,
        user_id: str,
        producer: str,
        timestamp: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        ingest_calls.append(
            {
                "system_label": system["label"],
                "text": text,
                "session_id": session_id,
                "user_id": user_id,
                "producer": producer,
                "timestamp": timestamp,
                "metadata": dict(metadata or {}),
            }
        )
        system["episodes"].append(text)

    def _recall_memory(system: dict[str, object], *, user_query: str, include_profile: bool = False):
        del include_profile
        return MemoryRecall(text=f"{system['label']} :: {user_query} :: {len(system['episodes'])} episodes")

    monkeypatch.setattr(memmachine_memory, "build_memmachine_memory_system", _build_system)
    monkeypatch.setattr(memmachine_memory, "ingest_episode", _ingest_episode)
    monkeypatch.setattr(memmachine_memory, "recall_memmachine_memory", _recall_memory)

    adapter = create_memmachine_memory_adapter(top_k=12, stm_record_budget=3, profile_max_turns=10)
    session = adapter.create_session()
    assert build_calls == [
        {
            "stm_record_budget": 3,
            "profile_max_turns": 10,
            "limit": 12,
            "expand_context": 3,
            "sentence_top_k": None,
            "episode_top_k": None,
            "profile_top_k": 12,
        }
    ]

    sample = BenchmarkSample(
        sample_id="memmachine-timestamps",
        benchmark_name="locomo",
        history_observations=[],
        history_turns=[
            ConversationTurn(
                turn_id="turn-1",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="user",
                speaker="Alice",
                text="likes tea.",
            ),
            ConversationTurn(
                turn_id="turn-2",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="assistant",
                speaker="Bob",
                text="likes coffee.",
            ),
        ],
        query=Query(text="What should be remembered?"),
        reference_answer="unused",
        metadata={
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "locomo_user_index": 1,
        },
    )
    session.load_case(sample)
    recall = session.recall(Query(text="What should be remembered?"), sample=sample)

    assert "memmachine-1 :: What should be remembered? :: 2 episodes" in recall.text
    assert recall.metadata["loaded_message_count"] == 2
    assert recall.metadata["recall_helper"] == "memmachine.recall"
    assert recall.metadata["conversation_user_id"] == "conversation:locomo-user-1"
    assert [call["system_label"] for call in ingest_calls] == ["memmachine-1", "memmachine-1"]
    assert ingest_calls[0]["text"] == "likes tea."
    assert ingest_calls[1]["text"] == "likes coffee."
    assert ingest_calls[0]["user_id"] == "conversation:locomo-user-1"
    assert ingest_calls[1]["user_id"] == "conversation:locomo-user-1"
    assert ingest_calls[0]["metadata"]["source_speaker"] == "Alice"
    assert ingest_calls[1]["metadata"]["source_speaker"] == "Bob"
    assert [call["timestamp"] for call in ingest_calls] == [
        "2026-04-17T08:00:00Z",
        "2026-04-17T08:00:00Z",
    ]


def test_create_amem_memory_adapter_uses_shared_conversation_system(monkeypatch) -> None:
    from memprimitive.example.classics import amem_memory

    build_calls: list[dict[str, object]] = []
    ingest_calls: list[dict[str, object]] = []

    def _build_system(
        *,
        note_namespace: str,
        candidate_k: int,
        recall_top_k: int,
    ) -> dict[str, object]:
        build_calls.append(
            {
                "note_namespace": note_namespace,
                "candidate_k": candidate_k,
                "recall_top_k": recall_top_k,
            }
        )
        return {"label": f"amem-{len(build_calls)}", "notes": []}

    def _ingest_note(
        system: dict[str, object],
        *,
        text: str,
        source: str = "dialogue",
        timestamp: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        ingest_calls.append(
            {
                "system_label": system["label"],
                "text": text,
                "source": source,
                "timestamp": timestamp,
                "metadata": dict(metadata or {}),
            }
        )
        system["notes"].append(text)

    def _recall_memory(system: dict[str, object], *, user_query: str):
        return MemoryRecall(text=f"{system['label']} :: {user_query} :: {len(system['notes'])} notes")

    monkeypatch.setattr(amem_memory, "build_amem_memory_system", _build_system)
    monkeypatch.setattr(amem_memory, "ingest_note", _ingest_note)
    monkeypatch.setattr(amem_memory, "recall_amem_memory", _recall_memory)

    adapter = create_amem_memory_adapter(top_k=12)
    session = adapter.create_session()
    assert build_calls == [
        {
            "note_namespace": "amem",
            "candidate_k": 5,
            "recall_top_k": 12,
        }
    ]

    sample = BenchmarkSample(
        sample_id="amem-timestamps",
        benchmark_name="locomo",
        history_observations=[],
        history_turns=[
            ConversationTurn(
                turn_id="turn-1",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="user",
                speaker="Alice",
                text="likes tea.",
            ),
            ConversationTurn(
                turn_id="turn-2",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="assistant",
                speaker="Bob",
                text="likes coffee.",
            ),
        ],
        query=Query(text="What should be remembered?"),
        reference_answer="unused",
        metadata={
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "locomo_user_index": 1,
        },
    )
    session.load_case(sample)
    recall = session.recall(Query(text="What should be remembered?"), sample=sample)

    assert "amem-1 :: What should be remembered? :: 2 notes" in recall.text
    assert recall.metadata["loaded_message_count"] == 2
    assert recall.metadata["recall_helper"] == "amem.recall"
    assert recall.metadata["conversation_user_id"] == "conversation:locomo-user-1"
    assert [call["system_label"] for call in ingest_calls] == ["amem-1", "amem-1"]
    assert ingest_calls[0]["text"] == "likes tea."
    assert ingest_calls[1]["text"] == "likes coffee."
    assert ingest_calls[0]["metadata"]["source_speaker"] == "Alice"
    assert ingest_calls[1]["metadata"]["source_speaker"] == "Bob"
    assert [call["timestamp"] for call in ingest_calls] == [
        "2026-04-17T08:00:00Z",
        "2026-04-17T08:00:00Z",
    ]


def test_dual_speaker_locomo_adapter_accepts_any_memory_binding() -> None:
    events_by_system: dict[str, list[str]] = {}

    class _Binding:
        name = "custom"

        def build_system(self) -> dict[str, object]:
            system_id = f"system-{len(events_by_system) + 1}"
            events_by_system[system_id] = []
            return {"system_id": system_id}

        def ingest_event(self, system: dict[str, object], event) -> None:
            events_by_system[str(system["system_id"])].append(f"{event.user_id}: {event.text} -> {event.context_text}")

        def recall(self, system: dict[str, object], query: Query, *, context):
            return MemoryRecall(
                text=f"{context.user_id} sees {len(events_by_system[str(system['system_id'])])} events for {query.text}"
            )

    adapter = create_dual_speaker_locomo_memory_adapter(lambda: _Binding(), name="custom", speaker_workers=1)
    sample = BenchmarkSample(
        sample_id="custom-binding",
        benchmark_name="locomo",
        history_observations=[],
        history_turns=[
            ConversationTurn(
                turn_id="turn-1",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="user",
                speaker="Alice",
                text="likes tea.",
            ),
            ConversationTurn(
                turn_id="turn-2",
                session_id="session-1",
                session_timestamp="2026-04-17T08:00:00Z",
                role="assistant",
                speaker="Bob",
                text="likes coffee.",
            ),
        ],
        query=Query(text="tea"),
        reference_answer="unused",
        metadata={"speaker_a": "Alice", "speaker_b": "Bob", "locomo_user_index": 1},
    )

    session = adapter.create_session()
    session.load_case(sample)
    recall = session.recall(sample.query, sample=sample)

    assert adapter.session_key(sample=sample) == "locomo-user-1"
    assert events_by_system["system-1"] == ["Alice_1: Alice: likes tea. -> Bob: likes coffee."]
    assert events_by_system["system-2"] == ["Bob_1: Bob: likes coffee. -> Alice: likes tea."]
    assert "Alice_1 sees 1 events for tea" in recall.text
    assert "Bob_1 sees 1 events for tea" in recall.text
    assert recall.metadata["recall_helper"] == "custom.recall"


def test_generic_memory_binding_adapter_ingests_longmemeval_turns() -> None:
    events: list[object] = []
    contexts: list[object] = []

    class _Binding:
        name = "custom-generic"

        def build_system(self) -> dict[str, object]:
            return {"events": events}

        def ingest_event(self, system: dict[str, object], event) -> None:
            system_events = system["events"]
            assert isinstance(system_events, list)
            system_events.append(event)

        def recall(self, system: dict[str, object], query: Query, *, context):
            contexts.append(context)
            system_events = system["events"]
            assert isinstance(system_events, list)
            return MemoryRecall(
                text=f"{context.user_id} sees {len(system_events)} events for {query.text}",
                source_ids=["source-1"],
                metadata={"custom_recall": True},
            )

    adapter = create_generic_memory_binding_adapter(lambda: _Binding(), name="custom-generic")
    sample = BenchmarkSample(
        sample_id="q-1",
        benchmark_name="longmemeval",
        history_observations=[],
        history_turns=[
            ConversationTurn(
                turn_id="session-a-turn-1",
                session_id="session-a",
                session_timestamp="2024-01-01",
                role="user",
                speaker="user",
                text="Alice likes tea.",
                metadata={"turn_index": 1, "original_field": "kept"},
            ),
            ConversationTurn(
                turn_id="session-a-turn-2",
                session_id="session-a",
                session_timestamp="2024-01-01",
                role="assistant",
                speaker="assistant",
                text="Noted.",
                metadata={"turn_index": 2},
            ),
        ],
        query=Query(text="Who likes tea?", metadata={"question_type": "single-hop"}),
        reference_answer="Alice",
        metadata={"variant": "s_cleaned", "question_date": "2024-01-02"},
    )

    session = adapter.create_session()
    session.load_case(sample)
    recall = session.recall(sample.query, sample=sample)

    assert isinstance(adapter, GenericMemoryBindingAdapter)
    assert len(events) == 2
    first_event = events[0]
    assert first_event.text == "Alice likes tea."
    assert first_event.context_text == ""
    assert first_event.user_id == "longmemeval:q-1"
    assert first_event.session_id == "session-a"
    assert first_event.turn_id == "session-a-turn-1"
    assert first_event.timestamp == "2024-01-01"
    assert first_event.role == "user"
    assert first_event.speaker == "user"
    assert first_event.metadata["benchmark"] == "longmemeval"
    assert first_event.metadata["sample_id"] == "q-1"
    assert first_event.metadata["query_metadata"] == {"question_type": "single-hop"}
    assert first_event.metadata["turn_index"] == 1
    assert first_event.metadata["original_field"] == "kept"
    assert first_event.metadata["turn_metadata"] == {"turn_index": 1, "original_field": "kept"}
    assert first_event.metadata["sample_metadata"] == {"variant": "s_cleaned", "question_date": "2024-01-02"}
    assert len(contexts) == 1
    assert contexts[0].sample_id == "q-1"
    assert contexts[0].user_id == "longmemeval:q-1"
    assert contexts[0].metadata == {"variant": "s_cleaned", "question_date": "2024-01-02"}
    assert recall.text == "longmemeval:q-1 sees 2 events for Who likes tea?"
    assert recall.source_ids == ["source-1"]
    assert recall.metadata["conversation_user_id"] == "longmemeval:q-1"
    assert recall.metadata["custom_recall"] is True
    assert recall.metadata["recall_helper"] == "custom-generic.recall"


def test_create_mem0_memory_adapter_defaults_to_upstream_top_k(monkeypatch) -> None:
    from memprimitive.example.classics import mem0_memory

    build_calls: list[dict[str, object]] = []

    def _build_system(*, recent_top_k: int, similar_top_k: int, recall_top_k: int) -> dict[str, object]:
        build_calls.append(
            {
                "recent_top_k": recent_top_k,
                "similar_top_k": similar_top_k,
                "recall_top_k": recall_top_k,
            }
        )
        return {"pairs": []}

    monkeypatch.setattr(mem0_memory, "build_mem0_memory_system", _build_system)
    adapter = create_mem0_memory_adapter()
    adapter.create_session()

    assert build_calls == [
        {
            "recent_top_k": 6,
            "similar_top_k": 5,
            "recall_top_k": 30,
        },
        {
            "recent_top_k": 6,
            "similar_top_k": 5,
            "recall_top_k": 30,
        },
    ]


def test_create_mem0_memory_adapter_accepts_write_time_similar_top_k(monkeypatch) -> None:
    from memprimitive.example.classics import mem0_memory

    build_calls: list[dict[str, object]] = []

    def _build_system(*, recent_top_k: int, similar_top_k: int, recall_top_k: int) -> dict[str, object]:
        build_calls.append(
            {
                "recent_top_k": recent_top_k,
                "similar_top_k": similar_top_k,
                "recall_top_k": recall_top_k,
            }
        )
        return {"pairs": []}

    monkeypatch.setattr(mem0_memory, "build_mem0_memory_system", _build_system)
    adapter = create_mem0_memory_adapter(top_k=17, similar_top_k=3)
    adapter.create_session()

    assert build_calls == [
        {
            "recent_top_k": 6,
            "similar_top_k": 3,
            "recall_top_k": 17,
        },
        {
            "recent_top_k": 6,
            "similar_top_k": 3,
            "recall_top_k": 17,
        },
    ]


def test_cli_memory_adapter_defaults_by_adapter_name(monkeypatch) -> None:
    from memprimitive.example.classics import mem0_memory
    from memprimitive.example.classics import memmachine_memory

    build_calls: list[dict[str, object]] = []
    memmachine_build_calls: list[dict[str, object]] = []

    def _build_system(*, recent_top_k: int, similar_top_k: int, recall_top_k: int) -> dict[str, object]:
        build_calls.append(
            {
                "recent_top_k": recent_top_k,
                "similar_top_k": similar_top_k,
                "recall_top_k": recall_top_k,
            }
        )
        return {"pairs": []}

    def _build_memmachine_system(
        *,
        stm_record_budget: int,
        profile_max_turns: int,
        limit: int,
        expand_context: int,
        sentence_top_k: int | None = None,
        episode_top_k: int | None = None,
        profile_top_k: int,
    ) -> dict[str, object]:
        memmachine_build_calls.append(
            {
                "stm_record_budget": stm_record_budget,
                "profile_max_turns": profile_max_turns,
                "limit": limit,
                "expand_context": expand_context,
                "sentence_top_k": sentence_top_k,
                "episode_top_k": episode_top_k,
                "profile_top_k": profile_top_k,
            }
        )
        return {"episodes": []}

    monkeypatch.setattr(mem0_memory, "build_mem0_memory_system", _build_system)
    monkeypatch.setattr(memmachine_memory, "build_memmachine_memory_system", _build_memmachine_system)

    minimal_adapter = _create_cli_memory_adapter("minimal", top_k=None)
    mem0_adapter = _create_cli_memory_adapter("mem0", top_k=None)
    memmachine_adapter = _create_cli_memory_adapter("memmachine", top_k=None)

    assert minimal_adapter.pipeline_factory().retrieval.top_k == 5
    mem0_adapter.create_session()
    memmachine_adapter.create_session()

    assert build_calls == [
        {
            "recent_top_k": 6,
            "similar_top_k": 5,
            "recall_top_k": 30,
        },
        {
            "recent_top_k": 6,
            "similar_top_k": 5,
            "recall_top_k": 30,
        },
    ]
    assert memmachine_build_calls == [
        {
            "stm_record_budget": 20,
            "profile_max_turns": 6,
            "limit": 30,
            "expand_context": 3,
            "sentence_top_k": None,
            "episode_top_k": None,
            "profile_top_k": 10,
        }
    ]


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


def test_run_benchmark_emits_progress_events() -> None:
    sample = _sample_with_turns("Progress memory.", sample_id="progress")
    events: list[tuple[str, str | None, int | None]] = []

    class _BenchmarkAdapter:
        name = "progressbench"

        def iter_samples(self, *, limit: int | None = None):
            del limit
            yield sample

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return f"{sample.sample_id}::{memory_recall.text}"

    def _progress(*, phase: str, sample: BenchmarkSample | None = None, total: int | None = None) -> None:
        events.append((phase, sample.sample_id if sample is not None else None, total))

    run_benchmark(
        _BenchmarkAdapter(),
        PipelineMemoryAdapter(name="recency", pipeline_factory=_build_recency_pipeline),
        answer_runner=_AnswerRunner(),
        progress_callback=_progress,
    )

    assert events == [
        ("init", None, 1),
        ("memory_load_start", "progress", 1),
        ("memory_loaded", "progress", 1),
        ("start", "progress", 1),
        ("done", "progress", 1),
        ("finish", None, 1),
    ]


def test_run_benchmark_can_truncate_history_turns_for_smoke_runs() -> None:
    sample = _sample_with_turns(*(f"turn text {index}" for index in range(12)), sample_id="truncated")
    loaded_counts: list[tuple[int, int]] = []

    class _BenchmarkAdapter:
        name = "truncatebench"

        def iter_samples(self, *, limit: int | None = None):
            del limit
            yield sample

    class _MemorySession:
        def load_case(self, sample: BenchmarkSample) -> None:
            loaded_counts.append((len(sample.history_turns), len(sample.history_observations)))

        def recall(self, query: Query, *, sample: BenchmarkSample | None = None):
            del query
            assert sample is not None
            return MemoryRecall(text=f"{len(sample.history_turns)} turns loaded")

    class _MemoryAdapter:
        name = "truncate-memory"

        def create_session(self):
            return _MemorySession()

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return memory_recall.text

    result = run_benchmark(
        _BenchmarkAdapter(),
        _MemoryAdapter(),
        answer_runner=_AnswerRunner(),
        max_history_turns=10,
    )

    assert loaded_counts == [(10, 10)]
    prediction = result.predictions[0]
    assert prediction.predicted_answer == "10 turns loaded"
    assert prediction.metadata["history_turn_count"] == 10
    assert prediction.metadata["original_history_turn_count"] == 12
    assert prediction.metadata["max_history_turns"] == 10


def test_run_benchmark_reuses_session_for_same_memory_key() -> None:
    samples = [
        _sample_with_turns("Shared memory.", sample_id=f"conv-1-qa-{index}")
        for index in range(1, 4)
    ]
    for sample in samples:
        sample.metadata["locomo_sample_id"] = "conv-1"
    created_count = 0
    loaded_sample_ids: list[str] = []
    recalled_sample_ids: list[str] = []
    init_memory_totals: list[int] = []

    class _BenchmarkAdapter:
        name = "reusebench"

        def iter_samples(self, *, limit: int | None = None):
            del limit
            yield from samples

    class _MemorySession:
        def load_case(self, sample: BenchmarkSample) -> None:
            loaded_sample_ids.append(sample.sample_id)

        def recall(self, query: Query, *, sample: BenchmarkSample | None = None):
            del query
            assert sample is not None
            recalled_sample_ids.append(sample.sample_id)
            return MemoryRecall(text=f"memory for {sample.sample_id}")

    class _MemoryAdapter:
        name = "reuse-memory"

        def create_session(self):
            nonlocal created_count
            created_count += 1
            return _MemorySession()

        def session_key(self, *, sample: BenchmarkSample) -> str:
            return str(sample.metadata["locomo_sample_id"])

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return memory_recall.text

    result = run_benchmark(
        _BenchmarkAdapter(),
        _MemoryAdapter(),
        answer_runner=_AnswerRunner(),
        progress_callback=lambda *, phase, memory_turn_total=None: (
            init_memory_totals.append(memory_turn_total) if phase == "init" else None
        ),
    )

    assert created_count == 1
    assert loaded_sample_ids == ["conv-1-qa-1"]
    assert recalled_sample_ids == ["conv-1-qa-1", "conv-1-qa-2", "conv-1-qa-3"]
    assert [prediction.predicted_answer for prediction in result.predictions] == [
        "memory for conv-1-qa-1",
        "memory for conv-1-qa-2",
        "memory for conv-1-qa-3",
    ]
    assert init_memory_totals == [1]


def test_run_benchmark_parallel_workers_preserve_prediction_order() -> None:
    samples = [_sample_with_turns(f"memory {index}", sample_id=f"sample-{index}") for index in range(1, 4)]

    class _BenchmarkAdapter:
        name = "parallelbench"

        def iter_samples(self, *, limit: int | None = None):
            del limit
            yield from samples

    class _MemorySession:
        def load_case(self, sample: BenchmarkSample) -> None:
            del sample

        def recall(self, query: Query, *, sample: BenchmarkSample | None = None):
            del query
            assert sample is not None
            return MemoryRecall(text=f"memory for {sample.sample_id}")

    class _MemoryAdapter:
        name = "parallel-memory"

        def create_session(self):
            return _MemorySession()

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            if sample.sample_id == "sample-1":
                time.sleep(0.02)
            return f"{sample.sample_id}::{memory_recall.text}"

    result = run_benchmark(
        _BenchmarkAdapter(),
        _MemoryAdapter(),
        answer_runner=_AnswerRunner(),
        max_workers=2,
    )

    assert [prediction.sample_id for prediction in result.predictions] == [
        "sample-1",
        "sample-2",
        "sample-3",
    ]


def test_run_benchmark_parallel_workers_process_whole_user_groups() -> None:
    samples = [
        _sample_with_turns("conv 1 memory", sample_id="conv-1-qa-1"),
        _sample_with_turns("conv 2 memory", sample_id="conv-2-qa-1"),
        _sample_with_turns("conv 1 followup", sample_id="conv-1-qa-2"),
        _sample_with_turns("conv 2 followup", sample_id="conv-2-qa-2"),
    ]
    for sample in samples:
        sample.metadata["locomo_sample_id"] = sample.sample_id.rsplit("-qa-", 1)[0]

    load_barrier = threading.Barrier(2)
    lock = threading.Lock()
    created_count = 0
    loaded_sample_ids: list[str] = []
    recall_threads_by_user: dict[str, set[int]] = {}
    progress_events: list[tuple[str, str | None, str | None, int | None]] = []

    class _BenchmarkAdapter:
        name = "parallel-userbench"

        def iter_samples(self, *, limit: int | None = None):
            del limit
            yield from samples

    class _MemorySession:
        def __init__(self) -> None:
            self.session_key = ""

        def load_case(self, sample: BenchmarkSample) -> None:
            self.session_key = str(sample.metadata["locomo_sample_id"])
            with lock:
                loaded_sample_ids.append(sample.sample_id)
            load_barrier.wait(timeout=2)

        def recall(self, query: Query, *, sample: BenchmarkSample | None = None):
            del query
            assert sample is not None
            user_key = str(sample.metadata["locomo_sample_id"])
            assert user_key == self.session_key
            with lock:
                recall_threads_by_user.setdefault(user_key, set()).add(threading.get_ident())
            return MemoryRecall(text=f"{self.session_key}:{sample.sample_id}")

    class _MemoryAdapter:
        name = "parallel-user-memory"

        def create_session(self):
            nonlocal created_count
            with lock:
                created_count += 1
            return _MemorySession()

        def session_key(self, *, sample: BenchmarkSample) -> str:
            return str(sample.metadata["locomo_sample_id"])

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return f"{sample.sample_id}::{memory_recall.text}"

    def _progress(
        *,
        phase: str,
        sample: BenchmarkSample | None = None,
        session_key: str | None = None,
        group_sample_index: int | None = None,
    ) -> None:
        if phase in {"memory_load_start", "memory_reuse", "start", "done"}:
            with lock:
                progress_events.append(
                    (phase, sample.sample_id if sample is not None else None, session_key, group_sample_index)
                )

    result = run_benchmark(
        _BenchmarkAdapter(),
        _MemoryAdapter(),
        answer_runner=_AnswerRunner(),
        max_workers=2,
        progress_callback=_progress,
    )

    assert created_count == 2
    assert sorted(loaded_sample_ids) == ["conv-1-qa-1", "conv-2-qa-1"]
    assert recall_threads_by_user.keys() == {"conv-1", "conv-2"}
    assert all(len(thread_ids) == 1 for thread_ids in recall_threads_by_user.values())
    assert [prediction.sample_id for prediction in result.predictions] == [
        "conv-1-qa-1",
        "conv-2-qa-1",
        "conv-1-qa-2",
        "conv-2-qa-2",
    ]
    assert [prediction.predicted_answer for prediction in result.predictions] == [
        "conv-1-qa-1::conv-1:conv-1-qa-1",
        "conv-2-qa-1::conv-2:conv-2-qa-1",
        "conv-1-qa-2::conv-1:conv-1-qa-2",
        "conv-2-qa-2::conv-2:conv-2-qa-2",
    ]
    assert ("memory_load_start", "conv-1-qa-1", "conv-1", None) in progress_events
    assert ("memory_load_start", "conv-2-qa-1", "conv-2", None) in progress_events
    assert ("memory_reuse", "conv-1-qa-2", "conv-1", 2) in progress_events
    assert ("memory_reuse", "conv-2-qa-2", "conv-2", 2) in progress_events


def test_mem0_locomo_load_case_emits_turn_progress(monkeypatch) -> None:
    sample = _sample_with_turns("Alice likes tea.", "Bob likes coffee.", "Alice likes cake.", sample_id="locomo-progress")
    sample.metadata.update({"speaker_a": "user", "speaker_b": "assistant", "locomo_sample_id": "locomo-progress"})
    events: list[tuple[str, int | None, int | None, str | None]] = []
    ingested: list[tuple[str, str]] = []

    def _fake_build_system(**kwargs):
        del kwargs
        return {"fake": object()}

    def _fake_ingest(system, *, user_text, assistant_text, session_id, turn_id, timestamp):
        del system, session_id, timestamp
        ingested.append((turn_id, f"{user_text} -> {assistant_text}"))

    monkeypatch.setattr(memory_adapters_module.mem0_memory, "build_mem0_memory_system", _fake_build_system)
    monkeypatch.setattr(memory_adapters_module.mem0_memory, "ingest_message_pair", _fake_ingest)

    session = create_mem0_memory_adapter(top_k=1).create_session()
    session.load_case(
        sample,
        progress_callback=lambda *, phase, turn_index=None, total_turns=None, turn_id=None: events.append(
            (phase, turn_index, total_turns, turn_id)
        ),
    )

    assert events == [
        ("memory_init", None, 3, None),
        ("memory_turn_done", 2, 3, "turn-2"),
        ("memory_turn_done", 3, 3, "turn-3"),
        ("memory_finish", None, 3, None),
    ]
    assert session.load_metadata["loaded_turn_count"] == 3
    assert len(ingested) == 3


def test_mem0_locomo_load_case_can_parallelize_speakers(monkeypatch) -> None:
    sample = _sample_with_turns("Alice likes tea.", "Bob likes coffee.", sample_id="locomo-parallel")
    sample.metadata.update({"speaker_a": "user", "speaker_b": "assistant", "locomo_sample_id": "locomo-parallel"})
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    thread_ids: set[int] = set()

    def _fake_build_system(**kwargs):
        del kwargs
        return {"fake": object()}

    def _fake_ingest(system, *, user_text, assistant_text, session_id, turn_id, timestamp):
        del system, user_text, assistant_text, session_id, turn_id, timestamp
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait(timeout=2)

    monkeypatch.setattr(memory_adapters_module.mem0_memory, "build_mem0_memory_system", _fake_build_system)
    monkeypatch.setattr(memory_adapters_module.mem0_memory, "ingest_message_pair", _fake_ingest)

    session = create_mem0_memory_adapter(top_k=1, speaker_workers=2).create_session()
    session.load_case(sample)

    assert len(thread_ids) == 2
    assert session.load_metadata["speaker_workers"] == 2


def test_mem0_adapter_groups_qa_by_locomo_sample_id() -> None:
    adapter = create_mem0_memory_adapter(top_k=1)
    sample = _sample_with_turns("Memory.", sample_id="conv-1-qa-7")
    sample.metadata["locomo_sample_id"] = "conv-1"

    assert adapter.session_key(sample=sample) == "conv-1"


def test_run_benchmark_merges_memory_metadata_into_prediction_metadata() -> None:
    sample = _sample_with_turns("Merged metadata memory.", sample_id="merged")

    class _BenchmarkAdapter:
        name = "mergedbench"

        def iter_samples(self, *, limit: int | None = None):
            if limit == 0:
                return
            yield sample

    class _MemorySession:
        def load_case(self, sample: BenchmarkSample) -> None:
            del sample

        def recall(self, query: Query, *, sample: BenchmarkSample | None = None):
            del query, sample
            return MemoryRecall(
                text="merged memory",
                metadata={
                    "speaker_1_memories": "Alice memory",
                    "speaker_2_memories": "Bob memory",
                    "num_speaker_1_memories": 1,
                    "num_speaker_2_memories": 1,
                    "speaker_1_user_id": "Alice_1",
                    "speaker_2_user_id": "Bob_1",
                },
            )

    class _MemoryAdapter:
        name = "merged-memory"

        def create_session(self):
            return _MemorySession()

    class _AnswerRunner:
        name = "fake-answer"

        def answer(self, *, sample: BenchmarkSample, memory_recall):
            return f"{sample.sample_id}::{memory_recall.text}"

    result = run_benchmark(_BenchmarkAdapter(), _MemoryAdapter(), answer_runner=_AnswerRunner())

    prediction = result.predictions[0]
    assert prediction.metadata["speaker_1_memories"] == "Alice memory"
    assert prediction.metadata["speaker_2_memories"] == "Bob memory"
    assert prediction.metadata["num_speaker_1_memories"] == 1
    assert prediction.metadata["speaker_1_user_id"] == "Alice_1"
    assert prediction.to_json_dict()["memory_metadata"]["speaker_2_user_id"] == "Bob_1"


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
