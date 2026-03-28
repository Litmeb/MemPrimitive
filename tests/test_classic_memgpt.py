from __future__ import annotations

from memprimitive.utils import _runtime
from memprimitive.classic_modules.memgpt import (
    MEMGPT_ARCHIVAL_LAYER,
    MEMGPT_CORE_BLOCK_HUMAN,
    MEMGPT_CORE_BLOCK_PERSONA,
    MEMGPT_CORE_LAYER,
    MEMGPT_QUEUE_LAYER,
    MEMGPT_RECALL_LAYER,
    MEMGPT_REQUIRED_CORE_BLOCKS,
    MEMGPT_WORKING_LAYER,
    MEMGPT_WORKING_SUMMARY_KEY,
)
from memprimitive.example.classics.memgpt import DEFAULT_WORKING_SUMMARY, MemGPTAgent, main


class FakeRuntime:
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = list([] if responses is None else responses)
        self.chat_calls: list[dict] = []
        self.summary_calls: list[dict] = []
        self.embed_calls: list[str] = []

    def chat_with_tools(self, *, system: str, user: str, tools: list[dict], temperature: float = 0.0):
        self.chat_calls.append({"system": system, "user": user, "tools": tools, "temperature": temperature})
        if not self.responses:
            return {
                "assistant_message": "Acknowledged.",
                "tool_calls": [],
            }
        return self.responses.pop(0)

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        tokens = [len(part) for part in str(text).split()[:4]]
        padded = tokens + [0] * (4 - len(tokens))
        return [float(value) for value in padded[:4]]

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())

    def summarize_records(self, *, records: list[dict], instruction: str, max_sentences: int = 3) -> str:
        self.summary_calls.append(
            {
                "records": records,
                "instruction": instruction,
                "max_sentences": max_sentences,
            }
        )
        snippets = [str(item.get("text", "")).strip() for item in records if str(item.get("text", "")).strip()]
        return "Recursive summary: " + " | ".join(snippets[:3])


def _use_fake_runtime(fake_runtime: FakeRuntime) -> None:
    _runtime._DEFAULT_RUNTIME = fake_runtime


def test_memgpt_agent_boots_five_region_store_and_required_blocks() -> None:
    fake_runtime = FakeRuntime()
    _use_fake_runtime(fake_runtime)

    agent = MemGPTAgent()

    assert agent.store.topology.layer_names == (
        MEMGPT_CORE_LAYER,
        MEMGPT_WORKING_LAYER,
        MEMGPT_QUEUE_LAYER,
        MEMGPT_RECALL_LAYER,
        MEMGPT_ARCHIVAL_LAYER,
    )
    assert agent.store.count(MEMGPT_CORE_LAYER) == len(MEMGPT_REQUIRED_CORE_BLOCKS)
    assert agent.store.count(MEMGPT_WORKING_LAYER) == 1
    assert agent.store.count(MEMGPT_QUEUE_LAYER) == 0
    assert agent.store.count(MEMGPT_RECALL_LAYER) == 0
    assert agent.store.count(MEMGPT_ARCHIVAL_LAYER) == 0
    assert agent.pipelines["queue_ingest_pipeline"].store is agent.store
    assert agent.pipelines["conversation_search_pipeline"].store is agent.store

    core_keys = {record.metadata["memgpt_key"] for record in agent.store.iter_records(MEMGPT_CORE_LAYER)}
    assert core_keys == {MEMGPT_CORE_BLOCK_PERSONA, MEMGPT_CORE_BLOCK_HUMAN}
    working_record = agent.store.iter_records(MEMGPT_WORKING_LAYER)[0]
    assert working_record.metadata["memgpt_key"] == MEMGPT_WORKING_SUMMARY_KEY
    assert working_record.text == DEFAULT_WORKING_SUMMARY


def test_memgpt_run_turn_appends_user_and_assistant_events_to_recall_history() -> None:
    fake_runtime = FakeRuntime(
        responses=[
            {
                "assistant_message": "I will keep that in mind.",
                "tool_calls": [],
            }
        ]
    )
    _use_fake_runtime(fake_runtime)
    agent = MemGPTAgent(queue_token_budget=50)

    final = agent.run_turn("Remember that I like green tea.")

    assert final == "I will keep that in mind."
    recall_records = agent.store.iter_records(MEMGPT_RECALL_LAYER)
    assert len(recall_records) == 2
    event_types = [record.metadata["memgpt"]["event_type"] for record in recall_records]
    assert event_types == ["user", "assistant"]
    assert agent.store.count(MEMGPT_QUEUE_LAYER) == 2


def test_memgpt_overflow_emits_warning_then_flushes_queue_into_recursive_summary() -> None:
    fake_runtime = FakeRuntime(
        responses=[
            {
                "assistant_message": "First reply with several tokens to fill the queue quickly.",
                "tool_calls": [],
            },
            {
                "assistant_message": "Second reply after the warning.",
                "tool_calls": [],
            },
        ]
    )
    _use_fake_runtime(fake_runtime)
    agent = MemGPTAgent(queue_token_budget=12)

    final = agent.run_turn("This user message is intentionally long enough to overflow the queue budget.")

    assert final == "Second reply after the warning."
    recall_event_types = [record.metadata["memgpt"]["event_type"] for record in agent.store.iter_records(MEMGPT_RECALL_LAYER)]
    assert "memory_warning" in recall_event_types
    assert fake_runtime.summary_calls
    assert agent._queue_token_count() <= agent.queue_token_budget
    working_record = agent.store.iter_records(MEMGPT_WORKING_LAYER)[0]
    assert working_record.text.startswith("Recursive summary:")
    assert agent.store.count(MEMGPT_QUEUE_LAYER) < len(recall_event_types)


def test_memgpt_archival_insert_writes_only_to_archival_memory() -> None:
    fake_runtime = FakeRuntime()
    _use_fake_runtime(fake_runtime)
    agent = MemGPTAgent()

    result = agent.run_tool(
        "archival_memory_insert",
        {"memory": "The release checklist lives in the launch runbook."},
    )

    assert result["status"] == "inserted"
    assert result["target_layer"] == MEMGPT_ARCHIVAL_LAYER
    assert agent.store.count(MEMGPT_ARCHIVAL_LAYER) == 1
    assert agent.store.count(MEMGPT_QUEUE_LAYER) == 0
    assert agent.store.count(MEMGPT_RECALL_LAYER) == 0
    archival_record = agent.store.iter_records(MEMGPT_ARCHIVAL_LAYER)[0]
    assert archival_record.embedding is not None


def test_memgpt_search_tools_dispatch_to_their_own_layers() -> None:
    fake_runtime = FakeRuntime()
    _use_fake_runtime(fake_runtime)
    agent = MemGPTAgent()

    agent.run_tool("archival_memory_insert", {"memory": "Archive: release checklist and rollback plan."})
    agent._record_event(
        text="User previously mentioned the rollback plan in conversation.",
        event_type="user",
        source="user",
    )

    conversation = agent.run_tool("conversation_search", {"query": "rollback"})
    archival = agent.run_tool("archival_memory_search", {"query": "rollback"})

    assert conversation["target_layer"] == MEMGPT_RECALL_LAYER
    assert archival["target_layer"] == MEMGPT_ARCHIVAL_LAYER
    assert all(item["layer"] == MEMGPT_RECALL_LAYER for item in conversation["items"])
    assert all(item["layer"] == MEMGPT_ARCHIVAL_LAYER for item in archival["items"])


def test_memgpt_core_memory_append_and_replace_update_keyed_blocks() -> None:
    fake_runtime = FakeRuntime()
    _use_fake_runtime(fake_runtime)
    agent = MemGPTAgent(persona="Base persona.", human="Base human.")

    append_result = agent.run_tool("core_memory_append", {"block": "persona", "value": "Likes concise output."})
    replace_result = agent.run_tool("core_memory_replace", {"block": "human", "value": "The human prefers bulleted updates."})

    assert append_result["block"] == "persona"
    assert replace_result["block"] == "human"
    persona_record = next(
        record for record in agent.store.iter_records(MEMGPT_CORE_LAYER) if record.metadata["memgpt_key"] == "persona"
    )
    human_record = next(
        record for record in agent.store.iter_records(MEMGPT_CORE_LAYER) if record.metadata["memgpt_key"] == "human"
    )
    assert "Base persona." in persona_record.text
    assert "Likes concise output." in persona_record.text
    assert human_record.text == "The human prefers bulleted updates."


def test_memgpt_heartbeat_loop_runs_tool_then_followup_model_call() -> None:
    fake_runtime = FakeRuntime(
        responses=[
            {
                "assistant_message": "",
                "tool_calls": [
                    {
                        "name": "archival_memory_insert",
                        "arguments": {"memory": "The project codename is Aurora."},
                    }
                ],
            },
            {
                "assistant_message": "Stored the codename in archival memory.",
                "tool_calls": [],
            },
        ]
    )
    _use_fake_runtime(fake_runtime)
    agent = MemGPTAgent()

    final = agent.run_turn("Please store the project codename.")

    assert final == "Stored the codename in archival memory."
    recall_event_types = [record.metadata["memgpt"]["event_type"] for record in agent.store.iter_records(MEMGPT_RECALL_LAYER)]
    assert "tool_result" in recall_event_types
    assert agent.store.count(MEMGPT_ARCHIVAL_LAYER) == 1
    assert len(fake_runtime.chat_calls) == 2


def test_memgpt_main_runs_demo_path(capsys) -> None:
    fake_runtime = FakeRuntime(
        responses=[
            {
                "assistant_message": "",
                "tool_calls": [
                    {
                        "name": "archival_memory_insert",
                        "arguments": {"memory": "The user prefers concise status updates."},
                    }
                ],
            },
            {
                "assistant_message": "I saved the durable preference.",
                "tool_calls": [],
            },
        ]
    )
    _use_fake_runtime(fake_runtime)

    main()

    captured = capsys.readouterr()
    assert "I saved the durable preference." in captured.out
    assert "store counts:" in captured.out
