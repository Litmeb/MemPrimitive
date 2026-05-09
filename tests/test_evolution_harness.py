from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest
from pathlib import Path
from unittest.mock import Mock

from memprimitive.evolution import deepseek_responses_shim as _deepseek_shim
from memprimitive.evolution import _codex
from memprimitive.evolution._codex import (
    _codex_failure_message,
    codex_exec_args,
    normalize_candidate_for_repo,
    run_orchestrator,
    run_worker_codex,
)
from memprimitive.evolution._process import CommandRunner, ProcessResult, load_repo_env_defaults, resolve_executable_args
from memprimitive.evolution._runner import (
    benchmark_command,
    build_leaderboard,
    collect_changed_files,
    command_for_host,
    copy_runtime_env_files,
    create_candidate_worktree,
    detect_env_file_changes,
    detect_ignored_protected_changes,
    ensure_control_worktree_is_usable,
    merge_resume_base_ref_into_worktree,
    parse_changed_files,
    python_command,
    resume_evolution_benchmark_only,
    run_focused_tests,
    short_worktree_slug,
    summarize_candidate_artifacts,
    validate_changed_files,
)
from memprimitive.evolution._types import (
    CandidateResult,
    CandidateSpec,
    EvolutionRunConfig,
    append_jsonl,
)
from memprimitive.evolution.search import main as search_main


class FakeCodexRunner(CommandRunner):
    def __init__(self, final_message: str) -> None:
        self.final_message = final_message
        self.calls: list[tuple[str, ...] | str] = []

    def run(
        self,
        args,
        *,
        cwd: Path,
        input_text: str | None = None,
        shell: bool = False,
        timeout: int | None = None,
        env=None,
    ) -> ProcessResult:
        del input_text, shell, timeout, env
        normalized_args = tuple(str(part) for part in args) if not isinstance(args, str) else args
        self.calls.append(normalized_args)
        if isinstance(normalized_args, tuple) and "--output-last-message" in normalized_args:
            output_path = Path(normalized_args[normalized_args.index("--output-last-message") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self.final_message, encoding="utf-8")
        return ProcessResult(
            args=normalized_args,
            cwd=cwd,
            returncode=0,
            stdout='{"event":"done"}\n',
            stderr="",
            duration_seconds=0.01,
        )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True)


def _git_commit(repo: Path, message: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "memprimitive").mkdir()
    (repo / "memprimitive" / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git_commit(repo, "init")
    return repo


def test_candidate_schema_normalizes_and_serializes() -> None:
    spec = CandidateSpec(
        id="Direct Hit Readout V2!",
        hypothesis="Expose direct hits.",
        allowed_files=["./memprimitive/example/classics/memmachine_memory.py"],
        implementation_prompt="Add a small readout branch.",
        focused_tests=[],
    )

    assert spec.id == "direct-hit-readout-v2"
    assert spec.allowed_files == ("memprimitive/example/classics/memmachine_memory.py",)
    assert spec.focused_tests
    assert CandidateSpec.from_json_dict(spec.to_json_dict()).id == spec.id


def test_append_jsonl_writes_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    append_jsonl(path, [{"a": 1}, {"b": 2}])

    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [{"a": 1}, {"b": 2}]


def test_codex_exec_args_include_sandbox_and_output_file(tmp_path: Path) -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b", codex_bin="codex-test")
    output = tmp_path / "final.md"

    args = codex_exec_args(
        config=config,
        cwd=tmp_path,
        sandbox="read-only",
        output_last_message=output,
        role="orchestrator",
    )

    assert args[:4] == ["codex-test", "--ask-for-approval", "never", "exec"]
    assert "--sandbox" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--model") + 1] == "deepseek-v4-pro"
    assert args[args.index("--profile") + 1] == "deepseek"
    assert 'model_reasoning_effort="medium"' in args
    assert args[args.index("--output-last-message") + 1] == str(output)
    assert args[-1] == "-"


def test_codex_exec_args_use_worker_mini_model(tmp_path: Path) -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b", codex_bin="codex-test")

    args = codex_exec_args(
        config=config,
        cwd=tmp_path,
        sandbox="danger-full-access",
        output_last_message=tmp_path / "final.md",
        role="worker",
    )

    assert args[args.index("--model") + 1] == "deepseek-v4-flash"
    assert args[args.index("--profile") + 1] == "deepseek"
    assert "model_reasoning_effort" not in " ".join(args)


def test_codex_exec_args_routes_deepseek_defaults_through_wsl_codex(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MEMPRIMITIVE_EVOLUTION_USE_WSL_CODEX", raising=False)
    monkeypatch.setattr(_codex.os, "name", "nt")
    monkeypatch.setattr(_codex.shutil, "which", lambda name: "C:/Windows/System32/wsl.exe" if name == "wsl.exe" else None)
    monkeypatch.setattr(_codex, "_discover_wsl_codex_bin", lambda: "/usr/local/bin/codex")
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")

    args = _codex.codex_exec_args(
        config=config,
        cwd=tmp_path,
        sandbox="read-only",
        output_last_message=tmp_path / "final.md",
        role="orchestrator",
    )

    assert args[:3] == ["wsl.exe", "bash", "-lc"]
    assert "/usr/local/bin/codex" in args[-1]
    assert "--profile deepseek" in args[-1]


def test_codex_exec_args_allows_explicit_wsl_codex_opt_out(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMPRIMITIVE_EVOLUTION_USE_WSL_CODEX", "0")
    monkeypatch.setattr(_codex.os, "name", "nt")
    monkeypatch.setattr(_codex.shutil, "which", lambda name: "C:/Windows/System32/wsl.exe" if name == "wsl.exe" else None)
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")

    args = _codex.codex_exec_args(
        config=config,
        cwd=tmp_path,
        sandbox="read-only",
        output_last_message=tmp_path / "final.md",
        role="orchestrator",
    )

    assert args[0] == "codex"


def test_command_runner_resolves_windows_cmd_shims(monkeypatch) -> None:
    monkeypatch.setattr("memprimitive.evolution._process.shutil.which", lambda name: "C:/Tools/codex.CMD" if name == "codex" else None)

    args = resolve_executable_args(["codex", "--version"], shell=False)

    assert args == ["C:/Tools/codex.CMD", "--version"]


def test_command_runner_writes_stdin_as_utf8(monkeypatch, tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["tool"], returncode=0, stdout="ok", stderr="")
    run_mock = Mock(return_value=completed)
    monkeypatch.setattr("memprimitive.evolution._process.subprocess.run", run_mock)

    result = CommandRunner().run(["tool"], cwd=tmp_path, input_text="中文 prompt")

    assert result.returncode == 0
    assert run_mock.call_args.kwargs["encoding"] == "utf-8"
    assert run_mock.call_args.kwargs["errors"] == "replace"


def test_repo_env_defaults_feed_subprocess_env_without_overrides(tmp_path: Path) -> None:
    (tmp_path / "memprimitive").mkdir()
    (tmp_path / "memprimitive" / ".env").write_text("DEEPSEEK_API_KEY=from-file\nEXISTING=from-file\n", encoding="utf-8")
    env = {"EXISTING": "from-env"}

    loaded_keys = load_repo_env_defaults(tmp_path, env)

    assert loaded_keys == ("DEEPSEEK_API_KEY", "EXISTING")
    assert env["DEEPSEEK_API_KEY"] == "from-file"
    assert env["EXISTING"] == "from-env"


def test_command_runner_marks_repo_env_for_wsl_inheritance(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "memprimitive").mkdir()
    (tmp_path / "memprimitive" / ".env").write_text("MEMPRIMITIVE_TEST_KEY=from-file\n", encoding="utf-8")
    completed = subprocess.CompletedProcess(args=["tool"], returncode=0, stdout="", stderr="")
    run_mock = Mock(return_value=completed)
    monkeypatch.setattr("memprimitive.evolution._process.subprocess.run", run_mock)

    CommandRunner().run(["tool"], cwd=tmp_path)

    env = run_mock.call_args.kwargs["env"]
    assert env["MEMPRIMITIVE_TEST_KEY"] == "from-file"
    assert "MEMPRIMITIVE_TEST_KEY/u" in env["WSLENV"].split(":")


def test_deepseek_shim_builds_chat_payload_from_responses_shape() -> None:
    payload = {
        "model": "deepseek-v4-pro",
        "instructions": "system instructions",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "developer note"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ],
        "tools": [{"type": "function", "name": "exec_command", "description": "run", "parameters": {"type": "object"}}],
        "tool_choice": "auto",
    }

    chat = _deepseek_shim._build_chat_payload(payload)

    assert chat["model"] == "deepseek-v4-pro"
    assert [message["role"] for message in chat["messages"]] == ["system", "system", "user"]
    assert chat["messages"][-1]["content"] == "hello"
    assert chat["tools"][0]["function"]["name"] == "exec_command"


def test_deepseek_shim_accepts_string_input() -> None:
    chat = _deepseek_shim._build_chat_payload(
        {
            "model": "deepseek-v4-pro",
            "input": "hello from string input",
        }
    )

    assert chat["messages"] == [{"role": "user", "content": "hello from string input"}]


def test_deepseek_shim_maps_chat_tool_call_to_responses_output() -> None:
    response = _deepseek_shim._chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"},
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
        "deepseek-v4-flash",
    )

    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["call_id"] == "call_1"
    assert response["output"][0]["name"] == "exec_command"
    assert response["usage"]["total_tokens"] == 7


def test_deepseek_shim_lists_models() -> None:
    handler = _deepseek_shim.ShimHandler.__new__(_deepseek_shim.ShimHandler)
    sent: dict[str, object] = {}

    def _capture(status: int, payload: dict[str, object]) -> None:
        sent["status"] = status
        sent["payload"] = payload

    handler.path = "/v1/models?client_version=0.130.0"
    handler._send_json = _capture  # type: ignore[method-assign]

    _deepseek_shim.ShimHandler.do_GET(handler)

    assert sent["status"] == 200
    payload = sent["payload"]
    assert isinstance(payload, dict)
    assert payload["object"] == "list"
    assert [item["id"] for item in payload["data"]] == ["deepseek-v4-pro", "deepseek-v4-flash"]


def test_deepseek_shim_groups_parallel_responses_tool_calls_for_chat() -> None:
    messages = _deepseek_shim._responses_input_to_chat_messages(
        {
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"},
                {"type": "function_call", "call_id": "call_2", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "repo"},
                {"type": "function_call_output", "call_id": "call_2", "output": "files"},
            ]
        }
    )

    assert len(messages) == 3
    assert messages[0]["role"] == "assistant"
    assert [tool_call["id"] for tool_call in messages[0]["tool_calls"]] == ["call_1", "call_2"]
    assert [message["role"] for message in messages[1:]] == ["tool", "tool"]


def test_deepseek_shim_merges_assistant_text_between_tool_call_and_output() -> None:
    messages = _deepseek_shim._responses_input_to_chat_messages(
        {
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I will inspect the repo."}]},
                {"type": "function_call_output", "call_id": "call_1", "output": "repo"},
            ]
        }
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "I will inspect the repo."
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["role"] == "tool"


def test_python_command_resolves_wsl_wrapper_under_windows_python() -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")

    command = python_command(config)

    assert command
    if os.name == "nt":
        assert command != "~/bin/winpy312"
        exe = json.loads(command)
        assert Path(exe).suffix.casefold() in {".exe", ".bat", ".cmd"}
    else:
        assert command == "~/bin/winpy312"


def test_python_command_falls_back_for_preexpanded_non_exe_shim(tmp_path: Path) -> None:
    shim = tmp_path / "bin" / "winpy312"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/bash\n", encoding="utf-8")
    config = EvolutionRunConfig(
        goal="test",
        rounds=1,
        candidates_per_round=1,
        target_binding="a:b",
        python_bin=str(shim),
    )
    command = python_command(config)
    if os.name == "nt":
        assert command == json.dumps(sys.executable)
    else:
        assert command == str(shim)


def test_command_for_host_rewrites_wsl_wrapper_under_windows_python() -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")

    command = command_for_host("~/bin/winpy312 -m pytest tests -v", config)

    if os.name == "nt":
        assert command == f"{python_command(config)} -m pytest tests -v"
    else:
        assert command == "~/bin/winpy312 -m pytest tests -v"


def test_run_orchestrator_uses_fake_codex_and_parses_candidates(tmp_path: Path) -> None:
    (tmp_path / "PROJECT_PROGRESS.md").write_text("progress", encoding="utf-8")
    (tmp_path / "memprimitive" / "example" / "classics").mkdir(parents=True)
    (tmp_path / "memprimitive" / "example" / "classics" / "memmachine_memory.py").write_text("# binding\n", encoding="utf-8")
    config = EvolutionRunConfig(
        goal="improve recall",
        rounds=1,
        candidates_per_round=1,
        target_binding="memprimitive.example.classics.memmachine_memory:create_memory_binding",
        codex_bin="codex-test",
    )
    runner = FakeCodexRunner(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "candidate-one",
                        "hypothesis": "try one local change",
                        "allowed_files": ["memprimitive/example/classics/memmachine_memory.py"],
                        "implementation_prompt": "change one thing",
                    }
                ]
            }
        )
    )

    candidates = run_orchestrator(
        repo_root=tmp_path,
        config=config,
        round_index=1,
        previous_feedback=None,
        artifact_dir=tmp_path / "artifacts",
        runner=runner,
    )

    assert candidates[0].id == "candidate-one"
    command = runner.calls[0]
    assert isinstance(command, tuple)
    assert command[command.index("--sandbox") + 1] == "read-only"


def test_candidate_normalization_rewrites_known_memmachine_test_path(tmp_path: Path) -> None:
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=[
            "tests/example/classics/test_memmachine_memory.py",
            "tests/classics/test_classics_memmachine_memory.py",
            "tests/test_classics_memmachine_memory.py",
            "tests/test_classics_memmachine_binding.py",
        ],
        implementation_prompt="test",
        focused_tests=[
            "~/bin/winpy312 -m pytest tests/example/classics/test_memmachine_memory.py -v",
            "~/bin/winpy312 -m pytest tests/classics/test_classics_memmachine_memory.py -v",
            "~/bin/winpy312 -m pytest tests/test_classics_memmachine_memory.py -v",
            "~/bin/winpy312 -m pytest tests/test_classics_memmachine_binding.py -v",
        ],
    )

    normalized = normalize_candidate_for_repo(tmp_path, candidate)

    assert normalized.allowed_files == ("tests/test_classics_memmachine.py",)
    assert normalized.focused_tests == (
        "~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v",
        "~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v",
        "~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v",
        "~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v",
    )


def test_run_focused_tests_falls_back_for_empty_memmachine_selector(tmp_path: Path) -> None:
    class FocusedRunner(CommandRunner):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(
            self,
            args,
            *,
            cwd: Path,
            input_text: str | None = None,
            shell: bool = False,
            timeout: int | None = None,
            env=None,
        ) -> ProcessResult:
            del input_text, shell, timeout, env
            command = str(args)
            self.calls.append(command)
            if len(self.calls) == 1:
                return ProcessResult(
                    args=command,
                    cwd=cwd,
                    returncode=5,
                    stdout="collected 4 items / 4 deselected / 0 selected\n",
                    stderr="",
                    duration_seconds=0.01,
                )
            return ProcessResult(
                args=command,
                cwd=cwd,
                returncode=0,
                stdout="4 passed\n",
                stderr="",
                duration_seconds=0.01,
            )

    candidate = normalize_candidate_for_repo(
        tmp_path,
        CandidateSpec(
            id="candidate",
            hypothesis="test",
            allowed_files=["memprimitive/example/classics/memmachine_memory.py"],
            implementation_prompt="test",
            focused_tests=[
                "~/bin/winpy312 -m pytest tests/test_classics_memmachine_memory.py -v -k 'rerank or cluster'"
            ],
        ),
    )
    runner = FocusedRunner()

    ok, commands = run_focused_tests(
        worktree_path=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        candidate=candidate,
        config=EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b"),
        runner=runner,
    )

    assert ok
    assert len(commands) == 2
    assert "tests/test_classics_memmachine.py -v -k 'rerank or cluster'" in runner.calls[0]
    assert runner.calls[1].endswith("-m pytest tests/test_classics_memmachine.py -v")


def test_codex_failure_message_extracts_jsonl_error(tmp_path: Path) -> None:
    result = ProcessResult(
        args=("codex", "exec"),
        cwd=tmp_path,
        returncode=1,
        stdout=json.dumps({"type": "error", "message": "usage limit hit"}) + "\n",
        stderr="wsl noisy stderr",
        duration_seconds=0.1,
    )

    message = _codex_failure_message(
        result,
        events_path=tmp_path / "events.jsonl",
        final_path=tmp_path / "final.md",
    )

    assert "usage limit hit" in message
    assert "events_path" in message


def test_run_worker_codex_uses_danger_full_access(tmp_path: Path) -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b", codex_bin="codex-test")
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=["memprimitive/example/classics/memmachine_memory.py"],
        implementation_prompt="edit only the file",
    )
    runner = FakeCodexRunner("done")

    result = run_worker_codex(
        worktree_path=tmp_path,
        config=config,
        candidate=candidate,
        artifact_dir=tmp_path / "artifacts",
        runner=runner,
    )

    assert result.returncode == 0
    command = runner.calls[0]
    assert isinstance(command, tuple)
    assert command[command.index("--sandbox") + 1] == "danger-full-access"


def test_worktree_creation_and_changed_file_whitelist(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    config = EvolutionRunConfig(
        goal="test",
        rounds=1,
        candidates_per_round=1,
        target_binding="a:b",
        worktree_root=str(tmp_path / "worktrees"),
        run_id="run-one",
    )
    candidate = CandidateSpec(
        id="allowed-change",
        hypothesis="change allowed file",
        allowed_files=["memprimitive/target.py"],
        implementation_prompt="change value",
    )

    worktree = create_candidate_worktree(
        repo_root=repo,
        config=config,
        candidate=candidate,
        round_index=1,
        runner=CommandRunner(),
    )
    (worktree / "memprimitive" / "target.py").write_text("VALUE = 2\n", encoding="utf-8")

    changed = collect_changed_files(worktree, CommandRunner())

    assert changed == ("memprimitive/target.py",)
    assert validate_changed_files(candidate, changed) == []


def test_short_worktree_slug_keeps_generated_paths_bounded() -> None:
    slug = short_worktree_slug("very-long-run-id-" + ("x" * 120), prefix_chars=18)

    assert slug.startswith("very-long-run-id-")
    assert len(slug) <= 29


def test_benchmark_command_maps_unknown_candidate_args_to_binding_kwargs(tmp_path: Path) -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=["memprimitive/target.py"],
        implementation_prompt="test",
        benchmark_args={"sentence_top_k": 80, "memory_binding_kwargs": {"expand_context": 3}},
    )

    command = benchmark_command(config, candidate, tmp_path / "predictions.jsonl")

    assert "--sentence-top-k" not in command
    assert "--memory-binding-kwargs" in command
    assert "sentence_top_k" in command
    assert "expand_context" in command


def test_benchmark_command_normalizes_cli_style_candidate_args(tmp_path: Path) -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=["memprimitive/target.py"],
        implementation_prompt="test",
        benchmark_args={
            "--memory-adapter": "binding",
            "--memory-binding": "ignored.module:create_memory_binding",
            "--memmachine-rerank": "true",
        },
    )

    command = benchmark_command(config, candidate, tmp_path / "predictions.jsonl")

    assert "--memory-adapter binding" in command
    assert "ignored.module:create_memory_binding" not in command
    assert '\\"rerank\\": true' in command
    assert "memmachine_rerank" not in command


def test_benchmark_command_uses_control_repo_benchmark_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=["memprimitive/target.py"],
        implementation_prompt="test",
    )

    command = benchmark_command(config, candidate, tmp_path / "predictions.jsonl", repo_root=repo_root)

    assert "--smoke-test" not in command
    assert f"--benchmark-root {json.dumps(str((repo_root / 'benchmarks').resolve()))}" in command


def test_benchmark_command_includes_search_default_memmachine_benchmark_settings(tmp_path: Path) -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=["memprimitive/target.py"],
        implementation_prompt="test",
    )

    command = benchmark_command(config, candidate, tmp_path / "predictions.jsonl")

    assert "--top-k 10" in command
    assert "--memmachine-stm-record-budget 20" in command
    assert "--memmachine-profile-max-turns 24" in command
    assert "--max-workers 10" in command
    assert "--llm-max-input-tokens 7000" in command


def test_control_worktree_dirty_guard_requires_explicit_opt_in(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "memprimitive" / "target.py").write_text("VALUE = 99\n", encoding="utf-8")
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")

    try:
        ensure_control_worktree_is_usable(repo, config, CommandRunner())
    except RuntimeError as exc:
        assert "control worktree has uncommitted changes" in str(exc)
    else:  # pragma: no cover - the guard must fail for this test repo.
        raise AssertionError("dirty guard did not reject a dirty control worktree")

    config.allow_dirty_control_worktree = True
    ensure_control_worktree_is_usable(repo, config, CommandRunner())


def test_changed_file_guard_rejects_disallowed_and_protected_paths() -> None:
    candidate = CandidateSpec(
        id="candidate",
        hypothesis="test",
        allowed_files=["memprimitive/target.py"],
        implementation_prompt="test",
    )

    reasons = validate_changed_files(
        candidate,
        ("memprimitive/target.py", "README.md", "memprimitive/.env", "benchmarks/outputs/x.jsonl"),
    )

    assert "path outside allowed_files: README.md" in reasons
    assert "protected path changed: memprimitive/.env" in reasons
    assert "protected path changed: benchmarks/outputs/x.jsonl" in reasons


def test_parse_changed_files_handles_renames_and_untracked() -> None:
    status = " M memprimitive/a.py\n?? memprimitive/new.py\nR  old.py -> memprimitive/renamed.py\n"

    assert parse_changed_files(status) == (
        "memprimitive/a.py",
        "memprimitive/new.py",
        "memprimitive/renamed.py",
    )


def test_env_copy_hash_detects_runtime_env_mutation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    (repo / "memprimitive").mkdir(parents=True)
    (worktree / "memprimitive").mkdir(parents=True)
    (repo / "memprimitive" / ".env").write_text("TOKEN=one\n", encoding="utf-8")

    hashes = copy_runtime_env_files(repo, worktree)
    (worktree / "memprimitive" / ".env").write_text("TOKEN=two\n", encoding="utf-8")

    assert detect_env_file_changes(worktree, hashes) == ["memprimitive/.env"]


def test_ignored_protected_changes_detects_outputs_and_new_env(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    (worktree / "benchmarks" / "outputs").mkdir(parents=True)
    (worktree / "benchmarks" / "outputs" / "bad.jsonl").write_text("{}", encoding="utf-8")
    (worktree / "memprimitive").mkdir()
    (worktree / "memprimitive" / "new.env").write_text("TOKEN=bad\n", encoding="utf-8")

    protected = detect_ignored_protected_changes(worktree, {})

    assert "benchmarks/outputs/bad.jsonl" in protected
    assert "memprimitive/new.env" in protected


def test_artifact_summary_and_leaderboard(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"retrieved_text": "Alice likes tea.", "retrieved_source_ids": ["r1"]}),
                json.dumps({"retrieved_text": "", "retrieved_source_ids": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    score_summary = tmp_path / "score_summary.json"
    score_summary.write_text(json.dumps({"overall": {"f1_score": 0.5, "bleu_score": 0.25}}), encoding="utf-8")

    diagnostics = summarize_candidate_artifacts(predictions_path=predictions, score_summary_path=score_summary)
    leaderboard = build_leaderboard(
        [
            CandidateResult(candidate_id="bad", status="failed", diagnostics={"overall_f1": 1.0}),
            CandidateResult(candidate_id="good", status="passed", diagnostics=diagnostics),
        ]
    )

    assert diagnostics["prediction_count"] == 2
    assert diagnostics["empty_recall_rate"] == 0.5
    assert diagnostics["source_id_coverage"] == 0.5
    assert leaderboard[0]["candidate_id"] == "good"


def test_search_cli_dry_run_outputs_preview(capsys) -> None:
    exit_code = search_main(["--goal", "test", "--rounds", "1", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["config"]["goal"] == "test"
    assert "PROJECT_PROGRESS.md" in payload["context_files"]


def test_search_cli_reports_runtime_error_without_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr("memprimitive.evolution.search.git_root", lambda cwd: cwd)

    def _fail(*, repo_root, config, runner=None, resume_benchmark_only: bool = False):
        del repo_root, config, runner, resume_benchmark_only
        raise RuntimeError("codex limit")

    monkeypatch.setattr("memprimitive.evolution.search.run_evolution_search", _fail)

    exit_code = search_main(["--goal", "test", "--rounds", "1"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"status": "failed", "error": "codex limit"}


def test_search_cli_forwards_resume_benchmark_only(monkeypatch, capsys) -> None:
    monkeypatch.setattr("memprimitive.evolution.search.git_root", lambda cwd: cwd)
    seen: dict[str, object] = {}

    def _run(*, repo_root, config, runner=None, resume_benchmark_only: bool = False):
        del runner
        seen["repo_root"] = repo_root
        seen["config"] = config
        seen["resume_benchmark_only"] = resume_benchmark_only
        return {"run_id": config.run_id, "artifact_dir": str(repo_root / "artifacts"), "leaderboard": []}

    monkeypatch.setattr("memprimitive.evolution.search.run_evolution_search", _run)

    exit_code = search_main(
        [
            "--goal",
            "test",
            "--rounds",
            "1",
            "--run-id",
            "resume-me",
            "--resume-benchmark-only",
        ]
    )

    assert exit_code == 0
    assert seen["resume_benchmark_only"] is True
    assert seen["config"].run_id == "resume-me"
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "resume-me"


def test_resume_merge_base_ref_requires_resume_benchmark_only() -> None:
    with pytest.raises(SystemExit) as exc_info:
        search_main(["--goal", "test", "--rounds", "1", "--resume-merge-base-ref"])
    assert exc_info.value.code != 0


def test_merge_resume_base_ref_into_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "memprimitive" / "harness.py").write_text("H = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git_commit(repo, "add harness")
    root_sha = subprocess.check_output(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo, text=True).strip()
    wt = tmp_path / "cand_wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", "evolve/test-cand", str(wt), root_sha],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    (wt / "memprimitive" / "target.py").write_text("VALUE = 99\n", encoding="utf-8")
    _git(wt, "add", ".")
    _git_commit(wt, "candidate")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    runner = CommandRunner()
    ok, records = merge_resume_base_ref_into_worktree(
        repo_root=repo,
        worktree_path=wt,
        artifact_dir=artifact_dir,
        base_ref="HEAD",
        runner=runner,
    )
    assert ok
    assert len(records) == 2
    assert (wt / "memprimitive" / "harness.py").read_text(encoding="utf-8") == "H = 1\n"
    assert "99" in (wt / "memprimitive" / "target.py").read_text(encoding="utf-8")


def test_resume_evolution_benchmark_only_reruns_only_benchmark_failures(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    output_root = repo / "benchmarks" / "outputs" / "evolve" / "resume-me"
    round_dir = output_root / "round-1"
    round_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text("{}", encoding="utf-8")
    artifact_dir = round_dir / "candidate-one"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "candidate.json").write_text("{}", encoding="utf-8")
    worktree = repo / "existing-worktree"
    worktree.mkdir()

    append_jsonl(
        round_dir / "proposals.jsonl",
        [
            CandidateSpec(
                id="candidate-one",
                hypothesis="test",
                allowed_files=["memprimitive/target.py"],
                implementation_prompt="test",
            ).to_json_dict()
        ],
    )
    (round_dir / "leaderboard.json").write_text(
        json.dumps(
            [
                CandidateResult(
                    candidate_id="candidate-one",
                    status="failed",
                    changed_files=("memprimitive/target.py",),
                    failed_stage="benchmark",
                    artifact_paths={
                        "candidate": str(artifact_dir / "candidate.json"),
                        "worktree": str(worktree),
                    },
                ).to_json_dict()
            ]
        ),
        encoding="utf-8",
    )

    class ResumeRunner(CommandRunner):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(
            self,
            args,
            *,
            cwd: Path,
            input_text: str | None = None,
            shell: bool = False,
            timeout: int | None = None,
            env=None,
        ) -> ProcessResult:
            del input_text, timeout, env
            command = str(args)
            self.calls.append(command)
            assert shell
            if "minimal_baseline" in command:
                (artifact_dir / "predictions.jsonl").write_text(
                    json.dumps({"retrieved_text": "evidence", "retrieved_source_ids": ["s1"]}) + "\n",
                    encoding="utf-8",
                )
            elif "generate_scores" in command:
                (artifact_dir / "score_summary.json").write_text(
                    json.dumps({"overall": {"f1_score": 0.5, "bleu_score": 0.25}}),
                    encoding="utf-8",
                )
            return ProcessResult(args=args, cwd=cwd, returncode=0, stdout="", stderr="", duration_seconds=0.01)

    result = resume_evolution_benchmark_only(
        repo_root=repo,
        config=EvolutionRunConfig(
            goal="test",
            rounds=1,
            candidates_per_round=1,
            target_binding="a:b",
            run_id="resume-me",
        ),
        runner=ResumeRunner(),
    )

    updated = json.loads((round_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert result["resume_benchmark_only"] is True
    assert updated[0]["status"] == "passed"
    assert updated[0]["failed_stage"] is None
    assert updated[0]["diagnostics"]["overall_f1"] == 0.5


def test_resume_evolution_benchmark_only_retries_benchmark_failures_in_parallel(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    output_root = repo / "benchmarks" / "outputs" / "evolve" / "resume-parallel"
    round_dir = output_root / "round-1"
    round_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text("{}", encoding="utf-8")
    worktree = repo / "existing-worktree"
    worktree.mkdir()

    candidates = []
    leaderboard_rows = []
    for candidate_id in ("candidate-one", "candidate-two"):
        artifact_dir = round_dir / candidate_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "candidate.json").write_text("{}", encoding="utf-8")
        candidates.append(
            CandidateSpec(
                id=candidate_id,
                hypothesis="test",
                allowed_files=["memprimitive/target.py"],
                implementation_prompt="test",
            ).to_json_dict()
        )
        leaderboard_rows.append(
            CandidateResult(
                candidate_id=candidate_id,
                status="failed",
                changed_files=("memprimitive/target.py",),
                failed_stage="benchmark",
                artifact_paths={
                    "candidate": str(artifact_dir / "candidate.json"),
                    "worktree": str(worktree),
                },
            ).to_json_dict()
        )
    append_jsonl(round_dir / "proposals.jsonl", candidates)
    (round_dir / "leaderboard.json").write_text(json.dumps(leaderboard_rows), encoding="utf-8")

    class ParallelResumeRunner(CommandRunner):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.lock = threading.Lock()
            self.active_benchmarks = 0
            self.max_active_benchmarks = 0

        def run(
            self,
            args,
            *,
            cwd: Path,
            input_text: str | None = None,
            shell: bool = False,
            timeout: int | None = None,
            env=None,
        ) -> ProcessResult:
            del input_text, timeout, env
            command = str(args)
            self.calls.append(command)
            assert shell
            artifact_dir = Path(cwd)
            if "minimal_baseline" in command:
                with self.lock:
                    self.active_benchmarks += 1
                    self.max_active_benchmarks = max(self.max_active_benchmarks, self.active_benchmarks)
                time.sleep(0.1)
                (artifact_dir / "predictions.jsonl").write_text(
                    json.dumps({"retrieved_text": "evidence", "retrieved_source_ids": ["s1"]}) + "\n",
                    encoding="utf-8",
                )
                with self.lock:
                    self.active_benchmarks -= 1
            elif "generate_scores" in command:
                (artifact_dir / "score_summary.json").write_text(
                    json.dumps({"overall": {"f1_score": 0.5, "bleu_score": 0.25}}),
                    encoding="utf-8",
                )
            return ProcessResult(args=args, cwd=cwd, returncode=0, stdout="", stderr="", duration_seconds=0.01)

    runner = ParallelResumeRunner()
    result = resume_evolution_benchmark_only(
        repo_root=repo,
        config=EvolutionRunConfig(
            goal="test",
            rounds=1,
            candidates_per_round=2,
            target_binding="a:b",
            run_id="resume-parallel",
            max_parallel_candidates=2,
        ),
        runner=runner,
    )

    updated = json.loads((round_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert result["resume_benchmark_only"] is True
    assert all(row["status"] == "passed" for row in updated)
    assert runner.max_active_benchmarks >= 2
