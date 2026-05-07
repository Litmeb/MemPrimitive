from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import Mock
from pathlib import Path

from memprimitive.evolution._codex import (
    _codex_failure_message,
    codex_exec_args,
    normalize_candidate_for_repo,
    run_orchestrator,
    run_worker_codex,
)
from memprimitive.evolution._process import CommandRunner, ProcessResult, resolve_executable_args
from memprimitive.evolution._runner import (
    benchmark_command,
    build_leaderboard,
    collect_changed_files,
    copy_runtime_env_files,
    create_candidate_worktree,
    detect_env_file_changes,
    detect_ignored_protected_changes,
    ensure_control_worktree_is_usable,
    parse_changed_files,
    python_command,
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


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "memprimitive").mkdir()
    (repo / "memprimitive" / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
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
    assert args[args.index("--model") + 1] == "gpt-5.4"
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

    assert args[args.index("--model") + 1] == "gpt-5.4-mini"
    assert "model_reasoning_effort" not in " ".join(args)


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


def test_python_command_resolves_wsl_wrapper_under_windows_python() -> None:
    config = EvolutionRunConfig(goal="test", rounds=1, candidates_per_round=1, target_binding="a:b")

    command = python_command(config)

    assert command
    if os.name == "nt":
        assert command != "~/bin/winpy312"
    else:
        assert command == "~/bin/winpy312"


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
        allowed_files=["tests/example/classics/test_memmachine_memory.py"],
        implementation_prompt="test",
        focused_tests=["~/bin/winpy312 -m pytest tests/example/classics/test_memmachine_memory.py -v"],
    )

    normalized = normalize_candidate_for_repo(tmp_path, candidate)

    assert normalized.allowed_files == ("tests/test_classics_memmachine.py",)
    assert normalized.focused_tests == ("~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v",)


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

    def _fail(*, repo_root, config):
        del repo_root, config
        raise RuntimeError("codex limit")

    monkeypatch.setattr("memprimitive.evolution.search.run_evolution_search", _fail)

    exit_code = search_main(["--goal", "test", "--rounds", "1"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"status": "failed", "error": "codex limit"}
