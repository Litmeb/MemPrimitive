"""Worktree runner, staged tests, and artifact feedback for evolution runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from ._codex import run_orchestrator, run_worker_codex
from ._process import CommandRunner, write_process_log
from ._types import (
    CandidateResult,
    CandidateSpec,
    CommandRecord,
    EvolutionRunConfig,
    append_jsonl,
    is_allowed_path,
    is_protected_path,
    normalize_repo_path,
    write_json,
)


ENV_FILES_TO_COPY = ("memprimitive/.env", "memprimitive/2.env")
WORKTREE_EXCLUDES = (
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "memprimitive/.env",
    "memprimitive/2.env",
    "*.env",
    "benchmarks/outputs/",
)

MEMMACHINE_CLASSICS_TEST_COMMAND = "~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v"


def python_command(config: EvolutionRunConfig) -> str:
    """Return a shell-safe Python command for the current host interpreter shape."""

    raw = str(config.python_bin).strip()
    if os.name != "nt":
        return raw
    expanded = os.path.expanduser(raw.replace("/", os.sep))
    path = Path(expanded)
    if path.is_file():
        ext = path.suffix.casefold()
        if ext in {".exe", ".bat", ".cmd"}:
            return json.dumps(str(path.resolve()))
        # Bash-only shims (e.g. ~/bin/winpy312) are often pre-expanded by the shell; cmd.exe cannot run them.
        return json.dumps(sys.executable)
    if raw.startswith("~/") or raw.startswith("~\\"):
        return json.dumps(sys.executable)
    return raw


def git_root(start: Path, runner: CommandRunner | None = None) -> Path:
    command_runner = runner or CommandRunner()
    result = command_runner.run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    if result.returncode != 0:
        raise RuntimeError("current directory is not inside a git repository.")
    return Path(result.stdout.strip())


def ensure_control_worktree_is_usable(repo_root: Path, config: EvolutionRunConfig, runner: CommandRunner) -> None:
    if config.allow_dirty_control_worktree:
        return
    result = runner.run(["git", "status", "--porcelain"], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed.")
    if result.stdout.strip():
        raise RuntimeError(
            "control worktree has uncommitted changes. Commit/stash them, or pass "
            "--allow-dirty-control-worktree with an explicit --base-ref to keep candidates based on a commit."
        )


def _run_git(repo_root: Path, args: list[str], runner: CommandRunner):
    return runner.run(["git", *args], cwd=repo_root)


def short_worktree_slug(value: str, *, prefix_chars: int = 32) -> str:
    normalized = normalize_repo_path(value).replace("/", "-") or "item"
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
    prefix = normalized[:prefix_chars].strip("-") or "item"
    return f"{prefix}-{digest}"


def create_candidate_worktree(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    candidate: CandidateSpec,
    round_index: int,
    runner: CommandRunner,
) -> Path:
    worktree_root = (repo_root / config.worktree_root).resolve()
    worktree_run_id = short_worktree_slug(config.run_id, prefix_chars=18)
    worktree_candidate_id = short_worktree_slug(candidate.id, prefix_chars=32)
    worktree_path = worktree_root / worktree_run_id / f"r{round_index}" / worktree_candidate_id
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    branch_name = f"evolve/{worktree_run_id}/r{round_index}/{worktree_candidate_id}"
    result = _run_git(
        repo_root,
        ["worktree", "add", "-b", branch_name, str(worktree_path), config.base_ref],
        runner,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to create worktree {worktree_path}")
    write_worktree_excludes(worktree_path, runner)
    return worktree_path


def merge_resume_base_ref_into_worktree(
    *,
    repo_root: Path,
    worktree_path: Path,
    artifact_dir: Path,
    base_ref: str,
    runner: CommandRunner,
) -> tuple[bool, list[CommandRecord]]:
    """Merge ``base_ref`` (resolved from the control repo) into a candidate worktree.

    Used when resuming benchmarks so updated harness code on ``base_ref`` is combined with
    the candidate branch; fails if Git reports merge conflicts.
    """

    resolve = runner.run(["git", "rev-parse", "--verify", base_ref], cwd=repo_root)
    records: list[CommandRecord] = [
        write_process_log(resolve, artifact_dir=artifact_dir, name="resume-merge-rev-parse")
    ]
    if resolve.returncode != 0:
        return False, records
    sha = resolve.stdout.strip()
    merge = runner.run(["git", "merge", "--no-edit", sha], cwd=worktree_path)
    records.append(write_process_log(merge, artifact_dir=artifact_dir, name="resume-merge-base-ref"))
    if merge.returncode != 0:
        return False, records
    return True, records


def rerun_benchmark_after_optional_resume_merge(
    *,
    repo_root: Path,
    worktree_path: Path,
    artifact_dir: Path,
    config: EvolutionRunConfig,
    candidate: CandidateSpec,
    runner: CommandRunner,
) -> tuple[bool, dict[str, Any], list[CommandRecord]]:
    prepend: list[CommandRecord] = []
    if config.resume_merge_base_ref:
        merge_ok, merge_records = merge_resume_base_ref_into_worktree(
            repo_root=repo_root,
            worktree_path=worktree_path,
            artifact_dir=artifact_dir,
            base_ref=config.base_ref,
            runner=runner,
        )
        prepend.extend(merge_records)
        if not merge_ok:
            return False, {"resume_merge_failed": True}, prepend
    bench_ok, diagnostics, bench_commands = run_benchmark_and_local_scoring(
        repo_root=repo_root,
        worktree_path=worktree_path,
        artifact_dir=artifact_dir,
        config=config,
        candidate=candidate,
        runner=runner,
    )
    return bench_ok, diagnostics, prepend + bench_commands


def write_worktree_excludes(worktree_path: Path, runner: CommandRunner) -> None:
    result = runner.run(["git", "rev-parse", "--git-path", "info/exclude"], cwd=worktree_path)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to locate worktree exclude file.")
    exclude_path = Path(result.stdout.strip())
    if not exclude_path.is_absolute():
        exclude_path = worktree_path / exclude_path
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8", errors="replace") if exclude_path.exists() else ""
    additions = [pattern for pattern in WORKTREE_EXCLUDES if pattern not in existing.splitlines()]
    if additions:
        with exclude_path.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write("\n".join(additions) + "\n")


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_runtime_env_files(repo_root: Path, worktree_path: Path) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for rel_path in ENV_FILES_TO_COPY:
        source = repo_root / rel_path
        target = worktree_path / rel_path
        if not source.exists():
            hashes[rel_path] = None
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[rel_path] = _file_sha256(target)
    return hashes


def detect_env_file_changes(worktree_path: Path, initial_hashes: dict[str, str | None]) -> list[str]:
    changed: list[str] = []
    for rel_path, before in initial_hashes.items():
        after = _file_sha256(worktree_path / rel_path)
        if before != after:
            changed.append(rel_path)
    return changed


def detect_ignored_protected_changes(worktree_path: Path, initial_env_hashes: dict[str, str | None]) -> list[str]:
    protected: list[str] = []
    known_env_paths = set(initial_env_hashes)
    env_candidates = set(worktree_path.rglob("*.env")) | set(worktree_path.rglob(".env"))
    for path in sorted(env_candidates):
        if not path.is_file():
            continue
        rel_path = path.relative_to(worktree_path).as_posix()
        if rel_path in known_env_paths and _file_sha256(path) == initial_env_hashes[rel_path]:
            continue
        protected.append(rel_path)

    outputs_dir = worktree_path / "benchmarks" / "outputs"
    if outputs_dir.exists():
        for path in sorted(outputs_dir.rglob("*")):
            if path.is_file():
                protected.append(path.relative_to(worktree_path).as_posix())
    return protected


def parse_changed_files(status_output: str) -> tuple[str, ...]:
    changed: list[str] = []
    for raw_line in status_output.splitlines():
        if not raw_line.strip():
            continue
        path_part = raw_line[3:] if len(raw_line) > 3 else raw_line
        if " -> " in path_part:
            path_part = path_part.rsplit(" -> ", 1)[-1]
        normalized = normalize_repo_path(path_part.strip())
        if normalized:
            changed.append(normalized)
    return tuple(dict.fromkeys(changed))


def is_generated_status_path(path: str | Path) -> bool:
    normalized = normalize_repo_path(path)
    return (
        "__pycache__/" in normalized
        or normalized.endswith(".pyc")
        or normalized.endswith(".pyo")
        or normalized.startswith(".pytest_cache/")
        or normalized.startswith(".ruff_cache/")
        or normalized.startswith(".mypy_cache/")
    )


def collect_changed_files(worktree_path: Path, runner: CommandRunner) -> tuple[str, ...]:
    result = runner.run(["git", "status", "--porcelain"], cwd=worktree_path)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed in candidate worktree.")
    return tuple(path for path in parse_changed_files(result.stdout) if not is_generated_status_path(path))


def materialize_diff(worktree_path: Path, artifact_dir: Path, runner: CommandRunner) -> CommandRecord:
    runner.run(["git", "add", "-N", "."], cwd=worktree_path)
    result = runner.run(["git", "diff", "--binary", "HEAD"], cwd=worktree_path)
    diff_path = artifact_dir / "diff.patch"
    diff_path.write_text(result.stdout, encoding="utf-8")
    return write_process_log(result, artifact_dir=artifact_dir, name="git-diff")


def validate_changed_files(candidate: CandidateSpec, changed_files: tuple[str, ...]) -> list[str]:
    reasons: list[str] = []
    for path in changed_files:
        if is_protected_path(path):
            reasons.append(f"protected path changed: {path}")
        elif not is_allowed_path(path, candidate.allowed_files):
            reasons.append(f"path outside allowed_files: {path}")
    return reasons


def run_shell_command(command: str, *, cwd: Path, artifact_dir: Path, name: str, runner: CommandRunner) -> CommandRecord:
    result = runner.run(command, cwd=cwd, shell=True)
    return write_process_log(result, artifact_dir=artifact_dir, name=name)


def command_for_host(command: str, config: EvolutionRunConfig) -> str:
    command = str(command)
    if os.name == "nt":
        return command.replace("~/bin/winpy312", python_command(config), 1)
    return command


def _focused_test_needs_memmachine_fallback(command: str, record: CommandRecord) -> bool:
    normalized = str(command).replace("\\", "/")
    if "tests/test_classics_memmachine.py" not in normalized:
        return False
    output_parts: list[str] = []
    if record.stdout_path:
        output_parts.append(Path(record.stdout_path).read_text(encoding="utf-8", errors="replace"))
    if record.stderr_path:
        output_parts.append(Path(record.stderr_path).read_text(encoding="utf-8", errors="replace"))
    output = "\n".join(output_parts).casefold()
    return record.returncode in {4, 5} and (
        "file or directory not found" in output
        or "no tests ran" in output
        or "0 selected" in output
    )


def run_static_checks(
    *,
    worktree_path: Path,
    artifact_dir: Path,
    changed_files: tuple[str, ...],
    config: EvolutionRunConfig,
    runner: CommandRunner,
) -> tuple[list[str], list[CommandRecord]]:
    reasons: list[str] = []
    commands: list[CommandRecord] = []

    diff_check = runner.run(["git", "diff", "--check"], cwd=worktree_path)
    commands.append(write_process_log(diff_check, artifact_dir=artifact_dir, name="git-diff-check"))
    if diff_check.returncode != 0:
        reasons.append("git diff --check failed")

    py_files = [path for path in changed_files if path.endswith(".py")]
    if py_files:
        quoted = " ".join(json.dumps(path) for path in py_files)
        compile_command = f"{python_command(config)} -m py_compile {quoted}"
        compile_record = run_shell_command(
            compile_command,
            cwd=worktree_path,
            artifact_dir=artifact_dir,
            name="py-compile",
            runner=runner,
        )
        commands.append(compile_record)
        if compile_record.returncode != 0:
            reasons.append("py_compile failed")

    return reasons, commands


def run_focused_tests(
    *,
    worktree_path: Path,
    artifact_dir: Path,
    candidate: CandidateSpec,
    config: EvolutionRunConfig,
    runner: CommandRunner,
) -> tuple[bool, list[CommandRecord]]:
    commands: list[CommandRecord] = []
    ok = True
    for index, command in enumerate(candidate.focused_tests, start=1):
        host_command = command_for_host(command, config)
        record = run_shell_command(
            host_command,
            cwd=worktree_path,
            artifact_dir=artifact_dir,
            name=f"focused-test-{index}",
            runner=runner,
        )
        commands.append(record)
        if record.returncode != 0:
            fallback_command = command_for_host(MEMMACHINE_CLASSICS_TEST_COMMAND, config)
            if host_command != fallback_command and _focused_test_needs_memmachine_fallback(command, record):
                fallback_record = run_shell_command(
                    fallback_command,
                    cwd=worktree_path,
                    artifact_dir=artifact_dir,
                    name=f"focused-test-{index}-fallback",
                    runner=runner,
                )
                commands.append(fallback_record)
                if fallback_record.returncode == 0:
                    continue
            ok = False
            break
    pytest_log = artifact_dir / "pytest.log"
    combined = []
    for record in commands:
        if record.stdout_path:
            combined.append(Path(record.stdout_path).read_text(encoding="utf-8", errors="replace"))
        if record.stderr_path:
            combined.append(Path(record.stderr_path).read_text(encoding="utf-8", errors="replace"))
    pytest_log.write_text("\n".join(combined), encoding="utf-8")
    return ok, commands


def _normalize_benchmark_arg_key(key: str) -> str:
    return str(key).strip().lstrip("-").replace("-", "_")


def _coerce_benchmark_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"none", "null"}:
            return None
    return value


def benchmark_command(
    config: EvolutionRunConfig,
    candidate: CandidateSpec,
    predictions_path: Path,
    *,
    repo_root: Path | None = None,
) -> str:
    binding_kwargs = dict(candidate.benchmark_args.get("memory_binding_kwargs") or {})
    benchmark_root = Path(config.benchmark_root)
    if repo_root is not None and not benchmark_root.is_absolute():
        benchmark_root = (repo_root / benchmark_root).resolve()
    known_cli_args = {
        "top_k",
        "similar_top_k",
        "mem0_speaker_workers",
        "memmachine_stm_record_budget",
        "memmachine_profile_max_turns",
        "max_workers",
        "llm_max_input_tokens",
    }
    reserved_args = {"memory_adapter", "memory_binding"}
    parts = [
        python_command(config),
        "-m memprimitive.benchmarking.minimal_baseline",
        f"--benchmark {config.benchmark}",
        f"--benchmark-root {json.dumps(str(benchmark_root))}",
        "--memory-adapter binding",
        f"--memory-binding {json.dumps(config.target_binding)}",
        "--no-progress",
        f"--output {json.dumps(str(predictions_path))}",
    ]
    if config.benchmark == "longmemeval":
        parts.append(f"--longmemeval-variant {json.dumps(config.longmemeval_variant)}")
    if config.locomo_users:
        parts.append("--locomo-users " + " ".join(json.dumps(user) for user in config.locomo_users))
    if config.benchmark_limit is not None:
        parts.append(f"--limit {int(config.benchmark_limit)}")
    if config.benchmark_top_k is not None:
        parts.append(f"--top-k {int(config.benchmark_top_k)}")
    if config.memmachine_stm_record_budget is not None:
        parts.append(f"--memmachine-stm-record-budget {int(config.memmachine_stm_record_budget)}")
    if config.memmachine_profile_max_turns is not None:
        parts.append(f"--memmachine-profile-max-turns {int(config.memmachine_profile_max_turns)}")
    if config.benchmark_max_workers is not None:
        parts.append(f"--max-workers {int(config.benchmark_max_workers)}")
    if config.max_history_turns is not None:
        parts.append(f"--max-history-turns {int(config.max_history_turns)}")
    if config.llm_max_input_tokens is not None:
        parts.append(f"--llm-max-input-tokens {int(config.llm_max_input_tokens)}")
    for raw_key, raw_value in sorted(candidate.benchmark_args.items()):
        key = _normalize_benchmark_arg_key(raw_key)
        value = _coerce_benchmark_arg_value(raw_value)
        if key == "memory_binding_kwargs":
            if isinstance(raw_value, dict):
                binding_kwargs.update(
                    {
                        str(item_key): _coerce_benchmark_arg_value(item_value)
                        for item_key, item_value in raw_value.items()
                    }
                )
            continue
        if key in reserved_args:
            continue
        if key not in known_cli_args:
            if key.startswith("memmachine_"):
                key = key.removeprefix("memmachine_")
            binding_kwargs[key] = value
            continue
        cli_key = str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(f"--{cli_key}")
        elif value is not None:
            parts.append(f"--{cli_key} {json.dumps(str(value))}")
    if binding_kwargs:
        parts.append(f"--memory-binding-kwargs {json.dumps(json.dumps(binding_kwargs, ensure_ascii=False))}")
    return " ".join(parts)


def run_benchmark_and_local_scoring(
    *,
    repo_root: Path,
    worktree_path: Path,
    artifact_dir: Path,
    config: EvolutionRunConfig,
    candidate: CandidateSpec,
    runner: CommandRunner,
) -> tuple[bool, dict[str, Any], list[CommandRecord]]:
    commands: list[CommandRecord] = []
    predictions_path = artifact_dir / "predictions.jsonl"
    metrics_path = artifact_dir / "metrics.json"
    summary_path = artifact_dir / "score_summary.json"

    benchmark_record = run_shell_command(
        benchmark_command(config, candidate, predictions_path, repo_root=repo_root),
        cwd=worktree_path,
        artifact_dir=artifact_dir,
        name="benchmark",
        runner=runner,
    )
    commands.append(benchmark_record)
    if benchmark_record.returncode != 0:
        return False, {}, commands

    eval_command = (
        f"{python_command(config)} -m memprimitive.benchmarking.evals "
        f"--input_file {json.dumps(str(predictions_path))} "
        f"--output_file {json.dumps(str(metrics_path))} "
        "--max_workers 4 --skip_llm_judge"
    )
    eval_record = run_shell_command(eval_command, cwd=worktree_path, artifact_dir=artifact_dir, name="eval-local", runner=runner)
    commands.append(eval_record)
    if eval_record.returncode != 0:
        return False, {}, commands

    summary_command = (
        f"{python_command(config)} -m memprimitive.benchmarking.generate_scores "
        f"--input_file {json.dumps(str(metrics_path))} "
        f"--output_file {json.dumps(str(summary_path))}"
    )
    summary_record = run_shell_command(
        summary_command,
        cwd=worktree_path,
        artifact_dir=artifact_dir,
        name="score-summary",
        runner=runner,
    )
    commands.append(summary_record)
    if summary_record.returncode != 0:
        return False, {}, commands

    diagnostics = summarize_candidate_artifacts(predictions_path=predictions_path, score_summary_path=summary_path)
    return True, diagnostics, commands


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                value = json.loads(raw_line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def summarize_candidate_artifacts(*, predictions_path: Path, score_summary_path: Path) -> dict[str, Any]:
    predictions = _load_jsonl_dicts(predictions_path)
    count = len(predictions)
    retrieved_texts = [str(item.get("retrieved_text", "")) for item in predictions]
    source_ids = [item.get("retrieved_source_ids", []) for item in predictions]
    source_id_counts = [len(value) if isinstance(value, list) else 0 for value in source_ids]
    diagnostics: dict[str, Any] = {
        "prediction_count": count,
        "empty_recall_count": sum(1 for text in retrieved_texts if not text.strip()),
        "empty_recall_rate": 0.0 if count == 0 else sum(1 for text in retrieved_texts if not text.strip()) / count,
        "source_id_coverage": 0.0 if count == 0 else sum(1 for item in source_id_counts if item > 0) / count,
        "avg_retrieved_chars": 0.0 if count == 0 else mean(len(text) for text in retrieved_texts),
        "avg_source_ids": 0.0 if count == 0 else mean(source_id_counts),
    }
    if score_summary_path.exists():
        summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict):
            diagnostics["score_summary"] = summary
            overall = summary.get("overall", {})
            if isinstance(overall, dict):
                diagnostics["overall_f1"] = overall.get("f1_score", 0.0)
                diagnostics["overall_bleu"] = overall.get("bleu_score", 0.0)
    return diagnostics


def run_expensive_llm_scoring(
    *,
    worktree_path: Path,
    artifact_dir: Path,
    config: EvolutionRunConfig,
    runner: CommandRunner,
) -> tuple[bool, dict[str, Any], list[CommandRecord]]:
    commands: list[CommandRecord] = []
    predictions_path = artifact_dir / "predictions.jsonl"
    metrics_path = artifact_dir / "metrics_llm.json"
    summary_path = artifact_dir / "score_summary_llm.json"
    eval_command = (
        f"{python_command(config)} -m memprimitive.benchmarking.evals "
        f"--input_file {json.dumps(str(predictions_path))} "
        f"--output_file {json.dumps(str(metrics_path))} "
        "--max_workers 4"
    )
    eval_record = run_shell_command(eval_command, cwd=worktree_path, artifact_dir=artifact_dir, name="eval-llm", runner=runner)
    commands.append(eval_record)
    if eval_record.returncode != 0:
        return False, {}, commands
    summary_command = (
        f"{python_command(config)} -m memprimitive.benchmarking.generate_scores "
        f"--input_file {json.dumps(str(metrics_path))} "
        f"--output_file {json.dumps(str(summary_path))}"
    )
    summary_record = run_shell_command(summary_command, cwd=worktree_path, artifact_dir=artifact_dir, name="score-summary-llm", runner=runner)
    commands.append(summary_record)
    if summary_record.returncode != 0:
        return False, {}, commands
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return True, {"llm_score_summary": summary}, commands


def run_candidate(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    round_index: int,
    candidate: CandidateSpec,
    round_artifact_dir: Path,
    runner: CommandRunner,
) -> CandidateResult:
    artifact_dir = round_artifact_dir / candidate.id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "candidate.json", candidate.to_json_dict())
    commands: list[CommandRecord] = []
    rejected_reasons: list[str] = []
    diagnostics: dict[str, Any] = {}
    artifact_paths = {
        "candidate": str(artifact_dir / "candidate.json"),
        "worker_events": str(artifact_dir / "worker.jsonl"),
        "worker_final": str(artifact_dir / "worker_final.md"),
        "diff": str(artifact_dir / "diff.patch"),
        "pytest_log": str(artifact_dir / "pytest.log"),
        "predictions": str(artifact_dir / "predictions.jsonl"),
        "metrics": str(artifact_dir / "metrics.json"),
        "score_summary": str(artifact_dir / "score_summary.json"),
    }

    worktree_path = create_candidate_worktree(
        repo_root=repo_root,
        config=config,
        candidate=candidate,
        round_index=round_index,
        runner=runner,
    )
    artifact_paths["worktree"] = str(worktree_path)
    env_hashes = copy_runtime_env_files(repo_root, worktree_path)

    worker_result = run_worker_codex(
        worktree_path=worktree_path,
        config=config,
        candidate=candidate,
        artifact_dir=artifact_dir,
        runner=runner,
    )
    commands.append(
        CommandRecord(
            command=worker_result.command_text,
            cwd=str(worker_result.cwd),
            returncode=worker_result.returncode,
            stdout_path=str(artifact_dir / "worker.stdout.log"),
            stderr_path=str(artifact_dir / "worker.stderr.log"),
            duration_seconds=round(worker_result.duration_seconds, 3),
        )
    )
    worker_final_path = artifact_dir / "worker_final.md"
    worker_final_message = worker_final_path.read_text(encoding="utf-8") if worker_final_path.exists() else ""
    if worker_result.returncode != 0:
        return CandidateResult(
            candidate_id=candidate.id,
            status="failed",
            failed_stage="worker",
            diagnostics={"worker_exit_code": worker_result.returncode},
            artifact_paths=artifact_paths,
            commands=tuple(commands),
            worker_final_message=worker_final_message,
        )

    commands.append(materialize_diff(worktree_path, artifact_dir, runner))
    changed_files = collect_changed_files(worktree_path, runner)
    env_changes = detect_env_file_changes(worktree_path, env_hashes)
    ignored_protected_changes = detect_ignored_protected_changes(worktree_path, env_hashes)
    rejected_reasons.extend(validate_changed_files(candidate, changed_files))
    rejected_reasons.extend(f"protected env file changed: {path}" for path in env_changes)
    rejected_reasons.extend(
        f"ignored protected path changed: {path}"
        for path in ignored_protected_changes
        if path not in set(env_changes)
    )

    static_reasons, static_commands = run_static_checks(
        worktree_path=worktree_path,
        artifact_dir=artifact_dir,
        changed_files=changed_files,
        config=config,
        runner=runner,
    )
    commands.extend(static_commands)
    rejected_reasons.extend(static_reasons)
    if rejected_reasons:
        return CandidateResult(
            candidate_id=candidate.id,
            status="rejected",
            changed_files=changed_files,
            rejected_reasons=tuple(rejected_reasons),
            failed_stage="static",
            artifact_paths=artifact_paths,
            commands=tuple(commands),
            worker_final_message=worker_final_message,
        )

    focused_ok, focused_commands = run_focused_tests(
        worktree_path=worktree_path,
        artifact_dir=artifact_dir,
        candidate=candidate,
        config=config,
        runner=runner,
    )
    commands.extend(focused_commands)
    if not focused_ok:
        return CandidateResult(
            candidate_id=candidate.id,
            status="failed",
            changed_files=changed_files,
            failed_stage="focused_tests",
            artifact_paths=artifact_paths,
            commands=tuple(commands),
            worker_final_message=worker_final_message,
        )

    benchmark_ok, benchmark_diagnostics, benchmark_commands = run_benchmark_and_local_scoring(
        repo_root=repo_root,
        worktree_path=worktree_path,
        artifact_dir=artifact_dir,
        config=config,
        candidate=candidate,
        runner=runner,
    )
    commands.extend(benchmark_commands)
    diagnostics.update(benchmark_diagnostics)
    if not benchmark_ok:
        return CandidateResult(
            candidate_id=candidate.id,
            status="failed",
            changed_files=changed_files,
            failed_stage="benchmark",
            diagnostics=diagnostics,
            artifact_paths=artifact_paths,
            commands=tuple(commands),
            worker_final_message=worker_final_message,
        )

    return CandidateResult(
        candidate_id=candidate.id,
        status="passed",
        changed_files=changed_files,
        diagnostics=diagnostics,
        artifact_paths=artifact_paths,
        commands=tuple(commands),
        worker_final_message=worker_final_message,
    )


def candidate_score(result: CandidateResult) -> tuple[float, float, float]:
    diagnostics = dict(result.diagnostics)
    return (
        float(diagnostics.get("overall_f1", 0.0) or 0.0),
        float(diagnostics.get("source_id_coverage", 0.0) or 0.0),
        -float(diagnostics.get("empty_recall_rate", 1.0) or 0.0),
    )


def build_leaderboard(results: list[CandidateResult]) -> list[dict[str, Any]]:
    rows = sorted(
        (result.to_json_dict() for result in results),
        key=lambda row: (
            row.get("status") == "passed",
            float(row.get("diagnostics", {}).get("overall_f1", 0.0) or 0.0),
            float(row.get("diagnostics", {}).get("source_id_coverage", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def build_orchestrator_feedback(
    *,
    round_index: int,
    results: list[CandidateResult],
    leaderboard: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = [result.candidate_id for result in results if result.status == "passed"]
    rejected = [result.candidate_id for result in results if result.status == "rejected"]
    failed = [result.candidate_id for result in results if result.status == "failed"]
    return {
        "round": round_index,
        "passed_candidates": passed,
        "rejected_candidates": rejected,
        "failed_candidates": failed,
        "leaderboard": leaderboard,
        "results": [result.to_json_dict() for result in results],
    }


def _load_candidate_specs(path: Path) -> list[CandidateSpec]:
    rows = _load_jsonl_dicts(path)
    return [CandidateSpec.from_json_dict(row) for row in rows]


def _load_round_results(path: Path) -> list[CandidateResult]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    results: list[CandidateResult] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        commands = tuple(
            CommandRecord(
                command=str(command.get("command", "")),
                cwd=str(command.get("cwd", "")),
                returncode=int(command.get("returncode", 0)),
                stdout_path=command.get("stdout_path"),
                stderr_path=command.get("stderr_path"),
                duration_seconds=float(command.get("duration_seconds", 0.0) or 0.0),
            )
            for command in row.get("commands", [])
            if isinstance(command, dict)
        )
        results.append(
            CandidateResult(
                candidate_id=str(row.get("candidate_id", "")),
                status=str(row.get("status", "")),
                changed_files=tuple(row.get("changed_files", []) or ()),
                rejected_reasons=tuple(row.get("rejected_reasons", []) or ()),
                failed_stage=row.get("failed_stage"),
                diagnostics=dict(row.get("diagnostics", {}) or {}),
                artifact_paths=dict(row.get("artifact_paths", {}) or {}),
                commands=commands,
                worker_final_message=str(row.get("worker_final_message", "")),
            )
        )
    return results


def promote_top_candidates(
    *,
    results: list[CandidateResult],
    config: EvolutionRunConfig,
    runner: CommandRunner,
) -> list[CandidateResult]:
    if config.promote_top_k <= 0:
        return results
    promoted_ids = {
        result.candidate_id
        for result in sorted(
            [item for item in results if item.status == "passed"],
            key=candidate_score,
            reverse=True,
        )[: config.promote_top_k]
    }
    updated: list[CandidateResult] = []
    for result in results:
        if result.candidate_id not in promoted_ids:
            updated.append(result)
            continue
        artifact_dir = Path(result.artifact_paths["candidate"]).parent
        worktree_path = Path(result.artifact_paths["worktree"])
        ok, diagnostics, commands = run_expensive_llm_scoring(
            worktree_path=worktree_path,
            artifact_dir=artifact_dir,
            config=config,
            runner=runner,
        )
        merged_diagnostics = {**dict(result.diagnostics), **diagnostics, "llm_scoring_passed": ok}
        updated.append(
            CandidateResult(
                candidate_id=result.candidate_id,
                status=result.status,
                changed_files=result.changed_files,
                rejected_reasons=result.rejected_reasons,
                failed_stage=result.failed_stage,
                diagnostics=merged_diagnostics,
                artifact_paths={
                    **dict(result.artifact_paths),
                    "llm_metrics": str(artifact_dir / "metrics_llm.json"),
                    "llm_score_summary": str(artifact_dir / "score_summary_llm.json"),
                },
                commands=tuple(list(result.commands) + commands),
                worker_final_message=result.worker_final_message,
            )
        )
    return updated


def resume_evolution_benchmark_only(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    runner: CommandRunner,
) -> dict[str, Any]:
    output_root = (repo_root / config.output_root / config.run_id).resolve()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"resume target not found: {manifest_path}")

    all_results: list[CandidateResult] = []
    for round_index in range(1, config.rounds + 1):
        round_dir = output_root / f"round-{round_index}"
        proposals_path = round_dir / "proposals.jsonl"
        leaderboard_path = round_dir / "leaderboard.json"
        if not proposals_path.exists():
            raise RuntimeError(f"missing candidate proposals for round {round_index}: {proposals_path}")
        if not leaderboard_path.exists():
            raise RuntimeError(f"missing round leaderboard for round {round_index}: {leaderboard_path}")

        candidates = {candidate.id: candidate for candidate in _load_candidate_specs(proposals_path)}
        existing_results = _load_round_results(leaderboard_path)
        resumed_results: list[CandidateResult] = []
        benchmark_retry_targets: list[tuple[int, CandidateResult, CandidateSpec, Path, Path]] = []
        for index, existing in enumerate(existing_results):
            candidate = candidates.get(existing.candidate_id)
            if candidate is None or existing.failed_stage != "benchmark":
                resumed_results.append(existing)
                continue
            worktree_value = existing.artifact_paths.get("worktree")
            if not worktree_value:
                raise RuntimeError(f"missing worktree path for candidate {existing.candidate_id}")
            benchmark_retry_targets.append(
                (
                    index,
                    existing,
                    candidate,
                    Path(worktree_value),
                    Path(existing.artifact_paths["candidate"]).parent,
                )
            )

        retried_results: dict[int, CandidateResult] = {}
        if benchmark_retry_targets:
            with ThreadPoolExecutor(max_workers=config.max_parallel_candidates) as executor:
                future_map = {
                    executor.submit(
                        rerun_benchmark_after_optional_resume_merge,
                        repo_root=repo_root,
                        worktree_path=worktree_path,
                        artifact_dir=artifact_dir,
                        config=config,
                        candidate=candidate,
                        runner=runner,
                    ): (index, existing)
                    for index, existing, candidate, worktree_path, artifact_dir in benchmark_retry_targets
                }
                for future in as_completed(future_map):
                    index, existing = future_map[future]
                    benchmark_ok, benchmark_diagnostics, benchmark_commands = future.result()
                    merged_diagnostics = {**dict(existing.diagnostics), **benchmark_diagnostics}
                    retried_results[index] = CandidateResult(
                        candidate_id=existing.candidate_id,
                        status="passed" if benchmark_ok else "failed",
                        changed_files=existing.changed_files,
                        rejected_reasons=existing.rejected_reasons,
                        failed_stage=None if benchmark_ok else "benchmark",
                        diagnostics=merged_diagnostics,
                        artifact_paths=existing.artifact_paths,
                        commands=tuple(list(existing.commands) + benchmark_commands),
                        worker_final_message=existing.worker_final_message,
                    )

        ordered_results: list[CandidateResult] = []
        for index, existing in enumerate(existing_results):
            ordered_results.append(retried_results.get(index, existing))
        resumed_results = ordered_results

        resumed_results = promote_top_candidates(results=resumed_results, config=config, runner=runner)
        leaderboard = build_leaderboard(resumed_results)
        write_json(round_dir / "leaderboard.json", leaderboard)
        feedback = build_orchestrator_feedback(
            round_index=round_index,
            results=resumed_results,
            leaderboard=leaderboard,
        )
        write_json(round_dir / "orchestrator_feedback.json", feedback)
        all_results.extend(resumed_results)

    final_leaderboard = build_leaderboard(all_results)
    write_json(output_root / "leaderboard.json", final_leaderboard)
    return {
        "run_id": config.run_id,
        "artifact_dir": str(output_root),
        "leaderboard": final_leaderboard,
        "resume_benchmark_only": True,
    }


def run_evolution_search(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    runner: CommandRunner | None = None,
    resume_benchmark_only: bool = False,
) -> dict[str, Any]:
    command_runner = runner or CommandRunner()
    ensure_control_worktree_is_usable(repo_root, config, command_runner)
    if resume_benchmark_only:
        return resume_evolution_benchmark_only(
            repo_root=repo_root,
            config=config,
            runner=command_runner,
        )
    output_root = (repo_root / config.output_root / config.run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config.to_json_dict(),
        "repo_root": str(repo_root),
    }
    write_json(output_root / "manifest.json", manifest)

    previous_feedback: dict[str, Any] | None = None
    all_results: list[CandidateResult] = []
    for round_index in range(1, config.rounds + 1):
        round_dir = output_root / f"round-{round_index}"
        candidates = run_orchestrator(
            repo_root=repo_root,
            config=config,
            round_index=round_index,
            previous_feedback=previous_feedback,
            artifact_dir=round_dir,
            runner=command_runner,
        )
        append_jsonl(round_dir / "proposals.jsonl", [candidate.to_json_dict() for candidate in candidates])

        round_results: list[CandidateResult] = []
        with ThreadPoolExecutor(max_workers=config.max_parallel_candidates) as executor:
            future_by_candidate = {
                executor.submit(
                    run_candidate,
                    repo_root=repo_root,
                    config=config,
                    round_index=round_index,
                    candidate=candidate,
                    round_artifact_dir=round_dir,
                    runner=command_runner,
                ): candidate
                for candidate in candidates
            }
            for future in as_completed(future_by_candidate):
                candidate = future_by_candidate[future]
                try:
                    round_results.append(future.result())
                except Exception as exc:  # pragma: no cover - defensive artifact path for unattended runs.
                    artifact_dir = round_dir / candidate.id
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    (artifact_dir / "runner_exception.txt").write_text(str(exc), encoding="utf-8")
                    round_results.append(
                        CandidateResult(
                            candidate_id=candidate.id,
                            status="failed",
                            failed_stage="runner",
                            artifact_paths={"runner_exception": str(artifact_dir / "runner_exception.txt")},
                        )
                    )

        round_results = promote_top_candidates(results=round_results, config=config, runner=command_runner)
        leaderboard = build_leaderboard(round_results)
        write_json(round_dir / "leaderboard.json", leaderboard)
        previous_feedback = build_orchestrator_feedback(
            round_index=round_index,
            results=round_results,
            leaderboard=leaderboard,
        )
        write_json(round_dir / "orchestrator_feedback.json", previous_feedback)
        all_results.extend(round_results)

    final_leaderboard = build_leaderboard(all_results)
    write_json(output_root / "leaderboard.json", final_leaderboard)
    return {
        "run_id": config.run_id,
        "artifact_dir": str(output_root),
        "leaderboard": final_leaderboard,
    }
