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


def python_command(config: EvolutionRunConfig) -> str:
    """Return a shell-safe Python command for the current host interpreter shape."""

    if os.name == "nt" and str(config.python_bin).startswith("~/"):
        return json.dumps(sys.executable)
    return str(config.python_bin)


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


def create_candidate_worktree(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    candidate: CandidateSpec,
    round_index: int,
    runner: CommandRunner,
) -> Path:
    worktree_root = (repo_root / config.worktree_root).resolve()
    worktree_path = worktree_root / config.run_id / f"round-{round_index}" / candidate.id
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    branch_name = f"evolve/{config.run_id}/r{round_index}/{candidate.id}"
    result = _run_git(
        repo_root,
        ["worktree", "add", "-b", branch_name, str(worktree_path), config.base_ref],
        runner,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to create worktree {worktree_path}")
    return worktree_path


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


def collect_changed_files(worktree_path: Path, runner: CommandRunner) -> tuple[str, ...]:
    result = runner.run(["git", "status", "--porcelain"], cwd=worktree_path)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed in candidate worktree.")
    return parse_changed_files(result.stdout)


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
    runner: CommandRunner,
) -> tuple[bool, list[CommandRecord]]:
    commands: list[CommandRecord] = []
    ok = True
    for index, command in enumerate(candidate.focused_tests, start=1):
        record = run_shell_command(
            command,
            cwd=worktree_path,
            artifact_dir=artifact_dir,
            name=f"focused-test-{index}",
            runner=runner,
        )
        commands.append(record)
        if record.returncode != 0:
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


def benchmark_command(config: EvolutionRunConfig, candidate: CandidateSpec, predictions_path: Path) -> str:
    parts = [
        python_command(config),
        "-m memprimitive.benchmarking.minimal_baseline",
        f"--benchmark {config.benchmark}",
        f"--benchmark-root {json.dumps(config.benchmark_root)}",
        "--memory-adapter binding",
        f"--memory-binding {json.dumps(config.target_binding)}",
        "--smoke-test",
        "--no-progress",
        f"--output {json.dumps(str(predictions_path))}",
    ]
    if config.benchmark == "longmemeval":
        parts.append(f"--longmemeval-variant {json.dumps(config.longmemeval_variant)}")
    if config.locomo_users:
        parts.append("--locomo-users " + " ".join(json.dumps(user) for user in config.locomo_users))
    if config.benchmark_limit is not None:
        parts.append(f"--limit {int(config.benchmark_limit)}")
    if config.max_history_turns is not None:
        parts.append(f"--max-history-turns {int(config.max_history_turns)}")
    for key, value in sorted(candidate.benchmark_args.items()):
        cli_key = str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(f"--{cli_key}")
        elif value is not None:
            parts.append(f"--{cli_key} {json.dumps(str(value))}")
    return " ".join(parts)


def run_benchmark_and_local_scoring(
    *,
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
        benchmark_command(config, candidate, predictions_path),
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


def run_evolution_search(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    command_runner = runner or CommandRunner()
    ensure_control_worktree_is_usable(repo_root, config, command_runner)
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
