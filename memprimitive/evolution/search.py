"""CLI for unattended MemPrimitive memory search and evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from ._codex import DEFAULT_CONTEXT_FILES, target_binding_source_path
from ._runner import git_root, run_evolution_search
from ._types import DEFAULT_FOCUSED_TESTS, EvolutionRunConfig


def _parse_locomo_users(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    users: list[str] = []
    for value in values:
        users.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(users)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, help="Natural-language search/evolution objective.")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument(
        "--target-binding",
        default="memprimitive.example.classics.memmachine_memory:create_memory_binding",
        help="MemorySystemBinding factory used by benchmark runs.",
    )
    parser.add_argument("--benchmark", choices=("locomo", "longmemeval", "dmr"), default="locomo")
    parser.add_argument("--locomo-users", nargs="+", default=None)
    parser.add_argument("--longmemeval-variant", default="s_cleaned")
    parser.add_argument("--benchmark-root", default="benchmarks")
    parser.add_argument("--benchmark-limit", type=int, default=None)
    parser.add_argument("--benchmark-top-k", type=int, default=10)
    parser.add_argument("--memmachine-stm-record-budget", type=int, default=20)
    parser.add_argument("--memmachine-profile-max-turns", type=int, default=24)
    parser.add_argument("--benchmark-max-workers", type=int, default=10)
    parser.add_argument("--max-history-turns", type=int, default=None)
    parser.add_argument(
        "--llm-max-input-tokens",
        type=int,
        default=7000,
        help="Forwarded to memprimitive.benchmarking.minimal_baseline for LoCoMo LLM answer caps.",
    )
    parser.add_argument("--worktree-root", default="../MemPrimitive-evolve-worktrees")
    parser.add_argument("--output-root", default="benchmarks/outputs/evolve")
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--allow-dirty-control-worktree", action="store_true")
    parser.add_argument("--max-parallel-candidates", type=int, default=1)
    parser.add_argument("--promote-top-k", type=int, default=0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default=None, help="Deprecated compatibility option: sets one model for both roles.")
    parser.add_argument("--codex-profile", default="deepseek")
    parser.add_argument("--orchestrator-model", default="deepseek-v4-pro")
    parser.add_argument("--orchestrator-reasoning-effort", default="medium")
    parser.add_argument("--worker-model", default="deepseek-v4-flash")
    parser.add_argument("--worker-reasoning-effort", default=None)
    parser.add_argument("--orchestrator-timeout-seconds", type=int, default=600)
    parser.add_argument("--worker-timeout-seconds", type=int, default=900)
    parser.add_argument("--python-bin", default="~/bin/winpy312")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--resume-benchmark-only",
        action="store_true",
        help="Reuse an existing run_id's candidate worktrees and rerun benchmark/scoring for candidates that previously failed at the benchmark stage.",
    )
    parser.add_argument("--context-char-limit", type=int, default=60000)
    parser.add_argument("--dry-run", action="store_true", help="Print config and command/context preview without writing files.")
    return parser


def config_from_args(args: argparse.Namespace) -> EvolutionRunConfig:
    return EvolutionRunConfig(
        goal=args.goal,
        rounds=args.rounds,
        candidates_per_round=args.candidates_per_round,
        target_binding=args.target_binding,
        benchmark=args.benchmark,
        locomo_users=_parse_locomo_users(args.locomo_users),
        longmemeval_variant=args.longmemeval_variant,
        benchmark_root=args.benchmark_root,
        worktree_root=args.worktree_root,
        output_root=args.output_root,
        base_ref=args.base_ref,
        allow_dirty_control_worktree=args.allow_dirty_control_worktree,
        max_parallel_candidates=args.max_parallel_candidates,
        promote_top_k=args.promote_top_k,
        benchmark_limit=args.benchmark_limit,
        benchmark_top_k=args.benchmark_top_k,
        memmachine_stm_record_budget=args.memmachine_stm_record_budget,
        memmachine_profile_max_turns=args.memmachine_profile_max_turns,
        benchmark_max_workers=args.benchmark_max_workers,
        max_history_turns=args.max_history_turns,
        llm_max_input_tokens=args.llm_max_input_tokens,
        codex_bin=args.codex_bin,
        codex_model=args.codex_model,
        codex_profile=args.codex_profile,
        orchestrator_model=args.orchestrator_model,
        orchestrator_reasoning_effort=args.orchestrator_reasoning_effort,
        worker_model=args.worker_model,
        worker_reasoning_effort=args.worker_reasoning_effort,
        orchestrator_timeout_seconds=args.orchestrator_timeout_seconds,
        worker_timeout_seconds=args.worker_timeout_seconds,
        python_bin=args.python_bin,
        run_id=args.run_id,
        context_char_limit=args.context_char_limit,
        dry_run=args.dry_run,
    )


def dry_run_payload(repo_root: Path, config: EvolutionRunConfig) -> dict[str, object]:
    target_path = target_binding_source_path(repo_root, config.target_binding)
    context_files = [path for path in DEFAULT_CONTEXT_FILES if (repo_root / path).exists()]
    if target_path is not None:
        context_files.append(target_path.relative_to(repo_root).as_posix())
    return {
        "mode": "dry-run",
        "repo_root": str(repo_root),
        "config": config.to_json_dict(),
        "context_files": context_files,
        "default_focused_tests": list(DEFAULT_FOCUSED_TESTS),
        "artifact_dir": str((repo_root / config.output_root / config.run_id).resolve()),
        "worktree_run_dir": str(((repo_root / config.worktree_root).resolve() / config.run_id)),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)
    repo_root = git_root(Path.cwd())
    if config.dry_run:
        print(json.dumps(dry_run_payload(repo_root, config), ensure_ascii=False, indent=2))
        return 0
    try:
        result = run_evolution_search(repo_root=repo_root, config=config, resume_benchmark_only=args.resume_benchmark_only)
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
