"""Dataclasses and JSON helpers for the evolution harness."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_FOCUSED_TESTS = (
    "~/bin/winpy312 -m pytest tests/test_classics_memmachine.py tests/test_benchmarking_adapters.py -v",
)

PROTECTED_PATH_PREFIXES = (
    ".git/",
    "benchmarks/outputs/",
)


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def slugify(value: str, *, default: str = "candidate") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip().casefold())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or default


def normalize_repo_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def is_protected_path(path: str | Path) -> bool:
    normalized = normalize_repo_path(path)
    if not normalized:
        return False
    if normalized == ".env" or normalized.endswith("/.env") or normalized.endswith(".env"):
        return True
    return any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in PROTECTED_PATH_PREFIXES)


def is_allowed_path(path: str | Path, allowed_files: tuple[str, ...]) -> bool:
    normalized = normalize_repo_path(path)
    for allowed in allowed_files:
        allowed_path = normalize_repo_path(allowed)
        if not allowed_path:
            continue
        if allowed_path.endswith("/"):
            if normalized.startswith(allowed_path):
                return True
        elif normalized == allowed_path:
            return True
    return False


@dataclass(slots=True)
class EvolutionRunConfig:
    goal: str
    rounds: int
    candidates_per_round: int
    target_binding: str
    benchmark: str = "locomo"
    locomo_users: tuple[str, ...] = ()
    longmemeval_variant: str = "s_cleaned"
    benchmark_root: str = "benchmarks"
    worktree_root: str = "../MemPrimitive-evolve-worktrees"
    output_root: str = "benchmarks/outputs/evolve"
    base_ref: str = "HEAD"
    allow_dirty_control_worktree: bool = False
    max_parallel_candidates: int = 1
    promote_top_k: int = 0
    benchmark_limit: int | None = None
    max_history_turns: int | None = None
    llm_max_input_tokens: int | None = None
    codex_bin: str = "codex"
    codex_model: str | None = None
    codex_profile: str | None = "deepseek"
    orchestrator_model: str | None = "deepseek-v4-pro"
    orchestrator_reasoning_effort: str | None = "medium"
    worker_model: str | None = "deepseek-v4-flash"
    worker_reasoning_effort: str | None = None
    orchestrator_timeout_seconds: int = 600
    worker_timeout_seconds: int = 900
    python_bin: str = "~/bin/winpy312"
    run_id: str = ""
    context_char_limit: int = 60000
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.goal = str(self.goal).strip()
        if not self.goal:
            raise ValueError("goal is required.")
        if self.rounds <= 0:
            raise ValueError("rounds must be positive.")
        if self.candidates_per_round <= 0:
            raise ValueError("candidates_per_round must be positive.")
        if self.max_parallel_candidates <= 0:
            raise ValueError("max_parallel_candidates must be positive.")
        if self.orchestrator_timeout_seconds <= 0:
            raise ValueError("orchestrator_timeout_seconds must be positive.")
        if self.worker_timeout_seconds <= 0:
            raise ValueError("worker_timeout_seconds must be positive.")
        self.target_binding = str(self.target_binding).strip()
        if not self.target_binding:
            raise ValueError("target_binding is required.")
        self.benchmark = str(self.benchmark).strip().casefold() or "locomo"
        self.locomo_users = tuple(str(item).strip() for item in self.locomo_users if str(item).strip())
        if self.llm_max_input_tokens is not None and self.benchmark != "locomo":
            raise ValueError("llm_max_input_tokens is only supported when benchmark is locomo.")
        if not self.run_id:
            self.run_id = f"{utc_timestamp()}_{slugify(self.goal, default='evolution')[:48]}"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateSpec:
    id: str
    hypothesis: str
    allowed_files: tuple[str, ...]
    implementation_prompt: str
    focused_tests: tuple[str, ...] = DEFAULT_FOCUSED_TESTS
    benchmark_args: dict[str, Any] = field(default_factory=dict)
    expected_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.id = slugify(self.id)
        self.hypothesis = str(self.hypothesis).strip()
        if not self.hypothesis:
            raise ValueError("CandidateSpec.hypothesis is required.")
        self.allowed_files = tuple(dict.fromkeys(normalize_repo_path(path) for path in self.allowed_files if str(path).strip()))
        if not self.allowed_files:
            raise ValueError("CandidateSpec.allowed_files must contain at least one file.")
        self.implementation_prompt = str(self.implementation_prompt).strip() or self.hypothesis
        self.focused_tests = tuple(str(command).strip() for command in self.focused_tests if str(command).strip())
        if not self.focused_tests:
            self.focused_tests = DEFAULT_FOCUSED_TESTS
        self.benchmark_args = dict(self.benchmark_args)
        self.expected_diagnostics = tuple(str(item).strip() for item in self.expected_diagnostics if str(item).strip())

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CandidateSpec":
        return cls(
            id=str(data.get("id", "")),
            hypothesis=str(data.get("hypothesis", "")),
            allowed_files=tuple(data.get("allowed_files") or ()),
            implementation_prompt=str(data.get("implementation_prompt", "")),
            focused_tests=tuple(data.get("focused_tests") or DEFAULT_FOCUSED_TESTS),
            benchmark_args=dict(data.get("benchmark_args") or {}),
            expected_diagnostics=tuple(data.get("expected_diagnostics") or ()),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommandRecord:
    command: str
    cwd: str
    returncode: int
    stdout_path: str | None = None
    stderr_path: str | None = None
    duration_seconds: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateResult:
    candidate_id: str
    status: str
    changed_files: tuple[str, ...] = ()
    rejected_reasons: tuple[str, ...] = ()
    failed_stage: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    commands: tuple[CommandRecord, ...] = ()
    worker_final_message: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "commands": [command.to_json_dict() for command in self.commands],
        }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
