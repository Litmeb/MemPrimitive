"""Codex CLI prompt construction and invocation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ._process import CommandRunner, ProcessResult, write_process_log
from ._types import CandidateSpec, EvolutionRunConfig, write_json


DEFAULT_CONTEXT_FILES = (
    "PROJECT_PROGRESS.md",
    "AGENTS.md",
    "DSL_SEMANTIC_OPERATION_DESIGN_AGENT_IDEAS.zh-CN.md",
    "DSL_SEMANTIC_OPERATION_IDEA_LIST.zh-CN.md",
    "DSL_SEMANTIC_OPERATION_MAP.zh-CN.md",
)


def target_binding_source_path(repo_root: Path, target_binding: str) -> Path | None:
    module_name = str(target_binding).replace(":", ".").rsplit(".", 1)[0]
    candidate = repo_root / (module_name.replace(".", "/") + ".py")
    return candidate if candidate.exists() else None


def read_context_documents(repo_root: Path, config: EvolutionRunConfig) -> list[tuple[str, str]]:
    paths = [repo_root / item for item in DEFAULT_CONTEXT_FILES]
    target_path = target_binding_source_path(repo_root, config.target_binding)
    if target_path is not None:
        paths.append(target_path)

    docs: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        if config.context_char_limit > 0 and len(text) > config.context_char_limit:
            text = text[: config.context_char_limit] + "\n\n[TRUNCATED BY EVOLUTION HARNESS]\n"
        docs.append((rel_path, text))
    return docs


def build_orchestrator_prompt(
    *,
    config: EvolutionRunConfig,
    round_index: int,
    context_documents: list[tuple[str, str]],
    previous_feedback: dict[str, Any] | None,
) -> str:
    docs = "\n\n".join(f"## {path}\n\n```text\n{text}\n```" for path, text in context_documents)
    feedback = json.dumps(previous_feedback or {}, ensure_ascii=False, indent=2)
    repair_instruction = ""
    if previous_feedback and not previous_feedback.get("passed_candidates"):
        repair_instruction = (
            "\nPrevious round had no passing candidates. Generate smaller, more local repair candidates and "
            "avoid repeating failed mutations."
        )
    return f"""You are the MemPrimitive memory-search orchestrator.

Your only job is to propose candidate code modifications. Do not edit files.
Return strict JSON only, with no markdown, no prose, and this exact top-level shape:
{{"candidates": [{{"id": "...", "hypothesis": "...", "allowed_files": ["..."], "implementation_prompt": "...", "focused_tests": ["..."], "benchmark_args": {{}}, "expected_diagnostics": ["..."]}}]}}

Run goal: {config.goal}
Round: {round_index} of {config.rounds}
Candidates requested: {config.candidates_per_round}
Target binding: {config.target_binding}
Benchmark: {config.benchmark}

Candidate rules:
- Each candidate must be a small semantic memory-operation mutation, not a broad rewrite.
- Each candidate must include allowed_files; worker edits will be mechanically rejected outside this whitelist.
- Prefer focused changes around retrieval, provenance, readout, evolution triggers, profile maintenance, or binding-facing behavior.
- Do not include protected files such as .env, *.env, .git, or benchmarks/outputs.
- Include focused pytest commands when a better focused test exists; otherwise use the default MemMachine regression commands.
- Include diagnostics that should change if the hypothesis is correct.
{repair_instruction}

Previous round feedback:
```json
{feedback}
```

Repository context:
{docs}
"""


def build_worker_prompt(*, config: EvolutionRunConfig, candidate: CandidateSpec) -> str:
    candidate_json = json.dumps(candidate.to_json_dict(), ensure_ascii=False, indent=2)
    return f"""You are a worker agent implementing exactly one MemPrimitive evolution candidate.

Read PROJECT_PROGRESS.md first and follow AGENTS.md. Use ~/bin/winpy312 for Python commands.

Candidate:
```json
{candidate_json}
```

Hard constraints:
- Edit only the candidate.allowed_files list.
- Do not edit .env, *.env, .git, benchmarks/outputs, or generated evolution artifacts.
- Preserve real retrieval/rerank/model behavior; do not replace it with mocks or heuristic stand-ins.
- Keep the change small and mechanism-level.
- Do not broaden the task beyond this candidate.

Implementation task:
{candidate.implementation_prompt}

After editing, provide a concise final message listing changed files, the mechanism change, and any tests you ran.
The outer harness will run the official three-layer checks.
"""


def codex_exec_args(
    *,
    config: EvolutionRunConfig,
    cwd: Path,
    sandbox: str,
    output_last_message: Path,
    json_events: bool = True,
) -> list[str]:
    args = [
        config.codex_bin,
        "exec",
        "-C",
        str(cwd),
        "--ask-for-approval",
        "never",
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(output_last_message),
    ]
    if json_events:
        args.append("--json")
    if config.codex_model:
        args.extend(["--model", config.codex_model])
    if config.codex_profile:
        args.extend(["--profile", config.codex_profile])
    args.append("-")
    return args


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Codex response JSON must be an object.")
    return value


def run_orchestrator(
    *,
    repo_root: Path,
    config: EvolutionRunConfig,
    round_index: int,
    previous_feedback: dict[str, Any] | None,
    artifact_dir: Path,
    runner: CommandRunner,
) -> list[CandidateSpec]:
    context_documents = read_context_documents(repo_root, config)
    prompt = build_orchestrator_prompt(
        config=config,
        round_index=round_index,
        context_documents=context_documents,
        previous_feedback=previous_feedback,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = artifact_dir / "orchestrator_prompt.md"
    final_path = artifact_dir / "orchestrator_final.md"
    events_path = artifact_dir / "orchestrator.jsonl"
    prompt_path.write_text(prompt, encoding="utf-8")
    result = runner.run(
        codex_exec_args(config=config, cwd=repo_root, sandbox="read-only", output_last_message=final_path),
        cwd=repo_root,
        input_text=prompt,
    )
    events_path.write_text(result.stdout, encoding="utf-8")
    write_process_log(result, artifact_dir=artifact_dir, name="orchestrator")
    if result.returncode != 0:
        raise RuntimeError(f"orchestrator codex exec failed with exit code {result.returncode}: {result.stderr}")
    payload = extract_json_object(final_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("orchestrator JSON must contain a candidates list.")
    specs = [CandidateSpec.from_json_dict(dict(item)) for item in candidates[: config.candidates_per_round]]
    if not specs:
        raise ValueError("orchestrator returned no candidates.")
    write_json(artifact_dir / "orchestrator_candidates.json", [spec.to_json_dict() for spec in specs])
    return specs


def run_worker_codex(
    *,
    worktree_path: Path,
    config: EvolutionRunConfig,
    candidate: CandidateSpec,
    artifact_dir: Path,
    runner: CommandRunner,
) -> ProcessResult:
    prompt = build_worker_prompt(config=config, candidate=candidate)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "worker_prompt.md").write_text(prompt, encoding="utf-8")
    final_path = artifact_dir / "worker_final.md"
    events_path = artifact_dir / "worker.jsonl"
    result = runner.run(
        codex_exec_args(
            config=config,
            cwd=worktree_path,
            sandbox="danger-full-access",
            output_last_message=final_path,
        ),
        cwd=worktree_path,
        input_text=prompt,
    )
    events_path.write_text(result.stdout, encoding="utf-8")
    write_process_log(result, artifact_dir=artifact_dir, name="worker")
    return result
