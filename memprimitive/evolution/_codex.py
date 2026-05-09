"""Codex CLI prompt construction and invocation."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
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

KNOWN_PATH_REWRITES = {
    "tests/test_classics_memmachine_binding.py": "tests/test_classics_memmachine.py",
    "tests/test_classics_memmachine_memory.py": "tests/test_classics_memmachine.py",
    "tests/classics/test_classics_memmachine_memory.py": "tests/test_classics_memmachine.py",
    "tests/example/classics/test_memmachine_memory.py": "tests/test_classics_memmachine.py",
}


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
Do not call tools or inspect the repository. The repository context below is already the input.
Your first response must be the strict JSON object; no analysis, no preamble, no markdown.
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
- Do not invent test files or `-k` selectors. Use exactly `~/bin/winpy312 -m pytest tests/test_classics_memmachine.py -v` unless you know an exact existing test is better.
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


def _windows_path_to_wsl(path: Path) -> str:
    value = str(path)
    normalized = value.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if not match:
        return normalized
    drive = match.group(1).casefold()
    rest = match.group(2)
    return f"/mnt/{drive}/{rest}"


def _discover_wsl_codex_bin() -> str:
    env_value = os.environ.get("MEMPRIMITIVE_EVOLUTION_WSL_CODEX_BIN", "").strip()
    if env_value:
        return env_value
    if not shutil.which("wsl.exe"):
        return "codex"
    command = (
        "ls -1d /mnt/c/Users/*/.cursor/extensions/openai.chatgpt-*/bin/linux-x86_64/codex "
        "2>/dev/null | sort -V | tail -n 1"
    )
    result = subprocess.run(
        ["wsl.exe", "bash", "-lc", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    path = result.stdout.strip()
    return path or "codex"


def _codex_exec_inner_args(
    *,
    config: EvolutionRunConfig,
    cwd: Path | str,
    sandbox: str,
    output_last_message: Path | str,
    json_events: bool,
    role: str,
) -> list[str]:
    args = [
        config.codex_bin,
        "--ask-for-approval",
        "never",
        "exec",
        "-C",
        str(cwd),
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(output_last_message),
    ]
    if json_events:
        args.append("--json")
    model, reasoning_effort = _model_settings_for_role(config, role=role)
    if model:
        args.extend(["--model", model])
    if reasoning_effort:
        args.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
    if config.codex_profile:
        args.extend(["--profile", config.codex_profile])
    args.append("-")
    return args


def _model_settings_for_role(config: EvolutionRunConfig, *, role: str) -> tuple[str | None, str | None]:
    if config.codex_model:
        return config.codex_model, None
    if role == "orchestrator":
        return config.orchestrator_model, config.orchestrator_reasoning_effort
    if role == "worker":
        return config.worker_model, config.worker_reasoning_effort
    raise ValueError(f"unknown codex role {role!r}.")


def _truthy_env(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _falsey_env(value: str) -> bool:
    return value.strip().casefold() in {"0", "false", "no", "off"}


def _uses_deepseek_codex_config(config: EvolutionRunConfig) -> bool:
    values = (
        config.codex_profile,
        config.codex_model,
        config.orchestrator_model,
        config.worker_model,
    )
    return any(str(value).strip().casefold().startswith("deepseek") for value in values if value)


def _should_use_wsl_codex(config: EvolutionRunConfig) -> bool:
    env_value = os.environ.get("MEMPRIMITIVE_EVOLUTION_USE_WSL_CODEX", "")
    if _truthy_env(env_value):
        return True
    if _falsey_env(env_value):
        return False
    return _uses_deepseek_codex_config(config)


def codex_exec_args(
    *,
    config: EvolutionRunConfig,
    cwd: Path,
    sandbox: str,
    output_last_message: Path,
    json_events: bool = True,
    role: str = "worker",
) -> list[str]:
    use_wsl_codex = _should_use_wsl_codex(config)
    if use_wsl_codex and os.name == "nt" and config.codex_bin == "codex" and shutil.which("wsl.exe"):
        wsl_codex_bin = _discover_wsl_codex_bin()
        wsl_cwd = _windows_path_to_wsl(cwd)
        wsl_output = _windows_path_to_wsl(output_last_message)
        inner_config = EvolutionRunConfig(
            **{
                **config.to_json_dict(),
                "codex_bin": wsl_codex_bin,
            }
        )
        inner_args = _codex_exec_inner_args(
            config=inner_config,
            cwd=wsl_cwd,
            sandbox=sandbox,
            output_last_message=wsl_output,
            json_events=json_events,
            role=role,
        )
        command = " ".join(shlex.quote(part) for part in inner_args)
        return ["wsl.exe", "bash", "-lc", command]
    return _codex_exec_inner_args(
        config=config,
        cwd=cwd,
        sandbox=sandbox,
        output_last_message=output_last_message,
        json_events=json_events,
        role=role,
    )


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


def _codex_failure_message(result: ProcessResult, *, events_path: Path, final_path: Path) -> str:
    parts = [f"codex exec failed with exit code {result.returncode}"]
    if result.stderr.strip():
        parts.append(f"stderr:\n{result.stderr.strip()}")
    if result.stdout.strip():
        messages: list[str] = []
        for raw_line in result.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                message = event.get("message")
                if message:
                    messages.append(str(message))
                error = event.get("error")
                if isinstance(error, dict) and error.get("message"):
                    messages.append(str(error["message"]))
        if messages:
            parts.append("codex events:\n" + "\n".join(dict.fromkeys(messages)))
        else:
            parts.append(f"stdout:\n{result.stdout.strip()}")
    parts.append(f"events_path: {events_path}")
    parts.append(f"final_path: {final_path}")
    return "\n\n".join(parts)


def normalize_candidate_for_repo(repo_root: Path, candidate: CandidateSpec) -> CandidateSpec:
    allowed_files = tuple(KNOWN_PATH_REWRITES.get(path, path) for path in candidate.allowed_files)
    focused_tests = []
    for command in candidate.focused_tests:
        normalized = command
        for old, new in KNOWN_PATH_REWRITES.items():
            normalized = normalized.replace(old, new)
        focused_tests.append(normalized)
    return CandidateSpec(
        id=candidate.id,
        hypothesis=candidate.hypothesis,
        allowed_files=allowed_files,
        implementation_prompt=candidate.implementation_prompt,
        focused_tests=tuple(focused_tests),
        benchmark_args=candidate.benchmark_args,
        expected_diagnostics=candidate.expected_diagnostics,
    )


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
        codex_exec_args(
            config=config,
            cwd=repo_root,
            sandbox="read-only",
            output_last_message=final_path,
            role="orchestrator",
        ),
        cwd=repo_root,
        input_text=prompt,
        timeout=config.orchestrator_timeout_seconds,
    )
    events_path.write_text(result.stdout, encoding="utf-8")
    write_process_log(result, artifact_dir=artifact_dir, name="orchestrator")
    if result.returncode != 0:
        raise RuntimeError(
            "orchestrator " + _codex_failure_message(result, events_path=events_path, final_path=final_path)
        )
    payload = extract_json_object(final_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("orchestrator JSON must contain a candidates list.")
    specs = [
        normalize_candidate_for_repo(repo_root, CandidateSpec.from_json_dict(dict(item)))
        for item in candidates[: config.candidates_per_round]
    ]
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
            role="worker",
        ),
        cwd=worktree_path,
        input_text=prompt,
        timeout=config.worker_timeout_seconds,
    )
    events_path.write_text(result.stdout, encoding="utf-8")
    write_process_log(result, artifact_dir=artifact_dir, name="worker")
    return result
