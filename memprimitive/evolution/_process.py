"""Subprocess helpers used by the evolution harness."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dotenv import dotenv_values

from ._types import CommandRecord


REPO_ENV_FILES = ("memprimitive/.env", "memprimitive/2.env")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def resolve_executable_args(args: Sequence[str] | str, *, shell: bool) -> Sequence[str] | str:
    if shell or isinstance(args, str) or not args:
        return args
    resolved = shutil.which(str(args[0]))
    if not resolved:
        return args
    return [resolved, *[str(part) for part in args[1:]]]


def load_repo_env_defaults(cwd: Path, env: dict[str, str]) -> tuple[str, ...]:
    loaded_keys: list[str] = []
    for rel_path in REPO_ENV_FILES:
        path = cwd / rel_path
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            key = str(key or "")
            if key and value is not None:
                env.setdefault(key, str(value))
                loaded_keys.append(key)
    return tuple(dict.fromkeys(loaded_keys))


def mark_wsl_inherited_env(env: dict[str, str], keys: Sequence[str]) -> None:
    entries = [item for item in env.get("WSLENV", "").split(":") if item]
    seen_names = {entry.split("/", 1)[0] for entry in entries}
    for key in keys:
        key = str(key)
        if not ENV_KEY_RE.match(key) or key in seen_names:
            continue
        entries.append(f"{key}/u")
        seen_names.add(key)
    if entries:
        env["WSLENV"] = ":".join(entries)


@dataclass(slots=True)
class ProcessResult:
    args: tuple[str, ...] | str
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def command_text(self) -> str:
        if isinstance(self.args, str):
            return self.args
        return " ".join(str(part) for part in self.args)


class CommandRunner:
    """Thin wrapper around subprocess so tests can provide a fake runner."""

    def run(
        self,
        args: Sequence[str] | str,
        *,
        cwd: Path,
        input_text: str | None = None,
        shell: bool = False,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        started = time.monotonic()
        merged_env = os.environ.copy()
        merged_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        wsl_inherited_keys = list(load_repo_env_defaults(cwd, merged_env))
        if env:
            merged_env.update({str(key): str(value) for key, value in env.items()})
            wsl_inherited_keys.extend(str(key) for key in env)
        mark_wsl_inherited_env(merged_env, wsl_inherited_keys)
        resolved_args = resolve_executable_args(args, shell=shell)
        try:
            completed = subprocess.run(
                resolved_args,
                cwd=str(cwd),
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                shell=shell,
                timeout=timeout,
                env=merged_env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return ProcessResult(
                args=str(resolved_args) if shell else tuple(str(part) for part in resolved_args),
                cwd=cwd,
                returncode=124,
                stdout=str(stdout),
                stderr=(str(stderr) + f"\nTIMEOUT after {timeout} seconds").strip(),
                duration_seconds=time.monotonic() - started,
            )
        return ProcessResult(
            args=str(resolved_args) if shell else tuple(str(part) for part in resolved_args),
            cwd=cwd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.monotonic() - started,
        )


def write_process_log(result: ProcessResult, *, artifact_dir: Path, name: str) -> CommandRecord:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / f"{name}.stdout.log"
    stderr_path = artifact_dir / f"{name}.stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return CommandRecord(
        command=result.command_text,
        cwd=str(result.cwd),
        returncode=result.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        duration_seconds=round(result.duration_seconds, 3),
    )
