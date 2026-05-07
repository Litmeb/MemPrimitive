"""Subprocess helpers used by the evolution harness."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ._types import CommandRecord


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
        if env:
            merged_env.update({str(key): str(value) for key, value in env.items()})
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            input=input_text,
            text=True,
            capture_output=True,
            shell=shell,
            timeout=timeout,
            env=merged_env,
        )
        return ProcessResult(
            args=str(args) if shell else tuple(str(part) for part in args),
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
