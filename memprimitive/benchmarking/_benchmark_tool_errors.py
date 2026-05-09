"""Shared tooling-error ledger for unattended benchmark runs.

When upstream LLMs emit invalid tool calls, the Agents SDK raises
``ModelBehaviorError`` before MemPrimitive executes the tool callbacks.
Benchmark ingestion paths catch that exception so evaluation can proceed.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    from agents.exceptions import ModelBehaviorError
except ImportError:  # pragma: no cover - agents is expected at benchmark runtime

    class ModelBehaviorError(Exception):
        """Stand-in used only if ``agents`` is missing from the interpreter."""


class BenchmarkToolErrorLog:
    """Thread-safe append-only log consumed by ``run_benchmark``."""

    __slots__ = ("_events", "_lock")

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        """Record one JSON-serializable event payload."""

        with self._lock:
            self._events.append(dict(event))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._events]


_active_tool_error_log: BenchmarkToolErrorLog | None = None


def push_benchmark_tool_error_log(log: BenchmarkToolErrorLog | None) -> BenchmarkToolErrorLog | None:
    """Install *log* for nested ``run_benchmark`` calls; restores previous."""

    global _active_tool_error_log
    prior = _active_tool_error_log
    _active_tool_error_log = log
    return prior


def append_benchmark_tool_error(payload: dict[str, Any]) -> None:
    log = _active_tool_error_log
    if log is not None:
        log.append(dict(payload))


__all__ = [
    "BenchmarkToolErrorLog",
    "ModelBehaviorError",
    "append_benchmark_tool_error",
    "push_benchmark_tool_error_log",
]
