"""Custom exceptions for MemPrimitive runtime composition checks."""

from __future__ import annotations


class IncompatibleCompositionError(ValueError):
    """Raised when a pipeline composition is invalid before execution begins."""

