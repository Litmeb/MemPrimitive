"""Shared helpers for baseline primitive implementations."""

from __future__ import annotations

from ..core import Packet


def copy_trace(packet: Packet) -> dict:
    """Shallow copy of packet.trace for stage-local mutation."""
    return dict(packet.trace)
