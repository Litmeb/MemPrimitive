#!/usr/bin/env python3
"""Launch the DeepSeek Responses shim without importing the full memprimitive package."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_shim_module():
    repo_root = Path(__file__).resolve().parents[1]
    shim_path = repo_root / "memprimitive" / "evolution" / "deepseek_responses_shim.py"
    spec = importlib.util.spec_from_file_location("memprimitive_deepseek_responses_shim", shim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load shim module from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_shim_module()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
