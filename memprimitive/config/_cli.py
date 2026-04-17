"""CLI helpers for MemPrimitive config loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..pipeline import MemoryPipeline
from ._loader import ConfigLoaderError, load_object_from_yaml


def _slot_summary(module_or_modules: Any) -> str:
    if isinstance(module_or_modules, tuple):
        return "[" + ", ".join(type(module).__name__ for module in module_or_modules) + "]"
    return type(module_or_modules).__name__


def _describe_object(obj: Any) -> str:
    if isinstance(obj, MemoryPipeline):
        return (
            f"MemoryPipeline("
            f"layers={list(obj.store.topology.layer_names)}, "
            f"retrieval={_slot_summary(obj.retrieval)}, "
            f"readout={_slot_summary(obj.readout)})"
        )
    return type(obj).__name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m memprimitive.config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Load a config file and validate its root object.")
    validate_parser.add_argument("path", help="Path to config.yml")
    validate_parser.add_argument("--root", help="Override the config's root object name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        config_path = Path(args.path)
        try:
            root_object = load_object_from_yaml(config_path, root=args.root)
        except ConfigLoaderError as error:
            print(f"Validation failed: {error}", file=sys.stderr)
            return 1

        print(f"Validation succeeded: {_describe_object(root_object)}")
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2

