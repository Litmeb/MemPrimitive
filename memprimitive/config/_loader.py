"""Declarative YAML loader for MemPrimitive object graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import importlib
from pathlib import Path
from typing import Any

import yaml

from ..pipeline import MemoryPipeline

_ROOT_PATH = "root"
_ALLOWED_VERSION = 1
_BASELINE_PREFIX = "memprimitive.baselines."
_PIPELINE_SLOT_NAMES = frozenset((*MemoryPipeline.INGEST_SLOTS, *MemoryPipeline.RECALL_SLOTS))
_TRIGGER_SLOT_NAMES = frozenset({"write_trigger", "evolution_trigger"})


class ConfigLoaderError(ValueError):
    """Base class for config document and object-resolution failures."""


class ConfigResolutionError(ConfigLoaderError):
    """Raised when a config node cannot be resolved into a runtime object."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _path_child(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _path_index(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _require_non_empty_string(value: Any, *, path: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigResolutionError(path, f"{field_name} must be a non-empty string.")
    return value.strip()


def _wrap_exception_message(prefix: str, error: Exception) -> str:
    return f"{prefix} ({type(error).__name__}: {error})"


def _normalize_callable_target(dotted_path: str, *, allow_baseline_shorthand: bool) -> str:
    normalized = dotted_path.strip()
    if "." not in normalized:
        return f"{_BASELINE_PREFIX}{normalized}"
    return normalized


@lru_cache(maxsize=None)
def _import_dotted_symbol(dotted_path: str) -> Any:
    parts = dotted_path.split(".")
    if len(parts) < 2:
        raise ImportError("dotted import must include both a module path and an attribute name.")

    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        attr_path = parts[split_index:]
        try:
            current: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                continue
            raise

        for attribute in attr_path:
            try:
                current = getattr(current, attribute)
            except AttributeError as error:
                raise ImportError(
                    f"import target {dotted_path!r} is missing attribute {attribute!r}."
                ) from error
        return current

    raise ImportError(f"could not import dotted symbol {dotted_path!r}.")


class _Resolver:
    def __init__(self, *, objects: Mapping[str, Any]) -> None:
        self._objects = dict(objects)
        self._cache: dict[str, Any] = {}
        self._resolving_stack: list[str] = []

    def resolve_root(self, root_name: str) -> Any:
        return self._resolve_object(root_name, requested_path=_ROOT_PATH)

    def resolve_value(
        self,
        node: Any,
        *,
        path: str,
        allow_baseline_shorthand: bool = False,
        pipeline_slot_hint: str | None = None,
    ) -> Any:
        if isinstance(node, Mapping):
            return self._resolve_mapping(
                node,
                path=path,
                allow_baseline_shorthand=allow_baseline_shorthand,
                pipeline_slot_hint=pipeline_slot_hint,
            )
        if isinstance(node, list):
            return [
                self.resolve_value(
                    item,
                    path=_path_index(path, index),
                    allow_baseline_shorthand=allow_baseline_shorthand,
                    pipeline_slot_hint=pipeline_slot_hint,
                )
                for index, item in enumerate(node)
            ]
        if isinstance(node, tuple):
            return tuple(
                self.resolve_value(
                    item,
                    path=_path_index(path, index),
                    allow_baseline_shorthand=allow_baseline_shorthand,
                    pipeline_slot_hint=pipeline_slot_hint,
                )
                for index, item in enumerate(node)
            )
        if allow_baseline_shorthand and isinstance(node, str):
            return self._resolve_implicit_call(
                node,
                path=path,
                pipeline_slot_hint=pipeline_slot_hint,
            )
        return node

    def _resolve_mapping(
        self,
        node: Mapping[str, Any],
        *,
        path: str,
        allow_baseline_shorthand: bool,
        pipeline_slot_hint: str | None,
    ) -> Any:
        if "$ref" in node:
            return self._resolve_ref(node, path=path)
        if "$import" in node:
            return self._resolve_import(node, path=path)
        if "$call" in node:
            return self._resolve_call(
                node,
                path=path,
                allow_baseline_shorthand=allow_baseline_shorthand,
                pipeline_slot_hint=pipeline_slot_hint,
            )
        return {
            key: self.resolve_value(
                value,
                path=_path_child(path, str(key)),
                allow_baseline_shorthand=allow_baseline_shorthand,
                pipeline_slot_hint=pipeline_slot_hint,
            )
            for key, value in node.items()
        }

    def _resolve_ref(self, node: Mapping[str, Any], *, path: str) -> Any:
        extras = set(node) - {"$ref"}
        if extras:
            raise ConfigResolutionError(
                path,
                f"$ref does not accept extra keys: {sorted(extras)}.",
            )
        object_name = _require_non_empty_string(node.get("$ref"), path=_path_child(path, "$ref"), field_name="$ref")
        return self._resolve_object(object_name, requested_path=path)

    def _resolve_import(self, node: Mapping[str, Any], *, path: str) -> Any:
        extras = set(node) - {"$import"}
        if extras:
            raise ConfigResolutionError(
                path,
                f"$import does not accept extra keys: {sorted(extras)}.",
            )
        dotted_path = _require_non_empty_string(
            node.get("$import"),
            path=_path_child(path, "$import"),
            field_name="$import",
        )
        try:
            return _import_dotted_symbol(dotted_path)
        except Exception as error:
            raise ConfigResolutionError(
                path,
                _wrap_exception_message(f"failed to import {dotted_path!r}", error),
            ) from error

    def _resolve_call(
        self,
        node: Mapping[str, Any],
        *,
        path: str,
        allow_baseline_shorthand: bool,
        pipeline_slot_hint: str | None,
    ) -> Any:
        extras = set(node) - {"$call", "args", "kwargs"}
        if extras:
            raise ConfigResolutionError(
                path,
                f"$call does not accept extra keys: {sorted(extras)}.",
            )

        raw_target = _require_non_empty_string(
            node.get("$call"),
            path=_path_child(path, "$call"),
            field_name="$call",
        )
        dotted_path = _normalize_callable_target(
            raw_target,
            allow_baseline_shorthand=allow_baseline_shorthand,
        )
        try:
            target = _import_dotted_symbol(dotted_path)
        except Exception as error:
            raise ConfigResolutionError(
                path,
                _wrap_exception_message(f"failed to import callable {dotted_path!r}", error),
            ) from error
        if not callable(target):
            raise ConfigResolutionError(path, f"import target {dotted_path!r} is not callable.")

        raw_args = node.get("args", [])
        raw_kwargs = node.get("kwargs", {})
        if isinstance(raw_args, (str, bytes, bytearray)) or not isinstance(raw_args, Sequence):
            raise ConfigResolutionError(_path_child(path, "args"), "$call args must be a list or tuple.")
        if not isinstance(raw_kwargs, Mapping):
            raise ConfigResolutionError(_path_child(path, "kwargs"), "$call kwargs must be a mapping.")

        args = [
            self.resolve_value(value, path=_path_index(_path_child(path, "args"), index))
            for index, value in enumerate(raw_args)
        ]
        kwargs = self._resolve_call_kwargs(
            target=target,
            raw_kwargs=raw_kwargs,
            path=path,
            pipeline_slot_hint=pipeline_slot_hint,
        )
        try:
            return target(*args, **kwargs)
        except ConfigLoaderError:
            raise
        except Exception as error:
            raise ConfigResolutionError(
                path,
                _wrap_exception_message(f"failed to call {dotted_path!r}", error),
            ) from error

    def _resolve_call_kwargs(
        self,
        *,
        target: Any,
        raw_kwargs: Mapping[str, Any],
        path: str,
        pipeline_slot_hint: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key, value in raw_kwargs.items():
            key_text = str(key)
            child_allow_baseline_shorthand = target is MemoryPipeline and key_text in _PIPELINE_SLOT_NAMES
            child_pipeline_slot_hint = key_text if child_allow_baseline_shorthand else None
            kwargs[key_text] = self.resolve_value(
                value,
                path=_path_child(_path_child(path, "kwargs"), key_text),
                allow_baseline_shorthand=child_allow_baseline_shorthand,
                pipeline_slot_hint=child_pipeline_slot_hint,
            )
        if pipeline_slot_hint in _TRIGGER_SLOT_NAMES and "slot" not in kwargs:
            kwargs["slot"] = pipeline_slot_hint
        return kwargs

    def _resolve_implicit_call(
        self,
        dotted_or_baseline_name: str,
        *,
        path: str,
        pipeline_slot_hint: str | None,
    ) -> Any:
        dotted_path = _normalize_callable_target(
            dotted_or_baseline_name,
            allow_baseline_shorthand=True,
        )
        try:
            target = _import_dotted_symbol(dotted_path)
        except Exception as error:
            raise ConfigResolutionError(
                path,
                _wrap_exception_message(f"failed to import implicit callable {dotted_path!r}", error),
            ) from error
        if not callable(target):
            raise ConfigResolutionError(path, f"implicit import target {dotted_path!r} is not callable.")

        kwargs: dict[str, Any] = {}
        if pipeline_slot_hint in _TRIGGER_SLOT_NAMES:
            kwargs["slot"] = pipeline_slot_hint
        try:
            return target(**kwargs)
        except Exception as error:
            raise ConfigResolutionError(
                path,
                _wrap_exception_message(f"failed to call implicit callable {dotted_path!r}", error),
            ) from error

    def _resolve_object(self, name: str, *, requested_path: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        if name not in self._objects:
            raise ConfigResolutionError(requested_path, f"unknown object reference {name!r}.")
        if name in self._resolving_stack:
            cycle = " -> ".join([*self._resolving_stack, name])
            raise ConfigResolutionError(requested_path, f"circular $ref detected: {cycle}.")

        object_path = _path_child("objects", name)
        self._resolving_stack.append(name)
        try:
            resolved = self.resolve_value(self._objects[name], path=object_path)
        finally:
            self._resolving_stack.pop()
        self._cache[name] = resolved
        return resolved


def _validate_document(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise ConfigResolutionError("", "config document must be a mapping.")

    if "version" not in config:
        raise ConfigResolutionError("version", "missing required top-level key.")
    version = config["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigResolutionError("version", "version must be an integer.")
    if version != _ALLOWED_VERSION:
        raise ConfigResolutionError("version", f"unsupported version {version!r}; expected {_ALLOWED_VERSION}.")

    if "root" not in config:
        raise ConfigResolutionError("root", "missing required top-level key.")
    root_name = _require_non_empty_string(config["root"], path="root", field_name="root")

    if "objects" not in config:
        raise ConfigResolutionError("objects", "missing required top-level key.")
    objects = config["objects"]
    if not isinstance(objects, Mapping):
        raise ConfigResolutionError("objects", "objects must be a mapping.")

    normalized_objects: dict[str, Any] = {}
    for key, value in objects.items():
        object_name = _require_non_empty_string(key, path="objects", field_name="object name")
        normalized_objects[object_name] = value
    return root_name, normalized_objects


def load_object_from_dict(config: Mapping[str, Any], *, root: str | None = None) -> Any:
    """Build the selected root object from a config mapping."""

    default_root_name, objects = _validate_document(config)
    selected_root = _require_non_empty_string(root, path="root", field_name="root override") if root is not None else default_root_name
    resolver = _Resolver(objects=objects)
    return resolver.resolve_root(selected_root)


def load_object_from_yaml(path: str | Path, *, root: str | None = None) -> Any:
    """Build the selected root object from a YAML config file."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigLoaderError(_wrap_exception_message(f"failed to read {config_path}", error)) from error
    except yaml.YAMLError as error:
        raise ConfigLoaderError(_wrap_exception_message(f"failed to parse YAML {config_path}", error)) from error

    try:
        return load_object_from_dict(payload, root=root)
    except ConfigLoaderError as error:
        raise ConfigLoaderError(f"{config_path}: {error}") from error


def load_pipeline_from_dict(config: Mapping[str, Any], *, root: str | None = None) -> MemoryPipeline:
    """Build a root object and require it to be a MemoryPipeline."""

    resolved = load_object_from_dict(config, root=root)
    if not isinstance(resolved, MemoryPipeline):
        expected = MemoryPipeline.__name__
        actual = type(resolved).__name__
        raise ConfigResolutionError(_ROOT_PATH, f"resolved root object must be {expected}, got {actual}.")
    return resolved


def load_pipeline_from_yaml(path: str | Path, *, root: str | None = None) -> MemoryPipeline:
    """Build a root object from YAML and require it to be a MemoryPipeline."""

    resolved = load_object_from_yaml(path, root=root)
    if not isinstance(resolved, MemoryPipeline):
        expected = MemoryPipeline.__name__
        actual = type(resolved).__name__
        raise ConfigLoaderError(f"{Path(path)}: {_ROOT_PATH}: resolved root object must be {expected}, got {actual}.")
    return resolved
