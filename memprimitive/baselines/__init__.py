"""Baseline stage-1 primitive implementations.

Concrete classes live in one module per DSL slot (see README.md in this package).
Public names are derived from each file's ``BASELINE_CLASSES`` (see ``registry.py``).
"""

from __future__ import annotations

from .registry import baseline_classes_by_slot, registered_baseline_class_names

__all__ = tuple(sorted(registered_baseline_class_names()))

_class_by_name: dict[str, type] | None = None


def _exports_dict() -> dict[str, type]:
    global _class_by_name
    if _class_by_name is None:
        _class_by_name = {
            cls.__name__: cls
            for classes in baseline_classes_by_slot().values()
            for cls in classes
        }
    return _class_by_name


def __getattr__(name: str) -> type:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        cls = _exports_dict()[name]
    except KeyError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    globals()[name] = cls
    return cls


def __dir__() -> list[str]:
    return sorted(set(__all__) | {k for k in globals() if not k.startswith("_")})
