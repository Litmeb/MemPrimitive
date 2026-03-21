"""Baseline stage-1 primitive implementations.

Concrete classes live in one module per DSL slot (see README.md in this package).
Public names are derived from each file's ``BASELINE_CLASSES`` (see ``registry.py``).
The ``simple`` submodule re-exports the same symbols for backward compatibility.
"""

from __future__ import annotations

from ..pipeline_slots import ALL_PIPELINE_SLOTS
from .registry import baseline_classes_by_slot

from .memory_evolution import *
from .evolution_trigger import *
from .organization import *
from .readout import *
from .representation import *
from .retrieval import *
from .unit_formation import *
from .write_trigger import *

_names: list[str] = []
_by_slot = baseline_classes_by_slot()
for _slot in ALL_PIPELINE_SLOTS:
    for _cls in _by_slot[_slot]:
        globals()[_cls.__name__] = _cls
        _names.append(_cls.__name__)

__all__ = tuple(_names)
