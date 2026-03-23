"""Backward-compatible aggregate of all stage-1 baselines.

Historically all baseline classes lived in this module. They are now split by
primitive slot (``unit_formation``, ``representation``, …). Import from
``memprimitive.baselines`` or from the slot module (e.g.
``memprimitive.baselines.unit_formation``) for new code.

This module mirrors the parent package's public names (see ``__init__.py``) so that
``from memprimitive.baselines.simple import PassThroughUnitFormation`` keeps working.
"""

from __future__ import annotations

import sys

_pkg = sys.modules[__package__]
__all__ = list(_pkg.__all__)
for _name in __all__:
    globals()[_name] = getattr(_pkg, _name)
