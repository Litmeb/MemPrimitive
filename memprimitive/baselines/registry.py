"""Aggregate per-file baseline registries into factory groups for the pipeline.

Each slot module under ``memprimitive.baselines`` should define:

- ``BASELINE_SLOT`` — string matching :attr:`~memprimitive.core.ModuleSpec.slot`
  for implementations in that file.
- ``BASELINE_CLASSES`` — tuple of concrete primitive classes for that slot.

Non-slot helper modules (no registry) are skipped. See ``README.md`` in this package.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

from ..interfaces import PrimitiveModule
from ..pipeline_slots import ALL_PIPELINE_SLOTS

BaselineFactory = Callable[[], PrimitiveModule]


def _default_factory_for_class(cls: type[PrimitiveModule], slot: str, *, top_k: int) -> BaselineFactory:
    """Instantiate a registered class; slot-specific kwargs live here (not per-class lists)."""
    if slot == "retrieval":
        return lambda c=cls, k=top_k: c(top_k=k)
    if slot == "readout" and cls.__name__ == "TemplateReadout":
        return lambda c=cls: c(prompt="{{ retrieved.items | join_text }}")
    if slot == "write_trigger":
        return lambda c=cls: c(slot="write_trigger")
    if slot == "evolution_trigger":
        if cls.__name__ == "PeriodicMaintenanceTrigger":
            from .trigger import NeverTrigger

            return lambda c=cls: c(
                slot="evolution_trigger",
                every_n=2,
                trigger=NeverTrigger(slot="evolution_trigger"),
            )
        return lambda c=cls: c(slot="evolution_trigger")
    return lambda c=cls: c()


@lru_cache(maxsize=1)
def baseline_classes_by_slot() -> dict[str, tuple[type[PrimitiveModule], ...]]:
    """Merge ``BASELINE_CLASSES`` from each slot module (imported via pkgutil)."""
    import memprimitive.baselines as pkg
    from .trigger import (
        AlwaysTrigger,
        BoundaryEventTrigger,
        IdleMaintenanceTrigger,
        LLMJudgeTrigger,
        NeverTrigger,
        PeriodicMaintenanceTrigger,
        RuntimeEventTrigger,
        ScalarRuleTrigger,
        StoreAllTrigger,
    )

    by_slot: dict[str, list[type[PrimitiveModule]]] = defaultdict(list)

    for mi in pkgutil.iter_modules(pkg.__path__):
        if mi.name.startswith("_") or mi.name == "registry":
            continue
        m = importlib.import_module(f"memprimitive.baselines.{mi.name}")
        has_classes = hasattr(m, "BASELINE_CLASSES")
        has_slot = hasattr(m, "BASELINE_SLOT")
        if has_classes != has_slot:
            raise RuntimeError(
                f"memprimitive.baselines.{mi.name}: define both BASELINE_SLOT and BASELINE_CLASSES, or neither."
            )
        if not has_classes:
            continue
        slot: str = m.BASELINE_SLOT
        for cls in m.BASELINE_CLASSES:
            if cls.spec.slot != slot:
                raise ValueError(
                    f"{cls.__name__}.spec.slot is {cls.spec.slot!r} but {mi.name}.BASELINE_SLOT is {slot!r}"
                )
            by_slot[slot].append(cls)

    by_slot["write_trigger"] = [
        AlwaysTrigger,
        StoreAllTrigger,
        BoundaryEventTrigger,
        RuntimeEventTrigger,
        ScalarRuleTrigger,
        LLMJudgeTrigger,
    ]
    by_slot["evolution_trigger"] = [
        NeverTrigger,
        StoreAllTrigger,
        BoundaryEventTrigger,
        RuntimeEventTrigger,
        ScalarRuleTrigger,
        LLMJudgeTrigger,
        PeriodicMaintenanceTrigger,
        IdleMaintenanceTrigger,
    ]

    missing = [s for s in ALL_PIPELINE_SLOTS if s not in by_slot or not by_slot[s]]
    if missing:
        raise RuntimeError(f"No BASELINE_CLASSES registered for required slot(s): {missing}")

    return {s: tuple(by_slot[s]) for s in ALL_PIPELINE_SLOTS}


def registered_baseline_class_names() -> set[str]:
    """Set of short class names listed in per-module ``BASELINE_CLASSES`` (for tests / exports)."""
    return {cls.__name__ for classes in baseline_classes_by_slot().values() for cls in classes}


def baseline_factory_groups(*, top_k: int) -> dict[str, tuple[BaselineFactory, ...]]:
    """One factory per registered class per slot (order matches ``BASELINE_CLASSES`` order)."""
    if top_k <= 0:
        raise ValueError("top_k must be positive for baseline_factory_groups.")

    classes_map = baseline_classes_by_slot()
    out: dict[str, tuple[BaselineFactory, ...]] = {}
    for slot in ALL_PIPELINE_SLOTS:
        classes = classes_map[slot]
        out[slot] = tuple(_default_factory_for_class(cls, slot, top_k=top_k) for cls in classes)
    return out


def instantiate_default_baseline_modules(*, top_k: int = 3) -> dict[str, PrimitiveModule]:
    """One module per slot using the first registered class in each slot."""
    groups = baseline_factory_groups(top_k=top_k)
    return {slot: groups[slot][0]() for slot in ALL_PIPELINE_SLOTS}


def materialize_pipeline_kwargs(
    factories_by_slot: Mapping[str, Sequence[BaselineFactory]],
    *,
    slot_order: Sequence[str] | None = None,
    choice_indices: Sequence[int] | None = None,
) -> dict[str, PrimitiveModule]:
    """Pick one factory per slot (by index) and instantiate modules."""
    order = tuple(slot_order) if slot_order is not None else ALL_PIPELINE_SLOTS
    if choice_indices is None:
        return {slot: factories_by_slot[slot][0]() for slot in order}

    if len(choice_indices) != len(order):
        raise ValueError("choice_indices length must match slot_order length.")

    out: dict[str, PrimitiveModule] = {}
    for slot, idx in zip(order, choice_indices, strict=True):
        group = factories_by_slot[slot]
        out[slot] = group[idx]()
    return out


def iter_baseline_pipeline_instances(*, top_k: int = 2):
    """Yield a :class:`~memprimitive.pipeline.MemoryPipeline` for each baseline combo."""
    from itertools import product

    from ..pipeline import MemoryPipeline

    groups = baseline_factory_groups(top_k=top_k)
    for combo in product(*(groups[s] for s in ALL_PIPELINE_SLOTS)):
        kwargs = {s: f() for s, f in zip(ALL_PIPELINE_SLOTS, combo, strict=True)}
        yield MemoryPipeline(**kwargs)
