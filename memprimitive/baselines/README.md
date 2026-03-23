# Baselines package layout

Stage-1 baseline implementations are **one Python module per DSL primitive slot**, aligned with `ModuleSpec.slot` and the abstract types in `memprimitive.interfaces`.

## Class documentation convention

Each **public baseline class** must have a **class-level docstring** (immediately under the `class` line) that:

1. **Behavior** — What the primitive does in one short paragraph (inputs/outputs at the `Packet` / `MemoryStore` level, not implementation trivia).
2. **Parameter constraints** — For `__init__`: valid ranges/types and how they affect behavior. For `run`: which `packet` fields must be set, length/alignment rules (e.g. `units` vs `decisions`), and whether the module mutates `store`.

New implementations in this package should follow the same pattern so readers can compare slots without opening every method body.

## File map


| File                  | Slot (`ModuleSpec.slot`) | Example class              |
| --------------------- | ------------------------ | -------------------------- |
| `unit_formation.py`   | `unit_formation`         | `PassThroughUnitFormation` |
| `representation.py`   | `representation`         | `BasicRepresentation`      |
| `write_trigger.py`    | `write_trigger`          | `AlwaysWriteTrigger`       |
| `organization.py`     | `organization`           | `AppendOrganization`       |
| `evolution_trigger.py`| `evolution_trigger`      | `NeverEvolutionTrigger`    |
| `memory_evolution.py` | `memory_evolution`       | `AppendOnlyEvolution`      |
| `retrieval.py`        | `retrieval`              | `RecencyRetrieval`         |
| `readout.py`          | `readout`                | `ConcatenateReadout`       |


## Shared helpers

- `_trace.py` — `copy_trace()` for shallow-copying `Packet.trace` before appending stage-local keys. Leading underscore marks it as package-internal; baseline authors should import it only from sibling modules.

## Per-file registry (source of truth)

Each **slot** module (e.g. `unit_formation.py`, `retrieval.py`) must end with:

- **`BASELINE_SLOT`** — string equal to that file’s `ModuleSpec.slot` for every class listed below.
- **`BASELINE_CLASSES`** — tuple of **concrete** primitive classes implemented in that file (add another class to the tuple when you add an alternative baseline for the same slot).

Helper modules that are not tied to a slot (e.g. future shared utilities) should define **neither** field so `registry.py` skips them.

`registry.py` **does not** list classes by hand: it imports submodules with `pkgutil`, merges `BASELINE_CLASSES` by `BASELINE_SLOT`, and builds factories. If two files use the same `BASELINE_SLOT`, their classes are **concatenated** (order: import order of submodules).

**Instantiation:** most slots use zero-arg `cls()`. The **`retrieval`** slot passes `top_k` from the caller (`baseline_factory_groups`, `create_baseline_pipeline`, tests). If a new retrieval class needs different constructor arguments, extend `_default_factory_for_class` in `registry.py` (structural rule, not a class list).

## Aggregator (`registry.py`)

- **`baseline_classes_by_slot()`** — merged `BASELINE_CLASSES` per slot (cached).
- **`registered_baseline_class_names()`** — all registered class short names (for `__all__` tests).
- **`baseline_factory_groups(top_k=...)`** — factories derived from registered classes; Cartesian product used in tests.
- **`instantiate_default_baseline_modules(top_k=...)`** — first class per slot; used by `create_baseline_pipeline`.
- **`iter_baseline_pipeline_instances(top_k=...)`** — one `MemoryPipeline` per combination.

Slot order is defined in `memprimitive.pipeline_slots` (`ALL_PIPELINE_SLOTS`, …). New slots require entries there and in `MemoryPipeline` validation.

## Pipeline composition rules (`memprimitive.pipeline`)

`MemoryPipeline` validates each injected module with:

1. **Abstract type** — `unit_formation` must be a `UnitFormationModule`, `retrieval` a `RetrievalModule`, etc.
2. **`ModuleSpec.slot`** — must match the pipeline position (e.g. a `RetrievalModule` with `spec.slot == "retrieval"`).

Swapping modules across slots or reusing the wrong ABC fails fast with `TypeError` / `ValueError`. Combinatorial tests rely on this so invalid pairings are caught without maintaining a separate exclusion list until you introduce incompatible third-party primitives.

## Public entry points

- **`__init__.py`** exposes all classes listed in per-file **`BASELINE_CLASSES`** (via `registry.baseline_classes_by_slot()`); no manual symbol list.
- **`simple.py`** mirrors the parent package namespace (`sys.modules[__package__]`) for backward compatibility. Older code may use `from memprimitive.baselines.simple import RecencyRetrieval`; new code should prefer the package root or the slot module.
- **Registry** is imported as `from memprimitive.baselines.registry import ...` (not duplicated in `__init__.__all__`).

## Extension guidelines

1. **Add a new implementation** for an existing slot: implement the class in that slot’s file and append it to **`BASELINE_CLASSES`** (package exports update automatically). Follow the **Class documentation convention** above.
2. **Add a new primitive slot** (future DSL versions): extend `memprimitive.interfaces`, add `pipeline_slots` and `MemoryPipeline` validation, add a new file with `BASELINE_SLOT` / `BASELINE_CLASSES`.
3. Keep **cross-slot helpers** minimal; if something is shared by many slots, consider `_trace.py` or a small `_util.py` rather than growing a second “god” module.

## Compatibility notes

- `create_baseline_pipeline` builds from `registry.instantiate_default_baseline_modules`; it does not depend on `simple.py`.
- Tests that import from `memprimitive.baselines` are unchanged.
- Any external code importing `memprimitive.baselines.simple` continues to work via the re-export layer.
