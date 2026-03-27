# Baselines package layout

Stage-1 baseline implementations are **one Python module per DSL primitive slot**, aligned with `ModuleSpec.slot` and the abstract types in `memprimitive.interfaces`.

## Class documentation convention

Each **public baseline class** must have a **class-level docstring** (immediately under the `class` line) that:

1. **Behavior** — What the primitive does in one short paragraph (inputs/outputs at the `Packet` / `MemoryStore` level, not implementation trivia).
2. **Parameter constraints** — For `__init__`: valid ranges/types and how they affect behavior. For `run`: which `packet` fields must be set, length/alignment rules (e.g. `units` vs `decisions`), and whether the module mutates `store`.

New implementations in this package should follow the same pattern so readers can compare slots without opening every method body.

## File map

Each **slot** below lists the **concrete classes** registered via `BASELINE_CLASSES` in the cited source file (this is what `registry.py` discovers and what `memprimitive.baselines` exports by name).

### Slot `unit_formation` — `unit_formation.py`

- `PassThroughUnitFormation`
- `SentenceSplitUnitFormation`
- `LineSplitUnitFormation`
- `WindowedUnitFormation`
- `MetadataHintUnitFormation`

### Slot `representation` — `representation.py`

- `BasicRepresentation`
- `KeywordRepresentation`

### Slot `write_trigger` — `write_trigger.py`

- `AlwaysWriteTrigger`
- `ThresholdWriteTrigger`

(The same module also exposes `compose_write_trigger` for assembling trigger-family adapters; those are not extra `BASELINE_CLASSES` entries.)

### Slot `organization` — `organization.py`

- `AppendOrganization`
- `ConditionalLayerOrganization`
- `GraphAppendOrganization`

### Slot `evolution_trigger` — `evolution_trigger.py`

- `NeverEvolutionTrigger`
- `ThresholdEvolutionTrigger`

(The same module also exposes `compose_evolution_trigger` for assembling trigger-family adapters; those are not extra `BASELINE_CLASSES` entries.)

### Slot `memory_evolution` — `memory_evolution.py`

- `AppendOnlyEvolution`
- `TraceOnlyEvolution`
- `SummaryRewriteEvolution`
- `LayerMoveEvolution`
- `GraphNeighborAppendEvolution`

### Slot `retrieval` — `retrieval.py`

- `RecencyRetrieval`
- `KeywordCountRetrieval`
- `EmbeddingSimilarityRetrieval`
- `TagRetrieval`
- `EntityRetrieval`
- `BM25Retrieval`
- `GraphNeighborRetrieval`
- `GraphSeedAndExpandRetrieval`
- `LayerAwareRetrieval`

`BM25Retrieval` uses Okapi BM25 from `rank-bm25` over lowercase whitespace tokens from record text plus
`metadata["representation"]["keywords"]` when present. It sorts by descending BM25 score, uses recency to break
ties, and falls back to recency when all candidates score zero.

### Slot `readout` — `readout.py`

- `ConcatenateReadout`
- `BulletListReadout`
- `GroupedByLayerReadout`
- `JSONReadout`
- `GraphReadout`

## Representation: supported element kinds (`representation.py`)

`BasicRepresentation` (and subclass `KeywordRepresentation`) take `elements: tuple[str, ...]`. Names must come from the package allow-list `_VALID_ELEMENTS`; unknown names raise `ValueError` at construction. Duplicates are removed while preserving order.

For each enabled element, the module updates `MemoryUnit` fields and/or `unit.metadata["representation"]`, then sets `representation_elements` to the sorted set of **actually produced** element tags (some extractions are no-ops if nothing is found, in which case that element tag is not added).

| Element | Main outputs (where) | Behavior summary |
| --- | --- | --- |
| `text` | `MemoryUnit.text` (trimmed), `normalized_text` | Always included when selected; ensures normalized casing helper path. |
| `embedding` | `MemoryUnit.embedding` | Loads `sentence_transformers.SentenceTransformer` for `embedding_model` (default `sentence-transformers/all-MiniLM-L6-v2`, overridable; env `MEMPRIMITIVE_EMBEDDING_MODEL`). Model instances are cached per model id on the class. |
| `triple` | `MemoryUnit.triples` | Uses `metadata["triples"]` hint if present; else regex heuristics for “X likes/prefers/… Y” and “X is Y” patterns. |
| `kv` | `MemoryUnit.kv` | Uses `metadata["kv"]` hint if present; else `key: value` patterns and triple-derived keys. |
| `entities` | `MemoryUnit.entities` | Uses `metadata["entities"]` hint if present; else capitalized-word heuristic (`_ENTITY_PATTERN`), filtering trivial tokens. |
| `tags` | `MemoryUnit.tags` | Uses `metadata["tags"]` hint if present; else keyword heuristics from text plus `unit_type`, entity/kv/triple richness tags. |
| `keywords` | `metadata["representation"]["keywords"]` only | Uses `metadata["keywords"]` hint if present; else word-frequency top tokens (stopword-filtered) plus entities/tags. |
| `summary` | `metadata["representation"]["summary"]` | Uses `metadata["summary"]` string hint if present. Else requires `api_key`, `base_url`, and `model` (constructor or `MEMPRIMITIVE_*`); calls OpenAI chat completions for a short factual summary. Empty text yields no summary. |
| `time_anchor` | `metadata["representation"]["time_anchor"]` | Uses `metadata["time_anchor"]` dict if present; else derives `timestamp` / `date` from `MemoryUnit.timestamp`. |
| `relation_tags` | `metadata["representation"]["relation_tags"]` | Uses `metadata["relation_tags"]` hint if present; else `predicate:…` tags from triples and `multi_entity` when ≥2 entities. |
| `source_type` | `metadata["representation"]["source_type"]` | From `MemoryUnit.metadata["source"]` when set. |
| `description` | `MemoryUnit.description` | If `unit.description` already set, kept. Else requires `api_key`, `base_url`, and `model` (constructor or `MEMPRIMITIVE_*`); calls OpenAI chat completions for a short description. |

Constructor env fallbacks (via `memprimitive/.env`): `MEMPRIMITIVE_EMBEDDING_MODEL`, `MEMPRIMITIVE_API_KEY`, `MEMPRIMITIVE_BASE_URL`, `MEMPRIMITIVE_MODEL`.

`KeywordRepresentation` is a thin subclass: default `elements=("text", "keywords", "tags")` (no `embedding` unless you pass a custom `elements` tuple). Its `ModuleSpec.output_guarantees` omit explicit embedding fields because the default path does not require them.

## Trigger family: signal, scorer, gate, policy (`_trigger_family.py`)

The DSL still has separate pipeline slots `write_trigger` and `evolution_trigger`. Both baseline modules are thin adapters around **`TriggerFamilyRunner`**, which composes **four roles** (not extra pipeline slots): **signals** (0+ providers), **one scorer**, **one gate**, **one policy**. Execution per unit index is fixed:

1. Merge `SignalMap`s from every `SignalProvider.provide(context, unit_index)`.
2. `score = scorer.score(signals)`.
3. `gate_open = gate.evaluate(context, unit_index, signals=signals, score=score)`.
4. `decision = policy.decide(score=score, gate_open=gate_open)`.

`TriggerContext` carries `packet`, `store`, `output_field`, and `trace_key`. `write_trigger` adapters fill `Packet.decisions`; `evolution_trigger` adapters fill `Packet.evolution_decisions` and default-require `placements` on the packet.

Factory entry points: **`compose_write_trigger`** (`write_trigger.py`) and **`compose_evolution_trigger`** (`evolution_trigger.py`), each building `_TriggerFamilyWriteAdapter` / `_TriggerFamilyEvolutionAdapter` with a custom `ModuleSpec`.

### Signal role — `SignalProvider` (`provide`)

| Implementation | `provide` behavior |
| --- | --- |
| `ConstantSignal` | One float under `signal_name` (default name `constant`, value `1.0`). |
| `UnitLengthSignal` | Unit text length, optionally divided by `normalize_by` (default `100.0`). |
| `KeywordMatchSignal` | **Requires** `packet.query`. Counts overlap between query tokens and representation keywords + unit text tokens. |
| `HasEntitySignal` | `1.0` if `unit.entities` non-empty else `0.0`. |
| `HasTripleSignal` | `1.0` if `unit.triples` non-empty else `0.0`. |
| `HasKVSignal` | `1.0` if `unit.kv` non-empty else `0.0`. |
| `TagMatchSignal` | **Requires** `packet.query`. Counts overlap between query tokens and `unit.tags`. |
| `LayerTargetSignal` | **Requires** `packet.placements`. `1.0` if `placements[unit_index].target_layer` is in `allowed_layers`. |
| `QueryOverlapSignal` | **Requires** `packet.query`. Token overlap between query and unit text. |
| `MetadataFlagSignal` | Reads a bool-ish metadata flag from a dotted path on `unit.metadata`, `observation.metadata`, or `query.metadata`; missing paths may raise or use a configured default. |
| `UnitTypeSignal` | `1.0` iff `unit.unit_type == expected_unit_type`. |
| `PlacementExistsSignal` | **Requires** aligned `packet.placements`. Emits `1.0` when the current unit already has a placement entry. |
| `PartitionKeyPresentSignal` | Checks one or more dotted metadata paths (for example TiM-like group/hash keys) and emits `1.0` iff any configured key is present. |
| `NeighborCountSignal` | Counts comparable vector neighbors already present in the target layer, using current unit embedding plus placement-derived layer targeting. |
| `TopNeighborSimilaritySignal` | Returns the best cosine similarity among comparable vector neighbors in the target layer, or `0.0` when none are available. |
| `OutcomeCorrectnessSignal` | `1.0` if observation metadata implies the trial failed; else `0.0`. |
| `FeedbackPresenceSignal` | `1.0` if supported feedback text exists in observation metadata; else `0.0`. |

### Scorer role — `ScoreAggregator` (`score`)

| Implementation | `score` behavior |
| --- | --- |
| `IdentityScorer` | Returns `signals[source]` (default source `constant`). |
| `WeightedSumScorer` | Σ `weight * signals[name]` for each entry in `weights`; raises if a named signal is missing. |
| `MaxScorer` | Maximum of listed `sources`. |
| `MinScorer` | Minimum of listed `sources`. |
| `AverageScorer` | Arithmetic mean of listed `sources`. |
| `ClippedWeightedSumScorer` | `WeightedSumScorer` result clipped to `[min_score, max_score]`. |

### Gate role — `Gate` (`evaluate`)

Hard predicate per unit after scoring; receives full `signals` and `score` for context-rich gates if needed.

| Implementation | `evaluate` behavior |
| --- | --- |
| `AlwaysOpenGate` | Always `True`. |
| `RequireEntityGate` | `True` iff unit has entities. |
| `RequireTripleGate` | `True` iff unit has triples. |
| `RequireTagGate` | `True` iff unit tags intersect `required_tags` (case-insensitive). |
| `LayerAllowedGate` | **Requires** `packet.placements`; `True` iff placement target layer ∈ `allowed_layers`. |
| `QueryPresentGate` | `True` iff `packet.query` is not `None`. |
| `SchemaPresentGate` | Checks dotted schema paths on a selected source object (`unit`, `unit.metadata`, `observation`, `query`, etc.) and opens only when the required fields are present. |
| `FeedbackSchemaGate` | Opens only when observation metadata exposes a parseable outcome or feedback schema. |
| `HasEmbeddingGate` | Opens only when the configured source exposes a non-empty embedding vector. |
| `VectorIndexReadyGate` | Opens only when the target layer exists and supports the `vector` index. |
| `GraphLayerGate` | Opens only when the target layer exists and has `shape="Graph"`. |

### Policy role — `DecisionPolicy` (`decide`)

Final boolean decision from `score` and `gate_open`.

| Implementation | `decide` behavior |
| --- | --- |
| `AlwaysPolicy` | Always `True`. |
| `NeverPolicy` | Always `False`. |
| `ThresholdPolicy` | `gate_open and score >= threshold`. |
| `BooleanGatePolicy` | Returns `gate_open` (ignores score except for trace). |
| `BandPassThresholdPolicy` | `gate_open and lower <= score <= upper`. |
| `ThresholdOrGatePolicy` | `gate_open or score >= threshold`. |

### Baseline wiring

- **`AlwaysWriteTrigger`** — `ConstantSignal` + `IdentityScorer` + `AlwaysOpenGate` + `AlwaysPolicy`.
- **`ThresholdWriteTrigger`** — `ConstantSignal` + `WeightedSumScorer({"constant": 1.0})` + `AlwaysOpenGate` + `ThresholdPolicy`.
- **`NeverEvolutionTrigger`** — same signals/scorer/gate as always-write, but **`NeverPolicy`**.
- **`ThresholdEvolutionTrigger`** — same as `ThresholdWriteTrigger` but writes **`evolution_decisions`** and uses evolution adapter `input_requirements` (`units`, `placements`).

## `MemoryStore` and topology (pipeline integration)

`MemoryPipeline` accepts an optional `store` keyword (`memprimitive.pipeline.MemoryPipeline`). If omitted, the pipeline constructs `MemoryStore()` with the default topology from `StoreTopology.single_flat_default()` (a single **Flat** layer named `"default"`).

Relevant types live in `memprimitive.core`:

- **`StoreTopology`** — Declares ordered layers; each layer is a **`StoreLayerSpec`** (`name`, `theme`, `shape` ∈ {`Flat`, `Graph`}, `indices`, `capacity`). Duplicate layer names are rejected.
- **`MemoryStore`** — Holds `topology`, per-layer record lists, `allow_topology_extend`, and helpers such as `append`, `has_layer`, `layer_shape`, `layer_supports_index`.

**Composition check:** `MemoryPipeline._validate_store_compatibility` walks nested organization modules (including children reached via `iter_child_modules`, e.g. dispatch wrappers). If any module is `GraphAppendOrganization`, the store topology must declare its `target_layer` with `shape="Graph"`; otherwise `IncompatibleCompositionError` is raised.

## Shared helpers

- `_trace.py` — `copy_trace()` for shallow-copying `Packet.trace` before appending stage-local keys. Leading underscore marks it as package-internal; baseline authors should import it only from sibling modules.
- `_trigger_family.py` — Shared trigger-family runner/adapters used by `write_trigger.py` and `evolution_trigger.py` factory helpers; not a slot module (no `BASELINE_SLOT` / `BASELINE_CLASSES`). See **Trigger family: signal, scorer, gate, policy** above for the four roles and every concrete class/method.

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

## Pipeline composition rules (`memprimitive.pipeline` and `memprimitive.dispatch`)

`MemoryPipeline` validates each injected module with:

1. **Abstract type** — `unit_formation` must be a `UnitFormationModule`, `retrieval` a `RetrievalModule`, etc.
2. **`ModuleSpec.slot`** — must match the pipeline position (e.g. a `RetrievalModule` with `spec.slot == "retrieval"`).

Swapping modules across slots or reusing the wrong ABC fails fast with `TypeError` / `ValueError`. Combinatorial tests rely on this so invalid pairings are caught without maintaining a separate exclusion list until you introduce incompatible third-party primitives.

### Sequential modules per slot (iterable)

For any slot keyword, you may pass **one module** or a **non-string `Iterable`** of modules of that slot’s ABC. The pipeline runs them **in iteration order**: each module’s `run` receives the `(packet, store)` produced by the previous module in that slot. Empty iterables raise `ValueError`. Strings and bytes are not treated as iterables of modules (they would not validate as primitives).

### Fan-out per slot (`Dispatch*` classes)

Slot-local **fan-out** is implemented by the `Dispatch*` classes in **`memprimitive.dispatch`** (also re-exported from `memprimitive`): `DispatchUnitFormation`, `DispatchRepresentation`, `DispatchWriteTrigger`, `DispatchOrganization`, `DispatchEvolutionTrigger`, `DispatchMemoryEvolution`, `DispatchRetrieval`, `DispatchReadout`.

Each dispatcher wraps several **child** modules of the same slot type. On `run`, it **deep-copies** the incoming `Packet` once per child; children are invoked **one after another** on the **same** `MemoryStore` (so store mutations from all branches accumulate). This is **not** multi-threaded parallelism. The **`primary_index`** child supplies the `Packet` that continues to later slots; per-child slot traces are merged under `trace["dispatch"][<slot>]`. Dispatchers expose **`iter_child_modules()`** so graph/topology validation can see nested `GraphAppendOrganization` children.

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
