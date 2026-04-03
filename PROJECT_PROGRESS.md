# MemPrimitive Project Progress

## Purpose

This file is the shared long-lived project status note for future agents. Keep it short, current, and focused on what still matters.

## Project Goal

`MemPrimitive` aims to describe agent memory systems as composable primitives rather than paper-specific pipelines. The target end state is:

- a unified primitive ontology / DSL
- an executable runtime built around `MemoryPipeline`
- re-expression of representative memory papers inside the same framework
- explicit constraints for safe composition and later architecture search
- eventual evaluation and motif discovery over the design space

## Current Status

The project is already beyond the concept stage.

- The stage-1 runtime is real and usable: slot-based composition, topology/store contracts, baseline modules, and a shared LLM/runtime path all exist.
- Chinese-facing documentation has been refreshed and is now much closer to the actual runtime surface.
- Several representative memory families now have decomposition work or reusable primitive support, with the repository increasingly centered on baseline modules rather than paper-specific wrappers.
- Literature coverage work is active. The repo is no longer focused only on a few showcase papers; it is trying to scale toward a broader set of memory papers and judge which are fully re-expressible versus only partially mappable.
- A trigger-focused literature pass now suggests the trigger subsystem is probably over-modeled relative to the literature: only a small minority of candidate papers appear genuinely trigger-centric, while most heterogeneity sits in representation, organization, maintenance, and retrieval.
- `TRIGGERS.md` has now been expanded from coarse `core/secondary/generic` labels into a 40-paper trigger mapping pass grounded in original-paper reading: each candidate is mapped to concrete trigger prototypes (`failure`, `capacity`, `scheduled`, `type-routing`, `subgoal`, `threshold`, `boolean gate`, `always`) plus named signals where relevant.
- Trigger redesign groundwork is now stronger than a qualitative survey: `TRIGGERS.md` also includes prototype occurrence counts plus per-paper provenance, and argues that `always` should remain a default behavior while `new-write conditioned` should likely be treated as an event hook rather than a heavyweight first-class trigger family.
- The trigger survey has been refined again: the old coarse `scheduled/offline` bucket was too broad and has now been split into `PeriodicTrigger`, `SessionEndTrigger`, and `IdleTrigger`, with a second pass over all 40 papers. That pass also downgraded several previously over-eager `scheduled/offline` assignments to explicit "not representable by the current trigger vocabulary" notes when the source only described a background or offline phase rather than a real trigger.
- `TRIGGERS.md` had a later UTF-8/PowerShell corruption incident on April 1, 2026; the canonical survey content has been restored from the earlier intact commit (`8dfc722`) so the file on disk again matches the original Codex-authored Chinese text rather than the mojibake version.
- The same April 1, 2026 UTF-8/PowerShell corruption pattern also affected `PLAN_ADD_MODULE.md` and `PAPER_DECOMPOSE.md`; both have likewise been restored from the earlier intact `8dfc722` state so the on-disk notes again match their original Chinese Codex-authored content.
- The documentation corruption cleanup has since been extended further: `DSLIO.md` was restored from its last intact pre-corruption state (`3b05409`), and `DSL_REFERENCE.zh-CN.md` was restored from its last intact pre-corruption state (`8dfc722`), so both DSL-facing docs are back to readable Chinese / intended text rather than mojibake.
- `DSL_REFERENCE.zh-CN.md` has now been expanded beyond the old slot-order skeleton: it enumerates the current public baseline modules under each pipeline slot, along with constructor parameters and short effect summaries, so the Chinese DSL reference is once again usable as a code-aligned lookup rather than only a placeholder outline.
- A new literature-side survey artifact now exists for the same 40-paper candidate pool: `ORGANIZATION_EVOLUTION_SURVEY.zh-CN.md` records a quick per-paper pass over `organization` and `evolution` methods, complementing the existing trigger-focused notes and making it easier to judge which heterogeneity really lies in storage structure, consolidation, graph update, profile maintenance, and cross-store migration rather than in trigger design alone.
- A follow-up pass on `subgoal-completion conditioned` made the category stricter as well: after re-checking the previously listed papers, `HiAgent` remains the clearest true fit, while several others were better reclassified as session-boundary or event-boundary triggers instead of subgoal completion.
- The follow-up trigger note has now been rewritten again at a more implementation-oriented level: `TRIGGER_SCORE_BOOLEAN_GATE_SURVEY.md` no longer primarily groups triggers by semantic labels like `threshold(score)` or `boolean gate`, but instead by concrete trigger implementation families for write/evolution (`PassThroughHook`, `StructuralBoundaryHook`, `RuntimeCallback`, `ExplicitScalarRule`, `LLMJudge`, `BackgroundScheduler`, `ControllerOrchestrator`) across the same 40-paper corpus.
- The code-aligned trigger rewrite plan in `TRIGGER_REWRITE_IMPLEMENTATION_PLAN.md` has now been updated again to match the current baseline runtime more closely: it no longer assumes separate write/evolution trigger class hierarchies, and instead treats richer trigger families as additions to the existing unified `TriggerModule` + `slot=` pattern in `memprimitive/baselines/trigger.py`, while still recording that true periodic/idle background maintenance likely needs a future dedicated maintenance entrypoint.
- The public baseline trigger surface has now been intentionally simplified again: the old trigger-family decomposition (`signal / scorer / gate / policy`) and compose-style trigger builders have been removed from the baseline API, and the repo now only exposes basic slot triggers (`AlwaysTrigger`, `ThresholdTrigger`, `NeverTrigger`) on that layer.
- That minimal trigger surface has now been expanded again in code, but the redundant input-pass-through alias has since been removed: `memprimitive/baselines/trigger.py` keeps a unified `TriggerModule + slot=` design with `AlwaysTrigger` / `NeverTrigger` / `ThresholdTrigger` plus direct richer families such as `BoundaryEventTrigger`, `RuntimeEventTrigger`, `ScalarRuleTrigger`, `ModelJudgeTrigger`, `PeriodicMaintenanceTrigger`, and `IdleMaintenanceTrigger`, while preserving the stable downstream contract of `packet.decisions` plus `trace["write_trigger"]` / `trace["evolution_trigger"]`.
- Trigger pressure support has now been filled in for the two main explicit/runtime trigger families: `ScalarRuleTrigger(signal_key="memory_pressure")` can compute per-layer `record_budget` / `token_budget` pressure directly from the current store, and `RuntimeEventTrigger(accepted_events=("memory_pressure",), pressure_threshold=...)` can now synthesize a runtime `memory_pressure` event from the same store-aware pressure snapshot instead of only matching externally supplied event strings.
- That pressure-aware trigger path now also populates the parallel store-side selection artifact consistently: both `RuntimeEventTrigger(memory_pressure)` and `ScalarRuleTrigger(signal_key="memory_pressure")` write the affected layer's full record set into `packet.decisions_store` when they fire, so downstream evolution/maintenance code can operate on the same selected store slice instead of only seeing broadcast unit decisions.
- Trigger trace/writeback semantics have also been tightened so later trigger stages no longer erase an already-selected `packet.decisions_store` merely by returning `None`; a trigger with no new store-side selection now preserves the existing store selection artifact, while a trigger that does produce a new selection still overwrites it. This matters for write-side `memory_pressure` triggers because their layer-wide selection must survive through the rest of the ingest pipeline.
- Trigger selection now has an explicit parallel store-side artifact as well: `Packet` includes `decisions_store`, `BoundaryEventTrigger` can map `session/turn/chunk/subgoal/episode` boundary hits onto existing store records with matching `*_id` metadata, and `RuntimeEventTrigger` can mark all records in the affected layer for `memory_pressure` events. This is intended as groundwork for later evolution/organization selectors without breaking the existing `packet.decisions` contract.
- The trigger surface now also includes a store-wide selector primitive: `StoreAllTrigger` can write every non-empty layer's full record set into `packet.decisions_store` while preserving the current `packet.decisions` value (including `None`). This makes it possible to compose "gate with one trigger, select full store with another" without overloading the gating mask.
- A first store-targeted hierarchical abstraction family has now been added on top of that selector path: `HierarchicalOrganization` and `HierarchicalEvolution` can consume `packet.decisions_store`, group selected source-layer records, and package higher-level provenance-bearing observations under `metadata["hierarchical"]`. Their persistence path is no longer hard-coded direct append; they now write through either a default child `MemoryPipeline.ingest()` (with `AppendOrganization(target_layer=...)`) or a caller-provided child pipeline that reuses the current store. This is the first baseline family that intentionally treats `decisions_store` as an execution-time store selector rather than only a trigger trace artifact.
- There is now a concrete end-to-end demonstration of one practical hierarchical-memory pattern instead of only primitive-level unit tests: a `session_end`-triggered pipeline can write raw dialogue turns into an `episodic` layer, use `HierarchicalEvolution` to generate one summary per `session_id` into a `session_summary` layer, and recall across both layers with `LayerAwareRetrieval` while keeping a final global embedding-similarity `top_k`.
- A follow-up coverage note now compares that executable trigger surface against the 40-paper mappings in `TRIGGERS.md`: the current code is strong on thin trigger semantics (`always`, threshold, boolean/failure events, periodic, idle, session-boundary) but still cannot natively express route-aware / multi-label trigger outputs, and only partially covers true capacity-native and subgoal-completion-heavy papers.
- The Reflexion-style maintenance path now has explicit end-to-end pipeline coverage in `tests/test_pipeline.py`: `ModelJudgeTrigger(slot="evolution_trigger")` can be wired through `ReflectionGenerationEvolution`, `BufferRetrieval`, and `PromptContextReadout`, and tests now cover both the positive path (reflection appended and recalled) and the blocked path (no reflection written when the judge rejects it).
- The runtime migration toward `openai-agents` has now started in executable code: shared LLM/runtime access is no longer centered on raw `openai` chat-completions calls, and the MemGPT classic loop now uses real `openai-agents` function tools plus `Agent + Runner`.
- Representation-time triple extraction is no longer heuristic-only: a dedicated `TripleRepresentation` now owns triple extraction with real LLM-backed direct and two-stage modes, and graph-style pipelines have been migrated away from `BasicRepresentation(..., "triple", ...)`.
- A fresh architecture review surfaced a likely trigger-boundary problem in the current ingest/evolution path: runtime trigger modules currently output only `packet.decisions` aligned to incoming `packet.units`, so the same mask is implicitly serving both as "should this stage fire?" and "which items are selected for the operation". That coupling is acceptable for simple append/no-op paths, but it mismatches the DSL's own `MemoryEvolution = selection + action + effect + trigger` decomposition and will become a design constraint for maintenance/evolution families whose real targets are existing store records, windows, or layer slices rather than the incoming units themselves.

## Best Current Reading

The project is currently in a "framework expansion" phase:

- the primitive runtime foundation is mostly in place
- documentation and examples are in decent shape
- the main pressure is now broader literature coverage and missing-module discovery
- full DSL execution, search, benchmark orchestration, and motif mining are still later-stage work

In short: the core framework is substantially built, but the research pipeline around it is not finished.

## What Seems Largely Done

These areas should generally be treated as established unless new work changes them materially:

- core slot-based runtime and `MemoryPipeline` direction
- topology/store contract checking as a first layer of legality validation
- baseline module ecosystem for composition experiments
- Chinese documentation pass that better reflects executable code
- reusable baseline/runtime support as a meaningful prototype layer rather than pure design prose

Avoid re-documenting these in detail unless there is a major architectural change.

## Main Open Gaps

### 1. DSL is still more documented than executable

There is still no fully realized end-to-end DSL path that cleanly goes from declarative config to validated executable pipeline and back.

### 2. Search-space formalization is incomplete

The repo has the ingredients for constrained search, but not a finished searchable representation of module coupling, legality, bundles, and topology-aware constraints.

### 3. Literature re-expression coverage is incomplete

This is the most immediate research-facing gap. Current decomposition work suggests:

- some papers are already close to the runtime
- others expose missing primitive families
- broader claims about coverage will require targeted module additions rather than only better prose
- trigger complexity should be treated carefully during this work: current survey notes suggest simplification is likely low-risk as long as the runtime still covers a few high-value motifs such as failure-triggered reflection, capacity/batch maintenance, subgoal completion, and type-routing writes
  - update: the baseline runtime has since been deliberately narrowed further than that older note implied; richer trigger motifs remain relevant as literature-analysis concepts, but they are no longer public baseline trigger primitives today

### 4. Evaluation / benchmark infrastructure is incomplete

There are tests and demos, but not yet a finished research benchmark harness for systematic comparison, logging, and analysis across configurations.

### 5. Motif discovery remains future work

The long-term "discover recurring memory motifs" goal depends on search + evaluation infrastructure that does not yet fully exist.

## Literature Coverage Snapshot

Current decomposition work suggests several useful boundary papers:

- **Mem0**: relatively close; likely reachable with targeted maintenance-oriented modules
- **HippoRAG**: only partially mappable; highlights graph retrieval / propagation gaps
- **AriGraph**: only partially mappable; highlights semantic + episodic graph maintenance gaps
- **HiAgent**: only partially mappable; highlights hierarchical working-memory management gaps
- **LightMem**: only partially mappable; highlights staged/offline memory maintenance gaps
- Trigger survey status:
  - `TRIGGERS.md` now records a candidate-pool pass over trigger heterogeneity
  - it no longer stops at coarse labels; it now includes per-paper prototype/signal mappings and more detailed trigger summaries
  - provisional conclusion: only a small minority of papers are truly trigger-centric
  - practical implication: future trigger refactors should bias toward simplification unless a paper specifically depends on richer trigger semantics
  - more specifically, only a handful of surveyed papers clearly require explicit periodic / session-end / idle semantics; many other "offline" cases are better modeled outside the trigger layer
- evolution boundary note:
  - current code still couples trigger output to per-input-unit boolean masks via `packet.decisions`
  - this is probably too narrow for faithful `selection` modeling in maintenance/evolution work such as `layer_slice`, `time_window`, `matched_by_entity`, `low_activity`, or other store-targeted selectors
  - a likely next refactor is to keep a lightweight trigger activation result while introducing a separate selection artifact for operation targets, instead of overloading trigger decisions to mean both activation and candidate choice
- Organization/evolution survey status:
  - `ORGANIZATION_EVOLUTION_SURVEY.zh-CN.md` now provides a parallel quick-pass over the same 40 papers focused only on memory organization and memory evolution
  - early pattern: the densest cross-paper variation appears in hierarchical store structure, event-centric unit formation, graph relinking, profile/event dual-track memory, consolidation, and cross-store migration
  - practical implication: missing-module discovery should likely prioritize organization/evolution families at least as much as richer trigger semantics

Practical interpretation: the framework is expressive enough to analyze these systems, but not yet broad enough to claim faithful coverage of all of them.

## Recommended Next Focus

If continuing the current roadmap, the highest-value next steps are:

1. add missing module families that unlock more paper-faithful re-expression
2. make hidden coupling and metadata contracts more explicit and machine-readable
3. build a concrete DSL/config-to-runtime bridge
4. only then push harder on general search/evaluation infrastructure

## Repository Notes

- The repo may contain unrelated in-progress changes; do not overwrite them casually.
- Full test runs can be slow. Prefer targeted tests for the area you changed.
- `openai-agents` is now an active dependency and should be treated as part of the runtime direction rather than a speculative future option.
- Trigger status note:
  - public baseline trigger API is no longer only the three constant baselines; it now includes code-shaped richer trigger classes on the same unified surface
  - the previous `OnInputTrigger` alias has been deleted because its default behavior duplicated `AlwaysTrigger`
  - trigger-family infrastructure and compose helpers remain removed
  - baseline trigger design should still be treated as the current source of truth
  - the implemented direction matches the rewrite plan: `packet.decisions` stays the stable downstream contract, write-stage decisions remain in `trace["write_trigger"]`, and richer trigger families are additive slot implementations rather than a return to the removed heavy trigger stack
  - `StoreAllTrigger` is now part of that additive surface as a selector-only trigger: it fills `packet.decisions_store` for all non-empty layers but intentionally preserves the incoming `packet.decisions` instead of recomputing it
  - dispatch semantics remain unchanged: if `StoreAllTrigger` runs inside `DispatchWriteTrigger` / `DispatchEvolutionTrigger` on a non-primary branch, its `decisions_store` stays only in branch trace and is not merged back into the returned packet; docs now warn users to prefer primary placement or serial trigger lists
  - `memory_pressure` is no longer only a documentation-level trigger example: current code now computes it from store usage against per-layer `record_budget` / `token_budget`, writes both dimensions plus the aggregated pressure value into trigger trace output, requires explicit `target_layer` on write-side scalar pressure checks, and defaults to placement-derived layer resolution on evolution-side pressure checks
- Current migration state:
  - `memprimitive/utils/_runtime.py` now routes text / JSON / summarization / reranking calls through `openai-agents`
  - primitive-layer LLM calls have been further unified: `memprimitive/baselines/representation.py` no longer creates a raw `OpenAI(...)` client for `summary` / `description`, and now uses the shared runtime path as well
  - triple extraction has been split out of `BasicRepresentation` into dedicated `TripleRepresentation`, with strict structured outputs and direct / two-stage extraction modes
  - the legacy `classics` examples and `classic_modules` wrappers have been removed so the repo surface now centers on baseline primitives plus demonstration examples

## TODO

- Replace remaining heuristic implementations with more realistic model-backed methods where appropriate.
- Improve graph edge linking / graph construction quality.
