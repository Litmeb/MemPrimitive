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
- A follow-up pass on `subgoal-completion conditioned` made the category stricter as well: after re-checking the previously listed papers, `HiAgent` remains the clearest true fit, while several others were better reclassified as session-boundary or event-boundary triggers instead of subgoal completion.
- The follow-up trigger note has now been rewritten again at a more implementation-oriented level: `TRIGGER_SCORE_BOOLEAN_GATE_SURVEY.md` no longer primarily groups triggers by semantic labels like `threshold(score)` or `boolean gate`, but instead by concrete trigger implementation families for write/evolution (`PassThroughHook`, `StructuralBoundaryHook`, `RuntimeCallback`, `ExplicitScalarRule`, `LLMJudge`, `BackgroundScheduler`, `ControllerOrchestrator`) across the same 40-paper corpus.
- A code-aligned trigger rewrite plan now exists in `TRIGGER_REWRITE_IMPLEMENTATION_PLAN.md`: it maps the literature-backed trigger families onto the current `write_trigger` / `evolution_trigger` slot API, recommends slot-specific concrete classes plus shared helpers, and explicitly records the main implementation constraint that true periodic/idle background maintenance will likely require a new maintenance entrypoint rather than only ordinary ingest.
- The public baseline trigger surface has now been intentionally simplified again: the old trigger-family decomposition (`signal / scorer / gate / policy`) and compose-style trigger builders have been removed from the baseline API, and the repo now only exposes basic slot triggers (`AlwaysWriteTrigger`, `ThresholdWriteTrigger`, `NeverEvolutionTrigger`, `ThresholdEvolutionTrigger`) on that layer.
- The runtime migration toward `openai-agents` has now started in executable code: shared LLM/runtime access is no longer centered on raw `openai` chat-completions calls, and the MemGPT classic loop now uses real `openai-agents` function tools plus `Agent + Runner`.
- Representation-time triple extraction is no longer heuristic-only: a dedicated `TripleRepresentation` now owns triple extraction with real LLM-backed direct and two-stage modes, and graph-style pipelines have been migrated away from `BasicRepresentation(..., "triple", ...)`.

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
  - public baseline trigger API is now minimal
  - trigger-family infrastructure and compose helpers have been removed
  - baseline trigger design should be treated as the current source of truth
  - forward trigger expansion should remain code-shaped rather than ontology-heavy: the new plan keeps `packet.decisions` / `packet.evolution_decisions` as the stable downstream contract and treats richer trigger families as additive slot implementations, not a return to the removed heavy trigger stack
- Current migration state:
  - `memprimitive/utils/_runtime.py` now routes text / JSON / summarization / reranking calls through `openai-agents`
  - primitive-layer LLM calls have been further unified: `memprimitive/baselines/representation.py` no longer creates a raw `OpenAI(...)` client for `summary` / `description`, and now uses the shared runtime path as well
  - triple extraction has been split out of `BasicRepresentation` into dedicated `TripleRepresentation`, with strict structured outputs and direct / two-stage extraction modes
  - the legacy `classics` examples and `classic_modules` wrappers have been removed so the repo surface now centers on baseline primitives plus demonstration examples

## TODO

- Replace remaining heuristic implementations with more realistic model-backed methods where appropriate.
- Improve graph edge linking / graph construction quality.
