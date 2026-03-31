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

- The stage-1 runtime is real and usable: slot-based composition, topology/store contracts, baseline modules, and classic-family examples all exist.
- Chinese-facing documentation has been refreshed and is now much closer to the actual runtime surface.
- Several classic or near-classic families already have runnable support or decomposition work, including Reflexion, MemGPT, and A-MEM-like paths.
- Literature coverage work is active. The repo is no longer focused only on a few showcase papers; it is trying to scale toward a broader set of memory papers and judge which are fully re-expressible versus only partially mappable.
- The runtime migration toward `openai-agents` has now started in executable code: shared LLM/runtime access is no longer centered on raw `openai` chat-completions calls, and the MemGPT classic loop now uses real `openai-agents` function tools plus `Agent + Runner`.

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
- classic-family support as a meaningful prototype layer rather than pure design prose

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
- Current migration state:
  - `memprimitive/utils/_runtime.py` now routes text / JSON / summarization / reranking calls through `openai-agents`
  - `memprimitive/example/classics/memgpt.py` no longer hand-rolls tool-call JSON parsing; it uses `openai-agents` tools and keeps only MemPrimitive-specific memory logic
  - primitive-layer LLM calls have been further unified: `memprimitive/baselines/representation.py` no longer creates a raw `OpenAI(...)` client for `summary` / `description`, and now uses the shared runtime path as well
  - targeted regressions for the migrated surface passed: `tests/test_classic_memgpt.py`, `tests/test_classic_amem.py`, and `tests/test_baselines.py` for a total of 129 passing tests in that focused run

## TODO

- Replace remaining heuristic implementations with more realistic model-backed methods where appropriate.
- Improve representation-time triple extraction.
- Improve graph edge linking / graph construction quality.
