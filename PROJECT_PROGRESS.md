# MemPrimitive Project Progress

Shared long-lived status note for future agents. Keep this file short, current, and decision-oriented. Detailed benchmark history, classics coverage, and broad survey notes now live in [PROJECT_BROAD_STATUS_ARCHIVE.md](PROJECT_BROAD_STATUS_ARCHIVE.md).

## Current Focus

The next project phase is centered on **memory search and memory evolution**.

The framework already has enough basic primitives, examples, and benchmark wiring to stop treating paper reconstruction as the main workstream. The useful next step is to make memory systems searchable, evolvable, and comparable as composed mechanisms:

- search over valid memory-pipeline designs rather than hand-picking paper-shaped examples
- expose retrieval, provenance, and readout behavior clearly enough for automated comparison
- make evolution/maintenance modules composable under explicit trigger, store, topology, and metadata contracts
- use LoCoMo/LongMemEval-style runs as feedback for mechanism search, not just as one-off reproductions

## Working Baseline

Treat these as the current foundation:

- `MemoryPipeline` is the strict validated public pipeline surface.
- `FreeMemoryPipeline` is the permissive ordered-runner surface for experiments.
- Baseline modules are the source of truth for framework capability; examples should compose them instead of hiding behavior in paper wrappers.
- `DSL_SEMANTIC_OPERATION_MAP.zh-CN.md` is the current semantic map for splitting code-level DSL modules into search-ready memory design moves, including hard legality constraints vs soft search variables.
- `DSL_SEMANTIC_OPERATION_IDEA_LIST.zh-CN.md` is the concise prioritized list of small semantic operations worth trying first in retrieval/evolution/search experiments.
- `DSL_SEMANTIC_OPERATION_DESIGN_AGENT_IDEAS.zh-CN.md` is the design-agent inspiration list that separates small modification cues from insertable pipeline components.
- `memprimitive/example/classics/memmachine_memory.py` is the strongest current reference for a search/evolution-heavy memory composition: working memory, STM consolidation, raw episodic LTM, sentence-derived indexing, parent/temporal expansion, cluster rerank, timestamped readout, and profile tools.
- LoCoMo adapter work established the preferred generic memory-system boundary: `build_system`, `ingest_event`, and `recall`, loadable through `--memory-adapter binding --memory-binding module:create_memory_binding`.
- Real runtime paths matter. Use `Runtime.embed()` and the dedicated `Runtime.rerank()` / `MEMPRIMITIVE_RERANK_*` path for retrieval work that depends on model behavior.
- WSL repo Python commands should use `~/bin/winpy312`, not bare `python`, `python3`, or `conda run`.

## Main Workstream

1. **Retrieval/search surface**
   - Make retrieved records, parent records, temporal neighbors, reranked candidates, and readout sections preserve stable provenance.
   - Move from flat `RetrievedSet`-only thinking toward grouped/provenance-aware recall views when a memory system naturally has layers or expansion steps.
   - Keep MemMachine-style contextualized retrieval as the immediate proving ground: direct episodic hits, sentence-hit parent expansion, temporal expansion, and rerank should be inspectable as separate stages.

2. **Evolution/maintenance surface**
   - Clarify which evolution modules mutate content, metadata, topology, or derived indexes.
   - Make trigger conditions and evolution side effects explicit enough for automated composition checks.
   - Prefer reusable evolution primitives over paper-specific helper glue when the behavior recurs: STM consolidation, profile feature updates, conflict/invalidation, summarization, link strengthening, and graph/entity maintenance.

3. **Search-space formalization**
   - Define machine-readable module compatibility: accepted stores, record metadata requirements, topology assumptions, trigger outputs, and representation dependencies.
   - Separate hard legality constraints from soft search preferences.
   - Capture common bundles only when the coupling is real, for example representation plus matching retrieval/index maintenance.

4. **Evaluation feedback loop**
   - Use benchmark artifacts to diagnose mechanism failures first: empty recall, lost source ids, malformed metadata, wrong runtime shape, or missing temporal evidence.
   - Keep LoCoMo and LongMemEval as the first search-feedback targets.
   - Treat scoring/experiment tracking as supporting infrastructure for mechanism search, not as the main project goal by itself.

## Near-Term Checkpoints

- Turn the semantic operation map into a small machine-readable compatibility schema when the next search/evolution implementation pass starts.
- Audit current retrieval and evolution modules for missing metadata/provenance declarations.
- Add the smallest compatibility schema needed to describe module legality and coupling.
- Run focused regressions around MemMachine retrieval/evolution behavior before broader benchmark reruns.
- When benchmark behavior looks wrong, inspect output JSONL/metrics before changing prompts.

## Boundaries

- Do not add speculative framework layers before the concrete retrieval/evolution contracts are clear.
- Do not hide retrieval or evolution behavior inside classics wrappers when it can be expressed by baseline modules.
- Do not replace real retrieval/rerank/model behavior with heuristic simulations unless the test is explicitly scoped to pure logic.
- Do not treat temporal/session neighbors as graph neighbors; preserve mechanism boundaries literally.
- Do not compare single-user or smoke-test artifacts directly with full paper scores.

## Deferred Work

The following remain useful but are no longer the active project center:

- expanding the classics catalog for its own sake
- strict reproduction of every paper/repo detail
- broad benchmark harness polish not tied to search/evolution diagnosis
- motif discovery over the full design space
- MSC/DMR runner cleanup unless needed for the search/evolution loop

See [PROJECT_BROAD_STATUS_ARCHIVE.md](PROJECT_BROAD_STATUS_ARCHIVE.md) for the broader status snapshot that used to live here.
