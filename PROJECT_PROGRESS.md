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
- LoCoMo benchmark scheduling now parallelizes by `memory_adapter.session_key(sample=...)` user group: each worker creates and loads one memory session, answers/scores that user's QA in stable order, and final predictions are re-sorted to the original benchmark sample order.
- `memprimitive.benchmarking.minimal_baseline` writes timestamped default outputs under `benchmarks/outputs` using dataset, memory adapter, smoke/full mode, user filter, and timestamp fields; explicit `--output` paths still override this.
- LoCoMo runs can cap the **answer** LLM request with `--llm-max-input-tokens N` (requires `--benchmark locomo`): prompts are trimmed to fit N tokens (tiktoken when installed) while keeping the final `Question:` / `Answer:` block for Mem0/MemMachine templates and the user request for minimal single-recall LoCoMo.
- Shared-conversation LoCoMo classics baselines now cover both `memmachine` and `amem`; A-MEM is wired through the same benchmark-facing boundary while keeping its graph-note write/evolution mechanism unchanged.
- `memprimitive.evolution.search` is the automated search/evolution harness: an orchestrator proposes allowed-file candidate mutations, isolated git worktrees run Codex workers, staged checks enforce the whitelist, and full `minimal_baseline` runs plus scoring artifacts feed the next round (evolution benchmarks no longer pass `--smoke-test`).
- `memprimitive.evolution.search` now defaults its Codex models to `deepseek-v4-pro` for the orchestrator and `deepseek-v4-flash` for workers; explicit CLI overrides still take precedence.
- DeepSeek-backed search currently runs through WSL Codex using the `deepseek` Codex profile and a local Responses-compatible shim at `http://127.0.0.1:8765/v1`; the harness loads repo env defaults and marks WSL inheritance so `DEEPSEEK_API_KEY` reaches WSL Codex without putting secrets on command lines.
- Search harness worktrees now use short hash slugs to avoid Windows path-length failures, normalize common hallucinated MemMachine test paths to `tests/test_classics_memmachine.py`, fall back to the real MemMachine regression when generated `-k` selectors select no tests, use the control repo's absolute benchmark data root, and normalize CLI-style candidate benchmark args before passing binding kwargs.
- `memprimitive.evolution.search` can now resume an existing run's benchmark stage with `--resume-benchmark-only --run-id <existing-run-id>`: it reloads `proposals.jsonl`, reuses saved candidate worktrees, reruns only candidates that previously failed at `failed_stage="benchmark"`, respects `--max-parallel-candidates` during that benchmark retry pass, and rewrites the round/final leaderboards without regenerating candidates.
- The latest DeepSeek search validation reached Codex worker, static checks, focused tests, LoCoMo benchmark invocation, and recovered benchmark scoring through the resume path; remaining failures should be interpreted from candidate artifacts rather than as search/Codex plumbing failures.
- A 2026-05-08 resume of run `20260507_160843_improve-memmachine-locomo-recall-provenance-and-` restored the final leaderboard with `--resume-benchmark-only --max-parallel-candidates 4` after a successful small `Runtime.rerank()` health check. Candidates `c1`, `c2`, and `c4` reran in parallel and passed LoCoMo smoke/local scoring with `prediction_count=10`, `empty_recall_rate=0.0`, and `source_id_coverage=1.0`; `c1` and `c4` tied at overall F1 `0.1899` / BLEU `0.1606`, `c2` scored F1 `0.1773` / BLEU `0.1536`, and `c3` remains statically rejected for changing `memprimitive/baselines/retrieval.py` outside its allowed file list.
- MemMachine benchmark runs can raise the profile write agent turn budget with `--memmachine-profile-max-turns` when real tool-call evolution needs more than the default 6 turns.
- MemMachine profile evolution now keeps its write-agent prompt compact for 4k-context local models by omitting full metadata dumps and truncating the selected episode text in the prompt; raw episodes remain fully stored for retrieval/readout.
- LongMemEval can now use the same CLI memory adapter names through a generic one-binding-per-sample adapter, while LoCoMo keeps its speaker-specific adapter behavior.
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
- Run the new harness first in dry-run mode, then with explicit `--base-ref HEAD --allow-dirty-control-worktree` while this checkout remains dirty, so candidate worktrees start from a committed ref rather than the control worktree's local edits.
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
