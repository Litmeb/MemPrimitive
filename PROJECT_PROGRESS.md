# MemPrimitive Project Progress

## Purpose

Shared long-lived status note for future agents. Keep it short, current, and focused on what still matters.

## Project Goal

`MemPrimitive` aims to express agent memory systems as composable primitives instead of paper-specific pipelines. The intended end state is:

- a unified primitive ontology / DSL
- an executable runtime centered on `MemoryPipeline`
- re-expression of representative memory papers inside one framework
- explicit composition constraints for later search and evaluation
- eventual motif discovery over the design space

## Current Position

The project is past the concept stage and now in a "framework expansion" phase.

- The core runtime is real and usable: slot-based composition, store/topology contracts, baseline modules, and a shared LLM runtime path already exist.
- Documentation is in much better shape and is broadly aligned with the executable surface.
- The repository has shifted from design prose to reusable baseline modules plus executable paper-style examples.
- Literature work has broadened beyond a few showcase systems and is now being used to judge what is already expressible, what is only approximate, and what still needs new primitive families.

## What To Treat As Established

Unless there is a major architectural change, these areas are mostly settled:

- `MemoryPipeline` as the strict validated public pipeline surface
- `FreeMemoryPipeline` as the permissive experimental ordered runner
- baseline modules as the main way to build systems
- `openai-agents` as the active runtime direction
- Chinese docs as useful, code-aligned reference material rather than placeholder notes

Avoid re-documenting these in detail unless something materially changes.

## Important Recent Progress

- Retrieval, prompt, readout, graph, and tool-calling surfaces are now broad enough to build nontrivial paper-style reconstructions from shared primitives rather than ad hoc wrappers.
- PromptPlan-driven tool visibility is now implemented for `LLMFunctionCallOrganization` / `LLMFunctionCallEvolution`: prompt-side recall branches can report retrieved record provenance, select which recall branches contribute to `visible_records`, and expose that visibility in trace metadata.
- PromptPlan labeled sub-recall now degrades cleanly on per-label empty recall-query overrides: an override that renders to `""` records `disabled_reason="empty_rendered_recall_query"` in prompt/readout metadata instead of failing `Query(text=...)` validation.
- Integration smoke coverage for real LLM / embedding baselines exists in `tests/test_smoke_real_model_modules.py` and can exercise the main LLM-backed baseline families when runtime credentials are configured.
- Embedding first-stage downshift is now implemented: ordinary record-level text embeddings can be declared per layer in `StoreLayerSpec.settings["embedding"]`, and `MemoryStore.append()` / `replace_record()` now auto-generate or refresh `record.embedding` for those layers.
- The implemented policy boundary is intentionally narrow: stage 1 only handles `mode="text"` with refresh on semantic text change. Entity embeddings, note-payload-derived embeddings, query embeddings, and the broader `UNIT_EMBEDDING_CONTRACT` redesign are still deliberately left in their existing specialized paths.
- Follow-up simplification landed in the expected moderate form: responsibility is more centralized and Mem0/tool-path manual embedding logic is smaller, but the harder embedding complexity still remains in graph/entity/note/query-specialized paths.
- `ConfigurableEmbeddingRepresentation` is now the generic text-configurable embedding representation primitive: render configurable text (including template-based text) from the current unit, embed that text, and record embedding-input provenance in `metadata["representation"]` / trace without rewriting the unit's main text-facing fields.
- `memprimitive/example/classics` now contains executable reconstructions rather than an empty placeholder. The most important current examples are:
  - COMEDY / compressive-memory style hierarchical maintenance
  - A-MEM / Agentic Memory
  - Mem0
  - Mem0g
  - RET-LLM / MemLLM-style triple memory
- Trigger literature review suggests trigger diversity is real but not the main source of cross-paper variation. The bigger long-term differences are in representation, organization, maintenance/evolution, and retrieval.

## Main Open Gaps

### 1. DSL is still more documented than executable

There is still no clean end-to-end path from declarative config to validated runnable pipeline and back.

### 2. Search-space formalization is incomplete

The repo has many ingredients for constrained search, but not yet a finished machine-readable representation of coupling, legality, bundles, and topology-aware constraints.

### 3. Faithful paper coverage is still incomplete

The framework is expressive enough for mechanism-level reconstructions of several systems, but not broad enough to claim faithful coverage yet. The main pressure is no longer "write more prose"; it is adding or refining the module families that close concrete behavioral gaps.

### 4. Evaluation infrastructure is incomplete

There are targeted tests and runnable demos, but not yet a benchmark harness for systematic comparison, logging, and analysis across configurations.

### 5. Motif discovery remains later-stage work

The eventual "discover recurring memory motifs" goal depends on the DSL bridge plus search/evaluation infrastructure that still does not fully exist.

## Coverage Snapshot

### Mem0 Family

- `mem0_memory.py` is reasonably close at the mechanism level.
- PromptPlan-controlled candidate visibility now closes the previous major update-scope mismatch: the Mem0 profile-update tools can be restricted to recalled profile candidates instead of the whole `profile` layer.
- `mem0g_memory.py` has moved closer to upstream structure by keeping both a profile/vector branch and a graph branch, but it is still only partially aligned.
- Mem0g now uses the same prompt-controlled visible-domain path for both its profile/vector tools and graph-maintenance tools.
- The biggest remaining Mem0g gaps are graph-native storage semantics, relation-level invalidation/update behavior, and full repo-style recall behavior.

### RET-LLM / MemLLM

- `ret_llm_memory.py` remains a mechanism-level reconstruction, not a faithful RET-LLM reproduction.
- The `MEM_READ` path is now closer to the paper than before: it scans the full `triple_memory` layer and returns all matching triplets instead of truncating to top-k retrieved records.
- The main remaining RET-LLM gaps are now clearer: there is still no paper-style unified controller with one fine-tuned model producing both `MEM_WRITE` and `MEM_READ` calls from raw natural-language input.
- The current memory is also structurally different from the paper: it stores graph-shaped `MemoryRecord`s with entity-level dedup/merge and retrieves top-k records, whereas RET-LLM describes a triplet-table memory that returns all matching triplets for a one- or two-slot query after exact-or-term-substitution lookup.
- Temporal/update behavior is still only approximate: the example lacks an explicit policy for handling conflicting time-varying facts, so it should not be treated as aligned to the paper's temporal QA behavior.

### A-MEM / Agentic Memory

- Paper + upstream repo review now suggests current baseline modules are sufficient for an A-MEM reconstruction without adding new primitives, as long as alignment follows the easier repo-consistent path rather than the paper's most ambitious wording.
- The clean mapping is now clearer:
  - note construction -> multiple `LLMRepresentation` fields (`context` / `keywords` / `tags` / `category` / `attributes`) + `ConfigurableEmbeddingRepresentation`
  - append note into graph memory -> `GraphAppendOrganization`
  - post-write note evolution -> bounded `LLMFunctionCallEvolution` with A-MEM-specific tools
  - retrieval -> `VectorGraphSeedAndExpandRetrieval`
- The main remaining work is example-level wiring and prompt/readout shaping, not baseline-family coverage.
- That example-level wiring now exists in `memprimitive/example/classics/amem_memory.py`, so A-MEM should no longer be treated only as a capability hypothesis. The current status is: executable mechanism-level reconstruction exists; remaining gaps are fidelity/prompt tuning questions rather than missing framework coverage.
- The old dedicated A-MEM note-construction representation has now been removed. The intended behavior is expressed by composing generic `LLMRepresentation` modules with `ConfigurableEmbeddingRepresentation`, rather than by preserving a special unit-level note-payload contract.
- The old A-MEM-specialized `GraphAppendLinkReadyOrganization` has now been removed. Its intended write role is folded back into `GraphAppendOrganization`, which is now the single baseline graph-append primitive used for both ordinary graph records and note-graph/A-MEM-style writes.
- Important paper/repo mismatch: the paper text says evolved neighbors may update context, keywords, and tags, but the released repos mostly implement context/tag updates only. The current framework matches the repo-side behavior more naturally; keyword rewrite should be treated as optional fidelity stretch, not a blocker.
- Retrieval alignment is also better with the repo interpretation: seed by embedding similarity, then expand by stored links / neighbors. Query keyword generation can be expressed with existing retrieval-query rewrite machinery rather than a new primitive.
- A new concrete design direction is now documented in `AMEM_FUNCTION_CALL_EVOLUTION_CONTRACT.md`: the current two-step A-MEM evolution can be collapsed into one `LLMFunctionCallEvolution` if execution uses a hard visible-record boundary plus two A-MEM-specific tools, one for current-record link strengthening and one for neighbor-only context/tag updates. This path is now the preferred repo-consistent refactor target.
- That refactor target is now implemented in the executable A-MEM path. `memprimitive/example/classics/amem_memory.py` now uses one bounded `LLMFunctionCallEvolution` with two A-MEM-specific tools:
  - `AMEM_STRENGTHEN_LINKS` for current-note `graph.links` plus optional current-note `tags`
  - `AMEM_UPDATE_NEIGHBOR` for neighbor-note `context` / `tags` only
- The old standalone A-MEM evolution baselines have now been removed from the baseline surface, tests, and docs so the function-call path is the only supported implementation route.
- Important repo-consistency correction: A-MEM evolution now treats field mutation asymmetrically, matching the released implementation rather than the older baseline rewrite path:
  - new/current note: update `links`, `tags`
  - neighbor note: update `context`, `tags`
  - neighbor evolution no longer rewrites `content`, `keywords`, or embeddings
- The A-MEM function-call evolution visible-set path is now back on embedding-similarity recall rather than a temporary recency fallback. Prompt-plan sub-recall can now pass a full `Query` object (including embedding) into child retrieval pipelines, so evolution-side candidate visibility can reuse the current note embedding directly.
- Backward-compatibility is preserved at the helper boundary: prompt/sub-recall utilities now accept full `Query` objects for embedding-aware child retrieval, but the old `query_text=...` call shape still works unchanged for existing callers such as mid-decoding memory-read tools.
- `LLMFunctionCallEvolution` itself is also now usable for this ingest-time/current-record pattern because it can select evolution targets from aligned `packet.units` / `packet.placements` / `packet.decisions`, not only from `decisions_store` or whole-layer scans.

### Reflexion

- Paper + upstream repo review now suggests the current framework can reproduce the Reflexion memory module without adding new primitives, as long as scope stays limited to memory and prompt-context injection rather than the full agent loop.
- The easiest alignment target is the released repo, which implements memory more simply than the paper's broad framing: failed trials generate short natural-language reflections, those reflections are stored in an episodic text buffer, and later trials prepend either the last trial, the reflection buffer, or both.
- The implementation path is now more general than the earlier dedicated-helper sketch:
  - raw failed-trial persistence -> `AppendOrganization` into `trial_buffer`
  - reflection extraction from newly written failed trajectories -> `HierarchicalEvolution(extract_mode="generate", selection_mode="latest_active_units")`
  - bounded episodic reflection buffer read -> `BufferRetrieval`
  - prompt construction for `base` / `last_trial` / `reflexion` / `last_trial_and_reflexion` -> `PromptContextReadout`
- `HierarchicalEvolution` now has two generic extensions that make this clean:
  - `record_text_field` lets generated payloads keep structured fields while using one field such as `reflection` as the stored record text
  - `retention_size` prunes the target layer to a bounded recency window, matching the repo-style small reflection buffer
- Important paper/repo mismatch: the paper describes a generic verbal reinforcement framework with episodic memory, but the public repo implementations are mostly plain append-only text memory plus prompt templating, sometimes with a recent-3 truncation rule. The current framework matches this repo-side interpretation better than a more ambitious generalized-learning reading.
- The memory-only orchestration now exists in `memprimitive/example/classics/reflexion_memory.py` and is covered by deterministic tests in `tests/test_classics_reflexion.py`.
- New primitive work is only needed if future goals expand beyond the repo-consistent Reflexion memory slice, such as trial-indexed multi-task memory partitioning, richer evaluator provenance, or tighter coupling between evaluation outcome and recall selection.
- Paper-fidelity gaps are now clearer for the memory slice:
  - reflection generation now conditions on prior retained reflections via prompt-side sub-recall, so the earlier "latest failed trial only" mismatch is closed without adding a Reflexion-specific primitive
  - the helper/default path now prefers an explicit full `trial_trace` for short-term memory and only falls back to `last_attempt`, so the earlier "single compressed attempt string only" mismatch is also closed at the example level
  - recall is still a pure recency window over one shared reflection buffer, so the example still lacks the paper's task-local trial loop semantics and can mix reflections across unrelated tasks unless the caller isolates stores manually; this is currently treated as an intentional research-prototype simplification rather than a primitive gap

### Other Boundary Papers

- `HippoRAG`: still highlights graph retrieval / propagation gaps.
- `AriGraph`: still highlights semantic + episodic graph maintenance gaps.
- `HiAgent`: still highlights hierarchical working-memory management gaps.
- `LightMem`: still highlights staged/offline maintenance gaps.

## Architecture Boundaries To Remember

- Trigger outputs are still easier to align to incoming units than to arbitrary store-side candidate subsets. This remains a real limitation for faithful maintenance/evolution modeling.
- Retrieval output is still mostly a flat `RetrievedSet`; richer grouped or provenance-aware recall views remain underdeveloped.
- Trigger work should be kept in proportion: literature review suggests triggers matter, but they are usually not the highest-leverage missing piece.

## Recommended Next Focus

If continuing the current roadmap, the highest-value next steps are:

1. add missing module families or selection mechanics that unlock more faithful paper re-expression
2. make hidden metadata and coupling contracts more explicit and machine-readable
3. build a concrete DSL/config-to-runtime bridge
4. push harder on search/evaluation only after the above is in better shape

## Repository Notes

- The repo may contain unrelated in-progress changes; do not overwrite them casually.
- Full test runs can be slow; prefer targeted tests for the area you changed.
- The baseline layer, not paper-specific wrappers, should remain the source of truth for framework capability.

## TODO

- Replace remaining heuristic paths with realistic model-backed methods where appropriate.
- Improve graph edge linking / graph construction quality.
