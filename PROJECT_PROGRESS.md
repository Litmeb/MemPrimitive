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
- Organization baselines now also include `FanoutIngestOrganization`, a reusable ingest-time helper that reads an iterable string field from `Observation.metadata` and fans those strings out through a child `MemoryPipeline.ingest(...)` path while aggregating child ingest trace.
- `FanoutIngestOrganization` now also supports reading that iterable string field from `packet.units[0].metadata["representation"]` when the field is not already present on `Observation.metadata`, so representation-driven extraction pipelines can fan out directly into child ingest paths without extra example glue.
- Retrieval baselines now also include `MetadataRetrieval`, a simple metadata-field filter that supports case-insensitive exact match by default plus regex matching, with one-level iterable-member matching for list/tuple/set-style metadata fields.
- The public baseline surface has been intentionally tightened again: `GraphNeighborAppendEvolution`, `BulletListReadout`, `GroupedByLayerReadout`, `GraphEntityAppendOrganization`, `TagRetrieval`, `ConditionalLayerOrganization`, `LineSplitUnitFormation`, `WindowedUnitFormation`, and `MetadataHintUnitFormation` are now removed rather than kept as extra baseline variants.
- The intended replacements are now explicit in code/tests/docs rather than preserved as compatibility aliases:
  - graph neighbor append compatibility -> `GraphLinkEvolution`
  - simple special-case readouts -> `ConcatenateReadout` or `TemplateReadout`
  - per-entity graph writes -> `GraphEntityDeduplicationAppendOrganization` where entity-level fanout/dedup is still needed, otherwise `GraphAppendOrganization`
  - tag-overlap retrieval -> other existing retrieval families such as `KeywordCountRetrieval`, `EntityRetrieval`, `EmbeddingSimilarityRetrieval`, or `LayerAwareRetrieval`
- PromptPlan-driven tool visibility is now implemented for `LLMFunctionCallOrganization` / `LLMFunctionCallEvolution`: prompt-side recall branches can report retrieved record provenance, select which recall branches contribute to `visible_records`, and expose that visibility in trace metadata.
- PromptPlan labeled sub-recall now degrades cleanly on per-label empty recall-query overrides: an override that renders to `""` records `disabled_reason="empty_rendered_recall_query"` in prompt/readout metadata instead of failing `Query(text=...)` validation.
- `DSL_REFERENCE.zh-CN.md` is being reworked into a code-aligned detailed API-style reference organized by slot/module IO rather than compact tables, so future agents can audit module behavior against real packet/store fields more reliably.
- A new concise local skill entry now exists at `.cursor/skills/memprimitive-dsl-brief/SKILL.md` so agents can start from a short MemPrimitive DSL guide and fall back to `DSL_REFERENCE.zh-CN.md` only for exact parameter/edge-case detail.
- That DSL skill now also contains a `references/` split of `DSL_REFERENCE.zh-CN.md` into per-section documents, and `SKILL.md` explicitly routes agents to the relevant section file before they load the full monolithic reference.
- Integration smoke coverage for real LLM / embedding baselines exists in `tests/test_smoke_real_model_modules.py` and can exercise the main LLM-backed baseline families when runtime credentials are configured.
- Embedding first-stage downshift is now implemented: ordinary record-level text embeddings can be declared per layer in `StoreLayerSpec.settings["embedding"]`, and `MemoryStore.append()` / `replace_record()` now auto-generate or refresh `record.embedding` for those layers.
- The implemented policy boundary is intentionally narrow: stage 1 only handles `mode="text"` with refresh on semantic text change. Entity embeddings, note-payload-derived embeddings, query embeddings, and the broader `UNIT_EMBEDDING_CONTRACT` redesign are still deliberately left in their existing specialized paths.
- Follow-up simplification landed in the expected moderate form: responsibility is more centralized and Mem0/tool-path manual embedding logic is smaller, but the harder embedding complexity still remains in graph/entity/note/query-specialized paths.
- `ConfigurableEmbeddingRepresentation` is now the generic text-configurable embedding representation primitive: render configurable text (including template-based text) from the current unit, embed that text, and record embedding-input provenance in `metadata["representation"]` / trace without rewriting the unit's main text-facing fields.
- `LLMRepresentation` now supports structured metadata-backed custom fields with `value_type=list[dict[str, str]]` in addition to `str`, `list[str]`, and `dict[str, str]`, using permissive normalization for JSON object lists.
- TiM thought extraction now uses that structured `LLMRepresentation(field="thoughts", value_type=list[dict[str, str]])` path plus a thin example-level normalization helper, instead of a fully hand-written direct runtime JSON call.
- `memprimitive/example/classics` now contains executable reconstructions rather than an empty placeholder. The most important current examples are:
  - COMEDY / compressive-memory style hierarchical maintenance
  - A-MEM / Agentic Memory
  - Mem0
  - Mem0g
  - RET-LLM / MemLLM-style triple memory
- The old `PlacementWithoutAppendOrganization` baseline has now been removed as redundant. The concrete places that had still been using it right before removal were:
  - `memprimitive/example/classics/mem0_memory.py`
  - `memprimitive/example/classics/mem0g_memory.py`
  - `memprimitive/example/classics/comedy_memory.py`
  These now use `NeverTrigger(slot="write_trigger") + AppendOrganization(...)` to preserve placement emission without ingest-time writes.
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
- The Mem0 profile write path is now organized as extraction-plus-fanout rather than one multi-fact maintenance step: an outer pipeline extracts `fact_list`, then `FanoutIngestOrganization` fans each fact into a child single-fact profile pipeline for add/update/delete decisions. The old helper-side per-fact recall stitching path has been removed.
- `mem0g_memory.py` has moved closer to upstream structure by keeping both a profile/vector branch and a graph branch, but it is still only partially aligned.
- Mem0g now uses the same prompt-controlled visible-domain path for both its profile/vector tools and graph-maintenance tools.
- Mem0g's profile/vector branch now follows the same extraction-plus-fanout organization as Mem0: outer fact extraction via `fact_list`, then per-fact child ingest through `FanoutIngestOrganization` rather than helper-side multi-fact recall assembly.
- Mem0g no longer does a separate example-level pre-pass for contextual graph hint extraction before graph ingest. The graph branch now feeds the dialogue pair directly into `TripleRepresentation`, so graph entity/triple extraction is expressed by the existing baseline representation path rather than by extra helper-side LLM JSON glue.
- Mem0g dual recall is now also expressed as one reusable pipeline surface: `TemplateReadout` drives labeled sub-recall through the existing profile and graph recall pipelines, replacing the old example-level `recall_all()` string-concatenation wrapper.
- Additional classics cleanup landed in the example layer rather than the baseline surface:
  - `tim_simple_memory.py` now caches its candidate recall pipeline and `LLMFunctionCallEvolution` module in the built system instead of rebuilding them for every thought update
  - `tim_memory.py` no longer computes query embeddings / hash buckets redundantly across `build_tim_query()` and `build_tim_recall_pipeline()`
  - `memory_sharing_memory.py` now reuses module-level default prompt constants instead of duplicating the same default judge/readout prompts inline
  - `recurrentgpt_memory.py` still uses repo-style labeled plain-text outputs, but its parsing path is now centralized around one generic labeled-section parser instead of several bespoke substring helpers
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
- The old `ReflectionGenerationEvolution` baseline has now been removed. The last direct test-only usage before removal was in `tests/test_pipeline_triggers_dispatch.py`; supported Reflexion behavior remains on the `HierarchicalEvolution`-based path in `memprimitive/example/classics/reflexion_memory.py`.
- New primitive work is only needed if future goals expand beyond the repo-consistent Reflexion memory slice, such as trial-indexed multi-task memory partitioning, richer evaluator provenance, or tighter coupling between evaluation outcome and recall selection.
- Paper-fidelity gaps are now clearer for the memory slice:
  - reflection generation now conditions on prior retained reflections via prompt-side sub-recall, so the earlier "latest failed trial only" mismatch is closed without adding a Reflexion-specific primitive
  - the helper/default path now prefers an explicit full `trial_trace` for short-term memory and only falls back to `last_attempt`, so the earlier "single compressed attempt string only" mismatch is also closed at the example level
- recall is still a pure recency window over one shared reflection buffer, so the example still lacks the paper's task-local trial loop semantics and can mix reflections across unrelated tasks unless the caller isolates stores manually; this is currently treated as an intentional research-prototype simplification rather than a primitive gap

### RecurrentGPT

- Paper + upstream repo review now suggests the current framework can reproduce the RecurrentGPT memory module without adding new primitives, as long as scope excludes the human/agent loop and aligns to the easier repo-consistent path.
- The most implementation-friendly interpretation is the released repo rather than the paper's stricter wording:
  - short-term memory -> one bounded natural-language summary record that is rewritten each step
  - long-term memory -> append-only paragraph memory with vector retrieval by current plan/instruction
  - prompt context assembly -> previous paragraph + current plan + rewritten short memory + retrieved long-term paragraphs
- The clean module mapping is now clearer:
  - paragraph/history persistence -> `AppendOrganization`
  - long-term recall by next-plan query -> `EmbeddingSimilarityRetrieval` + `ConcatenateReadout` or `TemplateReadout`
  - short-memory rewrite -> `HierarchicalEvolution(extract_mode="generate", selection_mode="latest_active_units", retention_size=1)`
  - full generation prompt assembly with recalled memory -> `TemplateReadout` / `PromptPlan`-based recalled-prompt composition, or equivalently `LLMRepresentation` prompts with sub-recall
- Important paper/repo mismatch: the paper/README often says long-term memory stores summaries of prior paragraphs, but the released repo actually appends raw generated paragraphs into the vector memory and retrieves those paragraphs directly. The current framework matches the repo-side behavior more naturally; paper-style summary memory is still possible by inserting one extra abstraction layer, not by adding a new primitive.
- The example-level wiring now exists in `memprimitive/example/classics/recurrentgpt_memory.py`. Its current status is: executable repo-style reconstruction exists for the memory module plus the simple writer/human-simulator loop; remaining gaps are prompt fidelity/tuning questions rather than missing framework coverage.

### Think-in-Memory (TiM)

- Paper review now suggests the current framework can reproduce the TiM memory module without adding new baseline primitive families, as long as scope excludes the surrounding agent loop and accepts a small amount of example-level orchestration/helper logic.
- The clearest current mapping is:
  - thought extraction / post-thinking -> `LLMFunctionCallOrganization` with repeated `ADD` calls, or helper orchestration plus `SentenceSplitUnitFormation` / `AppendOrganization`
  - thought storage -> flat semantic layer with vector index plus top-level metadata such as `hash_bucket`, head entity, relation, tail entity, and provenance
  - TiM retrieval -> example-level bucket computation, then `MetadataRetrieval(field="hash_bucket", source="store")` followed by `EmbeddingSimilarityRetrieval(source="retrieved")`
  - TiM maintenance within one bucket -> `LLMFunctionCallEvolution` using existing `ADD` / `UPDATE` / `DELETE` tools, where merge is represented as `UPDATE` one kept record plus `DELETE` redundant records
- Important paper ambiguity remains explicit:
  - the paper clearly requires insert / forget / merge within a hash group, but it does not specify a deterministic execution protocol for conflict detection, merge target choice, output schema, or exact invocation timing
  - the prompt examples for forget / merge are illustrative only; they are not enough to derive one unique implementation
- Paper-first audit note:
  - latest paper-only re-audit suggests the main previously identified TiM memory mismatches are now closed at the example level:
    - multiple newly extracted thoughts are grouped by bucket and updated in one bucket-level batch rather than one-by-one
    - same-bucket maintenance exposes the full hash group to forget/merge, matching the paper's group-local organization intent
    - post-thinking thought extraction now receives existing historical thoughts together with the current Q-R pair, matching the paper's "incorporates both historical and new thoughts" requirement at the memory-module level
  - current boundary note:
    - `memprimitive/example/classics/tim_memory.py` should now be treated as reasonably paper-aligned for the memory mechanism itself, while still excluding the broader surrounding agent loop
- Current framework boundary:
  - mechanism-level TiM reconstruction is feasible now without new primitives
  - strict paper-faithful LSH as a first-class reusable primitive does not exist yet; current alignment would compute/query bucket ids in wrapper code and store them as ordinary metadata
  - if future work wants TiM-style hash retrieval and same-bucket maintenance to become declarative reusable baselines rather than example glue, a dedicated hash-bucketing representation/retrieval path would still be a worthwhile later refinement
  - thought extraction is now slightly less ad hoc than before: the example reuses generic `LLMRepresentation` for multi-thought structured extraction, while still keeping bucket grouping and bucket-local update orchestration in thin TiM-specific helper code

### Memory Sharing / INMS

- Paper + upstream repo review now suggests the current framework can reproduce the repo-consistent memory module of `Memory Sharing for Large Language Model based Agents` without adding new baseline primitive families, as long as scope excludes the multi-agent control loop and treats retriever updating as example-level orchestration rather than a new declarative module.
- The easiest alignment target is the released repo `GHupppp/InteractiveMemorySharingLLM`, not the paper's cleaner prose:
  - stored memory is the full retrieved/enhanced prompt plus the generated answer, not just the raw user query plus answer
  - write filtering is an LLM rubric score with a simple threshold gate
  - retrieval at inference is cosine similarity over encoder embeddings of concatenated QA strings
  - the advertised BM25 + LLM labeling + online training path mainly appears in the update procedure for the retriever, not in the final recall scoring path
- The clean framework mapping is now clearer:
  - shared domain memory pool -> one shared `MemoryStore` layer, or one layer per domain when reproducing domain-pool vs single-pool experiments
  - memory write filtering -> `LLMJudgeTrigger(decision_mode="score")`
  - memory append -> `AppendOrganization`
  - retrieval seed / fallback -> `EmbeddingSimilarityRetrieval` for the repo-consistent main path, optionally `BM25Retrieval` or `LayerAwareRetrieval` if reproducing the repo's BM25-first candidate mining logic around retriever updates
  - prompt assembly from retrieved QA examples -> `TemplateReadout` or `ConcatenateReadout`
- Important paper/repo mismatch: the paper frames retriever learning as a central part of the mechanism, while the public repo's executed path is looser and partly inconsistent:
  - new memories are scored and appended with a fixed threshold
  - retriever training uses BM25 candidate mining plus LLM-generated binary labels
  - recall itself later uses encoder cosine similarity on stored QA strings, not the classifier head output
- Current framework boundary:
  - mechanism-level repo-style reconstruction is feasible now without adding new baseline modules
  - fully first-class declarative support for "online-train the retriever whenever a memory is accepted" still does not exist as a primitive; that piece would need to live in wrapper/orchestration code unless a future trainable-retriever primitive is added
- That repo-consistent memory-only wiring now exists in `memprimitive/example/classics/memory_sharing_memory.py` and has deterministic coverage in `tests/test_classics_memory_sharing.py`.
- The implemented scope is intentionally narrow and explicit:
  - accepted examples are stored as prompt-answer memory records in a shared vector-backed pool
  - write filtering is LLM-judge gated with a numeric threshold
  - repo-style rubric selection is now represented too: the judge prompt resolves `Literature` / `Logic` / `Plan` / `Total` from explicit `grading_category` or from domain aliases such as `literal_creation`, `logic_problem_solving`, `plan_generation`, and `one_pool`
  - recall now follows a single-layer, repo-style pool-locking approximation: records stay in one shared layer, then retrieval first uses `MetadataRetrieval` to lock candidates to the resolved pool/domain (`Literature` / `Logic` / `Plan` / `Total`) and only then applies `EmbeddingSimilarityRetrieval(source="retrieved")` within that candidate subset
  - online retriever training remains a documented example-level stub/hook rather than an implemented primitive or baseline family
- Paper-first audit note: the current example should still not be described as paper-aligned at the memory-mechanism level. The main paper-relevant mismatches are now clearer:
  - the paper makes domain-specific pre-established scoring rubrics a first-class part of memory selection, including per-rubric score ranges and a final aggregate score; the current example collapses that into one generic LLM score prompt plus threshold
  - the paper makes accepted memories immediately participate in online retriever training via BM25 candidate mining, LLM contradiction-style scoring, positive/negative labeling, and retriever optimization; the current example only logs accepted record ids in `pending_retriever_updates` and performs no retriever update
  - the paper explicitly supports prompt-less answer-only memories for initial bootstrap and manually seeded initial pools; the current example only accepts non-empty prompt-answer pairs and has no dedicated bootstrap path
  - the paper's experimental setup allocates separate memory pools per domain, while the current example exposes one shared layer and only carries `domain` as metadata unless wrapper code partitions stores manually
- Upstream/reconstruction boundary note:
  - `memprimitive/example/classics/memory_sharing_memory.py` intentionally targets a narrow repo-consistent memory slice rather than the full paper loop
  - the file header already states that multi-agent orchestration is out of scope and retriever online training is left as an explicit placeholder hook

### Other Boundary Papers

- `MemGPT`: latest paper + upstream repo audit suggests the current framework can cover a mechanism-level, memory-only reconstruction without adding new baseline primitive families if scope is limited to the storage/retrieval/update surfaces and allows thin example-level orchestration.
- The clean reusable mapping is:
  - editable in-context core memory blocks -> ordinary layers plus single-record/low-budget retention and `LLMFunctionCallEvolution` / `UPDATE`-style rewrites
  - recall memory -> append-only conversation/history layer with recency or metadata/date-filtered retrieval
  - archival memory -> append-only semantic layer with vector retrieval
  - prompt assembly -> `TemplateReadout` / `PromptPlan`-style explicit reconstruction of visible memory blocks and recalled context
- The main blocker to paper/repo-faithful MemGPT memory remains the queue-manager side, not the storage backends:
  - there is no first-class primitive for a FIFO in-context message queue with head-pinned recursive summary plus explicit reinsertion of retrieved recall results into the active queue
  - memory-pressure warnings / warning-threshold vs flush-threshold behavior are not first-class declarative trigger surfaces
  - recursive compaction of evicted queue spans into the special summary slot can be approximated with existing evolution/readout building blocks, but is not currently one clean primitive path
- Therefore MemGPT should currently be treated like RecurrentGPT / Reflexion in spirit but with a stricter caveat:
  - mechanism-level memory reconstruction is feasible now with example-level glue
  - stronger paper/repo fidelity would likely benefit from one dedicated queue-manager / compaction family rather than only more prompt wiring
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
