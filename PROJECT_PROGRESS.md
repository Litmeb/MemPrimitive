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
- `evaluate.md` 中列出的大部分目标论文现已批量下载到 `paper/`，并在 `paper/DOWNLOAD_MANIFEST.md` 里维护了一份来源清单；当前唯一明确留空的是 `Ego-LLaVA`，因为还没有稳定定位到同名开放论文 PDF。
- 外部 memory 代码生态调研新增了一条重要判断：公开实现并不是清一色“不接数据库”。更产品化/平台化的 memory 项目（如 Mem0、Letta、LightMem）往往显式接向量库、SQL/pgvector、图库或 Redis；更典型的论文复现仓库（如 Reflexion、RecurrentGPT、InteractiveMemorySharingLLM）则通常只用文件、BM25 或进程内结构，最多把 SQLite 当缓存或轻量本地状态。

## What To Treat As Established

Unless there is a major architectural change, these areas are mostly settled:

- `MemoryPipeline` as the strict validated public pipeline surface
- `FreeMemoryPipeline` as the permissive experimental ordered runner
- baseline modules as the main way to build systems
- `openai-agents` as the active runtime direction
- Chinese docs as useful, code-aligned reference material rather than placeholder notes

Avoid re-documenting these in detail unless something materially changes.

## Important Recent Progress

- A new root-level English `README.md` now exists and is aligned with the current repo surface:
  - explains the project goal in terms of composable memory primitives rather than paper-specific pipelines
  - documents config loading, shipped example configs, and the current benchmark CLI entrypoint
  - clarifies that benchmark raw files must be placed by the user into the expected `benchmarks/` subfolders before CLI evaluation
  - gives a concise architecture overview spanning `core` / `pipeline` / `baselines` / `config` / `example` / `benchmarking`
  - now explicitly frames the architecture as LEGO-like slot composition, points readers to `DSL_REFERENCE.zh-CN.md`, and uses `memprimitive/example/demonstration/embedding_similarity_retrieval.py` as the small concrete assembly example
- Retrieval, prompt, readout, graph, and tool-calling surfaces are now broad enough to build nontrivial paper-style reconstructions from shared primitives rather than ad hoc wrappers.
- A first declarative config bridge now exists under `memprimitive/config/`:
  - single-file YAML config with fixed `version/root/objects` shape
  - recursive `$call` / `$import` / `$ref` object-graph resolution
  - explicit shared-object reuse for `MemoryStore` and nested child `MemoryPipeline`
  - `MemoryPipeline` slot-level shorthand now exists too: slot positions can use baseline short names such as `PassThroughUnitFormation` or `$call: RecencyRetrieval`, and trigger slots auto-fill `slot=...`
  - root-loading APIs (`load_object_from_yaml`, `load_pipeline_from_yaml`) plus `python -m memprimitive.config validate ...`
  - example configs under `memprimitive/example/config/` and dedicated tests in `tests/test_config_loader.py`
- Raw benchmark assets are now staged locally under `benchmarks/` for evaluation prototyping:
  - `benchmarks/LoCoMo/` contains the core LoCoMo benchmark files copied from `snap-research/locomo` (`locomo10.json` plus persona source JSON and repo metadata/license)
  - `benchmarks/MSC/` contains the `nayohan/multi_session_chat` train/validation/test parquet shards
  - `benchmarks/DMR/` contains MemGPT's `MSC-Self-Instruct` JSONL benchmark data
  - `benchmarks/LongMemEval/` contains the cleaned LongMemEval splits, including the large `longmemeval_m_cleaned.json`
  This closes the "data not downloaded yet" part of benchmark setup.
- A first benchmark harness slice now exists in `memprimitive/benchmarking/minimal_baseline.py`:
  - unified normalized sample adapters for `LoCoMo`, `LongMemEval`, and `DMR`
  - a minimal one-layer pipeline baseline using `ingest(...)` + single `recall(...)`
  - a thin outer answer runner that sends retrieved memory plus the query to the real OpenAI-compatible runtime
  - a CLI entrypoint that can run limited smoke/debug jobs and write JSONL predictions
- Mem0 LoCoMo benchmark answering now has a dedicated prompt-aligned path:
  - `memprimitive/benchmarking/prompts.py` stays close to the upstream Mem0 answer prompts, including the timestamp and relative-time wording
  - `Mem0LoCoMoAnswerRunner` renders the LoCoMo prompt with `speaker_a` / `speaker_b` labels and the single-profile recall text
  - `minimal_baseline.py` now automatically selects that runner when `--benchmark locomo --memory-adapter mem0`
- The Mem0 LoCoMo benchmark adapter now uses a per-speaker dual-system session shape:
  - each sample/session builds two independent `build_mem0_memory_system(...)` instances, one for `speaker_a` and one for `speaker_b`
  - pairwise ingest now preserves speaker labels in the stored text and flips user/assistant roles for the opposite speaker view
  - recall returns split `speaker_1` / `speaker_2` memory text plus user ids and line-count metadata, and the answer runner now prefers those split fields when rendering prompts
  - `FanoutIngestOrganization` now returns parent-unit placements after child fanout ingest, so Mem0 fact fanout pipelines can pass the default evolution-trigger stage without `packet.placements` errors
  - LoCoMo Mem0 benchmark runs now reuse one loaded memory session per `locomo_sample_id` / user conversation, so the raw dialogue is converted to memories once and then shared across that user's QA samples
- LoCoMo benchmark output compatibility has been tightened further:
  - sample metadata now carries `adversarial_answer` alongside existing `evidence` / `qa_category`
  - benchmark predictions now copy LoCoMo recall metadata into the general `prediction.metadata` payload so speaker ids and memory counts are easy to read without depending on `memory_metadata`
  - the Mem0-style evaluator now accepts both legacy JSONL and prediction-shaped JSONL without misclassifying multi-line JSONL as a single JSON object
- The benchmark CLI can now filter LoCoMo by `--locomo-users` using comma-separated conversation indices, `sample_id`s, or speaker names, and can run either the legacy minimal pipeline or the classics Mem0 reconstruction via `--memory-adapter minimal|mem0`.
- The benchmark CLI now shows `tqdm` progress bars by default: one for Mem0 LoCoMo raw-dialogue memory generation by turn, and one for QA answering by sample/user; pass `--no-progress` to disable them.
- Mem0 LoCoMo benchmark throughput has an initial parallelism pass:
  - the local Mem0 adapter can ingest and recall the two speaker-specific memory systems concurrently via `speaker_workers`
  - the CLI exposes `--mem0-speaker-workers` for that path and `--max-workers` for parallel QA recall/answer work
  - this closes the biggest harness-level gap versus upstream Mem0's threaded LoCoMo add path, though per-speaker turn order remains intentionally sequential because each write depends on prior memory state
- The benchmark runner/CLI now supports smoke runs that cap cost: `--smoke-test` limits to the first 10 QA samples and first 10 history turns per sample, while `--max-history-turns N` exposes the turn cap independently.
- Benchmark adapter top-k routing is now aligned with the CLI: `--memory-adapter mem0` can receive `--top-k` and forwards it into the Mem0 builder, while the CLI default stays adapter-specific (`minimal=5`, `mem0=30`).
- Mem0-style LoCoMo evaluation assets have been moved into `memprimitive/benchmarking/` as runnable local modules:
  - `prompts.py` carries the Mem0 answer-prompt templates
  - `evals.py` reads current benchmark JSONL predictions or grouped Mem0 JSON and writes BLEU1/F1 plus optional LLM-judge metrics
  - `generate_scores.py` summarizes those metric files by category and overall
- That early harness has now been refactored into a more general adapter layer under `memprimitive/benchmarking/`:
  - normalized shared benchmark types now include `ConversationTurn`, richer `BenchmarkSample`, `MemoryRecall`, and prediction/run-result containers
  - official benchmark adapters now cover `LoCoMo` and `LongMemEval` through one common `BenchmarkAdapter` protocol
  - memory-side evaluation can now wrap plain `MemoryPipeline`, `FreeMemoryPipeline`, YAML-loaded pipelines, helper-style system dicts, and pairwise dialogue-ingest systems through `MemoryAdapter` / `MemorySession`
  - a ready-made `mem0` benchmark adapter preset now exists on top of that function/pairwise adapter layer
  - `minimal_baseline.py` is now mainly a compatibility wrapper plus CLI entrypoint rather than the whole benchmarking implementation
- The benchmark harness is still intentionally narrow:
  - baseline 1 is single-recall only, not a tool-calling `MEM_READ` loop
  - `MSC` is not wired into this first baseline because its natural task shape is dialogue continuation rather than QA
  - scoring/metrics aggregation is still not implemented beyond writing predictions and references
- Local upstream-repo survey now sharpens one strategic judgment for future framework planning:
  - lightweight paper-style memory repos often keep the actual memory code simple (plain Python lists, JSONL/file append, in-process embedding similarity, prompt concatenation)
  - the steep engineering cost usually appears later in one of two places:
    - platform/runtime work such as durable storage, multi-backend vector or graph support, APIs, background jobs, and consistency concerns
    - genuinely specialized memory algorithms such as explicit triple stores, graph propagation, or offline indexing/update pipelines
  - so "memory code is easy" and "memory code is intrinsically hard" are both too coarse; the sampled ecosystem is clearly mixed
- A follow-up survey focused specifically on lightweight vs algorithmic memory is now documented and expanded in `LIGHTWEIGHT_ALGORITHMIC_MEMORY_SURVEY.zh-CN.md`.
  - the sampled space now looks more like a spectrum than a binary split:
    - very light prompt/buffer systems (`Reflexion`, `RecurrentGPT`) are still usually cheaper to implement ad hoc for one-off use
    - medium summary/profile/maintenance systems (`MemoChat`, `Memory Bank`, parts of `MOOM`) are now the clearest place where MemPrimitive could plausibly save real work
    - heavy graph/probe/planning systems (`AriGraph`, `ComoRAG`, plus earlier `HippoRAG` / `MemLLM`) keep most of their difficulty in bespoke algorithms and controllers outside the generic slots
  - the broader conclusion remains: MemPrimitive mainly removes mechanism/orchestration complexity, not bespoke algorithm complexity
  - an additional strategic note now has code evidence too: a thin unified adapter/evaluation layer likely has better adoption potential than expecting outside authors to migrate whole memory systems into the full internal abstraction stack
  - official public Think-in-Memory code was still not confirmed in this survey; only a community demo was found, so TiM code-level judgments remain low-confidence
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
- Write-path LLM tool calling now defaults to tolerant failure handling for `LLMFunctionCallOrganization` / `LLMFunctionCallEvolution`: failed tool executions are recorded and skipped by default, `max_retry` can rerun the whole agent with prior error context from a clean store snapshot, and `raise_on_tool_error=True` enables final failure surfacing while legacy `strict_tools=True` still raises immediately.
- PromptPlan labeled sub-recall now degrades cleanly on per-label empty recall-query overrides: an override that renders to `""` records `disabled_reason="empty_rendered_recall_query"` in prompt/readout metadata instead of failing `Query(text=...)` validation.
- `DSL_REFERENCE.zh-CN.md` is being reworked into a code-aligned detailed API-style reference organized by slot/module IO rather than compact tables, so future agents can audit module behavior against real packet/store fields more reliably.
- A new concise local skill entry now exists at `.cursor/skills/memprimitive-dsl-brief/SKILL.md` so agents can start from a short MemPrimitive DSL guide and fall back to `DSL_REFERENCE.zh-CN.md` only for exact parameter/edge-case detail.
- That DSL skill now also contains a `references/` split of `DSL_REFERENCE.zh-CN.md` into per-section documents, and `SKILL.md` explicitly routes agents to the relevant section file before they load the full monolithic reference.
- Integration smoke coverage for real LLM / embedding baselines exists in `tests/test_smoke_real_model_modules.py` and can exercise the main LLM-backed baseline families when runtime credentials are configured.
- Embedding first-stage downshift is now implemented: ordinary record-level text embeddings can be declared per layer in `StoreLayerSpec.settings["embedding"]`, and `MemoryStore.append()` / `replace_record()` now auto-generate or refresh `record.embedding` for those layers.
- The implemented policy boundary is intentionally narrow: stage 1 only handles `mode="text"` with refresh on semantic text change. Entity embeddings, note-payload-derived embeddings, query embeddings, and the broader `UNIT_EMBEDDING_CONTRACT` redesign are still deliberately left in their existing specialized paths.
- Follow-up simplification landed in the expected moderate form: responsibility is more centralized and Mem0/tool-path manual embedding logic is smaller, but the harder embedding complexity still remains in graph/entity/note/query-specialized paths.
- `ConfigurableEmbeddingRepresentation` is now the generic text-configurable embedding representation primitive: render configurable text (including template-based text) from the current unit, embed that text, and record embedding-input provenance in `metadata["representation"]` / trace without rewriting the unit's main text-facing fields.
- Embedding runtime now supports an optional OpenAI-compatible API provider via independent `MEMPRIMITIVE_EMBEDDING_PROVIDER=openai`, `MEMPRIMITIVE_EMBEDDING_API_KEY`, `MEMPRIMITIVE_EMBEDDING_BASE_URL`, and `MEMPRIMITIVE_EMBEDDING_MODEL`; local `sentence-transformers` remains the default.
- Runtime JSON coercion now tolerates common real-model formatting drift in strict JSON paths: Markdown fenced JSON is accepted, and a single truncated top-level array/object with balanced strings can be repaired before parsing. This specifically protects LLMRepresentation list/dict extraction during long Mem0 LoCoMo runs.
- `LLMRepresentation` now supports structured metadata-backed custom fields with `value_type=list[dict[str, str]]` in addition to `str`, `list[str]`, and `dict[str, str]`, using permissive normalization for JSON object lists.
- TiM thought extraction now uses that structured `LLMRepresentation(field="thoughts", value_type=list[dict[str, str]])` path plus a thin example-level normalization helper, instead of a fully hand-written direct runtime JSON call.
- Official Mnemis code is now confirmed public (`microsoft/Mnemis`), but the repo only includes the `global_selection` module plus figures/results/paper. There is no end-to-end reproduction or graph-construction pipeline, and the released code depends on Graphiti + Neo4j + external LLM credentials.
- Paper-setting audit for reproduction now clarifies one important configuration detail: the paper's main embedding setup is not the repo-local MiniLM placeholder. The PDF states that the main experiments use `Qwen3-Embedding-0.6B` with its embedding dimension reduced from `1024` to `128` via MRL; `all-MiniLM-L6-v2` at dimension `384` appears only as an additional comparison embedding in the ablation-style evaluation.
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

### 1. Declarative DSL bridge exists but is still partial

There is now a clean v1 path from single-file declarative config to validated runnable root `MemoryPipeline`, but the broader DSL bridge is still incomplete:

- no reverse serialization from live pipeline/store back into config
- no finished config surface for multi-pipeline system dicts or full classics builders
- no include/override/composition layer beyond one YAML file
- no machine-readable search-space schema layered on top of the config bridge

### 2. Search-space formalization is incomplete

The repo has many ingredients for constrained search, but not yet a finished machine-readable representation of coupling, legality, bundles, and topology-aware constraints.

### 3. Faithful paper coverage is still incomplete

The framework is expressive enough for mechanism-level reconstructions of several systems, but not broad enough to claim faithful coverage yet. The main pressure is no longer "write more prose"; it is adding or refining the module families that close concrete behavioral gaps.

### 4. Evaluation infrastructure is incomplete

There are targeted tests, runnable demos, a unified benchmark adapter layer for `LoCoMo` / `LongMemEval`, and a backward-compatible minimal baseline path (with legacy `DMR` loading still preserved in that wrapper), but evaluation infrastructure is still far from complete:

- only one very simple shared-answer baseline is wired end to end
- `MSC` is still outside the runner
- no benchmark-specific scoring or aggregate analysis is implemented yet
- there is now a thin multi-system adapter surface, but not yet a richer experiment tracking / comparison workflow

### 5. Motif discovery remains later-stage work

The eventual "discover recurring memory motifs" goal depends on the DSL bridge plus search/evaluation infrastructure that still does not fully exist.

## Coverage Snapshot

### Mem0 Family

- `mem0_memory.py` is reasonably close at the mechanism level.
- PromptPlan-controlled candidate visibility now closes the previous major update-scope mismatch: the Mem0 profile-update tools can be restricted to recalled profile candidates instead of the whole `profile` layer.
- LoCoMo benchmark fidelity is now materially better on the timestamp axis: the LoCoMo dataset adapter preserves each turn's `session_timestamp`, `PairwiseDialogueMemoryAdapter` now forwards that value into `ingest_message_pair(...)`, and the Mem0 helper path writes the same dataset timestamp into the pair observation plus the derived `recent_dialogue` and `profile` records. The current `mem0_memory.py` builder intentionally omits a `conversation_summary_update_pipeline`, so the `conversation_summary` layer is only an optional context hook and stays empty by default. The per-speaker mismatch is also improved now that `create_mem0_memory_adapter()` builds two independent Mem0 systems and returns split speaker recall fields; the main remaining difference from upstream `add.py` is still batch granularity, since this repo uses pairwise turn helpers rather than the upstream whole-conversation threaded write loop.
- Latest LoCoMo Mem0 audit note: `benchmarks/outputs/locomo_mem0_user1_predictions_after_fix.jsonl` is only conversation/user 1 (`conv-26`), with 154 predictions and 152 scored items after skipping category 5, so its `llm_score=0.8092` should not be compared directly with the Mem0 paper's full-LoCoMo Table 2 scores. The paper table reports `Mem0=66.88` and `Mem0g=68.44` over the entire LoCoMo dataset; this local user1 run also exposes a larger answer context in practice (about 4k retrieved tokens on average) than the paper table's Mem0 memory-token figure.
- Follow-up chunk-level audit: the local Mem0 LoCoMo path still differs from released Mem0 evaluation code in important ways. Upstream `add.py` batches each session's chronological role-labeled messages in size-2 chunks and passes only that parsed chunk to fact extraction, with update-time similar-memory search fixed at `top_k=5`. The local reconstruction reorders each pair into target-speaker-as-`user` and exposes `recent_messages`, `pair_text`, and unit metadata/timestamp to extraction/update LLM calls. Write-time `similar_top_k` is now independently configurable and defaults to 5, matching upstream's candidate-count default, while recall `top_k` remains separately controlled. These differences can inflate memory richness and cross-speaker leakage, so the current run is useful as a mechanism stress test but not a strict Mem0 reproduction.
- Mem0 LoCoMo speed alignment has started: benchmark `--top-k` still controls recall top-k, while new `--similar-top-k` / `create_mem0_memory_adapter(similar_top_k=...)` controls write-time similar-memory candidate count and defaults to 5. The Mem0 reconstruction also no longer runs a separate conversation-summary LLM on every pair; it still keeps the summary metadata field for prompt compatibility, but the local Mem0 path now only appends recent dialogue after fact extraction/update.
- LoCoMo performance audit: the local Mem0 reconstruction is expected to run much slower than upstream Mem0 eval even when answer quality is similar. For each speaker-pair ingest, it does fact extraction and then fans each extracted fact into its own `LLMFunctionCallEvolution` agent call; upstream Mem0 batches all extracted facts from the chunk into one update prompt. The benchmark adapter also loads the two speaker systems sequentially, while upstream `add.py` writes the two speaker views on two threads. On the user1 output, recall emits 30 profile lines per speaker (60 total, about 19k retrieved characters per QA), whereas upstream eval defaults to a smaller search context.
- The Mem0 profile write path is now organized as extraction-plus-fanout rather than one multi-fact maintenance step: an outer pipeline extracts `fact_list`, then `FanoutIngestOrganization` fans each fact into a child single-fact profile pipeline for add/update/delete decisions. The old helper-side per-fact recall stitching path has been removed.
- The Mem0 fact extraction prompt has now been tightened to mirror upstream `custom_instructions` more closely while keeping the existing JSON list output shape: it now calls out self-contained memories, the person's name instead of generic `user`, emotional states/reactions, ongoing journeys/future plans, specific dates/timeframes, and user-message-only extraction with assistant text treated as conversational context.
- Mem0 profile UPDATE/DELETE tools now reject invisible or hallucinated record ids as explicit no-op tool effects instead of raising through the whole benchmark run. This keeps invalid model tool calls from mutating memory while allowing long LoCoMo-style runs to continue.
- `mem0g_memory.py` has moved closer to upstream structure by keeping both a profile/vector branch and a graph branch, but it is still only partially aligned.
- Mem0g now uses the same prompt-controlled visible-domain path for both its profile/vector tools and graph-maintenance tools.
- Mem0g's profile/vector branch now follows the same extraction-plus-fanout organization as Mem0: outer fact extraction via `fact_list`, then per-fact child ingest through `FanoutIngestOrganization` rather than helper-side multi-fact recall assembly.
- Mem0g no longer does a separate example-level pre-pass for contextual graph hint extraction before graph ingest. The graph branch now feeds the dialogue pair directly into `TripleRepresentation`, so graph entity/triple extraction is expressed by the existing baseline representation path rather than by extra helper-side LLM JSON glue.
- Mem0g dual recall is now also expressed as one reusable pipeline surface: `TemplateReadout` drives labeled sub-recall through the existing profile and graph recall pipelines, replacing the old example-level `recall_all()` string-concatenation wrapper.
- Mem0 / Mem0g benchmark-facing profile recall now preserves timestamps on each retrieved memory line (`timestamp: memory`) through a small local Mem0-family readout helper, while leaving the global `ConcatenateReadout` primitive unchanged for other systems.
- Tool-call organization/evolution now commits retry snapshots back into the original shared `MemoryStore` object instead of replacing a pipeline-local store reference. This fixed a Mem0 LoCoMo failure mode where profile writes appeared in `LLMFunctionCallEvolution` traces but were invisible to the sibling recall pipeline, causing empty benchmark recall.
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

- `MemMachine` (arXiv:2604.04853): latest paper/docs/upstream-code audit suggests the current framework can now cover the contextualized episodic recall spine of the memory system at a mechanism level. The remaining gaps are not the sentence-hit -> parent -> temporal-context -> cluster-rerank recall path, but the surrounding STM consolidation, structured profile maintenance, and optional retrieval-agent orchestration.
- What the current framework can already approximate without new modules:
  - working/short-term episodes -> bounded temporal layer plus `BufferRetrieval` / `RecencyRetrieval`
  - STM/session summary -> `HierarchicalEvolution` or `LLMRepresentation` + summary rewrite/retention
  - long-term raw episodes -> append-only episodic layer with temporal/session metadata
  - sentence-level vector index -> `SentenceSplitUnitFormation` + embedding representation into a separate sentence layer
  - profile memory -> Mem0-style `LLMRepresentation` extraction plus `LLMFunctionCallEvolution` add/update/delete over a profile layer
  - multi-layer context assembly -> `LayerAwareRetrieval` and `TemplateReadout`
- Newly covered reusable module:
  - `ParentEpisodeExpansionRetrieval` now covers the sentence/derivative hit -> source episode expansion step using explicit metadata/provenance parent ids, without record-text parsing.
  - `TemporalNeighborExpansionRetrieval` now covers bounded previous/following episode expansion around retrieved nucleus episodes within matching session/user/agent scope, with chronological dedupe and per-nucleus cluster trace.
  - `EpisodeClusterRerankRetrieval` now covers the final contextualized episodic recall stage: consume temporal episode clusters, rerank clusters through the shared runtime reranker, unify/dedupe under an episode budget with nucleus-near fallback when budget is tight, and return final episodes chronologically.
- Follow-up audit note: `EpisodeClusterRerankRetrieval` has focused code coverage, is registered/exported, and now has a matching `DSL_REFERENCE.zh-CN.md` retrieval-section entry.
- Together, those three retrieval modules cover MemMachine's sentence-derived hit -> parent episode -> temporal neighbor cluster -> cluster-level rerank/unify/chronological-return path without adding a combined paper-specific `ContextualizedEpisodeRetrieval`.
- Modules still needed for a faithful reconstruction:
  - `STMConsolidationEvolution`: expose STM overflow as an explicit event that summarizes evicted episodes, retains/update a session summary, and copies raw evicted episodes into LTM. Current sliding-window trimming does not preserve enough eviction provenance for this declaratively.
  - `ProfileFeatureEvolution`: maintain structured profile features with category/tag/feature/value plus citations/source episode ids, supporting add/delete/consolidate or upsert-style updates rather than only free-text profile records.
  - Optional `RetrievalAgentRetrieval`: route direct retrieval vs split-query vs chain-of-query, run sub-queries/iterative rewrites, and rerank final candidates against the concatenated multi-query history.
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
- If future work adds durable storage backends, treat that as a separate runtime/backend-semantics layer rather than as evidence that current paper-style primitives already abstract over real databases. Current baseline code still assumes easy full scans and synchronous record mutation much more often than DB-backed repos do.

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
