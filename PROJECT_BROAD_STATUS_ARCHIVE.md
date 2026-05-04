# MemPrimitive Broad Status Archive

Archived from `PROJECT_PROGRESS.md` when the active project focus moved to memory search and memory evolution. This file is reference context, not the current work board.

## Broad Project Goal

`MemPrimitive` expresses agent memory systems as composable primitives instead of paper-specific pipelines. The intended end state remains:

- a unified primitive ontology / DSL
- an executable runtime centered on `MemoryPipeline`
- representative memory papers re-expressed inside one framework
- explicit composition constraints for later search and evaluation
- eventual motif discovery over the memory-system design space

## Broad Framework Position

- Core runtime is usable: slot-based composition, store/topology contracts, baseline modules, shared LLM/runtime paths, and real-model smoke coverage exist.
- Documentation is mostly code-aligned. `README.md` gives the root orientation; `DSL_REFERENCE.zh-CN.md` is the detailed API-style reference; `.cursor/skills/memprimitive-dsl-brief/` is the short agent entrypoint.
- `memprimitive/example/classics/` contains executable mechanism-level reconstructions, including Mem0, Mem0g, MemMachine, A-MEM, RET-LLM/MemLLM, Reflexion, RecurrentGPT, TiM, memory sharing, and COMEDY-style maintenance.
- Benchmarking moved from a one-off minimal script toward a reusable adapter layer. LoCoMo and LongMemEval are the main normalized targets; DMR remains in the legacy compatibility wrapper; MSC is still not wired into the runner.
- The local paper/code survey suggests MemPrimitive is most useful for medium-complexity memory mechanisms: profile/summary/maintenance/retrieval orchestration. Very light prompt-buffer systems may be cheaper ad hoc; heavy graph/planning systems still keep important bespoke algorithm work outside generic slots.

## Implemented Capability Highlights

- Declarative config bridge under `memprimitive/config/`: single-file YAML with `version/root/objects`, `$call` / `$import` / `$ref`, shared object reuse, slot-level shorthand, root loaders, validation CLI, examples, and tests.
- Runtime model paths:
  - OpenAI-compatible embeddings through `MEMPRIMITIVE_EMBEDDING_*`, with local `sentence-transformers` as default.
  - Dedicated OpenAI-compatible rerank path through `MEMPRIMITIVE_RERANK_*` and `POST {base_url}/rerank`; rerank no longer uses chat-LLM prompt emulation.
  - JSON coercion tolerates common real-model drift such as fenced JSON and repairable truncated arrays/objects.
- Important reusable baseline/module additions include `FanoutIngestOrganization`, `MetadataRetrieval`, `ConfigurableEmbeddingRepresentation`, `RerankerRetrieval`, profile feature write tools, prompt-plan visible-record control, and MemMachine-oriented parent/temporal/cluster retrieval plus STM consolidation.
- Baseline surface has also been intentionally tightened. Removed special cases should generally map to `GraphLinkEvolution`, `ConcatenateReadout`, `TemplateReadout`, `GraphAppendOrganization`, `GraphEntityDeduplicationAppendOrganization`, or existing retrieval families instead of compatibility aliases.
- Raw benchmark assets are staged locally under `benchmarks/` for LoCoMo, LongMemEval, DMR, and MSC; `paper/DOWNLOAD_MANIFEST.md` tracks downloaded paper assets.

## Benchmark Status Snapshot

- Main CLI entrypoint remains `memprimitive/benchmarking/minimal_baseline.py`, but it is now mostly a compatibility wrapper over the adapter layer.
- Normalized benchmark-side types and adapters cover LoCoMo and LongMemEval. Memory-side adapters can wrap plain pipelines, YAML-loaded pipelines, helper-style system dicts, pairwise dialogue-ingest systems, and generic memory bindings.
- New systems can plug into LoCoMo-style evaluation through `--memory-adapter binding --memory-binding module:create_memory_binding` if they expose `build_system`, `ingest_event`, and `recall`.
- LoCoMo CLI supports user filtering, smoke/cost caps, progress bars, adapter-specific top-k controls, Mem0 speaker parallelism, and max-worker recall/answer parallelism.
- Mem0-style scoring utilities live in `memprimitive/benchmarking/evals.py` and `generate_scores.py`, but broader experiment tracking and comparison workflow is still thin.

Important fidelity notes:

- Mem0 LoCoMo uses two speaker-view memory systems and split speaker recall. It is useful as a mechanism stress test, but not a strict upstream reproduction because local ingest uses pairwise helper paths and per-fact fanout, while upstream batches chronological messages and updates extracted facts more compactly.
- MemMachine LoCoMo should use one shared conversation memory, not two speaker memories. The repaired path ingests raw messages individually, immediately indexes LTM/sentence records, recalls with `limit=30` and `expand_context=3`, preserves source ids, and renders timestamped memory sections.
- Do not compare the single-user Mem0 LoCoMo artifact directly with full LoCoMo paper scores; it covers only one conversation/user subset and uses a larger answer context than the paper table setting.

## Classics Coverage Snapshot

### Mem0 / Mem0g

- Mem0 is reasonably close at the mechanism level: fact extraction, profile maintenance, prompt-plan visible candidate control, timestamp-preserving recall, and LoCoMo dual-speaker evaluation wiring exist.
- Remaining Mem0 fidelity gaps are mainly upstream batching/update semantics, runtime cost, possible cross-speaker leakage from local helper shape, and strict reproduction of paper token/context settings.
- Mem0g keeps both profile/vector and graph branches, but graph-native storage semantics, relation-level invalidation/update behavior, and full repo-style recall behavior remain only partial.

### MemMachine

- Core memory-layer reconstruction is covered at the mechanism level: working memory, STM consolidation/overflow, raw episodic LTM, sentence-derived index, parent expansion, temporal neighbor expansion, cluster rerank, timestamped readout, and structured profile tools.
- `memmachine_memory.py` and the LoCoMo adapter should be treated as the current preferred runtime shape.
- Optional `RetrievalAgentRetrieval` is still not implemented: direct vs split-query vs chain-of-query routing and multi-query rerank remain a future refinement, not a core memory-layer blocker.

### A-MEM / Agentic Memory

- Executable mechanism-level reconstruction exists in `amem_memory.py`.
- The preferred implementation is generic composition: LLM note fields, configurable embedding text, `GraphAppendOrganization`, and one bounded `LLMFunctionCallEvolution` with A-MEM-specific tools for current-note link strengthening and neighbor context/tag updates.
- Remaining gaps are mostly prompt/fidelity tuning. Keyword rewrite should be treated as optional stretch because released repos emphasize context/tag updates more than full keyword mutation.

### RET-LLM / MemLLM

- `ret_llm_memory.py` remains a mechanism-level reconstruction, not a faithful reproduction.
- The `MEM_READ` path scans the full triple layer and can return all matching triplets, but there is still no paper-style unified fine-tuned controller that emits both `MEM_WRITE` and `MEM_READ`.
- Temporal conflict handling and exact triplet-table semantics remain approximate.

### Reflexion / RecurrentGPT

- Both are feasible without new primitive families when scoped to memory and prompt-context injection rather than the full agent loop.
- `reflexion_memory.py` uses generic hierarchical evolution and bounded reflection buffers. It still lacks task-local trial-loop semantics unless callers isolate stores manually.
- `recurrentgpt_memory.py` follows the released repo shape: bounded short memory rewrite, append-only paragraph memory, vector recall, and prompt assembly.

### TiM

- `tim_memory.py` should be treated as reasonably paper-aligned for the memory mechanism itself, while still excluding the surrounding agent loop.
- Hash grouping, bucket-local forget/merge, and thought extraction are implemented through example-level helper orchestration plus generic representations/retrieval/evolution.
- A first-class reusable hash-bucketing representation/retrieval path would still be useful later if TiM-style behavior should become declarative rather than example glue.

### Memory Sharing / INMS

- `memory_sharing_memory.py` intentionally targets a narrow repo-consistent memory slice: accepted prompt-answer examples, LLM-judge write filtering, domain/pool metadata, embedding recall, and a placeholder hook for retriever updates.
- It should not be described as fully paper-aligned: online retriever training, separate experimental memory pools, bootstrap prompt-less memories, and rubric-specific score semantics are not fully reproduced.

## Other Boundary Systems

- MemGPT is feasible as a memory-only mechanism with example-level glue, but paper/repo fidelity likely needs a dedicated queue-manager/compaction primitive family.
- HippoRAG and AriGraph still point to graph retrieval, propagation, and graph-maintenance gaps.
- HiAgent still points to hierarchical working-memory management gaps.
- LightMem still points to staged/offline maintenance gaps.
- Mnemis code is public, but the release is not an end-to-end reproduction; it depends on Graphiti, Neo4j, and external LLM credentials.

## Architecture Boundaries To Remember

- Trigger outputs are still easier to align to incoming units than to arbitrary store-side candidate subsets.
- Retrieval output is still mostly a flat `RetrievedSet`; richer grouped/provenance-aware recall views remain underdeveloped.
- Durable storage backends are a separate runtime/backend-semantics concern. Current baseline code still assumes easy full scans and synchronous record mutation more often than DB-backed product systems do.
- Trigger diversity is real, but literature review suggests representation, organization, maintenance/evolution, and retrieval differences usually matter more.

## Repository Notes

- The repo may contain unrelated in-progress changes; do not overwrite them casually.
- Prefer targeted tests for the changed area. Full test runs can be slow.
- When benchmark behavior looks wrong, inspect output JSONL/metrics first: empty recall, missing source ids, or malformed metadata often explain score collapses faster than prompt speculation.
