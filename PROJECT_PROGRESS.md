# MemPrimitive Project Progress

Last updated: 2026-03-27

## Purpose of this document

This file is the durable project-memory document for agents working in this repository.

When an agent learns something important about:

- project goals
- current implementation status
- major architectural decisions
- remaining gaps
- milestone progress

it should update this file so the next agent can continue from an up-to-date state instead of re-reading the whole repository from scratch.

## Project in one paragraph

`MemPrimitive` is trying to turn agent-memory research from a collection of named methods into a compositional, mechanism-level design space. Instead of treating MemGPT, Reflexion, TiM, A-MEM, and similar systems as isolated end-to-end architectures, the project decomposes memory into reusable primitive slots such as unit formation, representation, write trigger, organization, evolution trigger, memory evolution, retrieval, and readout. The intended end state is a unified DSL plus runtime substrate that can express existing methods, compare them slot-by-slot, enforce composition constraints, support search over valid configurations, and eventually surface recurring design motifs.

## What the project wants to do

Based on the repository docs and code, the target is not merely "build a memory library". The project appears to want all of the following:

1. Define a unified ontology for agent-memory mechanisms.
2. Provide a shared runtime interface so primitives from different families can plug into one `MemoryPipeline`.
3. Re-express classic systems as configurations or module families inside that common framework.
4. Represent topology, primitive choice, and constraints explicitly enough to support constrained architecture search.
5. Make it possible to compare memory systems at the mechanism level rather than only paper-vs-paper.
6. Eventually derive recurring high-performing memory motifs from systematic exploration.

The clearest statement of this appears in [README.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/README.md), with supporting detail in [DSLIO.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/DSLIO.md), [DSLgrammar.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/DSLgrammar.md), and [Primitives.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/Primitives.md).

## What has already been done

### 1. A usable stage-1 runtime exists

The core package is not just a concept sketch. `memprimitive` already implements a concrete stage-1 runtime:

- Core IR/data objects exist in [memprimitive/core.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/core.py): `Observation`, `MemoryUnit`, `MemoryRecord`, `Placement`, `Query`, `RetrievedSet`, `Readout`, `Packet`, `StoreLayerSpec`, `StoreTopology`, `MemoryStore`, and `ModuleSpec`.
- A fixed-slot pipeline exists in [memprimitive/pipeline.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/pipeline.py): ingest side and recall side are wired through explicit primitive slots.
- Slot interfaces exist in [memprimitive/interfaces.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/interfaces.py).
- Slot-local fan-out/dispatch exists in [memprimitive/dispatch.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/dispatch.py).

This means the project already has a real executable substrate for composing memory primitives.

### 2. Topology and compatibility checking are already partially formalized

This is one of the most important completed pieces because it turns the design space into something machine-checkable instead of purely descriptive:

- `StoreTopology` and `StoreLayerSpec` encode layers, shapes, indices, capacities, and settings.
- `ModuleSpec` carries requirement/guarantee metadata.
- `MemoryPipeline` validates type compatibility, slot compatibility, and some store/layer requirements before execution.
- Graph-layer compatibility is enforced explicitly.

This is evidence that the project has already moved beyond prose taxonomy into constrained composition.

### 3. Stage-1 baseline primitive families are substantial

The baseline layer is broad, not trivial. Per [memprimitive/baselines/README.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/README.md) and tests, the repository already has:

- Multiple unit formation modules.
- Multiple representation strategies, including embeddings, triples, entities, tags, keywords, summary/description hooks, and metadata-aware variants.
- Trigger-family infrastructure with signals, scorers, gates, and policies.
- Several organization modules, including append, conditional routing, and graph append.
- Multiple evolution implementations.
- Multiple retrieval implementations, including recency, keyword count, BM25, tag/entity retrieval, embedding similarity, and layer-aware retrieval.
- Multiple readout formats.
- Baseline registry/auto-discovery for slot implementations.

So "primitive slot space" is already populated with enough diversity to make the framework meaningful.

### 4. Demonstration scripts already show composition patterns

The examples under [memprimitive/example/demonstration/README.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/demonstration/README.md) and sibling scripts already exercise:

- minimal pipeline assembly
- composed trigger families
- multi-layer topology stores
- embedding retrieval
- layer-aware retrieval
- graph organization and retrieval
- conditional routing
- dispatch-style fan-out
- Reflexion-style trigger examples

This means the project already demonstrates the framework's compositional claims in runnable form.

### 4.1 Graph baseline family is now a reusable stage-1 pipeline base

The repository now has a more complete graph-oriented baseline family, not just isolated demo behavior:

- `GraphAppendOrganization` writes a stabilized graph metadata payload (`graph.layer`, `graph.shape`, `graph.entities`, `graph.triples`, `graph.links`, `graph.node_count`, `graph.link_count`, `graph.last_linked_at`, `graph.link_history`).
- `GraphNeighborRetrieval` supports explicit seed-id neighbor recall on graph layers.
- `GraphSeedAndExpandRetrieval` provides a simplified baseline seed-and-expand graph retrieval path using query-token/entity scoring plus one-hop graph expansion.
- `GraphNeighborAppendEvolution` appends graph links for newly written graph records without touching non-graph layers.
- `GraphReadout` renders retrieved graph payloads in a stable graph-readable format.
- `memprimitive/example/demonstration/graph_baseline_pipeline.py` now shows the full graph loop: ingest -> link evolution -> neighbor recall -> readout.

This stage is important because it turns graph memory into a reusable test base for later graph-dependent motifs instead of leaving it as a one-off organization demo.

### 4.2 Trigger-family shared infrastructure now covers classic motif prerequisites

The shared trigger framework in [memprimitive/baselines/_trigger_family.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/_trigger_family.py) now goes beyond the initial constant/query-style examples and includes reusable classic-motif infrastructure:

- Existing Reflexion-oriented pieces remain in place: `OutcomeCorrectnessSignal`, `FeedbackPresenceSignal`, `FeedbackSchemaGate`.
- New common signals now cover metadata/unit/placement/partition readiness and neighbor-based graph evolution prerequisites:
  - `MetadataFlagSignal`
  - `UnitTypeSignal`
  - `PlacementExistsSignal`
  - `PartitionKeyPresentSignal`
  - `NeighborCountSignal`
  - `TopNeighborSimilaritySignal`
- New common gates now cover generic schema readiness plus graph/vector capability checks:
  - `SchemaPresentGate`
  - `HasEmbeddingGate`
  - `VectorIndexReadyGate`
  - `GraphLayerGate`

Practical impact:

- TiM-like inferred trigger decompositions can now be expressed with shared `signal -> scorer -> gate -> policy` pieces instead of bespoke logic.
- A-MEM-like neighbor-triggered evolution can now reuse shared neighbor-count / top-similarity signals plus graph/vector readiness gates.
- Family-specific trigger modules for later phases can be assembled through `compose_write_trigger` / `compose_evolution_trigger` without changing the pipeline API.

Validation status:

- `tests/test_baselines.py` now includes dedicated coverage for each of the new trigger-family signals/gates, including normal paths, missing-field failures where applicable, and blocked graph/vector readiness paths.
- `python -m memprimitive.example.demonstration.trigger_family_infrastructure` provides a runnable TiM-like plus A-MEM-like shared infrastructure demo.

### 4.3 Graph-dependent trigger and evolution baselines now exist as stage-3 building blocks

The baseline graph family is no longer limited to append-plus-link demos. The repository now has reusable graph-dependent trigger/evolution primitives that can be composed into a fuller graph pipeline:

- `NeighborExistsEvolutionTrigger` in [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py) implements the motif-guide's neighbor-exists trigger as a trigger-family composition, not a bespoke black-box trigger.
- `compose_graph_neighbor_evolution_trigger(...)` provides a reusable graph evolution trigger composer over `NeighborCountSignal`, `TopNeighborSimilaritySignal`, `HasEmbeddingGate`, `VectorIndexReadyGate`, and `GraphLayerGate`.
- `GraphLinkEvolution` in [memprimitive/baselines/memory_evolution.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/memory_evolution.py) performs graph-layer-local link evolution using same-layer neighbor candidates, `MemoryStore.add_graph_links`, and safe graph metadata rewrites.
- `GraphNeighborContextTraceEvolution` adds the simplified/trace-first neighbor-context update baseline. It can stay trace-only or conservatively write a namespaced `graph.neighbor_context` snapshot.
- The older `GraphNeighborAppendEvolution` now remains as a backward-compatible wrapper on the newer graph-link evolution path instead of diverging as a separate implementation.

Practical impact:

- There is now a clear stage-3 baseline path for graph-dependent evolution motifs before implementing more paper-faithful A-MEM-style controllers.
- Later A-MEM-like work can reuse a stable end-to-end substrate rather than bundling trigger, link evolution, and retrieval logic into one family-specific module.
- Graph evolution writes are intentionally scoped to the target graph layer, and metadata rewrites stay under the `metadata["graph"]` namespace.

Validation status:

- `tests/test_baselines.py` now includes dedicated coverage for the graph neighbor trigger composer, `NeighborExistsEvolutionTrigger`, `GraphLinkEvolution`, `GraphNeighborContextTraceEvolution`, and a full ingest -> organization -> evolution_trigger -> memory_evolution -> retrieval -> readout pipeline.
- `python -m memprimitive.example.demonstration.graph_dependent_pipeline` provides the new minimal stage-3 graph pipeline demonstration.

### 4.4 Non-graph trigger-family motif baselines now exist for TiM / Reflexion / MemGPT-style triggers

The stage-1 trigger-family substrate now also covers the non-graph trigger motifs called out by the motif DSL layer guide, without introducing new pipeline slots or family-specific trigger black boxes:

- `MetadataGatedWriteTrigger` in [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py) expresses TiM-style metadata-gated write as `UnitTypeSignal + MetadataFlagSignal + MinScorer + AlwaysOpenGate + ThresholdPolicy`.
- `KeyReadyWriteTrigger` in [memprimitive/baselines/write_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/write_trigger.py) expresses key/presence-gated write for keyed or partition-addressable storage paths such as MemGPT-style upsert flows.
- `OutcomeConditionedEvolutionTrigger` in [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py) expresses Reflexion-style failure-triggered reflection using the shared `OutcomeCorrectnessSignal`, `FeedbackPresenceSignal`, and `FeedbackSchemaGate`.
- `NewWriteEvolutionTrigger` in [memprimitive/baselines/evolution_trigger.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/evolution_trigger.py) expresses TiM-style "new write triggers local maintenance" using `UnitTypeSignal`, `PlacementExistsSignal`, `PartitionKeyPresentSignal`, `MinScorer`, and `LayerAllowedGate`.

The same slot modules now also expose motif-oriented builders:

- `compose_metadata_gated_write_trigger`
- `compose_key_ready_write_trigger`
- `compose_outcome_conditioned_evolution_trigger`
- `compose_new_write_evolution_trigger`

Practical impact:

- TiM / Reflexion / MemGPT-style trigger motifs can now be reconstructed directly inside the baseline slot files rather than only inside classic-family modules or ad hoc examples.
- The trigger-family four-piece decomposition is now visible in stable baseline class names, which makes these motifs easier to compare, test, and later search over.
- Existing Reflexion demonstrations were updated to use the stable `OutcomeConditionedEvolutionTrigger` baseline, and a new demonstration now shows a partition-ready unit opening TiM-style local maintenance.

Validation status:

- `tests/test_baselines.py` now includes dedicated coverage for `MetadataGatedWriteTrigger`, `KeyReadyWriteTrigger`, `OutcomeConditionedEvolutionTrigger`, and `NewWriteEvolutionTrigger`.
- The following demonstrations run successfully:
  - `python -m memprimitive.example.demonstration.reflexion_trigger_failed_trial`
  - `python -m memprimitive.example.demonstration.reflexion_trigger_success_trial`
  - `python -m memprimitive.example.demonstration.partition_ready_local_maintenance`

### 4.5 Reflexion-like back-half motifs are now baseline-first instead of classic-only

The Reflexion family is no longer represented only by classic-family classes. Its reusable back-half motifs now live in the baseline slot files and the classic wrapper composes them for backward compatibility:

- `PlacementWithoutAppendOrganization` in [memprimitive/baselines/organization.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/organization.py) expresses the placement-without-append organization motif for ephemeral trial packets.
- `OutcomeConditionedEvolutionTrigger` remains the shared failure-trigger path and is now the underlying implementation for the classic `TrialFailureEvolutionTrigger`.
- `ReflectionGenerationEvolution` in [memprimitive/baselines/memory_evolution.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/memory_evolution.py) separates the generic memory-evolution skeleton from the benchmark/prompt residual through a `reflection_generator` / `prompt_builder` boundary.
- `BufferRetrieval` in [memprimitive/baselines/retrieval.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/retrieval.py) expresses bounded temporal buffer recall without query-dependent search.
- `PromptContextReadout` in [memprimitive/baselines/readout.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/baselines/readout.py) renders switchable next-trial prompt context (`base`, `last_trial`, `reflexion`, `last_trial_and_reflexion`).

Practical impact:

- Reflexion-like modules can now participate in the baseline registry and be compared at the slot level instead of only through a family-specific classic file.
- The classic API in [memprimitive/classic_modules/reflexion.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py) is preserved via thin wrappers and compatibility traces.
- A full Reflexion-like closed-loop demo now exists at `python -m memprimitive.example.demonstration.reflexion_reflection_cycle`.

Validation status:

- `tests/test_baselines.py` now covers success-suppressed reflection generation, failure-triggered reflection append, buffer-window retrieval truncation, and prompt-context strategy switching.
- Existing classic Reflexion tests continue to target the old public class names while exercising the baseline-backed implementation.

### 4.6 A-MEM-like graph-note motifs now exist as baseline-first modules on top of the graph pipeline

The graph pipeline is no longer limited to generic append/link/readout primitives. It now also covers the A-MEM-like enriched-note path through reusable slot modules plus thin classic wrappers:

- Representation:
  - `SemanticFieldEnrichmentRepresentation` generates repaired note payloads (`content`, `note_text`, `context`, `keywords`, `tags`, `category`, `attributes`) through the shared classic runtime.
  - `RetrievalOrientedEmbeddingRepresentation` turns that note payload into retrieval-oriented embedding text and writes embeddings plus stable `metadata["representation"]`.
- Write trigger:
  - `LLMJudgedWriteTrigger` adds a reusable LLM-judged write path with explicit `decision/reason/confidence` trace output.
- Organization:
  - `GraphAppendLinkReadyOrganization` appends enriched notes into graph layers while initializing link-ready graph metadata.
- Evolution:
  - Existing `NeighborExistsEvolutionTrigger` remains the generic trigger-family decomposition for graph-neighbor-triggered evolution.
  - `LinkStrengtheningEvolution` now performs LLM-selected graph link strengthening plus current-note tag refresh.
  - `NeighborContextUpdateEvolution` rewrites linked neighbors' note context/tags with schema repair.
- Retrieval and readout:
  - `VectorGraphSeedAndExpandRetrieval` provides the vector-seed + graph-expansion retrieval path used by A-MEM-like pipelines.
  - `NoteRenderReadout` renders enriched note payloads into readable note/context/tag output.

Classic-layer impact:

- `memprimitive/classic_modules/amem.py` is now mostly a compatibility layer over the new baseline modules.
- The classic wrapper keeps the paper residual mainly in the higher-level A-MEM evolution controller prompt/decision loop and naming compatibility.
- `memprimitive/example/classics/amem_agentic_memory.py` still exposes the familiar classic entry point, while `memprimitive/example/demonstration/amem_like_graph_cycle.py` shows the same family as explicit baseline slot composition and now uses the shared classic runtime instead of an in-script fake demo runtime.
- The shared classic runtime JSON coercion is now more tolerant of extra prose around a valid JSON block, which matters for real-API demonstrations where some models occasionally prepend or append commentary despite a strict-JSON instruction.

Validation status:

- `tests/test_baselines.py` now covers:
  - note-schema repair across representation stages
  - graph/vector store precondition validation for link-ready graph organization
  - vector-seed retrieval neighbor expansion
  - link-strengthening + neighbor-context writeback
- Existing `tests/test_classic_amem.py` should continue to exercise the public classic A-MEM API on top of the baseline-backed implementation.

### 5. Classic method families have been partially reconstructed

This is a major step toward the stated goal of re-expressing literature inside one framework:

- TiM modules exist under [memprimitive/classic_modules/tim.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/tim.py).
- Reflexion modules exist under [memprimitive/classic_modules/reflexion.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/reflexion.py).
- MemGPT modules exist under [memprimitive/classic_modules/memgpt.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/memgpt.py).
- A-MEM modules exist under [memprimitive/classic_modules/amem.py](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/amem.py).

There are also example workflows for these families under [memprimitive/example/classics](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/example/classics) and dedicated tests under [tests](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/tests).

### 6. The repository already acknowledges search constraints and hidden coupling

[memprimitive/classic_modules/SEARCH_COMPATIBILITY.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/memprimitive/classic_modules/SEARCH_COMPATIBILITY.md) is especially important because it shows the project has already identified a core research challenge:

- classic modules fit the common slot interface
- but they are not fully orthogonal
- hidden coupling still lives in metadata schemas, topology assumptions, and family-specific invariants

This document effectively marks the current boundary between "composition works" and "safe architecture search is not finished yet".

### 7. Tests cover a large portion of the current framework

The test suite indicates meaningful implementation maturity:

- core data model tests
- pipeline validation/composition tests
- baseline primitive tests
- classic family tests
- classic composition tests

This strongly suggests the repository is already in a serious prototyping phase rather than an idea-only phase.

**Running tests:** The suite is large; a full run can take a long time. For day-to-day work, prefer running only the tests relevant to your change (e.g., a single file or a narrowed `pytest` selection). Slow full runs are normal—allow extra time if you do run everything.

## What is not finished yet

The repository is meaningfully advanced, but it is not at the final stated research goal yet. The biggest missing pieces appear to be the following.

### 1. The DSL is documented more than it is implemented

There is a lot of design material in [DSLIO.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/DSLIO.md), [DSLgrammar.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/DSLgrammar.md), and [Primitives.md](D:/Git/MemPrimitive-MemEngineDemo/MemPrimitive/Primitives.md), but the executable code currently centers on Python classes and `MemoryPipeline`, not on a fully implemented parser/serializer/compiler for the proposed declarative DSL.

Practical implication:

- the conceptual DSL exists
- the runtime substrate exists
- the bridge from formal DSL spec to end-to-end executable configs still seems incomplete

### 2. Search is still mostly a research direction, not a completed subsystem

The project talks extensively about searchable configuration space, but there does not yet appear to be a complete architecture-search engine that:

- enumerates valid candidates
- applies constraints automatically
- samples/searches across topology + module + hyperparameter choices
- runs evaluation loops
- records results systematically
- mines recurring motifs from those results

The closest current groundwork is compatibility metadata, topology checks, module registries, and the search-compatibility note.

### 3. Hidden semantic coupling is not fully surfaced into machine-readable constraints

This is probably the single biggest blocker between today's framework and true constrained search.

The current runtime can validate things like:

- slot type
- slot name
- graph/vector/index requirements
- target layer existence/shape

But the repository itself notes that it still cannot fully validate:

- required metadata namespaces on units/records/queries
- family-specific record schemas
- family-specific trace contracts
- bundle-only search modes
- topology-family compatibility beyond current hard-coded checks

Until those are explicit and machine-readable, many "valid" combinations will still only be syntactically valid rather than semantically sound.

The graph family is improved here, but still only partially formalized:

- graph retrieval still uses simplified heuristic seed scoring rather than a general vector-seed abstraction
- graph link evolution is baseline-safe and local, not yet paper-faithful A-MEM agentic evolution
- graph metadata contracts are stabilized in code, but not yet surfaced as a full searchable ontology layer
- there is now an LLM-driven neighbor-note rewrite path, but the full graph-note metadata contract is still not elevated into a searchable ontology layer
- the classic A-MEM evolution controller still keeps prompt-level residual logic at the wrapper layer instead of exposing every paper residual as search metadata

### 4. Classic reconstructions are present, but not yet turned into a stable search-ready ontology

The classic families are implemented, but they still appear to function more like:

- paper-aligned module families
- example-compatible workflows
- manually reasoned search bundles

rather than a fully standardized searchable ontology with explicit coupling metadata.

### 5. Evaluation and benchmark story is still incomplete

There are tests and examples, but the repo does not yet look like it has a finished research benchmark harness that would let the team say:

- here is the controlled experiment protocol
- here is how configurations are compared
- here is how results are logged and analyzed
- here is how motif discovery is measured

The presence of `MemEngine` and `A-mem` suggests external or comparative work is happening around the main framework, but that does not yet read as a finished integrated evaluation pipeline for the full design-space agenda.

### 6. The "recurring motifs" end goal is still future work

The docs repeatedly frame motif discovery as a final research output, but there is no obvious completed subsystem yet that mines motifs from search/evaluation results. That is still an aspirational later-stage deliverable.

## Best current reading of project status

The project seems to be in this phase:

1. The conceptual research framing is well developed.
2. The stage-1 executable substrate is real and fairly strong.
3. Baseline primitive coverage is already broad enough to support composition experiments.
4. Several classic methods have been re-expressed as module families.
5. Constraint-aware search has been recognized as the right direction.
6. The missing jump is from "composable runtime + documented design space" to "fully explicit DSL + safe search + evaluation + motif mining".

Short version:

The project has already built the foundation and a meaningful prototype framework. It has not yet completed the full research program it describes in the README.

## Highest-priority remaining work

If the goal is to actually complete the repository's stated agenda, the most important next steps appear to be:

### Stage-gate note: readiness for phase 2 (search design)

Current best judgment: the repository is ready to enter **phase 2**, but only if phase 2 starts with **search-space formalization** rather than immediately implementing a search algorithm.

Why this is now reasonable:

- stage-1 runtime and slot interfaces are already stable enough to serve as the executable substrate
- baseline slot coverage is broad enough that the search space is non-trivial and worth designing around
- topology/index/shape validation already provides a real first layer of hard constraints
- classic-family compatibility has already been analyzed in `memprimitive/classic_modules/SEARCH_COMPATIBILITY.md`

What is still missing before "full search" can be safe:

- machine-readable coupling metadata beyond current `ModuleSpec`
- explicit declarations of required/produced unit, record, and query metadata contracts
- topology-family and bundle-level search constraints
- a canonical candidate/config representation that separates free modules, semi-coupled bundles, and bundle-only families

Practical implication:

- **Yes**: begin phase 2 by designing the search substrate, candidate schema, constraint model, and legality checks.
- **Not yet**: jump straight to unrestricted enumeration, scoring, or optimization over all modules as if the space were already fully orthogonal.

### A. Make hidden coupling explicit

Add machine-readable metadata for each module family, likely along the lines already proposed in `SEARCH_COMPATIBILITY.md`, such as:

- family id
- coupling level
- required/produced unit metadata
- required/produced record metadata
- required query metadata
- topology family
- safe search mode

Without this, architecture search will remain fragile.

### B. Build the DSL-to-runtime bridge

Implement a concrete path from the design docs to execution:

- config schema or parser
- validation layer
- compiler/builder into `MemoryPipeline` and topology objects
- serialization round-trip for classic/system configs

This would turn the current research language from documentation into executable infrastructure.

### C. Build a constrained search subsystem

The project likely needs a dedicated subsystem that can:

- enumerate legal configurations
- bundle strongly coupled families when needed
- reject invalid mixes before execution
- sweep hyperparameters
- persist results and traces

### D. Build a benchmark/evaluation harness

To support the research claims, the project still needs a repeatable experiment layer:

- datasets/tasks
- metrics
- run orchestration
- result logging
- comparison tooling

### E. Build motif analysis on top of search results

Only after the above pieces exist can the project really produce the "recurring motif" output described in the README.

## Practical guidance for future agents

When continuing work in this repository, do not assume the project is "mostly docs only" or "already done". A more accurate framing is:

- the framework core is already real
- the primitive runtime is usable
- the baseline layer is substantial
- the classic-family mapping work is substantial
- the full declarative DSL, search system, and evaluation/motif layers are still incomplete

Future agents should update this document whenever they:

- add or change major primitive slots
- introduce search metadata or search infrastructure
- implement a real DSL parser/compiler/builder
- add benchmark/evaluation pipelines
- materially change the estimate of what is complete vs incomplete

## Repository snapshot noticed during this read

At the time of writing, the worktree already contained unrelated in-progress changes outside this document update, including modifications in:

- `memprimitive/baselines/_trigger_family.py`
- `memprimitive/example/demonstration/README.md`
- `tests/test_baselines.py`

and several untracked files under `A-mem/` plus new demonstration scripts. Future agents should treat those as existing work and avoid overwriting them casually.
