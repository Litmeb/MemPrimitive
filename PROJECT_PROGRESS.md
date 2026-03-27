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
