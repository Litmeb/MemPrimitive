# Classic Module Search Compatibility

This note summarizes the current search-time compatibility of the four classic module families that are actually present under `memprimitive/classic_modules/`:

- `tim.py`
- `reflexion.py`
- `memgpt.py`
- `amem.py`

The goal is not to restate the papers, but to describe which modules can safely enter module search, where the hard constraints are, and where the current runtime still has hidden semantic coupling.

## Bottom Line

These classic modules are compatible at the `MemoryPipeline` interface level, but they are not fully orthogonal.

- They all implement the shared slot interfaces and can be inserted into `MemoryPipeline`.
- `MemoryPipeline` can already reject many invalid combinations through `ModuleSpec.store_requirements`, `ModuleSpec.layer_requirements`, and `validate_store(...)`.
- The remaining problem is hidden coupling through `unit.metadata`, `record.metadata`, layer naming, and family-specific invariants.

So the right search regime is:

- search with constraints
- search by family bundles when coupling is high
- avoid unrestricted Cartesian products across all classic modules

## Coupling Levels

| Family | Current slot coverage | Coupling level | Recommended search unit |
| --- | --- | --- | --- |
| `tim` | full 8-slot family | medium-high | mostly as a bundled family, or split into early-stage vs late-stage bundles |
| `reflexion` | `organization`, `evolution_trigger`, `memory_evolution`, `retrieval`, `readout` | medium | as a back-half family that can hang off generic front-half pipelines |
| `memgpt` | `organization`, `retrieval`, `readout` | high | only inside a MemGPT-shaped topology family |
| `amem` | `representation`, `write_trigger`, `organization`, `evolution_trigger`, `memory_evolution`, `retrieval`, `readout` | very high | as a tightly bundled graph-memory family |

## Search Matrix

### TiM

| Slot | Module | Requires | Produces | Hidden semantic coupling | Search advice |
| --- | --- | --- | --- | --- | --- |
| `unit_formation` | `TimThoughtUnitFormation` | `observation.text` | `units`, `units.metadata.tim` | emits TiM-specific thought schema | safe to search, but strongest when paired with TiM downstream |
| `representation` | `TimThoughtRepresentation` | `units` | `units.embedding`, `units.metadata.representation` | writes `tim.hash_index`, `tim.group_id`, `tim.summary` | should usually stay paired with TiM organization/evolution/retrieval |
| `write_trigger` | `TimThoughtWriteTrigger` | `units` | `decisions` | expects `unit_type == "tim_thought"` and `metadata.tim.write` semantics | low risk if TiM unit formation stays upstream |
| `organization` | `TimThoughtMemoryOrganization` | declared `thought_memory` layer | `placements` | rebuilds and depends on TiM bucket index | not a good free-mix module |
| `evolution_trigger` | `TimBudgetEvolutionTrigger` | declared `thought_memory` layer, `units`, `placements` | `evolution_decisions` | assumes written units are TiM thoughts | keep with TiM organization/evolution |
| `memory_evolution` | `TimThoughtMemoryEvolution` | declared `thought_memory` layer, aligned `units/placements/evolution_decisions` | evolution effects in trace | depends on TiM bucket/group structure inside store records | strongly bundled |
| `retrieval` | `TimThoughtMemoryRetrieval` | declared `thought_memory` layer, `query.text` | `retrieved` | assumes records were bucketed by TiM representation/evolution | strongly bundled |
| `readout` | `TimThoughtReadout` | `retrieved.items` | `readout` | weak coupling, mainly formats TiM retrieval trace | can be swapped more freely than TiM retrieval |

TiM bundle recommendation:

- safe bundle: `TimThoughtRepresentation + TimThoughtMemoryOrganization + TimBudgetEvolutionTrigger + TimThoughtMemoryEvolution + TimThoughtMemoryRetrieval`
- semi-free front-half search: `TimThoughtUnitFormation + TimThoughtWriteTrigger`
- do not search `TimThoughtMemoryRetrieval` as an isolated retriever over generic stores

### Reflexion

| Slot | Module | Requires | Produces | Hidden semantic coupling | Search advice |
| --- | --- | --- | --- | --- | --- |
| `organization` | `ReflexionTrialOrganization` | `units`, `decisions` | `placements` | intentionally does not append trial records | safe if you want Reflexion-style trial handling |
| `evolution_trigger` | `TrialFailureEvolutionTrigger` | `units`, `observation.metadata` | `evolution_decisions` | relies on `metadata.reflexion` task/result fields | search only in feedback-aware tasks |
| `memory_evolution` | `ReflectionMemoryEvolution` | declared reflection layer, `units`, `placements`, `evolution_decisions`, `observation.metadata` | reflection records + trace | relies on question, scratchpad, feedback schema in observation metadata | bundle with Reflexion trigger |
| `retrieval` | `ReflexionMemoryRetrieval` | declared reflection layer, `query.text` | `retrieved` | simple bounded memory-buffer semantics | relatively portable if reflection layer exists |
| `readout` | `ReflexionContextReadout` / `ReflexionPrependedReadout` | declared reflection layer, `query.text`, `retrieved.items` | `readout` | assumes retrieved items are reflections for next-trial context | portable inside reflection-memory workflows |

Reflexion bundle recommendation:

- safe bundle: `TrialFailureEvolutionTrigger + ReflectionMemoryEvolution + ReflexionMemoryRetrieval + ReflexionPrependedReadout`
- easiest hybridization path: pair Reflexion back-half with generic or baseline front-half modules
- main non-topology constraint is observation metadata schema, not store shape

### MemGPT

| Slot | Module | Requires | Produces | Hidden semantic coupling | Search advice |
| --- | --- | --- | --- | --- | --- |
| `organization` | `MemGPTKeyedUpsertOrganization` | declared target layer, `units`, `decisions` | `placements` | each written unit must carry `unit.metadata[key_name]`; semantics assume block-like keyed memory | only search inside keyed-memory families |
| `retrieval` | `MemGPTPagedRetrieval` | declared target layer, `query.text`, records with embeddings | `retrieved` | expects page/page_size style querying and MemGPT-like target layers | do not treat as a generic embedding retriever |
| `readout` | `MemGPTSearchReadout` | `retrieved.items` | JSON search payload | assumes tool-style search output, not plain recall text | search only when downstream agent expects tool payloads |

MemGPT topology recommendation:

- `core_memory`
- `working_memory`
- `conversation_queue`
- `recall_storage`
- `archival_memory`

MemGPT bundle recommendation:

- safe bundle: `MemGPTKeyedUpsertOrganization + MemGPTPagedRetrieval + MemGPTSearchReadout`
- do not mix freely with generic readouts unless you explicitly want to abandon MemGPT's tool-facing interface
- do not put MemGPT modules into stores that are not layer-named and role-partitioned in the MemGPT style

### A-MEM

| Slot | Module | Requires | Produces | Hidden semantic coupling | Search advice |
| --- | --- | --- | --- | --- | --- |
| `representation` | `AMEMAgenticRepresentation` | `units`, classic runtime LLM | `units.embedding`, `units.metadata.amem`, `units.metadata.representation` | produces agentic note schema with context, tags, keywords, category | should stay with A-MEM downstream |
| `write_trigger` | `AMEMAgenticWriteTrigger` | `units`, classic runtime LLM if enabled | `decisions` | expects A-MEM metadata on units | moderately tied to A-MEM representation |
| `organization` | `AMEMAgenticOrganization` | graph layer, indices `graph/vector/keyword/tag`, `units`, `decisions` | `placements` | writes graph-structured record metadata and link-ready schema | not portable outside graph family |
| `evolution_trigger` | `AMEMAgenticEvolutionTrigger` | vector index, `units`, `placements` | `evolution_decisions` | expects unit embeddings and candidate-neighbor semantics | tightly tied to A-MEM representation/organization |
| `memory_evolution` | `AMEMAgenticEvolution` | graph layer, vector index, classic runtime LLM, aligned `units/placements/evolution_decisions` | graph-link and rewrite effects | depends on A-MEM note payloads and graph-neighbor semantics | strongly bundled |
| `retrieval` | `AMEMEnhancedRetrieval` | graph layer, vector index, `query.text` | `retrieved` | assumes graph links, optional neighbor expansion, optional agentic rerank/query expansion | strongly bundled |
| `readout` | `AMEMAgenticReadout` | `query.text`, `retrieved.items` | `readout` | formats A-MEM note payloads and retrieval trace | keep with A-MEM retrieval |

A-MEM topology recommendation:

- one declared `Graph` layer
- indices include at least `graph` and `vector`
- usually also `keyword` and `tag`

A-MEM bundle recommendation:

- safe bundle: `AMEMAgenticRepresentation + AMEMAgenticOrganization + AMEMAgenticEvolutionTrigger + AMEMAgenticEvolution + AMEMEnhancedRetrieval + AMEMAgenticReadout`
- optional swap point: `AMEMAgenticWriteTrigger` can be replaced by simpler write policies if you intentionally weaken paper alignment
- do not search A-MEM retrieval/evolution as isolated generic modules

## What the Current Runtime Can Validate

The current runtime can already validate:

- slot type compatibility
- `ModuleSpec.slot` correctness
- required store indices such as `index:graph` and `index:vector`
- required target-layer shape such as `target_layer_shape:Graph`
- existence of declared family-specific layers through `validate_store(...)`

This is enough to reject many bad mixes early.

Examples already covered by tests:

- Reflexion requires a declared reflection layer
- MemGPT modules require the declared budget/search layers
- TiM modules require the declared `thought_memory` layer
- A-MEM modules require graph shape and graph index

## What the Current Runtime Cannot Validate Yet

These are the main hidden couplings you still need to surface for robust search:

- unit metadata namespaces such as `metadata.tim`, `metadata.amem`, `metadata.reflexion`
- required per-unit keys such as `memgpt_key`
- record metadata shapes such as graph links, TiM bucket ids, A-MEM note payloads
- query metadata contracts such as MemGPT paging
- family-specific trace contracts consumed downstream
- whether a module is conceptually generic, family-specific, or "paper-faithful only"

## Recommended Search Policy

Use three search granularities instead of one.

### 1. Fully free modules

Mostly baseline modules and a few weakly coupled readouts.

### 2. Semi-coupled bundles

Use this for:

- Reflexion back-half family
- TiM front-half or TiM readout-only variants

### 3. Strong family bundles

Use this for:

- MemGPT family
- A-MEM family
- TiM retrieval/evolution family

## Suggested Explicit Constraints To Add

To make search safe, each candidate module should eventually declare additional searchable metadata beyond `ModuleSpec`.

Recommended fields:

- `family_id`
- `coupling_level`
- `requires_unit_metadata`
- `produces_unit_metadata`
- `requires_record_metadata`
- `produces_record_metadata`
- `requires_query_metadata`
- `topology_family`
- `safe_search_mode`

Example values:

- `family_id="tim"`
- `coupling_level="high"`
- `requires_unit_metadata=("tim.group_id",)`
- `produces_record_metadata=("graph.links",)`
- `topology_family="graph_memory"`
- `safe_search_mode="bundle_only"`

## Practical Rule Of Thumb

If a module only depends on slot IO, it can enter ordinary module search.

If a module depends on family-specific metadata or a family-specific store shape, it should enter constrained search.

If a module depends on both family metadata and family topology, it should enter bundle search only.
