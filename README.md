# MemPrimitive

MemPrimitive is a research framework for agent memory systems. Instead of presenting one more paper-specific memory architecture, it treats memory as a composition of reusable primitives that can be re-expressed, compared, and eventually searched inside one shared design space.

The repository currently provides:

- a strict slot-based runtime centered on `MemoryPipeline`
- baseline primitive modules for ingest, maintenance, and recall
- a declarative YAML config loader
- executable examples and paper-style reconstructions
- a small benchmark harness for early evaluation work

## What is MemPrimitive?

MemPrimitive starts from a simple question:

Can we describe agent memory at the mechanism level, in a unified and composable way, instead of as isolated end-to-end methods?

Most memory systems repeat the same core operations:

- form memory units from observations
- represent those units in a structured or retrievable way
- decide what should be written
- organize or update stored memory
- retrieve relevant records later
- turn recalled state into agent-usable context

MemPrimitive makes those operations explicit as primitive slots:

- `unit_formation`
- `representation`
- `write_trigger`
- `organization`
- `evolution_trigger`
- `memory_evolution`
- `retrieval`
- `readout`

This is the core idea behind [`IDEA.md`](IDEA.md): a memory system is not one monolithic module, but a chain of composable mechanisms. Once memory is expressed that way, the project can support three research goals:

- re-express diverse memory papers in one shared language
- compare mechanisms instead of only comparing whole systems
- move toward constrained search over memory configurations and later motif discovery

So MemPrimitive is best understood as a research-oriented memory DSL and runtime, not just a single memory method.

## Quick Start

### 1. Install dependencies

MemPrimitive requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
```

### 2. Configure the runtime for LLM-backed modules

Some examples and the benchmark answer runner use the OpenAI-compatible runtime in `memprimitive/utils/_runtime.py`. Set these environment variables, or place them in `memprimitive/.env`:

```env
MEMPRIMITIVE_API_KEY=...
MEMPRIMITIVE_BASE_URL=...
MEMPRIMITIVE_MODEL=...
MEMPRIMITIVE_EMBEDDING_PROVIDER=sentence_transformers
MEMPRIMITIVE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Notes:

- Embeddings default to local `sentence-transformers`; `MEMPRIMITIVE_EMBEDDING_MODEL` is optional in that mode and defaults to `sentence-transformers/all-MiniLM-L6-v2`.
- To use an OpenAI-compatible embeddings API instead, set `MEMPRIMITIVE_EMBEDDING_PROVIDER=openai` plus `MEMPRIMITIVE_EMBEDDING_API_KEY`, `MEMPRIMITIVE_EMBEDDING_BASE_URL`, and `MEMPRIMITIVE_EMBEDDING_MODEL`.
- Simple config validation does not need model credentials.
- End-to-end execution of LLM-backed examples does need them.

### 3. Load a pipeline from YAML config

The config bridge lives in `memprimitive/config/` and builds runnable object graphs from a single YAML file.

Python API:

```python
from memprimitive import Observation, Query
from memprimitive.config import load_pipeline_from_yaml

pipeline = load_pipeline_from_yaml("memprimitive/example/config/simple_pipeline.yml")
pipeline.ingest(Observation(text="Alice likes jasmine tea.", source="dialogue"))
readout = pipeline.recall(Query(text="What does Alice like?"))

print(readout.text)
```

CLI validation:

```bash
python -m memprimitive.config validate memprimitive/example/config/simple_pipeline.yml
```

The loader currently supports three special directives:

- `$call` to import and call a class or function
- `$import` to import a named symbol without calling it
- `$ref` to reuse a previously declared object instance

### 4. Try the shipped example configs

The fastest way to see the config surface is to use the example files under `memprimitive/example/config/`.

`simple_pipeline.yml` is the minimal baseline path:

- pass-through unit formation
- basic text representation
- append organization
- recency retrieval
- concatenate readout

`recalled_prompt_pipeline.yml` is a richer example:

- nested child recall pipeline
- imported query-builder function
- recalled-context prompt composition
- LLM-backed representation

Validate both examples from the CLI:

```bash
python -m memprimitive.config validate memprimitive/example/config/simple_pipeline.yml
python -m memprimitive.config validate memprimitive/example/config/recalled_prompt_pipeline.yml
```

Load one of them directly from Python:

```python
from memprimitive.config import load_pipeline_from_yaml

pipeline = load_pipeline_from_yaml(
    "memprimitive/example/config/recalled_prompt_pipeline.yml"
)
```

Beyond config examples, the repository also includes:

- `memprimitive/example/demonstration/` for focused runnable demos
- `memprimitive/example/classics/` for paper-style memory reconstructions such as Mem0, A-MEM, Reflexion, RecurrentGPT, and RET-LLM-style memory slices

### 5. Run evaluation from the CLI

The early benchmark harness lives in `memprimitive/benchmarking/`, and the current CLI exposes a minimal single-recall baseline.

Before running it, you need to place the raw benchmark files into the expected folders under `benchmarks/` yourself. The current loaders expect paths such as:

- `benchmarks/LoCoMo/data/locomo10.json`
- `benchmarks/LongMemEval/longmemeval_s_cleaned.json`
- `benchmarks/LongMemEval/longmemeval_m_cleaned.json`
- `benchmarks/LongMemEval/longmemeval_oracle.json`
- `benchmarks/DMR/msc_self_instruct.jsonl`

If those files are missing, the CLI will fail at load time. See `benchmarks/README.md` for the intended dataset layout.

Example commands:

```bash
python -m memprimitive.benchmarking.minimal_baseline --benchmark locomo --limit 5 --output benchmarks/outputs/locomo_smoke.jsonl
python -m memprimitive.benchmarking.minimal_baseline --benchmark longmemeval --longmemeval-variant s_cleaned --limit 10 --top-k 5 --output benchmarks/outputs/longmemeval_smoke.jsonl
python -m memprimitive.benchmarking.minimal_baseline --benchmark dmr --limit 10 --output benchmarks/outputs/dmr_smoke.jsonl
```

The CLI writes JSONL predictions containing the query, reference answer, predicted answer, retrieved text, and source ids.

Important limitations of the current benchmark path:

- it is intentionally a minimal baseline, not a full paper-faithful evaluation suite
- the default answer runner uses the configured OpenAI-compatible runtime
- embedding retrieval uses the configured sentence-transformer model

## Architecture

The easiest way to think about MemPrimitive is as a LEGO-like memory construction kit. A memory system is built by plugging modules into fixed pipeline slots, then swapping individual pieces without rewriting the whole stack. That is the main architectural bet of the project: reusable parts first, paper-specific wrappers second.

At a high level, MemPrimitive is a stateful dataflow system with an ingest half and a recall half:

```text
Ingest:
Observation
  -> unit_formation
  -> representation
  -> write_trigger
  -> organization
  -> evolution_trigger
  -> memory_evolution
  -> MemoryStore

Recall:
Query
  -> retrieval
  -> readout
  -> Readout
```

The main architectural pieces are:

### 1. Core types and store

`memprimitive/core.py` defines the shared runtime objects:

- `Observation` as external input
- `MemoryUnit` as the intermediate memory representation
- `MemoryRecord` as the stored form
- `Query`, `RetrievedSet`, and `Readout` for the recall path
- `MemoryStore`, `StoreTopology`, and `StoreLayerSpec` for layered memory structure

The store keeps layer/index structure explicit and also records composition contracts registered by modules.

### 2. Pipeline runners

`memprimitive/pipeline.py` exposes two execution styles:

- `MemoryPipeline`, the strict public runner that enforces slot ordering and module compatibility
- `FreeMemoryPipeline`, the permissive ordered runner for experimental compositions

`MemoryPipeline` is the default surface for stable, validated systems.

### 3. Slot-oriented primitives

`memprimitive/baselines/` groups concrete implementations by primitive slot:

- unit formation
- representation
- trigger
- organization
- memory evolution
- retrieval
- readout

Each module carries a `ModuleSpec`, so slot identity, side effects, and composition expectations remain inspectable rather than hidden in ad hoc glue code.

If you want the detailed build reference for those slots and modules, start with [`DSL_REFERENCE.zh-CN.md`](DSL_REFERENCE.zh-CN.md). It is the most complete code-aligned reference for the current public surface.

### 4. Declarative config layer

`memprimitive/config/` is the bridge from YAML to runnable runtime objects. It uses a fixed `version / root / objects` shape and supports:

- callable construction with `$call`
- named imports with `$import`
- shared-object reuse with `$ref`
- nested child pipelines and shared stores

This makes the config layer a practical first step toward a fuller memory DSL.

### 5. Examples and evaluation layer

The repository is organized so that the same primitive runtime can be used in several ways:

- `memprimitive/example/config/` for declarative YAML examples
- `memprimitive/example/demonstration/` for focused runtime demos
- `memprimitive/example/classics/` for paper-style reconstructions
- `memprimitive/benchmarking/` for normalized benchmark adapters, memory adapters, and CLI evaluation

That separation is intentional: the project wants one shared mechanism layer underneath configs, examples, and benchmarking, rather than a different implementation style for each.

### 6. Example: building a pipeline like LEGO

[`memprimitive/example/demonstration/embedding_similarity_retrieval.py`](memprimitive/example/demonstration/embedding_similarity_retrieval.py) is a good small example of the project style:

```python
pipeline = MemoryPipeline(
    unit_formation=PassThroughUnitFormation(),
    representation=BasicRepresentation(elements=("text", "embedding")),
    write_trigger=AlwaysTrigger(),
    organization=AppendOrganization(),
    retrieval=EmbeddingSimilarityRetrieval(top_k=2),
    readout=ConcatenateReadout(),
)
```

This example is useful because each line corresponds to one explicit primitive choice:

- `PassThroughUnitFormation()` keeps each incoming observation as one memory unit.
- `BasicRepresentation(elements=("text", "embedding"))` says each unit should carry text plus an embedding.
- `AlwaysTrigger()` says every formed unit should be written.
- `AppendOrganization()` writes each accepted unit into the store.
- `EmbeddingSimilarityRetrieval(top_k=2)` retrieves the two most semantically similar records at recall time.
- `ConcatenateReadout()` turns the retrieved records into one plain-text readout.

That is the LEGO-like property in practice: if you want a different memory behavior, you change one brick at a time. For example, you can keep the same write path but swap retrieval, or keep the same retrieval and change only readout, without redesigning the whole pipeline.

The same idea scales upward:

- small demos assemble a handful of baseline modules directly in Python
- YAML configs assemble the same kinds of objects declaratively
- classic reconstructions assemble larger paper-style systems from the same shared parts

So the architecture is not just "there are modules"; it is that the repository is organized around assembling memory systems from interchangeable parts with stable slot semantics.
