"""Shared utilities and helpers for MemPrimitive.

Submodules
----------
_runtime
    ``Runtime`` wrapper around OpenAI-compatible LLM and
    sentence-transformer embedding backends, plus the module-level
    singleton ``get_runtime()``.

_amem_family
    A-MEM-style note helpers: note-payload schema repair, retrieval-oriented
    embedding text construction, record<->note payload conversion, and
    embedding-based candidate collection for graph-note pipelines.

_example_dialogue
    Reusable example-level dialogue helpers for building/rendering message
    pairs and recalling prompt context text from a pipeline.

_graph_family
    Graph-family helpers: graph metadata normalization, record rewrite
    utilities, and link-history management for graph baseline modules.

_mem0_family
    Shared Mem0/Mem0g example helpers: per-fact profile recall, fixed-layer
    profile write tools, pair-context builders, and common turn orchestration.

_profile_feature_tools
    MemMachine-style structured profile-feature write tools for
    category/tag/feature/value records with source episode citations.

_reflexion_family
    Reflexion-like helpers: control parsing, prompt-context formatting,
    reflection generation payloads, and strategy constants.

_template_readout
    Template readout helpers: safe render-context projection, lightweight
    template parsing, filter application, and structured block rendering.

_template
    Shared prompt-template helpers for `PromptPlan` construction/rendering,
    sub-recall execution, and stable lightweight projections of
    packet/query/unit/record data.

_trace
    Lightweight packet-trace copy utility.

exceptions
    Custom exceptions for MemPrimitive runtime composition checks
    (``IncompatibleCompositionError``).
"""
