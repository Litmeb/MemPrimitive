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

_graph_family
    Graph-family helpers: graph metadata normalization, record rewrite
    utilities, and link-history management for graph baseline modules.

_reflexion_family
    Reflexion-like helpers: control parsing, prompt-context formatting,
    reflection generation payloads, and strategy constants.

_trace
    Lightweight packet-trace copy utility.

exceptions
    Custom exceptions for MemPrimitive runtime composition checks
    (``IncompatibleCompositionError``).
"""
