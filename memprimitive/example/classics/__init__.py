"""Classic paper-style examples live under ``memprimitive.example.classics``."""

from .amem_memory import build_amem_memory_system, ingest_note as ingest_amem_note, recall_notes as recall_amem_notes
from .reflexion_memory import (
    build_reflexion_memory_system,
    ingest_failed_trial as ingest_reflexion_failed_trial,
    recall_reflection_context,
    recall_reflections,
)
from .recurrentgpt_memory import (
    bootstrap_recurrentgpt_story,
    build_recurrentgpt_memory_system,
    current_short_memory,
    recall_related_paragraphs,
    run_recurrentgpt_iteration,
    run_recurrentgpt_loop,
)
