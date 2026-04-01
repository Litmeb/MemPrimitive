"""Shared helpers for Reflexion-like baseline and classic modules.

This helper exists because the same control parsing and prompt-context
formatting logic is reused across multiple baseline slots plus the
backward-compatible classic Reflexion wrappers. The decomposition is inferred
from the Reflexion family motif guide rather than copied from a paper-defined
module boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final

from ._runtime import get_runtime


DEFAULT_REFLECTION_LAYER: Final[str] = "reflections"
DEFAULT_TRIAL_LAYER: Final[str] = "trial_buffer"
DEFAULT_MEMORY_SIZE: Final[int] = 3

STRATEGY_NONE: Final[str] = "base"
STRATEGY_LAST_ATTEMPT: Final[str] = "last_trial"
STRATEGY_REFLEXION: Final[str] = "reflexion"
STRATEGY_LAST_ATTEMPT_AND_REFLEXION: Final[str] = "last_trial_and_reflexion"
VALID_PROMPT_CONTEXT_STRATEGIES: Final[frozenset[str]] = frozenset(
    {
        STRATEGY_NONE,
        STRATEGY_LAST_ATTEMPT,
        STRATEGY_REFLEXION,
        STRATEGY_LAST_ATTEMPT_AND_REFLEXION,
    }
)

REFLECTION_HEADER: Final[str] = (
    "You have attempted to answer following question before and failed. "
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question."
)
REFLECTION_AFTER_LAST_TRIAL_HEADER: Final[str] = (
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question."
)
LAST_TRIAL_HEADER: Final[str] = (
    "You have attempted to answer the following question before and failed. "
    "Below is the last trial you attempted to answer the question."
)


def normalize_text(value: Any) -> str:
    """Return a whitespace-normalized string representation."""

    return " ".join(str(value).strip().split())


def reflexion_controls(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Collect Reflexion-like control fields from top-level and nested metadata."""

    controls: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return controls

    nested = payload.get("reflexion")
    if isinstance(nested, dict):
        controls.update(nested)

    for key in (
        "question",
        "task",
        "scratchpad",
        "last_attempt",
        "is_correct",
        "success",
        "feedback",
        "evaluator_feedback",
        "answer",
        "trial_index",
        "strategy",
    ):
        if key in payload and key not in controls:
            controls[key] = payload[key]
    return controls


def coerce_bool(value: Any) -> bool | None:
    """Parse common bool-ish values used by trial feedback payloads."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "success", "passed", "correct"}:
            return True
        if normalized in {"false", "no", "0", "failure", "failed", "incorrect"}:
            return False
    return None


def question_from_payload(payload: dict[str, Any]) -> str:
    """Extract the task/question text from observation-like metadata."""

    controls = reflexion_controls(payload)
    return normalize_text(controls.get("question") or controls.get("task") or payload.get("text") or "")


def scratchpad_from_payload(payload: dict[str, Any]) -> str:
    """Extract the current trial trace or last attempt text."""

    controls = reflexion_controls(payload)
    return normalize_text(
        controls.get("scratchpad")
        or controls.get("last_attempt")
        or payload.get("text")
        or ""
    )


def feedback_from_payload(payload: dict[str, Any]) -> str:
    """Extract evaluator feedback text from supported metadata fields."""

    controls = reflexion_controls(payload)
    return normalize_text(controls.get("evaluator_feedback") or controls.get("feedback") or "")


def strategy_from_query_metadata(payload: dict[str, Any] | None, fallback: str) -> str:
    """Resolve prompt-context strategy from query metadata or a fallback."""

    controls = reflexion_controls(payload)
    raw = controls.get("strategy")
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized in VALID_PROMPT_CONTEXT_STRATEGIES:
            return normalized
    return fallback


def last_attempt_from_query_metadata(payload: dict[str, Any] | None) -> str:
    """Extract the last attempt text from query metadata."""

    controls = reflexion_controls(payload)
    return normalize_text(controls.get("last_attempt", ""))


def format_reflections(reflections: list[str], *, header: str = REFLECTION_HEADER) -> str:
    """Render reflection texts into the canonical prompt-context block."""

    if not reflections:
        return ""
    parts = [header]
    for index, reflection in enumerate(reflections, start=1):
        parts.append(f"Reflection {index}:")
        parts.append(normalize_text(reflection))
    return "\n".join(parts).strip()


def format_last_attempt(question: str, scratchpad: str) -> str:
    """Render the canonical last-attempt section for prompt context."""

    return "\n".join(
        [
            LAST_TRIAL_HEADER,
            f"Question: {question}",
            normalize_text(scratchpad),
        ]
    ).strip()


def build_prompt_context(
    *,
    strategy: str,
    question: str,
    last_attempt: str,
    reflections: list[str],
) -> str:
    """Render the final prompt-prepended context for the next trial."""

    sections: list[str] = []
    if strategy == STRATEGY_NONE:
        return ""

    if strategy in {STRATEGY_LAST_ATTEMPT, STRATEGY_LAST_ATTEMPT_AND_REFLEXION} and last_attempt:
        sections.append(format_last_attempt(question, last_attempt))

    if strategy in {STRATEGY_REFLEXION, STRATEGY_LAST_ATTEMPT_AND_REFLEXION} and reflections:
        header = REFLECTION_AFTER_LAST_TRIAL_HEADER if sections else REFLECTION_HEADER
        sections.append(format_reflections(reflections, header=header))

    return "\n\n".join(section for section in sections if section).strip()


@dataclass(slots=True, frozen=True)
class ReflectionGenerationPayload:
    """Normalized inputs for reflection-generation evolution."""

    question: str
    scratchpad: str
    evaluator_feedback: str
    prior_reflections: tuple[str, ...]
    observation_metadata: dict[str, Any]
    unit_metadata: dict[str, Any]


ReflectionGenerator = Callable[[ReflectionGenerationPayload], str]
ReflectionPromptBuilder = Callable[[ReflectionGenerationPayload], tuple[str, str]]


def default_reflection_prompt_builder(payload: ReflectionGenerationPayload) -> tuple[str, str]:
    """Build the default reflection-generation prompt.

    The memory-evolution skeleton stays generic; this prompt wording is the
    benchmark/prompt residual that approximates the public Reflexion setup.
    """

    return (
        "You are an advanced reasoning agent that can improve based on self reflection. "
        "Given a previous reasoning trial, diagnose the likely failure and propose a concise, "
        "high-level plan for the next attempt. Return a short reflection beginning with 'Reflection'.",
        (
            f"question: {payload.question}\n"
            f"previous_trial: {payload.scratchpad}\n"
            f"evaluator_feedback: {payload.evaluator_feedback}\n"
            f"prior_reflections: {list(payload.prior_reflections)}"
        ),
    )


def runtime_reflection_generator(
    payload: ReflectionGenerationPayload,
    *,
    prompt_builder: ReflectionPromptBuilder | None = None,
) -> str:
    """Generate a reflection using the classic runtime and a prompt residual."""

    system, user = (prompt_builder or default_reflection_prompt_builder)(payload)
    return get_runtime().text(system=system, user=user).strip()
