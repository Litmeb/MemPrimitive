from __future__ import annotations

from ..core import Query
from ..pipeline import MemoryPipeline


def build_dialogue_pair_messages(user_text: str, assistant_text: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def render_messages_for_prompt(messages: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).strip() or "unknown"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def recall_context_text(
    pipeline: MemoryPipeline,
    *,
    query_text: str,
) -> str:
    recalled = pipeline.recall(Query(text=query_text))
    return recalled.text.strip()
