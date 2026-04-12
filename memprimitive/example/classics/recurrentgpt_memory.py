"""Mechanism-level reconstruction of RecurrentGPT with the repo-style loop.

This file intentionally aligns to the released `aiwaves-cn/RecurrentGPT` code
path more than to the paper's stricter prose.

Important alignment note:

- This implementation is a reconstruction of the released repo behavior, not a
  paper-first normalization of RecurrentGPT.
- Long-term memory follows the released repo's actual code path: it stores raw
  prior paragraphs in a vector-backed memory and retrieves those paragraphs
  directly. The paper/README often describe long-term memory as storing
  summaries of prior paragraphs instead.
- Long-term memory write timing also follows the released repo: bootstrap seeds
  long memory with only the first two paragraphs, and later writer steps append
  the previous input paragraph after generation. This is not the paper's
  cleaner "generate paragraph -> append its summary in the same step" story.
- The recurrent content state also follows the released repo: the human
  simulator extends the writer's new paragraph, and that extended paragraph is
  what gets fed into the next writer step and later appended into long memory.
  The paper's wording makes the human role sound closer to plan selection and
  revision only.
- Where this file differs from the paper in those ways, treat that as an
  upstream paper/repo inconsistency in the original RecurrentGPT release, not
  as a defect in this reconstruction.

1. long-term memory stores prior paragraphs directly in a vector-backed layer,
2. the current plan/instruction queries that paragraph memory,
3. the writer prompt jointly produces a new paragraph, a rewritten short memory,
   and three next-step plans,
4. a lightweight "human simulator" selects one plan, extends the new paragraph,
   and rewrites the next-step plan, and
5. the paragraph appended to long-term memory is the previous human-finalized
   paragraph, on the next writer step, matching the upstream control flow.

This keeps the causal memory loop faithful to the repo while still reusing the
framework's existing primitives for memory storage and retrieval.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import AppendOrganization, BasicRepresentation, ConcatenateReadout, EmbeddingSimilarityRetrieval, RecencyRetrieval
from memprimitive.utils._runtime import get_runtime


DEFAULT_INIT_PROMPT = (
    "Please write a {type} novel about {topic} with about 50 chapters. Follow the format below precisely:\n\n"
    "Begin with the name of the novel.\n"
    "Next, write an outline for the first chapter. The outline should describe the background and the beginning of the novel.\n"
    "Write the first three paragraphs with their indication of the novel based on your outline.\n"
    "Write in a novelistic style and take your time to set the scene.\n"
    "Write a summary that captures the key information of the three paragraphs.\n"
    "Finally, write three different instructions for what to write next, each containing around five sentences.\n"
    "Each instruction should present a possible, interesting continuation of the story.\n"
    "The output format should follow these guidelines:\n"
    "Name:\n"
    "Outline:\n"
    "Paragraph 1:\n"
    "Paragraph 2:\n"
    "Paragraph 3:\n"
    "Summary:\n"
    "Instruction 1:\n"
    "Instruction 2:\n"
    "Instruction 3:\n\n"
    "Make sure to be precise and follow the output format strictly.\n"
    "Formatting rules:\n"
    "- Output plain text only.\n"
    "- Do not use Markdown, bullets, numbering, bold markers, or code fences.\n"
    "- Keep the field labels exactly as written above.\n"
    "- Put each field label on its own line, followed by its content."
)

DEFAULT_WRITER_PROMPT_TEMPLATE = (
    "I need you to help me write a novel.\n"
    "Now I give you a memory (a brief summary) of around 400 words. Use it to store the key content of what has been written so that you can keep track of very long context.\n"
    "For each turn, I will give you the current short memory, the previously written paragraph, instructions on what to write next, and related long-term memory paragraphs.\n"
    "I need you to write:\n"
    "1. Output Paragraph: the next paragraph of the novel. The output paragraph should contain around 20 sentences and should follow the input instructions.\n"
    "2. Output Memory: First explain what in the input memory is no longer necessary and why, then explain what needs to be added and why. After that, write Updated Memory.\n"
    "The updated memory should only store key information and should never exceed 20 sentences or 500 words.\n"
    "3. Output Instruction: provide 3 different possible interesting continuations, each around 5 sentences.\n\n"
    "Here are the inputs:\n"
    "Input Memory: {short_memory}\n"
    "Input Paragraph: {input_paragraph}\n"
    "Input Instruction: {input_instruction}\n"
    "Input Related Paragraphs: {related_paragraphs}\n\n"
    "Now start writing and strictly follow this format:\n"
    "Output Paragraph:\n"
    "<the paragraph text>\n"
    "Output Memory:\n"
    "Rational:\n"
    "<what is no longer necessary and what should be added>\n"
    "Updated Memory:\n"
    "<the rewritten short memory>\n"
    "Output Instruction:\n"
    "Instruction 1:\n"
    "<plan 1>\n"
    "Instruction 2:\n"
    "<plan 2>\n"
    "Instruction 3:\n"
    "<plan 3>\n\n"
    "Formatting rules you must obey exactly:\n"
    "- Output plain text only.\n"
    "- Do not use Markdown.\n"
    "- Do not use bold markers such as **.\n"
    "- Do not use bullets, numbering, XML, JSON, or code fences.\n"
    "- Keep the labels exactly as written: Output Paragraph, Output Memory, Rational, Updated Memory, Output Instruction, Instruction 1, Instruction 2, Instruction 3.\n"
    "- Write each label on its own line and put the content on the following lines.\n"
    "- Do not rename labels.\n"
    "- Do not omit any section.\n"
    "- Do not merge Instruction 1/2/3 together.\n"
    "- End after Instruction 3. Do not add any note, explanation, or extra heading.\n\n"
    "Write like a novelist and do not move too fast when writing the next instructions. "
    "Remember the chapter will contain many paragraphs and the novel many chapters, so leave room for future stories.\n"
    "{new_character_prompt}"
)

DEFAULT_PLAN_SELECTION_PROMPT_TEMPLATE = (
    "Now imagine you are a helpful assistant helping a novelist with decision making.\n"
    "You will be given a previously written paragraph, a paragraph written by a ChatGPT writing assistant, a summary of the main storyline maintained by the assistant, and 3 different possible plans of what to write next.\n"
    "Select the most interesting and suitable plan proposed by the assistant.\n\n"
    "Previously written paragraph:\n{previous_paragraph}\n\n"
    "The summary of the main storyline maintained by the assistant:\n{memory}\n\n"
    "The new paragraph written by the assistant:\n{writer_new_paragraph}\n\n"
    "Three plans of what to write next proposed by the assistant:\n{plans}\n\n"
    "Now start choosing and strictly follow this format:\n"
    "Selected Plan:\n"
    "<copy exactly one of the three candidate plans>\n"
    "Reason:\n"
    "<one short reason>\n\n"
    "Formatting rules:\n"
    "- Output plain text only.\n"
    "- Do not use Markdown, bullets, numbering, bold markers, or code fences.\n"
    "- Keep the labels exactly as written above.\n"
)

DEFAULT_HUMAN_PROMPT_TEMPLATE = (
    "Now imagine you are a novelist writing with the help of ChatGPT.\n"
    "You will be given a previously written paragraph, a paragraph written by your ChatGPT assistant, a summary of the main storyline maintained by your assistant, and one selected plan for what to write next.\n"
    "I need you to write:\n"
    "1. Extended Paragraph: extend the new paragraph written by the ChatGPT assistant to about twice its length.\n"
    "2. Selected Plan: copy the plan proposed by the ChatGPT assistant.\n"
    "3. Revised Plan: revise the selected plan into an outline of the next paragraph.\n\n"
    "Previously written paragraph:\n{previous_paragraph}\n\n"
    "The summary of the main storyline maintained by your assistant:\n{memory}\n\n"
    "The new paragraph written by your assistant:\n{writer_new_paragraph}\n\n"
    "The selected plan of what to write next:\n{selected_plan}\n\n"
    "Now start writing and strictly follow this format:\n"
    "Extended Paragraph:\n"
    "<the extended paragraph>\n"
    "Selected Plan:\n"
    "<copy the selected plan>\n"
    "Revised Plan:\n\n"
    "<the revised next-paragraph outline>\n"
    "Formatting rules:\n"
    "- Output plain text only.\n"
    "- Do not use Markdown, bullets, numbering, bold markers, or code fences.\n"
    "- Keep the labels exactly as written above.\n"
    "- Write each label on its own line and put the content on following lines.\n"
    "- Do not add any extra heading or trailing note.\n\n"
    "Write like a novelist and do not move too fast when revising the plan for the next paragraph. "
    "Leave room for future stories."
)


@dataclass
class RecurrentGPTState:
    """Current orchestration state for the repo-style RecurrentGPT loop."""

    title: str
    outline: str
    short_memory: str
    current_paragraph: str
    current_instruction: str
    previous_context: str
    candidate_instructions: list[str] = field(default_factory=list)
    step_index: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

def _sanitize_model_output(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    replacements = {
        "：": ":",
        "**Output Paragraph**": "Output Paragraph",
        "**Output Memory**": "Output Memory",
        "**Updated Memory**": "Updated Memory",
        "**Output Instruction**": "Output Instruction",
        "**Instruction 1**": "Instruction 1",
        "**Instruction 2**": "Instruction 2",
        "**Instruction 3**": "Instruction 3",
        "### Output Paragraph": "Output Paragraph",
        "### Output Memory": "Output Memory",
        "### Updated Memory": "Updated Memory",
        "### Output Instruction": "Output Instruction",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"^[ \t>*-]+", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"(?m)^(\d+\.\s*)(Output Paragraph|Output Memory|Updated Memory|Output Instruction|Instruction 1|Instruction 2|Instruction 3)\b", r"\2", normalized)
    normalized = re.sub(r"(?m)^(Output Paragraph|Output Memory|Updated Memory|Output Instruction|Instruction 1|Instruction 2|Instruction 3)\s*$", r"\1:", normalized)
    normalized = re.sub(r"(?m)^(Output Paragraph|Output Memory|Updated Memory|Output Instruction|Instruction 1|Instruction 2|Instruction 3)\s*:\s*", r"\1: ", normalized)
    return normalized


def _parse_labeled_sections(text: str, labels: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    normalized = _sanitize_model_output(text)
    positions: list[tuple[int, str, str]] = []
    for label in labels:
        marker = f"{label}:"
        index = normalized.find(marker)
        if index >= 0:
            positions.append((index, label, marker))
    positions.sort()

    sections: dict[str, str] = {}
    for position_index, (start, label, marker) in enumerate(positions):
        content_start = start + len(marker)
        content_end = positions[position_index + 1][0] if position_index + 1 < len(positions) else len(normalized)
        sections[label] = normalized[content_start:content_end].strip()
    return normalized, sections


def _require_labeled_sections(
    sections: dict[str, str],
    *,
    required: tuple[str, ...],
    error_message: str,
    normalized_text: str,
) -> None:
    missing = [label for label in required if not sections.get(label, "").strip()]
    if not missing:
        return
    preview = normalized_text[:1200]
    raise ValueError(f"{error_message}\nMissing sections: {missing}.\nResponse preview:\n{preview}")


def _parse_init_output(text: str) -> dict[str, Any]:
    normalized, sections = _parse_labeled_sections(
        text,
        (
            "Name",
            "Outline",
            "Paragraph 1",
            "Paragraph 2",
            "Paragraph 3",
            "Summary",
            "Instruction 1",
            "Instruction 2",
            "Instruction 3",
        ),
    )
    _require_labeled_sections(
        sections,
        required=("Name", "Outline", "Paragraph 1", "Paragraph 2", "Paragraph 3", "Summary"),
        error_message="Failed to parse initialization output with the expected RecurrentGPT format.",
        normalized_text=normalized,
    )
    instructions = [
        sections.get("Instruction 1", "").strip(),
        sections.get("Instruction 2", "").strip(),
        sections.get("Instruction 3", "").strip(),
    ]
    instructions = [item for item in instructions if item]
    parsed = {
        "name": sections["Name"],
        "outline": sections["Outline"],
        "paragraph_1": sections["Paragraph 1"],
        "paragraph_2": sections["Paragraph 2"],
        "paragraph_3": sections["Paragraph 3"],
        "summary": sections["Summary"],
        "instructions": instructions,
    }
    if len(parsed["instructions"]) != 3:
        raise ValueError("Initialization output must contain exactly 3 instructions.")
    return parsed


def _parse_writer_output(text: str) -> dict[str, Any]:
    normalized, sections = _parse_labeled_sections(
        text,
        (
            "Output Paragraph",
            "Output Memory",
            "Updated Memory",
            "Output Instruction",
            "Instruction 1",
            "Instruction 2",
            "Instruction 3",
        ),
    )
    paragraph = sections.get("Output Paragraph", "").strip()
    updated_memory = sections.get("Updated Memory", "").strip() or sections.get("Output Memory", "").strip()
    instructions = [
        sections.get("Instruction 1", "").strip(),
        sections.get("Instruction 2", "").strip(),
        sections.get("Instruction 3", "").strip(),
    ]
    instructions = [item for item in instructions if item]
    if not paragraph or not updated_memory or len(instructions) != 3:
        preview = normalized[:1200]
        raise ValueError(
            "Failed to parse writer output with the expected RecurrentGPT format.\n"
            f"Parsed paragraph={bool(paragraph)} updated_memory={bool(updated_memory)} instruction_count={len(instructions)}.\n"
            f"Response preview:\n{preview}"
        )
    return {
        "output_paragraph": paragraph,
        "updated_memory": updated_memory,
        "instructions": instructions,
        "raw_response": text,
    }


def _parse_selected_plan(text: str) -> str:
    normalized, sections = _parse_labeled_sections(text, ("Selected Plan", "Reason"))
    plan = sections.get("Selected Plan", "").strip()
    if not plan:
        first_line = normalized.strip().splitlines()[0] if normalized.strip() else ""
        if first_line.startswith("Selected Plan:"):
            plan = first_line[len("Selected Plan:") :].strip()
    if not plan:
        raise ValueError("Failed to parse selected plan.")
    return plan


def _parse_human_output(text: str) -> dict[str, str]:
    normalized, sections = _parse_labeled_sections(text, ("Extended Paragraph", "Selected Plan", "Revised Plan"))
    _require_labeled_sections(
        sections,
        required=("Extended Paragraph", "Revised Plan"),
        error_message="Failed to parse human-simulator output.",
        normalized_text=normalized,
    )
    return {
        "extended_paragraph": sections["Extended Paragraph"],
        "selected_plan": sections.get("Selected Plan", "").strip(),
        "revised_plan": sections["Revised Plan"],
        "raw_response": text,
    }


def _format_related_paragraphs(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "None"
    return "\n".join(f"Related Paragraph {index}: {line}" for index, line in enumerate(lines, start=1))


def _format_plan_choices(plans: list[str]) -> str:
    return "\n".join(f"Instruction {index}: {plan}" for index, plan in enumerate(plans, start=1))


def _append_log(log_path: str | None, header: str, content: str) -> None:
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{header}\n{content}\n\n")


def _generate_with_parse_retry(
    *,
    prompt: str,
    system_prompt: str,
    parser,
    response_log_path: str | None,
    log_header: str,
    retry_limit: int,
    temperature: float,
):
    runtime = get_runtime()
    last_error: Exception | None = None
    last_response = ""
    for attempt in range(1, retry_limit + 1):
        response = runtime.text(
            system=system_prompt,
            user=prompt,
            temperature=temperature,
        )
        last_response = response
        _append_log(response_log_path, f"{log_header} (attempt {attempt}):", response)
        try:
            return parser(response)
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(
        f"{log_header} failed after {retry_limit} attempts.\n"
        f"Last parse error: {last_error}\n"
        f"Last response preview:\n{str(last_response)[:1600]}"
    )


def build_recurrentgpt_memory_system(
    *,
    related_top_k: int = 2,
    long_memory_layer: str = "long_memory",
    short_memory_layer: str = "short_memory",
    new_character_prob: float = 0.1,
    generation_retry_limit: int = 5,
) -> dict[str, object]:
    """Build a RecurrentGPT-style memory system plus orchestration state."""

    if related_top_k <= 0:
        raise ValueError("related_top_k must be positive.")
    if not 0.0 <= float(new_character_prob) <= 1.0:
        raise ValueError("new_character_prob must be in [0, 1].")
    if int(generation_retry_limit) <= 0:
        raise ValueError("generation_retry_limit must be positive.")

    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name=long_memory_layer,
                theme="semantic",
                indices=("temporal", "vector"),
                settings={"embedding": {"enabled": True, "mode": "text", "refresh_on_update": "semantic_text_change"}},
            ),
            StoreLayerSpec(
                name=short_memory_layer,
                theme="working",
                indices=("temporal",),
                capacity="sliding_window",
                settings={"record_budget": 1},
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    long_memory_write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer=long_memory_layer),
        store=store,
    )
    short_memory_write_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer=short_memory_layer),
        store=store,
    )
    long_memory_recall_pipeline = MemoryPipeline(
        retrieval=EmbeddingSimilarityRetrieval(top_k=related_top_k, layer=long_memory_layer),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )
    short_memory_recall_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer=short_memory_layer),
        readout=ConcatenateReadout(separator="\n"),
        store=store,
    )

    return {
        "store": store,
        "long_memory_write_pipeline": long_memory_write_pipeline,
        "short_memory_write_pipeline": short_memory_write_pipeline,
        "long_memory_recall_pipeline": long_memory_recall_pipeline,
        "short_memory_recall_pipeline": short_memory_recall_pipeline,
        "long_memory_layer": long_memory_layer,
        "short_memory_layer": short_memory_layer,
        "related_top_k": related_top_k,
        "new_character_prob": float(new_character_prob),
        "generation_retry_limit": int(generation_retry_limit),
    }


def current_short_memory(system: dict[str, object]) -> str:
    """Return the latest short-memory text, if any."""

    store = system["store"]
    assert isinstance(store, MemoryStore)
    short_memory_layer = str(system["short_memory_layer"])
    records = store.iter_records(short_memory_layer)
    return records[-1].text if records else ""


def recall_related_paragraphs(system: dict[str, object], *, instruction: str) -> str:
    """Retrieve RecurrentGPT-style related paragraphs from long-term memory."""

    long_memory_recall_pipeline = system["long_memory_recall_pipeline"]
    assert isinstance(long_memory_recall_pipeline, MemoryPipeline)
    query = Query(text=instruction, embedding=list(get_runtime().embed(instruction)))
    return long_memory_recall_pipeline.recall(query).text


def _write_short_memory(system: dict[str, object], *, text: str, step_index: int) -> None:
    short_memory_write_pipeline = system["short_memory_write_pipeline"]
    assert isinstance(short_memory_write_pipeline, MemoryPipeline)
    short_memory_write_pipeline.ingest(
        Observation(
            text=text,
            source="recurrentgpt_short_memory",
            metadata={"step_index": step_index},
        )
    )


def _append_long_memory(system: dict[str, object], *, text: str, step_index: int) -> None:
    long_memory_write_pipeline = system["long_memory_write_pipeline"]
    assert isinstance(long_memory_write_pipeline, MemoryPipeline)
    long_memory_write_pipeline.ingest(
        Observation(
            text=text,
            source="recurrentgpt_paragraph",
            metadata={"step_index": step_index},
        )
    )


def initialize_recurrentgpt_story(
    system: dict[str, object],
    *,
    topic: str,
    story_type: str = "science fiction",
    init_prompt_template: str = DEFAULT_INIT_PROMPT,
    response_log_path: str | None = None,
) -> dict[str, Any]:
    """Generate the repo-style initial story seed."""

    prompt = init_prompt_template.format(type=story_type, topic=topic)
    return _generate_with_parse_retry(
        prompt=prompt,
        system_prompt="You initialize RecurrentGPT long-form story generation. Follow the requested output format exactly.",
        parser=_parse_init_output,
        response_log_path=response_log_path,
        log_header="Initialization output",
        retry_limit=int(system.get("generation_retry_limit", 5)),
        temperature=0.7,
    )


def select_plan(
    *,
    previous_paragraph: str,
    writer_new_paragraph: str,
    memory: str,
    candidate_plans: list[str],
    response_log_path: str | None = None,
) -> str:
    """Select one next-step plan using the repo-style human simulator prompt."""

    prompt = DEFAULT_PLAN_SELECTION_PROMPT_TEMPLATE.format(
        previous_paragraph=previous_paragraph,
        writer_new_paragraph=writer_new_paragraph,
        memory=memory,
        plans=_format_plan_choices(candidate_plans),
    )
    selected_plan = _generate_with_parse_retry(
        prompt=prompt,
        system_prompt="You select the most suitable next plot plan. Follow the requested output format exactly.",
        parser=_parse_selected_plan,
        response_log_path=response_log_path,
        log_header="Selected plan",
        retry_limit=5,
        temperature=0.2,
    )
    if selected_plan in candidate_plans:
        return selected_plan
    for candidate in candidate_plans:
        if selected_plan in candidate or candidate in selected_plan:
            return candidate
    return candidate_plans[0]


def human_step(
    *,
    previous_paragraph: str,
    writer_new_paragraph: str,
    memory: str,
    selected_plan: str,
    response_log_path: str | None = None,
) -> dict[str, str]:
    """Run the repo-style human simulator that extends the paragraph and revises the plan."""

    prompt = DEFAULT_HUMAN_PROMPT_TEMPLATE.format(
        previous_paragraph=previous_paragraph,
        writer_new_paragraph=writer_new_paragraph,
        memory=memory,
        selected_plan=selected_plan,
    )
    return _generate_with_parse_retry(
        prompt=prompt,
        system_prompt="You act as the human collaborator in RecurrentGPT. Follow the requested output format exactly.",
        parser=_parse_human_output,
        response_log_path=response_log_path,
        log_header="Human output",
        retry_limit=5,
        temperature=0.7,
    )


def bootstrap_recurrentgpt_story(
    system: dict[str, object],
    *,
    topic: str,
    story_type: str = "science fiction",
    response_log_path: str | None = None,
) -> RecurrentGPTState:
    """Run the repo-style initialization plus the initial human-simulator step."""

    init_output = initialize_recurrentgpt_story(
        system,
        topic=topic,
        story_type=story_type,
        response_log_path=response_log_path,
    )

    _append_long_memory(system, text=init_output["paragraph_1"], step_index=0)
    _append_long_memory(system, text=init_output["paragraph_2"], step_index=0)
    _write_short_memory(system, text=init_output["summary"], step_index=0)

    previous_context = f"{init_output['paragraph_1']}\n{init_output['paragraph_2']}"
    selected_plan = select_plan(
        previous_paragraph=previous_context,
        writer_new_paragraph=init_output["paragraph_3"],
        memory=init_output["summary"],
        candidate_plans=init_output["instructions"],
        response_log_path=response_log_path,
    )
    human_output = human_step(
        previous_paragraph=previous_context,
        writer_new_paragraph=init_output["paragraph_3"],
        memory=init_output["summary"],
        selected_plan=selected_plan,
        response_log_path=response_log_path,
    )

    state = RecurrentGPTState(
        title=init_output["name"],
        outline=init_output["outline"],
        short_memory=init_output["summary"],
        current_paragraph=human_output["extended_paragraph"],
        current_instruction=human_output["revised_plan"],
        previous_context=previous_context,
        candidate_instructions=list(init_output["instructions"]),
        step_index=0,
        history=[
            {
                "phase": "bootstrap",
                "init_output": init_output,
                "selected_plan": selected_plan,
                "human_output": human_output,
            }
        ],
    )
    return state


def writer_step(
    system: dict[str, object],
    state: RecurrentGPTState,
    *,
    response_log_path: str | None = None,
) -> dict[str, Any]:
    """Run one repo-style writer step."""

    import random

    related_paragraphs = recall_related_paragraphs(system, instruction=state.current_instruction)
    new_character_prompt = (
        "If it is reasonable, you can introduce a new character in the output paragraph and add it into the memory."
        if random.random() < float(system["new_character_prob"])
        else ""
    )
    prompt = DEFAULT_WRITER_PROMPT_TEMPLATE.format(
        short_memory=state.short_memory,
        input_paragraph=state.current_paragraph,
        input_instruction=state.current_instruction,
        related_paragraphs=_format_related_paragraphs(related_paragraphs),
        new_character_prompt=new_character_prompt,
    )
    parsed = _generate_with_parse_retry(
        prompt=prompt,
        system_prompt="You are the RecurrentGPT writing assistant. Follow the requested output format exactly.",
        parser=_parse_writer_output,
        response_log_path=response_log_path,
        log_header="Writer output",
        retry_limit=int(system.get("generation_retry_limit", 5)),
        temperature=0.7,
    )
    parsed["retrieved_related_paragraphs"] = related_paragraphs
    return parsed


def run_recurrentgpt_iteration(
    system: dict[str, object],
    state: RecurrentGPTState,
    *,
    response_log_path: str | None = None,
) -> RecurrentGPTState:
    """Advance the repo-style writer/human loop by one full iteration."""

    writer_output = writer_step(system, state, response_log_path=response_log_path)

    _append_long_memory(system, text=state.current_paragraph, step_index=state.step_index + 1)
    _write_short_memory(system, text=writer_output["updated_memory"], step_index=state.step_index + 1)

    selected_plan = select_plan(
        previous_paragraph=state.current_paragraph,
        writer_new_paragraph=writer_output["output_paragraph"],
        memory=writer_output["updated_memory"],
        candidate_plans=writer_output["instructions"],
        response_log_path=response_log_path,
    )
    human_output = human_step(
        previous_paragraph=state.current_paragraph,
        writer_new_paragraph=writer_output["output_paragraph"],
        memory=writer_output["updated_memory"],
        selected_plan=selected_plan,
        response_log_path=response_log_path,
    )

    state.short_memory = writer_output["updated_memory"]
    state.previous_context = state.current_paragraph
    state.current_paragraph = human_output["extended_paragraph"]
    state.current_instruction = human_output["revised_plan"]
    state.candidate_instructions = list(writer_output["instructions"])
    state.step_index += 1
    state.history.append(
        {
            "phase": "iteration",
            "step_index": state.step_index,
            "writer_output": writer_output,
            "selected_plan": selected_plan,
            "human_output": human_output,
        }
    )
    return state


def run_recurrentgpt_loop(
    system: dict[str, object],
    *,
    topic: str,
    story_type: str = "science fiction",
    iterations: int = 3,
    response_log_path: str | None = None,
) -> RecurrentGPTState:
    """Run bootstrap plus several repo-style iterations."""

    state = bootstrap_recurrentgpt_story(
        system,
        topic=topic,
        story_type=story_type,
        response_log_path=response_log_path,
    )
    for _ in range(iterations):
        state = run_recurrentgpt_iteration(
            system,
            state,
            response_log_path=response_log_path,
        )
    return state


def main() -> None:
    system = build_recurrentgpt_memory_system()
    state = run_recurrentgpt_loop(
        system,
        topic="a distant beacon awakening an abandoned city",
        story_type="science fiction",
        iterations=1,
    )

    store = system["store"]
    assert isinstance(store, MemoryStore)

    print("title:")
    print(state.title)
    print()

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("current short memory:")
    print(current_short_memory(system))
    print()

    print("latest current paragraph:")
    print(state.current_paragraph)
    print()

    print("current instruction:")
    print(state.current_instruction)
    print()

    print("long-memory paragraphs:")
    pprint([record.text for record in store.iter_records("long_memory")])


if __name__ == "__main__":
    main()
