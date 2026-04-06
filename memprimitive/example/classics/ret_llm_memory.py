"""Mechanism-level reconstruction of RET-LLM / MemLLM-style explicit memory.

This repo focuses on modularizing memory mechanisms rather than providing a
canonical agent-loop implementation. Accordingly, this example keeps the memory
system inside MemPrimitive pipelines, but runs the final answer step as a normal
agent loop whose ``MEM_READ`` tool simply delegates to ``pipeline.recall(...)``.

It also does not reproduce the paper's unified controller + text-API workflow
in which one fine-tuned model decides when to emit ``MEM_WRITE`` / ``MEM_READ``
calls from raw natural-language input. That behavior depends on the paper's
training setup, and training-faithful reproduction is intentionally out of
scope for this repository.

The paper also does not specify a concrete mechanism for update / temporal
correction beyond the high-level claim that a modifiable memory can absorb new
facts, so this example does not implement a dedicated update policy for that
behavior. In this repository, similar capabilities could instead be assembled
with components such as ``GraphDeduplicationAppendOrganization`` or
``LLMFunctionCallOrganization`` / ``LLMFunctionCallEvolution`` with tools like
``GRAPH_UPDATE``.

The resulting system still follows the intended memory-side structure:

1. informative text is split into sentences,
2. each sentence is converted into relation triples,
3. the triples are stored in an explicit external memory layer, and
4. answer generation can call ``MEM_READ`` to query that memory during the
   outer agent loop.

This remains a mechanism-level reconstruction, not a training-faithful clone of
the paper's later MemLLM setup. In particular, memory still uses graph-shaped
``MemoryRecord`` nodes rather than separate entity / relation tables.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, Readout, RetrievedSet, StoreLayerSpec, StoreTopology
from memprimitive.core import ModuleSpec
from memprimitive.baselines import (
    AlwaysTrigger,
    GraphEntityDeduplicationAppendOrganization,
    SentenceSplitUnitFormation,
    TemplateReadout,
    TripleMemoryRetrieval,
    TripleRepresentation,
)
from memprimitive.interfaces import RetrievalModule
from memprimitive.utils._mid_decoding_tools import (
    ReadoutToolCallContext,
    ToolExecutionState,
    build_runtime_tools,
    normalize_readout_tool_specs,
    project_tool_specs_for_prompt,
)
from memprimitive.utils._runtime import Runtime
from memprimitive.utils._template import structured_prompt, text_prompt
from memprimitive.utils._template import ensure_prompt_plan, render_prompt_plan


def _parse_structured_triple_query(text: str) -> dict[str, str | None]:
    parts = [part.strip() for part in str(text).split(">>")]
    if len(parts) != 3:
        raise ValueError("RET-LLM MEM_READ requires a structured triple query: 'subject >> relation >> object'.")
    return {
        "subject": None if parts[0] in {"", "*"} else parts[0],
        "relation": None if parts[1] in {"", "*"} else parts[1],
        "object": None if parts[2] in {"", "*"} else parts[2],
    }


def _format_structured_triple_query(query: dict[str, str | None]) -> str:
    return " >> ".join(query.get(slot) or "*" for slot in ("subject", "relation", "object"))


def _record_triples(record: Any) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    sources = [
        record.metadata.get("representation", {}).get("triples", []),
        record.metadata.get("graph", {}).get("triples", []),
    ]
    for source in sources:
        if not isinstance(source, list):
            continue
        for value in source:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                continue
            triple = tuple(str(item).strip() for item in value)
            if len(triple) != 3 or not all(triple) or triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
    return triples


def _triple_matches_query_spec(triple: tuple[str, str, str], query_spec: dict[str, str | None]) -> bool:
    subject, relation, obj = triple
    if query_spec.get("subject") is not None and subject.casefold() != str(query_spec["subject"]).casefold():
        return False
    if query_spec.get("relation") is not None and relation.casefold() != str(query_spec["relation"]).casefold():
        return False
    if query_spec.get("object") is not None and obj.casefold() != str(query_spec["object"]).casefold():
        return False
    return True


def _matched_triplet_entries_from_store(
    store: MemoryStore,
    *,
    query_spec: dict[str, str | None],
    layer: str = "triple_memory",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for record in store.iter_records(layer):
        for triple in _record_triples(record):
            if not _triple_matches_query_spec(triple, query_spec):
                continue
            entries.append(
                {
                    "record_id": record.record_id,
                    "triple": triple,
                    "text": f"{triple[0]} >> {triple[1]} >> {triple[2]}",
                }
            )
    return entries


def _matched_triplet_entries(packet: Packet) -> list[dict[str, str]]:
    retrieved = packet.retrieved if packet.retrieved is not None else RetrievedSet()
    entries: list[dict[str, str]] = []
    for score in retrieved.scores:
        if not isinstance(score, dict):
            continue
        record_id = str(score.get("record_id", "")).strip()
        matched_triples = score.get("matched_triples", [])
        if not isinstance(matched_triples, list):
            continue
        for triple in matched_triples:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            normalized = tuple(str(item).strip() for item in triple)
            if len(normalized) != 3 or not all(normalized):
                continue
            entries.append(
                {
                    "record_id": record_id,
                    "text": f"{normalized[0]} >> {normalized[1]} >> {normalized[2]}",
                }
            )
    return entries


class RETLLMTripleMemoryRetrieval(RetrievalModule):
    spec = ModuleSpec(
        name="ret_llm_triple_memory_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, inner: TripleMemoryRetrieval, *, fallback_similarity_threshold: float = 0.7) -> None:
        self.inner = inner
        self.fallback_similarity_threshold = float(fallback_similarity_threshold)

    def get_requires_contracts(self) -> frozenset[str]:
        return self.inner.get_requires_contracts()

    def get_produces_contracts(self) -> frozenset[str]:
        return self.inner.get_produces_contracts()

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("RETLLMTripleMemoryRetrieval requires packet.query.")
        original_query = packet.query
        active_query = original_query
        matched_entries = _matched_triplet_entries_from_store(
            store,
            query_spec=_parse_structured_triple_query(original_query.text),
        )
        if not matched_entries:
            resolved_query = self._resolve_query_terms(original_query, store)
            if resolved_query is not None:
                active_query = resolved_query
                matched_entries = _matched_triplet_entries_from_store(
                    store,
                    query_spec=_parse_structured_triple_query(resolved_query.text),
                )
        packet = replace(
            packet,
            query=active_query,
            retrieved=self._retrieved_set_from_entries(store, matched_entries),
        )
        trace = dict(packet.trace)
        trace["ret_llm"] = {
            "matched_triples": list(matched_entries),
        }
        return replace(packet, trace=trace), store

    def _embed_text(self, text: str) -> list[float]:
        return self.inner._embed_text(text)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        return self.inner._cosine_similarity(left, right)

    @staticmethod
    def _has_match(packet: Packet) -> bool:
        retrieved = packet.retrieved if packet.retrieved is not None else RetrievedSet()
        return bool(retrieved.items)

    @staticmethod
    def _retrieved_set_from_entries(store: MemoryStore, matched_entries: list[dict[str, Any]]) -> RetrievedSet:
        grouped: dict[str, list[tuple[str, str, str]]] = {}
        for entry in matched_entries:
            record_id = str(entry.get("record_id", "")).strip()
            triple = entry.get("triple")
            if not record_id or not isinstance(triple, tuple) or len(triple) != 3:
                continue
            grouped.setdefault(record_id, []).append(triple)

        items = []
        scores = []
        for record in store.iter_records("triple_memory"):
            record_matches = grouped.get(record.record_id, [])
            if not record_matches:
                continue
            items.append(record)
            scores.append(
                {
                    "record_id": record.record_id,
                    "rank": len(items),
                    "score": float(len(record_matches)),
                    "strategy": "ret_llm_triplet_table_scan",
                    "retrieval_mode": "table_scan",
                    "matched_triples": list(record_matches),
                }
            )
        return RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": RETLLMTripleMemoryRetrieval.spec.name,
                "retrieval_mode": "table_scan",
                "matched_candidate_count": len(items),
                "matched_triplet_count": len(matched_entries),
            },
        )

    def _resolve_query_terms(self, query: Query, store: MemoryStore) -> Query | None:
        query_spec = _parse_structured_triple_query(query.text)
        resolved = dict(query_spec)
        changed = False
        memory_terms = self._collect_memory_terms(store)
        for slot in ("subject", "relation", "object"):
            query_value = query_spec[slot]
            if query_value is None:
                continue
            candidates = memory_terms[slot]
            if query_value.casefold() in {candidate.casefold() for candidate in candidates}:
                continue
            replacement = self._nearest_existing_term(query_value, candidates)
            if replacement is None:
                return None
            resolved[slot] = replacement
            changed = True
        if not changed:
            return None
        metadata = dict(query.metadata or {})
        metadata["ret_llm_fallback"] = {
            "original_query": query.text,
            "resolved_query": _format_structured_triple_query(resolved),
        }
        return Query(text=_format_structured_triple_query(resolved), metadata=metadata)

    @staticmethod
    def _collect_memory_terms(store: MemoryStore) -> dict[str, set[str]]:
        collected = {"subject": set(), "relation": set(), "object": set()}
        for record in store.iter_records("triple_memory"):
            for subject, relation, obj in _record_triples(record):
                if subject:
                    collected["subject"].add(subject)
                if relation:
                    collected["relation"].add(relation)
                if obj:
                    collected["object"].add(obj)
        return collected

    def _nearest_existing_term(self, query_value: str, candidates: set[str]) -> str | None:
        if not candidates:
            return None
        query_embedding = self._embed_text(query_value)
        scored = [
            (
                self._cosine_similarity(query_embedding, self._embed_text(candidate)),
                candidate,
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1].casefold()))
        best_score, best_candidate = scored[0]
        if best_score < self.fallback_similarity_threshold:
            return None
        return best_candidate


@dataclass(slots=True)
class RETLLMMemoryReadPipeline:
    store: MemoryStore
    retrieval: RETLLMTripleMemoryRetrieval
    readout: TemplateReadout
    fallback_similarity_threshold: float = 0.7

    def recall(self, query: Query) -> Readout:
        readout = self._recall_once(query)
        if readout.text.strip():
            return readout

        resolved_query = self._resolve_query_terms(query)
        if resolved_query is None:
            return readout
        resolved_readout = self._recall_once(resolved_query)
        if resolved_readout.metadata is not None:
            resolved_readout.metadata.setdefault("ret_llm_fallback", {})
            resolved_readout.metadata["ret_llm_fallback"].update(
                {
                    "used": True,
                    "original_query": query.text,
                    "resolved_query": resolved_query.text,
                }
            )
        return resolved_readout

    def _recall_once(self, query: Query) -> Readout:
        packet, _ = self.retrieval.run(Packet(query=query), self.store)
        trace = dict(packet.trace)
        trace["ret_llm"] = {
            "matched_triples": _matched_triplet_entries(packet),
        }
        packet = replace(packet, trace=trace)
        packet, _ = self.readout.run(packet, self.store)
        if packet.readout is None:
            raise RuntimeError("RETLLMMemoryReadPipeline produced no readout.")
        return packet.readout

    def _resolve_query_terms(self, query: Query) -> Query | None:
        query_spec = _parse_structured_triple_query(query.text)
        resolved = dict(query_spec)
        changed = False
        memory_terms = self._collect_memory_terms()
        for slot in ("subject", "relation", "object"):
            query_value = query_spec[slot]
            if query_value is None:
                continue
            candidates = memory_terms[slot]
            if query_value.casefold() in {candidate.casefold() for candidate in candidates}:
                continue
            replacement = self._nearest_existing_term(query_value, candidates)
            if replacement is None:
                return None
            resolved[slot] = replacement
            changed = True
        if not changed:
            return None
        metadata = dict(query.metadata or {})
        metadata["ret_llm_fallback"] = {
            "original_query": query.text,
            "resolved_query": _format_structured_triple_query(resolved),
        }
        return Query(text=_format_structured_triple_query(resolved), metadata=metadata)

    def _collect_memory_terms(self) -> dict[str, set[str]]:
        collected = {"subject": set(), "relation": set(), "object": set()}
        for record in self.store.iter_records("triple_memory"):
            for subject, relation, obj in _record_triples(record):
                if subject:
                    collected["subject"].add(subject)
                if relation:
                    collected["relation"].add(relation)
                if obj:
                    collected["object"].add(obj)
        return collected

    def _nearest_existing_term(self, query_value: str, candidates: set[str]) -> str | None:
        if not candidates:
            return None
        query_embedding = self.retrieval._embed_text(query_value)
        scored = [
            (
                self.retrieval._cosine_similarity(query_embedding, self.retrieval._embed_text(candidate)),
                candidate,
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1].casefold()))
        best_score, best_candidate = scored[0]
        if best_score < self.fallback_similarity_threshold:
            return None
        return best_candidate


@dataclass(slots=True)
class RETLLMSystem:
    store: MemoryStore
    write_pipeline: MemoryPipeline
    mem_read_pipeline: Any
    answer_prompt: Any
    max_turns: int
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    embedding_model: str | None = None
    answer_agent_runner: Any | None = None

    def memorize(self, text: str, *, source: str = "document") -> None:
        self.write_pipeline.ingest(Observation(text=text, source=source))

    def mem_read(self, structured_query: str) -> str:
        return self.mem_read_pipeline.recall(Query(text=structured_query)).text

    def answer(self, question: str) -> str:
        packet = Packet(query=Query(text=question), retrieved=RetrievedSet())
        tool_specs = normalize_readout_tool_specs(
            ["MEM_READ"],
            module_name="ret_llm_answer_loop",
            retrieve_pipeline=self.mem_read_pipeline,
        )
        state = ToolExecutionState()
        tool_context = ReadoutToolCallContext(packet=packet, store=self.store, retrieve_pipeline=self.mem_read_pipeline)
        tools = build_runtime_tools(tool_specs, context=tool_context, state=state, strict_tools=True)
        rendered_prompt, _prompt_trace, updated_store = render_prompt_plan(
            ensure_prompt_plan(
                self.answer_prompt,
                metadata_mode="prompt",
                context_builder=lambda _packet, _store: {
                    "tools": project_tool_specs_for_prompt(tool_specs),
                },
            ),
            packet=packet,
            store=self.store,
        )
        tool_context.store = updated_store
        runner = self.answer_agent_runner or self._run_answer_agent
        final_text = runner(
            rendered_prompt=rendered_prompt,
            tools=tools,
            context={"query_text": question},
        ).strip()
        self.store = tool_context.store
        return final_text

    def _run_answer_agent(self, *, rendered_prompt: str, tools: list[Any], context: dict[str, Any]) -> str:
        runtime = Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )
        runtime.require_llm(capability="RETLLMSystem.answer")
        return str(
            runtime.run_agent(
                name="MemPrimitiveRETLLMAnswerAgent",
                instructions=(
                    "You answer the question using the provided prompt and may call the provided tools "
                    "when explicit memory lookup is needed. Use zero or more tool calls, then finish "
                    "with a plain-text final answer."
                ),
                input_text=json.dumps({"prompt": rendered_prompt, "context": context}, ensure_ascii=False),
                temperature=0.0,
                tools=tools,
                max_turns=self.max_turns,
            )
            or ""
        )


def build_ret_llm_memory_system(
    *,
    mem_read_top_k: int = 6,
    max_turns: int = 6,
    mem_read_fallback_similarity_threshold: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    embedding_model: str | None = None,
) -> RETLLMSystem:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="triple_memory",
                theme="semantic",
                shape="Graph",
                indices=("graph", "entity"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    write_pipeline = MemoryPipeline(
        unit_formation=SentenceSplitUnitFormation(),
        representation=TripleRepresentation(
            method="two_stage",
            prompt=text_prompt(
                "Extract grounded knowledge triples from the sentence.\n"
                "Keep only facts explicitly supported by the sentence.\n"
                "Use canonical entity names where possible.\n"
                "Use short relation phrases.\n"
                "Do not infer missing facts.\n\n"
                "Sentence:\n{{ unit.text }}"
            ),
            embed_entities=True,
        ),
        write_trigger=AlwaysTrigger(),
        organization=GraphEntityDeduplicationAppendOrganization(
            target_layer="triple_memory",
            threshold=0.85,
        ),
        store=store,
    )

    mem_read_pipeline = RETLLMMemoryReadPipeline(
        store=store,
        retrieval=RETLLMTripleMemoryRetrieval(
            TripleMemoryRetrieval(
                top_k=mem_read_top_k,
                layer="triple_memory",
                candidate_similarity_threshold=1.0,
                final_similarity_threshold=1.0,
            ),
            fallback_similarity_threshold=mem_read_fallback_similarity_threshold,
        ),
        readout=TemplateReadout(
            prompt=structured_prompt(
                {
                    "blocks": [
                        {
                            "id": "matched_triples",
                            "title": "Matched Triples",
                            "condition": "trace.packet.ret_llm.matched_triples | length",
                            "repeat_over": "trace.packet.ret_llm.matched_triples",
                            "item_template": "{{ item.text }}",
                            "separator": "\n",
                        },
                    ]
                }
            )
        ),
        fallback_similarity_threshold=mem_read_fallback_similarity_threshold,
    )

    answer_prompt = structured_prompt(
        {
            "blocks": [
                {
                    "id": "task",
                    "title": "Task",
                    "template": (
                        "You are answering with a RET-LLM-style explicit memory.\n"
                        "This repo only supplies the modular memory components; the current answer step is a normal "
                        "agent loop outside the memory pipeline.\n"
                        "You may call MEM_READ one or more times during reasoning.\n"
                        "MEM_READ accepts structured triple queries written as 'subject >> relation >> object'.\n"
                        "MEM_READ returns only the matched triplet set.\n"
                        "Use '*' for unknown slots. Ground one or two slots when possible.\n"
                        "If memory returns no match, say the memory does not contain the needed fact instead of inventing one.\n"
                        "When memory does return a match, ground the final answer in those facts."
                    ),
                },
                {
                    "id": "question",
                    "title": "Question",
                    "template": "{{ query.text }}",
                },
                {
                    "id": "tools",
                    "title": "Available Tools",
                    "repeat_over": "tools",
                    "item_template": "- {{ item.name }}: {{ item.description }}",
                    "separator": "\n",
                },
            ]
        }
    )

    return RETLLMSystem(
        store=store,
        write_pipeline=write_pipeline,
        mem_read_pipeline=mem_read_pipeline,
        answer_prompt=answer_prompt,
        max_turns=max_turns,
        api_key=api_key,
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
    )


def main() -> None:
    system = build_ret_llm_memory_system()

    system.memorize(
        "Washington D.C. is the capital of the United States. "
        "Marie Curie discovered polonium with Pierre Curie. "
        "The album Alla Mia Eta contains the song Il Regalo Piu Grande."
    )

    print("records per layer:")
    pprint({name: system.store.count(name) for name in system.store.topology.layer_names})
    print()

    print("stored memory records:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "triples": record.metadata.get("representation", {}).get("triples", []),
            }
            for record in system.store.iter_records("triple_memory")
        ]
    )
    print()

    print("memory read:")
    print(system.mem_read("Washington D.C. >> capital of >> *"))
    print()

    print("agent-loop answer:")
    print(system.answer("What is the capital of the United States?"))


if __name__ == "__main__":
    main()
