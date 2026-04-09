"""Baseline: representation primitive."""

from __future__ import annotations

from collections import Counter
import inspect
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Final, get_args, get_origin

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ..contracts import (
    UNIT_EMBEDDING_CONTRACT,
    UNIT_ENTITIES_CONTRACT,
    UNIT_TAGS_CONTRACT,
    normalize_contracts,
)
from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet, _representation_summary_from_unit
from ..interfaces import RepresentationModule

from ..utils._amem_family import DEFAULT_EMBEDDING_VERSION
from ..utils._template import (
    PromptPlan,
    ensure_prompt_plan,
    looks_like_template,
    project_unit_for_template,
    render_prompt_plan,
    text_prompt,
)
from ..utils._trace import copy_trace

_VALID_ELEMENTS: Final[tuple[str, ...]] = (
    "text",
    "embedding",
    "keywords",
    "time_anchor",
    "source_type",
)
_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "have",
        "has",
        "was",
        "were",
        "are",
        "is",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "or",
        "as",
        "it",
        "their",
        "they",
        "them",
        "we",
        "you",
        "your",
    }
)
_MEMPRIMITIVE_ENV_PATH: Final[Path] = Path(__file__).resolve().parents[1] / ".env"


class _TripleRelationship(BaseModel):
    subject: str
    predicate: str
    object: str


class _TripleDirectOutput(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relationships: list[_TripleRelationship] = Field(default_factory=list)


class _TripleEntityOutput(BaseModel):
    entities: list[str] = Field(default_factory=list)


class _TripleRelationshipOutput(BaseModel):
    relationships: list[_TripleRelationship] = Field(default_factory=list)


def _dedupe_non_empty_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _normalize_hinted_triples(value: Any) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    if not isinstance(value, list):
        return triples
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            subject = str(item[0]).strip()
            predicate = str(item[1]).strip()
            obj = str(item[2]).strip()
            if subject and predicate and obj:
                triples.append((subject, predicate, obj))
    return list(dict.fromkeys(triples))


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list of strings.")
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


def _normalize_string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        value_text = str(item).strip()
        if key_text and value_text:
            normalized[key_text] = value_text
    return normalized

class BasicRepresentation(RepresentationModule):
    """Build a configurable representation-element set for each unit.

    ``run`` requires ``packet.units``. The module writes representation outputs to
    top-level ``MemoryUnit`` fields and mirrors a compact summary into
    ``metadata["representation"]`` for backward compatibility. The store is
    unchanged.
    """

    spec = ModuleSpec(
        name="basic_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=(
            "units.text",
            "units.representation_elements",
            "units.normalized_text",
            "units.metadata.representation",
        ),
    )
    _embedding_cache: ClassVar[dict[str, SentenceTransformer]] = {}
    requires_contracts = frozenset()

    def __init__(
        self,
        *,
        elements: tuple[str, ...] = ("text", "embedding"),
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        normalized = tuple(dict.fromkeys(elements))
        unsupported = [element for element in normalized if element not in _VALID_ELEMENTS]
        if unsupported:
            raise ValueError(f"Unsupported representation element(s): {unsupported}.")

        self.elements = normalized
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.api_key = api_key if api_key is not None else env.get("MEMPRIMITIVE_API_KEY", "")
        self.base_url = base_url if base_url is not None else env.get("MEMPRIMITIVE_BASE_URL", "")
        self.model = model if model is not None else env.get("MEMPRIMITIVE_MODEL", "")
        self.produces_contracts = normalize_contracts(
            [UNIT_EMBEDDING_CONTRACT] if "embedding" in self.elements else []
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("BasicRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            represented_unit = self._represent_unit(unit)
            represented_units.append(represented_unit)
            per_unit_trace.append(
                {
                    "unit_id": represented_unit.unit_id,
                    "elements": list(represented_unit.representation_elements),
                }
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in represented_units],
            "elements": list(self.elements),
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _represent_unit(self, unit: MemoryUnit) -> MemoryUnit:
        normalized_text = unit.text.strip()
        normalized_casefold = normalized_text.casefold()

        elements = set(unit.representation_elements)
        text_value = normalized_text
        embedding = list(unit.embedding) if unit.embedding is not None else None
        triples = list(unit.triples)
        representation_meta: dict[str, Any] = dict(unit.metadata.get("representation", {}))

        if "text" in self.elements:
            elements.add("text")
        if "embedding" in self.elements:
            embedding = self._embed_text(normalized_text)
            elements.add("embedding")
        if "keywords" in self.elements:
            keywords = self._extract_keywords(unit, normalized_text)
            if keywords:
                representation_meta["keywords"] = keywords
                elements.add("keywords")
        if "time_anchor" in self.elements:
            time_anchor = self._build_time_anchor(unit)
            if time_anchor:
                representation_meta["time_anchor"] = time_anchor
                elements.add("time_anchor")
        if "source_type" in self.elements:
            source_type = self._build_source_type(unit)
            if source_type:
                representation_meta["source_type"] = source_type
                elements.add("source_type")

        represented = replace(
            unit,
            text=text_value,
            representation_elements=tuple(sorted(elements)),
            normalized_text=normalized_casefold,
            embedding=embedding,
            triples=triples,
        )
        return replace(
            represented,
            metadata={
                **represented.metadata,
                "representation": {
                    **_representation_summary_from_unit(represented),
                    **representation_meta,
                },
            },
        )

    def _embed_text(self, text: str) -> list[float]:
        model = self._embedding_cache.get(self.embedding_model)
        if model is None:
            model = SentenceTransformer(self.embedding_model)
            self._embedding_cache[self.embedding_model] = model
        return [float(value) for value in model.encode(text, normalize_embeddings=True).tolist()]

    def _extract_keywords(self, unit: MemoryUnit, text: str) -> list[str]:
        hinted = unit.metadata.get("keywords")
        if isinstance(hinted, list) and hinted:
            return list(dict.fromkeys(str(item).casefold() for item in hinted if str(item).strip()))

        counts = Counter(
            token.casefold()
            for token in _WORD_PATTERN.findall(text)
            if token.casefold() not in _STOPWORDS
        )
        return [token for token, _ in counts.most_common(6)]

    def _build_time_anchor(self, unit: MemoryUnit) -> dict[str, str] | None:
        timestamp = unit.metadata.get("time_anchor") if isinstance(unit.metadata.get("time_anchor"), dict) else None
        if timestamp:
            return {str(key): str(value) for key, value in timestamp.items()}
        raw = unit.timestamp
        if not raw:
            return None
        return {
            "timestamp": raw,
            "date": raw.split("T", 1)[0],
        }

    def _build_source_type(self, unit: MemoryUnit) -> str | None:
        source = unit.metadata.get("source")
        if source is None:
            return None
        return str(source).strip() or None

class TripleRepresentation(RepresentationModule):
    """LLM-backed triple extraction with direct and two-stage modes."""

    spec = ModuleSpec(
        name="triple_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=(
            "units.triples",
            "units.entities",
            "units.representation_elements",
            "units.metadata.representation",
        ),
    )
    requires_contracts = frozenset()
    produces_contracts = frozenset({UNIT_ENTITIES_CONTRACT})
    _VALID_METHODS: ClassVar[frozenset[str]] = frozenset({"direct", "two_stage"})

    def __init__(
        self,
        *,
        method: str = "direct",
        prompt: PromptPlan | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        embed_extracted: bool = False,
        embed_entities: bool = False,
    ) -> None:
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        normalized_method = str(method).strip().lower()
        if normalized_method not in self._VALID_METHODS:
            raise ValueError(f"TripleRepresentation method must be one of: {sorted(self._VALID_METHODS)}.")
        if isinstance(prompt, str):
            normalized_prompt = prompt.strip()
            if not normalized_prompt:
                raise ValueError("TripleRepresentation prompt must be non-empty when provided.")
        else:
            normalized_prompt = prompt
        self.method = normalized_method
        self.prompt = normalized_prompt
        self.api_key = api_key if api_key is not None else env.get("MEMPRIMITIVE_API_KEY", "")
        self.base_url = base_url if base_url is not None else env.get("MEMPRIMITIVE_BASE_URL", "")
        self.model = model if model is not None else env.get("MEMPRIMITIVE_MODEL", "")
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.embed_extracted = bool(embed_extracted)
        self.embed_entities = bool(embed_entities)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TripleRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            represented_unit, extraction_trace, store = self._dispatch_represent_unit(packet, unit, store)
            represented_units.append(represented_unit)
            per_unit_trace.append(
                {
                    "unit_id": represented_unit.unit_id,
                    "method": self.method,
                    **extraction_trace,
                }
            )

        trace = copy_trace(packet)
        prompt_plan = ensure_prompt_plan(self.prompt, metadata_mode="prompt") if self.prompt is not None else None
        trace["representation"] = {
            "module": self.spec.name,
            "method": self.method,
            "embed_extracted": self.embed_extracted,
            "embed_entities": self.embed_entities,
            "prompt_is_template": bool(
                prompt_plan is not None
                and (
                    prompt_plan.mode == "structured"
                    or (isinstance(prompt_plan.template, str) and "{{" in prompt_plan.template and "}}" in prompt_plan.template)
                )
            ),
            "unit_ids": [unit.unit_id for unit in represented_units],
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _dispatch_represent_unit(
        self,
        packet: Packet,
        unit: MemoryUnit,
        store: MemoryStore,
    ) -> tuple[MemoryUnit, dict[str, Any], MemoryStore]:
        parameters = tuple(inspect.signature(self._represent_unit).parameters.values())
        if len(parameters) <= 1:
            represented_unit, extraction_trace = self._represent_unit(unit)  # type: ignore[misc]
            return represented_unit, extraction_trace, store
        return self._represent_unit(unit, packet, store)

    def _represent_unit(
        self,
        unit: MemoryUnit,
        packet: Packet | None = None,
        store: MemoryStore | None = None,
    ) -> tuple[MemoryUnit, dict[str, Any], MemoryStore]:
        current_store = store if store is not None else MemoryStore()
        normalized_text = unit.text.strip()
        normalized_casefold = normalized_text.casefold()
        hinted_triples = _normalize_hinted_triples(unit.metadata.get("triples"))
        hinted_entities = unit.metadata.get("entities")
        if hinted_triples:
            hinted_entity_values = [str(item) for item in hinted_entities] if isinstance(hinted_entities, list) else []
            entities = _dedupe_non_empty_strings(
                [
                    *hinted_entity_values,
                    *(subject for subject, _, _ in hinted_triples),
                    *(obj for _, _, obj in hinted_triples),
                ]
            )
            extraction_trace = {
                "source": "metadata_hint",
                "entities": entities,
                "triple_count": len(hinted_triples),
            }
            return (
                self._replace_unit(unit, normalized_text, normalized_casefold, entities, hinted_triples),
                extraction_trace,
                current_store,
            )

        runtime = self._runtime()
        runtime.require_llm(capability=f"TripleRepresentation method {self.method!r}")
        prompt_trace: dict[str, Any] = {}
        rendered_prompt: str | None = None
        if packet is not None and store is not None and self.prompt is not None:
            rendered_prompt, prompt_trace, current_store = self._render_prompt(packet, unit, store)

        if self.method == "direct":
            payload = self._extract_direct(runtime=runtime, text=normalized_text, prompt=rendered_prompt)
            extraction_trace = {
                "source": "llm_direct",
                "entities": payload["entities"],
                "triple_count": len(payload["triples"]),
                **prompt_trace,
            }
        else:
            payload = self._extract_two_stage(runtime=runtime, text=normalized_text, prompt=rendered_prompt)
            extraction_trace = {
                "source": "llm_two_stage",
                "entities": payload["entities"],
                "triple_count": len(payload["triples"]),
                **prompt_trace,
            }
        return (
            self._replace_unit(unit, normalized_text, normalized_casefold, payload["entities"], payload["triples"]),
            extraction_trace,
            current_store,
        )

    def _replace_unit(
        self,
        unit: MemoryUnit,
        normalized_text: str,
        normalized_casefold: str,
        entities: list[str],
        triples: list[tuple[str, str, str]],
    ) -> MemoryUnit:
        elements = set(unit.representation_elements)
        embedding = list(unit.embedding) if unit.embedding is not None else None
        entity_embeddings: dict[str, list[float]] = {}
        if triples:
            elements.add("triple")
        if entities:
            elements.add("entities")
        runtime = None
        if self.embed_extracted or self.embed_entities:
            runtime = self._runtime()
        embedding_text = self._build_embedding_text(entities=entities, triples=triples)
        if self.embed_extracted and embedding_text is not None:
            assert runtime is not None
            embedding = list(runtime.embed(embedding_text))
            elements.add("embedding")
        if self.embed_entities:
            assert runtime is not None
            entity_embeddings = {
                entity: list(runtime.embed(entity))
                for entity in _dedupe_non_empty_strings([str(entity) for entity in entities])
            }
        represented = replace(
            unit,
            text=normalized_text,
            normalized_text=normalized_casefold,
            embedding=embedding,
            entities=entities,
            triples=triples,
            representation_elements=tuple(sorted(elements)),
        )
        return replace(
            represented,
            metadata={
                **represented.metadata,
                "representation": {
                    **_representation_summary_from_unit(represented),
                    **({"entity_embeddings": entity_embeddings} if entity_embeddings else {}),
                },
            },
        )

    @staticmethod
    def _build_embedding_text(
        *,
        entities: list[str],
        triples: list[tuple[str, str, str]],
    ) -> str | None:
        normalized_entities = _dedupe_non_empty_strings([str(entity) for entity in entities])
        normalized_triples = list(
            dict.fromkeys(
                (
                    str(subject).strip(),
                    str(predicate).strip(),
                    str(obj).strip(),
                )
                for subject, predicate, obj in triples
                if str(subject).strip() and str(predicate).strip() and str(obj).strip()
            )
        )
        if not normalized_entities and not normalized_triples:
            return None
        lines: list[str] = []
        if normalized_entities:
            lines.append("entities:")
            lines.extend(f"- {entity}" for entity in normalized_entities)
        if normalized_triples:
            lines.append("triples:")
            lines.extend(f"- {subject} | {predicate} | {obj}" for subject, predicate, obj in normalized_triples)
        return "\n".join(lines)

    def _runtime(self):
        from ..utils._runtime import Runtime

        return Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )

    def _render_prompt(self, packet: Packet, unit: MemoryUnit, store: MemoryStore) -> tuple[str, dict[str, Any], MemoryStore]:
        plan = ensure_prompt_plan(
            self.prompt,
            metadata_mode="prompt",
            context_builder=lambda packet, current_store: {
                "unit": project_unit_for_template(unit),
            },
        )
        return render_prompt_plan(plan, packet=packet, store=store)

    def _extract_direct(self, *, runtime, text: str, prompt: str | None = None) -> dict[str, list[Any]]:
        instructions = (
            "Extract a clean knowledge-graph representation from the text. "
            "Return only entities explicitly grounded in the text and relationships that can be expressed as triples. "
            "Each relationship must have non-empty subject, predicate, and object."
        )
        if prompt is not None:
            instructions = (
                "You extract a clean knowledge-graph representation from the provided text. "
                "Follow the user-provided prompt exactly when deciding what entities and relationships to keep. "
                "Return only grounded entities and relationships that can be expressed as triples. "
                "Each relationship must have non-empty subject, predicate, and object."
            )
        output = runtime.run_agent(
            name="MemPrimitiveTripleDirectAgent",
            instructions=instructions,
            input_text=json.dumps(
                {"text": text, **({"prompt": prompt} if prompt is not None else {})},
                ensure_ascii=False,
            ),
            temperature=0.0,
            max_turns=1,
            output_type=_TripleDirectOutput,
        )
        return self._normalize_triple_output(output)

    def _extract_two_stage(self, *, runtime, text: str, prompt: str | None = None) -> dict[str, list[Any]]:
        entity_instructions = (
            "Extract the canonical entities explicitly mentioned in the text. "
            "Return only grounded entities and avoid duplicates."
        )
        relation_instructions = (
            "Extract relationships from the text using the provided entities as anchors. "
            "Only emit relationships that are directly supported by the text and can be written as triples. "
            "Prefer the provided canonical entity strings for subject/object values."
        )
        if prompt is not None:
            entity_instructions = (
                "You extract canonical entities from the provided text. "
                "Follow the user-provided prompt exactly when deciding which grounded entities matter. "
                "Return only grounded entities and avoid duplicates."
            )
            relation_instructions = (
                "You extract relationships from the provided text using the provided entities as anchors. "
                "Follow the user-provided prompt exactly when deciding which grounded relationships matter. "
                "Only emit relationships directly supported by the text that can be written as triples. "
                "Prefer the provided canonical entity strings for subject/object values."
            )
        entity_output = runtime.run_agent(
            name="MemPrimitiveTripleEntityAgent",
            instructions=entity_instructions,
            input_text=json.dumps(
                {"text": text, **({"prompt": prompt} if prompt is not None else {})},
                ensure_ascii=False,
            ),
            temperature=0.0,
            max_turns=1,
            output_type=_TripleEntityOutput,
        )
        if isinstance(entity_output, BaseModel):
            raw_entities = entity_output.model_dump().get("entities", [])
        elif isinstance(entity_output, dict):
            raw_entities = entity_output.get("entities", [])
        else:
            raw_entities = getattr(entity_output, "entities", [])
        entities = _dedupe_non_empty_strings([str(item) for item in raw_entities] if isinstance(raw_entities, list) else [])
        relationship_output = runtime.run_agent(
            name="MemPrimitiveTripleRelationAgent",
            instructions=relation_instructions,
            input_text=json.dumps(
                {"text": text, "entities": entities, **({"prompt": prompt} if prompt is not None else {})},
                ensure_ascii=False,
            ),
            temperature=0.0,
            max_turns=1,
            output_type=_TripleRelationshipOutput,
        )
        normalized = self._normalize_triple_output(relationship_output)
        normalized["entities"] = _dedupe_non_empty_strings([*entities, *normalized["entities"]])
        return normalized

    def _normalize_triple_output(self, output: Any) -> dict[str, Any]:
        if isinstance(output, BaseModel):
            payload = output.model_dump()
        elif isinstance(output, dict):
            payload = dict(output)
        else:
            raise ValueError("TripleRepresentation requires structured model output.")

        raw_relationships = payload.get("relationships", [])
        triples: list[tuple[str, str, str]] = []
        for item in raw_relationships if isinstance(raw_relationships, list) else []:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if subject and predicate and obj:
                triples.append((subject, predicate, obj))
        triples = list(dict.fromkeys(triples))
        raw_entities = payload.get("entities", [])
        raw_entity_values = [str(item) for item in raw_entities] if isinstance(raw_entities, list) else []
        entities = _dedupe_non_empty_strings(
            [
                *raw_entity_values,
                *(subject for subject, _, _ in triples),
                *(obj for _, _, obj in triples),
            ]
        )
        return {"entities": entities, "triples": triples}


class LLMRepresentation(RepresentationModule):
    """Extract one representation field per unit with an LLM."""

    spec = ModuleSpec(
        name="llm_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=(
            "units.representation_elements",
            "units.metadata.representation",
        ),
    )
    requires_contracts = frozenset()

    _LIST_FIELDS: ClassVar[frozenset[str]] = frozenset({"tags", "entities", "keywords", "relation_tags"})
    _DICT_FIELDS: ClassVar[frozenset[str]] = frozenset({"time_anchor", "kv"})
    _VALUE_KIND_TEXT: ClassVar[str] = "text"
    _VALUE_KIND_LIST: ClassVar[str] = "list"
    _VALUE_KIND_DICT: ClassVar[str] = "dict"

    def __init__(
        self,
        *,
        field: str,
        prompt: PromptPlan | str,
        value_type: Any | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        normalized_field = str(field).strip()
        if not normalized_field:
            raise ValueError("LLMRepresentation requires a non-empty field.")
        if isinstance(prompt, str):
            normalized_prompt = prompt.strip()
        else:
            normalized_prompt = prompt
        if isinstance(normalized_prompt, str) and not normalized_prompt:
            raise ValueError("LLMRepresentation requires a non-empty prompt.")
        self.field = normalized_field
        self.prompt = normalized_prompt
        self.value_type = value_type
        self._declared_value_kind = self._resolve_declared_value_kind(value_type)
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.api_key = api_key if api_key is not None else env.get("MEMPRIMITIVE_API_KEY", "")
        self.base_url = base_url if base_url is not None else env.get("MEMPRIMITIVE_BASE_URL", "")
        self.model = model if model is not None else env.get("MEMPRIMITIVE_MODEL", "")
        produces: list[str] = []
        if self.field == "entities":
            produces.append(UNIT_ENTITIES_CONTRACT)
        if self.field == "tags":
            produces.append(UNIT_TAGS_CONTRACT)
        self.produces_contracts = normalize_contracts(produces)

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("LLMRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            represented_unit, unit_trace, store = self._represent_unit(packet, unit, store)
            represented_units.append(represented_unit)
            per_unit_trace.append({"unit_id": represented_unit.unit_id, **unit_trace})

        trace = copy_trace(packet)
        prompt_plan = ensure_prompt_plan(self.prompt, metadata_mode="prompt")
        trace["representation"] = {
            "module": self.spec.name,
            "field": self.field,
            "declared_value_type": self._declared_value_type_name(),
            "prompt_is_template": prompt_plan.mode == "structured" or (isinstance(prompt_plan.template, str) and "{{" in prompt_plan.template and "}}" in prompt_plan.template),
            "unit_ids": [unit.unit_id for unit in represented_units],
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _represent_unit(self, packet: Packet, unit: MemoryUnit, store: MemoryStore) -> tuple[MemoryUnit, dict[str, Any], MemoryStore]:
        normalized_text = unit.text.strip()
        normalized_casefold = normalized_text.casefold()
        payload, prompt_trace, store = self._extract_field(packet, unit, normalized_text, store)
        elements = set(unit.representation_elements)
        elements.add(self.field)
        representation_meta: dict[str, Any] = dict(unit.metadata.get("representation", {}))

        updated = replace(
            unit,
            text=normalized_text,
            normalized_text=normalized_casefold,
        )
        if self.field == "tags":
            updated = replace(updated, tags=list(payload))
        elif self.field == "entities":
            updated = replace(updated, entities=list(payload))
        elif self.field == "kv":
            updated = replace(updated, kv=dict(payload))
        elif self.field == "description":
            updated = replace(updated, description=payload)
        else:
            representation_meta[self.field] = payload

        updated = replace(updated, representation_elements=tuple(sorted(elements)))
        updated = replace(
            updated,
            metadata={
                **updated.metadata,
                "representation": {
                    **_representation_summary_from_unit(updated),
                    **representation_meta,
                },
            },
        )

        trace_value = payload
        if isinstance(payload, list):
            trace_value = list(payload)
        elif isinstance(payload, dict):
            trace_value = dict(payload)
        return updated, {
            "field": self.field,
            "kind": self._value_kind(),
            "declared_value_type": self._declared_value_type_name(),
            "value": trace_value,
            **prompt_trace,
        }, store

    def _value_kind(self) -> str:
        if self._declared_value_kind is not None:
            return self._declared_value_kind
        if self.field in self._LIST_FIELDS:
            return self._VALUE_KIND_LIST
        if self.field in self._DICT_FIELDS:
            return self._VALUE_KIND_DICT
        return self._VALUE_KIND_TEXT

    def _declared_value_type_name(self) -> str | None:
        if self.value_type is None:
            return None
        if self._declared_value_kind == self._VALUE_KIND_TEXT:
            return "str"
        if self._declared_value_kind == self._VALUE_KIND_LIST:
            return "list[str]"
        if self._declared_value_kind == self._VALUE_KIND_DICT:
            return "dict[str, str]"
        return str(self.value_type)

    @classmethod
    def _resolve_declared_value_kind(cls, value_type: Any | None) -> str | None:
        if value_type is None:
            return None
        if value_type is str:
            return cls._VALUE_KIND_TEXT

        origin = get_origin(value_type)
        args = get_args(value_type)
        if origin is list and args == (str,):
            return cls._VALUE_KIND_LIST
        if origin is dict and args == (str, str):
            return cls._VALUE_KIND_DICT

        raise ValueError(
            "LLMRepresentation value_type only supports str, list[str], or dict[str, str]."
        )

    def _runtime(self):
        from ..utils._runtime import Runtime

        return Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )

    def _extract_field(self, packet: Packet, unit: MemoryUnit, text: str, store: MemoryStore) -> tuple[Any, dict[str, Any], MemoryStore]:
        rendered_prompt, prompt_trace, store = self._render_prompt(packet, unit, store)
        user = json.dumps(
            {
                "field": self.field,
                "prompt": rendered_prompt,
                "unit": {
                    "text": text,
                    "unit_type": unit.unit_type,
                    "timestamp": unit.timestamp,
                    "existing_tags": unit.tags,
                    "existing_entities": unit.entities,
                    "existing_description": unit.description,
                    "existing_metadata": unit.metadata,
                },
            },
            ensure_ascii=False,
        )
        kind = self._value_kind()
        if kind == self._VALUE_KIND_LIST:
            return _normalize_string_list(self._llm_json(user=user)), prompt_trace, store
        if kind == self._VALUE_KIND_DICT:
            return _normalize_string_dict(self._llm_json(user=user)), prompt_trace, store
        return self._llm_text(user=user), prompt_trace, store

    def _render_prompt(self, packet: Packet, unit: MemoryUnit, store: MemoryStore) -> tuple[str, dict[str, Any], MemoryStore]:
        plan = ensure_prompt_plan(
            self.prompt,
            metadata_mode="prompt",
            context_builder=lambda packet, current_store: {
                "field": self.field,
                "unit": project_unit_for_template(unit),
            },
        )
        return render_prompt_plan(plan, packet=packet, store=store)

    def _llm_text(self, *, user: str) -> str:
        runtime = self._runtime()
        runtime.require_llm(capability=f"LLMRepresentation field {self.field!r}")
        output = runtime.text(
            system=(
                "You extract one requested representation field for a memory unit. "
                "Follow the user-provided prompt exactly and return plain text only."
            ),
            user=user,
            temperature=0.0,
        ).strip()
        if not output:
            raise ValueError(f"LLMRepresentation field {self.field!r} returned empty text.")
        return output

    def _llm_json(self, *, user: str) -> Any:
        runtime = self._runtime()
        runtime.require_llm(capability=f"LLMRepresentation field {self.field!r}")
        kind = self._value_kind()
        shape = "a strict JSON array of strings" if kind == self._VALUE_KIND_LIST else "a strict JSON object"
        return runtime.json(
            system=(
                "You extract one requested representation field for a memory unit. "
                f"Follow the user-provided prompt exactly and return {shape}."
            ),
            user=user,
        )

class ConfigurableEmbeddingRepresentation(RepresentationModule):
    """Build embeddings from configurable text or template-rendered text.

    Constructor: ``embedding_text`` controls the text sent to the embedding
    runtime. It may be a literal string, a template string, or a full
    :class:`~memprimitive.utils._template.PromptPlan`. When omitted, the module
    embeds ``unit.text``. ``embedding_version`` is recorded in trace and
    representation metadata as embedding provenance.

    ``run`` requires ``packet.units`` and does not mutate ``store``. Units keep
    their original identity and text-facing fields while gaining embedding
    vectors plus a structured ``metadata['representation']`` projection that
    records the embedding input text.
    """

    spec = ModuleSpec(
        name="configurable_embedding_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.embedding", "units.metadata.representation"),
    )
    requires_contracts = frozenset()
    produces_contracts = frozenset({UNIT_EMBEDDING_CONTRACT})

    def __init__(
        self,
        *,
        embedding_text: PromptPlan | str | None = None,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        embedding_model: str | None = None,
    ) -> None:
        self.embedding_text = embedding_text if embedding_text is not None else text_prompt("{{ unit.text }}")
        self.embedding_version = embedding_version
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("ConfigurableEmbeddingRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            embedding_input_text, prompt_trace, store = self._render_embedding_text(packet, unit, store)
            embedding = self._embed_text(embedding_input_text)
            represented_unit = replace(
                unit,
                embedding=embedding,
                representation_elements=tuple(sorted({*unit.representation_elements, "embedding"})),
            )
            representation_meta = {
                **_representation_summary_from_unit(represented_unit),
                "embedding_input_text": embedding_input_text,
                "embedding_input_preview": embedding_input_text[:200],
                "embedding_version": self.embedding_version,
            }
            representation_meta.pop("enhanced_embedding_text", None)
            represented_units.append(
                replace(
                    represented_unit,
                    metadata={
                        **represented_unit.metadata,
                        "representation": {
                            **representation_meta,
                        },
                    },
                )
            )
            per_unit_trace.append(
                {
                    "unit_id": unit.unit_id,
                    "embedding_version": self.embedding_version,
                    "embedding_input_text": embedding_input_text,
                    "embedding_input_preview": embedding_input_text[:200],
                    **prompt_trace,
                }
            )

        trace = copy_trace(packet)
        prompt_plan = ensure_prompt_plan(self.embedding_text, metadata_mode="prompt")
        trace["representation"] = {
            "module": self.spec.name,
            "embedding_version": self.embedding_version,
            "prompt_is_template": prompt_plan.mode == "structured" or looks_like_template(str(prompt_plan.template)),
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _embed_text(self, text: str) -> list[float]:
        from ..utils._runtime import get_runtime

        return list(get_runtime().embed(text))

    def _render_embedding_text(
        self,
        packet: Packet,
        unit: MemoryUnit,
        store: MemoryStore,
    ) -> tuple[str, dict[str, Any], MemoryStore]:
        rendered, prompt_trace, store = render_prompt_plan(
            ensure_prompt_plan(
                self.embedding_text,
                metadata_mode="prompt",
                context_builder=lambda current_packet, current_store: {
                    "unit": project_unit_for_template(unit),
                },
            ),
            packet=packet,
            store=store,
        )
        normalized = rendered.strip()
        if not normalized:
            raise ValueError("ConfigurableEmbeddingRepresentation requires non-empty embedding text.")
        return normalized, prompt_trace, store


BASELINE_SLOT: Final[str] = "representation"
BASELINE_CLASSES: Final[tuple[type[RepresentationModule], ...]] = (
    BasicRepresentation,
    TripleRepresentation,
    LLMRepresentation,
    ConfigurableEmbeddingRepresentation,
)
