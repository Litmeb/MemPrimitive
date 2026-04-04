"""Baseline: representation primitive."""

from __future__ import annotations

from collections import Counter
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Final

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ..contracts import (
    UNIT_EMBEDDING_CONTRACT,
    UNIT_ENTITIES_CONTRACT,
    UNIT_NOTE_PAYLOAD_CONTRACT,
    UNIT_TAGS_CONTRACT,
    normalize_contracts,
)
from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet, _representation_summary_from_unit
from ..interfaces import RepresentationModule

from ..utils._amem_family import (
    DEFAULT_CATEGORY,
    DEFAULT_EMBEDDING_VERSION,
    DEFAULT_NOTE_NAMESPACE,
    repair_note_payload,
    representation_from_note_payload,
)
from ..utils._template import (
    looks_like_template,
    metadata_from_resolution_state,
    project_unit_for_template,
    render_prompt_template,
    render_prompt_template_with_recall,
    utc_now_iso,
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
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        normalized_method = str(method).strip().lower()
        if normalized_method not in self._VALID_METHODS:
            raise ValueError(f"TripleRepresentation method must be one of: {sorted(self._VALID_METHODS)}.")
        self.method = normalized_method
        self.api_key = api_key if api_key is not None else env.get("MEMPRIMITIVE_API_KEY", "")
        self.base_url = base_url if base_url is not None else env.get("MEMPRIMITIVE_BASE_URL", "")
        self.model = model if model is not None else env.get("MEMPRIMITIVE_MODEL", "")
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("TripleRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            represented_unit, extraction_trace = self._represent_unit(unit)
            represented_units.append(represented_unit)
            per_unit_trace.append(
                {
                    "unit_id": represented_unit.unit_id,
                    "method": self.method,
                    **extraction_trace,
                }
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "method": self.method,
            "unit_ids": [unit.unit_id for unit in represented_units],
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _represent_unit(self, unit: MemoryUnit) -> tuple[MemoryUnit, dict[str, Any]]:
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
            return self._replace_unit(unit, normalized_text, normalized_casefold, entities, hinted_triples), extraction_trace

        runtime = self._runtime()
        runtime.require_llm(capability=f"TripleRepresentation method {self.method!r}")

        if self.method == "direct":
            payload = self._extract_direct(runtime=runtime, text=normalized_text)
            extraction_trace = {"source": "llm_direct", "entities": payload["entities"], "triple_count": len(payload["triples"])}
        else:
            payload = self._extract_two_stage(runtime=runtime, text=normalized_text)
            extraction_trace = {"source": "llm_two_stage", "entities": payload["entities"], "triple_count": len(payload["triples"])}
        return (
            self._replace_unit(unit, normalized_text, normalized_casefold, payload["entities"], payload["triples"]),
            extraction_trace,
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
        if triples:
            elements.add("triple")
        if entities:
            elements.add("entities")
        represented = replace(
            unit,
            text=normalized_text,
            normalized_text=normalized_casefold,
            entities=entities,
            triples=triples,
            representation_elements=tuple(sorted(elements)),
        )
        return replace(
            represented,
            metadata={
                **represented.metadata,
                "representation": _representation_summary_from_unit(represented),
            },
        )

    def _runtime(self):
        from ..utils._runtime import Runtime

        return Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )

    def _extract_direct(self, *, runtime, text: str) -> dict[str, list[Any]]:
        output = runtime.run_agent(
            name="MemPrimitiveTripleDirectAgent",
            instructions=(
                "Extract a clean knowledge-graph representation from the text. "
                "Return only entities explicitly grounded in the text and relationships that can be expressed as triples. "
                "Each relationship must have non-empty subject, predicate, and object."
            ),
            input_text=json.dumps({"text": text}, ensure_ascii=False),
            temperature=0.0,
            max_turns=1,
            output_type=_TripleDirectOutput,
        )
        return self._normalize_triple_output(output)

    def _extract_two_stage(self, *, runtime, text: str) -> dict[str, list[Any]]:
        entity_output = runtime.run_agent(
            name="MemPrimitiveTripleEntityAgent",
            instructions=(
                "Extract the canonical entities explicitly mentioned in the text. "
                "Return only grounded entities and avoid duplicates."
            ),
            input_text=json.dumps({"text": text}, ensure_ascii=False),
            temperature=0.0,
            max_turns=1,
            output_type=_TripleEntityOutput,
        )
        entities = _dedupe_non_empty_strings(list(getattr(entity_output, "entities", [])))
        relationship_output = runtime.run_agent(
            name="MemPrimitiveTripleRelationAgent",
            instructions=(
                "Extract relationships from the text using the provided entities as anchors. "
                "Only emit relationships that are directly supported by the text and can be written as triples. "
                "Prefer the provided canonical entity strings for subject/object values."
            ),
            input_text=json.dumps({"text": text, "entities": entities}, ensure_ascii=False),
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

    def __init__(
        self,
        *,
        field: str,
        prompt: str,
        retrieve_pipeline=None,
        recall_query_template: str | None = None,
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
        normalized_prompt = str(prompt).strip()
        if not normalized_prompt:
            raise ValueError("LLMRepresentation requires a non-empty prompt.")
        self.field = normalized_field
        self.prompt = normalized_prompt
        self.retrieve_pipeline = retrieve_pipeline
        self.recall_query_template = None if recall_query_template is None else str(recall_query_template)
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
            represented_unit, unit_trace, store = self._represent_unit(unit, store)
            represented_units.append(represented_unit)
            per_unit_trace.append({"unit_id": represented_unit.unit_id, **unit_trace})

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "field": self.field,
            "prompt_is_template": looks_like_template(self.prompt),
            "unit_ids": [unit.unit_id for unit in represented_units],
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _represent_unit(self, unit: MemoryUnit, store: MemoryStore) -> tuple[MemoryUnit, dict[str, Any], MemoryStore]:
        normalized_text = unit.text.strip()
        normalized_casefold = normalized_text.casefold()
        payload, prompt_trace, store = self._extract_field(unit, normalized_text, store)
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
            "value": trace_value,
            **prompt_trace,
        }, store

    def _value_kind(self) -> str:
        if self.field in self._LIST_FIELDS:
            return "list"
        if self.field in self._DICT_FIELDS:
            return "dict"
        return "text"

    def _runtime(self):
        from ..utils._runtime import Runtime

        return Runtime(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            embedding_model=self.embedding_model,
        )

    def _extract_field(self, unit: MemoryUnit, text: str, store: MemoryStore) -> tuple[Any, dict[str, Any], MemoryStore]:
        rendered_prompt, prompt_trace, store = self._render_prompt(unit, store)
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
        if self.field in self._LIST_FIELDS:
            return _normalize_string_list(self._llm_json(user=user)), prompt_trace, store
        if self.field in self._DICT_FIELDS:
            return _normalize_string_dict(self._llm_json(user=user)), prompt_trace, store
        return self._llm_text(user=user), prompt_trace, store

    def _render_prompt(self, unit: MemoryUnit, store: MemoryStore) -> tuple[str, dict[str, Any], MemoryStore]:
        if not looks_like_template(self.prompt):
            return self.prompt, {
                "prompt_is_template": False,
                "rendered_prompt": self.prompt,
                "rendered_prompt_preview": self.prompt[:200],
                "missing_variables": [],
                "recalled_prompt": "",
                "recalled_prompt_preview": "",
                "recall_prompt": {
                    "enabled": False,
                    "disabled_reason": "prompt_not_template",
                    "rendered_recall_query": "",
                    "rendered_recall_query_preview": "",
                    "recall_query_template": self.recall_query_template,
                    "recall_query_template_is_template": bool(
                        self.recall_query_template and looks_like_template(self.recall_query_template)
                    ),
                    "matched": False,
                    "recalled_prompt": "",
                    "recalled_prompt_preview": "",
                    "missing_variables": [],
                    "resolved_variables": [],
                    "used_record_ids": [],
                    "used_group_ids": [],
                    "filter_trace": [],
                },
            }, store
        context = {
            "field": self.field,
            "unit": project_unit_for_template(unit),
            "runtime": {"now": utc_now_iso()},
        }
        rendered_prompt, metadata, store = render_prompt_template_with_recall(
            self.prompt,
            context,
            store=store,
            retrieve_pipeline=self.retrieve_pipeline,
            recall_query_template=self.recall_query_template,
        )
        return rendered_prompt, metadata, store

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
        shape = "a strict JSON array of strings" if self.field in self._LIST_FIELDS else "a strict JSON object"
        return runtime.json(
            system=(
                "You extract one requested representation field for a memory unit. "
                f"Follow the user-provided prompt exactly and return {shape}."
            ),
            user=user,
        )

class SemanticFieldEnrichmentRepresentation(RepresentationModule):
    """Generate enriched note fields for later graph organization and retrieval.

    Constructor: ``note_namespace`` must be a non-empty metadata namespace used
    to store the repaired note payload. ``strict_llm`` is currently required to
    be ``True`` because heuristic fallback is intentionally not supported.

    ``run`` requires ``packet.units``. It preserves unit identity and primary
    text while adding a repaired note schema plus traceable context/tags under
    ``unit.metadata[note_namespace]``. The store is unchanged.
    """

    spec = ModuleSpec(
        name="semantic_field_enrichment_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.metadata.note", "units.metadata.representation"),
    )
    requires_contracts = frozenset()
    produces_contracts = frozenset({UNIT_NOTE_PAYLOAD_CONTRACT})

    def __init__(
        self,
        *,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        strict_llm: bool = True,
        default_category: str = DEFAULT_CATEGORY,
    ) -> None:
        if not str(note_namespace).strip():
            raise ValueError("SemanticFieldEnrichmentRepresentation requires a non-empty note_namespace.")
        if not strict_llm:
            raise ValueError("SemanticFieldEnrichmentRepresentation requires strict_llm=True.")
        self.note_namespace = str(note_namespace).strip()
        self.strict_llm = strict_llm
        self.default_category = default_category

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("SemanticFieldEnrichmentRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            payload = self._analyze_unit(unit)
            represented_units.append(
                replace(
                    unit,
                    description=payload["context"],
                    tags=list(payload["tags"]),
                    metadata={
                        **unit.metadata,
                        self.note_namespace: dict(payload),
                    },
                )
            )
            per_unit_trace.append(
                {
                    "unit_id": unit.unit_id,
                    "content": payload["content"],
                    "context": payload["context"],
                    "keywords": list(payload["keywords"]),
                    "tags": list(payload["tags"]),
                    "category": payload["category"],
                    "namespace": self.note_namespace,
                }
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "note_namespace": self.note_namespace,
            "strict_llm": self.strict_llm,
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _analyze_unit(self, unit: MemoryUnit) -> dict[str, Any]:
        from ..utils._runtime import get_runtime

        runtime = get_runtime()
        runtime.require_llm(capability="SemanticFieldEnrichmentRepresentation")
        raw = runtime.json(
            system=(
                "You are the note generator for downstream graph memory modules. "
                "Return JSON with fields: content, note_text, context, keywords, tags, category, attributes."
            ),
            user=json.dumps(
                {
                    "content": unit.text,
                    "unit_text": unit.text,
                    "unit_type": unit.unit_type,
                    "existing_tags": unit.tags,
                    "existing_metadata": unit.metadata,
                },
                ensure_ascii=False,
            ),
        )
        if not isinstance(raw, dict):
            raise ValueError("SemanticFieldEnrichmentRepresentation must receive a JSON object response.")
        return repair_note_payload(raw, fallback_content=unit.text, default_category=self.default_category)


class RetrievalOrientedEmbeddingRepresentation(RepresentationModule):
    """Embed enriched note fields into a retrieval-oriented composite representation.

    Constructor: ``note_namespace`` selects which repaired note payload to read.
    ``embedding_version`` is written into trace and representation metadata so
    downstream retrieval/readout modules can reason about the embedding source.

    ``run`` requires ``packet.units`` and does not mutate ``store``. Units keep
    their original identity while gaining embedding vectors plus a structured
    ``metadata['representation']`` projection derived from the note payload.
    """

    spec = ModuleSpec(
        name="retrieval_oriented_embedding_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.embedding", "units.metadata.representation"),
    )
    requires_contracts = frozenset({UNIT_NOTE_PAYLOAD_CONTRACT})
    produces_contracts = frozenset({UNIT_EMBEDDING_CONTRACT, UNIT_NOTE_PAYLOAD_CONTRACT})

    def __init__(
        self,
        *,
        note_namespace: str = DEFAULT_NOTE_NAMESPACE,
        default_category: str = DEFAULT_CATEGORY,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        embedding_model: str | None = None,
    ) -> None:
        if not str(note_namespace).strip():
            raise ValueError("RetrievalOrientedEmbeddingRepresentation requires a non-empty note_namespace.")
        self.note_namespace = str(note_namespace).strip()
        self.default_category = default_category
        self.embedding_version = embedding_version
        load_dotenv(_MEMPRIMITIVE_ENV_PATH, override=False)
        env = os.environ
        self.embedding_model = embedding_model or env.get(
            "MEMPRIMITIVE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("RetrievalOrientedEmbeddingRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            payload = repair_note_payload(
                unit.metadata.get(self.note_namespace),
                fallback_content=unit.text,
                default_category=self.default_category,
            )
            representation = representation_from_note_payload(payload, embedding_version=self.embedding_version)
            embedding = self._embed_text(representation["enhanced_embedding_text"])
            represented_units.append(
                replace(
                    unit,
                    text=payload["content"],
                    normalized_text=payload["content"].casefold(),
                    embedding=embedding,
                    description=payload["context"],
                    tags=list(payload["tags"]),
                    representation_elements=tuple(
                        sorted(
                            {
                                *unit.representation_elements,
                                "text",
                                "embedding",
                                "description",
                                "keywords",
                                "tags",
                            }
                        )
                    ),
                    metadata={
                        **unit.metadata,
                        self.note_namespace: {
                            **payload,
                            "enhanced_embedding_text": representation["enhanced_embedding_text"],
                            "embedding_version": self.embedding_version,
                        },
                        "representation": {
                            **_representation_summary_from_unit(
                                replace(
                                    unit,
                                    text=payload["content"],
                                    normalized_text=payload["content"].casefold(),
                                    embedding=embedding,
                                    description=payload["context"],
                                    tags=list(payload["tags"]),
                                )
                            ),
                            **representation,
                        },
                    },
                )
            )
            per_unit_trace.append(
                {
                    "unit_id": unit.unit_id,
                    "namespace": self.note_namespace,
                    "embedding_version": self.embedding_version,
                    "enhanced_embedding_text": representation["enhanced_embedding_text"],
                }
            )

        trace = copy_trace(packet)
        trace["representation"] = {
            "module": self.spec.name,
            "note_namespace": self.note_namespace,
            "embedding_version": self.embedding_version,
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _embed_text(self, text: str) -> list[float]:
        from ..utils._runtime import get_runtime

        return list(get_runtime().embed(text))


BASELINE_SLOT: Final[str] = "representation"
BASELINE_CLASSES: Final[tuple[type[RepresentationModule], ...]] = (
    BasicRepresentation,
    TripleRepresentation,
    LLMRepresentation,
    SemanticFieldEnrichmentRepresentation,
    RetrievalOrientedEmbeddingRepresentation,
)
