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
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet, _representation_summary_from_unit
from ..interfaces import RepresentationModule

from ..utils._amem_family import (
    DEFAULT_CATEGORY,
    DEFAULT_EMBEDDING_VERSION,
    DEFAULT_NOTE_NAMESPACE,
    repair_note_payload,
    representation_from_note_payload,
)
from ..utils._trace import copy_trace

_VALID_ELEMENTS: Final[tuple[str, ...]] = (
    "text",
    "embedding",
    "triple",
    "kv",
    "entities",
    "tags",
    "description",
    "keywords",
    "summary",
    "time_anchor",
    "relation_tags",
    "source_type",
)
_ENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_KV_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Za-z][A-Za-z0-9_ ]{0,40}?)\s*:\s*([^.;,\n]+)")
_LIKES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][a-z]+)\s+(likes|prefers|loves|hates|works on|studies)\s+([^.;,\n]+)", re.I)
_IS_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][a-z]+)\s+is\s+([^.;,\n]+)", re.I)
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
        kv = dict(unit.kv)
        entities = list(unit.entities)
        tags = list(unit.tags)
        description = unit.description
        representation_meta: dict[str, Any] = dict(unit.metadata.get("representation", {}))

        if "text" in self.elements:
            elements.add("text")
        if "embedding" in self.elements:
            embedding = self._embed_text(normalized_text)
            elements.add("embedding")
        if "triple" in self.elements:
            triples = self._extract_triples(unit, normalized_text)
            if triples:
                elements.add("triple")
        if "kv" in self.elements:
            kv = self._extract_kv(unit, normalized_text)
            if kv:
                elements.add("kv")
        if "entities" in self.elements:
            entities = self._extract_entities(unit, normalized_text)
            if entities:
                elements.add("entities")
        if "tags" in self.elements:
            tags = self._extract_tags(unit, normalized_text, kv=kv, entities=entities, triples=triples)
            if tags:
                elements.add("tags")
        if "keywords" in self.elements:
            keywords = self._extract_keywords(unit, normalized_text, tags=tags, entities=entities)
            if keywords:
                representation_meta["keywords"] = keywords
                elements.add("keywords")
        if "summary" in self.elements:
            summary = self._build_summary(
                unit,
                normalized_text,
                kv=kv,
                entities=entities,
                triples=triples,
                tags=tags,
            )
            if summary:
                representation_meta["summary"] = summary
                elements.add("summary")
        if "time_anchor" in self.elements:
            time_anchor = self._build_time_anchor(unit)
            if time_anchor:
                representation_meta["time_anchor"] = time_anchor
                elements.add("time_anchor")
        if "relation_tags" in self.elements:
            relation_tags = self._build_relation_tags(unit, triples=triples, entities=entities)
            if relation_tags:
                representation_meta["relation_tags"] = relation_tags
                elements.add("relation_tags")
        if "source_type" in self.elements:
            source_type = self._build_source_type(unit)
            if source_type:
                representation_meta["source_type"] = source_type
                elements.add("source_type")
        if "description" in self.elements:
            description = self._describe_unit(unit, normalized_text, kv=kv, entities=entities, triples=triples, tags=tags)
            if description:
                elements.add("description")

        represented = replace(
            unit,
            text=text_value,
            representation_elements=tuple(sorted(elements)),
            normalized_text=normalized_casefold,
            embedding=embedding,
            triples=triples,
            kv=kv,
            entities=entities,
            tags=tags,
            description=description,
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

    def _extract_triples(self, unit: MemoryUnit, text: str) -> list[tuple[str, str, str]]:
        hinted = unit.metadata.get("triples")
        if isinstance(hinted, list):
            triples = []
            for item in hinted:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    triples.append((str(item[0]), str(item[1]), str(item[2])))
            if triples:
                return triples

        triples: list[tuple[str, str, str]] = []
        for match in _LIKES_PATTERN.finditer(text):
            subject, predicate, obj = match.groups()
            triples.append((subject.strip(), predicate.lower().strip(), obj.strip()))
        for match in _IS_PATTERN.finditer(text):
            subject, obj = match.groups()
            triples.append((subject.strip(), "is", obj.strip()))
        return triples

    def _extract_kv(self, unit: MemoryUnit, text: str) -> dict[str, str]:
        hinted = unit.metadata.get("kv")
        if isinstance(hinted, dict) and hinted:
            return {str(key): str(value) for key, value in hinted.items()}

        kv: dict[str, str] = {}
        for key, value in _KV_PATTERN.findall(text):
            kv[key.strip().casefold().replace(" ", "_")] = value.strip()
        for subject, predicate, obj in _LIKES_PATTERN.findall(text):
            kv[f"{subject.strip().casefold()}_{predicate.lower().strip().replace(' ', '_')}"] = obj.strip()
        for subject, value in _IS_PATTERN.findall(text):
            kv[f"{subject.strip().casefold()}_state"] = value.strip()
        return kv

    def _extract_entities(self, unit: MemoryUnit, text: str) -> list[str]:
        hinted = unit.metadata.get("entities")
        if isinstance(hinted, list) and hinted:
            return list(dict.fromkeys(str(item) for item in hinted))

        entities = [match.group(1).strip() for match in _ENTITY_PATTERN.finditer(text)]
        return list(dict.fromkeys(entity for entity in entities if entity.casefold() not in {"the", "a", "an"}))

    def _extract_tags(
        self,
        unit: MemoryUnit,
        text: str,
        *,
        kv: dict[str, str],
        entities: list[str],
        triples: list[tuple[str, str, str]],
    ) -> list[str]:
        hinted = unit.metadata.get("tags")
        if isinstance(hinted, list) and hinted:
            return list(dict.fromkeys(str(item) for item in hinted))

        tags: list[str] = [unit.unit_type]
        lowered = text.casefold()
        keyword_to_tag = {
            "graph": "graph",
            "memory": "memory",
            "code": "code",
            "skill": "skill",
            "preference": "preference",
            "likes": "preference",
            "prefers": "preference",
            "works on": "work",
            "study": "learning",
        }
        for keyword, tag in keyword_to_tag.items():
            if keyword in lowered:
                tags.append(tag)
        if entities:
            tags.append("entity_rich")
        if kv:
            tags.append("structured_kv")
        if triples:
            tags.append("structured_triple")
        return list(dict.fromkeys(tag for tag in tags if tag))

    def _extract_keywords(self, unit: MemoryUnit, text: str, *, tags: list[str], entities: list[str]) -> list[str]:
        hinted = unit.metadata.get("keywords")
        if isinstance(hinted, list) and hinted:
            return list(dict.fromkeys(str(item).casefold() for item in hinted if str(item).strip()))

        counts = Counter(
            token.casefold()
            for token in _WORD_PATTERN.findall(text)
            if token.casefold() not in _STOPWORDS
        )
        ranked = [token for token, _ in counts.most_common(6)]
        extras = [entity.casefold() for entity in entities] + [tag.casefold() for tag in tags if len(tag) > 2]
        return list(dict.fromkeys(ranked + extras))

    def _openai_plain_text(self, *, element: str, system: str, user: str) -> str:
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError(
                f"BasicRepresentation element {element!r} requires an OpenAI-compatible API: "
                "set MEMPRIMITIVE_API_KEY, MEMPRIMITIVE_BASE_URL, and MEMPRIMITIVE_MODEL "
                "(or pass api_key, base_url, model to the constructor). "
                "Heuristic fallback is not supported."
            )
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _build_summary(
        self,
        unit: MemoryUnit,
        text: str,
        *,
        kv: dict[str, str],
        entities: list[str],
        triples: list[tuple[str, str, str]],
        tags: list[str],
    ) -> str | None:
        hinted = unit.metadata.get("summary")
        if isinstance(hinted, str) and hinted.strip():
            return hinted.strip()
        collapsed = " ".join(text.split())
        if not collapsed:
            return None
        prompt = (
            "Write a concise summary (one to three short sentences) of the main facts in this memory unit, "
            "suitable for retrieval and indexing.\n"
            f"unit_type: {unit.unit_type}\n"
            f"text: {text}\n"
            f"entities: {entities}\n"
            f"tags: {tags}\n"
            f"kv: {kv}\n"
            f"triples: {triples}\n"
            "Return plain text only."
        )
        out = self._openai_plain_text(
            element="summary",
            system="You produce short factual summaries of memory units.",
            user=prompt,
        )
        return out or None

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

    def _build_relation_tags(
        self,
        unit: MemoryUnit,
        *,
        triples: list[tuple[str, str, str]],
        entities: list[str],
    ) -> list[str]:
        hinted = unit.metadata.get("relation_tags")
        if isinstance(hinted, list) and hinted:
            return list(dict.fromkeys(str(item) for item in hinted if str(item).strip()))

        relation_tags: list[str] = []
        for _, predicate, _ in triples:
            relation_tags.append(f"predicate:{predicate.replace(' ', '_')}")
        if len(entities) >= 2:
            relation_tags.append("multi_entity")
        return list(dict.fromkeys(relation_tags))

    def _build_source_type(self, unit: MemoryUnit) -> str | None:
        source = unit.metadata.get("source")
        if source is None:
            return None
        return str(source).strip() or None

    def _describe_unit(
        self,
        unit: MemoryUnit,
        text: str,
        *,
        kv: dict[str, str],
        entities: list[str],
        triples: list[tuple[str, str, str]],
        tags: list[str],
    ) -> str:
        if unit.description is not None:
            return unit.description
        prompt = (
            "Write one concise natural-language description for this memory unit.\n"
            f"unit_type: {unit.unit_type}\n"
            f"text: {text}\n"
            f"entities: {entities}\n"
            f"tags: {tags}\n"
            f"kv: {kv}\n"
            f"triples: {triples}\n"
            "Return plain text only."
        )
        return self._openai_plain_text(
            element="description",
            system="You produce short factual descriptions for memory representations.",
            user=prompt,
        )


class KeywordRepresentation(BasicRepresentation):
    """Thin wrapper around ``BasicRepresentation`` for keyword-oriented indexing.

    Constructor keeps the same embedding/API options as ``BasicRepresentation``
    but defaults the element set to ``text + keywords + tags``.
    """

    spec = ModuleSpec(
        name="keyword_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=(
            "units.text",
            "units.representation_elements",
            "units.metadata.representation",
        ),
    )

    def __init__(
        self,
        *,
        elements: tuple[str, ...] = ("text", "keywords", "tags"),
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            elements=elements,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            model=model,
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
        from ..utils._runtime import get_classic_runtime

        runtime = get_classic_runtime()
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
        from ..utils._runtime import get_classic_runtime

        return list(get_classic_runtime().embed(text))


BASELINE_SLOT: Final[str] = "representation"
BASELINE_CLASSES: Final[tuple[type[RepresentationModule], ...]] = (
    BasicRepresentation,
    KeywordRepresentation,
    SemanticFieldEnrichmentRepresentation,
    RetrievalOrientedEmbeddingRepresentation,
)
