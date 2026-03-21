"""Baseline: representation primitive."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Final

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..core import MemoryStore, MemoryUnit, ModuleSpec, Packet, _representation_summary_from_unit
from ..interfaces import RepresentationModule

from ._trace import copy_trace

_VALID_ELEMENTS: Final[tuple[str, ...]] = ("text", "embedding", "triple", "kv", "entities", "tags", "description")
_ENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_KV_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Za-z][A-Za-z0-9_ ]{0,40}?)\s*:\s*([^.;,\n]+)")
_LIKES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][a-z]+)\s+(likes|prefers|loves|hates|works on|studies)\s+([^.;,\n]+)", re.I)
_IS_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][a-z]+)\s+is\s+([^.;,\n]+)", re.I)


def _load_local_env() -> dict[str, str]:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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
        env = _load_local_env()
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
                "representation": _representation_summary_from_unit(represented),
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
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError(
                "Description generation requires MEMPRIMITIVE_API_KEY, MEMPRIMITIVE_BASE_URL, and MEMPRIMITIVE_MODEL."
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You produce short factual descriptions for memory representations."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()


BASELINE_SLOT: Final[str] = "representation"
BASELINE_CLASSES: Final[tuple[type[RepresentationModule], ...]] = (BasicRepresentation,)
