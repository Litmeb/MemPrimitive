"""Voyager classic workstream support.

This module sketches a mechanism-faithful skill-memory loop:

- `SkillExtractor` turns skill snippets into memory units.
- `CodeWithDescriptionRepresentation` keeps code and description paired.
- `UpsertByKeySkillLibrary` stores skills by stable key and updates in place.
- `MixedSkillRetrieval` combines key, code, description, tag, keyword, and embedding signals.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import math
import re
from typing import Any, Final

from ..core import MemoryRecord, MemoryStore, MemoryUnit, ModuleSpec, Observation, Packet, Placement, Query, RetrievedSet, StoreLayerSpec
from ..exceptions import IncompatibleCompositionError
from ..interfaces import OrganizationModule, RepresentationModule, RetrievalModule, UnitFormationModule
from ._runtime import get_classic_runtime

_CODE_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>.*?)```", re.S)
_HEADER_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?i)^(skill|skill key|key|name|description|desc)\s*[:=-]\s*(.+)$")
_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?m)^#{1,6}\s*(.+)$")
_FUNCTION_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "with",
    }
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_key(value: Any) -> str:
    cleaned = _clean_text(value).casefold()
    if not cleaned:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return re.sub(r"_+", "_", normalized)


def _dedupe(values: list[Any] | tuple[Any, ...] | set[Any]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _clean_text(raw)
        if not text:
            continue
        normalized = _normalize_key(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def _as_text_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(_clean_text(item) for item in value if _clean_text(item))
    return ()


def _tokenize(text: str) -> list[str]:
    normalized = _clean_text(text).casefold().replace("_", " ")
    return [token for token in _TOKEN_PATTERN.findall(normalized) if token and token not in _STOPWORDS]


def _runtime_embedding(text: str) -> list[float]:
    return get_classic_runtime().embed(text)


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    if len(left) != len(right):
        return 0.0
    numerator = sum(lv * rv for lv, rv in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _skill_card_text(key: str, description: str, code: str, *, language: str = "") -> str:
    parts: list[str] = []
    if key:
        parts.append(f"Skill: {key}")
    if description:
        parts.append(f"Description: {description}")
    if code:
        if parts:
            parts.append("")
        parts.append(f"```{language}".rstrip())
        parts.append(code.rstrip())
        parts.append("```")
    return "\n".join(parts).strip()


def _strip_header_lines(text: str) -> str:
    lines: list[str] = []
    for line in _clean_text(text).splitlines():
        if _HEADER_LINE_PATTERN.match(line.strip()):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _parse_header(text: str) -> dict[str, str]:
    key = ""
    description = ""
    heading = ""
    for line in _clean_text(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        header_match = _HEADER_LINE_PATTERN.match(stripped)
        if header_match:
            label = header_match.group(1).casefold()
            value = header_match.group(2).strip()
            if label in {"skill", "skill key", "key", "name"} and not key:
                key = value
            elif label in {"description", "desc"} and not description:
                description = value
            continue
        if not heading:
            heading_match = _HEADING_PATTERN.match(stripped)
            if heading_match:
                heading = heading_match.group(1).strip()
    return {"key": key or heading, "description": description}


def _infer_skill_key(*, description: str, code: str, fallback_index: int) -> str:
    for candidate in (code, description):
        match = _FUNCTION_NAME_PATTERN.search(candidate)
        if match:
            return _normalize_key(match.group(1))
    first_line = code.splitlines()[0] if code.splitlines() else ""
    normalized = _normalize_key(description or first_line)
    return normalized or f"skill_{fallback_index + 1}"


def _keyword_list(*parts: str, limit: int = 8) -> list[str]:
    counts = Counter()
    for part in parts:
        counts.update(token for token in _tokenize(part) if token not in _STOPWORDS)
    return [token for token, _ in counts.most_common(limit)]


def _extract_tags(*, skill_type: str, language: str, user_tags: tuple[str, ...]) -> list[str]:
    tags = [skill_type]
    if language:
        tags.append(f"language:{_normalize_key(language)}")
    tags.extend(user_tags)
    return list(dict.fromkeys(tag for tag in tags if tag))


def _skill_hint_values(hint: Any) -> dict[str, Any]:
    if isinstance(hint, dict):
        return dict(hint)
    if isinstance(hint, str):
        return {"text": hint}
    return {"text": str(hint)}


def _profile_from_skill_mapping(mapping: dict[str, Any], *, fallback_text: str, fallback_index: int) -> dict[str, Any]:
    key = _normalize_key(
        mapping.get("skill_key")
        or mapping.get("key")
        or mapping.get("name")
        or mapping.get("title")
        or mapping.get("id")
        or ""
    )
    code = _clean_text(mapping.get("code") or mapping.get("snippet") or mapping.get("body") or mapping.get("text") or fallback_text)
    description = _clean_text(mapping.get("description") or mapping.get("summary") or mapping.get("doc"))
    language = _clean_text(mapping.get("language") or mapping.get("lang"))
    skill_type = _normalize_key(mapping.get("skill_type") or mapping.get("category") or mapping.get("type") or "skill") or "skill"
    tags = _dedupe(_as_text_list(mapping.get("tags")))
    aliases = _dedupe(_as_text_list(mapping.get("aliases")))
    extra_metadata = {
        key_name: value
        for key_name, value in mapping.items()
        if key_name
        not in {
            "skill_key",
            "key",
            "name",
            "title",
            "id",
            "code",
            "snippet",
            "body",
            "text",
            "description",
            "summary",
            "doc",
            "language",
            "lang",
            "skill_type",
            "category",
            "type",
            "tags",
            "aliases",
        }
    }

    if not key:
        key = _infer_skill_key(description=description or fallback_text, code=code or fallback_text, fallback_index=fallback_index)
    if not description:
        description = _parse_header(fallback_text).get("description", "") or key.replace("_", " ")
    if not code:
        code = fallback_text.strip()

    return {
        "key": key,
        "code": code,
        "description": description,
        "language": language,
        "skill_type": skill_type,
        "tags": tags,
        "aliases": aliases,
        "metadata": extra_metadata,
    }


def _profile_from_text(text: str, *, fallback_index: int) -> list[dict[str, Any]]:
    body = _clean_text(text)
    if not body:
        return []

    snippets: list[dict[str, Any]] = []
    matches = list(_CODE_FENCE_PATTERN.finditer(body))
    if matches:
        cursor = 0
        for idx, match in enumerate(matches):
            prefix = body[cursor:match.start()]
            header = _parse_header(prefix)
            code = _clean_text(match.group("code"))
            language = _clean_text(match.group("language"))
            prefix_description = header["description"] or ""
            if not prefix_description:
                prefix_lines = [line.strip() for line in prefix.splitlines() if line.strip()]
                if prefix_lines:
                    prefix_description = prefix_lines[-1]
            key = _normalize_key(header["key"]) or _infer_skill_key(
                description=prefix_description or prefix,
                code=code,
                fallback_index=fallback_index + idx,
            )
            snippets.append(
                {
                    "key": key,
                    "code": code,
                    "description": prefix_description or key.replace("_", " "),
                    "language": language,
                    "skill_type": "skill",
                    "tags": (),
                    "aliases": (),
                    "metadata": {},
                }
            )
            cursor = match.end()
        return snippets

    header = _parse_header(body)
    stripped = _strip_header_lines(body)
    description = header["description"]
    key = _normalize_key(header["key"]) or _infer_skill_key(
        description=description or body,
        code=stripped or body,
        fallback_index=fallback_index,
    )
    code = stripped or body
    snippets.append(
        {
            "key": key,
            "code": code,
            "description": description or key.replace("_", " "),
            "language": "",
            "skill_type": "skill",
            "tags": (),
            "aliases": (),
            "metadata": {},
        }
    )
    return snippets


def _snippet_from_hint(hint: Any, *, fallback_text: str, fallback_index: int) -> dict[str, Any]:
    return _profile_from_skill_mapping(_skill_hint_values(hint), fallback_text=fallback_text, fallback_index=fallback_index)


def _snippet_card_metadata(snippet: dict[str, Any], *, observation: Observation, snippet_index: int) -> dict[str, Any]:
    skill = {
        "key": snippet["key"],
        "code": snippet["code"],
        "description": snippet["description"],
        "language": snippet["language"],
        "skill_type": snippet["skill_type"],
        "tags": list(snippet["tags"]),
        "aliases": list(snippet["aliases"]),
        "snippet_index": snippet_index,
        **snippet["metadata"],
    }
    return {
        **observation.metadata,
        "source": observation.source,
        "provenance": {
            "observation_id": observation.observation_id,
            "source": observation.source,
            "snippet_index": snippet_index,
        },
        "skill": skill,
        "skill_key": snippet["key"],
        "skill_type": snippet["skill_type"],
        "description": snippet["description"],
        "language": snippet["language"],
    }


def _snippet_to_unit(snippet: dict[str, Any], *, observation: Observation, snippet_index: int) -> MemoryUnit:
    tags = _extract_tags(skill_type=snippet["skill_type"], language=snippet["language"], user_tags=snippet["tags"])
    text = _skill_card_text(snippet["key"], snippet["description"], snippet["code"], language=snippet["language"])
    return MemoryUnit(
        text=text,
        unit_type=snippet["skill_type"] or "skill",
        timestamp=observation.timestamp,
        description=snippet["description"] or None,
        tags=tags,
        metadata=_snippet_card_metadata(snippet, observation=observation, snippet_index=snippet_index),
    )


def _skill_profile_from_unit(unit: MemoryUnit) -> dict[str, Any]:
    skill_meta = unit.metadata.get("skill") if isinstance(unit.metadata.get("skill"), dict) else {}
    representation = unit.metadata.get("representation") if isinstance(unit.metadata.get("representation"), dict) else {}
    key = _normalize_key(
        skill_meta.get("key")
        or unit.metadata.get("skill_key")
        or representation.get("skill_key")
        or representation.get("key")
        or unit.text
    )
    code = _clean_text(skill_meta.get("code") or representation.get("code") or unit.metadata.get("code") or unit.text)
    description = _clean_text(
        skill_meta.get("description")
        or representation.get("description")
        or unit.description
        or unit.metadata.get("description")
    )
    language = _clean_text(skill_meta.get("language") or representation.get("language") or unit.metadata.get("language"))
    skill_type = _normalize_key(skill_meta.get("skill_type") or unit.metadata.get("skill_type") or unit.unit_type or "skill") or "skill"
    tags = _dedupe(list(unit.tags) + list(_as_text_list(skill_meta.get("tags"))) + list(_as_text_list(representation.get("tags"))))
    aliases = _dedupe(_as_text_list(skill_meta.get("aliases")))
    keywords = representation.get("keywords") if isinstance(representation.get("keywords"), list) else _keyword_list(key, description, code, " ".join(tags))
    return {
        "key": key,
        "code": code,
        "description": description,
        "language": language,
        "skill_type": skill_type,
        "tags": tags,
        "aliases": aliases,
        "keywords": _dedupe(keywords),
        "text": unit.text,
    }


def _skill_profile_from_record(record: MemoryRecord) -> dict[str, Any]:
    representation = record.metadata.get("representation") if isinstance(record.metadata.get("representation"), dict) else {}
    skill_meta = record.metadata.get("skill") if isinstance(record.metadata.get("skill"), dict) else {}
    key = _normalize_key(
        record.metadata.get("skill_key")
        or skill_meta.get("key")
        or representation.get("skill_key")
        or representation.get("key")
        or record.text
    )
    code = _clean_text(skill_meta.get("code") or representation.get("code") or record.text)
    description = _clean_text(skill_meta.get("description") or representation.get("description") or record.metadata.get("description") or record.text)
    language = _clean_text(skill_meta.get("language") or representation.get("language") or record.metadata.get("language"))
    skill_type = _normalize_key(skill_meta.get("skill_type") or record.metadata.get("skill_type") or "skill") or "skill"
    tags = _dedupe(
        list(_as_text_list(record.metadata.get("tags")))
        + list(_as_text_list(skill_meta.get("tags")))
        + list(_as_text_list(representation.get("tags")))
    )
    aliases = _dedupe(_as_text_list(skill_meta.get("aliases")))
    keywords = representation.get("keywords") if isinstance(representation.get("keywords"), list) else _keyword_list(key, description, code, " ".join(tags))
    embedding = record.embedding if record.embedding is not None else _runtime_embedding("\n".join([key, description, code, " ".join(tags)]))
    return {
        "key": key,
        "code": code,
        "description": description,
        "language": language,
        "skill_type": skill_type,
        "tags": tags,
        "aliases": aliases,
        "keywords": _dedupe(keywords),
        "embedding": embedding,
        "text": record.text,
    }


def _skill_profile_from_query(query: Query) -> dict[str, Any]:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    key = _normalize_key(metadata.get("skill_key") or metadata.get("skill") or metadata.get("key") or query.text)
    text = _clean_text(query.text)
    return {
        "key": key,
        "text": text,
        "tokens": _tokenize(text),
        "embedding": query.embedding if query.embedding is not None else _runtime_embedding(text),
    }


def _score_skill_match(query_profile: dict[str, Any], record_profile: dict[str, Any]) -> tuple[float, dict[str, float]]:
    query_tokens = set(query_profile["tokens"])
    record_tokens = set(_tokenize(record_profile["text"]))
    code_tokens = set(_tokenize(record_profile["code"]))
    description_tokens = set(_tokenize(record_profile["description"]))
    keyword_tokens = set(record_profile["keywords"])
    tag_tokens = set(record_profile["tags"])

    key_score = 0.0
    if query_profile["key"] and record_profile["key"]:
        if query_profile["key"] == record_profile["key"]:
            key_score = 1.0
        elif query_profile["key"] in record_profile["key"] or record_profile["key"] in query_profile["key"]:
            key_score = 0.7

    code_overlap = float(len(query_tokens & code_tokens))
    description_overlap = float(len(query_tokens & description_tokens))
    keyword_overlap = float(len(query_tokens & keyword_tokens))
    tag_overlap = float(len(query_tokens & tag_tokens))
    lexical_overlap = float(len(query_tokens & record_tokens))
    embedding_similarity = _cosine_similarity(query_profile["embedding"], record_profile["embedding"])

    score = (
        key_score * 5.0
        + code_overlap * 1.2
        + description_overlap * 1.6
        + keyword_overlap * 1.1
        + tag_overlap * 0.9
        + lexical_overlap * 0.25
        + embedding_similarity * 0.75
    )
    return score, {
        "key": key_score,
        "code_overlap": code_overlap,
        "description_overlap": description_overlap,
        "keyword_overlap": keyword_overlap,
        "tag_overlap": tag_overlap,
        "lexical_overlap": lexical_overlap,
        "embedding_similarity": embedding_similarity,
    }


@dataclass(slots=True)
class SkillSnippet:
    """Structured Voyager skill snippet extracted from an observation."""

    key: str
    code: str
    description: str = ""
    language: str = ""
    skill_type: str = "skill"
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.key = _normalize_key(self.key)
        self.code = _clean_text(self.code)
        self.description = _clean_text(self.description)
        self.language = _clean_text(self.language)
        self.skill_type = _normalize_key(self.skill_type) or "skill"
        self.tags = _dedupe(list(self.tags))
        self.aliases = _dedupe(list(self.aliases))
        self.metadata = dict(self.metadata)

    def card_text(self) -> str:
        return _skill_card_text(self.key, self.description, self.code, language=self.language)

    def to_unit(self, observation: Observation, *, snippet_index: int) -> MemoryUnit:
        snippet = {
            "key": self.key,
            "code": self.code,
            "description": self.description,
            "language": self.language,
            "skill_type": self.skill_type,
            "tags": self.tags,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }
        return _snippet_to_unit(snippet, observation=observation, snippet_index=snippet_index)


class SkillExtractor(UnitFormationModule):
    """Turn skill snippets into Voyager skill units."""

    spec = ModuleSpec(
        name="voyager_skill_extractor",
        slot="unit_formation",
        input_requirements=("observation.text",),
        output_guarantees=("units", "units.metadata.skill", "units.metadata.skill_key"),
    )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.observation is None:
            raise ValueError("SkillExtractor requires packet.observation.")

        observation = packet.observation
        snippets = self._extract_snippets(observation)
        units = [snippet.to_unit(observation, snippet_index=index) for index, snippet in enumerate(snippets)]

        trace = dict(packet.trace)
        trace["unit_formation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in units],
            "unit_count": len(units),
            "skill_keys": [unit.metadata.get("skill_key") for unit in units],
            "extraction_mode": "metadata" if observation.metadata.get("skills") or observation.metadata.get("skill") else "text",
        }
        return replace(packet, units=units, trace=trace), store

    def _extract_snippets(self, observation: Observation) -> list[SkillSnippet]:
        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        skills = metadata.get("skills")
        if isinstance(skills, list) and skills:
            return [
                SkillSnippet(**_snippet_from_hint(hint, fallback_text=observation.text, fallback_index=index))
                for index, hint in enumerate(skills)
            ]

        skill = metadata.get("skill")
        if skill:
            return [SkillSnippet(**_snippet_from_hint(skill, fallback_text=observation.text, fallback_index=0))]

        return [SkillSnippet(**snippet) for snippet in _profile_from_text(observation.text, fallback_index=0)]


class CodeWithDescriptionRepresentation(RepresentationModule):
    """Keep code and description together while adding deterministic skill features."""

    spec = ModuleSpec(
        name="voyager_code_with_description_representation",
        slot="representation",
        input_requirements=("units",),
        output_guarantees=("units.text", "units.description", "units.embedding", "units.metadata.representation"),
    )

    def __init__(self, *, elements: tuple[str, ...] = ("text", "code", "description", "embedding", "keywords", "tags")) -> None:
        normalized = tuple(dict.fromkeys(elements))
        unsupported = [element for element in normalized if element not in {"text", "code", "description", "embedding", "keywords", "tags", "language", "skill_key"}]
        if unsupported:
            raise ValueError(f"Unsupported Voyager representation element(s): {unsupported}.")
        self.elements = normalized

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("CodeWithDescriptionRepresentation requires packet.units.")

        represented_units: list[MemoryUnit] = []
        per_unit_trace: list[dict[str, Any]] = []
        for unit in packet.units:
            represented_unit = self._represent_unit(unit)
            represented_units.append(represented_unit)
            per_unit_trace.append(
                {
                    "unit_id": represented_unit.unit_id,
                    "skill_key": represented_unit.metadata.get("skill_key"),
                    "elements": list(represented_unit.representation_elements),
                    "has_embedding": represented_unit.embedding is not None,
                }
            )

        trace = dict(packet.trace)
        trace["representation"] = {
            "module": self.spec.name,
            "unit_ids": [unit.unit_id for unit in represented_units],
            "elements": list(self.elements),
            "per_unit": per_unit_trace,
        }
        return replace(packet, units=represented_units, trace=trace), store

    def _represent_unit(self, unit: MemoryUnit) -> MemoryUnit:
        profile = _skill_profile_from_unit(unit)
        representation_meta = dict(unit.metadata.get("representation", {})) if isinstance(unit.metadata.get("representation"), dict) else {}
        representation_meta.update(
            {
                "skill_key": profile["key"],
                "code": profile["code"],
                "description": profile["description"],
                "language": profile["language"],
                "skill_type": profile["skill_type"],
                "keywords": profile["keywords"],
                "tags": list(profile["tags"]),
                "aliases": list(profile["aliases"]),
            }
        )
        if "embedding" in self.elements:
            embedding = _runtime_embedding(
                "\n".join(
                    part
                    for part in (
                        profile["key"],
                        profile["description"],
                        profile["code"],
                        " ".join(profile["tags"]),
                        " ".join(profile["keywords"]),
                    )
                    if part
                )
            )
        else:
            embedding = unit.embedding

        tags = _extract_tags(skill_type=profile["skill_type"], language=profile["language"], user_tags=tuple(profile["tags"]))
        representation_elements = tuple(dict.fromkeys(("text", *self.elements)))
        return replace(
            unit,
            text=_skill_card_text(profile["key"], profile["description"], profile["code"], language=profile["language"]),
            description=profile["description"] or unit.description,
            embedding=embedding,
            tags=tags,
            representation_elements=representation_elements,
            normalized_text=_clean_text(unit.text).casefold(),
            metadata={
                **unit.metadata,
                "skill_key": profile["key"],
                "skill_type": profile["skill_type"],
                "language": profile["language"],
                "description": profile["description"],
                "skill": {
                    **(unit.metadata.get("skill") if isinstance(unit.metadata.get("skill"), dict) else {}),
                    "key": profile["key"],
                    "code": profile["code"],
                    "description": profile["description"],
                    "language": profile["language"],
                    "skill_type": profile["skill_type"],
                    "tags": list(profile["tags"]),
                    "aliases": list(profile["aliases"]),
                },
                "representation": representation_meta,
            },
        )


class UpsertByKeySkillLibrary(OrganizationModule):
    """Upsert Voyager skills into a key-addressed skill library layer."""

    spec = ModuleSpec(
        name="voyager_upsert_by_key_skill_library",
        slot="organization",
        input_requirements=("units", "decisions"),
        output_guarantees=("placements",),
        side_effects=("modify_store", "append_records"),
    )

    def __init__(
        self,
        *,
        target_layer: str = "skill_library",
        key_field: str = "skill_key",
        allow_create_layer: bool = True,
        layer_theme: str = "skill",
        layer_indices: tuple[str, ...] = ("keyword", "tag", "vector"),
    ) -> None:
        self.target_layer = target_layer
        self.key_field = key_field
        self.allow_create_layer = allow_create_layer
        self.layer_theme = layer_theme
        self.layer_indices = layer_indices

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.target_layer):
            if not self.allow_create_layer:
                raise IncompatibleCompositionError(
                    f"UpsertByKeySkillLibrary requires declared layer {self.target_layer!r}."
                )
            return
        for index_name in ("keyword", "tag"):
            if not store.layer_supports_index(self.target_layer, index_name):
                raise IncompatibleCompositionError(
                    f"UpsertByKeySkillLibrary requires index {index_name!r} on layer {self.target_layer!r}."
                )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.units is None:
            raise ValueError("UpsertByKeySkillLibrary requires packet.units.")
        if packet.decisions is None:
            raise ValueError("UpsertByKeySkillLibrary requires packet.decisions.")
        if len(packet.units) != len(packet.decisions):
            raise ValueError("UpsertByKeySkillLibrary requires decisions aligned with units.")

        self._ensure_layer(store)
        placements = [Placement(unit_id=unit.unit_id, target_layer=self.target_layer) for unit in packet.units]
        written_record_ids: list[str] = []
        written_unit_ids: list[str] = []
        updated_record_ids: list[str] = []
        inserted_record_ids: list[str] = []
        upserted_keys: list[str] = []
        skipped_units = 0

        for unit, decision, placement in zip(packet.units, packet.decisions, placements, strict=True):
            if not decision:
                skipped_units += 1
                continue
            skill_key = self._skill_key_from_unit(unit)
            record, was_update = self._upsert_unit(store, unit, skill_key)
            written_record_ids.append(record.record_id)
            written_unit_ids.append(unit.unit_id)
            upserted_keys.append(skill_key)
            if was_update:
                updated_record_ids.append(record.record_id)
            else:
                inserted_record_ids.append(record.record_id)

        trace = dict(packet.trace)
        trace["organization"] = {
            "module": self.spec.name,
            "target_layer": self.target_layer,
            "write_mode": "upsert_by_key",
            "key_field": self.key_field,
            "placements": [
                {"unit_id": placement.unit_id, "target_layer": placement.target_layer}
                for placement in placements
            ],
            "written_record_ids": written_record_ids,
            "written_unit_ids": written_unit_ids,
            "updated_record_ids": updated_record_ids,
            "inserted_record_ids": inserted_record_ids,
            "upserted_keys": upserted_keys,
            "skipped_unit_count": skipped_units,
        }
        return replace(packet, placements=placements, trace=trace), store

    def _ensure_layer(self, store: MemoryStore) -> None:
        if store.has_layer(self.target_layer):
            return
        if not self.allow_create_layer:
            raise ValueError(f"Target layer {self.target_layer!r} is not declared in the store topology.")
        store.topology = store.topology.with_added_layer(
            StoreLayerSpec(name=self.target_layer, theme=self.layer_theme, indices=self.layer_indices)
        )
        store.layers[self.target_layer] = []

    def _skill_key_from_unit(self, unit: MemoryUnit) -> str:
        skill_meta = unit.metadata.get("skill") if isinstance(unit.metadata.get("skill"), dict) else {}
        key = _normalize_key(
            unit.metadata.get("skill_key")
            or skill_meta.get("key")
            or unit.metadata.get(self.key_field)
            or unit.text
        )
        if key:
            return key
        return _infer_skill_key(description=unit.description or unit.text, code=unit.text, fallback_index=0)

    def _upsert_unit(self, store: MemoryStore, unit: MemoryUnit, skill_key: str) -> tuple[MemoryRecord, bool]:
        existing_index = None
        existing_record = None
        for index, record in enumerate(store.layers[self.target_layer]):
            record_meta = record.metadata if isinstance(record.metadata, dict) else {}
            record_skill = record_meta.get("skill") if isinstance(record_meta.get("skill"), dict) else {}
            record_representation = record_meta.get("representation") if isinstance(record_meta.get("representation"), dict) else {}
            record_key = _normalize_key(
                record_meta.get(self.key_field)
                or record_skill.get("key")
                or record_representation.get("skill_key")
                or record_representation.get("key")
                or record.text
            )
            if record_key == skill_key:
                existing_index = index
                existing_record = record
                break

        if existing_index is None or existing_record is None:
            fresh_record = MemoryRecord.from_unit(
                unit=unit,
                layer=self.target_layer,
                sequence_id=store.next_sequence_id(),
            )
            store.layers[self.target_layer].append(fresh_record)
            return fresh_record, False

        fresh_record = MemoryRecord.from_unit(unit=unit, layer=self.target_layer, sequence_id=0)
        merged_metadata = {**existing_record.metadata, **fresh_record.metadata, self.key_field: skill_key}
        updated_record = replace(fresh_record, record_id=existing_record.record_id, unit_id=unit.unit_id, metadata=merged_metadata)
        store.layers[self.target_layer][existing_index] = updated_record
        return updated_record, True


class MixedSkillRetrieval(RetrievalModule):
    """Blend exact-key, lexical, tag, keyword, and embedding skill retrieval."""

    spec = ModuleSpec(
        name="voyager_mixed_skill_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, *, top_k: int = 5, layer: str = "skill_library") -> None:
        if top_k <= 0:
            raise ValueError("MixedSkillRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def validate_store(self, store: MemoryStore) -> None:
        if not store.has_layer(self.layer):
            raise IncompatibleCompositionError(
                f"MixedSkillRetrieval requires declared layer {self.layer!r}."
            )
        for index_name in ("keyword", "tag"):
            if not store.layer_supports_index(self.layer, index_name):
                raise IncompatibleCompositionError(
                    f"MixedSkillRetrieval requires index {index_name!r} on layer {self.layer!r}."
                )

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("MixedSkillRetrieval requires packet.query.")
        if not store.has_layer(self.layer):
            retrieved = RetrievedSet(
                items=[],
                scores=[],
                trace={
                    "module": self.spec.name,
                    "top_k": self.top_k,
                    "candidate_count": 0,
                    "used_recency_fallback": False,
                    "skill_layer": self.layer,
                },
            )
            trace = dict(packet.trace)
            trace["retrieval"] = retrieved.trace
            return replace(packet, retrieved=retrieved, trace=trace), store

        query_profile = _skill_profile_from_query(packet.query)
        all_records = list(reversed(store.iter_records(self.layer)))
        candidates = []
        signal_map: dict[str, dict[str, float]] = {}
        for order_index, record in enumerate(all_records):
            record_profile = _skill_profile_from_record(record)
            _, signals = _score_skill_match(query_profile, record_profile)
            signal_map[record.record_id] = signals
            candidates.append(
                {
                    "id": record.record_id,
                    "order_index": order_index,
                    "skill_key": record_profile["key"],
                    "description": record_profile["description"],
                    "code": record_profile["code"],
                    "language": record_profile["language"],
                    "tags": list(record_profile["tags"]),
                    "keywords": list(record_profile["keywords"]),
                    "signals": signals,
                }
            )

        reranked = get_classic_runtime().rerank(
            query=f"{packet.query.text}\nskill_key:{query_profile['key']}",
            candidates=candidates,
            task="Voyager skill retrieval over code, description, tags, and semantic skill intent",
            top_k=self.top_k,
        )
        ranking = {item["id"]: item for item in reranked}
        used_recency_fallback = not bool(reranked)
        if used_recency_fallback:
            items = all_records[: self.top_k]
        else:
            items = [
                next(record for record in all_records if record.record_id == item["id"])
                for item in reranked
            ]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "score": float(ranking.get(record.record_id, {}).get("score", 0.0)),
                "strategy": "mixed_skill",
                "signals": signal_map.get(record.record_id, {}),
                "rationale": str(ranking.get(record.record_id, {}).get("rationale", "")).strip(),
            }
            for rank, record in enumerate(items, start=1)
        ]
        retrieved = RetrievedSet(
            items=items,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "candidate_count": len(all_records),
                "skill_layer": self.layer,
                "query_key": query_profile["key"],
                "used_recency_fallback": used_recency_fallback,
            },
        )
        trace = dict(packet.trace)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


__all__ = (
    "CodeWithDescriptionRepresentation",
    "MixedSkillRetrieval",
    "SkillExtractor",
    "SkillSnippet",
    "UpsertByKeySkillLibrary",
)
