"""Composition-contract names and helpers for pipeline legality checks."""

from __future__ import annotations

from typing import Iterable

UNIT_EMBEDDING_CONTRACT = "unit.embedding"
UNIT_ENTITIES_CONTRACT = "unit.entities"
UNIT_TAGS_CONTRACT = "unit.tags"
UNIT_PARTITION_KEY_CONTRACT = "unit.partition_key"
RECORD_GRAPH_LINKS_CONTRACT = "record.graph_links"
RECORD_NOTE_PAYLOAD_CONTRACT = "record.note_payload"
RECORD_REFLECTION_PAYLOAD_CONTRACT = "record.reflection_payload"

QUERY_TEXT_CONTRACT = "query.text"
QUERY_EMBEDDING_CONTRACT = "query.embedding"
QUERY_FEEDBACK_SCHEMA_CONTRACT = "query.feedback_schema"

TOPOLOGY_GRAPH_LAYER_CONTRACT = "topology.graph_layer"
TOPOLOGY_VECTOR_INDEX_CONTRACT = "topology.vector_index"
TOPOLOGY_KEYWORD_INDEX_CONTRACT = "topology.keyword_index"
TOPOLOGY_TAG_INDEX_CONTRACT = "topology.tag_index"
TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT = "topology.graph_vector_layer"


def normalize_contracts(values: Iterable[str]) -> frozenset[str]:
    """Return a stable frozenset of non-empty contract names."""

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        contract = str(value).strip()
        if not contract or contract in seen:
            continue
        seen.add(contract)
        normalized.append(contract)
    return frozenset(normalized)


__all__ = [
    "QUERY_EMBEDDING_CONTRACT",
    "QUERY_FEEDBACK_SCHEMA_CONTRACT",
    "QUERY_TEXT_CONTRACT",
    "RECORD_GRAPH_LINKS_CONTRACT",
    "RECORD_NOTE_PAYLOAD_CONTRACT",
    "RECORD_REFLECTION_PAYLOAD_CONTRACT",
    "TOPOLOGY_GRAPH_LAYER_CONTRACT",
    "TOPOLOGY_GRAPH_VECTOR_LAYER_CONTRACT",
    "TOPOLOGY_KEYWORD_INDEX_CONTRACT",
    "TOPOLOGY_TAG_INDEX_CONTRACT",
    "TOPOLOGY_VECTOR_INDEX_CONTRACT",
    "UNIT_EMBEDDING_CONTRACT",
    "UNIT_ENTITIES_CONTRACT",
    "UNIT_PARTITION_KEY_CONTRACT",
    "UNIT_TAGS_CONTRACT",
    "normalize_contracts",
]
