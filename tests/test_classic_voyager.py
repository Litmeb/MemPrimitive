from __future__ import annotations

import pytest

from memprimitive import MemoryPipeline, MemoryStore, Observation, Packet, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import AlwaysWriteTrigger, ConcatenateReadout
from memprimitive.classic_modules.voyager import (
    CodeWithDescriptionRepresentation,
    MixedSkillRetrieval,
    SkillExtractor,
    UpsertByKeySkillLibrary,
)


pytestmark = pytest.mark.usefixtures("require_real_classic_runtime")


def _skill_store() -> MemoryStore:
    return MemoryStore(
        topology=StoreTopology.from_layers(
            [
                StoreLayerSpec(
                    name="skill_library",
                    theme="skill",
                    indices=("keyword", "tag", "vector"),
                ),
            ]
        )
    )


def _ingest_skill(store: MemoryStore, text: str, *, source: str = "skill") -> tuple[Packet, MemoryStore]:
    packet = Packet(observation=Observation(text=text, source=source))
    packet, store = SkillExtractor().run(packet, store)
    packet, store = CodeWithDescriptionRepresentation().run(packet, store)
    packet, store = AlwaysWriteTrigger().run(packet, store)
    packet, store = UpsertByKeySkillLibrary(target_layer="skill_library").run(packet, store)
    return packet, store


def test_skill_extractor_parses_fenced_skill_text_into_card_units() -> None:
    packet_out, _ = SkillExtractor().run(
        Packet(
            observation=Observation(
                text=(
                    "Skill: craft_planks\n"
                    "Description: Turns logs into planks at a bench.\n\n"
                    "```python\n"
                    "def craft_planks(logs):\n"
                    "    return [log[:4] for log in logs]\n"
                    "```"
                ),
                source="skill",
            )
        ),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert len(packet_out.units) == 1
    unit = packet_out.units[0]
    assert unit.unit_type == "skill"
    assert unit.text.startswith("Skill: craft_planks")
    assert "def craft_planks" in unit.metadata["skill"]["code"]
    assert unit.metadata["skill_key"] == "craft_planks"
    assert unit.description == "Turns logs into planks at a bench."


def test_skill_extractor_can_materialize_multiple_metadata_hints() -> None:
    packet_out, _ = SkillExtractor().run(
        Packet(
            observation=Observation(
                text="Voyager skill notes",
                source="skill",
                metadata={
                    "skills": [
                        {
                            "key": "craft_planks",
                            "description": "Turns logs into planks at a bench.",
                            "code": "def craft_planks(logs):\n    return [log[:4] for log in logs]",
                            "language": "python",
                            "tags": ["craft"],
                        },
                        {
                            "key": "repair_shield",
                            "description": "Repairs a shield using iron.",
                            "code": "def repair_shield(shield):\n    return shield",
                            "language": "python",
                            "tags": ["repair"],
                        },
                    ]
                },
            )
        ),
        MemoryStore(),
    )

    assert packet_out.units is not None
    assert [unit.metadata["skill_key"] for unit in packet_out.units] == ["craft_planks", "repair_shield"]


def test_code_with_description_representation_keeps_code_description_and_embedding() -> None:
    packet, store = SkillExtractor().run(
        Packet(
            observation=Observation(
                text=(
                    "Skill: craft_planks\n"
                    "Description: Turns logs into planks at a bench.\n\n"
                    "```python\n"
                    "def craft_planks(logs):\n"
                    "    return [log[:4] for log in logs]\n"
                    "```"
                ),
                source="skill",
            )
        ),
        MemoryStore(),
    )

    packet_out, _ = CodeWithDescriptionRepresentation().run(packet, store)
    unit = packet_out.units[0]

    assert unit.embedding is not None
    assert "code" in unit.representation_elements
    assert "description" in unit.representation_elements
    assert "embedding" in unit.representation_elements
    assert unit.metadata["representation"]["code"].startswith("def craft_planks")
    assert unit.metadata["representation"]["description"] == "Turns logs into planks at a bench."
    assert unit.metadata["skill_key"] == "craft_planks"


def test_upsert_by_key_skill_library_replaces_existing_skill_record() -> None:
    store = _skill_store()

    _, store = _ingest_skill(
        store,
        (
            "Skill: craft_planks\n"
            "Description: Turns logs into planks at a bench.\n\n"
            "```python\n"
            "def craft_planks(logs):\n"
            "    return [log[:4] for log in logs]\n"
            "```"
        ),
    )
    _, store = _ingest_skill(
        store,
        (
            "Skill: craft_planks\n"
            "Description: Turns logs into planks faster at a bench.\n\n"
            "```python\n"
            "def craft_planks(logs):\n"
            "    return [log.strip()[:4] for log in logs]\n"
            "```"
        ),
        source="skill_update",
    )

    assert store.count("skill_library") == 1
    record = store.iter_records("skill_library")[0]
    assert record.metadata["skill_key"] == "craft_planks"
    assert record.metadata["skill"]["description"] == "Turns logs into planks faster at a bench."
    assert "faster at a bench" in record.text


def test_mixed_skill_retrieval_prioritizes_exact_key_and_uses_mixed_signals() -> None:
    store = _skill_store()
    _, store = _ingest_skill(
        store,
        (
            "Skill: craft_planks\n"
            "Description: Turns logs into planks at a bench.\n\n"
            "```python\n"
            "def craft_planks(logs):\n"
            "    return [log[:4] for log in logs]\n"
            "```"
        ),
    )
    _, store = _ingest_skill(
        store,
        (
            "Skill: repair_shield\n"
            "Description: Repairs a shield using iron.\n\n"
            "```python\n"
            "def repair_shield(shield):\n"
            "    return shield\n"
            "```"
        ),
        source="skill",
    )
    _, store = _ingest_skill(
        store,
        (
            "Skill: cook_stew\n"
            "Description: Cooks a warm stew for the team.\n\n"
            "```python\n"
            "def cook_stew(ingredients):\n"
            "    return ingredients\n"
            "```"
        ),
        source="skill",
    )

    packet_out, _ = MixedSkillRetrieval(top_k=2).run(Packet(query=Query(text="craft planks")), store)
    assert packet_out.retrieved is not None
    assert packet_out.retrieved.items[0].metadata["skill_key"] == "craft_planks"
    assert packet_out.retrieved.scores[0]["signals"]["key"] == 1.0

    packet_out, _ = MixedSkillRetrieval(top_k=1).run(
        Packet(query=Query(text="need a repair move", metadata={"skill_key": "repair_shield"})),
        store,
    )
    assert packet_out.retrieved.items[0].metadata["skill_key"] == "repair_shield"
    assert packet_out.retrieved.trace["used_recency_fallback"] is False


def test_voyager_pipeline_round_trip_upserts_and_reads_out_skill_cards() -> None:
    store = _skill_store()
    pipeline = MemoryPipeline(
        store=store,
        unit_formation=SkillExtractor(),
        representation=CodeWithDescriptionRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
        organization=UpsertByKeySkillLibrary(target_layer="skill_library"),
        retrieval=MixedSkillRetrieval(top_k=2, layer="skill_library"),
        readout=ConcatenateReadout(separator="\n\n---\n\n"),
    )

    pipeline.ingest(
        Observation(
            text=(
                "Skill: craft_planks\n"
                "Description: Turns logs into planks at a bench.\n\n"
                "```python\n"
                "def craft_planks(logs):\n"
                "    return [log[:4] for log in logs]\n"
                "```"
            ),
            source="skill",
        )
    )
    pipeline.ingest(
        Observation(
            text=(
                "Skill: craft_planks\n"
                "Description: Turns logs into planks faster at a bench.\n\n"
                "```python\n"
                "def craft_planks(logs):\n"
                "    return [log.strip()[:4] for log in logs]\n"
                "```"
            ),
            source="skill_update",
        )
    )

    readout = pipeline.recall(Query(text="craft planks"))

    assert store.count("skill_library") == 1
    assert "faster at a bench" in readout.text
    assert readout.source_ids == ["rec-1"]
