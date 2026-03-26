"""Voyager (Wang et al., 2023) - motif sketch.

From the repo root (recommended)::

    python -m memprimitive.example.classics.voyager

Or from this directory (script adds the repo root to ``sys.path``)::

    python voyager.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import AlwaysWriteTrigger, ConcatenateReadout
from memprimitive.classic_modules.voyager import (
    CodeWithDescriptionRepresentation,
    MixedSkillRetrieval,
    SkillExtractor,
    UpsertByKeySkillLibrary,
)


def main() -> None:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(
                name="skill_library",
                theme="skill",
                indices=("keyword", "tag", "vector"),
            ),
        ]
    )
    store = MemoryStore(topology=topology)

    pipeline = MemoryPipeline(
        store=store,
        unit_formation=SkillExtractor(),
        representation=CodeWithDescriptionRepresentation(),
        write_trigger=AlwaysWriteTrigger(),
        organization=UpsertByKeySkillLibrary(target_layer="skill_library"),
        retrieval=MixedSkillRetrieval(top_k=3, layer="skill_library"),
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

    print(readout.text)
    print("source record ids:", readout.source_ids)
    print("stored skill count:", store.count("skill_library"))


if __name__ == "__main__":
    main()
