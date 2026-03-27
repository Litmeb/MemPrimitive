"""Demo: a partition-ready TiM-style unit opens local-maintenance evolution.

From the repo root (recommended)::

    python -m memprimitive.example.demonstration.partition_ready_local_maintenance

Or from this directory::

    python partition_ready_local_maintenance.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryStore, MemoryUnit, Packet, Placement
from memprimitive.baselines import NewWriteEvolutionTrigger


def main() -> None:
    packet = Packet(
        units=[
            MemoryUnit(
                text="Alice prefers jasmine tea.",
                unit_id="unit-tim-ready",
                unit_type="tim_thought",
                metadata={"tim": {"group_id": "alice-profile", "write": True}},
            )
        ],
        placements=[Placement(unit_id="unit-tim-ready", target_layer="thought_memory")],
    )

    packet_out, _ = NewWriteEvolutionTrigger().run(packet, MemoryStore())

    print("evolution_decisions:", packet_out.evolution_decisions)
    print("evolution_trigger trace:")
    pprint(packet_out.trace["evolution_trigger"])


if __name__ == "__main__":
    main()
