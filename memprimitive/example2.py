"""End-to-end example showing the simplest trigger-family composition style.

From the repo root (recommended)::

    python -m memprimitive.example2

Or from this directory (script adds the repo root to ``sys.path``)::

    python example2.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Running as ``python memprimitive/example2.py`` leaves ``__package__`` unset; repo root must be on path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memprimitive import MemoryPipeline, Observation, Query
from memprimitive.baselines import (
    AppendOnlyEvolution,
    AppendOrganization,
    BasicRepresentation,
    ConcatenateReadout,
    PassThroughUnitFormation,
    RecencyRetrieval,
)
from memprimitive.baselines._trigger_family import (
    AlwaysOpenGate,
    BooleanGatePolicy,
    ConstantSignal,
    ThresholdPolicy,
    WeightedSumScorer,
)
from memprimitive.baselines.evolution_trigger import compose_evolution_trigger
from memprimitive.baselines.write_trigger import compose_write_trigger


def main() -> None:
    # The pipeline shape stays the same; only the trigger slots are composed inline.
    pipeline = MemoryPipeline(
        unit_formation=PassThroughUnitFormation(),
        representation=BasicRepresentation(),
        write_trigger=compose_write_trigger(
            name="demo_threshold_write_trigger",
            signal_providers=(ConstantSignal(signal_name="importance_hint", value=0.8),),
            scorer=WeightedSumScorer(weights={"importance_hint": 1.0}),
            gate=AlwaysOpenGate(),
            policy=ThresholdPolicy(threshold=0.5),
        ),
        organization=AppendOrganization(),
        evolution_trigger=compose_evolution_trigger(
            name="demo_boolean_evolution_trigger",
            signal_providers=(ConstantSignal(signal_name="after_write_ready", value=1.0),),
            scorer=WeightedSumScorer(weights={"after_write_ready": 1.0}),
            gate=AlwaysOpenGate(),
            policy=BooleanGatePolicy(),
        ),
        memory_evolution=AppendOnlyEvolution(),
        retrieval=RecencyRetrieval(top_k=2),
        readout=ConcatenateReadout(),
    )

    packet = pipeline.ingest(Observation(text="The user is exploring compositional triggers.", source="notes"))

    print("write_trigger trace:")
    pprint(packet.trace["write_trigger"])
    print()

    print("evolution_trigger trace (extra evolution remains disabled here):")
    pprint(packet.trace["evolution_trigger"])
    print()

    readout = pipeline.recall(Query(text="What is the user exploring?"))

    print("readout text:", readout.text)
    print("source record ids:", readout.source_ids)


if __name__ == "__main__":
    main()
