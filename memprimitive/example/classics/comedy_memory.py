"""COMEDY-style memory module reconstructed with baseline MemPrimitive modules.

This example intentionally mirrors the paper-style pattern requested by the
project notes:

1. Raw dialogue turns are appended into an ``episodic`` layer.
2. A session-close observation uses ``HierarchicalOrganization`` to extract one
   session-level memory from the just-finished session.
3. Every ``compress_every_n`` sessions, a maintenance observation uses
   ``PeriodicMaintenanceTrigger`` wrapped around ``StoreAllTrigger`` to select
   all existing session memories and ``HierarchicalEvolution`` to compress them
   into one new ``compressive_memory`` record.
4. Current reply generation can then read only the newest compressive memory via
   ``RecencyRetrieval(top_k=1)`` on the ``compressive_memory`` layer.

Repeated compressive writes are intentional here because the original paper
pattern rewrites a fresh compressed memory snapshot at each maintenance step.

The current runtime always executes the ``memory_evolution`` slot once the
pipeline reaches it, so this example submits maintenance observations only on
actual maintenance ticks. That keeps ``HierarchicalEvolution`` from falling back
to a full source-layer scan on non-maintenance turns while still using the
requested in-pipeline ``PeriodicMaintenanceTrigger(StoreAllTrigger(...))``
composition when compression does run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from memprimitive import MemoryPipeline, MemoryStore, Observation, Query, StoreLayerSpec, StoreTopology
from memprimitive.baselines import (
    AppendOrganization,
    BasicRepresentation,
    BoundaryEventTrigger,
    ConcatenateReadout,
    HierarchicalEvolution,
    HierarchicalOrganization,
    NeverTrigger,
    PeriodicMaintenanceTrigger,
    PlacementWithoutAppendOrganization,
    RecencyRetrieval,
    StoreAllTrigger,
)


def build_comedy_memory_system(*, compress_every_n: int = 2) -> dict[str, object]:
    topology = StoreTopology.from_layers(
        [
            StoreLayerSpec(name="episodic", theme="session_memory", indices=("temporal", "vector")),
            StoreLayerSpec(name="session_memory", theme="semantic", indices=("temporal", "vector")),
            StoreLayerSpec(name="compressive_memory", theme="semantic", indices=("temporal", "vector")),
        ]
    )
    store = MemoryStore(topology=topology)

    turn_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        organization=AppendOrganization(target_layer="episodic"),
        store=store,
    )

    session_memory_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=BoundaryEventTrigger(
            slot="write_trigger",
            accepted_events=("session_end",),
        ),
        organization=HierarchicalOrganization(
            source_layer="episodic",
            extract_mode="generate",
            extract_fields=("summary",),
            group_by=("session_id",),
            target_layer="session_memory",
        ),
        evolution_trigger=NeverTrigger(slot="evolution_trigger"),
        store=store,
    )

    compressive_memory_pipeline = MemoryPipeline(
        representation=BasicRepresentation(elements=("text",)),
        write_trigger=NeverTrigger(slot="write_trigger"),
        organization=PlacementWithoutAppendOrganization(target_layer="compressive_memory"),
        evolution_trigger=PeriodicMaintenanceTrigger(
            every_n=compress_every_n,
            trigger=StoreAllTrigger(slot="evolution_trigger"),
        ),
        memory_evolution=HierarchicalEvolution(
            source_layer="session_memory",
            extract_mode="generate",
            extract_fields=("summary",),
            target_layer="compressive_memory",
        ),
        store=store,
    )

    reply_memory_pipeline = MemoryPipeline(
        retrieval=RecencyRetrieval(top_k=1, layer="compressive_memory"),
        readout=ConcatenateReadout(),
        store=store,
    )

    return {
        "store": store,
        "turn_pipeline": turn_pipeline,
        "session_memory_pipeline": session_memory_pipeline,
        "compressive_memory_pipeline": compressive_memory_pipeline,
        "reply_memory_pipeline": reply_memory_pipeline,
        "compress_every_n": compress_every_n,
    }


def ingest_session(
    system: dict[str, object],
    *,
    session_index: int,
    session_id: str,
    turns: list[str],
) -> None:
    turn_pipeline = system["turn_pipeline"]
    session_memory_pipeline = system["session_memory_pipeline"]
    compressive_memory_pipeline = system["compressive_memory_pipeline"]
    compress_every_n = int(system["compress_every_n"])

    assert isinstance(turn_pipeline, MemoryPipeline)
    assert isinstance(session_memory_pipeline, MemoryPipeline)
    assert isinstance(compressive_memory_pipeline, MemoryPipeline)

    for turn_index, text in enumerate(turns, start=1):
        turn_pipeline.ingest(
            Observation(
                text=text,
                source="dialogue",
                metadata={
                    "session_id": session_id,
                    "turn_id": f"{session_id}-turn-{turn_index}",
                },
            )
        )

    session_memory_pipeline.ingest(
        Observation(
            text=f"close session {session_id}",
            source="session_controller",
            metadata={
                "session_id": session_id,
                "trigger": {"events": ["session_end"]},
            },
        )
    )

    if session_index % compress_every_n == 0:
        compressive_memory_pipeline.ingest(
            Observation(
                text=f"periodic compressive maintenance after {session_index} sessions",
                source="maintenance_controller",
                metadata={
                    "session_id": session_id,
                    "trigger": {
                        "schedule": {"tick": session_index},
                    },
                },
            )
        )


def build_reply_context(system: dict[str, object], *, user_query: str) -> str:
    reply_memory_pipeline = system["reply_memory_pipeline"]
    assert isinstance(reply_memory_pipeline, MemoryPipeline)
    return reply_memory_pipeline.recall(Query(text=user_query)).text


def main() -> None:
    system = build_comedy_memory_system(compress_every_n=2)
    store = system["store"]
    assert isinstance(store, MemoryStore)

    ingest_session(
        system,
        session_index=1,
        session_id="sess-1",
        turns=[
            "Alice is implementing graph retrieval for long multi-session conversations.",
            "She wants the assistant to remember the graph debugging context in future replies.",
        ],
    )
    ingest_session(
        system,
        session_index=2,
        session_id="sess-2",
        turns=[
            "Alice now compares hierarchical session memory against compressive memory.",
            "She decides the system should periodically compress all session summaries into one new memory.",
        ],
    )
    ingest_session(
        system,
        session_index=3,
        session_id="sess-3",
        turns=[
            "Alice asks for the latest compressed context before the next answer is generated.",
            "The assistant should read only the newest compressive memory record.",
        ],
    )

    print("records per layer:")
    pprint({name: store.count(name) for name in store.topology.layer_names})
    print()

    print("session-level memories:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "group_key": record.metadata.get("hierarchical", {}).get("group_key"),
                "source_record_ids": record.metadata.get("hierarchical", {}).get("source_record_ids"),
            }
            for record in store.iter_records("session_memory")
        ]
    )
    print()

    print("compressive memories:")
    pprint(
        [
            {
                "record_id": record.record_id,
                "text": record.text,
                "source_record_ids": record.metadata.get("hierarchical", {}).get("source_record_ids"),
            }
            for record in store.iter_records("compressive_memory")
        ]
    )
    print()

    print("latest compressive memory for reply generation:")
    print(build_reply_context(system, user_query="What should the assistant remember right now?"))


if __name__ == "__main__":
    main()
