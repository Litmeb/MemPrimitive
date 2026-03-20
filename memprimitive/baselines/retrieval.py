"""Baseline: retrieval primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from ..core import MemoryStore, ModuleSpec, Packet, RetrievedSet
from ..interfaces import RetrievalModule

from ._trace import copy_trace


class RecencyRetrieval(RetrievalModule):
    """Retrieve up to ``top_k`` records: keyword filter when possible, else by recency.

    Constructor: ``top_k`` must be a positive integer. ``layer`` selects
    ``store.iter_records(layer)``; ``None`` means all layers (order follows
    ``MemoryStore.iter_records``).

    ``run`` requires ``packet.query``. Tokenizes query text on whitespace; if any
    token matches substring-wise in a record, keeps only matches (order: newest
    first among candidates); otherwise takes the ``top_k`` newest records overall.
    Does not mutate the store. Populates ``packet.retrieved`` and score dicts
    (rank/strategy, not dense similarity scores).
    """

    spec = ModuleSpec(
        name="recency_retrieval",
        slot="retrieval",
        input_requirements=("query.text",),
        output_guarantees=("retrieved.items", "retrieved.scores"),
    )

    def __init__(self, top_k: int = 3, layer: str | None = None) -> None:
        if top_k <= 0:
            raise ValueError("RecencyRetrieval requires top_k > 0.")
        self.top_k = top_k
        self.layer = layer

    def run(self, packet: Packet, store: MemoryStore) -> tuple[Packet, MemoryStore]:
        if packet.query is None:
            raise ValueError("RecencyRetrieval requires packet.query.")

        all_records = store.iter_records(self.layer)
        query_tokens = {token for token in packet.query.text.casefold().split() if token}

        ordered = list(reversed(all_records))
        filtered_records = ordered
        keyword_mode = False
        if query_tokens:
            matching_records = [
                record
                for record in ordered
                if any(token in record.text.casefold() for token in query_tokens)
            ]
            if matching_records:
                filtered_records = matching_records
                keyword_mode = True

        selected_records = filtered_records[: self.top_k]
        scores = [
            {
                "record_id": record.record_id,
                "rank": rank,
                "strategy": "keyword+recency" if keyword_mode else "recency",
            }
            for rank, record in enumerate(selected_records, start=1)
        ]
        retrieved = RetrievedSet(
            items=selected_records,
            scores=scores,
            trace={
                "module": self.spec.name,
                "top_k": self.top_k,
                "matched_by_keyword": keyword_mode,
                "candidate_count": len(all_records),
            },
        )
        trace = copy_trace(packet)
        trace["retrieval"] = retrieved.trace
        return replace(packet, retrieved=retrieved, trace=trace), store


BASELINE_SLOT: Final[str] = "retrieval"
BASELINE_CLASSES: Final[tuple[type[RetrievalModule], ...]] = (RecencyRetrieval,)
