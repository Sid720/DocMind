"""Offline retrieval metrics for reproducible RAG quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluation:
    recall_at_k: float
    mean_reciprocal_rank: float
    hit_rate: float
    cases: int


def evaluate_rankings(
    expected: list[set[str]], retrieved: list[list[str]]
) -> RetrievalEvaluation:
    """Compute macro recall@k, MRR, and hit rate from chunk-id rankings."""
    if not expected or len(expected) != len(retrieved):
        raise ValueError("Expected and retrieved rankings must have equal non-zero length")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    hits = 0
    for relevant, ranking in zip(expected, retrieved, strict=True):
        if not relevant:
            raise ValueError("Every evaluation case needs at least one relevant chunk")
        matched = relevant.intersection(ranking)
        recalls.append(len(matched) / len(relevant))
        first_rank = next(
            (index for index, chunk_id in enumerate(ranking, 1) if chunk_id in relevant),
            None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        hits += int(bool(matched))
    count = len(expected)
    return RetrievalEvaluation(
        recall_at_k=round(sum(recalls) / count, 4),
        mean_reciprocal_rank=round(sum(reciprocal_ranks) / count, 4),
        hit_rate=round(hits / count, 4),
        cases=count,
    )

