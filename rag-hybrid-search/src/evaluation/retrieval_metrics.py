"""Pure-function retrieval quality metrics.

Computed from the ranked list of retrieved chunks and the golden
`expected_sources` list. No LLM calls — these are exact, deterministic,
and cheap. Use them in CI; reserve the LLM-judge metrics for offline runs.

Source matching uses filename basenames (e.g. 'rate-limiting.md') because
the golden dataset records expected_sources as filenames, not full paths.
"""
from __future__ import annotations

from collections.abc import Sequence


def _normalize(name: str) -> str:
    """Strip directory components so 'data/raw/x/rate-limiting.md' matches 'rate-limiting.md'."""
    return name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip().lower()


def hit_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """1.0 if any expected source appears in the top-k retrieved, else 0.0.

    Returns 0.0 when expected_sources is empty (the question is unanswerable
    — no retrieval result should be considered a 'hit').
    """
    if not expected_sources:
        return 0.0
    expected = {_normalize(s) for s in expected_sources}
    top_k = [_normalize(s) for s in retrieved_sources[:k]]
    return 1.0 if any(s in expected for s in top_k) else 0.0


def mrr_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant document in top-k, else 0.0.

    The first relevant document at rank r contributes 1/r. If no relevant
    document is found in top-k, the score is 0.0.
    """
    if not expected_sources:
        return 0.0
    expected = {_normalize(s) for s in expected_sources}
    for rank, src in enumerate(retrieved_sources[:k], start=1):
        if _normalize(src) in expected:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Fraction of top-k retrieved that are relevant. Useful when a question
    has multiple expected sources and we want to measure recall-ish behavior.
    """
    if not expected_sources or k <= 0:
        return 0.0
    expected = {_normalize(s) for s in expected_sources}
    top_k = [_normalize(s) for s in retrieved_sources[:k]]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if s in expected)
    return hits / min(k, len(top_k))
