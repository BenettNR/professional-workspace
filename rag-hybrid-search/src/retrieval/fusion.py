"""Reciprocal Rank Fusion (RRF) for hybrid dense + sparse retrieval.

RRF formula: score(d) = Σ_r 1 / (k + rank_r(d))
where k=60 is the standard smoothing constant and r ranges over result lists.

The fused list is then passed to the cross-encoder reranker for a second-pass
precision boost.
"""
from __future__ import annotations

import structlog

from src.models import RetrievedChunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.reranker import Reranker

log = structlog.get_logger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Merge dense and sparse result lists via weighted RRF.

    Returns the top_k unique chunks ordered by descending fusion score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, result in enumerate(dense_results):
        cid = result.chunk.id
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (rrf_k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = result

    for rank, result in enumerate(sparse_results):
        cid = result.chunk.id
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (rrf_k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = result
        else:
            # Merge rank/score info into existing entry
            chunk_map[cid].sparse_rank = result.sparse_rank
            chunk_map[cid].sparse_score = result.sparse_score

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused = []
    for cid in sorted_ids:
        rc = chunk_map[cid]
        rc.fusion_score = scores[cid]
        fused.append(rc)

    log.debug(
        "rrf_fusion",
        dense_candidates=len(dense_results),
        sparse_candidates=len(sparse_results),
        fused=len(fused),
    )
    return fused


class HybridRetriever:
    """Orchestrates dense → sparse → RRF fusion → reranking."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        fusion_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._fusion_top_k = fusion_top_k
        self._rerank_top_k = rerank_top_k

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        dense_only: bool = False,
    ) -> list[RetrievedChunk]:
        """Run full hybrid retrieval pipeline.

        Args:
            query: Raw query text (used by BM25 and reranker).
            query_embedding: Pre-computed query embedding (used by ChromaDB).
            dense_only: Skip sparse retrieval (for ablation comparison).
        """
        dense_results = await self._dense.retrieve(query_embedding, top_k=self._dense_top_k)

        if dense_only or not dense_results:
            candidates = dense_results[: self._fusion_top_k]
        else:
            sparse_results = await self._sparse.retrieve(query, top_k=self._sparse_top_k)
            candidates = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                rrf_k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                top_k=self._fusion_top_k,
            )

        reranked = await self._reranker.rerank(query, candidates, top_k=self._rerank_top_k)
        return reranked
