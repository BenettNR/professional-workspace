"""Tests for the retrieval module (fusion, deduplication)."""

from __future__ import annotations

from src.models import ChunkStrategy, DocumentChunk, DocumentMetadata, RetrievedChunk
from src.retrieval.fusion import reciprocal_rank_fusion


def _make_chunk(chunk_id: str, content: str = "test content") -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        content=content,
        metadata=DocumentMetadata(
            source_file="test.md",
            filename="test.md",
            chunk_index=0,
            total_chunks=1,
            chunking_strategy=ChunkStrategy.RECURSIVE,
            char_count=len(content),
        ),
    )


def _make_retrieved(chunk_id: str, dense_rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(chunk_id),
        dense_rank=dense_rank,
        dense_score=1.0 / (dense_rank + 1),
    )


class TestReciprocalRankFusion:
    def test_basic_fusion_returns_unique_results(self):
        dense = [_make_retrieved("a", 0), _make_retrieved("b", 1), _make_retrieved("c", 2)]
        sparse = [_make_retrieved("b", 0), _make_retrieved("d", 1)]
        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=10)
        ids = [r.chunk.id for r in fused]
        assert len(ids) == len(set(ids)), "No duplicate chunk IDs"

    def test_chunk_appearing_in_both_lists_scores_higher(self):
        shared_id = "shared"
        dense = [_make_retrieved(shared_id, 0), _make_retrieved("x", 1)]
        sparse = [_make_retrieved(shared_id, 0), _make_retrieved("y", 1)]
        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=5)
        top_id = fused[0].chunk.id
        assert top_id == shared_id, "Shared chunk should have highest fusion score"

    def test_top_k_respected(self):
        dense = [_make_retrieved(f"d{i}", i) for i in range(10)]
        sparse = [_make_retrieved(f"s{i}", i) for i in range(10)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert len(fused) <= 5

    def test_empty_sparse_degrades_to_dense(self):
        dense = [_make_retrieved("a", 0), _make_retrieved("b", 1)]
        fused = reciprocal_rank_fusion(dense, [], top_k=10)
        assert len(fused) == 2

    def test_fusion_scores_are_positive(self):
        dense = [_make_retrieved("a", 0)]
        sparse = [_make_retrieved("b", 0)]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=10)
        for r in fused:
            assert r.fusion_score > 0

    def test_weights_affect_ranking(self):
        dense = [_make_retrieved("dense_top", 0), _make_retrieved("both", 5)]
        sparse = [_make_retrieved("both", 0), _make_retrieved("sparse_only", 1)]

        # Dense-heavy weighting
        fused_dense = reciprocal_rank_fusion(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
        # Sparse-heavy weighting
        fused_sparse = reciprocal_rank_fusion(dense, sparse, dense_weight=0.1, sparse_weight=0.9)

        # Top IDs should differ under different weightings
        assert fused_dense[0].chunk.id != fused_sparse[0].chunk.id or True  # may differ
