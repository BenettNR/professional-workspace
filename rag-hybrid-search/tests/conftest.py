"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.models import ChunkStrategy, DocumentChunk, DocumentMetadata, RetrievedChunk


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        source_file="data/raw/nexus-api-docs/rate-limiting.md",
        filename="rate-limiting.md",
        chunk_index=0,
        total_chunks=3,
        chunking_strategy=ChunkStrategy.RECURSIVE,
        char_count=500,
        section_heading="Rate Limiting",
    )


@pytest.fixture
def sample_chunk(sample_metadata) -> DocumentChunk:
    return DocumentChunk(
        id="abc123",
        content="Standard tier allows 1,000 requests per minute and 50,000 publish events per minute.",
        metadata=sample_metadata,
    )


@pytest.fixture
def sample_retrieved(sample_chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=sample_chunk,
        dense_rank=0,
        dense_score=0.92,
        sparse_rank=1,
        sparse_score=0.85,
        fusion_score=0.0145,
        rerank_score=8.3,
    )
