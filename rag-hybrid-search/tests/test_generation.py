"""Tests for generation utilities (citation parsing, confidence scoring)."""
from __future__ import annotations

import pytest

from src.generation.citation_verifier import _extract_claim_sentences, _parse_citation_numbers
from src.generation.confidence import _citation_coverage, _retrieval_confidence
from src.models import Citation, ChunkStrategy, DocumentChunk, DocumentMetadata, RetrievedChunk


# ── Citation parsing ──────────────────────────────────────────────────────────

class TestCitationParsing:
    def test_parse_single_citation(self):
        answer = "The rate limit is 1,000 req/min [1]."
        assert _parse_citation_numbers(answer) == [1]

    def test_parse_multiple_citations(self):
        answer = "See [1] and [3] for details. Also check [2]."
        assert _parse_citation_numbers(answer) == [1, 2, 3]

    def test_parse_no_citations(self):
        answer = "There are no citations in this answer."
        assert _parse_citation_numbers(answer) == []

    def test_parse_duplicate_citations_deduplicated(self):
        answer = "Mentioned in [1] and again [1]."
        assert _parse_citation_numbers(answer) == [1]

    def test_extract_claim_sentence(self):
        answer = "The sky is blue. The rate limit is 1,000 req/min [1]. This is good."
        claim = _extract_claim_sentences(answer, 1)
        assert "[1]" in claim
        assert "rate limit" in claim

    def test_extract_claim_no_match_returns_context(self):
        answer = "No citation here."
        claim = _extract_claim_sentences(answer, 99)
        # Should return empty string when citation not found
        assert claim == ""


# ── Confidence scoring ────────────────────────────────────────────────────────

class TestConfidenceScoring:
    def _make_chunk(self) -> DocumentChunk:
        return DocumentChunk(
            id="test",
            content="content",
            metadata=DocumentMetadata(
                source_file="test.md",
                filename="test.md",
                chunk_index=0,
                total_chunks=1,
                chunking_strategy=ChunkStrategy.RECURSIVE,
                char_count=7,
            ),
        )

    def test_citation_coverage_all_verified(self):
        citations = [
            Citation(1, "c1", "f.md", "text", verified=True),
            Citation(2, "c2", "f.md", "text", verified=True),
        ]
        assert _citation_coverage(citations) == 1.0

    def test_citation_coverage_none_verified(self):
        citations = [
            Citation(1, "c1", "f.md", "text", verified=False),
        ]
        assert _citation_coverage(citations) == 0.0

    def test_citation_coverage_empty_gives_full_score(self):
        assert _citation_coverage([]) == 1.0

    def test_citation_coverage_partial(self):
        citations = [
            Citation(1, "c1", "f.md", "text", verified=True),
            Citation(2, "c2", "f.md", "text", verified=False),
        ]
        assert _citation_coverage(citations) == 0.5

    def test_retrieval_confidence_empty(self):
        assert _retrieval_confidence([]) == 0.0

    def test_retrieval_confidence_uses_rerank_score(self):
        chunk = self._make_chunk()
        rc = RetrievedChunk(chunk=chunk, rerank_score=0.0)  # sigmoid(0) = 0.5
        conf = _retrieval_confidence([rc])
        assert 0.49 < conf < 0.51

    def test_retrieval_confidence_high_rerank_score(self):
        chunk = self._make_chunk()
        rc = RetrievedChunk(chunk=chunk, rerank_score=10.0)  # sigmoid(10) ≈ 1.0
        conf = _retrieval_confidence([rc])
        assert conf > 0.99

    def test_retrieval_confidence_falls_back_to_fusion(self):
        chunk = self._make_chunk()
        rc = RetrievedChunk(chunk=chunk, fusion_score=0.05)
        conf = _retrieval_confidence([rc])
        assert conf > 0
