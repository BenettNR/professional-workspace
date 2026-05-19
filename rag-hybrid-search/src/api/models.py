"""Pydantic request/response schemas for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, Field

# ── /v1/ask ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    dense_only: bool = Field(
        default=False,
        description="Use dense-only retrieval (for hybrid vs dense ablation)",
    )
    chunk_strategy: str | None = Field(
        default=None,
        description="Override chunking strategy filter (not used at query time; for info only)",
    )


class CitationOut(BaseModel):
    number: int
    source_file: str
    text_excerpt: str
    verified: bool
    verification_reason: str


class ConfidenceOut(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite: float


class ChunkOut(BaseModel):
    id: str
    content: str
    source_file: str
    filename: str
    chunk_index: int
    chunking_strategy: str
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float
    rerank_score: float | None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    confidence: ConfidenceOut
    retrieved_chunks: list[ChunkOut]
    insufficient_info: bool
    missing_info_message: str | None


# ── /v1/documents ─────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename: str
    source_file: str
    chunking_strategy: str
    total_chunks: int


class DocumentsResponse(BaseModel):
    total_documents: int
    documents: list[DocumentInfo]


# ── /v1/ingest ────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    chunks_skipped: int
    strategy: str
    message: str
