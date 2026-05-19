"""Shared domain models used across all pipeline layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class DocumentMetadata:
    source_file: str  # original file path (relative to raw_data_dir)
    filename: str  # basename of the source file
    chunk_index: int  # 0-based position within the document
    total_chunks: int  # total chunks produced from this document
    chunking_strategy: ChunkStrategy
    char_count: int
    section_heading: str | None = None
    page_number: int | None = None  # PDF pages only

    def to_chroma_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict compatible with ChromaDB metadata storage."""
        return {
            "source_file": self.source_file,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunking_strategy": self.chunking_strategy.value,
            "char_count": self.char_count,
            "section_heading": self.section_heading or "",
            "page_number": self.page_number if self.page_number is not None else -1,
        }

    @classmethod
    def from_chroma_dict(cls, data: Mapping[str, Any]) -> DocumentMetadata:
        return cls(
            source_file=data["source_file"],
            filename=data["filename"],
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            chunking_strategy=ChunkStrategy(data["chunking_strategy"]),
            char_count=data["char_count"],
            section_heading=data.get("section_heading") or None,
            page_number=data["page_number"] if data.get("page_number", -1) >= 0 else None,
        )


@dataclass
class DocumentChunk:
    id: str  # stable UUID derived from source + chunk_index
    content: str
    metadata: DocumentMetadata


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class Citation:
    number: int  # as it appears in the answer [1], [2], …
    chunk_id: str
    source_file: str
    text_excerpt: str  # first 200 chars of the supporting chunk
    verified: bool = False
    verification_reason: str = ""


@dataclass
class ConfidenceScore:
    retrieval_confidence: float  # avg normalised relevance of top-k chunks
    citation_coverage: float  # verified_citations / total_citations
    answer_completeness: float  # LLM-as-judge: did answer cover all sub-questions
    composite: float  # weighted average of the three dimensions

    def is_sufficient(self, threshold: float) -> bool:
        return self.composite >= threshold


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: list[Citation]
    confidence: ConfidenceScore
    retrieved_chunks: list[RetrievedChunk]
    insufficient_info: bool = False
    missing_info_message: str | None = None


@dataclass
class IngestedDocument:
    filename: str
    raw_path: str
    chunks: list[DocumentChunk]
    strategy: ChunkStrategy
    skipped_duplicates: int = 0
