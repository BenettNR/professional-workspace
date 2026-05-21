"""Three chunking strategies: fixed-size, recursive, and semantic.

All chunkers expose an async `chunk()` method for interface uniformity.
Fixed and Recursive are synchronous internally; Semantic awaits embeddings.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from src.models import ChunkStrategy

if TYPE_CHECKING:
    from src.providers.embedding import EmbeddingProvider


class TextChunker(ABC):
    @property
    @abstractmethod
    def strategy(self) -> ChunkStrategy: ...

    @abstractmethod
    async def chunk(self, text: str, **kwargs: Any) -> list[str]: ...


class FixedSizeChunker(TextChunker):
    """Simple character-count splitting with configurable overlap."""

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.FIXED

    async def chunk(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> list[str]:
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
        )
        return splitter.split_text(text) or [text]


class RecursiveChunker(TextChunker):
    """Structure-aware splitting that respects markdown heading hierarchy."""

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.RECURSIVE

    async def chunk(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> list[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text) or [text]


class SemanticChunker(TextChunker):
    """Splits at topic boundaries detected via embedding cosine-similarity drops.

    Requires an Embedder to compute sentence-window embeddings. Expensive —
    use when retrieval quality matters more than indexing speed.
    """

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.SEMANTIC

    async def chunk(
        self,
        text: str,
        breakpoint_threshold: float = 0.3,
        **kwargs: Any,
    ) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 2:
            return [text]

        window_size = 3
        windows = []
        for i in range(len(sentences)):
            start = max(0, i - window_size // 2)
            end = min(len(sentences), i + window_size // 2 + 1)
            windows.append(" ".join(sentences[start:end]))

        embeddings = await self._embedder.embed_documents(windows)
        emb_array = np.array(embeddings)

        # Cosine similarity between adjacent windows
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = emb_array / norms

        breakpoints = [0]
        for i in range(1, len(normed)):
            similarity = float(np.dot(normed[i - 1], normed[i]))
            if similarity < (1.0 - breakpoint_threshold):
                breakpoints.append(i)
        breakpoints.append(len(sentences))

        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk_text = " ".join(sentences[breakpoints[i] : breakpoints[i + 1]]).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks or [text]


def get_chunker(strategy: ChunkStrategy, embedder: EmbeddingProvider | None = None) -> TextChunker:
    """Factory: return the appropriate chunker for the given strategy."""
    if strategy == ChunkStrategy.FIXED:
        return FixedSizeChunker()
    if strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker()
    if strategy == ChunkStrategy.SEMANTIC:
        if embedder is None:
            raise ValueError("SemanticChunker requires an EmbeddingProvider instance")
        return SemanticChunker(embedder)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
