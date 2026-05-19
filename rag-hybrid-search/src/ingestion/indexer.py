"""Orchestrates the full ingestion pipeline.

Flow for each document:
  load → chunk → embed → deduplicate → upsert ChromaDB → rebuild BM25
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi

from src.exceptions import IndexingError
from src.models import ChunkStrategy, DocumentChunk, DocumentMetadata, IngestedDocument
from src.ingestion.chunkers import get_chunker
from src.ingestion.deduplication import DuplicateDetector
from src.ingestion.embedder import Embedder
from src.ingestion.loaders import DocumentLoaderRegistry

log = structlog.get_logger(__name__)


def _chunk_id(source_file: str, chunk_index: int) -> str:
    """Stable, deterministic ID for a chunk."""
    raw = f"{source_file}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class BM25State:
    """Persisted BM25 index state."""

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        chunk_ids: list[str],
        bm25: BM25Okapi,
    ) -> None:
        self.tokenized_corpus = tokenized_corpus
        self.chunk_ids = chunk_ids
        self.bm25 = bm25

    @classmethod
    def empty(cls) -> BM25State:
        bm25 = BM25Okapi([[]])
        return cls([], [], bm25)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tokenized_corpus": self.tokenized_corpus,
                    "chunk_ids": self.chunk_ids,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> BM25State:
        with open(path, "rb") as f:
            data = pickle.load(f)
        corpus = data["tokenized_corpus"]
        ids = data["chunk_ids"]
        bm25 = BM25Okapi(corpus) if corpus else BM25Okapi([[]])
        return cls(corpus, ids, bm25)


class DocumentIndexer:
    """Ingests documents into both ChromaDB (dense) and BM25 (sparse) indexes."""

    def __init__(
        self,
        embedder: Embedder,
        collection,            # chromadb.Collection
        bm25_index_path: Path,
        dedup_threshold: float = 0.95,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        semantic_breakpoint_threshold: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._collection = collection
        self._bm25_path = bm25_index_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._semantic_threshold = semantic_breakpoint_threshold
        self._loader = DocumentLoaderRegistry()
        self._deduplicator = DuplicateDetector(collection, threshold=dedup_threshold)

    def _load_bm25_state(self) -> BM25State:
        if self._bm25_path.exists():
            return BM25State.load(self._bm25_path)
        return BM25State.empty()

    def _rebuild_bm25(
        self,
        existing: BM25State,
        new_chunks: list[DocumentChunk],
        new_embeddings: list[list[float]],
    ) -> BM25State:
        new_tokens = [chunk.content.lower().split() for chunk in new_chunks]
        new_ids = [chunk.id for chunk in new_chunks]

        combined_corpus = existing.tokenized_corpus + new_tokens
        combined_ids = existing.chunk_ids + new_ids

        if not combined_corpus:
            return BM25State.empty()

        bm25 = BM25Okapi(combined_corpus)
        state = BM25State(combined_corpus, combined_ids, bm25)
        state.save(self._bm25_path)
        return state

    async def ingest_file(
        self,
        file_path: Path,
        strategy: ChunkStrategy | None = None,
    ) -> IngestedDocument:
        strategy = strategy or ChunkStrategy.RECURSIVE
        log.info("ingesting_file", path=str(file_path), strategy=strategy.value)

        content, extra_meta = self._loader.load(file_path)
        chunker = get_chunker(strategy, embedder=self._embedder)

        raw_chunks = await chunker.chunk(
            content,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            breakpoint_threshold=self._semantic_threshold,
        )

        chunks = [
            DocumentChunk(
                id=_chunk_id(str(file_path), i),
                content=text,
                metadata=DocumentMetadata(
                    source_file=str(file_path),
                    filename=file_path.name,
                    chunk_index=i,
                    total_chunks=len(raw_chunks),
                    chunking_strategy=strategy,
                    char_count=len(text),
                    section_heading=extra_meta.get("section_heading"),
                    page_number=extra_meta.get("page_number"),
                ),
            )
            for i, text in enumerate(raw_chunks)
        ]

        embeddings = await self._embedder.embed_texts([c.content for c in chunks])

        unique_chunks, unique_embeddings = await self._deduplicator.filter_duplicates(
            chunks, embeddings
        )
        skipped = len(chunks) - len(unique_chunks)

        if unique_chunks:
            try:
                self._collection.upsert(
                    ids=[c.id for c in unique_chunks],
                    embeddings=unique_embeddings,
                    documents=[c.content for c in unique_chunks],
                    metadatas=[c.metadata.to_chroma_dict() for c in unique_chunks],
                )
            except Exception as exc:
                raise IndexingError(f"ChromaDB upsert failed: {exc}") from exc

            existing_state = self._load_bm25_state()
            self._rebuild_bm25(existing_state, unique_chunks, unique_embeddings)

        log.info(
            "ingestion_complete",
            filename=file_path.name,
            chunks=len(chunks),
            indexed=len(unique_chunks),
            skipped=skipped,
        )
        return IngestedDocument(
            filename=file_path.name,
            raw_path=str(file_path),
            chunks=unique_chunks,
            strategy=strategy,
            skipped_duplicates=skipped,
        )

    async def ingest_directory(
        self,
        directory: Path,
        strategy: ChunkStrategy | None = None,
        recursive: bool = True,
    ) -> list[IngestedDocument]:
        extensions = self._loader.supported_extensions()
        pattern = "**/*" if recursive else "*"
        files = [
            p
            for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in extensions
        ]
        log.info("ingesting_directory", path=str(directory), files=len(files))
        results = []
        for file_path in sorted(files):
            result = await self.ingest_file(file_path, strategy)
            results.append(result)
        return results

    def list_indexed_documents(self) -> list[dict]:
        """Return unique documents currently in the ChromaDB collection."""
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        seen: dict[str, dict] = {}
        for meta in metadatas:
            fname = meta.get("filename", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "source_file": meta.get("source_file", ""),
                    "chunking_strategy": meta.get("chunking_strategy", ""),
                    "total_chunks": meta.get("total_chunks", 0),
                }
        return list(seen.values())
