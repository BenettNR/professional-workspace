"""Sparse retrieval via BM25 keyword matching.

BM25 excels at exact-term matching (function names, config keys, error codes)
that semantic search often misses — precisely why hybrid search outperforms
dense-only RAG on technical documentation.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from src.exceptions import RetrievalError
from src.models import DocumentChunk, DocumentMetadata, RetrievedChunk

log = structlog.get_logger(__name__)


class SparseRetriever:
    """BM25 retrieval over a pickled corpus index."""

    def __init__(self, index_path: Path, collection=None) -> None:
        self._index_path = index_path
        self._collection = collection  # used to hydrate chunk metadata
        self._state: dict | None = None  # lazy loaded

    def _load_state(self) -> dict | None:
        if not self._index_path.exists():
            return None
        if self._state is None:
            with open(self._index_path, "rb") as f:
                self._state = pickle.load(f)
        return self._state

    def invalidate_cache(self) -> None:
        """Force reload on next retrieval (call after re-indexing)."""
        self._state = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        state = self._load_state()
        if state is None:
            log.warning("sparse_retriever_no_index", path=str(self._index_path))
            return []

        tokenized_corpus: list[list[str]] = state["tokenized_corpus"]
        chunk_ids: list[str] = state["chunk_ids"]

        if not tokenized_corpus:
            return []

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = query.lower().split()
            scores: np.ndarray = bm25.get_scores(query_tokens)
        except Exception as exc:
            raise RetrievalError(f"BM25 scoring failed: {exc}") from exc

        # Normalise scores to [0, 1] for fusion
        max_score = scores.max()
        normalised = (scores / max_score) if max_score > 0 else scores

        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved: list[RetrievedChunk] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            cid = chunk_ids[idx]
            chunk = self._hydrate_chunk(cid, tokenized_corpus[idx])
            if chunk is None:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    sparse_rank=rank,
                    sparse_score=float(normalised[idx]),
                )
            )

        log.debug("sparse_retrieval", results=len(retrieved))
        return retrieved

    def _hydrate_chunk(
        self,
        chunk_id: str,
        tokens: list[str],
    ) -> DocumentChunk | None:
        """Reconstruct a DocumentChunk from ChromaDB using the chunk ID."""
        if self._collection is None:
            # Minimal stub when no collection available
            return DocumentChunk(
                id=chunk_id,
                content=" ".join(tokens),
                metadata=DocumentMetadata(
                    source_file="unknown",
                    filename="unknown",
                    chunk_index=0,
                    total_chunks=1,
                    chunking_strategy=__import__(
                        "src.models", fromlist=["ChunkStrategy"]
                    ).ChunkStrategy.RECURSIVE,
                    char_count=len(" ".join(tokens)),
                ),
            )

        try:
            result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            if not docs:
                return None
            return DocumentChunk(
                id=chunk_id,
                content=docs[0],
                metadata=DocumentMetadata.from_chroma_dict(metas[0]),
            )
        except Exception:
            return None
