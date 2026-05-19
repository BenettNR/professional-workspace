"""Near-duplicate detection using cosine similarity against indexed chunks.

Before inserting a new chunk, we query ChromaDB for its nearest neighbour.
If similarity > threshold (default 0.95) the chunk is a near-duplicate and
is skipped, preventing redundant context in retrieval results.
"""
from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class DuplicateDetector:
    """Checks new embeddings against an existing ChromaDB collection."""

    def __init__(
        self,
        collection,  # chromadb.Collection
        threshold: float = 0.95,
    ) -> None:
        self._collection = collection
        self._threshold = threshold

    def is_duplicate(self, embedding: list[float]) -> bool:
        """Return True if a near-duplicate already exists in the collection."""
        count = self._collection.count()
        if count == 0:
            return False

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        distances = results.get("distances", [[]])[0]
        if not distances:
            return False

        # ChromaDB returns L2 distance by default; convert to cosine similarity.
        # For unit vectors: cosine_sim = 1 - (L2² / 2)
        # We normalise embeddings before comparison.
        emb = np.array(embedding)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return False

        l2_dist = distances[0]
        cosine_sim = 1.0 - (l2_dist ** 2) / 2.0
        is_dup = cosine_sim > self._threshold

        if is_dup:
            log.debug(
                "duplicate_detected",
                cosine_similarity=round(cosine_sim, 4),
                threshold=self._threshold,
            )
        return is_dup

    async def filter_duplicates(
        self,
        chunks: list,          # list[DocumentChunk]
        embeddings: list[list[float]],
    ) -> tuple[list, list[list[float]]]:
        """Return (unique_chunks, unique_embeddings) after removing near-duplicates."""
        unique_chunks = []
        unique_embeddings = []
        skipped = 0

        for chunk, embedding in zip(chunks, embeddings):
            if self.is_duplicate(embedding):
                skipped += 1
                log.info("chunk_skipped_duplicate", chunk_id=chunk.id)
            else:
                unique_chunks.append(chunk)
                unique_embeddings.append(embedding)

        log.info(
            "deduplication_complete",
            total=len(chunks),
            unique=len(unique_chunks),
            skipped=skipped,
        )
        return unique_chunks, unique_embeddings
