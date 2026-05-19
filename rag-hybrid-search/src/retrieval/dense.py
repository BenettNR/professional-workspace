"""Dense retrieval via ChromaDB vector similarity search."""
from __future__ import annotations

import structlog

from src.exceptions import RetrievalError
from src.models import DocumentChunk, DocumentMetadata, RetrievedChunk

log = structlog.get_logger(__name__)


class DenseRetriever:
    """Queries ChromaDB and returns top-k chunks ranked by cosine similarity."""

    def __init__(self, collection) -> None:  # chromadb.Collection
        self._collection = collection

    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        count = self._collection.count()
        if count == 0:
            log.warning("dense_retriever_empty_collection")
            return []

        n_results = min(top_k, count)
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(f"ChromaDB query failed: {exc}") from exc

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # Convert L2 distance to approximate cosine similarity
            cosine_sim = max(0.0, 1.0 - (dist ** 2) / 2.0)
            chunk = DocumentChunk(
                id=chunk_id,
                content=doc,
                metadata=DocumentMetadata.from_chroma_dict(meta),
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_rank=rank,
                    dense_score=cosine_sim,
                )
            )

        log.debug("dense_retrieval", results=len(retrieved))
        return retrieved
