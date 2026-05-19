"""Voyage AI embeddings with async batching and retry logic.

Voyage AI is Anthropic's recommended embedding partner. Using separate
input_type values for documents vs. queries meaningfully improves retrieval
precision on bi-encoder models like voyage-3.
"""
from __future__ import annotations

import asyncio
from typing import Literal

import structlog
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.exceptions import EmbeddingError

log = structlog.get_logger(__name__)

InputType = Literal["document", "query"]


class Embedder:
    """Async wrapper around the Voyage AI embeddings endpoint.

    Batches requests to stay within the 128-doc-per-request limit and
    retries on transient failures.
    """

    def __init__(self, api_key: str, model: str, batch_size: int = 128) -> None:
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model
        self._batch_size = min(batch_size, 128)  # Voyage AI hard limit

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _embed_batch(
        self, texts: list[str], input_type: InputType
    ) -> list[list[float]]:
        try:
            result = await self._client.embed(
                texts,
                model=self._model,
                input_type=input_type,
            )
            return result.embeddings
        except Exception as exc:
            raise EmbeddingError(f"Voyage AI embedding call failed: {exc}") from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus documents for indexing (input_type='document')."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            log.debug("embedding_batch", start=i, size=len(batch))
            embeddings = await self._embed_batch(batch, input_type="document")
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (input_type='query' for better retrieval precision)."""
        results = await self._embed_batch([text], input_type="query")
        return results[0]

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper — use only outside of a running event loop."""
        return asyncio.run(self.embed_texts(texts))
