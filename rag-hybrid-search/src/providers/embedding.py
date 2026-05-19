"""EmbeddingProvider Protocol and concrete implementations.

The Protocol uses structural typing so consumers depend on shape, not on a
particular base class. `@runtime_checkable` allows isinstance() in tests.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.exceptions import ProviderError

log = structlog.get_logger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async interface for producing query and document embeddings.

    Implementations MUST:
      - Expose a stable `name` (used for Chroma collection naming and logs).
      - Expose `embedding_dim` so consumers can validate index compatibility.
      - Treat `embed_query` and `embed_documents` as semantically distinct —
        bi-encoders like Voyage benefit from asymmetric encoding.
    """

    name: str
    embedding_dim: int

    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


_VOYAGE_DIMS: dict[str, int] = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
}


class VoyageEmbeddingProvider:
    """Voyage AI implementation of EmbeddingProvider.

    Uses asymmetric input_type ('document' vs 'query') for better retrieval
    precision on bi-encoder models. Batches document calls to stay within
    Voyage's 128-doc-per-request limit. Retries transient failures with
    exponential backoff.
    """

    def __init__(self, api_key: str, model: str, batch_size: int = 128) -> None:
        if model not in _VOYAGE_DIMS:
            raise ProviderError(
                f"Unknown Voyage model '{model}'. "
                f"Known models: {sorted(_VOYAGE_DIMS)}"
            )
        self._client = voyageai.AsyncClient(api_key=api_key)  # type: ignore[attr-defined]
        self._model = model
        self._batch_size = min(batch_size, 128)
        self.name = f"voyage:{model}"
        self.embedding_dim = _VOYAGE_DIMS[model]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            result = await self._client.embed(
                texts, model=self._model, input_type=input_type
            )
        except Exception as exc:
            raise ProviderError(f"Voyage embedding call failed: {exc}") from exc
        # Voyage's SDK return type is loosely typed; floats are what voyage-3 returns.
        return [list(map(float, vec)) for vec in result.embeddings]

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self._embed([text], input_type="query")
        return embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            log.debug("embedding_batch", start=i, size=len(batch))
            embeddings = await self._embed(batch, input_type="document")
            all_embeddings.extend(embeddings)
        return all_embeddings
