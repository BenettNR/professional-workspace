"""Local sentence-transformers EmbeddingProvider for offline demo mode.

Wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB, CPU-friendly).
The model is lazy-loaded on first use to keep import-time cheap — important
for CI that imports modules to run mypy or lint without ever calling encode().

The synchronous `model.encode()` is wrapped in `asyncio.to_thread` so callers
remain async-compatible without blocking the event loop on CPU-bound work.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class LocalSentenceTransformerEmbeddingProvider:
    """Offline EmbeddingProvider using sentence-transformers on CPU.

    Defaults: all-MiniLM-L6-v2 (384-dim). Suitable for ~10k-chunk corpora;
    not as accurate as voyage-3 but requires zero API keys and runs locally.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, embedding_dim: int = _DEFAULT_DIM) -> None:
        self.name = f"local:{model_name}"
        self.embedding_dim = embedding_dim
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading_local_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        def _run() -> list[list[float]]:
            model = self._load_model()
            arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [vec.tolist() for vec in arr]

        return await asyncio.to_thread(_run)

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self._encode([text])
        return embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._encode(texts)
