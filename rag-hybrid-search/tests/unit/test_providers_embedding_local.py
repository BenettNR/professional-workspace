"""Tests for LocalSentenceTransformerEmbeddingProvider.

The actual sentence-transformers model is mocked — we don't want unit tests
to download an 80MB model from huggingface every CI run. The integration that
proves the real model works happens at the `make demo` smoke-test layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.providers.embedding import EmbeddingProvider
from src.providers.embedding_local import LocalSentenceTransformerEmbeddingProvider


def _fake_model(dim: int = 384):
    """Return a stand-in for sentence_transformers.SentenceTransformer."""

    def encode(texts: list[str], **kwargs) -> np.ndarray:
        return np.array([[0.01 * (i + 1)] * dim for i, _ in enumerate(texts)])

    model = MagicMock()
    model.encode.side_effect = encode
    return model


class TestLocalEmbeddingProviderShape:
    def test_satisfies_protocol(self):
        provider = LocalSentenceTransformerEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_default_name_and_dim(self):
        provider = LocalSentenceTransformerEmbeddingProvider()
        assert provider.name == "local:sentence-transformers/all-MiniLM-L6-v2"
        assert provider.embedding_dim == 384

    def test_custom_model_name(self):
        provider = LocalSentenceTransformerEmbeddingProvider(
            model_name="sentence-transformers/all-mpnet-base-v2", embedding_dim=768
        )
        assert provider.name == "local:sentence-transformers/all-mpnet-base-v2"
        assert provider.embedding_dim == 768


class TestLocalEmbeddingProviderLazyLoad:
    def test_model_not_loaded_at_construction(self):
        with patch(
            "src.providers.embedding_local.LocalSentenceTransformerEmbeddingProvider._load_model"
        ) as load:
            LocalSentenceTransformerEmbeddingProvider()
            load.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_loaded_exactly_once_across_calls(self):
        provider = LocalSentenceTransformerEmbeddingProvider()
        provider._model = _fake_model()  # pretend already loaded

        # If we call _load_model with model already set, no construction happens.
        with patch("sentence_transformers.SentenceTransformer") as st_constructor:
            await provider.embed_query("test1")
            await provider.embed_documents(["test2", "test3"])
            st_constructor.assert_not_called()


class TestLocalEmbeddingProviderEncoding:
    @pytest.mark.asyncio
    async def test_embed_query_returns_single_vector(self):
        provider = LocalSentenceTransformerEmbeddingProvider()
        provider._model = _fake_model()

        result = await provider.embed_query("hello")

        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_documents_returns_vector_per_text(self):
        provider = LocalSentenceTransformerEmbeddingProvider()
        provider._model = _fake_model()

        result = await provider.embed_documents(["a", "b", "c"])

        assert len(result) == 3
        assert all(len(vec) == 384 for vec in result)

    @pytest.mark.asyncio
    async def test_embed_documents_empty_list_returns_empty(self):
        provider = LocalSentenceTransformerEmbeddingProvider()
        provider._model = _fake_model()

        result = await provider.embed_documents([])

        assert result == []
