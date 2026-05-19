"""Tests for EmbeddingProvider implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.exceptions import ProviderError
from src.providers.embedding import EmbeddingProvider, VoyageEmbeddingProvider


class TestVoyageEmbeddingProviderShape:
    def test_satisfies_protocol(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        assert isinstance(provider, EmbeddingProvider)

    def test_exposes_name_and_dim(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        assert provider.name == "voyage:voyage-3"
        assert provider.embedding_dim == 1024

    def test_rejects_unknown_model(self):
        with pytest.raises(ProviderError) as exc_info:
            VoyageEmbeddingProvider(api_key="test-key", model="voyage-99-imaginary")
        assert "Unknown Voyage model" in str(exc_info.value)


class TestVoyageEmbeddingProviderQuery:
    @pytest.mark.asyncio
    async def test_embed_query_calls_sdk_with_query_input_type(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        fake_result = MagicMock()
        fake_result.embeddings = [[0.1] * 1024]
        provider._client.embed = AsyncMock(return_value=fake_result)

        result = await provider.embed_query("hello")

        assert result == [0.1] * 1024
        provider._client.embed.assert_awaited_once_with(
            ["hello"], model="voyage-3", input_type="query"
        )

    @pytest.mark.asyncio
    async def test_embed_query_wraps_sdk_error_as_provider_error(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        provider._client.embed = AsyncMock(side_effect=RuntimeError("rate limited"))

        with pytest.raises(ProviderError) as exc_info:
            await provider.embed_query("hello")

        assert "rate limited" in str(exc_info.value)


class TestVoyageEmbeddingProviderDocuments:
    @pytest.mark.asyncio
    async def test_embed_documents_uses_document_input_type(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        fake_result = MagicMock()
        fake_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        provider._client.embed = AsyncMock(return_value=fake_result)

        result = await provider.embed_documents(["a", "b"])

        assert len(result) == 2
        call = provider._client.embed.call_args
        assert call.kwargs["input_type"] == "document"

    @pytest.mark.asyncio
    async def test_embed_documents_empty_list_returns_empty(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        provider._client.embed = AsyncMock()

        result = await provider.embed_documents([])

        assert result == []
        provider._client.embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_documents_batches_over_limit(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3", batch_size=2)

        def make_result(n: int) -> MagicMock:
            r = MagicMock()
            r.embeddings = [[0.0] * 1024 for _ in range(n)]
            return r

        provider._client.embed = AsyncMock(
            side_effect=[make_result(2), make_result(2), make_result(1)]
        )

        result = await provider.embed_documents(["a", "b", "c", "d", "e"])

        assert len(result) == 5
        assert provider._client.embed.await_count == 3
