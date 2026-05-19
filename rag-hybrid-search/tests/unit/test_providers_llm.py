"""Tests for LLMProvider implementations."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.exceptions import ProviderError
from src.providers.llm import AnthropicLLMProvider, LLMProvider


class TestAnthropicLLMProviderShape:
    def test_satisfies_protocol(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        assert isinstance(provider, LLMProvider)

    def test_exposes_name(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        assert provider.name == "anthropic:claude-sonnet-4-6"


class TestAnthropicLLMProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_text_content(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="Hello, world.")]
        provider._client.messages.create = AsyncMock(return_value=fake_response)

        result = await provider.complete(
            system="be terse", user="say hi", max_tokens=100
        )

        assert result == "Hello, world."

    @pytest.mark.asyncio
    async def test_complete_passes_system_user_and_max_tokens(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="ok")]
        provider._client.messages.create = AsyncMock(return_value=fake_response)

        await provider.complete(system="SYS", user="USR", max_tokens=42)

        call = provider._client.messages.create.call_args
        assert call.kwargs["model"] == "claude-sonnet-4-6"
        assert call.kwargs["system"] == "SYS"
        assert call.kwargs["max_tokens"] == 42
        assert call.kwargs["messages"] == [{"role": "user", "content": "USR"}]

    @pytest.mark.asyncio
    async def test_complete_wraps_sdk_error_as_provider_error(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        provider._client.messages.create = AsyncMock(side_effect=RuntimeError("nope"))

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(system="s", user="u", max_tokens=10)

        assert "nope" in str(exc_info.value)
