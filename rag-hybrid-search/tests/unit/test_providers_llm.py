"""Tests for LLMProvider implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from src.exceptions import ProviderError
from src.providers.llm import AnthropicLLMProvider, LLMProvider


def _fake_response(text: str) -> MagicMock:
    """Build a response whose content[0] passes isinstance(TextBlock)."""
    block = MagicMock(spec=anthropic.types.TextBlock)
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


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
        provider._client.messages.create = AsyncMock(return_value=_fake_response("Hello, world."))

        result = await provider.complete(system="be terse", user="say hi", max_tokens=100)

        assert result == "Hello, world."

    @pytest.mark.asyncio
    async def test_complete_passes_system_user_and_max_tokens(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        provider._client.messages.create = AsyncMock(return_value=_fake_response("ok"))

        await provider.complete(system="SYS", user="USR", max_tokens=42)

        call = provider._client.messages.create.call_args
        assert call.kwargs["model"] == "claude-sonnet-4-6"
        assert call.kwargs["system"] == "SYS"
        assert call.kwargs["max_tokens"] == 42
        assert call.kwargs["messages"] == [{"role": "user", "content": "USR"}]

    @pytest.mark.asyncio
    async def test_complete_rejects_non_text_block(self):
        """If Claude returns a ThinkingBlock or ToolUseBlock, raise ProviderError."""
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        non_text = MagicMock(spec=anthropic.types.ThinkingBlock)
        response = MagicMock()
        response.content = [non_text]
        provider._client.messages.create = AsyncMock(return_value=response)

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(system="s", user="u", max_tokens=10)

        assert "TextBlock" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_wraps_sdk_error_as_provider_error(self):
        provider = AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-6")
        provider._client.messages.create = AsyncMock(side_effect=RuntimeError("nope"))

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(system="s", user="u", max_tokens=10)

        assert "nope" in str(exc_info.value)
