"""LLMProvider Protocol and concrete implementations.

A single `complete(system, user, max_tokens) -> str` method is the lowest
common denominator across vendors. Streaming, tool use, and structured
output are deliberately out of scope for this Protocol — when we need them,
they will be additive (extending the interface, not breaking it).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.exceptions import ProviderError

log = structlog.get_logger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Async interface for single-turn LLM completion."""

    name: str

    async def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class AnthropicLLMProvider:
    """Anthropic implementation of LLMProvider."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self.name = f"anthropic:{model}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except Exception as exc:
            raise ProviderError(f"Anthropic completion failed: {exc}") from exc
