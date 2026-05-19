# PR-1: Provider Abstraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract vendor SDK calls behind `EmbeddingProvider` and `LLMProvider` Protocols so the pipeline depends on interfaces, not on Voyage/Anthropic SDKs directly. Behavior-preserving refactor — all existing tests must pass unchanged at the end.

**Architecture:** Two new Protocols in a new `src/providers/` package, with one concrete implementation each (`VoyageEmbeddingProvider`, `AnthropicLLMProvider`). All consumers (`RAGGenerator`, `CitationVerifier`, `ConfidenceScorer`, `DocumentIndexer`, semantic chunker, `/v1/ask` route) take the Protocol type instead of constructing SDK clients. `src/api/dependencies.py` is the only place SDK keys are read; new config fields (`embedding_backend`, `llm_backend`) gate which concrete class is wired.

**Tech Stack:** Python 3.11+, `typing.Protocol`, `anthropic`, `voyageai`, `pytest`, `pytest-asyncio`, `mypy --strict`. No new dependencies.

---

## Spec Reference

This plan implements §4.1 of [`docs/superpowers/specs/2026-05-19-portfolio-polish-design.md`](../specs/2026-05-19-portfolio-polish-design.md).

## File Structure

**New files:**
- `src/providers/__init__.py` — package init, re-exports Protocols and concrete classes
- `src/providers/embedding.py` — `EmbeddingProvider` Protocol + `VoyageEmbeddingProvider`
- `src/providers/llm.py` — `LLMProvider` Protocol + `AnthropicLLMProvider`
- `tests/unit/__init__.py` — marker for unit-test package
- `tests/unit/test_providers_embedding.py` — VoyageEmbeddingProvider tests with mocked SDK
- `tests/unit/test_providers_llm.py` — AnthropicLLMProvider tests with mocked SDK

**Modified files:**
- `src/exceptions.py` — add `ProviderError` base class
- `src/config.py` — add `embedding_backend` and `llm_backend` Literal fields
- `src/generation/generator.py` — accept `LLMProvider`, not `api_key`/`model`
- `src/generation/citation_verifier.py` — accept `LLMProvider`
- `src/generation/confidence.py` — accept `LLMProvider`
- `src/ingestion/indexer.py` — accept `EmbeddingProvider` (typing only — duck-typed method calls stay the same)
- `src/ingestion/chunkers.py` — accept `EmbeddingProvider` (typing only)
- `src/api/routes/ask.py` — type hint `embedder: EmbeddingProvider`
- `src/api/dependencies.py` — wire providers based on config; remove direct SDK construction from `get_generator`
- `scripts/seed.py` — construct `VoyageEmbeddingProvider` instead of `Embedder`

**Deleted files:**
- `src/ingestion/embedder.py` — logic moved into `VoyageEmbeddingProvider`

---

## Task 1: Add `ProviderError` to exceptions

**Files:**
- Modify: `src/exceptions.py`

- [ ] **Step 1: Open `src/exceptions.py` and add the new exception**

Edit `src/exceptions.py`, appending after the last existing exception:

```python
class ProviderError(RAGError):
    """Raised when a provider (embedding or LLM) is misconfigured or unreachable."""
```

- [ ] **Step 2: Verify mypy strict still passes**

Run: `uv run mypy --strict src/`
Expected: zero errors. If `uv` is unavailable, use `python -m mypy --strict src/`.

- [ ] **Step 3: Commit**

```bash
git add src/exceptions.py
git commit -m "feat(exceptions): add ProviderError for provider-layer failures"
```

---

## Task 2: Create `EmbeddingProvider` Protocol + `VoyageEmbeddingProvider`

**Files:**
- Create: `src/providers/__init__.py`
- Create: `src/providers/embedding.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_providers_embedding.py`

### Step 2.1: Write the failing tests first

- [ ] **Step 2.1.1: Create `tests/unit/__init__.py`** (empty file)

```python
```

- [ ] **Step 2.1.2: Create `tests/unit/test_providers_embedding.py`**

```python
"""Tests for EmbeddingProvider implementations."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exceptions import ProviderError
from src.providers.embedding import EmbeddingProvider, VoyageEmbeddingProvider


class TestVoyageEmbeddingProviderShape:
    def test_satisfies_protocol(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        # Structural check — runtime_checkable Protocol
        assert isinstance(provider, EmbeddingProvider)

    def test_exposes_name_and_dim(self):
        provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3")
        assert provider.name == "voyage:voyage-3"
        assert provider.embedding_dim == 1024


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
        provider = VoyageEmbeddingProvider(
            api_key="test-key", model="voyage-3", batch_size=2
        )

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
```

- [ ] **Step 2.1.3: Run the tests to verify they fail with import error**

Run: `uv run pytest tests/unit/test_providers_embedding.py -v`
Expected: `ImportError: cannot import name 'EmbeddingProvider' from 'src.providers.embedding'` (or similar — module does not yet exist).

### Step 2.2: Implement the Protocol and Voyage adapter

- [ ] **Step 2.2.1: Create `src/providers/__init__.py`**

```python
"""Vendor-agnostic provider interfaces for embeddings and LLM completion.

The pipeline depends on these Protocols, not on Voyage/Anthropic SDKs directly.
Adding a new vendor means adding one new class here — no changes to retrieval
or generation code.
"""
from src.providers.embedding import EmbeddingProvider, VoyageEmbeddingProvider
from src.providers.llm import AnthropicLLMProvider, LLMProvider

__all__ = [
    "EmbeddingProvider",
    "VoyageEmbeddingProvider",
    "LLMProvider",
    "AnthropicLLMProvider",
]
```

- [ ] **Step 2.2.2: Create `src/providers/embedding.py`**

```python
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
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model
        self._batch_size = min(batch_size, 128)
        if model not in _VOYAGE_DIMS:
            raise ProviderError(
                f"Unknown Voyage model '{model}'. "
                f"Known models: {sorted(_VOYAGE_DIMS)}"
            )
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
            return result.embeddings
        except Exception as exc:
            raise ProviderError(f"Voyage embedding call failed: {exc}") from exc

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
```

- [ ] **Step 2.2.3: Run the tests — expect pass**

Run: `uv run pytest tests/unit/test_providers_embedding.py -v`
Expected: all 7 tests pass.

- [ ] **Step 2.2.4: Run mypy strict**

Run: `uv run mypy --strict src/providers/`
Expected: zero errors.

- [ ] **Step 2.2.5: Commit**

```bash
git add src/providers/__init__.py src/providers/embedding.py tests/unit/__init__.py tests/unit/test_providers_embedding.py
git commit -m "feat(providers): add EmbeddingProvider Protocol with Voyage implementation"
```

---

## Task 3: Create `LLMProvider` Protocol + `AnthropicLLMProvider`

**Files:**
- Create: `src/providers/llm.py`
- Create: `tests/unit/test_providers_llm.py`

### Step 3.1: Write the failing tests

- [ ] **Step 3.1.1: Create `tests/unit/test_providers_llm.py`**

```python
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
```

- [ ] **Step 3.1.2: Run the tests — expect failure (module missing)**

Run: `uv run pytest tests/unit/test_providers_llm.py -v`
Expected: `ImportError: cannot import name 'LLMProvider' from 'src.providers.llm'`

### Step 3.2: Implement the Protocol and Anthropic adapter

- [ ] **Step 3.2.1: Create `src/providers/llm.py`**

```python
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
```

- [ ] **Step 3.2.2: Run the tests — expect pass**

Run: `uv run pytest tests/unit/test_providers_llm.py -v`
Expected: all 5 tests pass.

- [ ] **Step 3.2.3: Run mypy strict**

Run: `uv run mypy --strict src/providers/`
Expected: zero errors.

- [ ] **Step 3.2.4: Commit**

```bash
git add src/providers/llm.py tests/unit/test_providers_llm.py src/providers/__init__.py
git commit -m "feat(providers): add LLMProvider Protocol with Anthropic implementation"
```

---

## Task 4: Add backend-selection fields to config

**Files:**
- Modify: `src/config.py:1-62`
- Modify: `.env.example`

- [ ] **Step 4.1: Add Literal import and backend fields to `src/config.py`**

Edit `src/config.py`. Replace the top of the file:

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
```

with:

```python
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
```

Then add these fields immediately after the existing Anthropic block (after the `llm_max_tokens` line):

```python
    # ── Backend selection ─────────────────────────────────────────────────────
    # 'voyage' uses the Voyage SDK; 'local' uses sentence-transformers (PR-2).
    embedding_backend: Literal["voyage", "local"] = "voyage"
    # 'anthropic' uses Claude; 'replay' uses fixture playback (PR-2).
    llm_backend: Literal["anthropic", "replay"] = "anthropic"
```

- [ ] **Step 4.2: Update `.env.example` to document the new fields**

Append to `.env.example`:

```
# ── Backend selection (PR-2 will add local/replay backends) ──────────────────
# EMBEDDING_BACKEND=voyage    # voyage | local
# LLM_BACKEND=anthropic       # anthropic | replay
```

- [ ] **Step 4.3: Verify config still loads**

Run: `uv run python -c "from src.config import settings; print(settings.embedding_backend, settings.llm_backend)"`
Expected: `voyage anthropic`

- [ ] **Step 4.4: Run mypy strict**

Run: `uv run mypy --strict src/`
Expected: zero errors.

- [ ] **Step 4.5: Commit**

```bash
git add src/config.py .env.example
git commit -m "feat(config): add embedding_backend and llm_backend selectors"
```

---

## Task 5: Refactor `RAGGenerator` to depend on `LLMProvider`

**Files:**
- Modify: `src/generation/generator.py`

- [ ] **Step 5.1: Rewrite `src/generation/generator.py` to take an LLMProvider**

Replace the entire content of `src/generation/generator.py`:

```python
"""Grounded generation with an LLMProvider.

The system prompt instructs the LLM to:
  1. Answer only from the provided context blocks
  2. Use bracketed citations [1], [2], … for every factual claim
  3. Explicitly state when context is insufficient rather than hallucinating
"""
from __future__ import annotations

import structlog

from src.exceptions import GenerationError, ProviderError
from src.generation.citation_verifier import CitationVerifier
from src.generation.confidence import ConfidenceScorer
from src.models import RAGResponse, RetrievedChunk
from src.providers.llm import LLMProvider

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a precise technical documentation assistant. Answer the user's question
using ONLY the numbered context blocks provided below. Do not use any external
knowledge.

Rules:
1. Every factual claim MUST include a bracketed citation matching a context number,
   e.g. "The rate limit is 1000 req/hr [1]."
2. If multiple contexts support a claim, cite all relevant ones: [1][3].
3. If the provided context is insufficient to fully answer the question, clearly
   state: "Based on the available documentation, I can confirm that [partial answer].
   However, I could not find information about [missing parts]."
4. Do NOT invent facts, functions, configs, or error codes not present in the context.
5. Be concise and precise — this is technical documentation, not general prose.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        source = rc.chunk.metadata.filename
        lines.append(f"[{i}] Source: {source}\n{rc.chunk.content}")
    return "\n\n---\n\n".join(lines)


class RAGGenerator:
    def __init__(
        self,
        llm: LLMProvider,
        max_tokens: int,
        confidence_threshold: float,
        verifier: CitationVerifier,
        scorer: ConfidenceScorer,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._threshold = confidence_threshold
        self._verifier = verifier
        self._scorer = scorer

    async def generate(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RAGResponse:
        if not retrieved_chunks:
            return self._insufficient_response(
                question, retrieved_chunks, "No relevant documents found."
            )

        context_block = _build_context_block(retrieved_chunks)
        user_message = f"Context:\n\n{context_block}\n\nQuestion: {question}"

        try:
            answer_text = await self._llm.complete(
                system=_SYSTEM_PROMPT,
                user=user_message,
                max_tokens=self._max_tokens,
            )
        except ProviderError as exc:
            raise GenerationError(str(exc)) from exc

        citations = await self._verifier.verify(answer_text, retrieved_chunks)
        confidence = await self._scorer.score(
            question=question,
            answer=answer_text,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
        )

        insufficient = not confidence.is_sufficient(self._threshold)
        missing_msg = None
        if insufficient:
            missing_msg = (
                "Retrieval confidence is below threshold. "
                "The answer may be incomplete — consider reviewing the source "
                "documents directly."
            )
            log.warning(
                "low_confidence_response",
                confidence=confidence.composite,
                threshold=self._threshold,
            )

        log.info(
            "generation_complete",
            question_len=len(question),
            answer_len=len(answer_text),
            citations=len(citations),
            confidence=round(confidence.composite, 3),
        )
        return RAGResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=retrieved_chunks,
            insufficient_info=insufficient,
            missing_info_message=missing_msg,
        )

    def _insufficient_response(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        reason: str,
    ) -> RAGResponse:
        from src.models import ConfidenceScore

        confidence = ConfidenceScore(
            retrieval_confidence=0.0,
            citation_coverage=0.0,
            answer_completeness=0.0,
            composite=0.0,
        )
        return RAGResponse(
            question=question,
            answer=f"I was unable to answer this question. {reason}",
            citations=[],
            confidence=confidence,
            retrieved_chunks=chunks,
            insufficient_info=True,
            missing_info_message=reason,
        )
```

- [ ] **Step 5.2: Run mypy on the changed file**

Run: `uv run mypy --strict src/generation/generator.py`
Expected: zero errors. (Will surface downstream errors in dependencies.py — that's expected and fixed in Task 9.)

- [ ] **Step 5.3: Commit**

```bash
git add src/generation/generator.py
git commit -m "refactor(generator): depend on LLMProvider and injected verifier/scorer"
```

---

## Task 6: Refactor `CitationVerifier` to depend on `LLMProvider`

**Files:**
- Modify: `src/generation/citation_verifier.py`

- [ ] **Step 6.1: Rewrite `src/generation/citation_verifier.py`**

Replace the entire content of `src/generation/citation_verifier.py`:

```python
"""Citation verification — the quality layer most RAG systems skip.

For each citation [N] in the generated answer, we:
  1. Parse the claim sentence(s) surrounding the citation marker
  2. Send (claim, chunk_N_content) to the LLM as LLM-as-judge
  3. Mark the citation as verified or flagged

This catches the most common RAG failure mode: confident answers with
citations that don't actually support the stated claim.
"""
from __future__ import annotations

import asyncio
import re

import structlog

from src.exceptions import CitationVerificationError, ProviderError
from src.models import Citation, RetrievedChunk
from src.providers.llm import LLMProvider

log = structlog.get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

_VERIFICATION_SYSTEM = """\
You are a fact-checking assistant. Determine whether the provided CLAIM is
directly supported by the PASSAGE. Answer with exactly one of:
  SUPPORTED   — the passage clearly supports the claim
  UNSUPPORTED — the passage does not support or contradicts the claim
  PARTIAL     — the passage partially supports the claim

Then on a new line, write a single-sentence reason.
"""


def _extract_claim_sentences(answer: str, citation_number: int) -> str:
    """Return the sentence(s) immediately surrounding citation [N]."""
    marker = f"[{citation_number}]"
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    relevant = [s for s in sentences if marker in s]
    if not relevant:
        idx = answer.find(marker)
        if idx >= 0:
            return answer[max(0, idx - 200) : idx + 200]
        return ""
    return " ".join(relevant)


def _parse_citation_numbers(answer: str) -> list[int]:
    matches = _CITATION_PATTERN.findall(answer)
    return sorted({int(m) for m in matches})


class CitationVerifier:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def verify(
        self,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        citation_numbers = _parse_citation_numbers(answer)
        if not citation_numbers:
            log.info("no_citations_found")
            return []

        chunk_map = {i + 1: rc for i, rc in enumerate(retrieved_chunks)}
        tasks = [
            self._verify_one(answer, num, chunk_map.get(num))
            for num in citation_numbers
        ]
        citations = await asyncio.gather(*tasks)
        verified_count = sum(1 for c in citations if c.verified)
        log.info(
            "citation_verification_complete",
            total=len(citations),
            verified=verified_count,
        )
        return list(citations)

    async def _verify_one(
        self,
        answer: str,
        number: int,
        rc: RetrievedChunk | None,
    ) -> Citation:
        if rc is None:
            return Citation(
                number=number,
                chunk_id="",
                source_file="",
                text_excerpt="",
                verified=False,
                verification_reason=(
                    f"Citation [{number}] references a non-existent context block."
                ),
            )

        claim = _extract_claim_sentences(answer, number)
        passage = rc.chunk.content[:800]

        try:
            verdict_text = await self._llm.complete(
                system=_VERIFICATION_SYSTEM,
                user=f"CLAIM:\n{claim}\n\nPASSAGE:\n{passage}",
                max_tokens=100,
            )
        except ProviderError as exc:
            raise CitationVerificationError(
                f"Verification call failed for citation [{number}]: {exc}"
            ) from exc

        lines = verdict_text.strip().split("\n", 1)
        verdict = lines[0].strip().upper()
        reason = lines[1].strip() if len(lines) > 1 else verdict_text

        verified = verdict == "SUPPORTED"
        log.debug(
            "citation_verdict",
            number=number,
            verdict=verdict,
            source=rc.chunk.metadata.filename,
        )
        return Citation(
            number=number,
            chunk_id=rc.chunk.id,
            source_file=rc.chunk.metadata.source_file,
            text_excerpt=rc.chunk.content[:200],
            verified=verified,
            verification_reason=reason,
        )
```

- [ ] **Step 6.2: Run the existing citation tests — they exercise only the pure functions**

Run: `uv run pytest tests/test_generation.py -v`
Expected: all tests pass (the pure-function tests don't touch the LLM).

- [ ] **Step 6.3: Commit**

```bash
git add src/generation/citation_verifier.py
git commit -m "refactor(citation): depend on LLMProvider instead of anthropic.AsyncAnthropic"
```

---

## Task 7: Refactor `ConfidenceScorer` to depend on `LLMProvider`

**Files:**
- Modify: `src/generation/confidence.py`

- [ ] **Step 7.1: Rewrite `src/generation/confidence.py`**

Replace the entire content of `src/generation/confidence.py`:

```python
"""Multi-dimensional confidence scoring.

Three dimensions:
  1. retrieval_confidence  — mean normalised relevance score of top-k chunks
  2. citation_coverage     — verified_citations / total_citations
  3. answer_completeness   — LLM-as-judge: did the answer cover all sub-questions?

Composite = 0.4 * retrieval + 0.4 * citation + 0.2 * completeness
"""
from __future__ import annotations

import math

import structlog

from src.exceptions import ProviderError
from src.models import Citation, ConfidenceScore, RetrievedChunk
from src.providers.llm import LLMProvider

log = structlog.get_logger(__name__)

_COMPLETENESS_SYSTEM = """\
You are evaluating whether an answer fully addresses a question.
Given the QUESTION and the ANSWER, rate answer completeness from 0.0 to 1.0:
  1.0 — all aspects of the question are addressed
  0.5 — partial coverage (some sub-questions answered, others missing)
  0.0 — the question is not answered at all

Respond with only the numeric score.
"""


def _retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    scores = []
    for rc in chunks:
        if rc.rerank_score is not None:
            scores.append(1.0 / (1.0 + math.exp(-rc.rerank_score)))
        elif rc.fusion_score > 0:
            scores.append(min(1.0, rc.fusion_score * 10))
        elif rc.dense_score is not None:
            scores.append(rc.dense_score)
    return sum(scores) / len(scores) if scores else 0.0


def _citation_coverage(citations: list[Citation]) -> float:
    if not citations:
        return 1.0
    verified = sum(1 for c in citations if c.verified)
    return verified / len(citations)


class ConfidenceScorer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def score(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        citations: list[Citation],
    ) -> ConfidenceScore:
        retrieval = _retrieval_confidence(retrieved_chunks)
        citation = _citation_coverage(citations)
        completeness = await self._score_completeness(question, answer)

        composite = 0.4 * retrieval + 0.4 * citation + 0.2 * completeness

        log.info(
            "confidence_scored",
            retrieval=round(retrieval, 3),
            citation=round(citation, 3),
            completeness=round(completeness, 3),
            composite=round(composite, 3),
        )
        return ConfidenceScore(
            retrieval_confidence=retrieval,
            citation_coverage=citation,
            answer_completeness=completeness,
            composite=composite,
        )

    async def _score_completeness(self, question: str, answer: str) -> float:
        try:
            text = await self._llm.complete(
                system=_COMPLETENESS_SYSTEM,
                user=f"QUESTION:\n{question}\n\nANSWER:\n{answer[:1000]}",
                max_tokens=10,
            )
            return float(text.strip())
        except (ProviderError, ValueError) as exc:
            log.warning("completeness_scoring_failed", error=str(exc))
            return 0.5
```

- [ ] **Step 7.2: Run the existing confidence tests — they exercise only the pure functions**

Run: `uv run pytest tests/test_generation.py::TestConfidenceScoring -v`
Expected: all tests pass.

- [ ] **Step 7.3: Commit**

```bash
git add src/generation/confidence.py
git commit -m "refactor(confidence): depend on LLMProvider instead of anthropic.AsyncAnthropic"
```

---

## Task 8: Update `indexer.py` and `chunkers.py` to type-hint `EmbeddingProvider`

**Files:**
- Modify: `src/ingestion/indexer.py:19,75`
- Modify: `src/ingestion/chunkers.py:17,85`

The current code calls `embedder.embed_texts(...)` and `embedder.embed_query(...)` — only those two methods. The new `EmbeddingProvider` Protocol uses `embed_documents` instead of `embed_texts`, so we have to update call sites.

- [ ] **Step 8.1: Update `src/ingestion/indexer.py` imports and signature**

In `src/ingestion/indexer.py`:

- Replace line 19:
  ```python
  from src.ingestion.embedder import Embedder
  ```
  with:
  ```python
  from src.providers.embedding import EmbeddingProvider
  ```

- Replace the `embedder: Embedder` parameter type on line 75 (the `DocumentIndexer.__init__` signature) with `embedder: EmbeddingProvider`.

- Find the call `await self._embedder.embed_texts(...)` on line 153 and replace `embed_texts` with `embed_documents`:
  ```python
  embeddings = await self._embedder.embed_documents([c.content for c in chunks])
  ```

- [ ] **Step 8.2: Update `src/ingestion/chunkers.py`**

In `src/ingestion/chunkers.py`:

- Replace line 17 (the import inside the type-check / runtime block — check exact form):
  ```python
  from src.ingestion.embedder import Embedder
  ```
  with:
  ```python
  from src.providers.embedding import EmbeddingProvider
  ```

- Replace the `embedder: Embedder` parameter on line 85 (the semantic chunker `__init__`) with `embedder: EmbeddingProvider`.

- Find the call `await self._embedder.embed_texts(...)` on line 108 and replace `embed_texts` with `embed_documents`:
  ```python
  embeddings = await self._embedder.embed_documents(windows)
  ```

- [ ] **Step 8.3: Re-read the two files to confirm no remaining references to the old class**

Use Grep tool: pattern `Embedder|embed_texts`, path `src/ingestion/`.
Expected: zero matches in `indexer.py` and `chunkers.py`. (Old `src/ingestion/embedder.py` is deleted in Task 11.)

- [ ] **Step 8.4: Run mypy strict**

Run: `uv run mypy --strict src/ingestion/`
Expected: zero errors.

- [ ] **Step 8.5: Run existing ingestion tests**

Run: `uv run pytest tests/test_ingestion.py -v`
Expected: all tests pass (or skip cleanly if they relied on the deleted module — fix them as part of this step if so).

- [ ] **Step 8.6: Commit**

```bash
git add src/ingestion/indexer.py src/ingestion/chunkers.py
git commit -m "refactor(ingestion): consume EmbeddingProvider Protocol"
```

---

## Task 9: Rewire `src/api/dependencies.py` and `src/api/routes/ask.py`

**Files:**
- Modify: `src/api/dependencies.py`
- Modify: `src/api/routes/ask.py:9,19`

- [ ] **Step 9.1: Replace `src/api/dependencies.py` entirely**

Replace the entire content of `src/api/dependencies.py`:

```python
"""Dependency injection container using functools.lru_cache singletons.

All heavyweight objects (models, DB connections, provider clients) are created
once and reused across requests. FastAPI's Depends() wires these into routes.

Concrete providers are selected based on `settings.embedding_backend` and
`settings.llm_backend`. PR-2 adds 'local' and 'replay' alternatives; this PR
ships 'voyage' and 'anthropic' only.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb

from src.config import settings
from src.exceptions import ProviderError
from src.generation.citation_verifier import CitationVerifier
from src.generation.confidence import ConfidenceScorer
from src.generation.generator import RAGGenerator
from src.ingestion.indexer import DocumentIndexer
from src.providers.embedding import EmbeddingProvider, VoyageEmbeddingProvider
from src.providers.llm import AnthropicLLMProvider, LLMProvider
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_backend == "voyage":
        return VoyageEmbeddingProvider(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    raise ProviderError(
        f"Unsupported embedding_backend '{settings.embedding_backend}'. "
        "Local backend is added in PR-2."
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    if settings.llm_backend == "anthropic":
        return AnthropicLLMProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
    raise ProviderError(
        f"Unsupported llm_backend '{settings.llm_backend}'. "
        "Replay backend is added in PR-2."
    )


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "l2"},
    )


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    """Back-compat alias retained for the ask route's `Depends` signature."""
    return get_embedding_provider()


@lru_cache(maxsize=1)
def get_dense_retriever() -> DenseRetriever:
    return DenseRetriever(collection=get_chroma_collection())


@lru_cache(maxsize=1)
def get_sparse_retriever() -> SparseRetriever:
    return SparseRetriever(
        index_path=Path(settings.bm25_index_path),
        collection=get_chroma_collection(),
    )


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(
        model_name=settings.reranker_model,
        use_llm=settings.use_llm_reranker,
        anthropic_api_key=settings.anthropic_api_key,
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense=get_dense_retriever(),
        sparse=get_sparse_retriever(),
        reranker=get_reranker(),
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        fusion_top_k=settings.fusion_top_k,
        rerank_top_k=settings.rerank_top_k,
    )


@lru_cache(maxsize=1)
def get_citation_verifier() -> CitationVerifier:
    return CitationVerifier(llm=get_llm_provider())


@lru_cache(maxsize=1)
def get_confidence_scorer() -> ConfidenceScorer:
    return ConfidenceScorer(llm=get_llm_provider())


@lru_cache(maxsize=1)
def get_generator() -> RAGGenerator:
    return RAGGenerator(
        llm=get_llm_provider(),
        max_tokens=settings.llm_max_tokens,
        confidence_threshold=settings.retrieval_confidence_threshold,
        verifier=get_citation_verifier(),
        scorer=get_confidence_scorer(),
    )


@lru_cache(maxsize=1)
def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        embedder=get_embedding_provider(),
        collection=get_chroma_collection(),
        bm25_index_path=Path(settings.bm25_index_path),
        dedup_threshold=settings.dedup_similarity_threshold,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        semantic_breakpoint_threshold=settings.semantic_breakpoint_threshold,
    )
```

- [ ] **Step 9.2: Update `src/api/routes/ask.py` type hint**

In `src/api/routes/ask.py`:

- Replace line 9:
  ```python
  from src.ingestion.embedder import Embedder
  ```
  with:
  ```python
  from src.providers.embedding import EmbeddingProvider
  ```

- Replace line 19 (the `embedder: Embedder = Depends(...)` line):
  ```python
      embedder: EmbeddingProvider = Depends(get_embedder),
  ```

- [ ] **Step 9.3: Run mypy strict on the whole src/ tree**

Run: `uv run mypy --strict src/`
Expected: zero errors.

- [ ] **Step 9.4: Boot the API and hit /health**

Run (in one terminal): `uv run uvicorn src.api.main:app --port 8000`
Run (in another): `curl -fsS http://localhost:8000/health`
Expected: `{"status":"ok","version":"0.1.0"}`. Then `Ctrl-C` the uvicorn process.

- [ ] **Step 9.5: Commit**

```bash
git add src/api/dependencies.py src/api/routes/ask.py
git commit -m "refactor(api): wire EmbeddingProvider and LLMProvider via DI"
```

---

## Task 10: Update `scripts/seed.py`

**Files:**
- Modify: `scripts/seed.py:32,63`

- [ ] **Step 10.1: Open `scripts/seed.py` and replace the embedder construction**

In `scripts/seed.py`:

- Replace line 32:
  ```python
  from src.ingestion.embedder import Embedder
  ```
  with:
  ```python
  from src.providers.embedding import VoyageEmbeddingProvider
  ```

- Replace the `Embedder(...)` construction around line 63 (`embedder = Embedder(...)`) with:
  ```python
  embedder = VoyageEmbeddingProvider(
      api_key=settings.voyage_api_key,
      model=settings.embedding_model,
      batch_size=settings.embedding_batch_size,
  )
  ```

- [ ] **Step 10.2: Smoke-check that the script parses**

Run: `uv run python -c "import scripts.seed"`
Expected: no errors.

- [ ] **Step 10.3: Commit**

```bash
git add scripts/seed.py
git commit -m "refactor(scripts): seed.py uses VoyageEmbeddingProvider"
```

---

## Task 11: Delete `src/ingestion/embedder.py`

**Files:**
- Delete: `src/ingestion/embedder.py`

- [ ] **Step 11.1: Confirm no remaining imports of the old module**

Use Grep tool: pattern `from src\.ingestion\.embedder|src\.ingestion\.embedder`, no path filter.
Expected: zero matches anywhere in the repo.

- [ ] **Step 11.2: Delete the file**

Run: `git rm src/ingestion/embedder.py`
Expected: file removed and staged.

- [ ] **Step 11.3: Run mypy strict on the whole tree**

Run: `uv run mypy --strict src/`
Expected: zero errors.

- [ ] **Step 11.4: Commit**

```bash
git commit -m "refactor(ingestion): remove legacy Embedder class, superseded by VoyageEmbeddingProvider"
```

---

## Task 12: Full verification + PR description

**Files:**
- None (verification only)

- [ ] **Step 12.1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass; coverage report prints.

- [ ] **Step 12.2: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: zero issues.

- [ ] **Step 12.3: Run mypy strict on everything**

Run: `uv run mypy --strict src/`
Expected: zero errors.

- [ ] **Step 12.4: End-to-end smoke test against the running API**

This step requires real API keys in `.env`. Skip if running CI/offline; the offline path lands in PR-2.

In one terminal: `uv run uvicorn src.api.main:app --port 8000`

In another terminal: send a real query:
```bash
curl -fsS -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the default rate limit?"}'
```

Expected: a 200 response with `answer`, `citations`, `confidence`, `retrieved_chunks` fields populated. The shape and values should be **identical** to what the same query produced before PR-1 (proof of behavior preservation).

`Ctrl-C` the uvicorn process when done.

- [ ] **Step 12.5: Open a PR**

```bash
git push -u origin <branch-name>
gh pr create --title "PR-1: Provider abstraction for embeddings and LLM" --body "$(cat <<'EOF'
## Summary
Pure refactor. Extracts vendor SDK calls behind `EmbeddingProvider` and `LLMProvider` Protocols so the pipeline depends on interfaces, not on Voyage/Anthropic SDKs directly.

Adding a new vendor (OpenAI embeddings, Cohere LLM) now requires one new class in `src/providers/` and zero changes to retrieval or generation code. This is the foundation that PR-2's offline demo mode builds on.

## What changed
- New `src/providers/` package: `EmbeddingProvider` Protocol + `VoyageEmbeddingProvider`; `LLMProvider` Protocol + `AnthropicLLMProvider`.
- `RAGGenerator`, `CitationVerifier`, `ConfidenceScorer` now take a `LLMProvider` instead of constructing the Anthropic client themselves.
- `DocumentIndexer` and the semantic chunker now take an `EmbeddingProvider`.
- `src/api/dependencies.py` is the only place SDK keys are read; it wires providers based on the new `embedding_backend` / `llm_backend` config fields.
- Old `src/ingestion/embedder.py` deleted; functionality moved into `VoyageEmbeddingProvider`.

## Design notes
- `Protocol` (structural typing) over `ABC` (nominal): consumers stay decoupled from any inheritance hierarchy; mocking is trivial; the Protocol can describe types we don't own. See ADR-003 (lands in PR-5).
- `@runtime_checkable` allows `isinstance()` checks in tests but is not used anywhere on the hot path.
- Retry logic with exponential backoff moved into the providers — pipeline code never sees transient SDK failures, only `ProviderError`.

## Test plan
- [ ] All existing tests pass without modification (behavior preservation)
- [ ] New provider tests pass with mocked SDKs (no network)
- [ ] `mypy --strict src/` is clean
- [ ] `ruff check` is clean
- [ ] End-to-end `POST /v1/ask` returns a structurally-identical response to pre-refactor

## Out of scope
- Local embedding backend (PR-2)
- Replay LLM backend (PR-2)
- Eval harness (PR-3)
- CI / badges (PR-4)
- README / ADRs (PR-5)
EOF
)"
```

Expected: PR opens; CI runs (will be set up in PR-4 — for now humans review).

- [ ] **Step 12.6: Mark this plan complete**

When PR-1 is merged, move on to writing the PR-2 plan.

---

## Self-Review Notes

**Spec coverage:** Every bullet in §4.1 of the spec maps to a task here.
- "New package `src/providers/`" → Tasks 2 + 3 ✓
- "EmbeddingProvider / LLMProvider Protocols" → Tasks 2 + 3 ✓
- "VoyageEmbeddingProvider / AnthropicLLMProvider" → Tasks 2 + 3 ✓
- "RAGGenerator takes LLMProvider" → Task 5 ✓
- "CitationVerifier and ConfidenceScorer take LLMProvider" → Tasks 6 + 7 ✓
- "Embedder takes EmbeddingProvider" → Task 8 ✓
- "dependencies.py wires concrete providers" → Task 9 ✓
- "Config fields embedding_backend / llm_backend" → Task 4 ✓
- "Public API unchanged" → verified in Step 12.4 ✓
- "All existing tests pass" → verified in Step 12.1 ✓

**Type consistency:**
- `EmbeddingProvider.embed_documents` (not `embed_texts`) used consistently across Tasks 2, 8, 10.
- `LLMProvider.complete(system, user, max_tokens)` signature used consistently in Tasks 3, 5, 6, 7.
- `RAGGenerator.__init__(llm, max_tokens, confidence_threshold, verifier, scorer)` matches both Task 5 (definition) and Task 9 (instantiation).

**Placeholders scanned:** none.
