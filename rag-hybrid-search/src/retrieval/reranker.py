"""Cross-encoder reranking — the second-pass precision boost.

Uses sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2 by default.
The model is lazy-loaded on first use (~80 MB download on first run).

Optionally falls back to LLM-as-judge scoring when use_llm=True.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING

import structlog

from src.models import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load the cross-encoder model once and cache it for the process lifetime."""
    from sentence_transformers import CrossEncoder

    log.info("loading_cross_encoder", model=model_name)
    model: CrossEncoder = CrossEncoder(model_name)
    return model


class Reranker:
    """Re-scores fusion candidates with a cross-encoder or LLM-as-judge."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_llm: bool = False,
        anthropic_api_key: str | None = None,
        llm_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._model_name = model_name
        self._use_llm = use_llm
        self._api_key = anthropic_api_key
        self._llm_model = llm_model

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if self._use_llm:
            scored = await self._llm_rerank(query, candidates)
        else:
            scored = await asyncio.get_event_loop().run_in_executor(
                None, self._cross_encoder_rerank, query, candidates
            )

        scored.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        result = scored[:top_k]
        log.debug("reranking_complete", candidates=len(candidates), kept=len(result))
        return result

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        model = _load_cross_encoder(self._model_name)
        pairs = [(query, rc.chunk.content) for rc in candidates]
        # CrossEncoder.predict's input type is over-broad (covers image/audio
        # variants too); a list of (str, str) tuples is the documented happy path.
        scores = model.predict(pairs)  # type: ignore[arg-type]

        for rc, score in zip(candidates, scores, strict=True):
            rc.rerank_score = float(score)
        return candidates

    async def _llm_rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Score each candidate with a lightweight LLM relevance judgement."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        async def score_one(rc: RetrievedChunk) -> RetrievedChunk:
            prompt = (
                f"Query: {query}\n\n"
                f"Passage: {rc.chunk.content[:500]}\n\n"
                "Rate how relevant this passage is to the query on a scale of 0.0 to 1.0. "
                "Respond with only the numeric score."
            )
            try:
                response = await client.messages.create(
                    model=self._llm_model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                block = response.content[0]
                if not isinstance(block, anthropic.types.TextBlock):
                    raise TypeError(f"Expected TextBlock, got {type(block).__name__}")
                rc.rerank_score = float(block.text.strip())
            except Exception:
                rc.rerank_score = rc.fusion_score  # fallback
            return rc

        return list(await asyncio.gather(*[score_one(rc) for rc in candidates]))
