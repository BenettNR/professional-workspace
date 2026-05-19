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
