"""Multi-dimensional confidence scoring.

Three dimensions:
  1. retrieval_confidence  — mean normalised relevance score of top-k chunks
  2. citation_coverage     — verified_citations / total_citations
  3. answer_completeness   — LLM-as-judge: did the answer cover all sub-questions?

Composite = 0.4 * retrieval + 0.4 * citation + 0.2 * completeness
"""
from __future__ import annotations

import anthropic
import structlog

from src.models import Citation, ConfidenceScore, RetrievedChunk

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
            # Normalise cross-encoder score to [0,1] via sigmoid approximation
            import math

            scores.append(1.0 / (1.0 + math.exp(-rc.rerank_score)))
        elif rc.fusion_score > 0:
            scores.append(min(1.0, rc.fusion_score * 10))
        elif rc.dense_score is not None:
            scores.append(rc.dense_score)
    return sum(scores) / len(scores) if scores else 0.0


def _citation_coverage(citations: list[Citation]) -> float:
    if not citations:
        return 1.0  # no citations made → no false citations; give benefit of the doubt
    verified = sum(1 for c in citations if c.verified)
    return verified / len(citations)


class ConfidenceScorer:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

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
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=10,
                system=_COMPLETENESS_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer[:1000]}",
                    }
                ],
            )
            return float(response.content[0].text.strip())
        except Exception as exc:
            log.warning("completeness_scoring_failed", error=str(exc))
            return 0.5  # conservative fallback
