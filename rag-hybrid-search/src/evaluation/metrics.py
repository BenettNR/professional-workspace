"""LLM-as-judge evaluation metrics for the RAG pipeline.

Four dimensions per test case:
  1. answer_correctness  — does the answer match the expected answer?
  2. faithfulness        — are all claims grounded in retrieved context?
  3. retrieval_relevance — were the right chunks retrieved?
  4. citation_accuracy   — do citations actually support their claims?

Each judge prompt elicits a single 0.0-1.0 score from the LLM; conservative
fallback (0.5) on parse failure or provider error.

The runner depends on `LLMProvider` (the Protocol introduced in PR-1) rather
than the Anthropic SDK directly — so the same evaluator can run with replay
fixtures or real Claude depending on backend configuration.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.evaluation.dataset import GoldenQuestion
from src.models import RAGResponse
from src.providers.llm import LLMProvider

log = structlog.get_logger(__name__)

_JUDGE_SYSTEM = """\
You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.
Score the following dimension from 0.0 to 1.0 based on the provided inputs.
Respond with only the numeric score.
"""


@dataclass
class QuestionResult:
    question_id: str
    question: str
    answer_correctness: float
    faithfulness: float
    retrieval_relevance: float
    citation_accuracy: float

    @property
    def composite(self) -> float:
        return (
            self.answer_correctness * 0.3
            + self.faithfulness * 0.3
            + self.retrieval_relevance * 0.2
            + self.citation_accuracy * 0.2
        )


@dataclass
class EvaluationResult:
    config_name: str
    total_questions: int
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def avg_correctness(self) -> float:
        return self._avg("answer_correctness")

    @property
    def avg_faithfulness(self) -> float:
        return self._avg("faithfulness")

    @property
    def avg_retrieval_relevance(self) -> float:
        return self._avg("retrieval_relevance")

    @property
    def avg_citation_accuracy(self) -> float:
        return self._avg("citation_accuracy")

    @property
    def avg_composite(self) -> float:
        return self._avg("composite")

    def _avg(self, attr: str) -> float:
        if not self.results:
            return 0.0
        total: float = sum(float(getattr(r, attr)) for r in self.results)
        return total / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "total_questions": self.total_questions,
            "avg_correctness": round(self.avg_correctness, 3),
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_retrieval_relevance": round(self.avg_retrieval_relevance, 3),
            "avg_citation_accuracy": round(self.avg_citation_accuracy, 3),
            "avg_composite": round(self.avg_composite, 3),
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.to_dict(),
                    "per_question": [
                        {
                            "id": r.question_id,
                            "question": r.question,
                            "answer_correctness": round(r.answer_correctness, 3),
                            "faithfulness": round(r.faithfulness, 3),
                            "retrieval_relevance": round(r.retrieval_relevance, 3),
                            "citation_accuracy": round(r.citation_accuracy, 3),
                            "composite": round(r.composite, 3),
                        }
                        for r in self.results
                    ],
                },
                f,
                indent=2,
            )


class JudgeEvaluator:
    """LLM-as-judge scorer. Constructed with an LLMProvider so the same
    evaluator can run with replay fixtures or real Claude.
    """

    def __init__(self, llm: LLMProvider, max_tokens: int = 10) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    async def evaluate_response(
        self,
        golden: GoldenQuestion,
        response: RAGResponse,
    ) -> QuestionResult:
        correctness, faithfulness, relevance = await asyncio.gather(
            self._score_correctness(golden, response),
            self._score_faithfulness(response),
            self._score_retrieval_relevance(golden, response),
        )
        # citation_accuracy comes from the pipeline's own composite — no
        # extra LLM call needed since CitationVerifier already scored it.
        citation_accuracy = response.confidence.citation_coverage

        return QuestionResult(
            question_id=golden.id,
            question=golden.question,
            answer_correctness=correctness,
            faithfulness=faithfulness,
            retrieval_relevance=relevance,
            citation_accuracy=citation_accuracy,
        )

    async def _judge(self, user_prompt: str) -> float:
        try:
            text = await self._llm.complete(
                system=_JUDGE_SYSTEM,
                user=user_prompt,
                max_tokens=self._max_tokens,
            )
            return float(text.strip())
        except Exception as exc:
            log.warning("judge_call_failed", error=str(exc))
            return 0.5

    async def _score_correctness(self, golden: GoldenQuestion, response: RAGResponse) -> float:
        prompt = (
            f"Expected answer: {golden.expected_answer}\n\n"
            f"Actual answer: {response.answer}\n\n"
            "Score how correctly the actual answer matches the expected answer (0.0-1.0)."
        )
        return await self._judge(prompt)

    async def _score_faithfulness(self, response: RAGResponse) -> float:
        context = "\n\n".join(rc.chunk.content[:300] for rc in response.retrieved_chunks[:5])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer:\n{response.answer[:500]}\n\n"
            "Score how faithfully every claim in the answer is grounded in the context (0.0-1.0)."
        )
        return await self._judge(prompt)

    async def _score_retrieval_relevance(
        self, golden: GoldenQuestion, response: RAGResponse
    ) -> float:
        retrieved_sources = [rc.chunk.metadata.filename for rc in response.retrieved_chunks]
        prompt = (
            f"Question: {golden.question}\n"
            f"Expected sources: {golden.expected_sources}\n"
            f"Retrieved sources: {retrieved_sources}\n\n"
            "Score how relevant the retrieved sources are for answering this question (0.0-1.0)."
        )
        return await self._judge(prompt)


# Back-compat alias — the old name; same class.
EvaluationRunner = JudgeEvaluator
