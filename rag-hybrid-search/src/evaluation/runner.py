"""Ablation runner — compares dense-only / hybrid / hybrid+rerank configs.

Loops over the configured set of pipeline configurations, runs each
question in the golden dataset through each config, and aggregates:
  - Retrieval metrics (hit@k, MRR) — pure-function, deterministic, cheap
  - LLM-judge metrics (correctness, faithfulness, etc.) — optional, costly
  - Per-stage latency (embed, retrieve, rerank, generate) — measured

Writes two artifacts per run:
  - eval/results/<utc-timestamp>-<config>.json  — full numeric breakdown
  - docs/eval-results.md                        — committed markdown summary

Why three configs:
  - dense-only:    just ChromaDB cosine — baseline
  - hybrid:        dense + sparse (BM25) fused via Reciprocal Rank Fusion
  - hybrid+rerank: hybrid + cross-encoder reranker on top-20 -> top-5

Comparing these proves the architecture isn't cargo-cult: each stage
should be measurably justified by the deltas in the table.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import structlog

from src.evaluation.dataset import GoldenDataset, GoldenQuestion
from src.evaluation.metrics import EvaluationResult, JudgeEvaluator, QuestionResult
from src.evaluation.retrieval_metrics import hit_at_k, mrr_at_k

if TYPE_CHECKING:
    from src.generation.generator import RAGGenerator
    from src.providers.embedding import EmbeddingProvider
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.fusion import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.retrieval.sparse import SparseRetriever

log = structlog.get_logger(__name__)

ConfigName = Literal["dense-only", "hybrid", "hybrid+rerank"]
ALL_CONFIGS: tuple[ConfigName, ...] = ("dense-only", "hybrid", "hybrid+rerank")


@dataclass
class StageLatencies:
    """Per-question latency captured for each pipeline stage (milliseconds)."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.rerank_ms + self.generate_ms


@dataclass
class PerQuestionRecord:
    question_id: str
    question: str
    category: str
    retrieved_sources: list[str]
    expected_sources: list[str]
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_10: float
    latency: StageLatencies
    judge: QuestionResult | None = None  # None when --no-llm-judge


@dataclass
class ConfigRunResult:
    config_name: ConfigName
    per_question: list[PerQuestionRecord] = field(default_factory=list)

    def _agg(self, attr: str) -> float:
        values: list[float] = [float(getattr(r, attr)) for r in self.per_question]
        return float(statistics.mean(values)) if values else 0.0

    def _agg_by_category(self, attr: str) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in self.per_question:
            buckets.setdefault(r.category, []).append(getattr(r, attr))
        return {cat: statistics.mean(vs) for cat, vs in buckets.items()}

    def _latency_percentile(self, stage: str, pct: float) -> float:
        values = sorted(float(getattr(r.latency, stage)) for r in self.per_question)
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(len(values) * pct) - 1))
        return values[idx]

    @property
    def aggregate(self) -> dict[str, float | dict[str, float]]:
        return {
            "hit@1": round(self._agg("hit_at_1"), 3),
            "hit@3": round(self._agg("hit_at_3"), 3),
            "hit@5": round(self._agg("hit_at_5"), 3),
            "mrr@10": round(self._agg("mrr_at_10"), 3),
            "hit@1_by_category": {
                k: round(v, 3) for k, v in self._agg_by_category("hit_at_1").items()
            },
            "latency_p50_total_ms": round(
                statistics.median(r.latency.total_ms for r in self.per_question)
                if self.per_question
                else 0.0,
                1,
            ),
            "latency_p95_total_ms": round(
                self._percentile_total(0.95),
                1,
            ),
            "latency_p50_embed_ms": round(self._latency_percentile("embed_ms", 0.5), 1),
            "latency_p50_retrieve_ms": round(
                self._latency_percentile("retrieve_ms", 0.5), 1
            ),
            "latency_p50_rerank_ms": round(
                self._latency_percentile("rerank_ms", 0.5), 1
            ),
            "latency_p50_generate_ms": round(
                self._latency_percentile("generate_ms", 0.5), 1
            ),
        }

    def _percentile_total(self, pct: float) -> float:
        values = sorted(r.latency.total_ms for r in self.per_question)
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(len(values) * pct) - 1))
        return values[idx]


def _timed_ms() -> float:
    return time.perf_counter() * 1000.0


class AblationRunner:
    """Runs the dataset through each configured pipeline variant.

    The runner is constructed with the pipeline pieces it needs and a
    `JudgeEvaluator` (or `None` to skip LLM-judge metrics). For each
    question, the embed/retrieve/rerank/generate stages are timed
    independently — total latency is the sum.
    """

    def __init__(
        self,
        embedder: "EmbeddingProvider",
        dense: "DenseRetriever",
        sparse: "SparseRetriever",
        reranker: "Reranker",
        generator: "RAGGenerator",
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        fusion_top_k: int = 20,
        rerank_top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        judge: JudgeEvaluator | None = None,
    ) -> None:
        self._embedder = embedder
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._generator = generator
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._fusion_top_k = fusion_top_k
        self._rerank_top_k = rerank_top_k
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._judge = judge

    async def run_config(
        self,
        config_name: ConfigName,
        dataset: GoldenDataset,
    ) -> ConfigRunResult:
        log.info("ablation_config_start", config=config_name, n=len(dataset))
        result = ConfigRunResult(config_name=config_name)
        for q in dataset:
            record = await self._run_one(config_name, q)
            result.per_question.append(record)
            log.debug(
                "ablation_question_done",
                config=config_name,
                id=q.id,
                hit1=record.hit_at_1,
                latency=record.latency.total_ms,
            )
        log.info(
            "ablation_config_done",
            config=config_name,
            n=len(result.per_question),
        )
        return result

    async def _run_one(
        self, config_name: ConfigName, q: GoldenQuestion
    ) -> PerQuestionRecord:
        from src.retrieval.fusion import reciprocal_rank_fusion

        latency = StageLatencies()

        t0 = _timed_ms()
        query_emb = await self._embedder.embed_query(q.question)
        latency.embed_ms = _timed_ms() - t0

        t0 = _timed_ms()
        dense_results = await self._dense.retrieve(query_emb, top_k=self._dense_top_k)

        if config_name == "dense-only":
            candidates = dense_results[: self._fusion_top_k]
        else:
            sparse_results = await self._sparse.retrieve(
                q.question, top_k=self._sparse_top_k
            )
            candidates = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                rrf_k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                top_k=self._fusion_top_k,
            )
        latency.retrieve_ms = _timed_ms() - t0

        if config_name == "hybrid+rerank":
            t0 = _timed_ms()
            reranked = await self._reranker.rerank(
                q.question, candidates, top_k=self._rerank_top_k
            )
            latency.rerank_ms = _timed_ms() - t0
            final_chunks = reranked
        else:
            final_chunks = candidates[: self._rerank_top_k]

        # Generation + LLM-judge (the judge call is optional)
        judge_result: QuestionResult | None = None
        t0 = _timed_ms()
        response = await self._generator.generate(
            question=q.question, retrieved_chunks=final_chunks
        )
        latency.generate_ms = _timed_ms() - t0

        if self._judge is not None:
            judge_result = await self._judge.evaluate_response(q, response)

        retrieved_sources = [
            rc.chunk.metadata.filename for rc in final_chunks
        ]

        return PerQuestionRecord(
            question_id=q.id,
            question=q.question,
            category=q.category,
            retrieved_sources=retrieved_sources,
            expected_sources=q.expected_sources,
            hit_at_1=hit_at_k(retrieved_sources, q.expected_sources, 1),
            hit_at_3=hit_at_k(retrieved_sources, q.expected_sources, 3),
            hit_at_5=hit_at_k(retrieved_sources, q.expected_sources, 5),
            mrr_at_10=mrr_at_k(retrieved_sources, q.expected_sources, 10),
            latency=latency,
            judge=judge_result,
        )


# Keep both names exported for back-compat with anything that imported the older one.
__all__ = [
    "ALL_CONFIGS",
    "AblationRunner",
    "ConfigName",
    "ConfigRunResult",
    "EvaluationResult",
    "PerQuestionRecord",
    "StageLatencies",
]
