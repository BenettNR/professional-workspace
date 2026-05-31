"""Pure-retrieval ablation comparison — zero LLM calls, free, ~30 sec.

Runs the 53-question golden dataset through dense-only / hybrid /
hybrid+rerank, computes hit@k and MRR using retrieval_metrics, prints
a side-by-side table. Use this to test whether the corpus has grown
to where hybrid retrieval starts paying off (vs. the small-corpus
result where dense-only won — see ADR-001's Validation update).

This skips generation entirely (hence "free"). Full eval with
generation + LLM-judge is `make eval` (costs real money).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from pathlib import Path

import structlog

from src.evaluation.dataset import GoldenDataset
from src.evaluation.retrieval_metrics import hit_at_k, mrr_at_k
from src.retrieval.fusion import reciprocal_rank_fusion


async def main(dataset_path: Path) -> None:
    from src.api.dependencies import (
        get_chroma_collection,
        get_dense_retriever,
        get_embedding_provider,
        get_reranker,
        get_sparse_retriever,
    )
    from src.config import settings

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

    dataset = GoldenDataset(dataset_path)
    n_chunks = get_chroma_collection().count()
    print(f"Corpus size: {n_chunks} chunks")
    print(f"Dataset: {dataset_path.name} ({len(dataset)} questions)")
    print()

    embedder = get_embedding_provider()
    dense = get_dense_retriever()
    sparse = get_sparse_retriever()
    reranker = get_reranker()

    configs = ("dense-only", "hybrid", "hybrid+rerank")
    results: dict[str, dict[str, list[float]]] = {
        c: {"hit@1": [], "hit@3": [], "hit@5": [], "mrr@10": []} for c in configs
    }

    for i, q in enumerate(dataset, 1):
        if i % 10 == 0:
            print(f"  ...{i}/{len(dataset)} questions")
        q_emb = await embedder.embed_query(q.question)
        dense_results = await dense.retrieve(q_emb, top_k=settings.dense_top_k)
        sparse_results = await sparse.retrieve(q.question, top_k=settings.sparse_top_k)
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
            top_k=settings.fusion_top_k,
        )
        reranked = await reranker.rerank(q.question, fused, top_k=settings.rerank_top_k)

        per_cfg = {
            "dense-only": [
                c.chunk.metadata.filename for c in dense_results[: settings.rerank_top_k]
            ],
            "hybrid": [c.chunk.metadata.filename for c in fused[: settings.rerank_top_k]],
            "hybrid+rerank": [c.chunk.metadata.filename for c in reranked],
        }

        for cfg, sources in per_cfg.items():
            results[cfg]["hit@1"].append(hit_at_k(sources, q.expected_sources, 1))
            results[cfg]["hit@3"].append(hit_at_k(sources, q.expected_sources, 3))
            results[cfg]["hit@5"].append(hit_at_k(sources, q.expected_sources, 5))
            results[cfg]["mrr@10"].append(mrr_at_k(sources, q.expected_sources, 10))

    print()
    print(f"Retrieval comparison @ {n_chunks} chunks (NO LLM calls)")
    print("=" * 64)
    print(f"{'Config':<16} {'hit@1':>8} {'hit@3':>8} {'hit@5':>8} {'MRR@10':>8}")
    print("-" * 64)
    for cfg in configs:
        h1 = statistics.mean(results[cfg]["hit@1"])
        h3 = statistics.mean(results[cfg]["hit@3"])
        h5 = statistics.mean(results[cfg]["hit@5"])
        mrr = statistics.mean(results[cfg]["mrr@10"])
        print(f"{cfg:<16} {h1:>8.3f} {h3:>8.3f} {h5:>8.3f} {mrr:>8.3f}")
    print("=" * 64)
    print()
    print("Compare to the 2026-05-27 first-live-run at 109 chunks:")
    print("  dense-only      0.811   0.849   0.887   0.839   <- best")
    print("  hybrid          0.736   0.811   0.849   0.774")
    print("  hybrid+rerank   0.717   0.849   0.868   0.784")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/golden_dataset.json"),
        help="Path to the golden dataset JSON",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dataset))
