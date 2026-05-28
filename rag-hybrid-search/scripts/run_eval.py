"""Run the ablation eval and write JSON + markdown results.

Default: runs all three configs (dense-only, hybrid, hybrid+rerank), writes
per-config JSON to eval/results/<utc-timestamp>-<config>.json, and (re)writes
docs/eval-results.md with the headline tables.

The index must be seeded first (`make seed`). Live API keys are required for
real evaluation; in offline mode the eval still runs but generation responses
come from replay fixtures (only the curated demo questions will get real
answers — other golden questions show the replay-miss diagnostic).

Usage:
    uv run python scripts/run_eval.py                       # all 3 configs
    uv run python scripts/run_eval.py --config hybrid       # just one
    uv run python scripts/run_eval.py --no-llm-judge        # retrieval only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.evaluation.dataset import GoldenDataset
from src.evaluation.metrics import JudgeEvaluator
from src.evaluation.runner import (
    ALL_CONFIGS,
    AblationRunner,
    ConfigName,
    ConfigRunResult,
)

log = structlog.get_logger(__name__)


def _format_aggregate_table(results: list[ConfigRunResult]) -> str:
    """Headline ablation table comparing all configs."""
    header = (
        "| Config | hit@1 | hit@3 | hit@5 | MRR@10 | p50 total ms | p95 total ms |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for r in results:
        agg = r.aggregate
        rows.append(
            f"| `{r.config_name}` | {agg['hit@1']} | {agg['hit@3']} | {agg['hit@5']} "
            f"| {agg['mrr@10']} | {agg['latency_p50_total_ms']} "
            f"| {agg['latency_p95_total_ms']} |"
        )
    return header + "\n" + "\n".join(rows)


def _format_latency_breakdown(results: list[ConfigRunResult]) -> str:
    header = (
        "| Config | embed p50 | retrieve p50 | rerank p50 | generate p50 |\n|---|---|---|---|---|"
    )
    rows = []
    for r in results:
        agg = r.aggregate
        rows.append(
            f"| `{r.config_name}` | {agg['latency_p50_embed_ms']} ms "
            f"| {agg['latency_p50_retrieve_ms']} ms "
            f"| {agg['latency_p50_rerank_ms']} ms "
            f"| {agg['latency_p50_generate_ms']} ms |"
        )
    return header + "\n" + "\n".join(rows)


def _format_by_category(results: list[ConfigRunResult]) -> str:
    if not results:
        return ""
    categories = sorted(
        {cat for r in results for cat in r.aggregate["hit@1_by_category"]}  # type: ignore[union-attr]
    )
    header_cells = ["Config"] + [f"hit@1 ({cat})" for cat in categories]
    header = "| " + " | ".join(header_cells) + " |\n|" + "---|" * len(header_cells)
    rows = []
    for r in results:
        by_cat = r.aggregate["hit@1_by_category"]
        assert isinstance(by_cat, dict)
        cells = [f"`{r.config_name}`"] + [str(by_cat.get(cat, "—")) for cat in categories]
        rows.append("| " + " | ".join(cells) + " |")
    return header + "\n" + "\n".join(rows)


def _build_markdown(
    results: list[ConfigRunResult], git_commit: str | None, judge_used: bool
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    judge_note = (
        "LLM-as-judge scores included."
        if judge_used
        else "LLM-as-judge scores skipped (`--no-llm-judge`)."
    )
    commit_note = f"`{git_commit[:12]}`" if git_commit else "_(no git commit recorded)_"
    n_questions = len(results[0].per_question) if results else 0

    return f"""# Evaluation Results

> **Last run:** {timestamp}  •  **Commit:** {commit_note}  •  **{judge_note}**
> Dataset: {n_questions} golden questions over the Nexus API docs corpus.

## Headline ablation

Each row is one pipeline configuration evaluated on the full golden dataset.

{_format_aggregate_table(results)}

**How to read this:** higher is better for retrieval metrics (hit@k, MRR); \
lower is better for latency. The deltas between rows tell you whether each \
stage of the pipeline is *measurably* justified. If hybrid does **not** beat \
dense-only here, that is a real finding, not a bug — see \
[ADR-001 Validation](adr/001-hybrid-retrieval.md#validation) for the analysis \
of when hybrid pays off (large/noisy corpora, exact-keyword-heavy queries) \
versus when dense-only is the better default (small clean corpora, strong embedder).

## Latency breakdown (per-stage, p50)

{_format_latency_breakdown(results)}

The cross-encoder rerank adds ~50–200 ms to each query but is expected to \
materially improve MRR on ambiguous queries. Compare the rows above to see \
the trade-off in your run.

## Retrieval quality by question category

{_format_by_category(results)}

`simple_lookup` should be near 1.0 for any working config (exact-keyword or \
direct paraphrase). `multi_hop` benefits most from rerank. `unanswerable` \
mixes truly-unanswerable questions (no `expected_sources`, which score 0 by \
design) with *partially*-answerable ones that do name a source — so this \
column is low but non-zero, and is best read as "did we retrieve the one \
doc that's at least tangentially relevant".

## How to reproduce

```bash
# 1. Seed the index (real or local embeddings — both work)
make seed

# 2. Run the full ablation
make eval

# Or run one config at a time:
uv run python scripts/run_eval.py --config dense-only
uv run python scripts/run_eval.py --config hybrid
uv run python scripts/run_eval.py --config hybrid+rerank

# Skip LLM-as-judge calls (faster, no Anthropic cost):
uv run python scripts/run_eval.py --no-llm-judge
```

Per-config raw JSON results live in [`eval/results/`](../eval/results/).
"""


def _read_git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def _run(
    configs: list[ConfigName],
    no_llm_judge: bool,
    output_json_dir: Path,
    output_md: Path,
    golden_path: Path,
) -> None:
    from src.api.dependencies import (
        get_dense_retriever,
        get_embedding_provider,
        get_generator,
        get_llm_provider,
        get_reranker,
        get_sparse_retriever,
    )
    from src.config import settings

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    dataset = GoldenDataset(golden_path)
    log.info("eval_start", configs=configs, n=len(dataset))

    judge = None if no_llm_judge else JudgeEvaluator(llm=get_llm_provider())

    runner = AblationRunner(
        embedder=get_embedding_provider(),
        dense=get_dense_retriever(),
        sparse=get_sparse_retriever(),
        reranker=get_reranker(),
        generator=get_generator(),
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        fusion_top_k=settings.fusion_top_k,
        rerank_top_k=settings.rerank_top_k,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        judge=judge,
    )

    results: list[ConfigRunResult] = []
    timestamp_slug = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for cfg in configs:
        cfg_result = await runner.run_config(cfg, dataset)
        results.append(cfg_result)

        json_path = output_json_dir / f"{timestamp_slug}-{cfg}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "config_name": cfg_result.config_name,
                    "aggregate": cfg_result.aggregate,
                    "per_question": [
                        {
                            "id": r.question_id,
                            "question": r.question,
                            "category": r.category,
                            "hit@1": r.hit_at_1,
                            "hit@3": r.hit_at_3,
                            "hit@5": r.hit_at_5,
                            "mrr@10": r.mrr_at_10,
                            "latency_ms": {
                                "embed": round(r.latency.embed_ms, 1),
                                "retrieve": round(r.latency.retrieve_ms, 1),
                                "rerank": round(r.latency.rerank_ms, 1),
                                "generate": round(r.latency.generate_ms, 1),
                                "total": round(r.latency.total_ms, 1),
                            },
                            "retrieved_sources": r.retrieved_sources,
                            "expected_sources": r.expected_sources,
                            "judge": (
                                {
                                    "correctness": r.judge.answer_correctness,
                                    "faithfulness": r.judge.faithfulness,
                                    "retrieval_relevance": r.judge.retrieval_relevance,
                                    "citation_accuracy": r.judge.citation_accuracy,
                                    "composite": r.judge.composite,
                                }
                                if r.judge is not None
                                else None
                            ),
                        }
                        for r in cfg_result.per_question
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("eval_config_written", path=str(json_path))

    md = _build_markdown(results, _read_git_commit(), judge_used=not no_llm_judge)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")
    print(f"\n[OK] Wrote {output_md}")
    print(f"     {len(results)} config(s) × {len(dataset)} questions")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ablation eval and write JSON + markdown summary."
    )
    parser.add_argument(
        "--config",
        choices=list(ALL_CONFIGS) + ["all"],
        default="all",
        help="Pipeline config to evaluate (default: all)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge calls; retrieval metrics still computed",
    )
    parser.add_argument(
        "--output-json-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory for per-config JSON results",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/eval-results.md"),
        help="Path for the committed markdown summary",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("eval/golden_dataset.json"),
        help="Path to the golden dataset JSON",
    )
    args = parser.parse_args()

    configs: list[ConfigName] = list(ALL_CONFIGS) if args.config == "all" else [args.config]

    asyncio.run(
        _run(
            configs=configs,
            no_llm_judge=args.no_llm_judge,
            output_json_dir=args.output_json_dir,
            output_md=args.output_md,
            golden_path=args.golden,
        )
    )


if __name__ == "__main__":
    main()
