# Evaluation Results

> **Last run:** 2026-05-27 15:09 UTC  •  **Commit:** `cf13a0088d60`  •  **LLM-as-judge scores included.**
> Dataset: 53 golden questions over the Nexus API docs corpus.

## Headline ablation

Each row is one pipeline configuration evaluated on the full golden dataset.

| Config | hit@1 | hit@3 | hit@5 | MRR@10 | p50 total ms | p95 total ms |
|---|---|---|---|---|---|---|
| `dense-only` | 0.811 | 0.849 | 0.887 | 0.839 | 8731.1 | 11985.4 |
| `hybrid` | 0.736 | 0.811 | 0.849 | 0.774 | 8466.6 | 13259.8 |
| `hybrid+rerank` | 0.717 | 0.849 | 0.868 | 0.784 | 8602.8 | 14854.1 |

**How to read this:** higher is better for retrieval metrics (hit@k, MRR); lower is better for latency. The deltas between rows tell you whether each stage of the pipeline is *measurably* justified. If hybrid does **not** beat dense-only here, that is a real finding, not a bug — see [ADR-001 Validation](adr/001-hybrid-retrieval.md#validation) for the analysis of when hybrid pays off (large/noisy corpora, exact-keyword-heavy queries) versus when dense-only is the better default (small clean corpora, strong embedder).

> **This run:** dense-only won hit@1 (0.811) and MRR (0.839) over hybrid+rerank (0.717 / 0.784). The 109-chunk demo corpus is too small and clean for hybrid retrieval to pay off — `voyage-3` dense recall is near-perfect, so BM25 + RRF mostly add noise. Full analysis in [ADR-001](adr/001-hybrid-retrieval.md#validation-update--2026-05-27-first-live-run-hypothesis-not-confirmed-on-this-corpus).

## Latency breakdown (per-stage, p50)

| Config | embed p50 | retrieve p50 | rerank p50 | generate p50 |
|---|---|---|---|---|
| `dense-only` | 292.5 ms | 3.4 ms | 0.0 ms | 8343.7 ms |
| `hybrid` | 281.2 ms | 14.2 ms | 0.0 ms | 8123.5 ms |
| `hybrid+rerank` | 281.9 ms | 14.4 ms | 297.9 ms | 7586.6 ms |

The cross-encoder rerank adds ~50–200 ms to each query but is expected to materially improve MRR on ambiguous queries. Compare the rows above to see the trade-off in your run.

## Retrieval quality by question category

| Config | hit@1 (ambiguous) | hit@1 (multi_hop) | hit@1 (simple_lookup) | hit@1 (unanswerable) |
|---|---|---|---|---|
| `dense-only` | 1.0 | 1.0 | 0.897 | 0.222 |
| `hybrid` | 1.0 | 0.857 | 0.862 | 0.111 |
| `hybrid+rerank` | 1.0 | 0.929 | 0.759 | 0.222 |

`simple_lookup` should be near 1.0 for any working config (exact-keyword or direct paraphrase). `multi_hop` benefits most from rerank. `unanswerable` mixes truly-unanswerable questions (no `expected_sources`, which score 0 by design) with *partially*-answerable ones that do name a source — so this column is low but non-zero, and is best read as "did we retrieve the one doc that's at least tangentially relevant".

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
