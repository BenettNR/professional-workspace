# Evaluation Results

> **Status:** Template — no real results recorded yet.
>
> The repo owner regenerates this file by running:
>
> ```bash
> make eval                                            # all 3 configs
> # or, for a fast retrieval-only sanity check:
> uv run python scripts/run_eval.py --no-llm-judge
> ```
>
> Live API keys are required for the LLM-as-judge dimensions; the
> retrieval metrics (hit@k, MRR) and per-stage latencies work in
> offline mode too. Per-config JSON results land in
> [`eval/results/`](../eval/results/).

## Headline ablation

| Config | hit@1 | hit@3 | hit@5 | MRR@10 | p50 total ms | p95 total ms |
|---|---|---|---|---|---|---|
| `dense-only` | — | — | — | — | — | — |
| `hybrid` | — | — | — | — | — | — |
| `hybrid+rerank` | — | — | — | — | — | — |

**How to read this:** higher is better for retrieval metrics; lower is
better for latency. The deltas between rows tell you whether each stage
of the pipeline is *measurably* justified.

## Latency breakdown (p50 per stage)

| Config | embed | retrieve | rerank | generate |
|---|---|---|---|---|
| `dense-only` | — | — | — | — |
| `hybrid` | — | — | — | — |
| `hybrid+rerank` | — | — | — | — |

## Retrieval quality by question category

| Config | simple_lookup | multi_hop | ambiguous | unanswerable |
|---|---|---|---|---|
| `dense-only` | — | — | — | — |
| `hybrid` | — | — | — | — |
| `hybrid+rerank` | — | — | — | — |

`simple_lookup` should approach 1.0 for any working config. `multi_hop`
typically benefits most from the cross-encoder rerank. `unanswerable` is
0.0 by design — the dataset records no `expected_sources`, so any
retrieval result is correctly counted as a non-hit.

## How to reproduce

```bash
make seed     # index the corpus with the active embedder backend
make eval     # run all three ablation configs and rewrite this file
```

Per-config raw JSON (with per-question breakdown) lives in
[`eval/results/`](../eval/results/).

## Methodology notes

- **Retrieval metrics** (`hit@1/3/5`, `MRR@10`) are computed by matching
  the basename of the retrieved chunk's source file against the
  golden dataset's `expected_sources` list. Path normalization handles
  Windows/Unix separators and case differences.
- **LLM-as-judge metrics** (correctness, faithfulness, retrieval relevance,
  citation accuracy) ask Claude to score each dimension 0.0–1.0. The same
  judge prompt is used across all configs — comparisons within a single
  run are valid; comparisons across runs require the same judge model.
- **Latency timings** are wall-clock at the Python layer (perf_counter),
  measured per stage on the host running the eval. Network round-trip to
  Anthropic/Voyage dominates the `embed` and `generate` stages in live
  mode.
