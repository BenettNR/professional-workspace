# PR-3: Eval Harness + Ablation Results — Implementation Plan

> **Stacks on PR-2.** Branch: `feat/rag-eval-harness` off `feat/rag-offline-mode`.

**Goal:** Produce a reproducible ablation table comparing `dense-only`, `hybrid` (RRF), and `hybrid+rerank` configurations on the 53-question golden dataset. Publish numbers in `docs/eval-results.md` and link from the README.

**Architecture:** Pure-function retrieval metrics (hit@k, MRR) computed from `expected_sources` (no LLM cost). Refactor existing LLM-judge metrics to use `LLMProvider` (PR-1 consistency). Per-stage latency timing baked into the runner. Markdown output is deterministic given the JSON results.

**Tech stack:** Existing — no new deps.

---

## Files

**New:**
- `src/evaluation/retrieval_metrics.py` — pure-function hit@k, MRR computed against expected_sources
- `src/evaluation/runner.py` — ablation orchestrator that loops over configs, times stages, writes JSON + markdown
- `scripts/run_eval.py` — CLI wrapper: `--config dense-only|hybrid|hybrid+rerank|all`
- `docs/eval-results.md` — template populated by the runner (committed as a placeholder until first run)
- `tests/unit/test_retrieval_metrics.py`

**Modified:**
- `src/evaluation/metrics.py` — refactor to depend on `LLMProvider` (was using anthropic SDK directly); keep behaviour identical
- `src/evaluation/__init__.py` — re-exports
- `Makefile` — `make eval` target already exists as a stub; will work after this PR

## Tasks

### Task 1: Retrieval metrics (pure functions, no LLM)
- [ ] Create `src/evaluation/retrieval_metrics.py` with: `hit_at_k`, `mrr_at_k` computed from the ranked list of retrieved source filenames vs the golden `expected_sources` list.
- [ ] Tests: `tests/unit/test_retrieval_metrics.py` — hit-at-1 perfect/miss/partial, MRR with first-relevant-at-rank-k, empty expected_sources edge case.
- [ ] Commit.

### Task 2: Refactor LLM-judge metrics to LLMProvider
- [ ] Rewrite `src/evaluation/metrics.py` to take an `LLMProvider` instead of constructing `anthropic.AsyncAnthropic`. Behavior identical. Existing 4-dimension judge scoring preserved.
- [ ] Commit.

### Task 3: Ablation runner
- [ ] Create `src/evaluation/runner.py` with `AblationRunner` that:
  - Loops over (dense-only, hybrid, hybrid+rerank) — each config builds a fresh `HybridRetriever` with the right flags
  - For each question: times each stage (embed / retrieve / rerank / generate), captures retrieved sources, runs LLM-judge metrics (optional, behind flag)
  - Aggregates: per-config averages, p50/p95 latency, retrieval metrics by query category
- [ ] No tests for the runner itself (integration-style); tested via the script invocation in Task 4.

### Task 4: CLI script
- [ ] `scripts/run_eval.py --config {dense-only,hybrid,hybrid+rerank,all} [--no-llm-judge] [--output-json PATH] [--output-md PATH]`
- [ ] Default outputs: `eval/results/<utc-timestamp>-<config>.json` and `docs/eval-results.md`
- [ ] `--no-llm-judge` skips LLM-as-judge calls (useful for fast iteration; retrieval metrics still computed)
- [ ] Commit.

### Task 5: Eval results markdown
- [ ] `docs/eval-results.md` placeholder: headline ablation table (empty), per-query-type breakdown, latency table, "How to reproduce" section pointing at `make eval`.
- [ ] The runner re-writes this file with real numbers on each invocation.
- [ ] Commit.

### Task 6: README link
- [ ] Update the README "Eval results" section to embed the headline table from `docs/eval-results.md` (or link to it once populated).
- [ ] Mark PR-3 as ✅ in the Roadmap.
- [ ] Commit.

### Task 7: Verification + PR
- [ ] All existing tests still pass
- [ ] New retrieval-metric tests pass
- [ ] `mypy --strict` clean on new files
- [ ] Push branch and open PR-3 against PR-2's branch.
