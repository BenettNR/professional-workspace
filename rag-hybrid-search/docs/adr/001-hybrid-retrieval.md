# ADR-001: Hybrid Retrieval (Dense + Sparse + Rerank)

**Status:** Accepted, with a validation caveat — the first live eval (2026-05-27) found dense-only outperforms hybrid on the current small corpus. See [Validation update](#validation-update--2026-05-27-first-live-run-hypothesis-not-confirmed-on-this-corpus).
**Date:** 2026-05-19
**Deciders:** Project owner (portfolio context)

## Context

The single most-tunable lever in a RAG system is **what you put in the LLM's context window**. A 5% improvement in retrieval quality typically beats a much larger investment in the generator (model size, prompt engineering, fine-tuning). So the retrieval design has to be deliberate.

The corpus this service targets is **technical API documentation**: rate limits with exact numbers, error codes like `NXE-1004`, configuration keys (`NEXUS_FLUSH_INTERVAL`), CLI invocations, regex patterns. Three retrieval failure modes appear repeatedly when only one strategy is used:

1. **Dense-only fails on exact-keyword queries.** A semantic embedding of `"What does error code NXE-1004 mean?"` blurs the literal `NXE-1004` into a generic notion of "error code lookup", which retrieves rate-limiting and authentication chunks alongside the actual one. Cosine similarity has no privileged signal for *exact term presence*.
2. **Sparse-only (BM25) fails on paraphrase.** `"How do I sign in?"` versus `"authentication setup"` — BM25 sees zero shared tokens after stopword removal and ranks unrelated chunks higher. Bag-of-words has no notion of semantic equivalence.
3. **Either-alone returns noisy top-K.** Even when the right document is in the top-10, the top-5 (what fits in context) may be diluted by near-duplicate or marginally-relevant chunks. The generator then either copies the wrong claim or hedges.

## Decision

Adopt a **three-stage hybrid retrieval pipeline**:

1. **Dense retrieval** via Voyage AI bi-encoder embeddings into ChromaDB. Asymmetric `input_type` (`"document"` for indexing, `"query"` for search) on Voyage models. Returns top-10 by cosine similarity.
2. **Sparse retrieval** via BM25 over a tokenized corpus index. Returns top-10 by BM25 score.
3. **Reciprocal Rank Fusion** merges the two ranked lists with `score(d) = Σ_r weight_r / (k + rank_r(d))`, `k=60` (the standard smoothing constant). Default weights 0.7 dense / 0.3 sparse. Returns top-20.
4. **Cross-encoder reranking** with `ms-marco-MiniLM-L-6-v2`. Sees `(query, candidate)` jointly (unlike bi-encoders, which encode them separately) and reorders the top-20 → top-5. The top-5 is what the generator sees.

The `dense_only=True` flag on `HybridRetriever.retrieve()` is preserved as an ablation lever (used by the eval runner; not exposed in the public API).

## Consequences

**Positive:**
- Each stage is **independently ablatable** — the eval runner (PR-3) generates a comparison table across `dense-only` / `hybrid` / `hybrid+rerank`. The architecture is not cargo-cult: every stage is measurably justified by the deltas in `docs/eval-results.md`.
- BM25 catches exact terms that dense retrieval blurs. Dense catches paraphrases that BM25 misses. The two are genuinely complementary on technical docs.
- Cross-encoder rerank is a precision boost on the top-K that bi-encoders alone can't match because they never see query and candidate together.

**Negative:**
- **Latency:** the rerank stage adds ~50–200 ms at p50 (CPU, on a small candidate set). Live numbers in `docs/eval-results.md`. For a synchronous API call this is acceptable; for streaming generation it's a noticeable tail-latency contributor.
- **Memory:** ChromaDB collection + BM25 pickle + cross-encoder model all live in process memory. Roughly +100 MB resident over dense-only.
- **More moving parts to maintain** — three retrievers, fusion, rerank — each with their own failure modes. The DI in `src/api/dependencies.py` is the firewall: every dependency has a single construction site.
- **RRF weights are tuned, not learned.** 0.7/0.3 is a reasonable default for technical corpora where dense should dominate; an adversarial workload could justify a different split. The eval harness makes it easy to validate the choice on new corpora.

**Neutral:**
- Adds the dependency on `sentence-transformers` (~80 MB cross-encoder model on first run). Considered acceptable — the same library is already a dep for the local-mode embedder in PR-2.

## Alternatives considered

### A. Dense-only with a larger embedder (e.g. `voyage-3-large`)
- **Why rejected:** trades latency and cost for marginal precision on the *paraphrase* axis. It does nothing for the exact-keyword failure mode (a larger bi-encoder still blurs `NXE-1004`). Higher embedding cost without solving the actual gap.

### B. HyDE (Hypothetical Document Embeddings)
- Generate a hypothetical answer first, embed *that*, retrieve against it.
- **Why rejected:** adds an LLM call to the critical path (+latency, +cost on every query) for a gain that's mostly redundant with what the cross-encoder rerank achieves on this corpus. Worth revisiting for queries with no shared vocabulary with the source documents (which is rare in this corpus).

### C. ColBERT (late-interaction multi-vector retrieval)
- **Why rejected:** the index size blows up (one vector per token, not per chunk) and the query-time cost is higher. The precision win is meaningful for ad-hoc open-domain search but smaller on a well-structured technical corpus where chunk boundaries already align with document structure. Worth revisiting for a corpus 10–100x larger.

### D. Ensemble of multiple rerankers (cross-encoder + LLM-judge)
- The reranker module supports `use_llm=True` to swap in an LLM-judge rerank. Could in principle run both and combine.
- **Why rejected as default:** doubles the rerank cost for a small precision delta. The cross-encoder is the right default; LLM-judge rerank is available as an opt-in for workloads where rerank quality dominates.

### E. Sparse-only (BM25)
- Considered for the offline / lightweight path. Rejected because the paraphrase failure mode is too common on real user queries; people don't write queries that look like documentation prose. PR-2's offline mode uses local sentence-transformers as the dense backend instead.

## Validation

The eval harness in `src/evaluation/runner.py` runs all three ablation configurations on the 53-question golden dataset and writes the comparison to `docs/eval-results.md`. The table is regenerated on every `make eval`.

### Validation update — 2026-05-27 (first live run): hypothesis NOT confirmed on this corpus

The first real ablation run produced a result that **contradicts the prediction above**:

| Config | hit@1 | hit@3 | MRR@10 |
|---|---|---|---|
| `dense-only` | **0.811** | 0.849 | **0.839** |
| `hybrid` | 0.736 | 0.811 | 0.774 |
| `hybrid+rerank` | 0.717 | **0.849** | 0.784 |

On this corpus, **dense-only wins hit@1 and MRR**. Hybrid and hybrid+rerank are *worse* on rank-1 precision. This ADR predicted hybrid+rerank would be measurably best; the data says otherwise. Reporting it rather than burying it — that's the entire point of having an eval harness.

**Why dense-only wins here (analysis, not excuse):**

1. **The corpus is small and clean** — 11 documents, 109 chunks, well-structured technical Markdown. Hybrid retrieval's value is recovering *exact-keyword* matches that dense embeddings blur (rare error codes, config keys) **on large, noisy corpora where dense recall degrades**. With 109 chunks and a strong embedder (`voyage-3`), dense retrieval almost never misses the relevant chunk, so BM25 + RRF mostly inject lower-quality candidates that *demote* the correct dense hit. Sparse retrieval helps when dense is weak; here dense isn't weak.
2. **The cross-encoder rerank trades rank-1 precision for top-3 recall.** `hybrid+rerank` has the lowest hit@1 (0.717) but ties dense on hit@3 (0.849) and beats plain hybrid on `multi_hop` queries (0.929 vs 0.857). The reranker reorders the candidate set; on a corpus where the dense top-1 was already correct, reordering can only hurt hit@1.
3. **Generation dominates latency** (~8,000 ms p50). The retrieval and rerank stages (3–300 ms) are negligible by comparison — so the *cost* side of the hybrid trade-off is also smaller than this ADR implied.

**What this means for the decision:**

- For *this* corpus and query distribution, **dense-only would be the right production default.** Hybrid is not paying for itself.
- Hybrid+rerank is expected to win on the conditions it was designed for: a **larger, noisier corpus** (10k+ chunks), a **higher proportion of exact-keyword queries** (error codes, parameter names, version strings) that dense blurs, or a **weaker/cheaper embedder**. None of those hold for the 109-chunk demo corpus.
- The architecture is kept as the *configurable default* (the `dense_only` flag and RRF weights are all tunable), but the honest recommendation for a small clean corpus is to run dense-only. The eval harness is what surfaces this per-corpus rather than guessing.

This is the intended workflow: build the capability, measure it, and let the data — not the original hypothesis — decide. A future change to a larger corpus should re-run `make eval` and revisit this conclusion.

## Links

- Reciprocal Rank Fusion: Cormack, Clarke, Buettcher (2009). [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- Cross-encoder ms-marco: [sbert.net pretrained cross-encoders](https://sbert.net/docs/pretrained_cross-encoders.html).
- Implementation: [src/retrieval/](../../src/retrieval/), [src/evaluation/runner.py](../../src/evaluation/runner.py).
