# ADR-001: Hybrid Retrieval (Dense + Sparse + Rerank)

**Status:** Accepted
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

The eval harness in `src/evaluation/runner.py` runs all three ablation configurations on the 53-question golden dataset and writes the comparison to `docs/eval-results.md`. The table is regenerated on every `make eval`. The decision in this ADR stands as long as the table shows `hybrid+rerank` is measurably best on the corpus this service serves.

If a future change makes one stage no longer pay for itself in that table, this ADR should be revisited rather than the stage silently kept.

## Links

- Reciprocal Rank Fusion: Cormack, Clarke, Buettcher (2009). [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- Cross-encoder ms-marco: [sbert.net pretrained cross-encoders](https://sbert.net/docs/pretrained_cross-encoders.html).
- Implementation: [src/retrieval/](../../src/retrieval/), [src/evaluation/runner.py](../../src/evaluation/runner.py).
