# ADR-002: LLM-as-Judge for Citation Verification

**Status:** Accepted
**Date:** 2026-05-19
**Deciders:** Project owner (portfolio context)

## Context

The most damaging RAG failure mode in production isn't "I don't know" — that's recoverable. It's **confident, well-cited, plausible-sounding answers where the cited chunks don't actually support the claim**. A user reads the answer, sees a bracketed `[1]`, glances at chunk [1], and trusts it. If the link is wrong, the user is misled and there's no signal that anything went off.

Three observed failure patterns in early prototypes:
1. **Paraphrased claim, related chunk.** The chunk talks about the same topic but doesn't say what the answer claims. (e.g. answer: "rate limit is 1500 req/min [1]"; chunk: actually says 1000.)
2. **Multi-chunk drift.** The LLM cites `[2]` for a fact that `[2]` doesn't contain, because that fact came from `[1]` and the citation was misnumbered.
3. **Fabricated specificity.** The LLM adds a number or constant that isn't in *any* retrieved chunk, with a citation to one that doesn't contain it.

We need a per-citation verification signal so the UI can flag unverified claims and the confidence score can reflect citation quality, not just the *presence* of citations.

## Decision

For each `[N]` marker in the generated answer:

1. Parse the **claim sentence** around the marker (the sentence(s) containing `[N]`).
2. Send `(claim, retrieved_chunk_N.content)` to Claude as **LLM-as-judge** with a focused system prompt that demands one of three verdicts: `SUPPORTED` / `PARTIAL` / `UNSUPPORTED`, plus a one-sentence reason.
3. Mark the `Citation.verified = True` only on `SUPPORTED`. `PARTIAL` and `UNSUPPORTED` are returned to the client with the reason text so the UI can display the disagreement.

The `citation_coverage` metric — the fraction of citations that come back verified — feeds the **composite confidence score**:

```
composite = 0.4 * retrieval_confidence + 0.4 * citation_coverage + 0.2 * answer_completeness
```

The two big weights are the things most under our control: did we retrieve well, did the generator stay grounded.

## Consequences

**Positive:**
- The most common production failure mode (citation drift) becomes **visible to the client** rather than silent. A reviewer sees `verified: false` alongside `verification_reason: "the passage discusses rate limits but specifies 1000 req/min, not 1500"` and immediately knows where the answer drifted.
- The signal is **composable with the rest of confidence scoring**. Low `citation_coverage` alone can drop a response below the threshold and trigger an `insufficient_info` warning even when retrieval looked confident.
- The verification step runs **in parallel** for each citation (`asyncio.gather` in `CitationVerifier.verify`). Latency cost is one LLM call's worth, not N.

**Negative:**
- **Cost:** one extra Claude call per citation. For a typical answer with 3 citations, that's 3 small calls per query (max ~100 tokens output each). Real numbers in `docs/eval-results.md` once the eval runs.
- **Latency:** the verification step is ~150–400 ms per call, parallelised. Adds ~one round-trip to the tail latency.
- **Judge has its own failure modes.** A Claude misclassification can flag a correct citation as `UNSUPPORTED` (false negative) or pass a bad one as `SUPPORTED` (false positive). The judge prompt is deliberately narrow — supports/unsupported/partial with a reason — to minimise this, but it's not eliminated.
- **Calibration drift across model versions.** Updating `LLM_MODEL` shifts the judge's distribution. A future PR should pin a separate `JUDGE_MODEL` env var.

**Neutral:**
- The `verification_reason` text shows up in API responses. Useful debugging signal; potentially noisy if surfaced naively in UI. The Streamlit frontend keeps it in a collapsible details panel rather than the headline.

## Alternatives considered

### A. Rule-based citation extraction (regex match the cited chunk text inside the answer)
- **Why rejected:** brittle on paraphrased claims, which is precisely when verification matters most. An answer saying "the standard tier allows a thousand requests per minute [1]" wouldn't string-match the chunk "Standard | 1,000 | …". The whole point is to catch *semantic* drift, not lexical mismatch.

### B. NLI (Natural Language Inference) models
- Use a fine-tuned NLI model (e.g. `roberta-large-mnli`) to score entailment between claim and passage.
- **Why rejected as default:** adds another local model dependency (~500 MB), GPU-friendly but slow on CPU, and the off-the-shelf NLI models underperform Claude on technical-domain entailment in early testing. Worth revisiting if Claude judge cost becomes the bottleneck — keep the option open.

### C. Self-consistency (sample the answer 3× and only return claims that appear in all three)
- **Why rejected:** triples generation cost; doesn't catch the case where the LLM consistently mis-cites the same way (which is a real failure mode when the chunks are similar enough).

### D. Skip verification, expose the bare composite confidence
- **Why rejected:** composite confidence with no per-citation breakdown is exactly the "feels objective, says nothing" problem we're trying to avoid. Reviewers need to see *which* citation is suspect.

## Implementation notes

- The judge prompt is in `src/generation/citation_verifier.py:_VERIFICATION_SYSTEM`. **It's published, not hidden**, so reviewers can audit the wording. A copy lives in `docs/eval-results.md` once the eval runs, for reproducibility.
- Citations that reference a non-existent chunk number (e.g. answer says `[5]` but only 4 chunks were retrieved) short-circuit to `verified=False` with a structured reason; no LLM call.
- Future improvement: 3× majority-vote on the judge calls would reduce per-call variance at 3× the cost. Worth measuring once a baseline eval run exists.

## Validation

The composite confidence score is exposed in every `/v1/ask` response. The Streamlit frontend renders a three-way gauge breakdown so a human reviewer can spot when one dimension is dragging the composite down. The eval harness includes `citation_accuracy` as one of its four LLM-judge dimensions — but that judge uses a *different* prompt to avoid the obvious circularity of using the same judge that produced the verdict.

## Links

- Implementation: [src/generation/citation_verifier.py](../../src/generation/citation_verifier.py), [src/generation/confidence.py](../../src/generation/confidence.py).
- Anthropic on LLM-as-judge: [Claude evaluation cookbook](https://github.com/anthropics/claude-cookbooks).
