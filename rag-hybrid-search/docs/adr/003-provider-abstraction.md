# ADR-003: Provider Abstraction (Embedding + LLM) and Offline Demo Mode

**Status:** Accepted
**Date:** 2026-05-19
**Deciders:** Project owner (portfolio context)

## Context

Two pressures converged:

1. **Vendor coupling was creeping into business logic.** `RAGGenerator`, `CitationVerifier`, and `ConfidenceScorer` each constructed their own `anthropic.AsyncAnthropic` client at initialisation. `Embedder` lived inside `src/ingestion/` and was tied to Voyage. Adding a new vendor (OpenAI embeddings, Cohere LLM, local sentence-transformers for offline mode) would touch every consumer.

2. **Portfolio reviewers should be able to see this work without paying.** Cloning the repo, getting an Anthropic key, getting a Voyage key, putting them in a `.env` is friction. Many reviewers won't bother — they'll skim the README and move on. The first 90 seconds need to produce a *working demo* with zero cost.

Both pressures pointed at the same engineering move: define an interface for the vendor-dependent parts, swap implementations behind it.

## Decision

Two `typing.Protocol`s in a new `src/providers/` package:

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str            # e.g. "voyage:voyage-3"; used in Chroma collection names + logs
    embedding_dim: int   # checked against the index at startup
    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

@runtime_checkable
class LLMProvider(Protocol):
    name: str
    async def complete(self, system: str, user: str, max_tokens: int) -> str: ...
```

Four concrete implementations:

| Provider | Class | Purpose |
|---|---|---|
| Voyage embeddings | `VoyageEmbeddingProvider` | Production / live mode |
| Local embeddings | `LocalSentenceTransformerEmbeddingProvider` | Offline mode; `all-MiniLM-L6-v2`, 384-dim, CPU |
| Anthropic LLM | `AnthropicLLMProvider` | Production / live mode |
| Replay LLM | `ReplayLLMProvider` | Offline mode; SHA256-keyed fixture playback |

`src/api/dependencies.py` is the **only** place SDK keys are read. Two new `Literal`-typed settings (`embedding_backend`, `llm_backend`) gate which concrete class is instantiated. A `model_validator` on `Settings` auto-detects missing or placeholder API keys and flips the backends to their offline variants without further configuration — so `git clone && make demo` Just Works on a fresh machine.

The `Chroma` collection name embeds the active embedder's identity (`rag_docs_voyage_voyage-3_1024`, `rag_docs_local_sentence-transformers-all-MiniLM-L6-v2_384`). The two embedders never collide; swapping backends never corrupts an existing index.

## Consequences

**Positive:**
- **Adding a vendor is one class, zero changes to retrieval or generation code.** OpenAI embeddings? Subclass that exposes `name`, `embedding_dim`, and two async methods. Add a branch in `dependencies.py`. Done.
- **Offline mode unlocks the portfolio.** A reviewer who clones the repo gets a working demo in under 2 minutes with no signup. The replay fixtures (recorded once by the owner) make the LLM responses *real* — not stubs, not "demo mode" placeholder text. The pipeline runs end-to-end with real retrieval, real citation verification on real chunks, and previously-recorded answers.
- **CI runs without secrets.** All 4 GitHub Actions jobs (lint, typecheck, test, docker-build) run in offline mode (`EMBEDDING_BACKEND=local LLM_BACKEND=replay`). No `VOYAGE_API_KEY` or `ANTHROPIC_API_KEY` in repo secrets for the standard CI path. The dispatchable eval workflow uses secrets but never runs automatically.
- **Eval and recording use the same abstraction.** The `_RecordingLLMProvider` wrapper in `scripts/build_replay_fixtures.py` is a thin decorator over `AnthropicLLMProvider` — structurally identical to the Protocol, captures every call. The replay file it writes is exactly what `ReplayLLMProvider` reads. The roundtrip is testable end-to-end.

**Negative:**
- **Replay drift.** If the corpus or pipeline parameters change after fixtures were recorded, the `(system, user, max_tokens)` SHA256 keys change and fixtures stop matching. Mitigation: `make replay-fixtures` regenerates idempotently; the fixture file records `git_commit` so drift is detectable.
- **Local embeddings have lower retrieval quality** than Voyage on this corpus. The offline-mode demo is honest about this — the UI banner says "answers replayed from a recorded fixture set" and the eval harness shows the gap when both backends are run.
- **`# type: ignore[arg-type]` on ChromaDB calls.** The SDK's strict typing rejects `list[list[float]]` even though it works at runtime — embeddings are accepted via numpy arrays too. The ignore comments are documented inline. Worth replacing with `numpy` arrays everywhere once the rest of the type drift in the SDK is addressed.

**Neutral:**
- `@runtime_checkable` Protocol over ABC is a deliberate Python choice. Structural typing keeps consumers decoupled from any inheritance hierarchy; `isinstance()` works in tests; mocking is trivial. No real downside on a small interface like this.
- `LLMProvider.complete` is intentionally minimal — `(system, user, max_tokens) -> str`. Streaming, tool use, structured output are out of scope. They'll be additive (new Protocol methods or new Protocols) when needed, not breaking changes.

## Alternatives considered

### A. Keep direct SDK calls; environment-gated branches inside each consumer
```python
class RAGGenerator:
    def __init__(self):
        if settings.offline_mode:
            self.client = ReplayClient(...)
        else:
            self.client = anthropic.AsyncAnthropic(...)
```
- **Why rejected:** that branch ends up in *every* consumer (verifier, scorer, generator). DI through a Protocol is one branch in `dependencies.py`. Plus the test mocking story is much worse — each consumer needs its own `monkeypatch`.

### B. `abc.ABC` base classes instead of `Protocol`
- **Why rejected:** ABC requires consumers to inherit from a class we own. Less flexible (third-party objects can't satisfy the interface). The duck-typing benefit of Protocol shines when wrapping vendor SDKs that we don't control.

### C. Real local LLM (TinyLlama, Phi-3-mini via transformers/llama-cpp)
- **Why rejected for offline mode:** the demo UX is much worse — 1–3 GB model download, slow on CPU (~30 s per answer), output quality noticeably worse than Claude. Replay fixtures give *the actual Claude response* the reviewer would have seen if they had paid for it. Considered "less real" — but the replay design is recognisable as a golden-snapshot test pattern, which is itself good engineering signal. Documented in the README banner so reviewers know what they're seeing.

### D. Hosted public demo (Fly.io / Render / Hugging Face Spaces)
- **Why rejected:** ~$5–15/mo recurring cost; one-line README impact for a service that already has a great cloneable story. The "clone and run in < 2 min" path is more impressive for the kind of reviewers this targets (engineering hiring managers and senior ICs), who want to see the code work locally rather than poke at a hosted instance.

## Implementation notes

- **Auto-detection logic** lives in `Settings._autodetect_backends` (model_validator). Explicit env-var values for `EMBEDDING_BACKEND` / `LLM_BACKEND` always win over auto-detection — important so the `rag-eval.yml` workflow can pin `voyage` / `anthropic` even when running on a CI runner with empty key vars.
- **Per-embedder Chroma collection naming:** `_safe_collection_name(provider_name, dim)` in `dependencies.py`. The transform `voyage:voyage-3 → rag_docs_voyage_voyage-3_1024` is one-way deterministic; an embedder dim mismatch fails fast at startup with a clear error.
- **Replay fixture key:** `SHA256(system + '\n' + user + '\n' + max_tokens)`. The full pipeline issues multiple LLM calls per question (generation + N citation verifications + completeness scoring), each with a unique `(system, user)` shape. Hashing the call shape rather than the user-facing question gives a stable key for every individual call without enumerating prompt templates.

## Validation

- All 89 unit tests pass with the abstraction in place; behavior is preserved end-to-end (verified by booting the API in live mode against the existing seeded index — see PR-1's verification section).
- CI's `test` job exercises the offline path (`EMBEDDING_BACKEND=local LLM_BACKEND=replay`) on every push — the abstraction proves itself there.
- A future "swap in OpenAI embeddings" PR will be the real test that the open/closed principle holds. The expectation: one new file in `src/providers/`, one new branch in `dependencies.py`, zero changes anywhere else.

## Future work / scope kept out

- **Streaming responses.** Would extend `LLMProvider` with a new method (`stream`). Additive; no breaking change to `complete`.
- **Tool use / structured output.** Same — additive Protocol methods.
- **Multi-tenant key isolation.** Per-request provider construction (rather than `lru_cache` singletons) once tenants exist. Not relevant for the current single-tenant scope.
- **Pinning `JUDGE_MODEL` separately from `LLM_MODEL`.** Mentioned in ADR-002 as future work; this ADR's abstraction makes it trivially supported once the config field exists.

## Links

- Implementation: [src/providers/](../../src/providers/), [src/api/dependencies.py](../../src/api/dependencies.py), [src/config.py](../../src/config.py).
- Recording infrastructure: [scripts/build_replay_fixtures.py](../../scripts/build_replay_fixtures.py).
- Related: ADR-001 (hybrid retrieval), ADR-002 (citation verification).
