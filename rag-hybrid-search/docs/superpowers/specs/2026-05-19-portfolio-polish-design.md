# Portfolio-Grade Polish — Design Spec

**Date:** 2026-05-19
**Project:** `rag-hybrid-search`
**Goal:** Transform the existing working RAG service into a portfolio piece that signals staff-level engineering to AI/ML, Backend/Platform, and Senior-generalist reviewers — while remaining runnable by anyone who clones the repo without paid API keys.

---

## 1. Context & Goals

### 1.1 What exists today

A working, well-structured RAG service: hybrid dense (Voyage) + sparse (BM25) retrieval, Reciprocal Rank Fusion, cross-encoder rerank, Claude generation with LLM-as-judge citation verification and composite confidence scoring. FastAPI + Streamlit + Docker Compose. Modular `src/` layout. `pyproject.toml` with ruff + mypy-strict configured. Tests scaffolded. A golden eval dataset stub exists.

### 1.2 What's missing

- No README (the single most-read artifact)
- Cannot run without paid API keys for Anthropic + Voyage
- No CI / no green badges
- No published eval results
- No ADRs / decision rationale
- No LICENSE
- No screenshots / demo media
- Provider coupling: vendor SDKs called directly from business logic

### 1.3 Success criteria

A reviewer who has never seen the repo can:
1. **Read the README in 90 seconds** and understand what it does, why it's interesting, and see real metrics.
2. **`git clone && make demo` in under 2 minutes** with zero API keys and see real answers with citations and confidence scores on real documents.
3. **Open the PR list** and see five atomic, well-described commits that tell a story of disciplined incremental delivery.
4. **Open `docs/adr/`** and see three ADRs explaining *why* the system is built this way and what was rejected.
5. **See green CI badges** for lint, type-check, tests, and coverage ≥80%.

### 1.4 Non-goals

- A hosted public demo (rejected; offline mode chosen instead)
- Production deployment infra (k8s, Terraform) — explicitly scoped out
- Local LLM inference (rejected; replay fixtures chosen instead — better UX, deterministic)
- UI polish beyond functional Streamlit (full-stack signal is not a target)
- Multi-tenancy, auth, rate-limiting hardening (mentioned in ADRs as future work; not built)

---

## 2. Design Principles

These principles are applied throughout and named explicitly in ADRs so reviewers see the intent, not just the artifact.

| Principle | Concrete application in this design |
|---|---|
| **Dependency inversion** | `Protocol`-based `EmbeddingProvider` and `LLMProvider`; pipeline depends on interfaces, not vendor SDKs |
| **Open/closed** | New providers (OpenAI, Cohere) require one new class and zero changes to retrieval / generation code |
| **Twelve-factor** | Config via env only; stateless processes; logs as structured event streams; port binding via uvicorn |
| **Idempotency** | Ingestion dedupes by content hash; `seed.py` is safe to re-run |
| **Graceful degradation** | Rerank failure → fall back to fusion-only ranking; LLM failure → return retrieved chunks + diagnostic |
| **Observability** | Per-stage latency timing; correlation ID propagated through every log line; structured (not formatted) logs |
| **Defense in depth** | Pydantic validation at API boundary; max chunk / query size limits; tenacity retries with exponential backoff on provider calls |
| **Forward-compatible storage** | Chroma collection name embeds embedder identity (`rag_docs_voyage_1024`, `rag_docs_local_384`) so swapping embedders never corrupts an index |
| **Determinism in tests** | Replay fixtures; seeded RNG; zero network in unit tests |
| **Single source of truth** | All tunables in `Settings`; no magic constants in business logic |
| **Test pyramid** | Unit (fast, deterministic) → integration (provider boundaries with mocks) → eval (system-level on golden set) |
| **Atomic reviewable PRs** | Five PRs, each independently mergeable, each with passing tests |
| **Cost awareness** | Eval doc publishes tokens/query; offline mode respects reviewer budgets; CI never spends money |
| **Type safety as a gate** | `mypy --strict` is enforced in CI, not just configured |

---

## 3. Architecture After Changes

```
┌──────────────────────────────────────────────────────────────────┐
│                       Streamlit UI (frontend/)                   │
└────────────────┬─────────────────────────────────────────────────┘
                 │ HTTP
┌────────────────▼─────────────────────────────────────────────────┐
│                   FastAPI (src/api/)                             │
│  /v1/ask  ·  /v1/ingest  ·  /v1/documents  ·  /health  ·  /docs  │
└────────────────┬─────────────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │   Pipeline      │
        │                 │
        │  Embedder  ────────► EmbeddingProvider (Protocol)
        │                 │       ├─ VoyageEmbeddingProvider
        │                 │       └─ LocalSentenceTransformerEmbeddingProvider
        │  Retriever      │
        │   ├─ Dense  ────────► ChromaDB (per-embedder collection)
        │   ├─ Sparse ────────► rank-bm25
        │   └─ RRF Fusion │
        │                 │
        │  Reranker  ─────────► sentence-transformers cross-encoder
        │                 │
        │  Generator ────────► LLMProvider (Protocol)
        │                 │       ├─ AnthropicLLMProvider
        │                 │       └─ ReplayLLMProvider (eval/replay_fixtures.json)
        │                 │
        │  Verifier  ────────► LLMProvider (same instance)
        │  Scorer    ────────► LLMProvider (same instance)
        └─────────────────┘
```

**Mode selection** happens once at startup in `src/api/dependencies.py`. Auto-detection rules:
- `VOYAGE_API_KEY` missing/placeholder → `embedding_backend = "local"`
- `ANTHROPIC_API_KEY` missing/placeholder → `llm_backend = "replay"`
- Explicit env vars override auto-detection
- Startup log states active mode loudly

---

## 4. Workstream — Five Atomic PRs

Each PR is independently reviewable; tests stay green at every PR boundary; the PR graph itself is a portfolio artifact.

### 4.1 PR-1: Provider abstraction (pure refactor)

**Estimate:** ~3 hours

**Changes:**
- New package `src/providers/` with:
  - `embedding.py` — `EmbeddingProvider` Protocol; `VoyageEmbeddingProvider` (moves logic from current `src/ingestion/embedder.py`)
  - `llm.py` — `LLMProvider` Protocol; `AnthropicLLMProvider` (extracts `messages.create` from `RAGGenerator`, `CitationVerifier`, `ConfidenceScorer`)
- `RAGGenerator`, `CitationVerifier`, `ConfidenceScorer` take an `LLMProvider` via constructor (no Anthropic client construction inside)
- `Embedder` (or its replacement) takes an `EmbeddingProvider`
- `src/api/dependencies.py` wires concrete providers based on config
- New config fields: `embedding_backend: Literal["voyage","local"]`, `llm_backend: Literal["anthropic","replay"]`
- Public API (FastAPI routes, response models) unchanged
- All existing tests pass without modification — proves the refactor is behavior-preserving

**Interface contract:**

```python
class EmbeddingProvider(Protocol):
    name: str            # used for Chroma collection naming
    embedding_dim: int   # validated against index on startup
    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class LLMProvider(Protocol):
    name: str
    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str: ...
```

`Protocol` over ABC is deliberate: structural typing keeps consumers decoupled from the inheritance hierarchy and makes testing/mocking trivial. ADR-003 captures this.

**Failure modes & handling:**
- Provider misconfiguration at startup → fail fast with clear error naming the missing config
- Embedding-dim mismatch between provider and existing index → fail fast with remediation hint (rebuild index)

**Verification:**
- `pytest -q` — all existing tests pass
- `mypy --strict src/` — passes
- Manual: `POST /v1/ask` returns identical response shape against existing seeded index

---

### 4.2 PR-2: Offline demo mode (no API keys required)

**Estimate:** ~5 hours

**Changes:**
- `LocalSentenceTransformerEmbeddingProvider` — wraps `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, CPU-friendly
- `ReplayLLMProvider` — loads `eval/replay_fixtures.json`, normalizes question (lowercase + collapsed whitespace), returns recorded `(answer, citations, confidence)`. On miss, returns a clear "this question isn't in the demo fixture set — try one of: [list]" response.
- `scripts/build_replay_fixtures.py` — runs real Voyage + Anthropic against a 12-question curated set over the Nexus corpus, captures complete `RAGResponse` for each, writes to `eval/replay_fixtures.json`. Run **once** by repo owner; output is committed to the repo.
- `config.py` auto-detection: missing API keys → corresponding backend defaults to local/replay; explicit env vars win
- Two Chroma collections side-by-side: `rag_docs_voyage_1024` and `rag_docs_local_384`. Seed script targets whichever matches the active embedder. Auto-seed when collection is empty on first run.
- Streamlit shows a clearly visible banner in demo mode listing the 12 curated questions
- `make demo` target: builds containers, seeds local-mode index from the bundled Nexus corpus, starts the stack, opens browser

**Curated demo question set** (12 questions, spanning all query types in the eval dataset, committed to `eval/demo_questions.json`):
- 4 factual (e.g., "What's the default rate limit?")
- 3 procedural (e.g., "How do I authenticate against the API?")
- 2 comparative (e.g., "What's the difference between webhook v1 and v2?")
- 3 error-lookup (e.g., "What does error code E2403 mean?")

**Failure modes & handling:**
- Replay miss → return diagnostic response with the available question list; not an error
- Fixture file missing/corrupted → fail fast at startup with clear remediation
- Local model first-run download fails → clear error with manual-download instructions

**Verification:**
- `make demo` on a fresh clone with no `.env` produces a working UI in < 2 minutes
- All 12 curated questions return non-error responses with citations
- Integration test in CI exercises the offline path end-to-end

---

### 4.3 PR-3: Eval harness + published results

**Estimate:** ~6 hours

**Changes:**
- Expand `eval/golden_dataset.json` to ~25 questions with schema:
  ```json
  {
    "id": "q001",
    "question": "...",
    "query_type": "factual|procedural|comparative|error-lookup",
    "relevant_chunk_ids": ["chunk_id_a", "chunk_id_b"],
    "expected_answer_facts": ["fact 1", "fact 2"]
  }
  ```
- `src/evaluation/metrics.py` implements:
  - Retrieval: hit@1, hit@3, hit@5, MRR@10
  - Generation: citation-coverage (fraction of factual claims with a verified citation), answer-completeness (LLM-judge fraction of `expected_answer_facts` present)
  - Latency: p50, p95 per stage (embed / retrieve / rerank / generate) + end-to-end
- `scripts/run_eval.py --config {dense-only,hybrid,hybrid+rerank,all}` writes:
  - Per-run JSON to `eval/results/<utc-timestamp>-<config>.json`
  - Markdown summary table to `docs/eval-results.md` (committed)
- `make eval` reproduces everything end-to-end against live providers

**Ablation matrix:**

| Config | Dense | Sparse (BM25) | Rerank |
|---|---|---|---|
| `dense-only` | ✓ | — | — |
| `hybrid` | ✓ | ✓ (RRF) | — |
| `hybrid+rerank` | ✓ | ✓ (RRF) | ✓ (cross-encoder) |

**Results doc structure:** headline table → per-query-type breakdown → latency table → cost-per-query table → reproducibility command → LLM-judge prompt (published for auditability).

**Failure modes & handling:**
- Provider rate-limit during eval → tenacity backoff; eval is restartable
- LLM-judge non-determinism → run each judge call 3× and majority-vote; document this in the results doc

**Verification:**
- `make eval` produces a complete `docs/eval-results.md` from scratch
- README hero table reflects the latest committed results
- Eval results show hybrid+rerank measurably outperforming dense-only (the whole point of the architecture)

---

### 4.4 PR-4: CI + quality gates

**Estimate:** ~3 hours

**Changes:**

`.github/workflows/ci.yml` — single workflow, 4 jobs running in parallel where independent:

1. **lint** — `ruff check .` + `ruff format --check .`
2. **typecheck** — `mypy --strict src/`
3. **test** — matrix on Python 3.11 + 3.12; `pytest --cov=src --cov-fail-under=80`; uploads coverage artifact. Runs in offline mode (no secrets needed) — relies on PR-2's replay fixtures.
4. **docker-build** — builds both Dockerfiles; runs `docker compose up -d` smoke test; curls `/health`; tears down.

`.github/workflows/eval.yml` — `workflow_dispatch` only (costs real money). Reads `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` from repo secrets. Documents reproduction.

`.pre-commit-config.yaml` — ruff (lint + format) + mypy. Optional for contributors but documents the discipline.

README badge row: CI status · coverage % · Python versions · license · ruff code style.

**Failure modes & handling:**
- Slow first-run pip install in CI → cache `uv` / pip wheels keyed by `pyproject.toml` hash
- Flaky tests → none expected (deterministic by design); if they appear, treat as bug, not retry

**Verification:**
- All five badges green on the main branch
- A trivial typo PR fails CI immediately

---

### 4.5 PR-5: README + ADRs + screenshots + LICENSE + Makefile

**Estimate:** ~5 hours — this is the headline artifact

**README structure** (top-down, scannable for 90-second readers):

```
# RAG Hybrid Search
> Production-grade RAG with hybrid retrieval, citation verification,
> and confidence scoring.

[badges: ci | coverage | python | license | ruff]

[hero GIF: docs/demo.gif — offline mode running]

## Why this exists
[2-3 sentences: LLMs hallucinate; citations matter; retrieval quality is the lever]

## Highlights
- Hybrid dense (Voyage) + sparse (BM25) with Reciprocal Rank Fusion
- Cross-encoder reranking for second-pass precision
- LLM-as-judge citation verification + composite confidence scoring
- Offline demo mode — no API keys required
- Reproducible eval harness with published ablation results

## Eval results
[headline table from docs/eval-results.md]
→ Full results: docs/eval-results.md  ·  Reproduce: `make eval`

## Architecture
[ASCII pipeline diagram from §3 above]
[brief paragraph]

## Quickstart
### Offline demo (no API keys, ~2 min)
- `git clone <repo> && cd rag-hybrid-search`
- `make demo`
- open http://localhost:8501

### Live mode (Anthropic + Voyage)
- `cp .env.example .env` and fill in keys
- `make up`

## Project layout
[tree of src/ with one-line purpose per module]

## Design decisions
- [ADR-001: Why hybrid retrieval](docs/adr/001-hybrid-retrieval.md)
- [ADR-002: LLM-as-judge for citation verification](docs/adr/002-citation-verification.md)
- [ADR-003: Provider abstraction & offline mode](docs/adr/003-provider-abstraction.md)

## Development
`make test` · `make lint` · `make typecheck` · `make eval`
Pre-commit hooks: `.pre-commit-config.yaml`
CI: `.github/workflows/ci.yml`

## License
MIT
```

**ADRs** in `docs/adr/`, Michael Nygard template (Context · Decision · Consequences · Alternatives Considered):

- **001-hybrid-retrieval.md** — why dense+sparse, why RRF, why a reranker; alternatives considered: HyDE, ColBERT, ensemble-of-rerankers; consequences populated from the measured ablation table in `docs/eval-results.md` (the MRR / latency deltas published there are the consequences cited here, by reference, so this ADR cannot drift from real numbers)
- **002-citation-verification.md** — why LLM-as-judge over rule-based citation extraction; failure modes (judge bias, cost, latency); how confidence composite is computed
- **003-provider-abstraction.md** — why Protocol over ABC; how offline mode works; what would change for production multi-tenant (per-tenant API key, per-tenant collection, audit log)

**Other artifacts:**
- `LICENSE` — MIT
- `CONTRIBUTING.md` — short: issue → PR flow, dev setup, link to CI
- `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`
- `docs/screenshots/` — 3 PNGs: ask flow, citation panel, confidence breakdown
- `docs/demo.gif` — 8-15s offline-mode capture (recorded once by repo owner)
- `Makefile` — `demo`, `up`, `down`, `test`, `lint`, `typecheck`, `eval`, `seed`, `replay-fixtures`
- `tasks.ps1` — PowerShell equivalent for Windows users without Make

**Verification:**
- A reviewer can complete the "Offline demo" Quickstart from a fresh clone on a clean machine in < 2 minutes
- All ADR links resolve; all images render in GitHub web UI
- `make help` lists every target with a description

---

## 5. Cross-Cutting Concerns

### 5.1 Observability

- Every request gets a UUID4 `request_id`; FastAPI middleware injects it into the `structlog` context for the duration of the request.
- Each pipeline stage logs `stage_start` / `stage_end` events with elapsed-ms.
- Provider calls log `provider_call` with `provider_name`, `latency_ms`, `tokens_in`, `tokens_out` where applicable.
- These logs are what the latency-per-stage eval metrics consume.

### 5.2 Error model

- Custom exceptions in `src/exceptions.py` (already exists) — `IngestionError`, `RetrievalError`, `GenerationError`, `ProviderError` (new).
- API layer maps these to HTTP 4xx/5xx with structured `{error_code, message, request_id}` JSON.
- Internal pipeline failures are logged at `error` level with the full stack trace; the API response carries the `request_id` for correlation.

### 5.3 Configuration validity

- Pydantic settings validates on import; the app refuses to start with an invalid config.
- Embedder-name vs Chroma-collection-name consistency is checked at first retrieval call; mismatch fails fast with remediation instructions.

### 5.4 Testing strategy

- **Unit tests** (`tests/unit/`) — pure functions, deterministic, no I/O. Fast (< 5s total). Cover: RRF math, dedup hashing, chunking strategies, citation parsing.
- **Integration tests** (`tests/integration/`) — exercise provider boundaries with stub providers. Cover: full `/v1/ask` flow in offline mode; ingestion idempotency; correlation-ID propagation.
- **Eval** (`scripts/run_eval.py`) — system-level on golden dataset. Not run in standard CI (costs money); run via `make eval` or `eval.yml` workflow_dispatch.
- Coverage target: ≥80% on `src/`, enforced in CI.

### 5.5 Security & secret hygiene

- `.env.example` documents required keys with placeholder values that are clearly invalid; `.gitignore` already excludes `.env`.
- Provider API keys never logged; structlog processors should mask any field matching `*_api_key`.
- CORS in production left as a documented TODO with a comment pointing at the unsafe `allow_origins=["*"]` line.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| sentence-transformers first-run model download fails offline | Medium | `make demo` pre-pulls the model in the Docker build step; document the manual fallback |
| Replay fixture file gets out of sync with Nexus corpus after a content edit | Low | `make replay-fixtures` regenerates; checksum of corpus stored alongside fixtures and validated at startup |
| CI matrix runs slow due to sentence-transformers cold install | Medium | Cache the model weights as a CI artifact keyed by version |
| Eval results doc drifts from current code | Medium | Eval JSON includes git commit SHA; results doc shows the SHA; README links latest committed run |
| LLM-judge non-determinism makes eval flaky | Medium | Majority-vote across 3 calls; published prompt in results doc; judge stability noted in ADR-002 |
| Scope creep into UI polish or production hardening | High | This spec is the gate. Anything not in §4 is explicitly out of scope. |

---

## 7. Out of Scope (documented as future work in ADRs)

- Hosted public demo
- Authentication, rate-limiting, multi-tenancy
- k8s / Terraform / production deployment
- Local LLM inference (transformers / llama.cpp)
- Full-stack UI redesign
- Streaming responses
- Conversation memory / multi-turn

---

## 8. Definition of Done

- [ ] All 5 PRs merged to `main`
- [ ] All CI badges green on `main`
- [ ] `git clone && make demo` works on a fresh machine in < 2 minutes (verified on Windows + macOS + Linux)
- [ ] README renders correctly on github.com with all images, badges, and ADR links resolving
- [ ] `docs/eval-results.md` shows real numbers for all 3 ablation configs
- [ ] All 3 ADRs published with Context / Decision / Consequences / Alternatives sections filled in
- [ ] LICENSE present (MIT)
- [ ] At least 3 screenshots + 1 demo GIF committed
- [ ] `mypy --strict src/` passes
- [ ] Test coverage ≥ 80% on `src/`
