# RAG Hybrid Search

[![CI](https://github.com/BenettNR/professional-workspace/actions/workflows/rag-ci.yml/badge.svg?branch=main)](https://github.com/BenettNR/professional-workspace/actions/workflows/rag-ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy --strict](https://img.shields.io/badge/mypy-strict-blue)](pyproject.toml)

> Production-grade Retrieval-Augmented Generation with hybrid dense + sparse retrieval, citation verification, and multi-dimensional confidence scoring.

A question-answering service over your own documents that returns **grounded answers with verified citations** and **explicit confidence scores** — not just plausible-sounding prose. Built to demonstrate the engineering decisions that separate a working RAG demo from a production-ready RAG service.

**Design decisions are written down.** Every non-obvious choice has an [ADR](docs/adr/) explaining what we considered and why we picked this. The [eval harness](docs/eval-results.md) proves the architecture isn't cargo-cult — every stage of the pipeline is independently ablatable and measurably justified.

---

## Why this exists

LLMs hallucinate. Citations matter. **Retrieval quality is the lever**: a 5% improvement in what you put in the context window beats a much larger investment in the LLM itself. This project is a deliberate study in how to build that lever — and then prove it works with a published ablation table.

Three failure modes RAG systems most often ship with, and how this addresses them:

| Failure mode | How this handles it |
|---|---|
| **Plausible answer, irrelevant citations** | `CitationVerifier` runs LLM-as-judge over every `[N]` marker against the cited chunk. Unverified citations are flagged in the response. |
| **Confident answer when retrieval missed** | `ConfidenceScorer` computes a composite of retrieval confidence, citation coverage, and answer completeness. Below threshold, the response is marked `insufficient_info`. |
| **Pure-semantic retrieval fails on exact-keyword queries** | Hybrid retrieval combines dense (Voyage) + sparse (BM25) via Reciprocal Rank Fusion; a cross-encoder reranker reorders the top-K candidates for second-pass precision. |

---

## Eval results

The pipeline ships with a reproducible ablation harness that compares **dense-only**, **hybrid** (dense + sparse + RRF), and **hybrid+rerank** configurations on a 53-question golden dataset over the Nexus API corpus. Metrics: `hit@1/3/5`, `MRR@10`, per-stage latency (p50/p95), plus LLM-as-judge dimensions when keys are available.

→ Full table: [`docs/eval-results.md`](docs/eval-results.md) — `make eval` to regenerate.

## Architecture

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
        ┌────────▼────────────────────────────────────┐
        │              Ingestion pipeline             │
        │  load → chunk → embed → dedupe → index      │
        └────────┬────────────────────────────────────┘
                 │
        ┌────────▼────────────────────────────────────┐
        │              Retrieval pipeline             │
        │                                             │
        │   ┌─── Dense (Voyage → ChromaDB) ───┐       │
        │   │                                 │       │
        │   │   Reciprocal Rank Fusion ──→ Cross-encoder rerank
        │   │                                 │       │
        │   └─── Sparse (BM25) ──────────────┘       │
        └────────┬────────────────────────────────────┘
                 │
        ┌────────▼────────────────────────────────────┐
        │              Generation pipeline            │
        │   Claude → CitationVerifier → ConfidenceScorer
        └─────────────────────────────────────────────┘
```

Each stage is in its own package with one responsibility:

- [src/ingestion/](src/ingestion/) — loaders, chunkers (fixed / recursive / semantic), embedder, deduplication, indexer
- [src/retrieval/](src/retrieval/) — dense retriever, sparse retriever, RRF fusion, cross-encoder reranker
- [src/generation/](src/generation/) — grounded generator, citation verifier, confidence scorer
- [src/evaluation/](src/evaluation/) — golden-dataset eval harness (under construction)
- [src/api/](src/api/) — FastAPI routes, Pydantic models, dependency injection

---

## Highlights

- **Hybrid retrieval** — dense semantic vectors via [Voyage AI](https://voyageai.com) (Anthropic's recommended embedding partner) + BM25 sparse keyword retrieval, fused with Reciprocal Rank Fusion (k=60, the standard smoothing constant).
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reorders the top-20 fused candidates down to a top-5 context window, trading a 200ms latency hit for a meaningful MRR bump.
- **Asymmetric embeddings** — documents and queries are embedded with different `input_type` values, which Voyage's bi-encoders use to improve retrieval precision.
- **Citation verification** — every `[N]` citation in the answer is fact-checked against the cited chunk using Claude as LLM-as-judge. Verdict and reason are returned with the response.
- **Composite confidence** — three dimensions (retrieval confidence, citation coverage, answer completeness) blended into a single score the API exposes alongside its components.
- **Pluggable chunking** — fixed-size, recursive (default), or semantic (embedding-distance breakpoint). Configured per-request or globally.
- **Idempotent ingestion** — duplicate detection by cosine-similarity threshold against existing chunks; re-running `seed.py` is safe.
- **Structured logging** — `structlog` with JSON output, per-stage timing, and key=value event fields throughout the pipeline.
- **Docker Compose** — one command spins up the API + Streamlit UI with shared volumes for persistent indexes.

---

## Quickstart

### Offline demo (no API keys, ~2 min) ⭐

```bash
git clone https://github.com/BenettNR/professional-workspace.git
cd professional-workspace/rag-hybrid-search
make demo                      # or: ./tasks.ps1 demo   (Windows)
```

The first run downloads the local embedding model (~80MB) and builds the local index from the bundled Nexus API documentation. After that:
- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs
- Health check → http://localhost:8000/health

In offline mode, Claude responses are **replayed from a recorded fixture set** for the 12 curated demo questions — the UI banner lists them. To regenerate fixtures with fresh Claude responses, set keys and run `make replay-fixtures` (one-time, ~$1).

### Live mode (real Anthropic + Voyage)

```bash
cp .env.example .env           # fill in VOYAGE_API_KEY and ANTHROPIC_API_KEY
make up                        # or: ./tasks.ps1 up
uv run python scripts/seed.py  # index the bundled corpus with real embeddings
```

### Run locally without Docker

```bash
uv venv && uv pip install -e ".[dev]"   # install deps into .venv
# Offline mode (no .env needed):
EMBEDDING_BACKEND=local LLM_BACKEND=replay uv run uvicorn src.api.main:app --reload --port 8000
# In another terminal:
EMBEDDING_BACKEND=local LLM_BACKEND=replay uv run streamlit run frontend/app.py
```

### Requirements

- Python 3.11+
- Docker Desktop (for `make demo` / `make up`)
- API keys (optional — only needed for live mode): [Anthropic](https://console.anthropic.com/) and [Voyage AI](https://dash.voyageai.com)
- Windows users without GNU Make: use `./tasks.ps1` instead of `make`

### Try a query

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the default rate limit for the standard tier?"}'
```

You'll get a JSON response with the answer, numbered citations (each with a verification verdict and reason), per-chunk retrieval scores at each pipeline stage (dense, sparse, fusion, rerank), and a composite confidence score with its three component dimensions.

---

## Configuration

All settings are in [.env.example](.env.example). Tunables you'll touch most:

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `voyage-3` | Also: `voyage-3-lite`, `voyage-code-3` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Any Claude model name |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | `0.7` / `0.3` | RRF weighting |
| `RERANK_TOP_K` | `5` | Chunks sent to the generator |
| `DEFAULT_CHUNK_STRATEGY` | `recursive` | Also: `fixed`, `semantic` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | Character counts |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.3` | Below this → `insufficient_info: true` |

---

## Project layout

```
rag-hybrid-search/
├── src/
│   ├── api/                  FastAPI app + routes + DI
│   ├── ingestion/            load → chunk → embed → dedupe → index
│   ├── retrieval/            dense + sparse + RRF + rerank
│   ├── generation/           Claude → verify citations → score confidence
│   ├── evaluation/           golden-dataset metrics (in progress)
│   ├── config.py             pydantic-settings (all tunables here)
│   ├── models.py             shared domain dataclasses
│   └── exceptions.py         RAG-specific exception hierarchy
├── frontend/app.py           Streamlit UI
├── scripts/seed.py           Seed the index from data/raw/
├── tests/                    pytest, mypy --strict configured
├── data/
│   ├── raw/                  Demo corpus (Nexus API docs, fictional)
│   └── chroma/               Persistent vector index (gitignored)
├── eval/                     Golden dataset, demo questions, replay fixtures
│   ├── golden_dataset.json
│   ├── demo_questions.json   12 curated questions for offline-mode replay
│   └── replay_fixtures.json  Recorded Claude responses (built once)
├── docs/
│   ├── adr/                  Architecture decision records (ADR-001..003)
│   ├── eval-results.md       Published ablation results
│   └── superpowers/          Design specs and implementation plans
├── .github/
│   ├── workflows/            CI (push/PR) + eval (workflow_dispatch)
│   └── ISSUE_TEMPLATE/
├── docker-compose.yml        api + frontend (live mode)
├── docker-compose.demo.yml   overlay (offline mode)
├── Dockerfile.api            FastAPI container
├── Dockerfile.frontend       Streamlit container
├── Makefile                  demo, up, test, lint, typecheck, eval, …
├── tasks.ps1                 PowerShell equivalent for Windows
├── pyproject.toml            Deps + ruff + mypy-strict + pytest config
└── .pre-commit-config.yaml   Local hooks mirroring CI gates
```

---

## Design decisions

Full ADRs live in [`docs/adr/`](docs/adr/) following the Michael Nygard template (Context → Decision → Consequences → Alternatives).

- **[ADR-001: Hybrid retrieval (dense + sparse + rerank)](docs/adr/001-hybrid-retrieval.md)** — Why dense alone fails on exact-keyword queries; why BM25 alone fails on paraphrase; why RRF + cross-encoder rerank is the right composition; alternatives considered (HyDE, ColBERT, ensemble rerankers) and why they were rejected.
- **[ADR-002: LLM-as-judge for citation verification](docs/adr/002-citation-verification.md)** — Why per-citation verification matters more than composite confidence alone; the failure modes it catches (paraphrased claim with related chunk, multi-chunk drift, fabricated specificity); the cost/latency trade and the alternatives (rule-based, NLI, self-consistency).
- **[ADR-003: Provider abstraction + offline demo mode](docs/adr/003-provider-abstraction.md)** — Why `Protocol` over `ABC`; how the offline mode (`sentence-transformers` + replay fixtures) lets reviewers run the full pipeline with zero API keys; the per-embedder Chroma collection naming that prevents index corruption when backends swap.

A condensed version of why composite confidence is three-dimensional, in case you skip the ADRs:

- **Retrieval confidence** alone catches "we had nothing relevant to say".
- **Citation coverage** alone catches "we hallucinated citations".
- **Answer completeness** alone catches "we partially answered".

Conflating them into one score loses the signal that tells you *what* went wrong. The API exposes all three plus a weighted composite (0.4 / 0.4 / 0.2) — clients can threshold on any dimension.

---

## Development

```bash
make test       # pytest with coverage
make lint       # ruff check + format check
make typecheck  # mypy --strict
make check      # all three above
```

`pyproject.toml` configures everything — ruff rules (`E F I N UP B SIM`), `mypy --strict`, pytest, coverage — in one place.

### Pre-commit hooks (optional, recommended)

```bash
uv pip install pre-commit
pre-commit install
```

Hooks match CI's gates exactly: ruff lint + format, mypy strict on `src/`, basic file hygiene. Failures show up locally instead of in CI.

### CI gates

GitHub Actions runs on every push/PR touching this project:
- `lint` — `ruff check` + `ruff format --check`
- `typecheck` — `mypy --strict src/`
- `test` — `pytest` on Python 3.11 and 3.12, coverage gated
- `docker-build` — `docker compose build` + `/health` smoke test

The full eval (with real Claude calls) lives in a separate `workflow_dispatch`-only workflow so it never runs automatically (and never bills the repo owner).

---

## Roadmap

This project is being polished into a portfolio piece via a five-PR series. Full spec at [docs/superpowers/specs/2026-05-19-portfolio-polish-design.md](docs/superpowers/specs/2026-05-19-portfolio-polish-design.md).

| PR | Status | What it ships |
|---|---|---|
| **PR-1: Provider abstraction** | ✅ Open ([#1](https://github.com/BenettNR/professional-workspace/pull/1)) | `EmbeddingProvider` / `LLMProvider` Protocols; pipeline depends on interfaces, not vendor SDKs. [Plan](docs/superpowers/plans/2026-05-19-pr1-provider-abstraction.md) |
| **PR-2: Offline demo mode** | ✅ Open ([#2](https://github.com/BenettNR/professional-workspace/pull/2)) | Zero-API-key local mode: sentence-transformers embeddings + Claude replay fixtures. `make demo` runs in < 2 min on a fresh clone. [Plan](docs/superpowers/plans/2026-05-19-pr2-offline-demo-mode.md) |
| **PR-3: Eval harness + results** | ✅ Open ([#3](https://github.com/BenettNR/professional-workspace/pull/3)) | 53-question golden dataset, ablation harness (dense-only / hybrid / hybrid+rerank), latency p50/p95 per stage, `make eval` reproducibility. Results: [`docs/eval-results.md`](docs/eval-results.md). [Plan](docs/superpowers/plans/2026-05-19-pr3-eval-harness.md) |
| **PR-4: CI + quality gates** | ✅ Open ([#4](https://github.com/BenettNR/professional-workspace/pull/4)) | GitHub Actions: ruff + mypy --strict + pytest (3.11 / 3.12 matrix) + docker build. Green badges above. |
| **PR-5: README + ADRs + screenshots** | 🚧 In progress | Three Michael-Nygard ADRs, demo GIF, screenshots, finished README hero. [Plan](#) |

---

## License

[MIT](LICENSE)
