# RAG Hybrid Search

> Production-grade Retrieval-Augmented Generation with hybrid dense + sparse retrieval, citation verification, and multi-dimensional confidence scoring.

A question-answering service over your own documents that returns **grounded answers with verified citations** and **explicit confidence scores** — not just plausible-sounding prose. Built to demonstrate the engineering decisions that separate a working RAG demo from a production-ready RAG service.

> **Status:** Functional scaffold. A five-PR portfolio-polish pass is in progress — see [Roadmap](#roadmap) and [docs/superpowers/specs/](docs/superpowers/specs/) for the design.

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
├── eval/golden_dataset.json  Eval questions + relevant-chunk labels
├── docs/superpowers/         Design specs and implementation plans
├── docker-compose.yml        api + frontend services
├── Dockerfile.api            FastAPI container
├── Dockerfile.frontend       Streamlit container
└── pyproject.toml            Deps + ruff + mypy-strict + pytest config
```

---

## Design decisions

These are summarized here; full ADRs land in PR-5 of the polish pass.

### Why hybrid (dense + sparse + rerank)?

Dense retrieval alone is weak on **exact-keyword** queries — error codes, API parameter names, version strings — because semantic similarity blurs them. BM25 alone is weak on **paraphrase** ("how do I sign in" vs "authentication setup"). RRF combines them without needing to calibrate score scales between methods. The cross-encoder rerank is a precision step: it sees query and candidate jointly (unlike bi-encoders) and reorders the top-20 to a top-5 with measurably better MRR. Each stage is independently ablatable — and PR-3 publishes the ablation table.

### Why LLM-as-judge for citation verification?

Rule-based citation extraction (regex match the chunk text in the answer) is brittle — paraphrased claims fail. LLM-as-judge with a focused prompt (claim + passage → SUPPORTED / UNSUPPORTED / PARTIAL) catches paraphrase and partial support. It costs a Claude call per citation, which is the right trade for production: the alternative is shipping confident-sounding hallucinated citations.

### Why composite confidence (3 dimensions, not 1)?

- **Retrieval confidence** alone catches "we had nothing relevant to say"
- **Citation coverage** alone catches "we hallucinated citations"
- **Answer completeness** alone catches "we partially answered"

Conflating them into one score loses the signal that tells you *what* went wrong. The API exposes all three plus a weighted composite (0.4 / 0.4 / 0.2) — clients can threshold on any dimension.

### Why `pydantic-settings` for config?

Twelve-factor: every tunable comes from the environment, every default is a typed Python value, validation happens at import time, and `--strict` mypy proves no settings access is mistyped.

---

## Development

```bash
uv run pytest -q              # run tests with coverage
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy --strict src/     # type-check (strict mode is enabled)
```

`pyproject.toml` configures all of these — coverage threshold, ruff rules (`E F I N UP B SIM`), and `mypy --strict` are pinned in one place.

---

## Roadmap

This project is being polished into a portfolio piece via a five-PR series. Full spec at [docs/superpowers/specs/2026-05-19-portfolio-polish-design.md](docs/superpowers/specs/2026-05-19-portfolio-polish-design.md).

| PR | Status | What it ships |
|---|---|---|
| **PR-1: Provider abstraction** | ✅ Open ([#1](https://github.com/BenettNR/professional-workspace/pull/1)) | `EmbeddingProvider` / `LLMProvider` Protocols; pipeline depends on interfaces, not vendor SDKs. [Plan](docs/superpowers/plans/2026-05-19-pr1-provider-abstraction.md) |
| **PR-2: Offline demo mode** | ✅ Open ([#2](https://github.com/BenettNR/professional-workspace/pull/2)) | Zero-API-key local mode: sentence-transformers embeddings + Claude replay fixtures. `make demo` runs in < 2 min on a fresh clone. [Plan](docs/superpowers/plans/2026-05-19-pr2-offline-demo-mode.md) |
| **PR-3: Eval harness + results** | 🚧 In progress | 53-question golden dataset, ablation harness (dense-only / hybrid / hybrid+rerank), latency p50/p95 per stage, `make eval` reproducibility. Results: [`docs/eval-results.md`](docs/eval-results.md). [Plan](docs/superpowers/plans/2026-05-19-pr3-eval-harness.md) |
| **PR-4: CI + quality gates** | Planned | GitHub Actions: ruff + mypy --strict + pytest (3.11 / 3.12 matrix) + docker build. Green badges in this README. |
| **PR-5: README + ADRs + screenshots** | Planned | Three Michael-Nygard ADRs, demo GIF, screenshots, finished README hero. |

---

## License

[MIT](LICENSE)
