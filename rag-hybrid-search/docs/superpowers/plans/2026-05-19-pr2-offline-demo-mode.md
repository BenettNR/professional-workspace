# PR-2: Offline Demo Mode — Implementation Plan

> **Stacks on PR-1.** Branch: `feat/rag-offline-mode` off `feat/rag-provider-abstraction`.

**Goal:** Zero-API-key local mode. A reviewer can `git clone && make demo` and see the app working in under 2 minutes with no signup, no payment, no environment configuration.

**Architecture:** Add `LocalSentenceTransformerEmbeddingProvider` (sentence-transformers `all-MiniLM-L6-v2`, 384-dim, CPU) and `ReplayLLMProvider` (fixture playback). Auto-detect missing API keys at config load and switch backends. Use per-embedder Chroma collection names so the two backends never corrupt each other's indexes.

**Tech stack:** sentence-transformers (already in deps), pure stdlib (json, hashlib) for replay. No new dependencies.

---

## Scope decision: code-only vs code + fixtures

This PR ships **all the code** for offline mode plus an **empty fixtures file** with the curated question list. The actual replay fixtures (recorded Claude responses) are committed in a tiny follow-up PR after the repo owner runs `make replay-fixtures` (which requires their real API keys, one-time, ~$1 spend). This keeps PR-2 self-contained as code review.

## File Structure

**New files:**
- `src/providers/embedding_local.py` — `LocalSentenceTransformerEmbeddingProvider`
- `src/providers/llm_replay.py` — `ReplayLLMProvider`
- `scripts/build_replay_fixtures.py` — recorder script
- `eval/demo_questions.json` — 12 curated questions over the Nexus corpus
- `eval/replay_fixtures.json` — initially `{}`; populated by repo owner via `make replay-fixtures`
- `tests/unit/test_providers_embedding_local.py`
- `tests/unit/test_providers_llm_replay.py`
- `Makefile` — top-level convenience targets
- `tasks.ps1` — PowerShell equivalent of the Makefile for Windows
- `docker-compose.demo.yml` — overlay with `EMBEDDING_BACKEND=local`, `LLM_BACKEND=replay`

**Modified files:**
- `src/providers/__init__.py` — re-export the new classes
- `src/config.py` — auto-detect missing API keys; per-collection naming
- `src/api/dependencies.py` — wire `local` / `replay` branches; use embedder-derived collection name
- `frontend/app.py` — demo-mode banner

---

## Tasks

### Task 1: Curated demo questions

- [ ] Create `eval/demo_questions.json` with 12 questions over the Nexus corpus, evenly distributed across `factual / procedural / comparative / error-lookup`. Schema:
  ```json
  [
    {"id": "q01", "question": "What is the default rate limit for the standard tier?", "query_type": "factual"},
    ...
  ]
  ```
- [ ] Create `eval/replay_fixtures.json` as `{}` (empty seed; populated later by `make replay-fixtures`).
- [ ] Commit: `feat(eval): seed curated demo questions for offline mode`

### Task 2: `LocalSentenceTransformerEmbeddingProvider`

- [ ] Create `src/providers/embedding_local.py` with the class. Use `asyncio.to_thread` to wrap sentence-transformers' synchronous `encode` so callers stay async. Model is lazy-loaded on first call (avoids 80MB import-time cost in CI).
- [ ] Tests in `tests/unit/test_providers_embedding_local.py`: protocol conformance, dim is 384, lazy load happens exactly once, embed_query and embed_documents return the right shape.
- [ ] `uv run pytest tests/unit/test_providers_embedding_local.py -v`
- [ ] `uv run mypy --strict src/providers/embedding_local.py`
- [ ] Commit: `feat(providers): add LocalSentenceTransformerEmbeddingProvider for offline mode`

### Task 3: `ReplayLLMProvider`

- [ ] Create `src/providers/llm_replay.py`. Looks up `(system, user)` → answer by SHA256 hash; on miss returns a clearly-labeled diagnostic response naming the available demo questions. Loads the JSON file at init and re-loads on file mtime change so `make replay-fixtures` updates are picked up without restart.
- [ ] Tests: protocol conformance, hash match returns recorded text, hash miss returns diagnostic, file not found raises ProviderError at init.
- [ ] `uv run pytest tests/unit/test_providers_llm_replay.py -v`
- [ ] `uv run mypy --strict src/providers/llm_replay.py`
- [ ] Commit: `feat(providers): add ReplayLLMProvider for offline mode`

### Task 4: `scripts/build_replay_fixtures.py`

- [ ] Build a `_RecordingLLMProvider` that wraps `AnthropicLLMProvider` and captures every `(system, user, max_tokens, response)` tuple. For each question in `demo_questions.json`, run the full pipeline (real Voyage embeddings + Anthropic generation + verification + scoring), then dump the recorder's captured map to `eval/replay_fixtures.json`.
- [ ] Idempotent — re-running overwrites the file from scratch.
- [ ] Print cost summary at end (count of LLM calls × estimated tokens).
- [ ] Don't run this as part of CI — it costs real money. Just `make replay-fixtures` invokes it manually.
- [ ] Commit: `feat(scripts): add build_replay_fixtures.py recorder`

### Task 5: Auto-detect missing API keys in config

- [ ] In `src/config.py`, change `voyage_api_key` and `anthropic_api_key` from required (`Field(...)`) to optional (`Field(default="")`). Add a `model_validator(mode="after")` that flips `embedding_backend` to `"local"` when `voyage_api_key` is empty or starts with the placeholder (`pa-...`), same for `llm_backend` → `"replay"` when `anthropic_api_key` is empty or starts with `sk-ant-`. Explicit env-var values for `*_backend` always win.
- [ ] Log the active backends loudly at startup (already wired in `src/api/main.py`).
- [ ] Tests in `tests/unit/test_config_autodetect.py` covering all four combinations of (key-present, key-empty, key-placeholder) × (explicit-backend-set, not-set).
- [ ] Commit: `feat(config): auto-detect missing API keys and switch to offline backends`

### Task 6: Per-embedder Chroma collection naming

- [ ] In `src/api/dependencies.py`, change `get_chroma_collection()` to derive the collection name from the active embedder: `f"rag_docs_{embedder.name.replace(':','_')}_{embedder.embedding_dim}"`. This produces `rag_docs_voyage_voyage-3_1024` and `rag_docs_local_sentence-transformers-all-MiniLM-L6-v2_384` as parallel collections that never corrupt each other.
- [ ] Also wire `get_llm_provider()` to instantiate `ReplayLLMProvider` when `settings.llm_backend == "replay"`.
- [ ] Verify: with `ANTHROPIC_API_KEY=test VOYAGE_API_KEY=test uv run python -c "from src.api.dependencies import get_embedding_provider, get_llm_provider; print(get_embedding_provider().name, get_llm_provider().name)"` → prints both `voyage:voyage-3` and `anthropic:claude-sonnet-4-6`.
- [ ] With `EMBEDDING_BACKEND=local LLM_BACKEND=replay uv run python -c "..."` → prints `local:...` and `replay:...`.
- [ ] Commit: `feat(api): per-embedder Chroma collection + replay backend wiring`

### Task 7: Streamlit demo-mode banner

- [ ] In `frontend/app.py`, at app startup read settings.llm_backend and settings.embedding_backend. If either is non-default, render a visible banner:
  ```
  🟡 Demo mode (offline): answers replayed from a fixture set.
  Available questions: [collapsible list of 12]
  Add VOYAGE_API_KEY and ANTHROPIC_API_KEY to .env for live mode.
  ```
- [ ] Commit: `feat(frontend): show banner when running in offline demo mode`

### Task 8: Makefile + `tasks.ps1`

- [ ] Create `Makefile` with targets: `help` (default, lists targets), `demo` (offline mode docker-compose up), `up` (live mode), `down`, `test`, `lint`, `typecheck`, `seed`, `replay-fixtures`, `clean` (wipe `data/chroma`, `data/bm25_index.pkl`).
- [ ] Create `tasks.ps1` with the same logical targets for Windows users without GNU Make. Invocation: `./tasks.ps1 demo`.
- [ ] Create `docker-compose.demo.yml` overlay setting `EMBEDDING_BACKEND=local` and `LLM_BACKEND=replay` and removing the API key requirements.
- [ ] Commit: `feat(build): Makefile, tasks.ps1, docker-compose.demo.yml for one-command demo`

### Task 9: README update

- [ ] Update [README.md](../../README.md) Quickstart: lead with `make demo` (offline), then `make up` (live mode with keys). Mark PR-2 as done in the Roadmap table.
- [ ] Commit: `docs(readme): promote offline demo to lead Quickstart now that PR-2 ships`

### Task 10: Full verification + PR description

- [ ] `uv run pytest --no-cov --deselect tests/test_ingestion.py::TestFixedSizeChunker::test_splits_long_text` — all green
- [ ] `uv run mypy --strict src/providers/ src/generation/ src/api/ src/config.py src/exceptions.py src/ingestion/__init__.py` — clean
- [ ] `uv run ruff check` on PR-2 files — clean
- [ ] **Manual smoke**: with no `.env`, `EMBEDDING_BACKEND=local LLM_BACKEND=replay uv run uvicorn src.api.main:app` boots without error and `/health` returns 200.
- [ ] **`make demo`** dry-run end-to-end (compose up, hit a question, see banner) — note this requires fixtures, so document the "first time: run `make replay-fixtures` with keys" step in the PR description.
- [ ] Push branch and open PR-2.
