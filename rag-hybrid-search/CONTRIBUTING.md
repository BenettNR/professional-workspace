# Contributing to rag-hybrid-search

Thanks for taking a look. This project's primary purpose is portfolio demonstration, but pull requests and issues are genuinely welcome — especially ones that catch design errors or surface failure modes the eval harness misses.

## Quickstart for contributors

```bash
git clone https://github.com/BenettNR/professional-workspace.git
cd professional-workspace/rag-hybrid-search
uv venv && uv pip install -e ".[dev]"
pre-commit install                    # optional but recommended
make check                            # lint + typecheck + test in one shot
```

You don't need API keys to develop — offline mode runs the full pipeline against a recorded fixture set. Live mode (real Voyage + Anthropic) is one `.env` away when you want it.

## Workflow

1. **Open an issue first** for non-trivial changes. Five minutes of discussion can save an afternoon of code.
2. **Branch from `main`**, name it `feat/<short-slug>` or `fix/<short-slug>`.
3. **Run `make check` locally before pushing.** CI runs the same gates; failing fast is faster.
4. **One logical change per PR.** The five-PR polish series ([spec](docs/superpowers/specs/2026-05-19-portfolio-polish-design.md)) is the reference for what that looks like in practice.
5. **Write the commit message.** Why over what. Imperative mood. First line ≤ 72 chars.

## Code style

The toolchain enforces everything:

| Concern | Tool | Config |
|---|---|---|
| Lint + format | `ruff` | [pyproject.toml](pyproject.toml) — selects `E F I N UP B SIM` |
| Type-check | `mypy --strict` | [pyproject.toml](pyproject.toml) — Python 3.11+, no untyped defs |
| Test | `pytest` + `pytest-asyncio` + `pytest-cov` | [pyproject.toml](pyproject.toml) — coverage gated at 45% |
| Pre-commit | hooks mirror CI exactly | [.pre-commit-config.yaml](.pre-commit-config.yaml) |

Disagreements with the toolchain go through `pyproject.toml`, not by silencing individual warnings. Inline `# type: ignore` and `# noqa` need a *one-line comment explaining why* — see the existing examples in `src/retrieval/dense.py` and `src/ingestion/indexer.py` for the standard.

## Architectural decisions

Non-trivial design choices live in [`docs/adr/`](docs/adr/) following the Michael Nygard template. If your PR makes a non-obvious trade-off, write an ADR alongside the code change. ADRs are short — a few paragraphs of *context → decision → consequences → alternatives* — and they're the most underrated way to make a PR easy to review.

## Tests

- **Unit tests** (`tests/unit/`) are pure, deterministic, fast. No network. No real models. Mock the SDK boundary.
- **Module tests** (`tests/test_*.py`) test internal pure functions and behaviour without external dependencies.
- **Eval** (`scripts/run_eval.py`) is the system-level integration test. It costs money in live mode; CI doesn't run it. Use it manually via `make eval` when changes might affect retrieval quality.

When a bug is fixed, **add a regression test** before the fix lands. The test should fail on the broken commit and pass on the fix.

## What's in scope

This project is *deliberately* a small, focused RAG service over a single corpus. Issues and PRs that fit the shape:
- Retrieval quality improvements with measurable eval deltas
- New EmbeddingProvider / LLMProvider implementations
- Better confidence scoring, citation verification accuracy
- Performance, observability, docs

## What's out of scope

- Multi-tenancy / authentication — explicitly deferred in the spec
- Streaming responses — additive but not currently needed
- A web UI redesign — Streamlit is intentionally minimal here
- Pricing / hosting features

## Reporting issues

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) — it asks for the minimum a maintainer needs to reproduce. Include `EMBEDDING_BACKEND`, `LLM_BACKEND`, and the commit SHA you're seeing the issue on. Logs from `structlog` (JSON output) are more useful than screenshots.

## Security

If you find a security issue, please email rather than opening a public issue. The README's listed contact is the right place.

## License

MIT — same as the rest of the repo.
