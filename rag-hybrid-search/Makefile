# Makefile for rag-hybrid-search.
#
# Most targets work on Mac/Linux out of the box. On Windows, install GNU Make:
#   winget install GnuWin32.Make
# or use the equivalent `./tasks.ps1` PowerShell script.
#
# Targets are grouped: dev (typecheck/lint/test), demo (offline), live (prod).

.DEFAULT_GOAL := help

PYTHON ?= uv run python
UV ?= uv

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help (default target)
	@echo "rag-hybrid-search — targets:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Quickstart:"
	@echo "  make demo     # offline, no API keys, ~2 min"
	@echo "  make up       # live mode, requires .env with API keys"

# ── Demo (offline mode, no API keys) ──────────────────────────────────────────

.PHONY: demo
demo: ## Run the app in offline demo mode (no API keys required)
	docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build

.PHONY: demo-down
demo-down: ## Stop the demo containers
	docker compose -f docker-compose.yml -f docker-compose.demo.yml down

# ── Live mode (requires real API keys in .env) ────────────────────────────────

.PHONY: up
up: ## Run the app in live mode (needs .env with VOYAGE/ANTHROPIC keys)
	docker compose up --build

.PHONY: down
down: ## Stop the live containers
	docker compose down

# ── Local development ─────────────────────────────────────────────────────────

.PHONY: install
install: ## Install all dependencies into .venv
	$(UV) pip install -e ".[dev]"

.PHONY: test
test: ## Run the test suite
	$(UV) run pytest

.PHONY: lint
lint: ## Run ruff lint check
	$(UV) run ruff check .

.PHONY: format
format: ## Run ruff format
	$(UV) run ruff format .

.PHONY: typecheck
typecheck: ## Run mypy --strict on the src tree
	$(UV) run mypy --strict src/

.PHONY: check
check: lint typecheck test ## Run all quality gates (lint + typecheck + test)

# ── Indexing and demo data ────────────────────────────────────────────────────

.PHONY: seed
seed: ## Seed the index from data/raw/ (uses active embedder backend)
	$(PYTHON) scripts/seed.py

.PHONY: replay-fixtures
replay-fixtures: ## Record Claude responses over demo questions (needs real keys)
	$(PYTHON) scripts/build_replay_fixtures.py

.PHONY: eval
eval: ## Run the full ablation eval (dense-only, hybrid, hybrid+rerank)
	$(PYTHON) scripts/run_eval.py

.PHONY: eval-fast
eval-fast: ## Run eval without LLM-as-judge (retrieval + latency only)
	$(PYTHON) scripts/run_eval.py --no-llm-judge

.PHONY: clean
clean: ## Wipe the persistent indexes (forces a re-seed next run)
	rm -rf data/chroma data/bm25_index.pkl

.PHONY: clean-all
clean-all: clean ## Wipe indexes + drop the venv
	rm -rf .venv .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
