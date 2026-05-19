"""Application settings.

Loaded from environment variables (or `.env` via pydantic-settings).

Auto-detection: if VOYAGE_API_KEY or ANTHROPIC_API_KEY is missing or is
still set to the .env.example placeholder, the corresponding backend
auto-switches to its offline variant (`local` for embeddings, `replay`
for LLM). An explicit EMBEDDING_BACKEND / LLM_BACKEND env var always wins.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_placeholder(key: str) -> bool:
    """Return True if a key value is missing or still the .env.example placeholder."""
    if not key:
        return True
    return key.startswith("pa-...") or key.startswith("sk-ant-...")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Voyage AI (embeddings) ────────────────────────────────────────────────
    voyage_api_key: str = Field(default="", description="Voyage AI API key (empty → local backend)")
    embedding_model: str = "voyage-3"
    embedding_batch_size: int = 128  # Voyage AI per-request limit

    # ── Anthropic (generation + LLM-as-judge) ─────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key (empty → replay backend)")
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048

    # ── Backend selection ─────────────────────────────────────────────────────
    # 'voyage' uses the Voyage SDK; 'local' uses sentence-transformers.
    embedding_backend: Literal["voyage", "local"] = "voyage"
    # 'anthropic' uses Claude; 'replay' uses fixture playback.
    llm_backend: Literal["anthropic", "replay"] = "anthropic"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_directory: str = "data/chroma"
    chroma_collection_name: str = "rag_documents"

    # ── BM25 ──────────────────────────────────────────────────────────────────
    bm25_index_path: str = "data/bm25_index.pkl"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    dense_top_k: int = 10
    sparse_top_k: int = 10
    fusion_top_k: int = 20   # candidates sent to reranker
    rerank_top_k: int = 5    # final chunks used for generation
    rrf_k: int = 60          # RRF smoothing constant (standard default)
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    dedup_similarity_threshold: float = 0.95

    # ── Chunking ──────────────────────────────────────────────────────────────
    default_chunk_strategy: str = "recursive"  # fixed | recursive | semantic
    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_breakpoint_threshold: float = 0.3

    # ── Generation ────────────────────────────────────────────────────────────
    retrieval_confidence_threshold: float = 0.3

    # ── Reranking ─────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_llm_reranker: bool = False

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # ── Demo/offline mode ─────────────────────────────────────────────────────
    replay_fixtures_path: str = "eval/replay_fixtures.json"
    demo_questions_path: str = "eval/demo_questions.json"

    # ── Data paths ────────────────────────────────────────────────────────────
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    @model_validator(mode="after")
    def _autodetect_backends(self) -> "Settings":
        """Default backends to offline variants if API keys are missing/placeholder.

        Explicit env-var values for EMBEDDING_BACKEND / LLM_BACKEND always win —
        we only flip the default if the user hasn't explicitly chosen.
        """
        embedding_explicit = "EMBEDDING_BACKEND" in os.environ
        llm_explicit = "LLM_BACKEND" in os.environ

        if not embedding_explicit and _is_placeholder(self.voyage_api_key):
            object.__setattr__(self, "embedding_backend", "local")
        if not llm_explicit and _is_placeholder(self.anthropic_api_key):
            object.__setattr__(self, "llm_backend", "replay")
        return self


# pydantic-settings populates fields from the environment at runtime;
# all fields have defaults now (empty string for keys), so Settings() is callable.
settings = Settings()
