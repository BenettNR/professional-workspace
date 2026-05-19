from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Voyage AI (embeddings) ────────────────────────────────────────────────
    voyage_api_key: str = Field(..., description="Voyage AI API key for embeddings")
    embedding_model: str = "voyage-3"
    embedding_batch_size: int = 128  # Voyage AI per-request limit

    # ── Anthropic (generation + LLM-as-judge) ─────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key for Claude Sonnet")
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048

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

    # ── Data paths ────────────────────────────────────────────────────────────
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"


settings = Settings()
