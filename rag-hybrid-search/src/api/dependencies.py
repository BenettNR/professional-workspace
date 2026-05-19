"""Dependency injection container using functools.lru_cache singletons.

All heavyweight objects (models, DB connections, provider clients) are created
once and reused across requests. FastAPI's Depends() wires these into routes.

Concrete providers are selected based on `settings.embedding_backend` and
`settings.llm_backend`. PR-2 adds 'local' and 'replay' alternatives; this PR
ships 'voyage' and 'anthropic' only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb

from src.config import settings
from src.exceptions import ProviderError
from src.generation.citation_verifier import CitationVerifier
from src.generation.confidence import ConfidenceScorer
from src.generation.generator import RAGGenerator
from src.ingestion.indexer import DocumentIndexer
from src.providers.embedding import EmbeddingProvider, VoyageEmbeddingProvider
from src.providers.embedding_local import LocalSentenceTransformerEmbeddingProvider
from src.providers.llm import AnthropicLLMProvider, LLMProvider
from src.providers.llm_replay import ReplayLLMProvider
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_backend == "voyage":
        return VoyageEmbeddingProvider(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_backend == "local":
        return LocalSentenceTransformerEmbeddingProvider()
    raise ProviderError(f"Unsupported embedding_backend '{settings.embedding_backend}'.")


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    if settings.llm_backend == "anthropic":
        return AnthropicLLMProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
    if settings.llm_backend == "replay":
        return ReplayLLMProvider(
            fixtures_path=Path(settings.replay_fixtures_path),
            demo_questions_path=Path(settings.demo_questions_path),
        )
    raise ProviderError(f"Unsupported llm_backend '{settings.llm_backend}'.")


def _safe_collection_name(provider_name: str, dim: int) -> str:
    """Derive a Chroma-safe collection name from the embedder identity.

    Chroma names must match `[a-zA-Z0-9._-]{3,512}` and avoid certain
    sequences. The provider name like 'voyage:voyage-3' becomes
    'rag_docs_voyage_voyage-3_1024'.
    """
    safe = provider_name.replace(":", "_").replace("/", "-")
    return f"rag_docs_{safe}_{dim}"


@lru_cache(maxsize=1)
def get_chroma_collection() -> chromadb.Collection:
    # Per-embedder collection naming: swapping embedders never corrupts
    # an existing index. Each embedder gets its own collection silo.
    embedder = get_embedding_provider()
    name = _safe_collection_name(embedder.name, embedder.embedding_dim)
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "l2"},
    )


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    """Back-compat alias retained for the ask route's `Depends` signature."""
    return get_embedding_provider()


@lru_cache(maxsize=1)
def get_dense_retriever() -> DenseRetriever:
    return DenseRetriever(collection=get_chroma_collection())


@lru_cache(maxsize=1)
def get_sparse_retriever() -> SparseRetriever:
    return SparseRetriever(
        index_path=Path(settings.bm25_index_path),
        collection=get_chroma_collection(),
    )


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(
        model_name=settings.reranker_model,
        use_llm=settings.use_llm_reranker,
        anthropic_api_key=settings.anthropic_api_key,
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense=get_dense_retriever(),
        sparse=get_sparse_retriever(),
        reranker=get_reranker(),
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        fusion_top_k=settings.fusion_top_k,
        rerank_top_k=settings.rerank_top_k,
    )


@lru_cache(maxsize=1)
def get_citation_verifier() -> CitationVerifier:
    return CitationVerifier(llm=get_llm_provider())


@lru_cache(maxsize=1)
def get_confidence_scorer() -> ConfidenceScorer:
    return ConfidenceScorer(llm=get_llm_provider())


@lru_cache(maxsize=1)
def get_generator() -> RAGGenerator:
    return RAGGenerator(
        llm=get_llm_provider(),
        max_tokens=settings.llm_max_tokens,
        confidence_threshold=settings.retrieval_confidence_threshold,
        verifier=get_citation_verifier(),
        scorer=get_confidence_scorer(),
    )


@lru_cache(maxsize=1)
def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        embedder=get_embedding_provider(),
        collection=get_chroma_collection(),
        bm25_index_path=Path(settings.bm25_index_path),
        dedup_threshold=settings.dedup_similarity_threshold,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        semantic_breakpoint_threshold=settings.semantic_breakpoint_threshold,
    )
