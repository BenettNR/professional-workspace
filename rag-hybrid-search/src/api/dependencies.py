"""Dependency injection container using functools.lru_cache singletons.

All heavyweight objects (models, DB connections) are created once and reused
across requests. FastAPI's Depends() wires these into route handlers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb

from src.config import settings
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import DocumentIndexer
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import RAGGenerator


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "l2"},
    )


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


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
def get_generator() -> RAGGenerator:
    return RAGGenerator(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        confidence_threshold=settings.retrieval_confidence_threshold,
    )


@lru_cache(maxsize=1)
def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        embedder=get_embedder(),
        collection=get_chroma_collection(),
        bm25_index_path=Path(settings.bm25_index_path),
        dedup_threshold=settings.dedup_similarity_threshold,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        semantic_breakpoint_threshold=settings.semantic_breakpoint_threshold,
    )
