"""Vendor-agnostic provider interfaces for embeddings and LLM completion.

The pipeline depends on these Protocols, not on Voyage/Anthropic SDKs directly.
Adding a new vendor means adding one new class here — no changes to retrieval
or generation code.
"""
from src.providers.embedding import EmbeddingProvider, VoyageEmbeddingProvider
from src.providers.embedding_local import LocalSentenceTransformerEmbeddingProvider
from src.providers.llm import AnthropicLLMProvider, LLMProvider
from src.providers.llm_replay import ReplayLLMProvider

__all__ = [
    "EmbeddingProvider",
    "VoyageEmbeddingProvider",
    "LocalSentenceTransformerEmbeddingProvider",
    "LLMProvider",
    "AnthropicLLMProvider",
    "ReplayLLMProvider",
]
