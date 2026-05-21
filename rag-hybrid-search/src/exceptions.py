class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""


class DocumentLoadError(RAGError):
    """Raised when a document cannot be loaded or parsed."""


class UnsupportedFormatError(DocumentLoadError):
    """Raised when no loader handles the given file extension."""


class ChunkingError(RAGError):
    """Raised when text splitting fails."""


class EmbeddingError(RAGError):
    """Raised when the embedding API call fails."""


class IndexingError(RAGError):
    """Raised when writing to ChromaDB or the BM25 index fails."""


class RetrievalError(RAGError):
    """Raised when a retrieval operation (dense or sparse) fails."""


class GenerationError(RAGError):
    """Raised when the LLM generation call fails."""


class CitationVerificationError(RAGError):
    """Raised when the citation verification step fails."""


class EvaluationError(RAGError):
    """Raised when the evaluation pipeline encounters an unrecoverable error."""


class ProviderError(RAGError):
    """Raised when a provider (embedding or LLM) is misconfigured or unreachable."""
