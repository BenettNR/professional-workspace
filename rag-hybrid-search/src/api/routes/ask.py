"""POST /v1/ask — full RAG pipeline: embed → retrieve → generate."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_embedder, get_generator, get_hybrid_retriever
from src.api.models import AskRequest, AskResponse, ChunkOut, CitationOut, ConfidenceOut
from src.generation.generator import RAGGenerator
from src.providers.embedding import EmbeddingProvider
from src.retrieval.fusion import HybridRetriever

router = APIRouter()

EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
RetrieverDep = Annotated[HybridRetriever, Depends(get_hybrid_retriever)]
GeneratorDep = Annotated[RAGGenerator, Depends(get_generator)]


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    embedder: EmbedderDep,
    retriever: RetrieverDep,
    generator: GeneratorDep,
) -> AskResponse:
    try:
        query_embedding = await embedder.embed_query(request.question)
        retrieved_chunks = await retriever.retrieve(
            query=request.question,
            query_embedding=query_embedding,
            dense_only=request.dense_only,
        )
        response = await generator.generate(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=response.question,
        answer=response.answer,
        citations=[
            CitationOut(
                number=c.number,
                source_file=c.source_file,
                text_excerpt=c.text_excerpt,
                verified=c.verified,
                verification_reason=c.verification_reason,
            )
            for c in response.citations
        ],
        confidence=ConfidenceOut(
            retrieval_confidence=response.confidence.retrieval_confidence,
            citation_coverage=response.confidence.citation_coverage,
            answer_completeness=response.confidence.answer_completeness,
            composite=response.confidence.composite,
        ),
        retrieved_chunks=[
            ChunkOut(
                id=rc.chunk.id,
                content=rc.chunk.content,
                source_file=rc.chunk.metadata.source_file,
                filename=rc.chunk.metadata.filename,
                chunk_index=rc.chunk.metadata.chunk_index,
                chunking_strategy=rc.chunk.metadata.chunking_strategy.value,
                dense_score=rc.dense_score,
                sparse_score=rc.sparse_score,
                fusion_score=rc.fusion_score,
                rerank_score=rc.rerank_score,
            )
            for rc in response.retrieved_chunks
        ],
        insufficient_info=response.insufficient_info,
        missing_info_message=response.missing_info_message,
    )
