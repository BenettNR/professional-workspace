"""GET /v1/documents — list all indexed documents."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.models import DocumentInfo, DocumentsResponse
from src.api.dependencies import get_indexer
from src.ingestion.indexer import DocumentIndexer

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    indexer: DocumentIndexer = Depends(get_indexer),
) -> DocumentsResponse:
    docs = indexer.list_indexed_documents()
    return DocumentsResponse(
        total_documents=len(docs),
        documents=[
            DocumentInfo(
                filename=d["filename"],
                source_file=d["source_file"],
                chunking_strategy=d["chunking_strategy"],
                total_chunks=d["total_chunks"],
            )
            for d in docs
        ],
    )
