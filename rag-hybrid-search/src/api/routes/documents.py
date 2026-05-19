"""GET /v1/documents — list all indexed documents."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_indexer
from src.api.models import DocumentInfo, DocumentsResponse
from src.ingestion.indexer import DocumentIndexer

router = APIRouter()

IndexerDep = Annotated[DocumentIndexer, Depends(get_indexer)]


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(indexer: IndexerDep) -> DocumentsResponse:
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
