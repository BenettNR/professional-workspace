"""POST /v1/ingest — upload and index a new document."""
from __future__ import annotations

import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.models import IngestResponse
from src.api.dependencies import get_indexer, get_sparse_retriever
from src.ingestion.indexer import DocumentIndexer
from src.retrieval.sparse import SparseRetriever
from src.models import ChunkStrategy

router = APIRouter()

_ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".pdf"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: str = Form(default="recursive"),
    indexer: DocumentIndexer = Depends(get_indexer),
    sparse_retriever: SparseRetriever = Depends(get_sparse_retriever),
) -> IngestResponse:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    try:
        chunk_strategy = ChunkStrategy(strategy)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{strategy}'. Choose: fixed, recursive, semantic",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    try:
        async with aiofiles.open(tmp_path, "wb") as out:
            content = await file.read()
            await out.write(content)

        result = await indexer.ingest_file(tmp_path, strategy=chunk_strategy)
        sparse_retriever.invalidate_cache()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        filename=filename,
        chunks_indexed=len(result.chunks),
        chunks_skipped=result.skipped_duplicates,
        strategy=strategy,
        message=f"Successfully indexed {len(result.chunks)} chunks from '{filename}'.",
    )
