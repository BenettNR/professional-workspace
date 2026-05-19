"""FastAPI application entry point.

Endpoints:
  POST /v1/ask        — hybrid RAG query
  GET  /v1/documents  — list indexed documents
  POST /v1/ingest     — upload and index a new document

OpenAPI docs available at /docs (Swagger) and /redoc.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ask, documents, ingest
from src.config import settings

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "startup",
        host=settings.api_host,
        port=settings.api_port,
        llm=settings.llm_model,
        embedding=settings.embedding_model,
        chunk_strategy=settings.default_chunk_strategy,
    )
    yield
    log.info("shutdown")


app = FastAPI(
    title="RAG Hybrid Search API",
    description=(
        "Production-grade Retrieval-Augmented Generation with hybrid dense+sparse search, "
        "Reciprocal Rank Fusion, cross-encoder reranking, and citation verification."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router, prefix="/v1", tags=["Query"])
app.include_router(documents.router, prefix="/v1", tags=["Documents"])
app.include_router(ingest.router, prefix="/v1", tags=["Ingestion"])


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
