"""Seed script — indexes the sample Nexus API documentation corpus.

Run from the project root after copying .env.example to .env and filling in your keys:

  python scripts/seed.py

Options:
  --strategy    fixed | recursive | semantic  (default: recursive)
  --clean       Drop and rebuild indexes from scratch
  --docs-dir    Override the docs directory path
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.config import settings
from src.models import ChunkStrategy

log = structlog.get_logger()


async def main(strategy: str, clean: bool, docs_dir: Path) -> None:
    import chromadb
    from src.ingestion.embedder import Embedder
    from src.ingestion.indexer import DocumentIndexer

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    log.info("seed_start", docs_dir=str(docs_dir), strategy=strategy, clean=clean)

    bm25_path = Path(settings.bm25_index_path)

    if clean:
        log.warning("clean_mode_wiping_indexes")
        chroma_dir = Path(settings.chroma_persist_directory)
        if chroma_dir.exists():
            import shutil
            shutil.rmtree(chroma_dir)
            log.info("chroma_wiped", path=str(chroma_dir))
        if bm25_path.exists():
            bm25_path.unlink()
            log.info("bm25_wiped", path=str(bm25_path))

    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "l2"},
    )

    embedder = Embedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    indexer = DocumentIndexer(
        embedder=embedder,
        collection=collection,
        bm25_index_path=bm25_path,
        dedup_threshold=settings.dedup_similarity_threshold,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        semantic_breakpoint_threshold=settings.semantic_breakpoint_threshold,
    )

    chunk_strategy = ChunkStrategy(strategy)
    results = await indexer.ingest_directory(docs_dir, strategy=chunk_strategy, recursive=True)

    total_chunks = sum(len(r.chunks) for r in results)
    total_skipped = sum(r.skipped_duplicates for r in results)

    log.info(
        "seed_complete",
        documents=len(results),
        total_chunks_indexed=total_chunks,
        total_duplicates_skipped=total_skipped,
        collection_size=collection.count(),
    )
    print(f"\n✓ Indexed {len(results)} documents → {total_chunks} chunks in ChromaDB")
    print(f"  Strategy: {strategy}")
    print(f"  Duplicates skipped: {total_skipped}")
    print(f"  Collection size: {collection.count()}")
    print(f"\nStart the API:      uvicorn src.api.main:app --reload")
    print(f"Start the frontend: streamlit run frontend/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the RAG index with sample documentation")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "semantic"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe existing indexes before seeding",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data/raw/nexus-api-docs"),
        help="Path to documentation directory",
    )
    args = parser.parse_args()
    asyncio.run(main(args.strategy, args.clean, args.docs_dir))
