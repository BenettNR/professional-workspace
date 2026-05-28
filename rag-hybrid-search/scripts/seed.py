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
    # Use the same DI container the API uses, so the collection name
    # (per-embedder, e.g. rag_docs_voyage_voyage-3_1024) is identical to what
    # the API and the replay recorder read from. A single source of truth
    # avoids the "seeded one collection, queried another" class of bug.
    from src.api.dependencies import (
        get_chroma_collection,
        get_embedding_provider,
        get_indexer,
    )

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
        # lru_cache singletons may hold a handle to the now-deleted collection
        get_chroma_collection.cache_clear()
        get_embedding_provider.cache_clear()
        get_indexer.cache_clear()

    embedder = get_embedding_provider()
    collection = get_chroma_collection()
    log.info("seed_target", collection=collection.name, embedder=embedder.name)

    indexer = get_indexer()

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
    print(f"\n[OK] Indexed {len(results)} documents -> {total_chunks} chunks in ChromaDB")
    print(f"  Strategy: {strategy}")
    print(f"  Duplicates skipped: {total_skipped}")
    print(f"  Collection size: {collection.count()}")
    print("\nStart the API:      uvicorn src.api.main:app --reload")
    print("Start the frontend: streamlit run frontend/app.py")


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
