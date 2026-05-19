"""Record Claude responses over the curated demo questions for offline replay.

Run with live API keys:

    VOYAGE_API_KEY=...  ANTHROPIC_API_KEY=...  \\
    uv run python scripts/build_replay_fixtures.py

For each question in eval/demo_questions.json, runs the full RAG pipeline
(real Voyage embeddings + real Claude generation + verification + scoring)
while wrapping the LLM provider with a recorder that captures every
(system, user, max_tokens, response) tuple. The captured calls are written
to eval/replay_fixtures.json keyed by SHA256(system + '\\n' + user + '\\n' + max_tokens),
which is exactly the key ReplayLLMProvider uses to look them up.

This is a one-time recording step that produces the data file backing
offline demo mode. After this runs, anyone can `make demo` without
ever needing an API key.

Cost: ~12 questions × 5-7 LLM calls each × ~500 tokens average ≈ ~$1.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.providers.llm import LLMProvider

log = structlog.get_logger(__name__)


def _call_hash(system: str, user: str, max_tokens: int) -> str:
    payload = f"{system}\n{user}\n{max_tokens}".encode()
    return hashlib.sha256(payload).hexdigest()


class _RecordingLLMProvider:
    """Wraps a real LLMProvider and captures every call."""

    def __init__(self, wrapped: LLMProvider) -> None:
        self._wrapped = wrapped
        self.name = wrapped.name
        self.calls: dict[str, str] = {}

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        response: str = await self._wrapped.complete(system, user, max_tokens)
        key = _call_hash(system, user, max_tokens)
        self.calls[key] = response
        return response


async def _record(
    questions_path: Path,
    fixtures_path: Path,
    git_commit: str | None,
) -> None:
    from src.api.dependencies import (
        get_chroma_collection,
        get_dense_retriever,
        get_embedding_provider,
        get_hybrid_retriever,
        get_indexer,
        get_reranker,
        get_sparse_retriever,
    )
    from src.config import settings
    from src.generation.citation_verifier import CitationVerifier
    from src.generation.confidence import ConfidenceScorer
    from src.generation.generator import RAGGenerator
    from src.providers.llm import AnthropicLLMProvider

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    questions_data = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = questions_data.get("questions", [])
    if not questions:
        raise SystemExit(f"No questions found in {questions_path}")

    log.info("recording_start", questions=len(questions), output=str(fixtures_path))

    # Force live backends regardless of env (we need real responses to record)
    real_llm = AnthropicLLMProvider(api_key=settings.anthropic_api_key, model=settings.llm_model)
    recorder = _RecordingLLMProvider(real_llm)

    # Build the pipeline with our recording wrapper in place of the LLM.
    # _RecordingLLMProvider is structurally compatible with LLMProvider —
    # exposes `name` and async `complete(system, user, max_tokens) -> str`.
    embedder = get_embedding_provider()
    retriever = get_hybrid_retriever()
    verifier = CitationVerifier(llm=recorder)
    scorer = ConfidenceScorer(llm=recorder)
    generator = RAGGenerator(
        llm=recorder,
        max_tokens=settings.llm_max_tokens,
        confidence_threshold=settings.retrieval_confidence_threshold,
        verifier=verifier,
        scorer=scorer,
    )

    # Touch eager-evaluated singletons so any setup happens before timing
    _ = get_chroma_collection()
    _ = get_dense_retriever()
    _ = get_sparse_retriever()
    _ = get_reranker()
    _ = get_indexer()

    start_time = time.time()
    for q in questions:
        q_text = q["question"]
        log.info("recording_question", id=q.get("id"), question=q_text)
        try:
            query_emb = await embedder.embed_query(q_text)
            chunks = await retriever.retrieve(query=q_text, query_embedding=query_emb)
            await generator.generate(question=q_text, retrieved_chunks=chunks)
        except Exception as exc:
            log.error("recording_question_failed", id=q.get("id"), error=str(exc))
            raise

    elapsed = time.time() - start_time

    output = {
        "_meta": {
            "description": (
                "Recorded Claude responses keyed by SHA256 of "
                "(system + '\\n' + user + '\\n' + max_tokens). "
                "Populated by `make replay-fixtures`. Do not edit by hand."
            ),
            "fixture_format_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "question_count": len(questions),
            "call_count": len(recorder.calls),
            "elapsed_seconds": round(elapsed, 1),
        },
        "calls": recorder.calls,
    }
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log.info(
        "recording_complete",
        path=str(fixtures_path),
        calls_captured=len(recorder.calls),
        elapsed_seconds=round(elapsed, 1),
    )
    print(
        f"\n[OK] Recorded {len(recorder.calls)} LLM calls across "
        f"{len(questions)} questions in {round(elapsed, 1)}s"
    )
    print(f"    -> {fixtures_path}")


def _read_git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record Claude responses for the offline demo replay set."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/demo_questions.json"),
        help="Path to demo questions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/replay_fixtures.json"),
        help="Path to write recorded fixtures",
    )
    args = parser.parse_args()

    if not os.environ.get("VOYAGE_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "[err] VOYAGE_API_KEY and ANTHROPIC_API_KEY must be set in the "
            "environment. This script records real Claude responses and "
            "cannot run in offline mode."
        )

    asyncio.run(_record(args.questions, args.output, _read_git_commit()))


if __name__ == "__main__":
    main()
