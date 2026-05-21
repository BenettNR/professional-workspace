"""Citation verification — the quality layer most RAG systems skip.

For each citation [N] in the generated answer, we:
  1. Parse the claim sentence(s) surrounding the citation marker
  2. Send (claim, chunk_N_content) to the LLM as LLM-as-judge
  3. Mark the citation as verified or flagged

This catches the most common RAG failure mode: confident answers with
citations that don't actually support the stated claim.
"""

from __future__ import annotations

import asyncio
import re

import structlog

from src.exceptions import CitationVerificationError, ProviderError
from src.models import Citation, RetrievedChunk
from src.providers.llm import LLMProvider

log = structlog.get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

_VERIFICATION_SYSTEM = """\
You are a fact-checking assistant. Determine whether the provided CLAIM is
directly supported by the PASSAGE. Answer with exactly one of:
  SUPPORTED   — the passage clearly supports the claim
  UNSUPPORTED — the passage does not support or contradicts the claim
  PARTIAL     — the passage partially supports the claim

Then on a new line, write a single-sentence reason.
"""


def _extract_claim_sentences(answer: str, citation_number: int) -> str:
    """Return the sentence(s) immediately surrounding citation [N]."""
    marker = f"[{citation_number}]"
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    relevant = [s for s in sentences if marker in s]
    if not relevant:
        idx = answer.find(marker)
        if idx >= 0:
            return answer[max(0, idx - 200) : idx + 200]
        return ""
    return " ".join(relevant)


def _parse_citation_numbers(answer: str) -> list[int]:
    matches = _CITATION_PATTERN.findall(answer)
    return sorted({int(m) for m in matches})


class CitationVerifier:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def verify(
        self,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        citation_numbers = _parse_citation_numbers(answer)
        if not citation_numbers:
            log.info("no_citations_found")
            return []

        chunk_map = {i + 1: rc for i, rc in enumerate(retrieved_chunks)}
        tasks = [self._verify_one(answer, num, chunk_map.get(num)) for num in citation_numbers]
        citations = await asyncio.gather(*tasks)
        verified_count = sum(1 for c in citations if c.verified)
        log.info(
            "citation_verification_complete",
            total=len(citations),
            verified=verified_count,
        )
        return list(citations)

    async def _verify_one(
        self,
        answer: str,
        number: int,
        rc: RetrievedChunk | None,
    ) -> Citation:
        if rc is None:
            return Citation(
                number=number,
                chunk_id="",
                source_file="",
                text_excerpt="",
                verified=False,
                verification_reason=(
                    f"Citation [{number}] references a non-existent context block."
                ),
            )

        claim = _extract_claim_sentences(answer, number)
        passage = rc.chunk.content[:800]

        try:
            verdict_text = await self._llm.complete(
                system=_VERIFICATION_SYSTEM,
                user=f"CLAIM:\n{claim}\n\nPASSAGE:\n{passage}",
                max_tokens=100,
            )
        except ProviderError as exc:
            raise CitationVerificationError(
                f"Verification call failed for citation [{number}]: {exc}"
            ) from exc

        lines = verdict_text.strip().split("\n", 1)
        verdict = lines[0].strip().upper()
        reason = lines[1].strip() if len(lines) > 1 else verdict_text

        verified = verdict == "SUPPORTED"
        log.debug(
            "citation_verdict",
            number=number,
            verdict=verdict,
            source=rc.chunk.metadata.filename,
        )
        return Citation(
            number=number,
            chunk_id=rc.chunk.id,
            source_file=rc.chunk.metadata.source_file,
            text_excerpt=rc.chunk.content[:200],
            verified=verified,
            verification_reason=reason,
        )
