"""Download the four arxiv RAG survey PDFs used by the Condition C eval.

Idempotent — skips files already present. Run before
`scripts/compare_retrieval.py --dataset eval/golden_dataset_rag_papers.json`
or before any UI / API ingest of the test corpus:

    uv run python scripts/download_rag_papers.py

The PDFs are not committed to this repo (third-party content). The
filenames here match the `expected_sources` entries in
`eval/golden_dataset_rag_papers.json` and the citations recorded by the
Condition C eval — keep them in sync.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# (arxiv_id, local_filename_used_by_dataset)
_PAPERS: list[tuple[str, str]] = [
    ("2312.10997", "rag-survey-gao-2024.pdf"),
    ("2410.12837", "2410.12837.pdf"),
    ("2402.19473", "2402.19473.pdf"),
    ("2409.14924", "2409.14924.pdf"),
]


def main() -> int:
    target_dir = Path("data/test-pdfs")
    target_dir.mkdir(parents=True, exist_ok=True)
    for arxiv_id, filename in _PAPERS:
        out = target_dir / filename
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {filename} already present ({out.stat().st_size:,} bytes)")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"[get ] {url} -> {out}")
        try:
            urllib.request.urlretrieve(url, out)  # noqa: S310 — arxiv only
        except Exception as exc:
            print(f"[err ] {url}: {exc}", file=sys.stderr)
            return 1
        print(f"       {out.stat().st_size:,} bytes")
    print()
    print(f"All papers in {target_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
