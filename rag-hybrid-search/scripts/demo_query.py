"""Pretty-print a /v1/ask response. Used by the live demo session.

Reads JSON from stdin (piped from curl) and prints answer, citations,
confidence, and the source filenames of the retrieved chunks.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    d = json.load(sys.stdin)
    ans = d["answer"]
    suffix = "…" if len(ans) > 600 else ""
    print("ANSWER:", ans[:600] + suffix)
    print()
    print("CITATIONS:")
    for c in d["citations"]:
        fname = c["source_file"].replace("\\", "/").split("/")[-1]
        badge = "OK" if c["verified"] else "X"
        print(f"  [{c['number']}] {badge}  {fname}")
        if c.get("verification_reason"):
            reason = c["verification_reason"][:120]
            print(f"        reason: {reason}")
    conf = d["confidence"]
    print(
        f"CONFIDENCE: composite={conf['composite']:.2f}  "
        f"(retrieval={conf['retrieval_confidence']:.2f}  "
        f"cites={conf['citation_coverage']:.2f}  "
        f"complete={conf['answer_completeness']:.2f})"
    )
    print("RETRIEVED top sources (deduped):")
    seen: set[str] = set()
    for ch in d["retrieved_chunks"]:
        f = ch["filename"]
        if f not in seen:
            seen.add(f)
            print(f"  - {f}")


if __name__ == "__main__":
    main()
