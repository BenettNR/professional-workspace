"""Replay LLMProvider — fixture-based playback for offline demo mode.

Recorded Claude responses live in `eval/replay_fixtures.json`, keyed by
SHA256 hash of (system + '\n' + user + '\n' + str(max_tokens)). On a hash
match we return the recorded text; on a miss we return a clearly-labeled
diagnostic response that names the available demo questions.

Why hash-keyed instead of question-keyed: the full pipeline issues
multiple LLM calls per question (generation + citation verifications +
completeness scoring), each with a unique (system, user) payload. Hashing
both gives a stable key for each individual call without enumerating the
prompt shapes.

The fixture file is reloaded if its mtime changes — so running
`make replay-fixtures` updates the live offline mode without restarting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import structlog

from src.exceptions import ProviderError

log = structlog.get_logger(__name__)


_MISS_TEMPLATE = (
    "Demo mode: this question isn't in the curated replay set. "
    "Try one of the {n} pre-recorded demo questions (see the banner above), "
    "or add VOYAGE_API_KEY and ANTHROPIC_API_KEY to .env for live mode."
)


def _call_hash(system: str, user: str, max_tokens: int) -> str:
    payload = f"{system}\n{user}\n{max_tokens}".encode()
    return hashlib.sha256(payload).hexdigest()


class ReplayLLMProvider:
    """Plays back recorded LLM responses from a fixture file.

    On a miss, returns a diagnostic response — never raises — so the
    demo UI shows a useful message rather than a stack trace.
    """

    def __init__(
        self,
        fixtures_path: Path,
        demo_questions_path: Path | None = None,
    ) -> None:
        self.name = "replay"
        self._path = fixtures_path
        self._questions_path = demo_questions_path
        self._mtime: float | None = None
        self._calls: dict[str, str] = {}
        self._available_questions: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise ProviderError(
                f"Replay fixtures file not found at {self._path}. "
                "Run `make replay-fixtures` to record fixtures with live keys."
            )
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"Failed to load replay fixtures from {self._path}: {exc}"
            ) from exc

        self._calls = data.get("calls", {})
        self._mtime = self._path.stat().st_mtime
        log.info(
            "replay_fixtures_loaded",
            path=str(self._path),
            recorded_calls=len(self._calls),
        )

        if self._questions_path and self._questions_path.exists():
            try:
                qdata: dict[str, Any] = json.loads(
                    self._questions_path.read_text(encoding="utf-8")
                )
                self._available_questions = [
                    q["question"] for q in qdata.get("questions", [])
                ]
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("failed_to_load_demo_questions", error=str(exc))

    def _reload_if_changed(self) -> None:
        if not self._path.exists():
            return
        current_mtime = self._path.stat().st_mtime
        if current_mtime != self._mtime:
            log.info("replay_fixtures_reload", path=str(self._path))
            self._load()

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self._reload_if_changed()
        key = _call_hash(system, user, max_tokens)
        if key in self._calls:
            return self._calls[key]
        log.warning(
            "replay_miss",
            key_prefix=key[:8],
            recorded_calls=len(self._calls),
        )
        return _MISS_TEMPLATE.format(n=len(self._available_questions))

    @property
    def available_questions(self) -> list[str]:
        return list(self._available_questions)
