"""Tests for ReplayLLMProvider."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.exceptions import ProviderError
from src.providers.llm import LLMProvider
from src.providers.llm_replay import ReplayLLMProvider, _call_hash


def _write_fixtures(tmp_path: Path, calls: dict[str, str]) -> Path:
    path = tmp_path / "replay_fixtures.json"
    path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    return path


def _write_questions(tmp_path: Path, questions: list[str]) -> Path:
    path = tmp_path / "demo_questions.json"
    path.write_text(
        json.dumps(
            {"questions": [{"id": f"q{i:02d}", "question": q} for i, q in enumerate(questions, 1)]}
        ),
        encoding="utf-8",
    )
    return path


class TestReplayLLMProviderShape:
    def test_satisfies_protocol(self, tmp_path: Path):
        fixtures = _write_fixtures(tmp_path, {})
        provider = ReplayLLMProvider(fixtures_path=fixtures)
        assert isinstance(provider, LLMProvider)

    def test_name_is_stable(self, tmp_path: Path):
        fixtures = _write_fixtures(tmp_path, {})
        provider = ReplayLLMProvider(fixtures_path=fixtures)
        assert provider.name == "replay"


class TestReplayLLMProviderLoading:
    def test_missing_fixtures_raises_provider_error(self, tmp_path: Path):
        with pytest.raises(ProviderError) as exc_info:
            ReplayLLMProvider(fixtures_path=tmp_path / "does-not-exist.json")
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_json_raises_provider_error(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json {", encoding="utf-8")
        with pytest.raises(ProviderError) as exc_info:
            ReplayLLMProvider(fixtures_path=path)
        assert "load replay fixtures" in str(exc_info.value).lower()

    def test_loads_demo_questions_when_provided(self, tmp_path: Path):
        fixtures = _write_fixtures(tmp_path, {})
        questions = _write_questions(tmp_path, ["What is X?", "How do I Y?"])
        provider = ReplayLLMProvider(
            fixtures_path=fixtures, demo_questions_path=questions
        )
        assert provider.available_questions == ["What is X?", "How do I Y?"]


class TestReplayLLMProviderComplete:
    @pytest.mark.asyncio
    async def test_hit_returns_recorded_text(self, tmp_path: Path):
        key = _call_hash("SYS", "USR", 100)
        fixtures = _write_fixtures(tmp_path, {key: "recorded answer"})
        provider = ReplayLLMProvider(fixtures_path=fixtures)

        result = await provider.complete(system="SYS", user="USR", max_tokens=100)

        assert result == "recorded answer"

    @pytest.mark.asyncio
    async def test_miss_returns_diagnostic_with_question_count(self, tmp_path: Path):
        fixtures = _write_fixtures(tmp_path, {})
        questions = _write_questions(tmp_path, ["A?", "B?", "C?"])
        provider = ReplayLLMProvider(
            fixtures_path=fixtures, demo_questions_path=questions
        )

        result = await provider.complete(
            system="SYS", user="unknown question", max_tokens=100
        )

        assert "Demo mode" in result
        assert "3 pre-recorded" in result

    @pytest.mark.asyncio
    async def test_different_max_tokens_is_a_different_key(self, tmp_path: Path):
        key_100 = _call_hash("SYS", "USR", 100)
        fixtures = _write_fixtures(tmp_path, {key_100: "for 100 tokens"})
        provider = ReplayLLMProvider(fixtures_path=fixtures)

        hit = await provider.complete(system="SYS", user="USR", max_tokens=100)
        miss = await provider.complete(system="SYS", user="USR", max_tokens=200)

        assert hit == "for 100 tokens"
        assert "Demo mode" in miss


class TestReplayLLMProviderReload:
    @pytest.mark.asyncio
    async def test_reloads_when_fixtures_file_mtime_changes(self, tmp_path: Path):
        key = _call_hash("SYS", "USR", 100)
        fixtures = _write_fixtures(tmp_path, {})
        provider = ReplayLLMProvider(fixtures_path=fixtures)

        before = await provider.complete(system="SYS", user="USR", max_tokens=100)
        assert "Demo mode" in before

        # Rewrite fixtures with a recorded answer; force a different mtime.
        import os
        import time

        time.sleep(0.01)
        fixtures.write_text(
            json.dumps({"calls": {key: "now recorded"}}), encoding="utf-8"
        )
        os.utime(fixtures, None)  # bump mtime

        after = await provider.complete(system="SYS", user="USR", max_tokens=100)
        assert after == "now recorded"
