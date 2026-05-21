"""Tests for config auto-detection of offline backends when API keys are missing."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip any provider/backend env vars and force a fresh Settings import."""
    for var in (
        "VOYAGE_API_KEY",
        "ANTHROPIC_API_KEY",
        "EMBEDDING_BACKEND",
        "LLM_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)
    # Force re-import so model_validator runs against the cleaned env
    import src.config

    importlib.reload(src.config)
    yield
    importlib.reload(src.config)


class TestAutoDetectDefaults:
    def test_missing_voyage_key_flips_to_local(self, clean_env: None) -> None:
        from src.config import settings

        assert settings.embedding_backend == "local"

    def test_missing_anthropic_key_flips_to_replay(self, clean_env: None) -> None:
        from src.config import settings

        assert settings.llm_backend == "replay"

    def test_placeholder_voyage_key_flips_to_local(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-...")
        import src.config

        importlib.reload(src.config)

        assert src.config.settings.embedding_backend == "local"

    def test_placeholder_anthropic_key_flips_to_replay(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
        import src.config

        importlib.reload(src.config)

        assert src.config.settings.llm_backend == "replay"


class TestRealKeysOverride:
    def test_real_voyage_key_keeps_voyage_backend(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-real_looking_key_xyz")
        import src.config

        importlib.reload(src.config)

        assert src.config.settings.embedding_backend == "voyage"

    def test_real_anthropic_key_keeps_anthropic_backend(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real_xyz")
        import src.config

        importlib.reload(src.config)

        assert src.config.settings.llm_backend == "anthropic"


class TestExplicitBackendWins:
    def test_explicit_voyage_backend_with_missing_key(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting EMBEDDING_BACKEND=voyage explicitly overrides auto-detection."""
        monkeypatch.setenv("EMBEDDING_BACKEND", "voyage")
        import src.config

        importlib.reload(src.config)

        # Auto-detect would set "local" but explicit env var wins
        assert src.config.settings.embedding_backend == "voyage"

    def test_explicit_local_backend_with_real_key(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting EMBEDDING_BACKEND=local with a real key still uses local."""
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-real_xyz")
        monkeypatch.setenv("EMBEDDING_BACKEND", "local")
        import src.config

        importlib.reload(src.config)

        assert src.config.settings.embedding_backend == "local"
