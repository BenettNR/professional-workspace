"""Tests for the ingestion module (loaders, chunkers)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.ingestion.chunkers import FixedSizeChunker, RecursiveChunker
from src.ingestion.loaders import DocumentLoaderRegistry, UnsupportedFormatError
from src.models import ChunkStrategy

# ── Loaders ───────────────────────────────────────────────────────────────────


class TestDocumentLoaderRegistry:
    def setup_method(self):
        self.registry = DocumentLoaderRegistry()

    def test_markdown_loader(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("# My Heading\n\nSome content here.", encoding="utf-8")
        content, meta = self.registry.load(md)
        assert "Some content here" in content
        assert meta.get("section_heading") == "My Heading"

    def test_plaintext_loader(self, tmp_path: Path):
        txt = tmp_path / "doc.txt"
        txt.write_text("Hello world.", encoding="utf-8")
        content, meta = self.registry.load(txt)
        assert content == "Hello world."
        assert meta == {}

    def test_html_loader(self, tmp_path: Path):
        html = tmp_path / "doc.html"
        html.write_text(
            "<html><head><title>Test Page</title></head>"
            "<body><p>Content here.</p><script>alert(1)</script></body></html>",
            encoding="utf-8",
        )
        content, meta = self.registry.load(html)
        assert "Content here" in content
        assert "alert(1)" not in content  # script stripped
        assert meta.get("section_heading") == "Test Page"

    def test_unsupported_extension(self, tmp_path: Path):
        xls = tmp_path / "data.xlsx"
        xls.write_text("data")
        with pytest.raises(UnsupportedFormatError):
            self.registry.load(xls)

    def test_markdown_no_heading(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("No heading, just text.", encoding="utf-8")
        _, meta = self.registry.load(md)
        assert meta.get("section_heading") is None


# ── Chunkers ──────────────────────────────────────────────────────────────────


class TestFixedSizeChunker:
    @pytest.mark.asyncio
    async def test_splits_long_text(self):
        chunker = FixedSizeChunker()
        text = "word " * 300  # ~1500 chars
        chunks = await chunker.chunk(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 120  # with overlap tolerance

    @pytest.mark.asyncio
    async def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker()
        text = "Short text."
        chunks = await chunker.chunk(text, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_strategy_value(self):
        assert FixedSizeChunker().strategy == ChunkStrategy.FIXED


class TestRecursiveChunker:
    @pytest.mark.asyncio
    async def test_respects_markdown_headings(self):
        chunker = RecursiveChunker()
        text = textwrap.dedent("""
            ## Section One

            Content of section one with enough words to fill a chunk properly.

            ## Section Two

            Content of section two which is completely different from section one.
        """).strip()
        chunks = await chunker.chunk(text, chunk_size=100, chunk_overlap=0)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_empty_text_returns_non_empty(self):
        chunker = RecursiveChunker()
        chunks = await chunker.chunk("x")
        assert len(chunks) >= 1

    def test_strategy_value(self):
        assert RecursiveChunker().strategy == ChunkStrategy.RECURSIVE
