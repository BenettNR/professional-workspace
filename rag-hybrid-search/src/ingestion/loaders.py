"""Multi-format document loaders.

Each loader returns (plaintext_content, extra_metadata). The registry
dispatches to the correct loader based on file extension.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.exceptions import DocumentLoadError, UnsupportedFormatError


class DocumentLoader(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    @abstractmethod
    def load(self, path: Path) -> tuple[str, dict[str, Any]]:
        """Return (content, extra_metadata_dict)."""
        ...


class MarkdownLoader(DocumentLoader):
    _EXTENSIONS = {".md", ".markdown"}

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self._EXTENSIONS

    def load(self, path: Path) -> tuple[str, dict[str, Any]]:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DocumentLoadError(f"Cannot read {path}: {exc}") from exc

        heading_match = re.search(r"^#{1,6}\s+(.+)", content, re.MULTILINE)
        return content, {
            "section_heading": heading_match.group(1).strip() if heading_match else None
        }


class PlainTextLoader(DocumentLoader):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def load(self, path: Path) -> tuple[str, dict[str, Any]]:
        try:
            return path.read_text(encoding="utf-8"), {}
        except OSError as exc:
            raise DocumentLoadError(f"Cannot read {path}: {exc}") from exc


class HTMLLoader(DocumentLoader):
    _EXTENSIONS = {".html", ".htm"}

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self._EXTENSIONS

    def load(self, path: Path) -> tuple[str, dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise DocumentLoadError("beautifulsoup4 is required for HTML loading") from exc

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DocumentLoadError(f"Cannot read {path}: {exc}") from exc

        soup = BeautifulSoup(raw, "html.parser")
        title_tag = soup.find("title")
        heading = title_tag.get_text(strip=True) if title_tag else None

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(separator="\n"), {"section_heading": heading}


class PDFLoader(DocumentLoader):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path) -> tuple[str, dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentLoadError("pypdf is required for PDF loading") from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise DocumentLoadError(f"Cannot parse PDF {path}: {exc}") from exc

        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages), {"page_count": len(reader.pages)}


class DocumentLoaderRegistry:
    """Dispatches file paths to the appropriate loader."""

    def __init__(self) -> None:
        self._loaders: list[DocumentLoader] = [
            MarkdownLoader(),
            PlainTextLoader(),
            HTMLLoader(),
            PDFLoader(),
        ]

    def load(self, path: Path) -> tuple[str, dict[str, Any]]:
        for loader in self._loaders:
            if loader.supports(path):
                return loader.load(path)
        raise UnsupportedFormatError(
            f"No loader registered for extension '{path.suffix}'. Supported: .md, .txt, .html, .pdf"
        )

    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
