from pathlib import Path

import pdfplumber
import pandas as pd

from extraction.models import RawDocument
from ingestion.base import BaseParser


class PDFParser(BaseParser):
    def can_parse(self, suffix: str) -> bool:
        return suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> RawDocument:
        pages_text: list[str] = []
        tables: list[pd.DataFrame] = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

                for table in page.extract_tables():
                    if not table:
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df = df.dropna(how="all")
                    if not df.empty:
                        tables.append(df)

        return RawDocument(
            source_file=str(file_path),
            format="pdf",
            text="\n".join(pages_text),
            tables=tables,
        )
