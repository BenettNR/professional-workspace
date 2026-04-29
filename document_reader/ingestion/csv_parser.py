from pathlib import Path

import pandas as pd

from extraction.models import RawDocument
from ingestion.base import BaseParser


class CSVParser(BaseParser):
    def can_parse(self, suffix: str) -> bool:
        return suffix.lower() in {".csv", ".xlsx", ".xls"}

    def parse(self, file_path: Path) -> RawDocument:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path, dtype=str, skip_blank_lines=True)
        else:
            df = pd.read_excel(file_path, dtype=str)

        df = df.dropna(how="all").reset_index(drop=True)

        return RawDocument(
            source_file=str(file_path),
            format=suffix.lstrip("."),
            text="",
            tables=[df],
        )
