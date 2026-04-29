from pathlib import Path

from ingestion.base import BaseParser
from ingestion.csv_parser import CSVParser
from ingestion.ocr_engine import OCRParser
from ingestion.pdf_parser import PDFParser

_PARSERS: list[BaseParser] = [PDFParser(), CSVParser(), OCRParser()]


def get_parser(file_path: Path) -> BaseParser:
    suffix = file_path.suffix.lower()
    for parser in _PARSERS:
        if parser.can_parse(suffix):
            return parser
    raise ValueError(f"No parser available for file type: {suffix!r}")
