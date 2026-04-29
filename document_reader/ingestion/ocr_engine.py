from pathlib import Path

import pytesseract
from PIL import Image

from extraction.models import RawDocument
from ingestion.base import BaseParser

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class OCRParser(BaseParser):
    def can_parse(self, suffix: str) -> bool:
        return suffix.lower() in _IMAGE_SUFFIXES

    def parse(self, file_path: Path) -> RawDocument:
        image = Image.open(file_path)
        # Upscale for better OCR accuracy on scanned bank statements
        if max(image.size) < 2000:
            scale = 2000 / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")

        return RawDocument(
            source_file=str(file_path),
            format="image",
            text=text,
            tables=[],
        )
