from abc import ABC, abstractmethod
from pathlib import Path

from extraction.models import RawDocument


class BaseParser(ABC):
    @abstractmethod
    def can_parse(self, suffix: str) -> bool: ...

    @abstractmethod
    def parse(self, file_path: Path) -> RawDocument: ...
