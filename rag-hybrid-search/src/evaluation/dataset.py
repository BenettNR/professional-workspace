"""Golden Q&A dataset management.

Loads the handcrafted evaluation set from eval/golden_dataset.json and
provides iteration helpers used by the evaluation runner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]
    category: Literal["simple_lookup", "multi_hop", "unanswerable", "ambiguous"]
    difficulty: Literal["easy", "medium", "hard"]
    notes: str = ""


class GoldenDataset:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._questions: list[GoldenQuestion] = []
        self._metadata: dict = {}
        self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        self._metadata = {k: v for k, v in data.items() if k != "questions"}
        self._questions = [GoldenQuestion(**q) for q in data["questions"]]

    @property
    def questions(self) -> list[GoldenQuestion]:
        return self._questions

    def by_category(self, category: str) -> list[GoldenQuestion]:
        return [q for q in self._questions if q.category == category]

    def by_difficulty(self, difficulty: str) -> list[GoldenQuestion]:
        return [q for q in self._questions if q.difficulty == difficulty]

    def __len__(self) -> int:
        return len(self._questions)

    def __iter__(self):
        return iter(self._questions)
