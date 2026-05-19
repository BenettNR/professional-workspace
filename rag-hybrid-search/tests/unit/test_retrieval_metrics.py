"""Tests for pure-function retrieval metrics."""

from __future__ import annotations

from src.evaluation.retrieval_metrics import (
    _normalize,
    hit_at_k,
    mrr_at_k,
    precision_at_k,
)


class TestNormalize:
    def test_strips_directory(self):
        assert _normalize("data/raw/nexus-api-docs/rate-limiting.md") == "rate-limiting.md"

    def test_strips_windows_path(self):
        assert _normalize("data\\raw\\rate-limiting.md") == "rate-limiting.md"

    def test_lowercases(self):
        assert _normalize("Rate-Limiting.md") == "rate-limiting.md"


class TestHitAtK:
    def test_hit_at_1_top_result_relevant(self):
        assert hit_at_k(["rate-limiting.md", "other.md"], ["rate-limiting.md"], 1) == 1.0

    def test_hit_at_1_top_result_irrelevant(self):
        assert hit_at_k(["other.md", "rate-limiting.md"], ["rate-limiting.md"], 1) == 0.0

    def test_hit_at_3_relevant_at_rank_3(self):
        assert (
            hit_at_k(
                ["a.md", "b.md", "rate-limiting.md", "c.md"],
                ["rate-limiting.md"],
                3,
            )
            == 1.0
        )

    def test_hit_at_3_relevant_outside_top_3(self):
        assert (
            hit_at_k(
                ["a.md", "b.md", "c.md", "rate-limiting.md"],
                ["rate-limiting.md"],
                3,
            )
            == 0.0
        )

    def test_empty_expected_sources_returns_0(self):
        assert hit_at_k(["a.md", "b.md"], [], 5) == 0.0

    def test_multiple_expected_any_match_counts(self):
        assert (
            hit_at_k(
                ["unrelated.md", "error-codes.md"],
                ["rate-limiting.md", "error-codes.md"],
                2,
            )
            == 1.0
        )


class TestMRRAtK:
    def test_first_result_relevant(self):
        assert mrr_at_k(["target.md", "x.md"], ["target.md"], 5) == 1.0

    def test_second_result_relevant(self):
        assert mrr_at_k(["x.md", "target.md", "y.md"], ["target.md"], 5) == 0.5

    def test_third_result_relevant(self):
        assert abs(mrr_at_k(["x.md", "y.md", "target.md"], ["target.md"], 5) - (1 / 3)) < 1e-9

    def test_no_relevant_in_top_k(self):
        assert mrr_at_k(["x.md", "y.md"], ["target.md"], 5) == 0.0

    def test_relevant_outside_top_k(self):
        # target.md is at rank 6, but k=5 — should be 0
        retrieved = ["a.md", "b.md", "c.md", "d.md", "e.md", "target.md"]
        assert mrr_at_k(retrieved, ["target.md"], 5) == 0.0

    def test_empty_expected_sources(self):
        assert mrr_at_k(["a.md"], [], 5) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert (
            precision_at_k(
                ["a.md", "b.md"],
                ["a.md", "b.md", "c.md"],
                2,
            )
            == 1.0
        )

    def test_half_relevant(self):
        assert (
            precision_at_k(
                ["a.md", "x.md"],
                ["a.md", "b.md"],
                2,
            )
            == 0.5
        )

    def test_none_relevant(self):
        assert (
            precision_at_k(
                ["x.md", "y.md"],
                ["a.md", "b.md"],
                2,
            )
            == 0.0
        )

    def test_empty_expected_sources(self):
        assert precision_at_k(["a.md"], [], 5) == 0.0

    def test_path_normalization_works(self):
        # Retrieved uses full paths, expected uses basenames — should still match
        assert (
            hit_at_k(
                ["data/raw/nexus-api-docs/rate-limiting.md"],
                ["rate-limiting.md"],
                1,
            )
            == 1.0
        )
