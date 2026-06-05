"""Tests for tqdm progress bars in the eval harness (issue #89).

Verifies that run_eval_csv and run_eval_tickers emit nested progress bars
(position=0 outer, position=1 inner) and never corrupt stdout.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hmm_futures_analysis.eval import run_eval_csv, run_eval_tickers

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "test_data" / "eval-results"
)
THRESHOLD = ("threshold",)


class TestEvalCsvTqdm:
    """Verify run_eval_csv uses nested tqdm bars."""

    @patch("hmm_futures_analysis.eval.tqdm")
    def test_outer_bar_over_csv_files(self, mock_tqdm_cls):
        """Outer tqdm wraps CSV files with position=0, leave=True, desc='CSVs'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable

        run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)

        outer_calls = [
            c for c in mock_tqdm_cls.call_args_list if c.kwargs.get("position") == 0
        ]
        assert len(outer_calls) == 1
        assert outer_calls[0].kwargs.get("leave") is True
        assert outer_calls[0].kwargs.get("desc") == "CSVs"

    @patch("hmm_futures_analysis.eval.tqdm")
    def test_inner_bar_over_engines(self, mock_tqdm_cls):
        """Inner tqdm wraps engines with position=1, leave=False."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable

        run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)

        inner_calls = [
            c for c in mock_tqdm_cls.call_args_list if c.kwargs.get("position") == 1
        ]
        assert len(inner_calls) >= 1  # one per CSV file
        for call in inner_calls:
            assert call.kwargs.get("leave") is False
            assert call.kwargs.get("desc", "").startswith("  ")

    @patch("hmm_futures_analysis.eval.tqdm")
    def test_nested_bar_count(self, mock_tqdm_cls):
        """2 CSVs × 1 engine: 1 outer + 2 inner = 3 tqdm calls."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable

        run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)

        assert mock_tqdm_cls.call_count == 3

    @patch("hmm_futures_analysis.eval.tqdm")
    def test_results_unchanged(self, mock_tqdm_cls):
        """Results are identical with or without tqdm passthrough."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable

        results = run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)

        assert len(results) == 2
        assert {r["ticker"] for r in results} == {"CRM", "0700_HK"}
        assert all(r["engine"] == "threshold" for r in results)


class TestEvalTickersTqdm:
    """Verify run_eval_tickers uses nested tqdm bars."""

    @patch("hmm_futures_analysis.eval._save_ticker_csv")
    @patch("hmm_futures_analysis.eval.tqdm")
    def test_outer_bar_over_tickers(self, mock_tqdm_cls, mock_save):
        """Outer tqdm wraps tickers with position=0, leave=True, desc='Tickers'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_save.return_value = str(FIXTURE_DIR / "CRM.csv")

        run_eval_tickers(("CRM",), csv_cache_dir="/tmp/fake", engines=THRESHOLD)

        outer_calls = [
            c for c in mock_tqdm_cls.call_args_list if c.kwargs.get("position") == 0
        ]
        assert len(outer_calls) == 1
        assert outer_calls[0].kwargs.get("leave") is True
        assert outer_calls[0].kwargs.get("desc") == "Tickers"

    @patch("hmm_futures_analysis.eval._save_ticker_csv")
    @patch("hmm_futures_analysis.eval.tqdm")
    def test_inner_bar_over_engines(self, mock_tqdm_cls, mock_save):
        """Inner tqdm wraps engines with position=1, leave=False."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_save.return_value = str(FIXTURE_DIR / "CRM.csv")

        run_eval_tickers(("CRM",), csv_cache_dir="/tmp/fake", engines=THRESHOLD)

        inner_calls = [
            c for c in mock_tqdm_cls.call_args_list if c.kwargs.get("position") == 1
        ]
        assert len(inner_calls) == 1
        assert inner_calls[0].kwargs.get("leave") is False
        assert inner_calls[0].kwargs.get("desc", "").startswith("  ")
