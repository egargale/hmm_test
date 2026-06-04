"""Tests for the multi-ticker evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hmm_futures_analysis.eval import (
    ALL_ENGINES,
    format_table,
    run_eval_csv,
    _extract_summary,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "test_data" / "eval-results"
THRESHOLD = ("threshold",)  # Fastest engine — use for most tests


class TestEvalCsv:
    """Test --eval-csv mode using golden fixtures."""

    def test_run_eval_csv_single_engine(self):
        """Single engine evaluation against both CSVs."""
        results = run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)
        assert len(results) == 2  # 2 tickers × 1 engine
        tickers = {r["ticker"] for r in results}
        assert tickers == {"CRM", "0700_HK"}
        assert all(r["engine"] == "threshold" for r in results)

    @pytest.mark.skip(reason="messina/hmm engines hang — runtime bug pending investigation")
    def test_run_eval_csv_two_engines(self):
        """Two-engine subset produces correct count."""
        results = run_eval_csv(str(FIXTURE_DIR), engines=("threshold", "messina"))
        assert len(results) == 4  # 2 tickers × 2 engines
        engines = {r["engine"] for r in results}
        assert engines == {"threshold", "messina"}

    def test_result_fields_present(self):
        """Each result has all expected fields."""
        results = run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)
        r = results[0]
        expected_keys = {
            "ticker", "engine", "regime", "signal", "sharpe",
            "max_drawdown", "n_trades", "win_rate", "profit_factor",
            "total_return", "wall_seconds",
        }
        assert set(r.keys()) == expected_keys
        assert isinstance(r["n_trades"], int)
        assert isinstance(r["wall_seconds"], float)

    def test_regime_values_valid(self):
        """Regime is one of the three canonical names."""
        results = run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)
        assert all(r["regime"] in ("bear", "sideways", "bull") for r in results)

    def test_invalid_directory(self):
        """Raises ValueError for non-existent directory."""
        with pytest.raises(ValueError, match="Not a directory"):
            run_eval_csv("/nonexistent/path")

    def test_empty_directory(self, tmp_path):
        """Raises ValueError for directory with no CSVs."""
        with pytest.raises(ValueError, match="No CSV files found"):
            run_eval_csv(str(tmp_path))


class TestFormatTable:
    """Test the markdown table formatter."""

    def test_empty_results(self):
        assert format_table([]) == "No results."

    def test_table_has_headers(self):
        results = run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)
        table = format_table(results)
        assert "ticker" in table
        assert "engine" in table
        assert "sharpe" in table

    def test_table_rows_match_results(self):
        results = run_eval_csv(str(FIXTURE_DIR), engines=THRESHOLD)
        table = format_table(results)
        lines = table.strip().split("\n")
        # header + separator + 2 data rows
        assert len(lines) == 4


class TestCliEvalMode:
    """Integration tests for the CLI eval flags."""

    def test_eval_csv_json_output(self):
        """--eval-csv --json produces valid JSON array."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hmm_futures_analysis.cli",
                "--eval-csv", str(FIXTURE_DIR),
                "--eval-engines", "threshold",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_eval_csv_table_output(self):
        """--eval-csv without --json prints table to stderr."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hmm_futures_analysis.cli",
                "--eval-csv", str(FIXTURE_DIR),
                "--eval-engines", "threshold",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout == ""
        assert "ticker" in result.stderr
        assert "CRM" in result.stderr

    @pytest.mark.skip(reason="messina/hmm engines hang — runtime bug pending investigation")
    def test_eval_csv_engine_filter(self):
        """--eval-engines filters correctly via CLI."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hmm_futures_analysis.cli",
                "--eval-csv", str(FIXTURE_DIR),
                "--eval-engines", "threshold,messina",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data) == 4
        engines = {r["engine"] for r in data}
        assert engines == {"threshold", "messina"}

    def test_eval_csv_invalid_engine(self):
        """Invalid engine name produces error."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hmm_futures_analysis.cli",
                "--eval-csv", str(FIXTURE_DIR),
                "--eval-engines", "bogus",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_eval_and_single_mutually_exclusive(self):
        """Can't mix eval and single-run flags."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hmm_futures_analysis.cli",
                "--eval-csv", str(FIXTURE_DIR),
                "--ticker", "SPY",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_eval_tickers_and_csv_mutually_exclusive(self):
        """Can't use both --eval-tickers and --eval-csv."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hmm_futures_analysis.cli",
                "--eval-tickers", "SPY",
                "--eval-csv", str(FIXTURE_DIR),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_no_args_error(self):
        """No args at all produces usage error."""
        result = subprocess.run(
            [sys.executable, "-m", "hmm_futures_analysis.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
