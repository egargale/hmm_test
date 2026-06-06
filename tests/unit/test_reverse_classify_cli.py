"""Tests for --reverse-classify CLI flag (Issue #102)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _run_regime(*args, timeout=120):
    """Run hmm-regime CLI with args, return CompletedProcess."""
    cmd = [sys.executable, "-m", "hmm_futures_analysis.cli"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=timeout,
    )


class TestReverseClassifyCLI:
    """--reverse-classify flag parses correctly and flows through pipeline."""

    @pytest.fixture(scope="class")
    def btc_csv(self):
        p = ROOT / "test_data" / "BTC.csv"
        if not p.exists():
            pytest.skip("BTC.csv not available")
        return str(p)

    def test_flag_parses_with_hmm_engine(self, btc_csv):
        """--reverse-classify with hmm engine produces valid JSON with warning."""
        result = _run_regime(
            "--csv", btc_csv,
            "--engine", "hmm",
            "--json",
            "--reverse-classify",
        )
        assert result.returncode == 0, (
            f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        data = json.loads(result.stdout)
        assert data["engine_info"].get("reverse_classify") is True
        assert data["engine_info"].get("lookahead_bias_warning") is True

    def test_flag_noop_with_threshold(self, btc_csv):
        """--reverse-classify with threshold engine is a no-op."""
        result_forward = _run_regime(
            "--csv", btc_csv,
            "--engine", "threshold",
            "--json",
        )
        result_reverse = _run_regime(
            "--csv", btc_csv,
            "--engine", "threshold",
            "--json",
            "--reverse-classify",
        )
        assert result_forward.returncode == 0
        assert result_reverse.returncode == 0

        forward = json.loads(result_forward.stdout)
        reverse = json.loads(result_reverse.stdout)

        # Threshold should be identical regardless of flag
        assert forward["current_regime"] == reverse["current_regime"]
        assert forward["signal"] == reverse["signal"]
        assert reverse["engine_info"].get("lookahead_bias_warning") is None

    def test_terminal_shows_lookahead_warning(self, btc_csv):
        """Terminal output shows LOOKAHEAD BIAS warning with --reverse-classify."""
        result = _run_regime(
            "--csv", btc_csv,
            "--engine", "hmm",
            "--reverse-classify",
        )
        assert result.returncode == 0
        assert "LOOKAHEAD BIAS" in result.stderr
        assert "lookahead" in result.stderr.lower()

    def test_terminal_no_warning_in_forward_mode(self, btc_csv):
        """Terminal output has no lookahead warning in forward mode."""
        result = _run_regime(
            "--csv", btc_csv,
            "--engine", "hmm",
        )
        assert result.returncode == 0
        assert "LOOKAHEAD BIAS" not in result.stderr
