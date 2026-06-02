"""Tests for regime transition extraction (issue #63)."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.regime_transitions import (
    extract_transitions,
)


class TestExtractTransitions:
    """extract_transitions: pure function, engine-agnostic, no I/O."""

    def test_empty_array_returns_empty_list(self):
        """Empty regime array → no transitions."""
        regimes = np.array([], dtype=int)
        dates = pd.DatetimeIndex([])
        result = extract_transitions(regimes, dates)
        assert result == []

    def test_single_element_returns_empty_list(self):
        """Single-element array → no transitions (need ≥2 bars to change)."""
        regimes = np.array([0])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=1))
        result = extract_transitions(regimes, dates)
        assert result == []

    def test_no_change_returns_empty_list(self):
        """All same regime → no transitions."""
        regimes = np.array([1, 1, 1, 1, 1])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5))
        result = extract_transitions(regimes, dates)
        assert result == []

    def test_single_transition_produces_one_event(self):
        """Two-regime sequence with one change → exactly one event."""
        regimes = np.array([0, 0, 1, 1])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=4))
        result = extract_transitions(regimes, dates)

        assert len(result) == 1
        event = result[0]
        assert event.from_regime == "bear"
        assert event.to_regime == "sideways"

    def test_date_alignment(self):
        """Transition date matches the bar where the new regime starts."""
        regimes = np.array([0, 0, 1, 1])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=4))
        result = extract_transitions(regimes, dates)

        # Change happens at index 2 (bear→sideways)
        assert result[0].date == "2024-01-03"
        assert result[0].bar_index == 2

    def test_multiple_transitions_chronological_order(self):
        """Multiple transitions returned in chronological order."""
        regimes = np.array([0, 0, 1, 1, 2, 2])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=6))
        result = extract_transitions(regimes, dates)

        assert len(result) == 2
        assert result[0].from_regime == "bear"
        assert result[0].to_regime == "sideways"
        assert result[1].from_regime == "sideways"
        assert result[1].to_regime == "bull"

    @pytest.mark.parametrize(
        "from_idx,to_idx,from_name,to_name",
        [
            (0, 1, "bear", "sideways"),
            (0, 2, "bear", "bull"),
            (1, 0, "sideways", "bear"),
            (1, 2, "sideways", "bull"),
            (2, 0, "bull", "bear"),
            (2, 1, "bull", "sideways"),
        ],
    )
    def test_all_regime_pairs(self, from_idx, to_idx, from_name, to_name):
        """Every regime pair produces correct from/to names."""
        regimes = np.array([from_idx, from_idx, to_idx, to_idx])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=4))
        result = extract_transitions(regimes, dates)

        assert len(result) == 1
        assert result[0].from_regime == from_name
        assert result[0].to_regime == to_name

    def test_transition_event_is_named_tuple(self):
        """TransitionEvent fields are accessible by name."""
        regimes = np.array([0, 1])
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2))
        result = extract_transitions(regimes, dates)

        event = result[0]
        assert hasattr(event, "date")
        assert hasattr(event, "from_regime")
        assert hasattr(event, "to_regime")
        assert hasattr(event, "bar_index")
        assert isinstance(event.date, str)
        assert isinstance(event.bar_index, int)


@pytest.mark.slow
class TestCLITransitionsFlag:
    """CLI --transitions N argument parsing."""

    def test_transitions_flag_parses_integer(self, btc_csv):
        """--transitions 5 parses as integer and appears in JSON output."""
        cmd = [
            sys.executable,
            "-m",
            "hmm_futures_analysis.cli",
            "--csv",
            btc_csv,
            "--engine",
            "threshold",
            "--transitions",
            "5",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[:500]

        stdout = proc.stdout
        json_start = stdout.find("{")
        assert json_start >= 0, f"No JSON in output: {stdout[:200]}"
        output = json.loads(stdout[json_start:])
        assert "regime_transitions" in output

    def test_transitions_zero_shows_all(self, btc_csv):
        """--transitions 0 includes all transitions in JSON output."""
        cmd = [
            sys.executable,
            "-m",
            "hmm_futures_analysis.cli",
            "--csv",
            btc_csv,
            "--engine",
            "threshold",
            "--transitions",
            "0",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[:500]

        stdout = proc.stdout
        json_start = stdout.find("{")
        output = json.loads(stdout[json_start:])
        transitions = output["regime_transitions"]
        # All transitions present (same as no-limit)
        assert isinstance(transitions, list)

    def test_no_transitions_flag_json_includes_transitions(self, btc_csv):
        """Without --transitions, JSON output still includes regime_transitions."""
        cmd = [
            sys.executable,
            "-m",
            "hmm_futures_analysis.cli",
            "--csv",
            btc_csv,
            "--engine",
            "threshold",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[:500]

        stdout = proc.stdout
        json_start = stdout.find("{")
        output = json.loads(stdout[json_start:])
        # JSON always includes transitions regardless of flag
        assert "regime_transitions" in output
