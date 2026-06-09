"""Tests for limit_transitions helper (issue #101)."""

import json
import subprocess
import sys

import pytest


def _make_transitions(n):
    """Create N fake transition dicts (chronological order, oldest first)."""
    return [
        {
            "date": f"2024-01-{i+1:02d}",
            "from_regime": "bear",
            "to_regime": "bull",
            "bar_index": i + 1,
        }
        for i in range(n)
    ]


from hmm_futures_analysis.presenter import limit_transitions as _apply_transitions_limit


class TestApplyTransitionsLimitPassthrough:
    """limit=None returns the list unchanged."""

    def test_none_limit_returns_same_list(self):
        transitions = _make_transitions(5)
        result = _apply_transitions_limit(transitions, limit=None)
        assert result is transitions

    def test_none_limit_empty_list(self):
        result = _apply_transitions_limit([], limit=None)
        assert result == []


class TestApplyTransitionsLimitZero:
    """limit=0 returns all transitions reversed (newest first)."""

    def test_zero_returns_all_reversed(self):
        transitions = _make_transitions(5)
        result = _apply_transitions_limit(transitions, limit=0)
        assert len(result) == 5
        assert result[0]["date"] == "2024-01-05"
        assert result[4]["date"] == "2024-01-01"

    def test_zero_returns_new_list(self):
        transitions = _make_transitions(3)
        result = _apply_transitions_limit(transitions, limit=0)
        assert result is not transitions

    def test_zero_empty_list(self):
        result = _apply_transitions_limit([], limit=0)
        assert result == []


class TestApplyTransitionsLimitPositiveN:
    """limit=N>0 returns N most recent transitions, newest first."""

    def test_one_returns_most_recent(self):
        transitions = _make_transitions(5)
        result = _apply_transitions_limit(transitions, limit=1)
        assert len(result) == 1
        assert result[0]["date"] == "2024-01-05"
        assert result[0]["bar_index"] == 5

    def test_three_of_five(self):
        transitions = _make_transitions(5)
        result = _apply_transitions_limit(transitions, limit=3)
        assert len(result) == 3
        assert result[0]["date"] == "2024-01-05"
        assert result[1]["date"] == "2024-01-04"
        assert result[2]["date"] == "2024-01-03"

    def test_limit_exceeds_length(self):
        """N > len(transitions) returns all reversed, no error."""
        transitions = _make_transitions(3)
        result = _apply_transitions_limit(transitions, limit=10)
        assert len(result) == 3
        assert result[0]["date"] == "2024-01-03"

    def test_positive_returns_new_list(self):
        transitions = _make_transitions(3)
        result = _apply_transitions_limit(transitions, limit=2)
        assert result is not transitions


@pytest.mark.slow
class TestCLITransitionsJSON:
    """CLI --transitions with --json: filtering and ordering."""

    def test_json_transitions_one_single_element(self, btc_csv):
        """--json --transitions 1 → regime_transitions has exactly 1 entry."""
        cmd = [
            sys.executable,
            "-m",
            "hmm_futures_analysis.cli",
            "--csv",
            btc_csv,
            "--engine",
            "threshold",
            "--transitions",
            "1",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[:500]

        json_start = proc.stdout.find("{")
        assert json_start >= 0
        output = json.loads(proc.stdout[json_start:])

        transitions = output["regime_transitions"]
        assert isinstance(transitions, list)
        assert len(transitions) == 1

    def test_json_transitions_zero_all_reversed(self, btc_csv):
        """--json --transitions 0 → all transitions, newest first."""
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

        json_start = proc.stdout.find("{")
        assert json_start >= 0
        output = json.loads(proc.stdout[json_start:])

        transitions = output["regime_transitions"]
        assert isinstance(transitions, list)
        assert len(transitions) > 1
        # Verify newest-first ordering (bar_index descending)
        indices = [t["bar_index"] for t in transitions]
        assert indices == sorted(indices, reverse=True)

    def test_json_no_transitions_flag_newest_first(self, btc_csv):
        """--json without --transitions → all transitions, newest first."""
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

        json_start = proc.stdout.find("{")
        assert json_start >= 0
        output = json.loads(proc.stdout[json_start:])

        transitions = output["regime_transitions"]
        assert isinstance(transitions, list)
        assert len(transitions) > 1
        # Verify newest-first ordering (bar_index descending)
        indices = [t["bar_index"] for t in transitions]
        assert indices == sorted(indices, reverse=True)
