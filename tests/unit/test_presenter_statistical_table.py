"""Issue #114 (Slice 3) — fmt=\"statistical_table\" in presenter.format_eval.

Regression tests for the missing statistical report formatter: the
cross-engine agreement summary (one row per ticker, engine votes, majority
column) and the buy-and-hold delta grid, per ADR-0028.

presenter.py must stay pure: these tests exercise a pure function only.
"""

from __future__ import annotations

import json

import pytest

from hmm_futures_analysis.presenter import format_eval

# Minimal rows carrying just what the statistical formatter reads.
ROWS = [
    {
        "ticker": "AAA",
        "engine": "threshold",
        "regime": "bear",
        "fleiss_kappa": 0.5,
        "bnh_delta": -1.206,
    },
    {
        "ticker": "AAA",
        "engine": "messina",
        "regime": "bear",
        "fleiss_kappa": 0.5,
        "bnh_delta": 1.5,
    },
    {
        "ticker": "AAA",
        "engine": "fshmm",
        "regime": "sideways",
        "fleiss_kappa": 0.5,
        "bnh_delta": 0.0,
    },
    {
        "ticker": "BBB",
        "engine": "threshold",
        "regime": "bear",
        "fleiss_kappa": 0.5,
        "bnh_delta": -0.5,
    },
    {
        "ticker": "BBB",
        "engine": "messina",
        "regime": "sideways",
        "fleiss_kappa": 0.5,
        "bnh_delta": 0.25,
    },
    {
        "ticker": "BBB",
        "engine": "fshmm",
        "regime": "bull",
        "fleiss_kappa": 0.5,
        "bnh_delta": 2.0,
    },
]


class TestStatisticalTable:
    """Agreement summary + buy-and-hold delta grid."""

    def test_agreement_table_has_engine_and_majority_columns(self):
        out = format_eval(ROWS, fmt="statistical_table")
        # header carries every engine plus a majority column
        assert "ticker" in out
        assert "majority" in out
        for engine in ("threshold", "messina", "fshmm"):
            assert engine in out

    def test_agreement_rows_are_per_ticker_with_votes(self):
        out = format_eval(ROWS, fmt="statistical_table")
        aaa_line = next(
            line for line in out.splitlines() if line.startswith("AAA")
        )
        bbb_line = next(
            line for line in out.splitlines() if line.startswith("BBB")
        )
        # AAA votes: bear, bear, sideways -> majority bear
        assert "bear" in aaa_line and "sideways" in aaa_line
        assert aaa_line.split()[-1] == "bear"
        # BBB votes: bear, sideways, bull -> three-way tie
        assert bbb_line.split()[-1] == "tie"

    def test_fleiss_kappa_line(self):
        out = format_eval(ROWS, fmt="statistical_table")
        assert "Fleiss' kappa = 0.500000" in out

    def test_fleiss_kappa_none_renders_na(self):
        rows = [dict(r, fleiss_kappa=None) for r in ROWS]
        out = format_eval(rows, fmt="statistical_table")
        assert "Fleiss' kappa = n/a" in out

    def test_fleiss_kappa_missing_omits_line(self):
        rows = [{k: v for k, v in r.items() if k != "fleiss_kappa"} for r in ROWS]
        out = format_eval(rows, fmt="statistical_table")
        assert "Fleiss' kappa" not in out

    def test_bnh_delta_grid_three_decimals(self):
        out = format_eval(ROWS, fmt="statistical_table")
        assert "buy-and-hold delta" in out.lower()
        # threshold row: AAA -1.206, BBB -0.500
        threshold_line = next(
            line for line in out.splitlines() if line.startswith("threshold")
        )
        assert "-1.206" in threshold_line
        assert "-0.500" in threshold_line

    def test_empty_results(self):
        assert format_eval([], fmt="statistical_table") == "No results."

    def test_missing_delta_renders_na(self):
        rows = [{k: v for k, v in r.items() if k != "bnh_delta"} for r in ROWS]
        out = format_eval(rows, fmt="statistical_table")
        assert "N/A" in out


class TestExistingFormatsUnchanged:
    """fmt=\"table\" and fmt=\"json\" paths stay byte-identical."""

    def test_table_empty(self):
        assert format_eval([], fmt="table") == "No results."

    def test_json_is_row_array(self):
        assert format_eval(ROWS, fmt="json") == (
            json.dumps(ROWS, indent=2, allow_nan=False) + "\n"
        )

    def test_unknown_fmt_still_raises(self):
        with pytest.raises(ValueError):
            format_eval(ROWS, fmt="bogus")
