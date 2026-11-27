"""Issue #114 (Slice 3) — the --eval-statistical CLI flag.

Regression tests for the missing CLI surface: the flag must parse, dispatch
run_eval_statistical with the parsed tickers/engines, print the
statistical_table markdown report to stderr (or the JSON row array to stdout
with --json), and respect the eval-flag mutual exclusivity rules.

Offline: run_eval_statistical is mocked, so no data loading or engine runs.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from hmm_futures_analysis.cli import main
from hmm_futures_analysis.eval import ALL_ENGINES

ROWS = [
    {
        "ticker": "SPY",
        "engine": "threshold",
        "regime": "bear",
        "fleiss_kappa": -0.052296,
        "bnh_delta": -1.206,
    },
    {
        "ticker": "SPY",
        "engine": "messina",
        "regime": "bear",
        "fleiss_kappa": -0.052296,
        "bnh_delta": -0.731,
    },
    {
        "ticker": "NVDA",
        "engine": "threshold",
        "regime": "sideways",
        "fleiss_kappa": -0.052296,
        "bnh_delta": -0.852,
    },
    {
        "ticker": "NVDA",
        "engine": "messina",
        "regime": "bull",
        "fleiss_kappa": -0.052296,
        "bnh_delta": -0.060,
    },
]


def _run(argv):
    with patch.object(sys, "argv", ["hmm-regime", *argv]):
        main()


class TestEvalStatisticalDispatch:
    """The flag reaches run_eval_statistical and prints the report."""

    @patch("hmm_futures_analysis.cli.run_eval_statistical")
    def test_dispatch_default_engines(self, mock_eval, capsys):
        mock_eval.return_value = ROWS
        _run(["--eval-statistical", "SPY,NVDA"])

        mock_eval.assert_called_once()
        kwargs = mock_eval.call_args.kwargs
        assert kwargs["tickers"] == ("SPY", "NVDA")
        assert kwargs["engines"] == ALL_ENGINES

        captured = capsys.readouterr()
        assert "Cross-engine agreement" in captured.err
        assert "buy-and-hold delta" in captured.err.lower()
        assert captured.out == ""

    @patch("hmm_futures_analysis.cli.run_eval_statistical")
    def test_eval_engines_filters(self, mock_eval, capsys):
        mock_eval.return_value = ROWS
        _run(
            [
                "--eval-statistical",
                "SPY,NVDA",
                "--eval-engines",
                "threshold,messina",
            ]
        )
        kwargs = mock_eval.call_args.kwargs
        assert kwargs["engines"] == ("threshold", "messina")

    @patch("hmm_futures_analysis.cli.run_eval_statistical")
    def test_json_flag_prints_rows_to_stdout(self, mock_eval, capsys):
        mock_eval.return_value = ROWS
        _run(["--eval-statistical", "SPY", "--json"])

        captured = capsys.readouterr()
        import json

        parsed = json.loads(captured.out)
        assert parsed == ROWS
        assert captured.err == ""

    @patch("hmm_futures_analysis.cli.run_eval_statistical")
    def test_min_train_threaded(self, mock_eval, capsys):
        mock_eval.return_value = ROWS
        _run(["--eval-statistical", "SPY", "--min-train", "200"])
        assert mock_eval.call_args.kwargs["min_train"] == 200

    @patch("hmm_futures_analysis.cli.run_eval_statistical")
    def test_cache_flags_threaded(self, mock_eval, capsys):
        mock_eval.return_value = ROWS
        _run(
            [
                "--eval-statistical",
                "SPY",
                "--eval-cache-dir",
                "/tmp/csv-cache",
                "--refresh",
                "--no-cache",
            ]
        )
        kwargs = mock_eval.call_args.kwargs
        assert kwargs["cache_dir"] == "/tmp/csv-cache"
        assert kwargs["refresh"] is True
        assert kwargs["no_cache"] is True


class TestEvalStatisticalExclusivity:
    """--eval-statistical is mutually exclusive with the other input flags."""

    @pytest.mark.parametrize(
        "other",
        [
            ["--ticker", "SPY"],
            ["--csv", "x.csv"],
            ["--eval-tickers", "SPY"],
            ["--eval-csv", "some-dir"],
        ],
    )
    def test_mutually_exclusive(self, other, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(["--eval-statistical", "SPY", *other])
        assert exc.value.code != 0
        assert "mutually exclusive" in capsys.readouterr().err

    def test_no_args_mentions_new_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _run([])
        assert exc.value.code != 0
        assert "--eval-statistical" in capsys.readouterr().err


class TestEvalStatisticalHelp:
    """--help shows the flag in the eval-mode group."""

    def test_help_lists_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--eval-statistical" in out
        # same usage-text format as the other eval flags
        assert "Comma-separated ticker list" in out
        # and it sits in the eval-mode group
        eval_group = out[out.index("eval mode:") :]
        assert "--eval-statistical" in eval_group
