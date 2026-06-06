"""Tests for CLI cache flags (--cache-dir, --refresh, --no-cache)."""

from __future__ import annotations

from hmm_futures_analysis.cli import main
from unittest.mock import patch


class TestCliCacheFlags:
    """CLI parses and threads cache flags correctly."""

    @patch("hmm_futures_analysis.cli.load_prices")
    @patch("hmm_futures_analysis.cli.pipeline_run")
    def test_cache_dir_threads_to_load_prices(self, mock_pipeline, mock_load):
        """--cache-dir is passed to load_prices in single-run mode."""
        import pandas as pd
        import sys

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_load.return_value = (
            pd.Series([100, 101, 102, 103, 104], index=dates),
            None,
            "SPY",
        )
        mock_pipeline.return_value._asdict.return_value = {
            "source": "SPY",
            "engine": "threshold",
            "current_regime": {"name": "bull", "index": 2},
            "signal": 0.5,
            "dates": {"start": "2024-01-01", "end": "2024-01-05"},
            "engine_info": {"method": "threshold", "features": 1},
            "regime_counts": {"bear": 0, "sideways": 0, "bull": 5},
            "next_state_probabilities": {"bear": 0.1, "sideways": 0.2, "bull": 0.7},
            "persistence_diagonal": {"bear": 0.9, "sideways": 0.8, "bull": 0.95},
            "transition_matrix": [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
            "stationary_distribution": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
            "forecast": {
                "1_step": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
                "5_step": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
                "20_step": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
            },
            "walk_forward": {
                "sharpe": 1.0,
                "max_drawdown": -0.1,
                "n_trades": 5,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "total_return": 0.05,
            },
            "disclaimer": "test",
            "regime_transitions": None,
            "duration_forecast": None,
        }

        with patch.object(sys, "argv", ["hmm-regime", "--ticker", "SPY", "--cache-dir", "/tmp/cache", "--json"]):
            main()

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs["cache_dir"] == "/tmp/cache"
        assert call_kwargs["refresh"] is False
        assert call_kwargs["no_cache"] is False

    @patch("hmm_futures_analysis.cli.load_prices")
    @patch("hmm_futures_analysis.cli.pipeline_run")
    def test_refresh_and_no_cache_threaded(self, mock_pipeline, mock_load):
        """--refresh and --no-cache are passed to load_prices."""
        import pandas as pd
        import sys

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_load.return_value = (
            pd.Series([100, 101, 102, 103, 104], index=dates),
            None,
            "SPY",
        )
        mock_pipeline.return_value._asdict.return_value = {
            "source": "SPY",
            "engine": "threshold",
            "current_regime": {"name": "bull", "index": 2},
            "signal": 0.5,
            "dates": {"start": "2024-01-01", "end": "2024-01-05"},
            "engine_info": {"method": "threshold", "features": 1},
            "regime_counts": {"bear": 0, "sideways": 0, "bull": 5},
            "next_state_probabilities": {"bear": 0.1, "sideways": 0.2, "bull": 0.7},
            "persistence_diagonal": {"bear": 0.9, "sideways": 0.8, "bull": 0.95},
            "transition_matrix": [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
            "stationary_distribution": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
            "forecast": {
                "1_step": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
                "5_step": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
                "20_step": {"bear": 0.2, "sideways": 0.3, "bull": 0.5},
            },
            "walk_forward": {
                "sharpe": 1.0,
                "max_drawdown": -0.1,
                "n_trades": 5,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "total_return": 0.05,
            },
            "disclaimer": "test",
            "regime_transitions": None,
            "duration_forecast": None,
        }

        with patch.object(sys, "argv", ["hmm-regime", "--ticker", "SPY", "--refresh", "--no-cache", "--json"]):
            main()

        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs["refresh"] is True
        assert call_kwargs["no_cache"] is True

    @patch("hmm_futures_analysis.cli.run_eval_tickers")
    def test_eval_cache_dir_alias(self, mock_run_eval):
        """--eval-cache-dir maps to cache_dir in eval mode."""
        import sys

        mock_run_eval.return_value = []

        with patch.object(
            sys,
            "argv",
            [
                "hmm-regime",
                "--eval-tickers",
                "SPY",
                "--eval-cache-dir",
                "/legacy/path",
                "--json",
            ],
        ):
            main()

        call_kwargs = mock_run_eval.call_args.kwargs
        assert call_kwargs["cache_dir"] == "/legacy/path"
