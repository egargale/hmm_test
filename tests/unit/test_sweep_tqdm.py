"""Tests for tqdm progress bars in hmm_sweep.py (issue #90).

Verifies that all 5 phases use tqdm with correct desc strings
and nested structure (outer=position 0, inner=position 1).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _dummy_summary(ticker, label=""):
    """Fast dummy result that satisfies extract_summary shape."""
    return {
        "ticker": ticker,
        "config_label": label,
        "params": {
            "n_states": 3,
            "pca_variance": None,
            "dwell_bars": 0,
            "hysteresis_delta": 0.0,
        },
        "regime": "bull",
        "signal": 0.5,
        "sharpe": 1.0,
        "max_drawdown": -0.1,
        "n_trades": 10,
        "win_rate": 0.6,
        "profit_factor": 1.2,
        "total_return": 0.3,
        "wall_seconds": 0.01,
        "engine_info": {"features": "?", "resolved_n_states": 3},
    }


def _make_spy_tqdm():
    """Return a tqdm spy that passes through iterables and records calls."""
    calls = []

    def spy_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    spy_tqdm.calls = calls
    return spy_tqdm


class TestHmmSweepTqdm:
    """Verify hmm_sweep.py uses tqdm for all 5 phases."""

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_phase1_outer_bar(self, mock_tqdm_cls, mock_run_config):
        """Phase 1: outer bar wraps ticker_csvs with desc='Phase 1: defaults'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        phase1_outer = [
            c for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("desc") == "Phase 1: defaults"
        ]
        assert len(phase1_outer) == 1
        assert phase1_outer[0].kwargs.get("position") == 0
        assert phase1_outer[0].kwargs.get("leave") is True

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_phase2a_outer_bar(self, mock_tqdm_cls, mock_run_config):
        """Phase 2a: outer bar desc='Phase 2a: n_states'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        phase2a_outer = [
            c for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("desc") == "Phase 2a: n_states"
        ]
        assert len(phase2a_outer) == 1
        assert phase2a_outer[0].kwargs.get("position") == 0

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_phase2a_inner_bar(self, mock_tqdm_cls, mock_run_config):
        """Phase 2a: inner bar wraps ns_vals with leave=False."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        inner = [
            c for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("desc", "").startswith("  ") and c.kwargs.get("position") == 1
        ]
        # At least one inner bar (Phase 2a has one per ticker)
        assert len(inner) >= 1
        for c in inner:
            assert c.kwargs.get("leave") is False

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_phase2b_outer_bar(self, mock_tqdm_cls, mock_run_config):
        """Phase 2b: outer bar desc='Phase 2b: pca_variance'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        phase2b = [
            c for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("desc") == "Phase 2b: pca_variance"
        ]
        assert len(phase2b) == 1

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_phase2c_outer_bar(self, mock_tqdm_cls, mock_run_config):
        """Phase 2c: outer bar desc='Phase 2c: dwell_bars'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        phase2c = [
            c for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("desc") == "Phase 2c: dwell_bars"
        ]
        assert len(phase2c) == 1

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_phase2d_outer_bar(self, mock_tqdm_cls, mock_run_config):
        """Phase 2d: outer bar desc='Phase 2d: hysteresis_delta'."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        phase2d = [
            c for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("desc") == "Phase 2d: hysteresis_delta"
        ]
        assert len(phase2d) == 1

    @patch("hmm_sweep.run_config")
    @patch("hmm_sweep.tqdm")
    def test_all_five_phases_present(self, mock_tqdm_cls, mock_run_config):
        """All 5 phase outer bars are present."""
        mock_tqdm_cls.side_effect = lambda iterable, **kwargs: iterable
        mock_run_config.return_value = _dummy_summary("TEST")

        from hmm_sweep import main
        main()

        expected_descs = [
            "Phase 1: defaults",
            "Phase 2a: n_states",
            "Phase 2b: pca_variance",
            "Phase 2c: dwell_bars",
            "Phase 2d: hysteresis_delta",
        ]
        actual_descs = [
            c.kwargs.get("desc")
            for c in mock_tqdm_cls.call_args_list
            if c.kwargs.get("position") == 0
        ]
        for desc in expected_descs:
            assert desc in actual_descs, f"Missing outer bar: {desc}"
