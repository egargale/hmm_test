"""Issue #114 (Slice 3) — run_eval_statistical and the shared statistical metrics.

Regression tests for the missing statistical evaluation surface:
``run_eval_statistical`` must exist in the eval module, return the same dict
shape as ``run_eval_tickers`` plus ``fleiss_kappa`` (panel scalar) and
``bnh_delta`` per row, and the shared metric implementations must reproduce
the Slice 2 harness values pinned in runs/statistical_report.txt (ADR-0027).

Offline: load_prices / pipeline_run are mocked; committed test_data/ CSVs are
used only as name-resolution fixtures.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.eval import (
    _make_config,
    buy_and_hold_stats,
    fleiss_components,
    fleiss_kappa,
    run_eval_statistical,
)

# ── Helpers ─────────────────────────────────────────────────────────────

_REGIME_INDEX = {"bear": 0, "sideways": 1, "bull": 2}


def _fake_result(source: str, engine: str, regime: str, sharpe: float = 1.0):
    """Minimal pipeline result accepted by eval._extract_summary."""
    return SimpleNamespace(
        source=source,
        engine=engine,
        current_regime={"name": regime, "index": _REGIME_INDEX[regime]},
        signal=0.0,
        walk_forward={
            "sharpe": sharpe,
            "max_drawdown": -0.1,
            "n_trades": 3,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "total_return": 0.2,
        },
    )


def _quiet_tqdm():
    return patch(
        "hmm_futures_analysis.eval.tqdm", side_effect=lambda it, **kw: it
    )


def _run_with_mocks(regimes, tickers, engines, prices=None, **kwargs):
    """run_eval_statistical with mocked data loading and pipeline.

    ``regimes`` maps (ticker, engine) -> regime name.
    """
    if prices is None:
        prices = pd.Series(
            np.linspace(100.0, 130.0, 10),
            index=pd.date_range("2024-01-01", periods=10, freq="B"),
        )
    config_to_engine = {
        type(_make_config(name)).__name__: name for name in engines
    }

    def _pipeline(**kw):
        engine = config_to_engine[type(kw["engine_config"]).__name__]
        regime = regimes[(kw["source"], engine)]
        return _fake_result(kw["source"], engine, regime)

    with (
        patch(
            "hmm_futures_analysis.eval.load_prices",
            return_value=(prices, None, "fixture"),
        ) as mock_load,
        patch(
            "hmm_futures_analysis.eval.pipeline_run", side_effect=_pipeline
        ),
        _quiet_tqdm(),
    ):
        results = run_eval_statistical(tickers, engines, **kwargs)
    return results, mock_load


# ── fleiss_kappa / fleiss_components ────────────────────────────────────


class TestFleissKappa:
    """Shared implementation matches the Slice 2 harness contract."""

    def test_kappa_matches_committed_report_values(self):
        """Vote matrix of runs/statistical_report.txt:87-91.

        The committed report pins P_bar = 0.340000, P_e = 0.372800 and
        kappa = -0.052296 (runs/statistical_report.txt:93-94).
        """
        counts = [
            [3, 1, 1],  # SPY:  bear x3, sideways, bull
            [0, 2, 3],  # NVDA
            [1, 3, 1],  # BTC
            [0, 3, 2],  # KO
            [1, 3, 1],  # XLE
        ]
        p_bar, p_e, kappa = fleiss_components(counts)
        assert p_bar == pytest.approx(0.340000, abs=1e-9)
        assert p_e == pytest.approx(0.372800, abs=1e-9)
        assert kappa == pytest.approx(-0.052296, abs=1e-4)

    def test_kappa_hand_computed_value(self):
        """Counts [[1,1,0],[0,0,2]] -> P_bar=0.5, P_e=0.375, kappa=0.2."""
        p_bar, p_e, kappa = fleiss_components([[1, 1, 0], [0, 0, 2]])
        assert p_bar == pytest.approx(0.5)
        assert p_e == pytest.approx(0.375)
        assert kappa == pytest.approx(0.2)
        assert fleiss_kappa([[1, 1, 0], [0, 0, 2]]) == pytest.approx(0.2)

    def test_kappa_undefined_raises(self):
        """Chance agreement == 1 -> ValueError (harness contract)."""
        with pytest.raises(ValueError):
            fleiss_kappa([[0, 0, 2], [0, 0, 3]])

    def test_single_rater_raises(self):
        """One rater per subject -> ValueError (harness contract)."""
        with pytest.raises(ValueError):
            fleiss_kappa([[1, 0, 0]])


# ── buy_and_hold_stats ──────────────────────────────────────────────────


class TestBuyAndHoldStats:
    """B&H stats use the engine window prices[min_train:] (ADR-0027)."""

    PRICES = np.array([100.0, 110.0, 105.0, 108.0, 107.0, 112.0, 111.0])

    def test_windowed_formula(self):
        """Sharpe on the window only: mean/std(ddof=1) * sqrt(252)."""
        min_train = 2
        stats = buy_and_hold_stats(self.PRICES, min_train=min_train)

        px = self.PRICES[min_train:]
        rets = np.diff(px) / px[:-1]
        expected_sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252)
        expected_total = px[-1] / px[0] - 1.0
        expected_dd = float(np.min(px / np.maximum.accumulate(px) - 1))

        assert stats["sharpe"] == pytest.approx(expected_sharpe)
        assert stats["total_return"] == pytest.approx(expected_total)
        assert stats["max_drawdown"] == pytest.approx(expected_dd)

    def test_window_not_full_series(self):
        """Windowed Sharpe must differ from the full-CSV B&H Sharpe."""
        min_train = 2
        windowed = buy_and_hold_stats(self.PRICES, min_train=min_train)

        rets = np.diff(self.PRICES) / self.PRICES[:-1]
        full = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252)

        assert windowed["sharpe"] != pytest.approx(full), (
            "test fixture must discriminate the engine window from the "
            "full series"
        )

    def test_empty_engine_window_raises(self):
        """Fewer than 2 prices after min_train is a clear ValueError."""
        with pytest.raises(ValueError, match="min_train"):
            buy_and_hold_stats(self.PRICES, min_train=len(self.PRICES))


# ── run_eval_statistical ────────────────────────────────────────────────


class TestRunEvalStatistical:
    """Rows carry the run_eval_tickers shape plus two fields."""

    REGIMES = {
        ("AAA", "threshold"): "bear",
        ("AAA", "messina"): "sideways",
        ("BBB", "threshold"): "bull",
        ("BBB", "messina"): "bull",
    }

    def test_row_schema(self):
        results, _ = _run_with_mocks(
            self.REGIMES,
            tickers=("AAA", "BBB"),
            engines=("threshold", "messina"),
            min_train=2,
        )
        base_keys = set(_extract_summary_keys())
        for row in results:
            assert base_keys <= set(row.keys())
            assert set(row.keys()) - base_keys == {"fleiss_kappa", "bnh_delta"}
        # ticker-major order, engines within ticker
        assert [(r["ticker"], r["engine"]) for r in results] == [
            ("AAA", "threshold"),
            ("AAA", "messina"),
            ("BBB", "threshold"),
            ("BBB", "messina"),
        ]

    def test_panel_kappa_shared_scalar(self):
        results, _ = _run_with_mocks(
            self.REGIMES,
            tickers=("AAA", "BBB"),
            engines=("threshold", "messina"),
            min_train=2,
        )
        kappas = {r["fleiss_kappa"] for r in results}
        # votes [[1,1,0],[0,0,2]] -> kappa 0.2, identical on every row
        assert len(kappas) == 1
        assert list(kappas)[0] == pytest.approx(0.2)

    def test_bnh_delta_per_row(self):
        prices = pd.Series(
            TestBuyAndHoldStats.PRICES,
            index=pd.date_range("2024-01-01", periods=7, freq="B"),
        )
        results, _ = _run_with_mocks(
            self.REGIMES,
            tickers=("AAA", "BBB"),
            engines=("threshold", "messina"),
            prices=prices,
            min_train=2,
        )
        px = TestBuyAndHoldStats.PRICES[2:]
        rets = np.diff(px) / px[:-1]
        bh_sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252)
        for row in results:
            assert row["bnh_delta"] == pytest.approx(row["sharpe"] - bh_sharpe)

    def test_fixture_csv_preferred_over_ticker(self, tmp_path, monkeypatch):
        """Ticker resolves to ./test_data/<T>.csv before the yfinance path."""
        monkeypatch.chdir(tmp_path)
        fixture = tmp_path / "test_data" / "FOO.csv"
        fixture.parent.mkdir()
        fixture.write_text("Date,Close\n2024-01-01,100\n")

        _, mock_load = _run_with_mocks(
            {("FOO", "threshold"): "bear", ("FOO", "messina"): "bull"},
            tickers=("FOO",),
            engines=("threshold", "messina"),
            min_train=2,
        )
        assert mock_load.call_args.kwargs["csv"] == str(fixture)
        assert "ticker" not in mock_load.call_args.kwargs

    def test_falls_back_to_ticker_path(self, tmp_path, monkeypatch):
        """No committed fixture -> the yfinance ticker path (ADR-0004)."""
        monkeypatch.chdir(tmp_path)
        _, mock_load = _run_with_mocks(
            {
                ("NOSUCHTICKER", "threshold"): "bear",
                ("NOSUCHTICKER", "messina"): "bull",
            },
            tickers=("NOSUCHTICKER",),
            engines=("threshold", "messina"),
            min_train=2,
        )
        assert mock_load.call_args.kwargs["ticker"] == "NOSUCHTICKER"

    def test_kappa_none_when_undefined(self):
        """Degenerate panels (<2 engines, or unanimity) get kappa None."""
        results, _ = _run_with_mocks(
            {("AAA", "threshold"): "bull"},
            tickers=("AAA",),
            engines=("threshold",),
            min_train=2,
        )
        assert results[0]["fleiss_kappa"] is None

        unanimous, _ = _run_with_mocks(
            {
                ("AAA", "threshold"): "bull",
                ("AAA", "messina"): "bull",
            },
            tickers=("AAA",),
            engines=("threshold", "messina"),
            min_train=2,
        )
        assert unanimous[0]["fleiss_kappa"] is None

    def test_cache_flags_threaded_to_load_prices(self, tmp_path, monkeypatch):
        """--eval-cache-dir/--refresh/--no-cache reach the ticker loader."""
        monkeypatch.chdir(tmp_path)
        _, mock_load = _run_with_mocks(
            {
                ("NOSUCHTICKER", "threshold"): "bear",
                ("NOSUCHTICKER", "messina"): "bull",
            },
            tickers=("NOSUCHTICKER",),
            engines=("threshold", "messina"),
            min_train=2,
            cache_dir="/tmp/cache",
            refresh=True,
            no_cache=False,
        )
        kwargs = mock_load.call_args.kwargs
        assert kwargs["cache_dir"] == "/tmp/cache"
        assert kwargs["refresh"] is True
        assert kwargs["no_cache"] is False


def _extract_summary_keys():
    return [
        "ticker",
        "engine",
        "regime",
        "signal",
        "sharpe",
        "max_drawdown",
        "n_trades",
        "win_rate",
        "profit_factor",
        "total_return",
        "wall_seconds",
    ]


class TestPipelineInvocationContract:
    """Pin the pipeline_run invocation convention (ADR-0028).

    ``run_eval_statistical`` calls ``pipeline_run`` with the CLI
    single-run filter defaults (``dwell_bars="auto"``,
    ``hysteresis_delta="auto"``) so its rows, votes and ``bnh_delta``
    match the Slice 2 harness cells (which run the CLI single-run path).
    """

    def test_filters_use_cli_single_run_defaults(self):
        prices = pd.Series(
            np.linspace(100.0, 130.0, 260),
            index=pd.date_range("2024-01-01", periods=260, freq="B"),
        )
        with (
            patch(
                "hmm_futures_analysis.eval.load_prices",
                return_value=(prices, None, "fixture"),
            ),
            patch(
                "hmm_futures_analysis.eval.pipeline_run",
                side_effect=lambda **kw: _fake_result(
                    kw["source"], "threshold", "bull"
                ),
            ) as mock_pipeline,
            _quiet_tqdm(),
        ):
            run_eval_statistical(("AAA",), engines=("threshold",))
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["dwell_bars"] == "auto"
        assert kwargs["hysteresis_delta"] == "auto"
        assert kwargs["min_train"] == 252
