"""Tests for opt-in signal-scaled walk-forward position sizing (Issue #85).

All fixtures are synthetic: no HMM fitting is required.  The position model is
direction-gated: the filtered regime decides the allowed direction (-1/0/+1),
and the position magnitude is ``abs(clip(P(bull) - P(bear), -1, 1))``.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.backtesting.performance_metrics import calculate_sharpe_ratio
from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig, ThresholdConfig
from hmm_futures_analysis.regime.engine_protocol import ClassifyOutput
from hmm_futures_analysis.regime.pipeline import run as pipeline_run
from hmm_futures_analysis.regime.walk_forward import (
    _STATE_MAP,
    _compute_trade_stats,
    _empty_result,
    _sized_position,
    _validate_sizing_args,
    _walk_forward_positions,
    walk_forward_backtest,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate a deterministic price series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(0.0001, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + returns), index=dates, name="close")


def _returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change(fill_method=None).dropna()


def _make_regimes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, n).astype(int)


def _make_posteriors(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    post = rng.random((n, 3))
    return post / post.sum(axis=1, keepdims=True)


def _build_regimes(
    min_train: int, regimes_after_train: list[int], n_bars: int
) -> np.ndarray:
    all_regimes = [1] * min_train + regimes_after_train
    while len(all_regimes) < n_bars:
        all_regimes.append(1)
    return np.array(all_regimes[:n_bars], dtype=int)


def _row(bear: float, sideways: float, bull: float) -> np.ndarray:
    return np.array([bear, sideways, bull], dtype=float)


def _row_for_signal(signal: float) -> np.ndarray:
    """Build a posterior row whose P(bull) - P(bear) equals *signal*."""
    return _row((1.0 - signal) / 2.0, 0.0, (1.0 + signal) / 2.0)


# Frozen snapshot captured from the pre-change binary implementation on the
# deterministic fixture below (git master 94835f5, pre-Issue-85).
_BINARY_SNAPSHOT = {
    "sharpe": 1.5912409619783923,
    "max_drawdown": -0.08628499972478848,
    "n_trades": 53,
    "win_rate": 0.6037735849056604,
    "profit_factor": 2.0401624210893825,
    "total_return": 0.2382874794707075,
}
_BINARY_POSITION_SUM = 29
_BINARY_POSITION_HASH = 3815


def _frozen_fixture():
    prices = _make_prices()
    n = len(prices) - 1
    regimes = _make_regimes(n, seed=0)
    posteriors = _make_posteriors(n, seed=1)
    return prices, regimes, posteriors


# ---------------------------------------------------------------------------
# Position formula
# ---------------------------------------------------------------------------


class TestPositionFormula:
    def test_strong_bull_long_is_full(self):
        assert _sized_position(_row_for_signal(0.90), 1, 0.0) == pytest.approx(0.90)

    def test_weak_bull_long_is_proportional(self):
        assert _sized_position(_row_for_signal(0.20), 1, 0.0) == pytest.approx(0.20)

    def test_strong_bear_short_is_full(self):
        assert _sized_position(_row_for_signal(-0.90), -1, 0.0) == pytest.approx(-0.90)

    def test_bull_label_with_bearish_signal_is_flat(self):
        assert _sized_position(_row_for_signal(-0.40), 1, 0.0) == 0.0

    def test_bear_label_with_bullish_signal_is_flat(self):
        assert _sized_position(_row_for_signal(0.40), -1, 0.0) == 0.0

    def test_sideways_label_is_always_flat(self):
        assert _sized_position(_row(0.0, 0.0, 1.0), 0, 0.0) == 0.0
        assert _sized_position(_row(1.0, 0.0, 0.0), 0, 0.0) == 0.0

    def test_overconfident_signal_is_clipped(self):
        assert _sized_position(_row(0.0, 0.0, 5.0), 1, 0.0) == pytest.approx(1.0)
        assert _sized_position(_row(5.0, 0.0, 0.0), -1, 0.0) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "row",
        [
            _row(np.nan, 0.1, 0.9),
            _row(0.1, 0.1, np.inf),
            _row(np.inf, 0.1, 0.1),
        ],
    )
    def test_non_finite_signal_is_flat(self, row):
        assert _sized_position(row, 1, 0.0) == 0.0
        assert _sized_position(row, -1, 0.0) == 0.0

    def test_negative_zero_is_normalised_to_zero(self):
        # directional <= 0 returns a genuine +0.0, not -0.0.
        position = _sized_position(_row(0.5, 0.5, 0.5), 1, 0.0)
        assert position == 0.0
        assert np.signbit(position) is np.False_


class TestPositionIntegration:
    def test_signal_positions_are_float_and_bounded(self):
        prices = _make_prices(n=120)
        returns = _returns(prices)
        n = len(returns)
        regimes = np.full(n, 2, dtype=int)
        posteriors = np.tile(_row(0.05, 0.05, 0.90), (n, 1))

        positions = _walk_forward_positions(
            returns,
            regimes=regimes,
            posteriors=posteriors,
            min_train=20,
            sizing_mode="signal",
        )
        assert positions.dtype == np.float64
        assert np.all(np.abs(positions) <= 1.0)
        # Warmup bars remain flat; traded bars are the strong long.
        assert np.all(positions[:20] == 0.0)
        assert np.allclose(positions[20:], 0.85)

    def test_binary_positions_remain_integer(self):
        prices = _make_prices(n=120)
        returns = _returns(prices)
        n = len(returns)
        regimes = _make_regimes(n, seed=3)
        positions = _walk_forward_positions(
            returns, regimes=regimes, min_train=20
        )
        assert np.issubdtype(positions.dtype, np.integer)
        assert set(np.unique(positions)).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Signal floor
# ---------------------------------------------------------------------------


class TestFloor:
    @pytest.mark.parametrize(
        "signal,expected",
        [
            (0.29, 0.0),  # below floor -> flat
            (0.30, 0.30),  # exactly at floor -> kept (strict <)
            (0.31, 0.31),  # above floor -> kept
        ],
    )
    def test_positive_floor_boundary(self, signal, expected):
        row = _row_for_signal(signal)
        assert _sized_position(row, 1, 0.30) == pytest.approx(expected)

    def test_negative_signal_is_floored_by_absolute_value(self):
        # raw signal = P(bear) - P(bull)
        assert _sized_position(_row_for_signal(-0.29), -1, 0.30) == 0.0
        assert _sized_position(_row_for_signal(-0.30), -1, 0.30) == pytest.approx(
            -0.30
        )

    def test_zero_floor_is_noop(self):
        row = _row(0.40, 0.40, 0.60)
        assert _sized_position(row, 1, 0.0) == pytest.approx(0.20)

    def test_invalid_floor_values_raise(self):
        for bad in (-0.01, 1.01, np.nan, np.inf, "not-a-number"):
            with pytest.raises(ValueError, match="signal_floor"):
                _validate_sizing_args("signal", bad)

    def test_floor_with_binary_mode_raises(self):
        with pytest.raises(ValueError, match="requires sizing_mode"):
            _validate_sizing_args("binary", 0.2)

    def test_valid_floor_is_normalised_to_float(self):
        assert _validate_sizing_args("signal", 0.2) == pytest.approx(0.2)
        assert isinstance(_validate_sizing_args("signal", 0.2), float)

    def test_floor_is_applied_in_full_backtest(self):
        # 0.2 signal is below a 0.3 floor -> no positions and no trades; the
        # same signal with a 0.1 floor opens one persistent trade.
        prices = _make_prices(n=80, seed=13)
        n = len(prices) - 1
        regimes = _build_regimes(20, [2] * (n - 20), n)
        posteriors = np.tile(_row_for_signal(0.2), (n, 1))

        floored = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=20, sizing_mode="signal", signal_floor=0.3,
        )
        unfloored = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=20, sizing_mode="signal", signal_floor=0.1,
        )

        assert floored["sizing_mode"] == "signal"
        assert floored["n_trades"] == 0
        assert floored["total_return"] == pytest.approx(0.0)
        assert unfloored["n_trades"] == 1


# ---------------------------------------------------------------------------
# Filter interaction
# ---------------------------------------------------------------------------


class TestFilterInteraction:
    def test_dwell_blocks_direction_change_like_binary(self):
        min_train = 5
        n_bars = 20
        regimes = _build_regimes(min_train, [2] * 15, n_bars)
        returns = _returns(_make_prices(n=n_bars))
        posteriors = np.tile(_row(0.05, 0.05, 0.90), (n_bars, 1))

        binary = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=3,
        )
        signal = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=3, sizing_mode="signal",
        )

        # First two bull bars are held sideways by dwell -> flat in both modes.
        assert binary[min_train] == 0
        assert binary[min_train + 1] == 0
        assert signal[min_train] == 0.0
        assert signal[min_train + 1] == 0.0
        # Third consecutive bull bar lets the switch through.
        assert binary[min_train + 2] == 1
        assert signal[min_train + 2] == pytest.approx(0.85)

    def test_hysteresis_blocks_direction_change_like_binary(self):
        min_train = 0
        n_bars = 6
        regimes = np.array([0, 0, 0, 2, 2, 2], dtype=int)
        returns = _returns(_make_prices(n=n_bars))
        posteriors = np.array(
            [
                [0.90, 0.05, 0.05],  # bear
                [0.90, 0.05, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],  # bull
                [0.05, 0.05, 0.90],
                [0.05, 0.05, 0.90],
            ]
        )

        binary = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, hysteresis_delta=0.3,
        )
        signal = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, hysteresis_delta=0.3, sizing_mode="signal",
        )

        # The hysteresis margin (0.90 - 0.90 = 0.0) never clears 0.3, so the
        # held bear direction is retained; bullish bars size to flat.
        assert np.all(binary[3:] == -1)
        assert np.all(signal[3:] == 0.0)

        # Without hysteresis the switch is allowed and signal sizing follows.
        signal_free = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, sizing_mode="signal",
        )
        assert signal_free[3] == pytest.approx(0.85)

    def test_randomised_sign_invariant_and_bounds(self):
        prices = _make_prices()
        returns = _returns(prices)
        n = len(returns)
        regimes = _make_regimes(n, seed=11)
        posteriors = _make_posteriors(n, seed=12)

        binary = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=20, dwell_bars=3, hysteresis_delta=0.2,
        )
        signal = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=20, dwell_bars=3, hysteresis_delta=0.2,
            sizing_mode="signal",
        )

        assert np.all(np.abs(signal) <= 1.0)
        for t in range(n):
            if binary[t] == 0:
                # No position may be opened in a sideways-held regime.
                assert signal[t] == 0.0
            else:
                assert np.sign(signal[t]) in (0.0, np.sign(binary[t]))

    def test_no_position_when_filter_holds_sideways(self):
        min_train = 5
        n_bars = 15
        regimes = _build_regimes(min_train, [2] * 10, n_bars)
        returns = _returns(_make_prices(n=n_bars))
        posteriors = np.tile(_row(0.01, 0.01, 0.98), (n_bars, 1))
        signal = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=20, sizing_mode="signal",
        )
        # All ten bull bars fall short of the twenty-bar dwell requirement.
        assert np.all(signal[min_train:] == 0.0)


# ---------------------------------------------------------------------------
# No lookahead
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_mutating_future_posteriors_does_not_change_past_positions(self):
        prices = _make_prices(n=120)
        returns = _returns(prices)
        n = len(returns)
        regimes = _make_regimes(n, seed=5)
        posteriors = _make_posteriors(n, seed=6)
        t0 = 80

        mutated = posteriors.copy()
        mutated[t0 + 1 :] = _make_posteriors(n - t0 - 1, seed=99)

        before = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=20, hysteresis_delta=0.1, sizing_mode="signal",
        )
        after = _walk_forward_positions(
            returns, regimes=regimes, posteriors=mutated,
            min_train=20, hysteresis_delta=0.1, sizing_mode="signal",
        )
        np.testing.assert_array_equal(before[: t0 + 1], after[: t0 + 1])

    def test_one_bar_pnl_lag_and_warmup_masking_unchanged(self):
        min_train = 20
        prices = _make_prices(n=120, seed=7)
        n = len(prices) - 1
        # One tradable bull bar followed by sideways.  The one-bar lag then
        # applies the bull position to the next bar's return, while same-day
        # P&L would be flat.
        regimes = _build_regimes(min_train, [2] + [1] * (n - min_train - 1), n)
        posteriors = np.tile(_row_for_signal(0.8), (n, 1))

        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=min_train, sizing_mode="signal",
        )
        returns = _returns(prices)
        positions = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, sizing_mode="signal",
        )

        # Warmup bars are flat, so no position can exist before min_train.
        assert np.all(positions[:min_train] == 0.0)
        assert positions[min_train] != 0.0

        # The P&L uses positions[t-1] for returns[t].  Reproduce that lag,
        # then mask the warmup bars exactly as the backtest does.
        pnl = np.zeros(n, dtype=float)
        pnl[1:] = positions[:-1] * returns.iloc[1:].values
        equity = np.cumprod(1.0 + pnl)
        equity[:min_train] = np.nan
        equity_series = pd.Series(equity, index=returns.index).dropna()

        expected_total_return = float(
            equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
        )
        expected_sharpe = calculate_sharpe_ratio(equity_series)
        assert result["total_return"] == pytest.approx(expected_total_return)
        assert result["sharpe"] == pytest.approx(expected_sharpe)

        # Same-day (unlagged) P&L produces a different total return here, so
        # the assertion above really does detect removal of the lag.
        unlagged_pnl = positions * returns.values
        unlagged_equity = np.cumprod(1.0 + unlagged_pnl)
        unlagged_equity[:min_train] = np.nan
        unlagged_series = pd.Series(
            unlagged_equity, index=returns.index
        ).dropna()
        unlagged_total_return = float(
            unlagged_series.iloc[-1] / unlagged_series.iloc[0] - 1.0
        )
        assert not np.isclose(expected_total_return, unlagged_total_return)

        assert result["sizing_mode"] == "signal"


# ---------------------------------------------------------------------------
# Threshold engine behavior and validation
# ---------------------------------------------------------------------------


class TestThresholdAndValidation:
    def test_direct_signal_mode_without_posteriors_raises(self):
        prices = _make_prices(n=60)
        regimes = _make_regimes(len(prices) - 1)
        with pytest.raises(ValueError, match="posterior"):
            walk_forward_backtest(
                prices, regimes=regimes, posteriors=None,
                min_train=20, sizing_mode="signal",
            )

    def test_pipeline_rejects_threshold_before_classification(self):
        # A RangeIndex series would fail _validate_prices; the threshold guard
        # fires first, proving the rejection happens before classification.
        with pytest.raises(ValueError, match="threshold engine is unsupported"):
            pipeline_run(
                pd.Series([1.0, 2.0]),
                source="test",
                engine_config=ThresholdConfig(),
                sizing_mode="signal",
            )

    def test_cli_threshold_signal_sizing_errors_with_json(self, btc_csv):
        from tests.conftest import run_regime

        result = run_regime(
            "--csv", btc_csv, "--engine", "threshold",
            "--signal-sizing", "--json",
        )
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert "signal sizing" in data["error"]
        assert "threshold" in data["error"]

    def test_cli_signal_floor_requires_signal_sizing(self, btc_csv):
        from tests.conftest import run_regime

        result = run_regime("--csv", btc_csv, "--signal-floor", "0.2", "--json")
        assert result.returncode == 2
        assert "--signal-floor requires --signal-sizing" in result.stderr

    @pytest.mark.parametrize("bad", ["-0.1", "1.1", "nan", "inf"])
    def test_cli_rejects_invalid_signal_floor(self, bad):
        from tests.conftest import run_regime

        result = run_regime(
            "--csv", "unused.csv", "--signal-sizing",
            "--signal-floor", bad, "--json",
        )
        assert result.returncode == 2
        assert "--signal-floor" in result.stderr

    def test_cli_rejects_signal_flags_in_eval_mode(self):
        from tests.conftest import run_regime

        result = run_regime(
            "--eval-csv", "test_data", "--signal-sizing",
            "--eval-engines", "threshold", "--json",
        )
        assert result.returncode == 2
        assert "not supported in eval mode" in result.stderr

    def test_invalid_sizing_mode_raises(self):
        prices = _make_prices(n=60)
        regimes = _make_regimes(len(prices) - 1)
        with pytest.raises(ValueError, match="sizing_mode"):
            walk_forward_backtest(
                prices, regimes=regimes, min_train=20, sizing_mode="scaled",
            )

    def test_malformed_posteriors_raise(self):
        prices = _make_prices(n=60)
        returns = _returns(prices)
        regimes = _make_regimes(len(returns))
        with pytest.raises(ValueError, match="posteriors with shape"):
            _walk_forward_positions(
                returns, regimes=regimes,
                posteriors=np.zeros((len(returns), 2)),
                min_train=20, sizing_mode="signal",
            )

    def test_cli_passes_sizing_arguments(self, monkeypatch):
        from hmm_futures_analysis import cli

        captured = {}
        prices = _make_prices(n=30)
        monkeypatch.setattr(
            cli, "load_prices", lambda **kwargs: (prices, None, "test")
        )
        monkeypatch.setattr(cli, "build_engine_config", lambda args: object())
        monkeypatch.setattr(
            cli, "pipeline_run", lambda **kwargs: captured.update(kwargs) or {}
        )
        monkeypatch.setattr(
            cli, "serialize_pipeline", lambda output, transitions_limit=None: {}
        )
        monkeypatch.setattr(
            sys, "argv",
            [
                "hmm-regime", "--csv", "x.csv", "--json", "--engine", "hmm",
                "--signal-sizing", "--signal-floor", "0.2",
            ],
        )
        cli.main()
        assert captured["sizing_mode"] == "signal"
        assert captured["signal_floor"] == pytest.approx(0.2)

    def test_cli_defaults_to_binary(self, monkeypatch):
        from hmm_futures_analysis import cli

        captured = {}
        prices = _make_prices(n=30)
        monkeypatch.setattr(
            cli, "load_prices", lambda **kwargs: (prices, None, "test")
        )
        monkeypatch.setattr(cli, "build_engine_config", lambda args: object())
        monkeypatch.setattr(
            cli, "pipeline_run", lambda **kwargs: captured.update(kwargs) or {}
        )
        monkeypatch.setattr(
            cli, "serialize_pipeline", lambda output, transitions_limit=None: {}
        )
        monkeypatch.setattr(
            sys, "argv", ["hmm-regime", "--csv", "x.csv", "--json"]
        )
        cli.main()
        assert captured["sizing_mode"] == "binary"
        assert captured["signal_floor"] == 0.0


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestSizingModeField:
    def test_default_result_is_binary(self):
        prices, regimes, posteriors = _frozen_fixture()
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=50,
        )
        assert result["sizing_mode"] == "binary"

    def test_signal_result_is_signal(self):
        prices, regimes, posteriors = _frozen_fixture()
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=50,
            sizing_mode="signal",
        )
        assert result["sizing_mode"] == "signal"

    def test_short_data_empty_result_carries_mode(self):
        prices = _make_prices(n=5)
        regimes = _make_regimes(len(prices) - 1)
        posteriors = np.tile(_row(0.1, 0.1, 0.8), (len(prices) - 1, 1))
        for mode in ("binary", "signal"):
            result = walk_forward_backtest(
                prices, regimes=regimes, posteriors=posteriors,
                min_train=10, sizing_mode=mode,
            )
            assert result["sizing_mode"] == mode
            assert result["n_trades"] == 0

    def test_no_valid_equity_empty_result_carries_mode(self):
        prices = _make_prices(n=20)
        regimes = _make_regimes(len(prices) - 1)
        posteriors = np.tile(_row(0.1, 0.1, 0.8), (len(prices) - 1, 1))
        for mode in ("binary", "signal"):
            result = walk_forward_backtest(
                prices, regimes=regimes, posteriors=posteriors,
                min_train=19, sizing_mode=mode,
            )
            assert result["sizing_mode"] == mode
            assert result["n_trades"] == 0

    def test_empty_result_helper_defaults_to_binary(self):
        assert _empty_result()["sizing_mode"] == "binary"
        assert _empty_result("signal")["sizing_mode"] == "signal"

    def test_pipeline_walk_forward_has_seven_keys(self):
        prices = _make_prices(n=120)
        result = pipeline_run(
            prices, source="test", engine_config=ThresholdConfig(), min_train=50,
        )
        assert set(result.walk_forward.keys()) == {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
            "sizing_mode",
        }
        assert result.walk_forward["sizing_mode"] == "binary"

    def test_pipeline_signal_mode_passes_posteriors(self, monkeypatch):
        import hmm_futures_analysis.regime.pipeline as pipeline_mod

        prices = _make_prices(n=80)
        n = len(prices) - 1
        regimes = np.full(n, 2, dtype=int)
        posteriors = np.tile(_row_for_signal(0.8), (n, 1))

        class _FakeEngine:
            def run_classify(self, prices, ohlcv, returns, min_train, **kwargs):
                return ClassifyOutput(
                    regimes=regimes,
                    posteriors=posteriors,
                    last_regime=2,
                    n_states=3,
                    n_features=5,
                )

        monkeypatch.setattr(
            pipeline_mod, "resolve_engine", lambda config: _FakeEngine()
        )
        result = pipeline_mod.run(
            prices,
            source="test",
            engine_config=HMMGenericConfig(),
            min_train=20,
            sizing_mode="signal",
        )
        assert result.walk_forward["sizing_mode"] == "signal"
        assert set(result.walk_forward) == {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
            "sizing_mode",
        }


# ---------------------------------------------------------------------------
# Binary compatibility
# ---------------------------------------------------------------------------


class TestBinaryRegression:
    def test_frozen_binary_metrics_are_unchanged(self):
        prices, regimes, posteriors = _frozen_fixture()
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=50,
        )
        for key, expected in _BINARY_SNAPSHOT.items():
            assert result[key] == pytest.approx(expected, rel=0.0, abs=1e-12)

    def test_frozen_binary_positions_are_unchanged(self):
        prices, regimes, posteriors = _frozen_fixture()
        positions = _walk_forward_positions(
            _returns(prices), regimes=regimes, posteriors=posteriors,
            min_train=50,
        )
        assert positions.dtype == np.int64 or np.issubdtype(
            positions.dtype, np.integer
        )
        assert int(positions.sum()) == _BINARY_POSITION_SUM
        assert int(np.dot(positions, np.arange(len(positions)))) == (
            _BINARY_POSITION_HASH
        )

    def test_sizing_mode_is_only_added_result_key(self):
        prices, regimes, posteriors = _frozen_fixture()
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=50,
        )
        assert set(result.keys()) == {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
            "sizing_mode",
            "metrics_reliable",
            "metrics_caveat",
        }

    def test_mapping_is_unchanged(self):
        assert _STATE_MAP == {0: -1, 1: 0, 2: 1}


# ---------------------------------------------------------------------------
# Two-state (degenerate-recovered) posteriors and trade stats
# ---------------------------------------------------------------------------


class TestTwoStatePosteriors:
    def test_sideways_bucket_near_zero_still_sizes(self):
        # 2-state auto-recovery leaves the sideways bucket near zero.
        assert _sized_position(_row(0.0, 0.0, 1.0), 1, 0.0) == pytest.approx(1.0)
        assert _sized_position(_row(1.0, 0.0, 0.0), -1, 0.0) == pytest.approx(-1.0)
        assert _sized_position(_row(0.4, 0.0, 0.6), 1, 0.0) == pytest.approx(0.2)


class TestTradeStats:
    def test_scaling_without_zero_crossing_is_one_trade(self):
        positions = np.array([0.0, 0.9, 0.2, 0.9, 0.0])
        equity = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        n_trades, win_rate, _ = _compute_trade_stats(positions, equity)
        assert n_trades == 1
        assert win_rate == pytest.approx(1.0)

    def test_zero_crossing_starts_a_new_trade(self):
        positions = np.array([0.0, 0.9, 0.0, 0.5, 0.0])
        equity = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        n_trades, _, _ = _compute_trade_stats(positions, equity)
        assert n_trades == 2

    def test_open_trade_at_end_is_counted(self):
        positions = np.array([0.0, 0.4, 0.8])
        equity = np.array([1.0, 1.1, 1.2])
        n_trades, _, _ = _compute_trade_stats(positions, equity)
        assert n_trades == 1
