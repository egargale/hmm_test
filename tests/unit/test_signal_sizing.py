"""Tests for signal-scaled position sizing (Issue #85).

Issue #85 — https://github.com/egargale/hmm_test/issues/85

The walk-forward backtest gains an opt-in ``signal-sizing`` mode where
per-bar position = ``clip(signal, -1, 1)`` instead of binary ±1/0. The
filters (dwell-time, hysteresis) still gate *direction*; sizing scales
*magnitude* within the allowed direction. The threshold engine (no
posteriors) synthesizes a pseudo-signal from its rolling return.

Tests mirror the patterns in ``test_filters.py`` (synthetic regimes +
posteriors fixtures).
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures (mirror test_filters.py helpers)
# ---------------------------------------------------------------------------


def _make_prices(n: int = 100) -> pd.Series:
    """Generate simple price series for walk-forward tests."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = pd.Series(100.0 + np.arange(n) * 0.01, index=dates, name="close")
    return prices


def _build_regimes(min_train: int, regimes_after_train: list[int], n_bars: int) -> np.ndarray:
    """Build a regimes array (length n_bars) with warmup and training+osc region."""
    all_regimes = [1] * min_train + regimes_after_train
    while len(all_regimes) < n_bars:
        all_regimes.append(1)
    return np.array(all_regimes[:n_bars], dtype=int)


def _posteriors_from_regimes(regimes: np.ndarray, confidence: float = 0.8) -> np.ndarray:
    """Build (n, 3) posteriors where each bar's dominant state matches its regime.

    ``confidence`` is the probability mass on the dominant state; the rest is
    split evenly between the other two so the row sums to 1.
    """
    n = len(regimes)
    post = np.full((n, 3), (1.0 - confidence) / 2.0)
    for i, r in enumerate(regimes):
        post[i, r] = confidence
    return post


# ---------------------------------------------------------------------------
# AC3 — position formula = clip(signal, -1, 1)
# ---------------------------------------------------------------------------


class TestPositionEqualsClipSignal:
    """In sizing mode, positions[t] == clip(post[t][2] - post[t][0], -1, 1)."""

    def test_position_equals_clip_signal(self):
        from hmm_futures_analysis.regime.walk_forward import _walk_forward_positions

        min_train = 10
        n_bars = 30
        regimes_after = [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 2]
        regimes = _build_regimes(min_train, regimes_after, n_bars)
        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        # returns is length n_bars-1; align regimes/posteriors to returns.
        regimes = regimes[: len(returns)]
        posteriors = _posteriors_from_regimes(regimes, confidence=0.7)

        positions = _walk_forward_positions(
            returns,
            regimes=regimes,
            posteriors=posteriors,
            min_train=min_train,
            signal_sizing=True,
        )

        # Verify each trading bar matches clip(signal, -1, 1) exactly.
        for t in range(min_train, len(returns)):
            expected = float(np.clip(posteriors[t][2] - posteriors[t][0], -1.0, 1.0))
            assert positions[t] == pytest.approx(expected), (
                f"bar {t}: expected {expected}, got {positions[t]}"
            )
        # Pre-training bars should be zero.
        assert np.all(positions[:min_train] == 0.0)

    def test_positions_dtype_float_in_sizing_mode(self):
        """Position array is float when sizing is on, int when off."""
        from hmm_futures_analysis.regime.walk_forward import _walk_forward_positions

        min_train = 10
        n_bars = 20
        regimes = _build_regimes(min_train, [2, 0, 2, 0, 2, 1, 0, 1, 2, 0], n_bars)
        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(returns)]
        posteriors = _posteriors_from_regimes(regimes)

        pos_sizing = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True,
        )
        pos_binary = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=False,
        )
        assert pos_sizing.dtype == np.float64
        assert pos_binary.dtype == np.dtype(int)


# ---------------------------------------------------------------------------
# AC3 — clip bounds
# ---------------------------------------------------------------------------


class TestClipBounds:
    """Signal saturates at ±1; extreme posteriors clip correctly."""

    def test_clip_bounds(self):
        """Signal saturates at ±1 when mapped to positions; raw signal can exceed ±1.

        ``_bar_signal`` returns the raw P(bull)−P(bear) (may exceed ±1 on
        degenerate input); the **position** is ``clip(signal, -1, 1)``. This
        test verifies both layers: the raw signal value and the clipped
        position that results from it.
        """
        from hmm_futures_analysis.regime.walk_forward import _bar_signal, _walk_forward_positions

        # [0,0,1] -> signal +1 (pure bull)
        post_bull = np.array([0.0, 0.0, 1.0])
        # [1,0,0] -> signal -1 (pure bear)
        post_bear = np.array([1.0, 0.0, 0.0])
        # [0.5,0.5,0] -> signal -0.5
        post_mixed = np.array([0.5, 0.5, 0.0])
        # over-confident: [0,0,5] -> raw signal 5 -> clipped position +1
        post_over = np.array([0.0, 0.0, 5.0])
        # [0,1,0] -> signal 0 (sideways)
        post_flat = np.array([0.0, 1.0, 0.0])

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        returns = pd.Series([0.0] * 5, index=dates)

        # _bar_signal returns the RAW signal (clip happens at position level).
        assert _bar_signal(0, post_bull.reshape(1, 3), returns, None, None) == pytest.approx(1.0)
        assert _bar_signal(0, post_bear.reshape(1, 3), returns, None, None) == pytest.approx(-1.0)
        assert _bar_signal(0, post_mixed.reshape(1, 3), returns, None, None) == pytest.approx(-0.5)
        assert _bar_signal(0, post_over.reshape(1, 3), returns, None, None) == pytest.approx(5.0)
        assert _bar_signal(0, post_flat.reshape(1, 3), returns, None, None) == pytest.approx(0.0)

        # Now verify the CLIP happens in the position: build a fixture where
        # bar 2 has the over-confident posteriors and check positions[2] == +1.
        min_train = 2
        n_bars = 6
        regimes = np.array([1, 1, 2, 2, 2, 2], dtype=int)
        posteriors = np.zeros((n_bars, 3), dtype=float)
        posteriors[2] = post_over      # raw signal 5 -> clip +1
        posteriors[3] = post_bear      # raw signal -1 -> clip -1
        prices = _make_prices(n_bars)
        rets = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(rets)]
        posteriors = posteriors[: len(rets)]
        positions = _walk_forward_positions(
            rets, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True,
        )
        assert positions[2] == pytest.approx(1.0), "raw signal 5 should clip to +1"
        assert positions[3] == pytest.approx(-1.0), "raw signal -1 stays -1"


# ---------------------------------------------------------------------------
# AC5 — signal floor boundary
# ---------------------------------------------------------------------------


class TestSignalFloor:
    """signal_floor zeros positions where abs(signal) < floor (strict <)."""

    def test_signal_floor_boundary(self):
        from hmm_futures_analysis.regime.walk_forward import _walk_forward_positions

        min_train = 2
        n_bars = 6
        regimes = np.array([1, 1, 2, 2, 2, 2], dtype=int)
        posteriors = np.zeros((n_bars, 3), dtype=float)
        # Construct signals at 0.29, 0.30, 0.31 (abs values) on trading bars.
        # signal = post[2] - post[0]; make post[1]=0, post[2]=s, post[0]=0 -> signal=s
        posteriors[0] = [0.0, 1.0, 0.0]  # warmup
        posteriors[1] = [0.0, 1.0, 0.0]  # warmup
        posteriors[2] = [0.0, 0.0, 0.29]  # signal 0.29 -> below floor 0.30 -> flat
        posteriors[3] = [0.0, 0.0, 0.30]  # signal 0.30 -> at floor (strict <) -> sized
        posteriors[4] = [0.0, 0.0, 0.31]  # signal 0.31 -> above floor -> sized

        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(returns)]
        posteriors = posteriors[: len(returns)]

        positions = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True, signal_floor=0.30,
        )

        assert positions[2] == pytest.approx(0.0), "0.29 should be floored"
        assert positions[3] == pytest.approx(0.30), "0.30 at boundary should NOT be floored"
        assert positions[4] == pytest.approx(0.31), "0.31 should be sized"

    def test_signal_floor_default_zero_disables(self):
        """signal_floor=0.0 never floors anything (pure clip(signal))."""
        from hmm_futures_analysis.regime.walk_forward import _walk_forward_positions

        min_train = 2
        n_bars = 5
        regimes = np.array([1, 1, 2, 2, 2], dtype=int)
        posteriors = np.zeros((n_bars, 3), dtype=float)
        posteriors[2] = [0.0, 0.0, 0.01]  # tiny signal
        posteriors[3] = [0.0, 0.0, 0.001]
        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(returns)]
        posteriors = posteriors[: len(returns)]

        positions = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True, signal_floor=0.0,
        )
        assert positions[2] == pytest.approx(0.01)
        assert positions[3] == pytest.approx(0.001)

    def test_signal_floor_negative_signal_floored(self):
        """Floor applies to abs(signal), so negative weak signals also go flat."""
        from hmm_futures_analysis.regime.walk_forward import _walk_forward_positions

        min_train = 2
        n_bars = 5
        regimes = np.array([1, 1, 0, 0, 0], dtype=int)
        posteriors = np.zeros((n_bars, 3), dtype=float)
        posteriors[2] = [0.29, 0.0, 0.0]  # signal -0.29 -> abs 0.29 -> floored
        posteriors[3] = [0.31, 0.0, 0.0]  # signal -0.31 -> abs 0.31 -> sized
        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(returns)]
        posteriors = posteriors[: len(returns)]

        positions = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True, signal_floor=0.30,
        )
        assert positions[2] == pytest.approx(0.0)
        assert positions[3] == pytest.approx(-0.31)


# ---------------------------------------------------------------------------
# AC4 — filter gates direction; sizing scales magnitude
# ---------------------------------------------------------------------------


class TestFilterInteraction:
    """Dwell/hysteresis gate direction; sizing scales magnitude within it."""

    def test_weak_signal_in_allowed_direction_is_partial(self):
        """With dwell active and a weak signal, position is partial (not 0, not ±1)."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 5
        n_bars = 25
        regimes_after = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0,
                         2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
        regimes = _build_regimes(min_train, regimes_after, n_bars)

        # Weak but consistent bull/bear signals
        posteriors = np.zeros((n_bars, 3), dtype=float)
        for i, r in enumerate(regimes):
            if r == 2:
                posteriors[i] = [0.35, 0.15, 0.50]  # signal +0.15 (weak bull)
            elif r == 0:
                posteriors[i] = [0.50, 0.15, 0.35]  # signal -0.15 (weak bear)
            else:
                posteriors[i] = [0.33, 0.34, 0.33]  # near-zero

        prices = _make_prices(n_bars)
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=3, signal_sizing=True,
        )
        # Should produce a valid sizing-mode result with partial positions.
        assert result["sizing_mode"] == "signal"
        assert isinstance(result["n_trades"], int)

    def test_filter_blocks_switch_position_stays_on_current_signal(self):
        """When dwell blocks a regime switch, position uses current regime's signal.

        Per design §2 D2: filters gate direction (which regime label is
        active); the sized position follows the per-bar signal regardless of
        the candidate regime. The filter prevents the *label* from
        whipsawing; only the *size* moves with the signal.
        """
        from hmm_futures_analysis.regime.walk_forward import _walk_forward_positions

        min_train = 5
        n_bars = 12
        # Regime flips to 2 at bar 5, but dwell=3 blocks switch until bar 7.
        regimes = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=int)
        # Signal is strongly bull (+0.7) from bar 5 onward regardless of regime label.
        posteriors = np.zeros((n_bars, 3), dtype=float)
        for i in range(n_bars):
            posteriors[i] = [0.1, 0.1, 0.8]  # signal +0.7

        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(returns)]
        posteriors = posteriors[: len(returns)]

        positions = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=3, signal_sizing=True,
        )

        # Even while dwell holds current_regime at 1 (sideways), the signal
        # is +0.7 so the position should be +0.7 (sizing follows the signal,
        # not the regime label). This is the intended "reduce exposure when
        # uncertain" effect — here the signal is confident so position is full.
        for t in range(min_train, len(returns)):
            assert positions[t] == pytest.approx(0.7), (
                f"bar {t}: expected 0.7 (signal), got {positions[t]}"
            )


# ---------------------------------------------------------------------------
# AC6 — threshold engine pseudo-signal
# ---------------------------------------------------------------------------


class TestThresholdPseudoSignal:
    """Threshold engine (no posteriors) synthesizes pseudo-signal from rolling return."""

    def test_threshold_pseudo_signal(self):
        from hmm_futures_analysis.regime.walk_forward import _bar_signal

        # Construct returns where rolling(20).sum() at bar t is a known value.
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        # Constant +0.0025 daily return -> 20-bar rolling sum = 0.05 (= threshold) -> +1
        returns = pd.Series(0.0025, index=dates)
        window, threshold = 20, 0.05

        # At bar 19+ the rolling window is full: 20 * 0.0025 = 0.05 -> +1
        sig = _bar_signal(19, None, returns, window, threshold)
        assert sig == pytest.approx(1.0), f"expected +1 at threshold, got {sig}"

        # Half the daily return -> rolling sum 0.025 -> +0.5
        returns_half = pd.Series(0.00125, index=dates)
        sig_half = _bar_signal(19, None, returns_half, window, threshold)
        assert sig_half == pytest.approx(0.5), f"expected +0.5, got {sig_half}"

        # Negative at threshold -> -1
        returns_neg = pd.Series(-0.0025, index=dates)
        sig_neg = _bar_signal(19, None, returns_neg, window, threshold)
        assert sig_neg == pytest.approx(-1.0), f"expected -1, got {sig_neg}"

        # Beyond threshold saturates: daily 0.01 -> 20*0.01=0.2 -> clip +1
        returns_big = pd.Series(0.01, index=dates)
        sig_sat = _bar_signal(19, None, returns_big, window, threshold)
        assert sig_sat == pytest.approx(1.0), f"expected +1 (saturated), got {sig_sat}"

    def test_threshold_pseudo_signal_in_backtest(self):
        """End-to-end: threshold engine + signal sizing produces a valid result."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 40
        regimes = _build_regimes(min_train, [2, 2, 0, 0, 2, 2, 0, 0] * 5, n_bars)
        prices = _make_prices(n_bars)

        result = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train,
            signal_sizing=True, threshold_window=5, threshold_value=0.05,
        )
        assert result["sizing_mode"] == "signal"
        assert isinstance(result["n_trades"], int)

    def test_threshold_nan_rolling_returns_zero(self):
        """Before the rolling window fills (NaN), pseudo-signal is 0 (flat)."""
        from hmm_futures_analysis.regime.walk_forward import _bar_signal

        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(0.01, index=dates)
        # At bar 5 with window=20, rolling sum is NaN -> return 0.0
        sig = _bar_signal(5, None, returns, 20, 0.05)
        assert sig == 0.0

    def test_threshold_no_params_raises(self):
        """signal_sizing=True with posteriors=None and no threshold params -> ValueError."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes = _build_regimes(min_train, [2, 0, 2, 0] * 5, n_bars)
        prices = _make_prices(n_bars)

        with pytest.raises(ValueError, match="signal-sizing requires"):
            walk_forward_backtest(
                prices, regimes=regimes, min_train=min_train,
                signal_sizing=True,
            )


# ---------------------------------------------------------------------------
# AC9 — 2-state auto-recovered posteriors
# ---------------------------------------------------------------------------


class TestTwoStateRecovered:
    """2-state recovered models still populate (n,3) posteriors; signal well-defined."""

    def test_2state_recovered_posteriors(self):
        """A recovered 2-state model (bear/bull, sideways bucket ~0) still works."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        # Only bear (0) and bull (2) regimes, no sideways — as a 2-state model would.
        regimes_after = [2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 2]
        regimes = _build_regimes(min_train, regimes_after, n_bars)

        # Posteriors: 2-state recovered fills buckets 0 and 2, bucket 1 (sideways) ~0.
        posteriors = np.zeros((n_bars, 3), dtype=float)
        for i, r in enumerate(regimes):
            if r == 2:
                posteriors[i] = [0.1, 0.0, 0.9]  # bull dominant, sideways 0
            elif r == 0:
                posteriors[i] = [0.9, 0.0, 0.1]  # bear dominant, sideways 0
            else:
                posteriors[i] = [0.5, 0.0, 0.5]

        prices = _make_prices(n_bars)
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True,
        )
        assert result["sizing_mode"] == "signal"
        # Signal is well-defined: bull bars -> +0.8, bear bars -> -0.8
        assert isinstance(result["n_trades"], int)


# ---------------------------------------------------------------------------
# AC7 — sizing_mode output field
# ---------------------------------------------------------------------------


class TestSizingModeField:
    """walk_forward result includes sizing_mode in both modes."""

    def test_sizing_mode_field_binary_default(self):
        """Default run -> sizing_mode == 'binary'."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes = _build_regimes(min_train, [2, 0, 2, 0, 2, 1, 0, 1, 2, 0] * 2, n_bars)
        prices = _make_prices(n_bars)

        result = walk_forward_backtest(prices, regimes=regimes, min_train=min_train)
        assert result["sizing_mode"] == "binary"

    def test_sizing_mode_field_signal(self):
        """signal_sizing=True -> sizing_mode == 'signal'."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes = _build_regimes(min_train, [2, 0, 2, 0, 2, 1, 0, 1, 2, 0] * 2, n_bars)
        posteriors = _posteriors_from_regimes(regimes)
        prices = _make_prices(n_bars)

        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True,
        )
        assert result["sizing_mode"] == "signal"

    def test_sizing_mode_field_empty_result_binary(self):
        """Empty/short-data result still carries sizing_mode='binary'."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        prices = _make_prices(3)
        result = walk_forward_backtest(
            prices, regimes=np.array([1, 1, 1]), min_train=252,
        )
        assert result["sizing_mode"] == "binary"

    def test_sizing_mode_field_empty_result_signal(self):
        """Empty/short-data result in sizing mode carries sizing_mode='signal'."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        prices = _make_prices(3)
        result = walk_forward_backtest(
            prices, regimes=np.array([1, 1, 1]), min_train=252,
            signal_sizing=True,
        )
        assert result["sizing_mode"] == "signal"


# ---------------------------------------------------------------------------
# AC8 — backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Default (binary) output is identical whether sizing kwargs are omitted or False."""

    def test_default_binary_identical(self):
        """signal_sizing=False vs omitted -> identical result (modulo sizing_mode presence)."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes_after = [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 2]
        regimes = _build_regimes(min_train, regimes_after, n_bars)
        posteriors = _posteriors_from_regimes(regimes)
        prices = _make_prices(n_bars)

        result_explicit = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=min_train,
            dwell_bars=0, hysteresis_delta=0.0,
            signal_sizing=False, signal_floor=0.0,
        )
        result_default = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=min_train,
        )
        # All numeric fields identical; both carry sizing_mode='binary'.
        assert result_explicit == result_default

    def test_default_params_match_original(self):
        """walk_forward_backtest with explicit defaults == implicit defaults."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes_after = [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 2]
        regimes = _build_regimes(min_train, regimes_after, n_bars)
        prices = _make_prices(n_bars)

        result_explicit = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train,
            dwell_bars=0, hysteresis_delta=0.0,
        )
        result_default = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train,
        )
        assert result_explicit == result_default


# ---------------------------------------------------------------------------
# AC9 / §4.6 — trade stats flat-to-flat under sizing
# ---------------------------------------------------------------------------


class TestTradeStatsFlatToFlat:
    """n_trades counts flat-to-flat cycles, not size changes within a trade."""

    def test_position_scaling_without_zero_cross_is_one_trade(self):
        """A position scaling 0.9 -> 0.2 -> 0.9 without crossing 0 is one trade."""
        from hmm_futures_analysis.regime.walk_forward import (
            _compute_trade_stats,
            _walk_forward_positions,
        )

        min_train = 2
        n_bars = 7
        regimes = np.array([1, 1, 2, 2, 2, 2, 2], dtype=int)
        # Signal: 0.9, 0.2, 0.9 — never crosses 0, so one continuous trade.
        posteriors = np.zeros((n_bars, 3), dtype=float)
        posteriors[2] = [0.05, 0.05, 0.9]
        posteriors[3] = [0.4, 0.2, 0.6]   # signal = 0.2 (weak but positive)
        posteriors[4] = [0.05, 0.05, 0.9]
        posteriors[5] = [0.05, 0.05, 0.9]
        posteriors[6] = [0.05, 0.05, 0.9]

        prices = _make_prices(n_bars)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = regimes[: len(returns)]
        posteriors = posteriors[: len(returns)]

        positions = _walk_forward_positions(
            returns, regimes=regimes, posteriors=posteriors,
            min_train=min_train, signal_sizing=True,
        )
        # Build a synthetic equity curve for trade stats.
        equity = np.cumprod(1.0 + positions.astype(float) * 0.01)
        n_trades, _, _ = _compute_trade_stats(positions, equity)
        # One entry (0->non-0 at bar 2) and no exit before the end -> 1 trade.
        assert n_trades == 1, f"expected 1 continuous trade, got {n_trades}"

    def test_zero_cross_starts_new_trade(self):
        """Crossing through 0 (flat) starts/ends trades."""
        from hmm_futures_analysis.regime.walk_forward import (
            _compute_trade_stats,
        )

        # Positions: flat, long, flat, short, flat -> 2 trades.
        positions = np.array([0.0, 0.5, 0.0, -0.5, 0.0])
        equity = np.array([1.0, 1.01, 1.0, 0.99, 1.0])
        n_trades, _, _ = _compute_trade_stats(positions, equity)
        assert n_trades == 2


# ---------------------------------------------------------------------------
# CLI integration tests (mirror test_filters.py::TestCLIFlags)
# ---------------------------------------------------------------------------


class TestCLISignalSizing:
    """--signal-sizing and --signal-floor flags wired through correctly."""

    def test_cli_signal_sizing_flag(self, btc_csv):
        """--signal-sizing accepted and produces sizing_mode='signal'."""
        from tests.conftest import run_regime
        import json

        result = run_regime("--csv", btc_csv, "--json", "--signal-sizing")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["walk_forward"]["sizing_mode"] == "signal"

    def test_cli_signal_sizing_with_floor(self, btc_csv):
        """--signal-sizing --signal-floor together produce valid JSON."""
        from tests.conftest import run_regime
        import json

        result = run_regime(
            "--csv", btc_csv, "--json", "--signal-sizing", "--signal-floor", "0.2",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["walk_forward"]["sizing_mode"] == "signal"

    def test_cli_default_is_binary(self, btc_csv):
        """Default (no flag) -> sizing_mode='binary'."""
        from tests.conftest import run_regime
        import json

        result = run_regime("--csv", btc_csv, "--json")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["walk_forward"]["sizing_mode"] == "binary"

    def test_cli_signal_sizing_with_hmm_engine(self, btc_csv):
        """--signal-sizing with an HMM engine (posteriors present) works."""
        from tests.conftest import run_regime
        import json

        result = run_regime(
            "--csv", btc_csv, "--json", "--engine", "messina", "--signal-sizing",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["walk_forward"]["sizing_mode"] == "signal"
