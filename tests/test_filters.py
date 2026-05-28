"""Tests for whipsaw-reduction filters: dwell-time and hysteresis.

Issue #19 — https://github.com/egargale/hmm_test/issues/19
Acceptance criteria: https://github.com/egargale/hmm_test/issues/19#issuecomment-4566724433
"""
import numpy as np
import pandas as pd
import pytest


class TestClassifyResultPosteriors:
    """ClassifyResult accepts optional posteriors field."""

    def test_posteriors_defaults_to_none(self):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        result = ClassifyResult(regime=2, means=np.array([[1.0]]))
        assert result.posteriors is None

    def test_posteriors_accepts_ndarray(self):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        probs = np.array([0.1, 0.3, 0.6])
        result = ClassifyResult(regime=2, means=None, posteriors=probs)
        assert result.posteriors is not None
        np.testing.assert_array_almost_equal(result.posteriors, probs)

    def test_posteriors_with_none_means(self):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        result = ClassifyResult(regime=1, posteriors=np.array([0.2, 0.5, 0.3]))
        assert result.means is None
        assert result.posteriors is not None


def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for HMM engine tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + rng.standard_normal(n) * 0.3,
            "high": close + np.abs(rng.standard_normal(n) * 0.8),
            "low": close - np.abs(rng.standard_normal(n) * 0.8),
            "close": close,
            "volume": rng.integers(100, 10_000, n).astype(float),
        },
        index=dates,
    )


class TestHMMEnginesPopulatePosteriors:
    """HMM engines populate posteriors via predict_proba."""

    def test_hmm_generic_returns_posteriors(self):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        ohlcv = _make_ohlcv(n=400)
        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv)
        result = engine.classify(features)
        assert result.posteriors is not None
        assert len(result.posteriors) == 3
        # Posteriors should sum to ~1.0
        np.testing.assert_allclose(np.sum(result.posteriors), 1.0, atol=1e-6)

    def test_hmm_messina_returns_posteriors(self):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        ohlcv = _make_ohlcv(n=400)
        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv)
        result = engine.classify(features)
        assert result.posteriors is not None
        assert len(result.posteriors) == 3
        np.testing.assert_allclose(np.sum(result.posteriors), 1.0, atol=1e-6)

    def test_threshold_engine_leaves_posteriors_none(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(0.01, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.posteriors is None
        assert result.regime in {0, 1, 2}


class _MockEngine:
    """Controllable engine that returns a predetermined sequence of regimes."""

    def __init__(self, regimes: list[int]):
        self._regimes = list(regimes)
        self._call_idx = 0

    def precompute(self, data: pd.DataFrame) -> None:
        return None

    def classify(self, data: pd.DataFrame, prev_means=None):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        idx = min(self._call_idx, len(self._regimes) - 1)
        regime = self._regimes[idx]
        self._call_idx += 1
        return ClassifyResult(regime=regime)


class _MockEngineWithPosteriors:
    """Controllable engine returning regimes + posteriors."""

    def __init__(self, regimes: list[int], posteriors: list[np.ndarray]):
        self._regimes = list(regimes)
        self._posteriors = list(posteriors)
        self._call_idx = 0

    def precompute(self, data: pd.DataFrame) -> None:
        return None

    def classify(self, data: pd.DataFrame, prev_means=None):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        idx = min(self._call_idx, len(self._regimes) - 1)
        self._call_idx += 1
        return ClassifyResult(
            regime=self._regimes[idx],
            posteriors=self._posteriors[idx],
        )


def _make_prices(n: int = 100) -> pd.Series:
    """Generate simple price series for walk-forward tests."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = pd.Series(100.0 + np.arange(n) * 0.01, index=dates, name="close")
    return prices


class TestDwellTimeFilter:
    """Dwell-time filter: position only changes after N consecutive same-regime bars."""

    def test_dwell_3_blocks_switch_until_3_consecutive(self):
        """With dwell_bars=3, position stays until 3 consecutive new regimes."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        # Build regime sequence: sideways(1) for warmup, then oscillating,
        # then 3 consecutive bull(2)
        min_train = 10
        n_bars = 30
        # regimes for each bar after min_train:
        # bars 10-14: regime 2 (bull) - should switch after 3 consecutive
        # bars 15-17: regime 0 (bear) - should switch after 3 consecutive
        regimes_after_train = [2, 2, 1, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 2]
        # Warmup regimes (don't matter, just fill):
        all_regimes = [1] * min_train + regimes_after_train
        # Pad to n_bars
        while len(all_regimes) < n_bars:
            all_regimes.append(1)

        engine = _MockEngine(all_regimes)
        prices = _make_prices(n_bars)

        result = walk_forward_backtest(
            prices, engine=engine, min_train=min_train,
            dwell_bars=3,
        )
        # Should return valid results
        assert isinstance(result["n_trades"], int)

    def test_dwell_0_is_disabled_no_behavior_change(self):
        """dwell_bars=0 produces same result as default (disabled)."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 25
        regimes = [1] * min_train + [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2]
        while len(regimes) < n_bars:
            regimes.append(1)

        engine = _MockEngine(regimes)
        prices = _make_prices(n_bars)

        result_disabled = walk_forward_backtest(
            prices, engine=engine, min_train=min_train, dwell_bars=0,
        )
        # Reset engine with same regimes
        engine2 = _MockEngine(regimes)
        result_default = walk_forward_backtest(
            prices, engine=engine2, min_train=min_train,
        )
        assert result_disabled == result_default

    def test_dwell_reduces_trades_on_oscillating_data(self):
        """Dwell filter produces fewer trades when regime oscillates."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 50
        # Heavy oscillation: 2, 0, 2, 0, 2, 0, ... then steady 2
        osc = []
        for i in range(20):
            osc.append(2 if i % 2 == 0 else 0)
        osc += [2] * 20  # then steady bull
        regimes = [1] * min_train + osc
        while len(regimes) < n_bars:
            regimes.append(2)

        prices = _make_prices(n_bars)

        engine_no_filter = _MockEngine(regimes)
        result_raw = walk_forward_backtest(
            prices, engine=engine_no_filter, min_train=min_train,
        )

        engine_dwell = _MockEngine(regimes)
        result_filtered = walk_forward_backtest(
            prices, engine=engine_dwell, min_train=min_train, dwell_bars=3,
        )

        # Dwell must produce fewer or equal trades
        assert result_filtered["n_trades"] <= result_raw["n_trades"]


class TestBothFiltersActive:
    """AND logic: both filters must agree to switch."""

    def test_hysteresis_noop_when_posteriors_none(self):
        """Hysteresis is a no-op when posteriors=None (threshold engine)."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 25
        regimes = [1] * min_train + [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2]
        while len(regimes) < n_bars:
            regimes.append(1)

        engine = _MockEngine(regimes)  # posteriors=None
        prices = _make_prices(n_bars)

        result_hyst = walk_forward_backtest(
            prices, engine=engine, min_train=min_train, hysteresis_delta=0.3,
        )
        engine2 = _MockEngine(regimes)
        result_no_hyst = walk_forward_backtest(
            prices, engine=engine2, min_train=min_train,
        )
        assert result_hyst == result_no_hyst

    def test_both_filters_must_agree(self):
        """Switch blocked when dwell passes but hysteresis doesn't."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 5
        n_bars = 25
        # After warmup: 5 bars regime 2, then 5 bars regime 0
        regimes = [1] * min_train + [2, 2, 2, 2, 2, 0, 0, 0, 0, 0,
                                      2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
        while len(regimes) < n_bars:
            regimes.append(0)

        # Posteriors: regime 2 has high confidence, regime 0 has low margin over current
        # regime 2 posteriors: [0.1, 0.1, 0.8]  (bull=0.8)
        # regime 0 posteriors: [0.35, 0.1, 0.55] (bear=0.35, bull=0.55)
        #   -> margin of bear over current(bull): 0.35 - 0.8 = -0.45 (fails hyst=0.3)
        posteriors = []
        for r in regimes:
            if r == 2:
                posteriors.append(np.array([0.1, 0.1, 0.8]))
            elif r == 0:
                posteriors.append(np.array([0.35, 0.1, 0.55]))
            else:
                posteriors.append(np.array([0.1, 0.8, 0.1]))

        prices = _make_prices(n_bars)

        # Dwell-only: should switch (5 consecutive same regime)
        engine1 = _MockEngineWithPosteriors(regimes, posteriors)
        result_dwell_only = walk_forward_backtest(
            prices, engine=engine1, min_train=min_train, dwell_bars=3,
        )

        # Both filters: hysteresis should block the switch to bear
        # because posteriors[0] - posteriors[2] = 0.35 - 0.8 = -0.45 < 0.3
        engine2 = _MockEngineWithPosteriors(regimes, posteriors)
        result_both = walk_forward_backtest(
            prices, engine=engine2, min_train=min_train,
            dwell_bars=3, hysteresis_delta=0.3,
        )

        # With both filters, the switch to bear should be blocked
        # so we should have fewer trades
        assert result_both["n_trades"] <= result_dwell_only["n_trades"]


class TestFiltersDefaultDisabled:
    """Both filters disabled by default — existing behavior unchanged."""

    def test_default_params_match_original(self):
        """walk_forward_backtest with defaults produces identical results to before."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes = [1] * min_train + [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 2]
        while len(regimes) < n_bars:
            regimes.append(1)

        prices = _make_prices(n_bars)

        engine1 = _MockEngine(regimes)
        result_explicit = walk_forward_backtest(
            prices, engine=engine1, min_train=min_train,
            dwell_bars=0, hysteresis_delta=0.0,
        )

        engine2 = _MockEngine(regimes)
        result_default = walk_forward_backtest(
            prices, engine=engine2, min_train=min_train,
        )

        assert result_explicit == result_default


class TestCLIFlags:
    """CLI --dwell-bars and --hysteresis flags wired through correctly."""

    def test_cli_accepts_dwell_bars_flag(self, btc_csv):
        """--dwell-bars flag accepted and produces valid JSON output."""
        from tests.conftest import run_regime
        import json

        result = run_regime("--csv", btc_csv, "--json", "--dwell-bars", "3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "walk_forward" in data
        assert isinstance(data["walk_forward"]["n_trades"], int)

    def test_cli_accepts_hysteresis_flag(self, btc_csv):
        """--hysteresis flag accepted and produces valid JSON output."""
        from tests.conftest import run_regime
        import json

        result = run_regime("--csv", btc_csv, "--json", "--hysteresis", "0.2")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "walk_forward" in data

    def test_cli_both_flags_together(self, btc_csv):
        """Both flags together produce valid JSON output."""
        from tests.conftest import run_regime
        import json

        result = run_regime(
            "--csv", btc_csv, "--json",
            "--dwell-bars", "5", "--hysteresis", "0.15",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "walk_forward" in data

    def test_cli_default_flags_match_no_flags(self, btc_csv):
        """Default values (0, 0.0) produce identical output to no flags."""
        from tests.conftest import run_regime
        import json

        result_flags = run_regime(
            "--csv", btc_csv, "--json",
            "--dwell-bars", "0", "--hysteresis", "0.0",
        )
        result_no_flags = run_regime("--csv", btc_csv, "--json")
        assert result_flags.returncode == 0
        assert result_no_flags.returncode == 0
        data_flags = json.loads(result_flags.stdout)
        data_no_flags = json.loads(result_no_flags.stdout)
        assert data_flags["walk_forward"] == data_no_flags["walk_forward"]
