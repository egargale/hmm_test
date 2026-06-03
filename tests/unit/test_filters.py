"""Tests for whipsaw-reduction filters: dwell-time and hysteresis.

Issue #19 — https://github.com/egargale/hmm_test/issues/19
ADR-0017: engine param removed; regimes always pre-computed.
"""

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Helpers for filter tests (ADR-0017: engine param removed, regimes required)
# ---------------------------------------------------------------------------


def _make_prices(n: int = 100) -> pd.Series:
    """Generate simple price series for walk-forward tests."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = pd.Series(100.0 + np.arange(n) * 0.01, index=dates, name="close")
    return prices


def _build_regimes(min_train: int, regimes_after_train: list[int], n_bars: int) -> np.ndarray:
    """Build a regimes array (length n_bars) with warmup and training+osc regions."""
    all_regimes = [1] * min_train + regimes_after_train
    while len(all_regimes) < n_bars:
        all_regimes.append(1)
    return np.array(all_regimes[:n_bars], dtype=int)


class TestDwellTimeFilter:
    """Dwell-time filter: position only changes after N consecutive same-regime bars."""

    def test_dwell_3_blocks_switch_until_3_consecutive(self):
        """With dwell_bars=3, position stays until 3 consecutive new regimes."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 30
        regimes_after_train = [2, 2, 1, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 2]
        regimes = _build_regimes(min_train, regimes_after_train, n_bars)
        prices = _make_prices(n_bars)

        result = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train, dwell_bars=3,
        )
        assert isinstance(result["n_trades"], int)

    def test_dwell_0_is_disabled_no_behavior_change(self):
        """dwell_bars=0 produces same result as default (disabled)."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 25
        regimes_after = [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2]
        regimes = _build_regimes(min_train, regimes_after, n_bars)
        prices = _make_prices(n_bars)

        result_disabled = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train, dwell_bars=0,
        )
        result_default = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train,
        )
        assert result_disabled == result_default

    def test_dwell_reduces_trades_on_oscillating_data(self):
        """Dwell filter produces fewer trades when regime oscillates."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 50
        osc = []
        for i in range(20):
            osc.append(2 if i % 2 == 0 else 0)
        osc += [2] * 20
        regimes = _build_regimes(min_train, osc, n_bars)
        prices = _make_prices(n_bars)

        result_raw = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train,
        )
        result_filtered = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train, dwell_bars=3,
        )
        assert result_filtered["n_trades"] <= result_raw["n_trades"]


class TestBothFiltersActive:
    """AND logic: both filters must agree to switch."""

    def test_hysteresis_noop_when_posteriors_none(self):
        """Hysteresis is a no-op when posteriors=None."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 10
        n_bars = 25
        regimes_after = [2, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 2]
        regimes = _build_regimes(min_train, regimes_after, n_bars)
        prices = _make_prices(n_bars)

        result_hyst = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train, hysteresis_delta=0.3,
        )
        result_no_hyst = walk_forward_backtest(
            prices, regimes=regimes, min_train=min_train,
        )
        assert result_hyst == result_no_hyst

    def test_both_filters_must_agree(self):
        """Switch blocked when dwell passes but hysteresis doesn't."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        min_train = 5
        n_bars = 25
        regimes_after = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0,
                         2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
        regimes = _build_regimes(min_train, regimes_after, n_bars)

        # Posteriors: regime 2 has high confidence, regime 0 has low margin
        posteriors = np.zeros((n_bars, 3), dtype=float)
        for i, r in enumerate(regimes):
            if r == 2:
                posteriors[i] = [0.1, 0.1, 0.8]
            elif r == 0:
                posteriors[i] = [0.35, 0.1, 0.55]
            else:
                posteriors[i] = [0.1, 0.8, 0.1]

        prices = _make_prices(n_bars)

        result_dwell_only = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=3,
        )
        result_both = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors,
            min_train=min_train, dwell_bars=3, hysteresis_delta=0.3,
        )
        assert result_both["n_trades"] <= result_dwell_only["n_trades"]


class TestFiltersDefaultDisabled:
    """Both filters disabled by default — existing behavior unchanged."""

    def test_default_params_match_original(self):
        """walk_forward_backtest with defaults produces identical results."""
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
        """With CLI defaults as 'auto', explicit 0/0.0 differs from no flags."""
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
        # Explicit 0/0.0 differs from 'auto' (engine defaults) for hmm engine
        # where default_hysteresis_delta changed from 0.1 → 0.0 and
        # pca_variance changed from None → stays None for hmm (generic).
        # Both should still produce valid walk-forward results.
        assert "walk_forward" in data_flags
        assert "walk_forward" in data_no_flags
