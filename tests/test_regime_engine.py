"""Tests for RegimeEngine protocol and engine registry (ADR-002)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_regime(*args):
    """Run hmm_futures_analysis/cli.py with args, return CompletedProcess."""
    cmd = [sys.executable, "-m", "hmm_futures_analysis.cli"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))


class TestProtocolConformance:
    """Each engine class satisfies the RegimeEngine protocol."""

    def test_threshold_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        engine = ThresholdEngine(window=20, threshold=0.05)
        assert isinstance(engine, RegimeEngine)

    def test_hmm_generic_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        assert isinstance(engine, RegimeEngine)

    def test_hmm_messina_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        assert isinstance(engine, RegimeEngine)


class TestThresholdEngine:
    """ThresholdEngine classify returns expected regime indices."""

    @pytest.fixture
    def returns_series(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.Series(np.random.normal(0.001, 0.02, n), index=dates)

    def test_classify_returns_regime_in_valid_range(self, returns_series):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns_series)
        assert result.regime in {0, 1, 2}
        assert result.means is None

    def test_precompute_returns_none(self, returns_series):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        engine = ThresholdEngine(window=20, threshold=0.05)
        assert engine.precompute(returns_series) is None

    def test_classify_bull_when_strong_positive_returns(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(0.01, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.regime == 2

    def test_classify_bear_when_strong_negative_returns(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(-0.01, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.regime == 0

    def test_classify_sideways_when_flat_returns(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(0.001, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.regime == 1


class TestHMMGenericEngine:
    """HMMGenericEngine precompute and classify."""

    @pytest.fixture
    def ohlcv_data(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        close = np.maximum(close, 1.0)
        return pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
                "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n).astype(float),
            },
            index=dates,
        )

    def test_precompute_returns_dataframe(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert features.shape[1] > 1

    def test_classify_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means is not None
        assert result.means.shape[0] == 3

    def test_classify_with_prev_means_preserves_labels(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        first = engine.classify(features)
        assert first.regime in {0, 1, 2}
        assert first.means is not None
        second = engine.classify(features, prev_means=first.means)
        assert second.regime in {0, 1, 2}
        assert second.means is not None

    def test_classify_with_2_states_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=2)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1}
        assert result.means is not None
        assert result.means.shape[0] == 2

    def test_classify_with_4_states_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=4)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2, 3}
        assert result.means is not None
        assert result.means.shape[0] == 4


class TestHMMMMessinaEngine:
    """HMMMMessinaEngine precompute and classify."""

    @pytest.fixture
    def ohlcv_data(self):
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        close = np.maximum(close, 1.0)
        return pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
                "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n).astype(float),
            },
            index=dates,
        )

    def test_precompute_returns_dataframe(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_classify_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means is not None
        assert result.means.shape[0] == 3

    def test_classify_with_prev_means(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        first = engine.classify(features)
        second = engine.classify(features, prev_means=first.means)
        assert second.regime in {0, 1, 2}

    def test_classify_with_2_states_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=2)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1}
        assert result.means is not None
        assert result.means.shape[0] == 2


class TestSelectNStates:
    """BIC-based state count selection (Issue #17)."""

    @pytest.fixture
    def synthetic_3state_features(self):
        """3-state synthetic feature data (each state draws from a different mean)."""
        np.random.seed(42)
        n_per = 100
        # Three Gaussians with well-separated means in 2D
        block_a = np.random.normal(loc=-5.0, scale=0.5, size=(n_per, 2))
        block_b = np.random.normal(loc=0.0, scale=0.5, size=(n_per, 2))
        block_c = np.random.normal(loc=5.0, scale=0.5, size=(n_per, 2))
        return np.vstack([block_a, block_b, block_c])

    def test_select_n_states_returns_int_in_range(self, synthetic_3state_features):
        from hmm_futures_analysis.regime.engines._hmm_shared import select_n_states

        result = select_n_states(synthetic_3state_features, max_states=5)
        assert isinstance(result, int)
        assert 2 <= result <= 5

    def test_select_n_states_picks_3_for_3state_data(self, synthetic_3state_features):
        """BIC should favor 3 states for data generated from 3 Gaussians."""
        from hmm_futures_analysis.regime.engines._hmm_shared import select_n_states

        result = select_n_states(synthetic_3state_features, max_states=6)
        assert result == 3

    def test_select_n_states_short_data_caps_max_states(self):
        """When data is short, max_states should be capped to avoid overfitting."""
        from unittest.mock import patch
        from hmm_futures_analysis.regime.engines._hmm_shared import select_n_states

        # Only 15 rows — should never attempt to fit more than 1 state
        # (15 // 10 = 1), which means max_states clamped to min(max_states, 1) = 1,
        # then floored to 2. So only k=2 should be tried.
        np.random.seed(42)
        tiny = np.random.randn(15, 2)

        original_fit = __import__(
            "hmm_futures_analysis.regime.engines._hmm_shared",
            fromlist=["_fit_hmm_on_slice"],
        )._fit_hmm_on_slice
        fit_calls = []

        def tracking_fit(features, n_states=3, random_state=42):
            fit_calls.append(n_states)
            return original_fit(features, n_states=n_states, random_state=random_state)

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_shared._fit_hmm_on_slice",
            side_effect=tracking_fit,
        ) as mock_fit:
            mock_fit.side_effect = lambda *a, **kw: (
                fit_calls.append(kw.get("n_states", 3)),
                original_fit(*a, **kw),
            )[1]
            result = select_n_states(tiny, max_states=6)

        # Should only try k=2, never k=3,4,5,6
        assert set(fit_calls) == {2}
        assert result == 2


class TestEngineRegistry:
    """ENGINE_REGISTRY maps strings to correct classes."""

    def test_registry_has_three_engines(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY

        assert set(ENGINE_REGISTRY.keys()) == {
            "threshold",
            "hmm",
            "messina",
            "robust_hmm",
        }

    def test_registry_threshold_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        assert ENGINE_REGISTRY["threshold"] is ThresholdEngine

    def test_registry_hmm_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        assert ENGINE_REGISTRY["hmm"] is HMMGenericEngine

    def test_registry_messina_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        assert ENGINE_REGISTRY["messina"] is HMMMMessinaEngine

    def test_invalid_engine_raises_key_error(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY

        with pytest.raises(KeyError):
            ENGINE_REGISTRY["invalid"]


class TestWalkForwardWithProtocol:
    """Walk-forward backtest works with protocol-based engine dispatch."""

    @pytest.fixture
    def prices(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        return pd.Series(close, index=dates, name="close")

    def test_walk_forward_with_mock_engine(self, prices):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        class MockEngine:
            def precompute(self, data):
                return None

            def classify(self, data, prev_means=None):
                return ClassifyResult(regime=2)

        result = walk_forward_backtest(prices, engine=MockEngine(), min_train=50)
        assert "sharpe" in result
        assert "n_trades" in result
        assert isinstance(result["n_trades"], int)

    def test_walk_forward_rejects_invalid_engine_string(self, prices):
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        with pytest.raises(ValueError, match=r"engine"):
            walk_forward_backtest(prices, engine="nonexistent")

    def test_walk_forward_hmm_engine_requires_ohlcv(self, prices):
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        with pytest.raises(ValueError, match=r"OHLCV"):
            walk_forward_backtest(prices, engine="hmm")

    def test_walk_forward_threshold_string_backward_compat(self, prices):
        """String 'threshold' still works through registry dispatch."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        result = walk_forward_backtest(prices, engine="threshold", min_train=50)
        assert "sharpe" in result
        assert isinstance(result["n_trades"], int)


class TestCLINStatesArg:
    """CLI --n-states accepts 'auto' and integers >= 2."""

    def test_n_states_auto_accepted(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "threshold", "--n-states", "auto", "--json"
        )
        assert proc.returncode == 0

    def test_n_states_3_backward_compat(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "threshold", "--n-states", "3", "--json"
        )
        assert proc.returncode == 0

    def test_n_states_1_rejected(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "hmm", "--n-states", "1", "--json"
        )
        assert proc.returncode != 0

    def test_n_states_invalid_string_rejected(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "hmm", "--n-states", "foo", "--json"
        )
        assert proc.returncode != 0


class TestPipelineAutoNStates:
    """Pipeline resolves n_states='auto' via BIC selection."""

    @pytest.fixture
    def prices_and_ohlcv(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        close = np.maximum(close, 1.0)
        ohlcv = pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
                "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n).astype(float),
            },
            index=dates,
        )
        prices = pd.Series(close, index=dates, name="close")
        return prices, ohlcv

    def test_auto_resolves_to_integer(self, prices_and_ohlcv):
        """pipeline_run with n_states='auto' resolves to an integer."""
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv
        output = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states="auto",
        )
        # Should succeed and return a valid result
        assert "engine_info" in output
        n_used = output["engine_info"]["n_states"]
        assert isinstance(n_used, int)
        assert n_used >= 2

    def test_auto_uses_bic_selection(self, prices_and_ohlcv):
        """n_states='auto' should use select_n_states under the hood."""
        from unittest.mock import patch
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv

        with patch(
            "hmm_futures_analysis.regime.pipeline.select_n_states",
            return_value=3,
        ) as mock_bic:
            _ = pipeline_run(
                prices,
                source="test",
                engine="hmm",
                ohlcv=ohlcv,
                n_states="auto",
            )
            mock_bic.assert_called_once()

    def test_auto_threshold_ignores_bic(self, prices_and_ohlcv):
        """n_states='auto' with threshold engine doesn't call select_n_states."""
        from unittest.mock import patch
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv

        with patch(
            "hmm_futures_analysis.regime.pipeline.select_n_states",
            return_value=3,
        ) as mock_bic:
            _ = pipeline_run(
                prices,
                source="test",
                engine="threshold",
                n_states="auto",
            )
            mock_bic.assert_not_called()
