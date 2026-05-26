"""Tests for RegimeEngine protocol and engine registry (ADR-002)."""
import numpy as np
import pandas as pd
import pytest


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
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult
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


class TestEngineRegistry:
    """ENGINE_REGISTRY maps strings to correct classes."""

    def test_registry_has_three_engines(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY

        assert set(ENGINE_REGISTRY.keys()) == {"threshold", "hmm", "messina"}

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
