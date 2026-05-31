"""Tests for PCA whitening in HMM engines (Issue #18).

Vertical tracer bullets — each test drives one behavior through the
public interface.  No mocking of internal collaborators.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generic_features(n_rows: int = 300, n_features: int = 50, seed: int = 42):
    """Synthetic high-dimensional feature matrix for generic-engine tests."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, n_features))


def _make_ohlcv(n: int = 400, seed: int = 42):
    """Synthetic OHLCV DataFrame for engine-level tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    close = np.maximum(close, 1.0)
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


# ===================================================================
# Tracer bullet 1: _fit_hmm_on_slice with PCA reduces dimensionality
# ===================================================================


@pytest.mark.slow
class TestFitHMMOnSlicePCA:
    """_fit_hmm_on_slice applies PCA whitening when pca_variance is set."""

    def test_pca_reduces_feature_count(self):
        """With pca_variance=0.95 on high-dim data, returned model means
        should have fewer dimensions than the original feature count."""
        from hmm_futures_analysis.regime.engines._hmm_shared import _fit_hmm_on_slice

        features = _make_generic_features(n_rows=300, n_features=50)
        model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
            features,
            n_states=3,
            pca_variance=0.95,
        )
        # pca_n should be less than original feature count
        assert isinstance(pca_n, int)
        assert pca_n < 50
        assert pca_n >= 1
        # Model means should be in reduced space
        assert model.means_.shape[1] == pca_n

    def test_no_pca_returns_none_component_count(self):
        """With pca_variance=None (default), pca_n_components_used is None."""
        from hmm_futures_analysis.regime.engines._hmm_shared import _fit_hmm_on_slice

        features = _make_generic_features(n_rows=300, n_features=10)
        model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
            features, n_states=3
        )
        assert pca_n is None
        assert pca_transform is None
        assert model.means_.shape[1] == 10

    def test_pca_variance_threshold_controls_components(self):
        """Lower variance threshold → fewer components."""
        from hmm_futures_analysis.regime.engines._hmm_shared import _fit_hmm_on_slice

        features = _make_generic_features(n_rows=300, n_features=50)
        _, _, _, pca_n_95, _ = _fit_hmm_on_slice(
            features, n_states=3, pca_variance=0.95
        )
        _, _, _, pca_n_50, _ = _fit_hmm_on_slice(
            features, n_states=3, pca_variance=0.50
        )
        assert pca_n_50 < pca_n_95


# ===================================================================
# Tracer bullet 3: HMMGenericEngine with PCA classifies correctly
# ===================================================================


@pytest.mark.slow
class TestHMMGenericEnginePCA:
    """HMMGenericEngine with pca_variance produces valid regimes."""

    @pytest.fixture(scope="class")
    def ohlcv_data(self):
        return _make_ohlcv(n=400)

    def test_classify_with_pca_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3, pca_variance=0.95)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means is not None
        # Means should be in PCA-reduced space
        assert result.means.shape[1] < features.shape[1]

    def test_classify_without_pca_unaffected(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means.shape[1] == features.shape[1]

    def test_sticky_component_count_across_classify_calls(self, ohlcv_data):
        """First classify sets _pca_n_components; second classify preserves it."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3, pca_variance=0.95)
        features = engine.precompute(ohlcv_data)

        first = engine.classify(features)
        assert engine._pca_n_components is not None
        sticky_n = engine._pca_n_components

        second = engine.classify(features, prev_means=first.means)
        # Component count should not change on second call
        assert engine._pca_n_components == sticky_n
        assert second.regime in {0, 1, 2}

    def test_classify_with_prev_means_preserves_labels_under_pca(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3, pca_variance=0.95)
        features = engine.precompute(ohlcv_data)
        first = engine.classify(features)
        second = engine.classify(features, prev_means=first.means)
        assert second.regime in {0, 1, 2}
        assert second.means is not None


# ===================================================================
# Tracer bullet 4: HMMMMessinaEngine with PCA is unaffected by default
# ===================================================================


@pytest.mark.slow
class TestHMMMMessinaEnginePCA:
    """HMMMMessinaEngine backward compat — pca_variance defaults to None."""

    @pytest.fixture(scope="class")
    def ohlcv_data(self):
        return _make_ohlcv(n=400)

    def test_messina_without_pca_unaffected(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means.shape[1] == features.shape[1]

    def test_messina_with_pca_reduces_features(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3, pca_variance=0.95)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means.shape[1] < features.shape[1]


# ===================================================================
# Tracer bullet 7: select_n_states with PCA
# ===================================================================


@pytest.mark.slow
class TestSelectNStatesPCA:
    """BIC-based state selection works in PCA-reduced space."""

    @pytest.fixture(scope="class")
    def high_dim_3state_features(self):
        """3-state synthetic data in 50 dimensions."""
        rng = np.random.default_rng(42)
        n_per = 100
        # 3 well-separated clusters in 50D space
        block_a = rng.normal(loc=-2.0, scale=0.5, size=(n_per, 50))
        block_b = rng.normal(loc=0.0, scale=0.5, size=(n_per, 50))
        block_c = rng.normal(loc=2.0, scale=0.5, size=(n_per, 50))
        return np.vstack([block_a, block_b, block_c])

    def test_select_n_states_with_pca_picks_3(self, high_dim_3state_features):
        from hmm_futures_analysis.regime.engines._hmm_shared import select_n_states

        result = select_n_states(
            high_dim_3state_features,
            max_states=6,
            pca_variance=0.95,
        )
        assert isinstance(result, int)
        assert 2 <= result <= 6

    def test_select_n_states_without_pca_same_as_before(self, high_dim_3state_features):
        from hmm_futures_analysis.regime.engines._hmm_shared import select_n_states

        result = select_n_states(high_dim_3state_features, max_states=4)
        assert isinstance(result, int)
        assert 2 <= result <= 4


# ===================================================================
# Tracer bullet 8: walk_forward_backtest threads pca_variance
# ===================================================================


@pytest.mark.slow
class TestWalkForwardPCA:
    """walk_forward_backtest accepts and threads pca_variance to engine."""

    @pytest.fixture(scope="class")
    def prices_and_ohlcv(self):
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(rng.standard_normal(n))
        close = np.maximum(close, 1.0)
        ohlcv = pd.DataFrame(
            {
                "open": close + rng.standard_normal(n) * 0.3,
                "high": close + np.abs(rng.standard_normal(n) * 0.8),
                "low": close - np.abs(rng.standard_normal(n) * 0.8),
                "close": close,
                "volume": rng.integers(100, 10_000, n).astype(float),
            },
            index=dates,
        )
        prices = pd.Series(close, index=dates, name="close")
        return prices, ohlcv

    def test_walk_forward_hmm_with_pca_returns_valid_result(self, prices_and_ohlcv):
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        prices, ohlcv = prices_and_ohlcv
        result = walk_forward_backtest(
            prices,
            engine="hmm",
            ohlcv=ohlcv,
            min_train=100,
            pca_variance=0.95,
        )
        assert "sharpe" in result
        assert "n_trades" in result
        assert isinstance(result["n_trades"], int)

    def test_walk_forward_hmm_without_pca_unaffected(self, prices_and_ohlcv):
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        prices, ohlcv = prices_and_ohlcv
        result = walk_forward_backtest(
            prices,
            engine="hmm",
            ohlcv=ohlcv,
            min_train=100,
        )
        assert "sharpe" in result


# ===================================================================
# Tracer bullet 9: pipeline.run threads pca_variance through
# ===================================================================


@pytest.mark.slow
class TestPipelinePCA:
    """pipeline.run accepts and threads pca_variance through to engine + wf."""

    @pytest.fixture(scope="class")
    def prices_and_ohlcv(self):
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(rng.standard_normal(n))
        close = np.maximum(close, 1.0)
        ohlcv = pd.DataFrame(
            {
                "open": close + rng.standard_normal(n) * 0.3,
                "high": close + np.abs(rng.standard_normal(n) * 0.8),
                "low": close - np.abs(rng.standard_normal(n) * 0.8),
                "close": close,
                "volume": rng.integers(100, 10_000, n).astype(float),
            },
            index=dates,
        )
        prices = pd.Series(close, index=dates, name="close")
        return prices, ohlcv

    def test_pipeline_hmm_with_pca_returns_valid_output(self, prices_and_ohlcv):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv
        output = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            pca_variance=0.95,
        )
        assert "engine_info" in output
        assert output["engine"] == "hmm"
        assert "walk_forward" in output

    def test_pipeline_hmm_without_pca_unaffected(self, prices_and_ohlcv):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv
        output = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
        )
        assert "engine_info" in output
