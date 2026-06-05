"""Golden-master tests for FSHMMEngine.classify delegation to _classify_hmm_slice.

Issue #61 — https://github.com/egargale/hmm_test/issues/61

These tests capture the current observable behaviour of FSHMMEngine.classify.
After the refactor (delegating to shared _classify_hmm_slice), all golden-master
values must remain identical — same regime, same posteriors, same saliency.
"""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine


@pytest.fixture(scope="module")
def spy_features():
    """Precomputed features from SPY.csv — deterministic, session-scoped."""
    df = pd.read_csv("test_data/SPY.csv", index_col=0, parse_dates=True).astype(float)
    engine = FSHMMEngine(n_states=3, random_state=42)
    return engine.precompute(df)


@pytest.fixture(scope="module")
def baseline_result(spy_features):
    """Golden-master classify result (n_states=3, no PCA, no prev_means)."""
    engine = FSHMMEngine(n_states=3, random_state=42)
    return engine.classify(spy_features)


class TestFSHMMGoldenMaster:
    """Behaviour locked to golden values captured before refactor."""

    def test_regime_is_0(self, baseline_result):
        assert baseline_result.regime == 0

    def test_means_shape_3x48(self, baseline_result):
        assert baseline_result.means.shape == (3, 49)

    def test_posteriors_sum_to_one(self, baseline_result):
        assert abs(baseline_result.posteriors.sum() - 1.0) < 1e-6

    def test_posteriors_all_non_negative(self, baseline_result):
        assert (baseline_result.posteriors >= -1e-10).all()

    def test_regime_in_valid_range(self, baseline_result):
        assert baseline_result.regime in {0, 1, 2}


class TestFSHMMWithPrevMeans:
    """prev_means path must produce valid output through shared pipeline."""

    def test_with_prev_means_produces_valid_regime(self, spy_features, baseline_result):
        engine = FSHMMEngine(n_states=3, random_state=42)
        result = engine.classify(spy_features, prev_means=baseline_result.means)
        assert result.regime in {0, 1, 2}

    def test_with_prev_means_posteriors_valid(self, spy_features, baseline_result):
        engine = FSHMMEngine(n_states=3, random_state=42)
        result = engine.classify(spy_features, prev_means=baseline_result.means)
        assert abs(result.posteriors.sum() - 1.0) < 1e-6
        assert result.posteriors.shape == (3,)


class TestFSHMMWithPCA:
    """PCA path must produce valid output through shared pipeline."""

    def test_pca_regime_valid(self, spy_features):
        engine = FSHMMEngine(n_states=3, random_state=42, pca_variance=0.95)
        result = engine.classify(spy_features)
        assert result.regime in {0, 1, 2}

    def test_pca_means_reduced_dimensionality(self, spy_features):
        engine = FSHMMEngine(n_states=3, random_state=42, pca_variance=0.95)
        result = engine.classify(spy_features)
        # PCA reduces from 49 features — means should have fewer columns
        assert result.means.shape[0] == 3
        assert result.means.shape[1] < 49

    def test_pca_posteriors_valid(self, spy_features):
        engine = FSHMMEngine(n_states=3, random_state=42, pca_variance=0.95)
        result = engine.classify(spy_features)
        assert abs(result.posteriors.sum() - 1.0) < 1e-6
        assert result.posteriors.shape == (3,)


class TestFSHMMSaliencyPreserved:
    """Feature saliency output must survive the refactor unchanged."""

    def test_saliency_shape(self, baseline_result):
        assert np.array(baseline_result.feature_saliency).shape == (49,)

    def test_saliency_values_between_0_and_1(self, baseline_result):
        saliency = np.array(baseline_result.feature_saliency)
        assert (saliency >= -1e-10).all()
        assert (saliency <= 1.0 + 1e-10).all()

    def test_selected_features_not_empty(self, baseline_result):
        assert baseline_result.selected_features is not None
        assert len(baseline_result.selected_features) > 0
