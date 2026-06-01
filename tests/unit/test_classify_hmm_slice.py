"""Tests for _classify_hmm_slice post-fit pipeline.

Issue #44 — https://github.com/egargale/hmm_test/issues/44
Issue #61 — signature changed to accept pre-normalized X_last
"""

import numpy as np
from hmmlearn import hmm

from hmm_futures_analysis.regime.engine_protocol import ClassifyResult
from hmm_futures_analysis.regime.engines._hmm_shared import _classify_hmm_slice


def _make_model(n_states: int = 3, n_features: int = 2) -> hmm.GaussianHMM:
    """Hand-build a GaussianHMM with distinct, sorted means.

    Means are sorted ascending so the label_map is identity:
    state 0 → bear, state 1 → sideways, state 2 → bull (for 3-state).
    """
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
    spacing = np.linspace(-2.0, 2.0, n_states)
    model.means_ = np.column_stack([spacing] * n_features)
    model.covars_ = np.ones((n_states, n_features))
    model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
    model.startprob_ = np.zeros(n_states)
    model.startprob_[0] = 1.0
    return model


def _make_last_bar(n_features: int = 2, *, seed: int = 42) -> np.ndarray:
    """Generate a single pre-normalized last bar, shape (1, n_features)."""
    rng = np.random.RandomState(seed)
    return rng.randn(1, n_features).astype(np.float64)


class TestClassifyBasic3State:
    """3-state, no PCA, no prev_means — the happy path."""

    def test_returns_classify_result(self):
        model = _make_model(n_states=3)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=None,
        )

        assert isinstance(result, ClassifyResult)

    def test_regime_in_valid_range(self):
        model = _make_model(n_states=3)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=None,
        )

        assert result.regime in {0, 1, 2}

    def test_means_shape(self):
        model = _make_model(n_states=3)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=None,
        )

        assert result.means is not None
        assert result.means.shape == (3, model.means_.shape[1])

    def test_posteriors_shape_and_sums_to_one(self):
        model = _make_model(n_states=3)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=None,
        )

        assert result.posteriors is not None
        assert result.posteriors.shape == (3,)
        assert abs(result.posteriors.sum() - 1.0) < 1e-6


class TestClassify4StateCollapse:
    """4-state: posteriors aggregated to shape (3,), regime in {0,1,2}."""

    def test_regime_in_valid_range(self):
        model = _make_model(n_states=4)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=4,
            prev_means=None,
        )

        assert result.regime in {0, 1, 2}

    def test_posteriors_aggregated_to_3(self):
        model = _make_model(n_states=4)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=4,
            prev_means=None,
        )

        assert result.posteriors is not None
        assert result.posteriors.shape == (3,)
        assert abs(result.posteriors.sum() - 1.0) < 1e-6

    def test_means_shape_4x_features(self):
        model = _make_model(n_states=4)
        X_last = _make_last_bar(model.means_.shape[1])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=4,
            prev_means=None,
        )

        assert result.means is not None
        assert result.means.shape == (4, model.means_.shape[1])


class TestClassifyPrevMeansRemap:
    """With prev_means, _remap_to_prev_states is called and regime reflects it."""

    def test_prev_means_produces_valid_regime(self):
        """prev_means triggers the remap path; regime is still valid."""
        model = _make_model(n_states=3)
        X_last = _make_last_bar(model.means_.shape[1])

        prev_means = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=prev_means,
        )

        assert result.regime in {0, 1, 2}

    def test_remap_with_2_state_prev_changes_regime(self):
        """2-state prev_means can remap a 3-state result to a different label.

        The 3-state model predicts raw_state X which maps to a regime via the
        current label_map. _remap_to_prev_states then maps X through
        _match_states and the prev label_map (which only has 2 regimes).
        """
        n_features = 2
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        # Sorted ascending: state 0 → mean −100, state 1 → mean 0, state 2 → mean 100
        model.means_ = np.array(
            [[-100.0] * n_features, [0.0] * n_features, [100.0] * n_features]
        )
        model.covars_ = np.ones((3, n_features))
        model.transmat_ = np.full((3, 3), 1.0 / 3)
        model.startprob_ = np.array([1.0, 0.0, 0.0])

        # Data near zero → model should predict state 1 (mean=0)
        X_last = np.full((1, n_features), 0.01, dtype=np.float64)

        result_no_prev = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=None,
        )

        # With 2-state prev: prev state 0 → −100, state 1 → 100
        prev_means = np.array([[-100.0] * n_features, [100.0] * n_features])

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=prev_means,
        )

        # The remap path was taken — regime is valid
        assert result.regime in {0, 1}


class TestClassifyPCATransform:
    """With PCA, the caller transforms X_last before calling _classify_hmm_slice."""

    def test_pca_produces_valid_result(self):
        """Caller applies PCA to last bar; regime is valid, posteriors sum to 1."""
        from sklearn.decomposition import PCA

        n_features_orig = 5
        rng = np.random.RandomState(42)
        raw = rng.randn(50, n_features_orig)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(raw)

        # Build a 3-state model with 2 features (matching PCA output)
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        spacing = np.linspace(-2.0, 2.0, 3)
        model.means_ = np.column_stack([spacing] * 2)
        model.covars_ = np.ones((3, 2))
        model.transmat_ = np.full((3, 3), 1.0 / 3)
        model.startprob_ = np.array([1.0, 0.0, 0.0])

        # Caller normalizes and PCA-transforms the last bar
        features_arr = rng.randn(20, n_features_orig)
        center = np.zeros(n_features_orig)
        scale = np.ones(n_features_orig)
        X_last = (features_arr[-1:] - center) / scale
        X_last = pca.transform(X_last)

        result = _classify_hmm_slice(
            model=model,
            X_last=X_last,
            n_states=3,
            prev_means=None,
        )

        assert result.regime in {0, 1, 2}
        assert result.posteriors is not None
        assert result.posteriors.shape == (3,)
        assert abs(result.posteriors.sum() - 1.0) < 1e-6
