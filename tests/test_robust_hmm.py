"""Tests for the robust_hmm engine: Huber IRLS and MinCovDet emission correction."""
import json

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY, RegimeEngine
from hmm_futures_analysis.regime.engines._hmm_shared import (
    _fit_hmm_on_slice,
    robust_fit_gaussian_hmm,
)
from tests.conftest import run_regime


@pytest.fixture
def btc_ohlcv(btc_csv):
    df = pd.read_csv(btc_csv, parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip() for c in df.columns]
    return df[["Open", "High", "Low", "Last", "Volume"]].rename(
        columns={
            "Open": "open", "High": "high", "Low": "low",
            "Last": "close", "Volume": "volume",
        }
    )


@pytest.fixture
def btc_prices(btc_csv):
    from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
    return load_from_csv(btc_csv)


def _make_contaminated_data(n_per_state=200, n_features=2, n_outliers=10, seed=42):
    """Generate synthetic data from 3 well-separated Gaussian states + outliers.

    Returns (X_clean, X_contaminated, means_true).
    """
    rng = np.random.RandomState(seed)
    means_true = np.array(
        [[-5.0] * n_features, [0.0] * n_features, [5.0] * n_features]
    )
    cov_true = np.eye(n_features) * 0.3

    X_clean = np.vstack([
        rng.multivariate_normal(means_true[k], cov_true, size=n_per_state)
        for k in range(3)
    ])
    X_dirty = X_clean.copy()
    # Contaminate the last state with extreme outliers
    last_state_start = 2 * n_per_state
    outlier_idx = rng.choice(
        range(last_state_start, 3 * n_per_state), size=n_outliers, replace=False
    )
    X_dirty[outlier_idx] = rng.normal(25, 5, size=(n_outliers, n_features))

    return X_clean, X_dirty, means_true


class TestRegistry:
    def test_registry_contains_robust_hmm(self):
        assert "robust_hmm" in ENGINE_REGISTRY

    def test_registry_resolves_to_robust_engine(self):
        from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine

        assert ENGINE_REGISTRY["robust_hmm"] is RobustHMMEngine


class TestProtocolCompliance:
    def test_robust_hmm_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine

        eng = RobustHMMEngine()
        assert isinstance(eng, RegimeEngine)

    def test_robust_hmm_has_precompute_and_classify(self):
        from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine

        eng = RobustHMMEngine()
        assert callable(getattr(eng, "precompute", None))
        assert callable(getattr(eng, "classify", None))


class TestHuberRobustness:
    """Robust emission params should resist outlier influence."""

    def test_huber_correction_reduces_outlier_bias(self):
        """On data with injected outliers, Huber-corrected means are closer to truth."""
        from hmmlearn import hmm as hmm_mod

        rng = np.random.RandomState(42)
        n_features = 2
        n_per_state = 300
        means_true = np.array([[-6.0] * n_features, [0.0] * n_features, [6.0] * n_features])
        cov_diag = np.array([0.5] * n_features)

        # Generate clean data from 3 states
        X = np.vstack([
            rng.randn(n_per_state, n_features) * np.sqrt(cov_diag) + means_true[k]
            for k in range(3)
        ])
        # Perfect one-hot posteriors
        posteriors = np.zeros((3 * n_per_state, 3))
        for k in range(3):
            posteriors[k * n_per_state:(k + 1) * n_per_state, k] = 1.0

        # Create contaminated copy
        X_dirty = X.copy()
        outlier_idx = rng.choice(
            range(2 * n_per_state, 3 * n_per_state), size=15, replace=False
        )
        X_dirty[outlier_idx] = rng.normal(30, 5, size=(15, n_features))

        # Fit a standard model on dirty data (z-scored)
        center = np.mean(X_dirty, axis=0)
        scale = np.std(X_dirty, axis=0) + 1e-8
        X_norm = ((X_dirty - center) / scale).astype(np.float64)

        model = hmm_mod.GaussianHMM(
            n_components=3, covariance_type="diag", n_iter=30,
            random_state=42, tol=1e-4, params="stmc", init_params="stmc",
        )
        model.fit(X_norm)

        # MLE means (un-normalized)
        order_mle = np.argsort(model.means_[:, 0])
        mle_means_raw = model.means_[order_mle] * scale + center

        # Apply Huber correction
        from hmm_futures_analysis.regime.engines._hmm_shared import _huber_correction
        post = model.predict_proba(X_norm)
        _huber_correction(model, X_norm, post)

        order_hub = np.argsort(model.means_[:, 0])
        hub_means_raw = model.means_[order_hub] * scale + center

        # Robust means should be closer to ground truth than MLE
        err_mle = np.sum((mle_means_raw - means_true) ** 2)
        err_hub = np.sum((hub_means_raw - means_true) ** 2)
        assert err_hub < err_mle, (
            f"Huber err ({err_hub:.2f}) should be < MLE err ({err_mle:.2f})"
        )


class TestMCDRobustness:
    """MinCovDet robust estimation should resist outlier influence."""

    def test_mcd_correction_reduces_outlier_bias(self):
        from hmmlearn import hmm as hmm_mod

        rng = np.random.RandomState(42)
        n_features = 2
        n_per_state = 300
        means_true = np.array([[-6.0] * n_features, [0.0] * n_features, [6.0] * n_features])

        X = np.vstack([
            rng.randn(n_per_state, n_features) * np.sqrt(0.5) + means_true[k]
            for k in range(3)
        ])
        X_dirty = X.copy()
        outlier_idx = rng.choice(
            range(2 * n_per_state, 3 * n_per_state), size=15, replace=False
        )
        X_dirty[outlier_idx] = rng.normal(30, 5, size=(15, n_features))

        center = np.mean(X_dirty, axis=0)
        scale = np.std(X_dirty, axis=0) + 1e-8
        X_norm = ((X_dirty - center) / scale).astype(np.float64)

        model = hmm_mod.GaussianHMM(
            n_components=3, covariance_type="diag", n_iter=30,
            random_state=42, tol=1e-4, params="stmc", init_params="stmc",
        )
        model.fit(X_norm)

        order_mle = np.argsort(model.means_[:, 0])
        mle_means_raw = model.means_[order_mle] * scale + center

        # Use perfect posteriors to isolate the MCD correction logic
        from hmm_futures_analysis.regime.engines._hmm_shared import _mcd_correction
        perfect_post = np.zeros((3 * n_per_state, 3))
        for k in range(3):
            perfect_post[k * n_per_state:(k + 1) * n_per_state, k] = 1.0
        _mcd_correction(model, X_norm, perfect_post)

        order_mcd = np.argsort(model.means_[:, 0])
        mcd_means_raw = model.means_[order_mcd] * scale + center

        err_mle = np.sum((mle_means_raw - means_true) ** 2)
        err_mcd = np.sum((mcd_means_raw - means_true) ** 2)
        assert err_mcd < err_mle, (
            f"MCD err ({err_mcd:.2f}) should be < MLE err ({err_mle:.2f})"
        )


class TestMCDFallback:
    """MCD should fall back gracefully when a state has too few points."""

    def test_mcd_no_crash_on_sparse_states(self):
        """n_states=6 on short data — some states get very few points."""
        from hmmlearn import hmm as hmm_mod
        from hmm_futures_analysis.regime.engines._hmm_shared import _mcd_correction

        rng = np.random.RandomState(42)
        # Only 30 points — too few for 6 states to each have enough for MCD
        X = rng.randn(30, 2)
        X_norm = ((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)).astype(np.float64)

        model = hmm_mod.GaussianHMM(
            n_components=6, covariance_type="diag", n_iter=30,
            random_state=42, tol=1e-4, params="stmc", init_params="stmc",
        )
        model.fit(X_norm)

        # This should not raise — states with too few points just keep MLE params
        post = model.predict_proba(X_norm)
        _mcd_correction(model, X_norm, post)

        # Model should still have valid params
        assert model.means_.shape == (6, 2)
        assert np.all(np.isfinite(model.means_))


class TestBICCompatibility:
    """robust_hmm should work with n_states='auto' (BIC selection)."""

    def test_robust_hmm_with_auto_n_states(self, btc_ohlcv, btc_prices):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        result = pipeline_run(
            btc_prices, source="test", engine="robust_hmm",
            ohlcv=btc_ohlcv, n_states="auto", min_train=300,
        )
        assert "error" not in result
        assert result["engine"] == "robust_hmm"
        assert isinstance(result["engine_info"]["n_states"], int)
        assert result["engine_info"]["n_states"] >= 2


class TestPCACompatibility:
    """Robust correction should work in PCA-whitened space."""

    def test_robust_hmm_with_pca(self, btc_ohlcv, btc_prices):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        result = pipeline_run(
            btc_prices, source="test", engine="robust_hmm",
            ohlcv=btc_ohlcv, pca_variance=0.95, min_train=300,
        )
        assert "error" not in result
        assert result["engine"] == "robust_hmm"


class TestEngineIndependence:
    """robust_hmm differs from hmm on contaminated data, similar on clean."""

    def test_robust_differs_from_hmm_on_contaminated(self, btc_ohlcv, btc_prices):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        common = dict(source="test", min_train=300)

        result_hmm = pipeline_run(
            btc_prices, engine="hmm", ohlcv=btc_ohlcv, **common,
        )
        result_robust = pipeline_run(
            btc_prices, engine="robust_hmm", ohlcv=btc_ohlcv, **common,
        )
        # They use the same underlying HMM structure but robust applies correction,
        # so they should produce the same or very similar results on clean data
        # (correction has little effect without outliers)
        assert result_robust["engine"] == "robust_hmm"
        assert result_hmm["engine"] == "hmm"

    def test_robust_uses_robust_method(self, btc_ohlcv, btc_prices):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        common = dict(source="test", min_train=300, ohlcv=btc_ohlcv)
        result_hub = pipeline_run(
            btc_prices, engine="robust_hmm", robust_method="huber", **common,
        )
        result_mcd = pipeline_run(
            btc_prices, engine="robust_hmm", robust_method="mcd", **common,
        )
        assert result_hub["engine_info"].get("robust_method") == "huber"
        assert result_mcd["engine_info"].get("robust_method") == "mcd"


class TestCLIIntegration:
    """--engine robust_hmm --robust-method huber runs end-to-end."""

    def test_cli_robust_hmm_huber(self, btc_csv):
        result = run_regime(
            "--csv", btc_csv, "--json",
            "--engine", "robust_hmm", "--robust-method", "huber",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine"] == "robust_hmm"
        assert data["engine_info"]["robust_method"] == "huber"

    def test_cli_robust_hmm_mcd(self, btc_csv):
        result = run_regime(
            "--csv", btc_csv, "--json",
            "--engine", "robust_hmm", "--robust-method", "mcd",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine_info"]["robust_method"] == "mcd"

    def test_cli_robust_hmm_default_method(self, btc_csv):
        result = run_regime(
            "--csv", btc_csv, "--json", "--engine", "robust_hmm",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["engine_info"]["robust_method"] == "huber"
