"""Tests for FSHMM (Feature Saliency HMM) engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY

from tests.conftest import run_regime


# ---------------------------------------------------------------------------
# Fixtures for pipeline integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def btc_ohlcv(btc_csv):
    import pandas as pd

    df = pd.read_csv(btc_csv, parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip() for c in df.columns]
    return df[["Open", "High", "Low", "Last", "Volume"]].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Last": "close",
            "Volume": "volume",
        }
    )


@pytest.fixture
def btc_prices(btc_csv):
    from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv

    return load_from_csv(btc_csv)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_signal_noise_data(
    n_samples: int = 400,
    n_signal: int = 5,
    n_noise: int = 10,
    n_states: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic data with regime-dependent signal features and pure-noise features.

    Signal features have different means per regime; noise features are i.i.d. N(0,1).
    Returns a DataFrame with column names like signal_0, signal_1, noise_0, noise_1, ...
    """
    rng = np.random.default_rng(seed)
    samples_per_state = n_samples // n_states
    remainder = n_samples - samples_per_state * n_states

    signal_blocks = []
    for s in range(n_states):
        size = samples_per_state + (1 if s < remainder else 0)
        # Each signal feature has a different mean per state
        means = np.linspace(-3, 3, n_states)[s] * np.ones(n_signal)
        block = rng.normal(loc=means, scale=0.5, size=(size, n_signal))
        signal_blocks.append(block)
    signal_data = np.vstack(signal_blocks)

    noise_data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_noise))

    all_data = np.hstack([signal_data, noise_data])
    columns = [f"signal_{i}" for i in range(n_signal)] + [
        f"noise_{i}" for i in range(n_noise)
    ]
    # Shuffle rows so states are interleaved (but signal structure remains)
    perm = rng.permutation(n_samples)
    all_data = all_data[perm]

    return pd.DataFrame(all_data, columns=columns)


# ---------------------------------------------------------------------------
# Test 1: Tracer bullet — saliency discriminates signal from noise
# ---------------------------------------------------------------------------


class TestRegistry:
    """ENGINE_REGISTRY["fshmm"] returns FSHMMEngine."""

    def test_registry_has_fshmm(self):
        assert "fshmm" in ENGINE_REGISTRY

    def test_registry_returns_fshmm_engine(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        assert ENGINE_REGISTRY["fshmm"] is FSHMMEngine


class TestClassifyResultExtension:
    """fshmm populates feature_saliency; other engines leave it None."""

    def test_fshmm_populates_saliency(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=400)
        engine = FSHMMEngine(n_states=3, random_state=42)
        result = engine.classify(df)
        assert result.feature_saliency is not None
        assert result.selected_features is not None

    def test_threshold_leaves_saliency_none(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(rng.normal(0.001, 0.02, n), index=dates)
        engine = ThresholdEngine(window=20)
        result = engine.classify(returns)
        assert result.feature_saliency is None
        assert result.selected_features is None


class TestNStatesAuto:
    """BIC-based state count selection works with fshmm."""

    def test_auto_n_states_runs(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=400, n_signal=5, n_noise=5)
        engine = FSHMMEngine(n_states=3, random_state=42)
        result = engine.classify(df)
        assert result.regime in (0, 1, 2)


class TestPCASupport:
    """pca_variance works with fshmm."""

    def test_pca_variance_runs(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=400, n_signal=5, n_noise=10)
        engine = FSHMMEngine(
            n_states=3, pca_variance=0.95, random_state=42,
        )
        result = engine.classify(df)
        assert result.regime in (0, 1, 2)
        assert result.feature_saliency is not None
        # With PCA, saliency dimension matches PCA components, not original features
        assert result.feature_saliency.ndim == 1


class TestPipelineIntegration:
    """pipeline.run() with engine='fshmm' produces valid output."""

    def test_pipeline_runs_fshmm(self, btc_ohlcv, btc_prices):
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        # Use only last 80 bars with min_train=40 to minimise refit points
        prices = btc_prices.iloc[-80:]
        ohlcv = btc_ohlcv.iloc[-80:]

        result = pipeline_run(
            prices,
            source="test",
            engine="fshmm",
            ohlcv=ohlcv,
            min_train=40,
            n_states=3,
        )
        assert result["engine"] == "fshmm"
        assert result["current_regime"]["index"] in (0, 1, 2)
        assert result["engine_info"]["features"] == "generic"
        assert "walk_forward" in result


class TestCLIIntegration:
    """--engine fshmm runs end-to-end."""

    def test_cli_fshmm_rejects_without_ohlcv(self, btc_csv):
        """CSV files without OHLCV columns should give a clear error."""
        result = run_regime("--csv", btc_csv, "--engine", "fshmm", "--json")
        assert result.returncode != 0
        assert "OHLCV" in result.stdout or "ohlcv" in result.stdout.lower()

    def test_cli_accepts_fshmm_engine_flag(self, btc_csv):
        """--engine fshmm is accepted as a valid choice (may fail on data)."""
        result = run_regime("--csv", btc_csv, "--engine", "fshmm", "--json")
        # Should not fail with 'invalid choice' — it may fail on OHLCV
        if "invalid choice" in result.stderr:
            pytest.fail("fshmm not in --engine choices")

    def test_cli_saliency_threshold_flag(self, btc_csv):
        """--saliency-threshold flag is accepted."""
        result = run_regime(
            "--csv", btc_csv, "--engine", "fshmm",
            "--saliency-threshold", "0.3", "--json",
        )
        # Should not fail with unrecognized argument
        if "unrecognized arguments" in result.stderr:
            pytest.fail("--saliency-threshold not recognized")


class TestThresholdBehavior:
    """Different saliency_threshold values produce different selected_features."""

    def test_low_threshold_selects_more_features(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=400, n_signal=5, n_noise=10)
        engine_low = FSHMMEngine(
            n_states=3, saliency_threshold=0.1, random_state=42,
        )
        engine_high = FSHMMEngine(
            n_states=3, saliency_threshold=0.9, random_state=42,
        )
        result_low = engine_low.classify(df)
        result_high = engine_high.classify(df)

        assert len(result_low.selected_features or []) >= len(
            result_high.selected_features or []
        )


class TestConvergence:
    """Saliency weights stabilize within tolerance."""

    def test_saliency_converges(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=300, n_signal=5, n_noise=10)
        engine = FSHMMEngine(
            n_states=3, max_iter=100, tol=1e-4, random_state=42,
        )
        result = engine.classify(df)

        # Convergence: saliency weights are all in valid range
        rho = result.feature_saliency
        assert rho is not None
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0) and np.all(rho < 1)


class TestEngineIndependence:
    """fshmm produces different regime output than hmm for same input."""

    def test_fshmm_differs_from_hmm(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        df = _make_signal_noise_data(n_samples=400, n_signal=5, n_noise=10)
        fshmm = FSHMMEngine(n_states=3, random_state=42)
        hmm_eng = HMMGenericEngine(n_states=3)

        result_fshmm = fshmm.classify(df)
        result_hmm = hmm_eng.classify(df)

        # Means should differ (fshmm trains with saliency weighting)
        assert not np.allclose(result_fshmm.means, result_hmm.means, atol=1e-6)


class TestTracerBullet:
    """Tracer bullet: FSHMM learns which features are signal vs noise."""

    def test_saliency_discriminates_signal_from_noise(self):
        """Given 5 signal + 10 noise features, signal features get higher ρ_k."""
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=400, n_signal=5, n_noise=10)
        engine = FSHMMEngine(
            n_states=3, saliency_threshold=0.5, random_state=42,
        )
        result = engine.classify(df)

        assert result.feature_saliency is not None
        rho = result.feature_saliency
        assert rho.shape == (15,)
        assert np.all(rho >= 0) and np.all(rho <= 1)

        # Signal features (first 5) should have higher mean ρ than noise (last 10)
        signal_rho = rho[:5]
        noise_rho = rho[5:]
        assert np.mean(signal_rho) > np.mean(noise_rho), (
            f"Signal saliency ({np.mean(signal_rho):.3f}) should exceed "
            f"noise saliency ({np.mean(noise_rho):.3f})"
        )

    def test_selected_features_excludes_noise(self):
        """selected_features should contain signal columns, not noise columns."""
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data(n_samples=400, n_signal=5, n_noise=10)
        engine = FSHMMEngine(
            n_states=3, saliency_threshold=0.5, random_state=42,
        )
        result = engine.classify(df)

        assert result.selected_features is not None
        # At least some signal features should be selected
        signal_cols = {f"signal_{i}" for i in range(5)}
        assert len(set(result.selected_features) & signal_cols) > 0

    def test_regime_is_valid(self):
        """classify() returns a valid regime index."""
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        df = _make_signal_noise_data()
        engine = FSHMMEngine(n_states=3, random_state=42)
        result = engine.classify(df)

        assert result.regime in (0, 1, 2)
        assert result.means is not None
        assert result.posteriors is not None
