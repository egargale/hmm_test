"""Regression tests for FSHMM performance and determinism.

These tests verify:
1. FSHMM classify() completes within a reasonable timeout
2. FSHMM results are deterministic (same random_state → same output)
3. Scipy logsumexp optimization produces valid results
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine


def _make_test_df(n_samples: int = 300, n_features: int = 15) -> pd.DataFrame:
    """Create a small synthetic feature DataFrame for testing."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (n_samples, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


class TestFSHMMPerformance:
    """FSHMM classify() performance regression tests."""

    def test_classify_completes_within_timeout(self):
        """FSHMM classify() should complete within 30s on 500×20 data."""
        rng = np.random.default_rng(42)
        n, d = 500, 20
        df = pd.DataFrame(rng.normal(0, 1, (n, d)))

        engine = FSHMMEngine(n_states=3, random_state=42, max_iter=30)

        start = time.monotonic()
        result = engine.classify(df)
        elapsed = time.monotonic() - start

        assert result.regime in (0, 1, 2)
        assert elapsed < 30.0, f"FSHMM classify took {elapsed:.1f}s, expected < 30s"

    def test_classify_small_data_is_fast(self):
        """FSHMM on small data should complete quickly."""
        rng = np.random.default_rng(42)
        n, d = 200, 10
        df = pd.DataFrame(rng.normal(0, 1, (n, d)))

        engine = FSHMMEngine(n_states=3, random_state=42, max_iter=20)

        start = time.monotonic()
        result = engine.classify(df)
        elapsed = time.monotonic() - start

        assert result.regime in (0, 1, 2)
        # Small data should be fast
        assert elapsed < 10.0, f"FSHMM small classify took {elapsed:.1f}s, expected < 10s"

    def test_pca_classify_is_reasonably_fast(self):
        """FSHMM with PCA should complete within reasonable time."""
        rng = np.random.default_rng(42)
        n, d = 400, 25
        df = pd.DataFrame(rng.normal(0, 1, (n, d)))

        engine = FSHMMEngine(
            n_states=3,
            pca_variance=0.95,
            random_state=42,
            max_iter=25,
        )

        start = time.monotonic()
        result = engine.classify(df)
        elapsed = time.monotonic() - start

        assert result.regime in (0, 1, 2)
        assert elapsed < 45.0, (
            f"FSHMM+PCA classify took {elapsed:.1f}s, expected < 45s"
        )


class TestFSHMMDeterminism:
    """FSHMM results must be deterministic for same inputs."""

    def test_same_seed_same_result(self):
        """Same random_state → same regime, posteriors, saliency."""
        df = _make_test_df()

        engine1 = FSHMMEngine(n_states=3, random_state=42)
        engine2 = FSHMMEngine(n_states=3, random_state=42)

        r1 = engine1.classify(df)
        r2 = engine2.classify(df)

        assert r1.regime == r2.regime
        np.testing.assert_array_almost_equal(r1.posteriors, r2.posteriors)
        np.testing.assert_array_almost_equal(
            r1.feature_saliency, r2.feature_saliency
        )

    def test_different_seed_may_differ(self):
        """Different random_state may produce different results."""
        df = _make_test_df()

        engine1 = FSHMMEngine(n_states=3, random_state=42)
        engine2 = FSHMMEngine(n_states=3, random_state=99)

        r1 = engine1.classify(df)
        r2 = engine2.classify(df)

        # Regime could be same or different — no assertion on equality
        # But both must be valid
        assert r1.regime in (0, 1, 2)
        assert r2.regime in (0, 1, 2)

    def test_saliency_summary_is_reproducible(self):
        """The saliency feature selection is reproducible."""
        df = _make_test_df(n_samples=300, n_features=10)

        engine1 = FSHMMEngine(
            n_states=3, saliency_threshold=0.3, random_state=42
        )
        engine2 = FSHMMEngine(
            n_states=3, saliency_threshold=0.3, random_state=42
        )

        r1 = engine1.classify(df)
        r2 = engine2.classify(df)

        # selected_features should be identical
        assert r1.selected_features == r2.selected_features

    def test_two_runs_same_engine_same_result(self):
        """Running the same engine twice on same data gives same result."""
        df = _make_test_df()

        engine = FSHMMEngine(n_states=3, random_state=42)
        r1 = engine.classify(df)
        r2 = engine.classify(df)

        assert r1.regime == r2.regime
        np.testing.assert_array_almost_equal(r1.feature_saliency, r2.feature_saliency)


class TestFSHMMResultsValid:
    """Verify that FSHMM results are valid after optimization."""

    def test_scipy_logsumexp_is_available(self):
        """scipy.special.logsumexp is importable."""
        from scipy.special import logsumexp
        assert callable(logsumexp)

    def test_results_are_valid_with_optimization(self):
        """FSHMM with scipy logsumexp produces valid regimes and saliency."""
        df = _make_test_df(n_samples=300, n_features=12)

        engine = FSHMMEngine(n_states=3, random_state=42, max_iter=20)

        result = engine.classify(df)

        assert result.regime in (0, 1, 2)
        assert result.posteriors is not None
        assert result.posteriors.shape == (3,)
        assert abs(result.posteriors.sum() - 1.0) < 1e-6
        assert result.feature_saliency is not None
        assert result.feature_saliency.shape == (12,)
        assert np.all(result.feature_saliency >= 0)
        assert np.all(result.feature_saliency <= 1)

    def test_saliency_discriminates_with_logsumexp(self):
        """With scipy logsumexp, signal features still get higher saliency."""
        rng = np.random.default_rng(42)
        n = 300
        n_signal = 5
        n_noise = 10

        # Signal features: regime-dependent means ±2
        signal = np.vstack([
            rng.normal(-2, 0.5, (n // 3, n_signal)),
            rng.normal(0, 0.5, (n // 3, n_signal)),
            rng.normal(2, 0.5, (n - 2 * (n // 3), n_signal)),
        ])
        # Noise features: i.i.d. N(0, 1)
        noise = rng.normal(0, 1, (n, n_noise))
        data = np.hstack([signal, noise])
        cols = [f"signal_{i}" for i in range(n_signal)] + [
            f"noise_{i}" for i in range(n_noise)
        ]
        df = pd.DataFrame(data, columns=cols)

        engine = FSHMMEngine(
            n_states=3,
            saliency_threshold=0.0,
            random_state=42,
            max_iter=25,
        )
        result = engine.classify(df)

        rho = result.feature_saliency
        signal_rho = np.mean(rho[:n_signal])
        noise_rho = np.mean(rho[n_signal:])

        assert signal_rho > noise_rho, (
            f"Signal saliency ({signal_rho:.3f}) should exceed "
            f"noise saliency ({noise_rho:.3f})"
        )
