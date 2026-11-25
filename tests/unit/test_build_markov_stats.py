"""Unit tests for pipeline._build_markov_stats helper."""

import numpy as np
import pandas as pd

from hmm_futures_analysis.regime.markov_chain import forecast_n_steps
from hmm_futures_analysis.regime.pipeline import MarkovStats, _build_markov_stats


class TestBuildMarkovStats:
    """_build_markov_stats computes MarkovStats from a regimes array."""

    @staticmethod
    def _make_index(n: int) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="B")

    # --- tracer bullet: smoke test ---

    def test_returns_markov_stats(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert isinstance(result, MarkovStats)

    # --- transition matrix ---

    def test_transition_matrix_shape_and_row_sums(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert result.transmat.shape == (3, 3)
        np.testing.assert_allclose(result.transmat.sum(axis=1), 1.0, atol=1e-8)

    # --- stationary distribution ---

    def test_stationary_distribution_sums_to_one(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        np.testing.assert_allclose(result.stationary.sum(), 1.0, atol=1e-8)

    # --- persistence ---

    def test_persistence_has_three_state_keys(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert set(result.persistence.keys()) == {"bear", "sideways", "bull"}

    # --- signal ---

    def test_signal_in_valid_range(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert -1.0 <= result.signal <= 1.0

    # --- current regime and probs ---

    def test_current_regime_is_last_element(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert result.current_regime == 0  # last element

    def test_current_probs_are_one_hot_for_current_regime(self):
        """Issue #112: current_probs is e_current, not T[current_regime]."""
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        np.testing.assert_array_equal(
            result.current_probs, np.eye(3)[result.current_regime]
        )

    def test_current_probs_is_one_hot(self):
        """Synthetic sequence pins the one-hot invariant without HMM fitting."""
        regimes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert result.current_regime == 0  # last element
        np.testing.assert_array_equal(
            result.current_probs, np.array([1.0, 0.0, 0.0])
        )
        # unit vector: exactly one 1, the rest 0
        assert result.current_probs.sum() == 1.0
        assert set(np.unique(result.current_probs)) <= {0.0, 1.0}

    def test_current_probs_one_hot_with_2x2_transmat(self):
        """Construction is shape-robust: 2×2 matrix → 2-element one-hot."""
        idx_4 = self._make_index(5)
        transmat2 = np.array([[0.7, 0.3], [0.4, 0.6]])

        result0 = _build_markov_stats(
            np.array([0, 1, 1, 0]), idx_4, hmm_transmat=transmat2
        )
        np.testing.assert_array_equal(
            result0.current_probs, np.array([1.0, 0.0])
        )

        result1 = _build_markov_stats(
            np.array([0, 1, 0, 1]), idx_4, hmm_transmat=transmat2
        )
        np.testing.assert_array_equal(
            result1.current_probs, np.array([0.0, 1.0])
        )

    def test_signal_is_transition_row_based(self):
        """signal stays P(bull) - P(bear) from T[current_regime] (wire
        contract) and must not be recomputed from the one-hot vector."""
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        expected = (
            result.transmat[result.current_regime, 2]
            - result.transmat[result.current_regime, 0]
        )
        np.testing.assert_allclose(result.signal, expected, rtol=0, atol=1e-15)
        # guard against recomputing signal from the one-hot vector: that would
        # yield exactly 0 or ±1 (here: e_bull - e_bear for e_0 = -1), whereas
        # the row-based wire value is data-derived.
        one_hot_signal = float(
            result.current_probs[2] - result.current_probs[0]
        )
        assert result.signal != one_hot_signal

    def test_forecast_horizons_match_matrix_powers(self):
        """With one-hot current_probs, forecast n-step equals e_current @ T^n
        = T^n[current_regime, :] (issue #112: forecast.1 = exactly T¹,
        and the 5/20-step algebra holds too)."""
        regimes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        for n in (1, 5, 20):
            # forecast_n_steps advances the one-hot row vector: e @ T^n =
            # row current_regime of T^n (for n=1 exactly T[current_regime, :])
            expected = (
                np.linalg.matrix_power(result.transmat, n)[result.current_regime]
            )
            actual = forecast_n_steps(result.transmat, result.current_probs, n)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    # --- regime counts ---

    def test_regime_counts_match_array(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        assert result.regime_counts["bear"] == 3  # 0 appears 3 times
        assert result.regime_counts["sideways"] == 4  # 1 appears 4 times
        assert result.regime_counts["bull"] == 3  # 2 appears 3 times

    # --- dates ---

    def test_dates_from_price_index(self):
        regimes = np.array([0, 1, 2, 1, 0])
        idx = self._make_index(6)  # len(regimes) + 1
        result = _build_markov_stats(regimes, idx)
        assert result.dates["start"] == "2024-01-01"
        assert result.dates["end"] == "2024-01-08"  # 6th business day

    # --- single-regime edge case ---

    def test_all_same_regime(self):
        regimes = np.ones(20, dtype=int)
        idx = self._make_index(21)
        result = _build_markov_stats(regimes, idx)
        assert result.current_regime == 1
        assert result.regime_counts["sideways"] == 20
        assert result.regime_counts["bear"] == 0
        assert result.regime_counts["bull"] == 0
