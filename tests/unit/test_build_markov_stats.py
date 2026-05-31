"""Unit tests for pipeline._build_markov_stats helper."""

import numpy as np
import pandas as pd

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

    def test_current_probs_are_row_of_transmat(self):
        regimes = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
        idx = self._make_index(len(regimes) + 1)
        result = _build_markov_stats(regimes, idx)
        np.testing.assert_array_equal(
            result.current_probs, result.transmat[result.current_regime]
        )

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
