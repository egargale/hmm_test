"""Integration tests for the regime detection pipeline."""
import json

import numpy as np
import pytest

from regime.markov_chain import (
    build_transition_matrix,
    classify_regimes,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
    forecast_n_steps,
)
from regime.walk_forward import walk_forward_backtest
from data_processing.csv_auto_detect import load_from_csv


class TestThresholdPipeline:
    """Test threshold-based regime classification pipeline."""

    def test_classify_returns_three_states(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        assert len(regimes) == len(returns)
        assert set(np.unique(regimes)).issubset({0, 1, 2})

    def test_transition_matrix_row_sums(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        assert transmat.shape == (3, 3)
        row_sums = transmat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_stationary_distribution_sums_to_one(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        stationary = compute_stationary_distribution(transmat)
        assert len(stationary) == 3
        np.testing.assert_allclose(sum(stationary), 1.0, atol=1e-8)

    def test_signal_in_range(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        current_regime = int(regimes[-1])
        signal = compute_signal(transmat[current_regime])
        assert -1.0 <= signal <= 1.0

    def test_walk_forward_returns_keys(self, btc_csv):
        prices = load_from_csv(btc_csv)
        result = walk_forward_backtest(prices)
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert "n_trades" in result

    def test_walk_forward_drawdown_negative_or_nan(self, btc_csv):
        prices = load_from_csv(btc_csv)
        result = walk_forward_backtest(prices)
        assert result["max_drawdown"] <= 0 or np.isnan(result["max_drawdown"])

    def test_forecast_n_steps(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        current_regime = int(regimes[-1])
        probs = transmat[current_regime]
        forecast = forecast_n_steps(transmat, probs, 5)
        assert len(forecast) == 3
        np.testing.assert_allclose(sum(forecast), 1.0, atol=1e-8)

    def test_small_dataset_handles_gracefully(self, futures_csv):
        prices = load_from_csv(futures_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        # Should still return a valid array even for small data
        assert len(regimes) == len(returns)

    def test_persistence_diagonal_keys(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        persistence = compute_persistence_diagonal(transmat)
        assert "bear" in persistence
        assert "sideways" in persistence
        assert "bull" in persistence
