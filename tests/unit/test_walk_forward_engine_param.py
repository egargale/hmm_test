"""Tests for walk_forward_backtest accepting only RegimeEngine instances.

Issue #48 — Remove _resolve_engine() and old dispatch from walk_forward.py

These tests verify the public interface contract: walk_forward_backtest
takes a RegimeEngine instance, not a string name and not engine-specific kwargs.
"""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine
from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest


def _make_prices(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate price series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(0.0001, 0.01, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates, name="close")
    return prices


class TestWalkForwardAcceptsEngineInstance:
    """walk_forward_backtest uses a RegimeEngine instance directly."""

    def test_threshold_engine_produces_valid_result(self):
        prices = _make_prices()
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = walk_forward_backtest(prices, engine=engine, min_train=50)
        # Result dict has all expected keys
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert "n_trades" in result
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "total_return" in result

    def test_threshold_engine_result_types(self):
        prices = _make_prices()
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = walk_forward_backtest(prices, engine=engine, min_train=50)
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["max_drawdown"], float)
        assert isinstance(result["n_trades"], int)
        assert isinstance(result["total_return"], float)


class TestWalkForwardRejectsStringEngine:
    """Passing engine as a string name is no longer supported."""

    def test_string_engine_raises_type_error(self):
        prices = _make_prices()
        with pytest.raises(TypeError, match="engine"):
            walk_forward_backtest(prices, engine="threshold", min_train=50)


class TestWalkForwardRejectsEngineKwargs:
    """Engine-specific kwargs are removed from the signature."""

    @pytest.mark.parametrize(
        "kwarg,val",
        [
            ("window", 20),
            ("threshold", 0.05),
            ("n_states", 3),
            ("ohlcv", None),
            ("pca_variance", 0.95),
            ("robust_method", "huber"),
            ("saliency_threshold", 0.5),
        ],
    )
    def test_engine_kwarg_raises_type_error(self, kwarg, val):
        prices = _make_prices()
        engine = ThresholdEngine()
        with pytest.raises(TypeError):
            walk_forward_backtest(prices, engine=engine, min_train=50, **{kwarg: val})


class TestWalkForwardPrecomputedPath:
    """Pre-computed regimes/posteriors bypass engine entirely."""

    def test_regimes_posteriors_path_works(self):
        prices = _make_prices()
        # regimes/posteriors aligned to returns length (n-1 from prices)
        n = len(prices) - 1
        regimes = np.random.default_rng(0).integers(0, 3, n)
        posteriors = np.random.default_rng(1).random((n, 3))
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        result = walk_forward_backtest(
            prices,
            engine=ThresholdEngine(),
            min_train=50,
            regimes=regimes,
            posteriors=posteriors,
        )
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["n_trades"], int)


class TestDeletedInternals:
    """_resolve_engine and _VALID_ENGINES no longer exist."""

    def test_resolve_engine_removed(self):
        import hmm_futures_analysis.regime.walk_forward as wf_mod

        assert not hasattr(wf_mod, "_resolve_engine")

    def test_valid_engines_removed(self):
        import hmm_futures_analysis.regime.walk_forward as wf_mod

        assert not hasattr(wf_mod, "_VALID_ENGINES")
