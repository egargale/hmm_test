"""Tests for walk_forward_backtest signature (ADR-0017).

ADR-0017: engine param removed; regimes is now required. Pre-computed regimes
always come from pipeline.run() which calls engine.run_classify().
"""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest


def _make_prices(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate price series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(0.0001, 0.01, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates, name="close")
    return prices


def _make_regimes(n: int, seed: int = 0) -> np.ndarray:
    """Generate random regime labels (0, 1, 2)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, n).astype(int)


class TestWalkForwardRequiredRegimes:
    """walk_forward_backtest requires regimes (ADR-0017)."""

    def test_regimes_arg_produces_valid_result(self):
        prices = _make_prices()
        n = len(prices) - 1  # returns length
        regimes = _make_regimes(n)
        result = walk_forward_backtest(prices, regimes=regimes, min_train=50)
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert "n_trades" in result
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "total_return" in result

    def test_regimes_with_posteriors_produces_valid_result(self):
        prices = _make_prices()
        n = len(prices) - 1
        regimes = _make_regimes(n)
        posteriors = np.random.default_rng(1).random((n, 3))
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        result = walk_forward_backtest(
            prices, regimes=regimes, posteriors=posteriors, min_train=50
        )
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["n_trades"], int)


class TestWalkForwardRejectsMissingRegimes:
    """regimes is required — engine param removed (ADR-0017)."""

    def test_missing_regimes_raises_type_error(self):
        prices = _make_prices()
        with pytest.raises(TypeError):
            walk_forward_backtest(prices, min_train=50)  # noqa: missing regimes

    def test_engine_kwarg_raises_type_error(self):
        prices = _make_prices()
        n = len(prices) - 1
        regimes = _make_regimes(n)
        with pytest.raises(TypeError):
            walk_forward_backtest(prices, regimes=regimes, engine="anything", min_train=50)


class TestWalkForwardResultTypes:
    """Result dict has correct types."""

    def test_result_types(self):
        prices = _make_prices()
        n = len(prices) - 1
        regimes = _make_regimes(n)
        result = walk_forward_backtest(prices, regimes=regimes, min_train=50)
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["max_drawdown"], float)
        assert isinstance(result["n_trades"], int)
        assert isinstance(result["total_return"], float)


class TestDeletedInternals:
    """_resolve_engine and _VALID_ENGINES no longer exist."""

    def test_resolve_engine_removed(self):
        import hmm_futures_analysis.regime.walk_forward as wf_mod
        assert not hasattr(wf_mod, "_resolve_engine")

    def test_valid_engines_removed(self):
        import hmm_futures_analysis.regime.walk_forward as wf_mod
        assert not hasattr(wf_mod, "_VALID_ENGINES")

    def test_walk_forward_does_not_import_regime_engine(self):
        """walk_forward.py no longer imports RegimeEngine (ADR-0017)."""
        import hmm_futures_analysis.regime.walk_forward as wf_mod
        import inspect
        source = inspect.getsource(wf_mod)
        assert "RegimeEngine" not in source
