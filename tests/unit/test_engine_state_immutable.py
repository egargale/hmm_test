"""Regression test for Issue F: engine.n_states immutability.

Ensures that after pipeline.run_classify(), the engine's n_states
attribute is unchanged (never mutated after __init__).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine
from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine
from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine
from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine


HMM_ENGINE_CLASSES = [
    HMMGenericEngine,
    HMMMMessinaEngine,
    RobustHMMEngine,
    FSHMMEngine,
]


def _make_synthetic_data(n: int = 500, seed: int = 42):
    """Create minimal synthetic OHLCV data for a single-engine run."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    ohlcv = pd.DataFrame(
        {
            "open": 100 + rng.randn(n).cumsum() * 0.5,
            "high": 101 + rng.randn(n).cumsum() * 0.5,
            "low": 99 + rng.randn(n).cumsum() * 0.5,
            "close": 100 + rng.randn(n).cumsum() * 0.5,
            "volume": rng.randint(1000, 10000, n),
        },
        index=dates,
    )
    prices = ohlcv["close"]
    returns = prices.pct_change().dropna()
    return prices, ohlcv, returns


@pytest.mark.parametrize("engine_class", HMM_ENGINE_CLASSES)
def test_n_states_unchanged_after_run_with_int(engine_class):
    """Engine n_states (int) should be unchanged after run_classify."""
    prices, ohlcv, returns = _make_synthetic_data()
    engine = engine_class(n_states=3, pca_variance=None)
    original_n_states = engine.n_states

    _ = engine.run_classify(prices, ohlcv, returns, min_train=100)

    assert engine.n_states == original_n_states, (
        f"{engine_class.__name__}: n_states mutated from "
        f"{original_n_states} to {engine.n_states}"
    )


@pytest.mark.parametrize("engine_class", HMM_ENGINE_CLASSES)
def test_n_states_unchanged_after_two_runs(engine_class):
    """Engine n_states should be unchanged after two consecutive runs."""
    prices, ohlcv, returns = _make_synthetic_data()
    engine = engine_class(n_states=3)

    _ = engine.run_classify(prices, ohlcv, returns, min_train=100)
    n_after_first = engine.n_states

    _ = engine.run_classify(prices, ohlcv, returns, min_train=100)
    n_after_second = engine.n_states

    assert n_after_first == 3, (
        f"{engine_class.__name__}: n_states changed after first run: "
        f"{n_after_first}"
    )
    assert n_after_second == 3, (
        f"{engine_class.__name__}: n_states changed after second run: "
        f"{n_after_second}"
    )


@pytest.mark.parametrize("engine_class", HMM_ENGINE_CLASSES)
def test_n_states_unchanged_after_run_with_auto(engine_class):
    """Engine n_states initially 'auto' stays 'auto' after run_classify."""
    prices, ohlcv, returns = _make_synthetic_data()
    engine = engine_class(n_states="auto", pca_variance=None)

    _ = engine.run_classify(prices, ohlcv, returns, min_train=100)

    assert engine.n_states == "auto", (
        f"{engine_class.__name__}: n_states mutated from 'auto' "
        f"to {engine.n_states!r}"
    )


@pytest.mark.parametrize("engine_class", HMM_ENGINE_CLASSES)
def test_result_n_states_is_resolved_int(engine_class):
    """ClassifyOutput.n_states is the resolved int, engine.n_states untouched."""
    prices, ohlcv, returns = _make_synthetic_data()
    engine = engine_class(n_states=3)

    result = engine.run_classify(prices, ohlcv, returns, min_train=100)

    assert isinstance(result.n_states, int), (
        f"ClassifyOutput.n_states should be int, got {type(result.n_states)}"
    )
    assert engine.n_states == 3, (
        f"Engine n_states should stay 3, got {engine.n_states}"
    )
