"""Unit tests for pipeline._classify_hmm helper."""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engine_protocol import HMMGenericConfig
from hmm_futures_analysis.regime.pipeline import ClassifyOutput, _classify_hmm


@pytest.fixture(scope="module")
def ohlcv_small():
    """Small synthetic OHLCV dataset for fast HMM fitting."""
    np.random.seed(42)
    n = 350
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "open": close + np.random.normal(0, 0.3, n),
            "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
            "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


class TestClassifyHmm:
    """_classify_hmm runs the walk-forward classify loop."""

    def test_returns_classify_output(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        result = _classify_hmm(eng, precomputed, returns, min_train=300)
        assert isinstance(result, ClassifyOutput)

    def test_regimes_same_length_as_returns(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)
        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        result = _classify_hmm(eng, precomputed, returns, min_train=300)
        assert len(result.regimes) == len(returns)

    def test_regimes_valid_state_range(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)
        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        result = _classify_hmm(eng, precomputed, returns, min_train=300)
        assert set(np.unique(result.regimes)).issubset({0, 1, 2})

    def test_posteriors_shape(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)
        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        result = _classify_hmm(eng, precomputed, returns, min_train=300)
        assert result.posteriors is not None
        assert result.posteriors.shape == (len(returns), 3)

    def test_warmup_bars_set(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)
        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        result = _classify_hmm(eng, precomputed, returns, min_train=300)
        assert result.warmup_bars == 300

    def test_engine_instance_returned(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)
        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        result = _classify_hmm(eng, precomputed, returns, min_train=300)
        assert result.engine_instance is eng

    def test_profiling_populates_phases(self, ohlcv_small):
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        returns = prices.pct_change(fill_method=None).dropna()
        config = HMMGenericConfig()
        eng = resolve_engine(config)
        eng_temp = HMMGenericEngine(n_states=3)
        precomputed = eng_temp.precompute(ohlcv_small)

        phases: dict[str, float] = {}
        classify_times: list[float] = []
        result = _classify_hmm(
            eng,
            precomputed,
            returns,
            min_train=300,
            profile=True,
            _phases=phases,
            _classify_times=classify_times,
        )
        assert "walk_forward_classify" in phases
        assert len(classify_times) > 0
