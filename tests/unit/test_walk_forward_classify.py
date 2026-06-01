"""Unit tests for _walk_forward_classify generator."""

import numpy as np
import pandas as pd

from hmm_futures_analysis.regime.engines._hmm_pipeline import _walk_forward_classify


class _StubEngine:
    """Minimal engine stub that returns configurable classify results."""

    def __init__(self, regime: int = 1):
        self._regime = regime
        self.call_count = 0

    def classify(self, data, prev_means=None):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        self.call_count += 1
        return ClassifyResult(
            regime=self._regime,
            means=np.array([[0.0]]),
            posteriors=np.array([0.1, 0.8, 0.1]),
        )

    def precompute(self, data):
        return None


class TestWalkForwardClassify:
    """_walk_forward_classify yields (t, ClassifyResult) for every bar >= min_train."""

    def test_precomputed_regimes_yields_correct_regime_values(self):
        """Precomputed-regimes mode: yields the exact regime for each bar."""
        n = 100
        returns = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="B")
        )
        regimes = np.array([1] * 50 + [2] * 50, dtype=int)
        min_train = 50

        results = list(
            _walk_forward_classify(
                returns,
                regimes=regimes,
                min_train=min_train,
            )
        )

        assert len(results) == n - min_train
        for t, result in results:
            assert t >= min_train
            assert result.regime == regimes[t]

    def test_precomputed_features_yields_result_per_bar(self):
        """Precomputed-features mode: yields a result for every bar >= min_train."""
        n = 100
        returns = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="B")
        )
        features = pd.DataFrame({"f1": np.random.randn(n)}, index=returns.index)
        engine = _StubEngine(regime=2)
        min_train = 50

        results = list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                profile=False,
            )
        )

        assert len(results) == n - min_train
        # All results should have regime=2 (the stub value)
        for t, result in results:
            assert t >= min_train
            assert result.regime == 2

    def test_continues_on_classify_error(self):
        """Generator keeps yielding last good result when classify raises an error."""
        n = 80
        returns = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="B")
        )
        features = pd.DataFrame({"f1": np.random.randn(n)}, index=returns.index)

        call_count = 0

        def _failing_classify(data, prev_means=None):
            from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("simulated failure")
            return ClassifyResult(
                regime=2, means=np.array([[0.0]]), posteriors=np.array([0.1, 0.8, 0.1])
            )

        engine = _StubEngine(regime=2)
        engine.classify = _failing_classify
        min_train = 50

        results = list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                profile=False,
            )
        )

        assert len(results) == n - min_train
        # Every bar must yield a valid regime (no crashes, no gaps)
        assert all(0 <= r.regime <= 2 for _, r in results)

    def test_profile_collects_timing(self):
        """With profile=True, classify times are recorded in _classify_times."""
        n = 100
        returns = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="B")
        )
        features = pd.DataFrame({"f1": np.random.randn(n)}, index=returns.index)
        engine = _StubEngine(regime=2)
        min_train = 50

        phases: dict[str, float] = {}
        times: list[float] = []
        list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                profile=True,
                _phases=phases,
                _classify_times=times,
            )
        )

        # Should have recorded classify call times
        assert len(times) > 0
        # All times should be positive floats
        assert all(isinstance(t, float) and t >= 0 for t in times)
