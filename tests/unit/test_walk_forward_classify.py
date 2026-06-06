"""Unit tests for _walk_forward_classify generator."""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engines._hmm_pipeline import _walk_forward_classify

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

    def test_final_bar_always_refits(self):
        """Final bar triggers a refit even when off the regular refit_every cadence."""
        # n=62, min_train=50 → refit_every=5
        # Regular refit bars: 50, 55, 60. Final bar 61 is off-cadence.
        n = 62
        min_train = 50
        returns = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="B")
        )
        features = pd.DataFrame({"f1": np.random.randn(n)}, index=returns.index)

        refit_bars: list[int] = []

        class _TrackingEngine(_StubEngine):
            def classify(self, data, prev_means=None):
                refit_bars.append(len(data))  # len(data) == t at refit time
                return super().classify(data, prev_means)

        engine = _TrackingEngine(regime=2)
        list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                profile=False,
            )
        )

        # The final bar (t = n - 1 = 61) must trigger a refit
        assert (n - 1) in refit_bars

    def test_intermediate_bars_carry_forward_unchanged(self):
        """Non-refit intermediate bars still carry forward the previous result."""
        # n=62, min_train=50 → refit_every=5
        # Regular refit bars: 50, 55, 60. Forced refit at 61.
        # Bars 51-54 carry forward from refit at 50.
        # Bars 56-59 carry forward from refit at 55.
        # Bar 60 is a regular refit. Bar 61 is the forced final refit.
        n = 62
        min_train = 50
        returns = pd.Series(
            np.random.randn(n), index=pd.date_range("2020-01-01", periods=n, freq="B")
        )
        features = pd.DataFrame({"f1": np.random.randn(n)}, index=returns.index)

        call_count = 0

        class _RegimeCountingEngine(_StubEngine):
            def classify(self, data, prev_means=None):
                nonlocal call_count
                call_count += 1
                # Return a unique regime per call so carry-forward is detectable
                return ClassifyResult(
                    regime=call_count + 10,  # 11, 12, 13, 14...
                    means=np.array([[0.0]]),
                    posteriors=np.array([0.1, 0.8, 0.1]),
                )

        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult

        engine = _RegimeCountingEngine(regime=0)
        results = list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                profile=False,
            )
        )

        # Refit at t=50 → regime=11, carried at t=51,52,53,54
        assert results[0][1].regime == 11  # t=50 (refit)
        assert results[1][1].regime == 11  # t=51 (carry)
        assert results[2][1].regime == 11  # t=52 (carry)
        assert results[3][1].regime == 11  # t=53 (carry)
        assert results[4][1].regime == 11  # t=54 (carry)
        # Refit at t=55 → regime=12, carried at t=56,57,58,59
        assert results[5][1].regime == 12  # t=55 (refit)
        assert results[6][1].regime == 12  # t=56 (carry)
        # Refit at t=60 → regime=13
        assert results[10][1].regime == 13  # t=60 (refit)
        # Forced refit at t=61 → regime=14
        assert results[11][1].regime == 14  # t=61 (forced final refit)

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


class TestWalkForwardClassifyReverse:
    """_walk_forward_classify with reverse=True iterates backward."""

    def test_reverse_yields_correct_count_and_range(self):
        """Reverse mode yields (t, result) for t from n-1 down to min_train."""
        n = 100
        min_train = 50
        returns = pd.Series(
            np.random.randn(n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        features = pd.DataFrame({"f1": np.random.randn(n)}, index=returns.index)
        engine = _StubEngine(regime=2)

        results = list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                reverse=True,
                profile=False,
            )
        )

        assert len(results) == n - min_train
        ts = [t for t, _ in results]
        # Should iterate from n-1 down to min_train
        assert ts == list(range(n - 1, min_train - 1, -1))

    def test_reverse_uses_backward_slices(self):
        """In reverse mode, classify receives data.iloc[t:] (not iloc[:t])."""
        n = 80
        min_train = 40
        returns = pd.Series(
            np.random.randn(n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        features = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=returns.index)

        observed_lengths: list[int] = []

        class _SliceTrackingEngine(_StubEngine):
            def classify(self, data, prev_means=None):
                observed_lengths.append(len(data))
                return super().classify(data, prev_means)

        engine = _SliceTrackingEngine(regime=1)
        list(
            _walk_forward_classify(
                returns,
                eng=engine,
                precomputed=features,
                min_train=min_train,
                reverse=True,
                profile=False,
            )
        )

        # First refit at t=n-1 → slice is features.iloc[n-1:] → len=1
        # Second refit at t=n-1-refit_every → slice is features.iloc[t:] → len=n-t
        # The first refit (t=n-1) should see a very short slice (1 row)
        assert observed_lengths[0] == 1
        # Later refits should see progressively longer slices
        assert observed_lengths[-1] > observed_lengths[0]
