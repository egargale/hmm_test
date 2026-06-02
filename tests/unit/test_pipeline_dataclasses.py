"""Unit tests for pipeline internal dataclasses."""

import numpy as np

from hmm_futures_analysis.regime.pipeline import ClassifyOutput, MarkovStats


class TestClassifyOutput:
    """ClassifyOutput carries intermediate state from the classify phase."""

    def test_default_construction(self):
        out = ClassifyOutput(regimes=np.array([0, 1, 2]))
        np.testing.assert_array_equal(out.regimes, [0, 1, 2])
        assert out.posteriors is None
        assert out.last_regime == 1
        assert out.warmup_bars is None

    def test_full_construction(self):
        posteriors = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        out = ClassifyOutput(
            regimes=np.array([0, 1]),
            posteriors=posteriors,
            last_regime=1,
            warmup_bars=252,
        )
        assert out.last_regime == 1
        assert out.warmup_bars == 252
        np.testing.assert_array_equal(out.posteriors, posteriors)


class TestMarkovStats:
    """MarkovStats carries computed Markov chain statistics."""

    def test_construction_with_all_fields(self):
        transmat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
        stats = MarkovStats(
            transmat=transmat,
            stationary=np.array([0.3, 0.4, 0.3]),
            persistence={"bear": 0.7, "sideways": 0.8, "bull": 0.5},
            signal=0.42,
            current_regime=2,
            current_probs=np.array([0.2, 0.3, 0.5]),
            regime_counts={"bear": 10, "sideways": 20, "bull": 15},
            dates={"start": "2024-01-01", "end": "2024-12-31"},
        )
        np.testing.assert_array_equal(stats.transmat, transmat)
        assert stats.signal == 0.42
        assert stats.current_regime == 2
        assert stats.regime_counts["sideways"] == 20
        assert stats.dates["start"] == "2024-01-01"
