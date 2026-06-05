"""Root cause characterization test for issue #93.

Threshold engine produces Sideways on HDB despite unambiguous bear conditions.
This is a known design limitation of the fixed ±5% threshold on moderate-vol
stocks (daily std ~1.7%). The test documents the behavior, not a bug to fix.

Root cause:
  - HDB daily std = 1.69%, 20-day rolling return std = 7.56%
  - ±5% threshold = ±0.66σ → ~51% of bars land in Sideways
  - Last 20-day return (-0.0494) is 0.06% inside the Sideways zone
  - Compare: high-vol asset (4% daily std) → ±5% = ±0.28σ → only 22% Sideways

Fix assessment:
  - window=40 detects the bear (-0.0720 < -0.05) but adds lag
  - threshold=0.04 detects the bear (-0.0494 < -0.04) but is arbitrary
  - Proper fix: adaptive threshold = k × daily_std × √window (requires engine change)
"""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine
from hmm_futures_analysis.regime.markov_chain import classify_regimes


@pytest.fixture(scope="module")
def hdb_data():
    """Load HDB test data."""
    df = pd.read_csv("test_data/HDB_clean.csv")
    close = df["close"]
    returns = close.pct_change().dropna()
    return close, returns


class TestThresholdHDBRootCause:
    """Characterization tests: threshold engine on HDB moderate-vol stock.

    These tests document the known limitation, not a regression to fix.
    They serve as guard rails for any future threshold engine changes.
    """

    def test_hdb_daily_std_below_2pct(self, hdb_data):
        """HDB daily std (1.69%) is below the 2% threshold engine suitability line."""
        _, returns = hdb_data
        assert returns.std() < 0.02, "HDB daily std should be < 2%"

    def test_default_threshold_produces_sideways(self, hdb_data):
        """Default params (window=20, threshold=0.05) classify last bar as Sideways."""
        _, returns = hdb_data
        engine = ThresholdEngine(window=20, threshold=0.05)
        output = engine.run_classify(
            prices=returns,
            ohlcv=None,
            returns=returns,
            min_train=20,
        )
        # Last regime is Sideways (1) — the known limitation
        assert output.last_regime == 1, (
            "Default threshold should produce Sideways on HDB (known limitation)"
        )

    def test_last_rolling_return_marginally_sideways(self, hdb_data):
        """Last 20-day rolling return (-0.0494) is 0.0006 inside the Sideways zone."""
        _, returns = hdb_data
        rolling_ret = returns.rolling(window=20).sum()
        last_ret = rolling_ret.dropna().iloc[-1]
        # The return is just barely inside [-0.05, +0.05]
        assert -0.05 < last_ret < 0.05, (
            f"Last rolling return ({last_ret:.4f}) should be inside [-0.05, +0.05]"
        )
        # And it's very close to the bear boundary
        assert abs(last_ret - (-0.05)) < 0.01, (
            f"Last rolling return ({last_ret:.4f}) should be close to bear boundary"
        )

    def test_sideways_dominant_with_default_params(self, hdb_data):
        """Default params produce > 50% Sideways bars on HDB."""
        _, returns = hdb_data
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        sideways_frac = (regimes == 1).sum() / len(regimes)
        assert sideways_frac > 0.50, (
            f"Sideways fraction ({sideways_frac:.1%}) should exceed 50% on HDB"
        )

    def test_threshold_0p04_detects_bear(self, hdb_data):
        """Lower threshold (0.04) correctly classifies last bar as Bear."""
        _, returns = hdb_data
        engine = ThresholdEngine(window=20, threshold=0.04)
        output = engine.run_classify(
            prices=returns,
            ohlcv=None,
            returns=returns,
            min_train=20,
        )
        assert output.last_regime == 0, (
            "threshold=0.04 should detect Bear on HDB"
        )

    def test_window_40_detects_bear(self, hdb_data):
        """Longer window (40) correctly classifies last bar as Bear."""
        _, returns = hdb_data
        engine = ThresholdEngine(window=40, threshold=0.05)
        output = engine.run_classify(
            prices=returns,
            ohlcv=None,
            returns=returns,
            min_train=40,
        )
        assert output.last_regime == 0, (
            "window=40 should detect Bear on HDB"
        )

    def test_five_percent_threshold_equals_0p66_sigma(self, hdb_data):
        """Root cause: ±5% = ±0.66σ for HDB's 20-day rolling return distribution.

        At 0.66σ, ~51% of bars fall in the Sideways zone by design.
        This is why the threshold engine is unsuitable for daily std < 2%.
        """
        _, returns = hdb_data
        daily_std = returns.std()
        rolling_std = daily_std * np.sqrt(20)
        z_score = 0.05 / rolling_std
        # ±0.66σ — halfway between 0 and 1σ
        assert 0.60 < z_score < 0.70, (
            f"z-score ({z_score:.2f}) should be ~0.66 for HDB with default params"
        )
