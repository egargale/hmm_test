"""Tests for walk-forward mid-stream degeneration detection and recovery.

Issue #98: When state-remapping instability inside the walk-forward loop
produces a degenerate cumulative regime distribution (any regime < 5% of
bars classified so far), the engine auto-downgrades to n_states=2 for
remaining refits.

This extends the pre-check auto-downgrade from Issue #91 (ADR-0018)
into the walk-forward loop itself.

Test structure:
- Unit tests with stub engines (fast): verify mid-stream detection logic
- CSV integration tests (slow): verify recovery on real data and
  non-regression on healthy tickers
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
from hmm_futures_analysis.regime.engine_protocol import ClassifyResult
from hmm_futures_analysis.regime.engines._hmm_pipeline import (
    _hmm_classify_pipeline,
)
from hmm_futures_analysis.regime.pipeline import run as pipeline_run

ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Load CSV → (prices, ohlcv). Handles both Title and lowercase cols."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    prices = df["close"].copy()
    prices.index = pd.DatetimeIndex(prices.index)
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
    ohlcv.index = prices.index
    return prices, ohlcv


@pytest.fixture(scope="module")
def hdb_data():
    """HDB (HDFC Bank ADR) — pre-check catches degenerate full-data fit."""
    p = ROOT / "test_data" / "HDB_clean.csv"
    if not p.exists():
        pytest.skip("HDB_clean.csv not available")
    return _load_csv(p)


@pytest.fixture(scope="module")
def hk_data():
    """0700.HK (Tencent) — degenerate full-data fit, pre-check fires."""
    p = ROOT / "test_data" / "0700_HK.csv"
    if not p.exists():
        pytest.skip("0700_HK.csv not available")
    return _load_csv(p)


@pytest.fixture(scope="module")
def spy_data():
    """SPY — degenerate full-data fit, pre-check fires."""
    p = ROOT / "test_data" / "SPY.csv"
    if not p.exists():
        pytest.skip("SPY.csv not available")
    return _load_csv(p)


# ===================================================================
# Unit tests with stub engines (fast — no real HMM fitting)
# ===================================================================


class TestMidStreamDegenerationUnit:
    """Fast unit tests using stub engines to verify mid-stream detection."""

    @staticmethod
    def _make_data(n: int = 300, seed: int = 42):
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2020-01-01", periods=n)
        returns = pd.Series(rng.normal(0.0002, 0.01, n), index=dates)
        features = pd.DataFrame(
            rng.randn(n, 5),
            index=dates,
            columns=["f0", "f1", "f2", "f3", "f4"],
        )
        prices = pd.Series(
            np.exp(np.cumsum(returns.values)), index=returns.index, name="close"
        )
        return prices, returns, features

    def test_degenerate_cumulative_triggers_downgrade(self):
        """Stub engine that collapses to all-sideways → n_states drops to 2."""
        prices, returns, features = self._make_data()
        min_train = 50

        class _DegeneratingEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0
            n_states_at_call: list[int] = []

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                self.n_states_at_call.append(self.n_states)
                # First 3 calls: balanced rotation
                # After that: all sideways → bear state collapses
                if self.call_count <= 3:
                    regime = (self.call_count - 1) % 3
                else:
                    regime = 1
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _DegeneratingEngine()
        result = _hmm_classify_pipeline(
            engine, prices, None, returns, min_train=min_train, profile=False
        )

        # Engine n_states is never mutated after __init__ (Issue F fix)
        assert engine.n_states == 3, (
            "Engine n_states should never be mutated by the pipeline"
        )
        # Recovery is recorded in engine_info
        assert result.engine_info is not None
        assert result.engine_info.get("walk_forward_degenerate_recovery") is True

    def test_healthy_cumulative_no_downgrade(self):
        """Stub engine with balanced regimes → n_states stays 3."""
        prices, returns, features = self._make_data()
        min_train = 50

        class _HealthyEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                regime = self.call_count % 3
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _HealthyEngine()
        result = _hmm_classify_pipeline(
            engine, prices, None, returns, min_train=min_train, profile=False
        )

        assert result.n_states == 3
        assert result.engine_info is None or not result.engine_info.get(
            "walk_forward_degenerate_recovery", False
        )

    def test_stderr_contains_mid_stream_warning(self):
        """Stderr contains '[walk-forward] mid-stream degeneration detected'."""
        prices, returns, features = self._make_data()
        min_train = 50

        class _DegeneratingEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                if self.call_count <= 3:
                    regime = (self.call_count - 1) % 3
                else:
                    regime = 1
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _DegeneratingEngine()
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            _hmm_classify_pipeline(
                engine, prices, None, returns, min_train=min_train, profile=False
            )
        finally:
            sys.stderr = old_stderr

        assert "[walk-forward] mid-stream degeneration detected" in captured.getvalue()

    def test_no_stderr_mid_stream_warning_on_healthy(self):
        """No mid-stream warning on healthy walk-forward."""
        prices, returns, features = self._make_data()
        min_train = 50

        class _HealthyEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                regime = self.call_count % 3
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _HealthyEngine()
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            _hmm_classify_pipeline(
                engine, prices, None, returns, min_train=min_train, profile=False
            )
        finally:
            sys.stderr = old_stderr

        assert "mid-stream degeneration" not in captured.getvalue()

    def test_degeneration_bar_recorded(self):
        """engine_info.degeneration_bar is the bar where recovery fired."""
        n = 300
        min_train = 50
        prices, returns, features = self._make_data(n)

        class _DegeneratingEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                if self.call_count <= 3:
                    regime = (self.call_count - 1) % 3
                else:
                    regime = 1
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _DegeneratingEngine()
        result = _hmm_classify_pipeline(
            engine, prices, None, returns, min_train=min_train, profile=False
        )

        assert result.engine_info is not None
        assert "degeneration_bar" in result.engine_info
        deg_bar = result.engine_info["degeneration_bar"]
        assert isinstance(deg_bar, int)
        assert deg_bar >= min_train
        assert deg_bar < n

    def test_precheck_fires_no_mid_stream(self):
        """When pre-check downgrades to n_states=2, mid-stream does not fire."""
        prices, returns, features = self._make_data()
        min_train = 50

        class _HealthyEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                regime = self.call_count % 3
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _HealthyEngine()
        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
            return_value=True,
        ):
            result = _hmm_classify_pipeline(
                engine, prices, None, returns, min_train=min_train, profile=False
            )

        assert result.engine_info is not None
        assert result.engine_info.get("degenerate_auto_recovered") is True
        assert not result.engine_info.get("walk_forward_degenerate_recovery", False)
        assert "degeneration_bar" not in result.engine_info

    def test_mid_stream_only_when_precheck_passes(self):
        """Mid-stream fires when pre-check passes but walk-forward degenerates."""
        prices, returns, features = self._make_data()
        min_train = 50

        class _DegeneratingEngine:
            n_states = 3
            pca_variance = None
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                if self.call_count <= 3:
                    regime = (self.call_count - 1) % 3
                else:
                    regime = 1
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _DegeneratingEngine()
        result = _hmm_classify_pipeline(
            engine, prices, None, returns, min_train=min_train, profile=False
        )

        # Pre-check should NOT fire (random features → balanced full-data fit)
        assert not result.engine_info.get("degenerate_auto_recovered", False)
        # Mid-stream SHOULD fire
        assert result.engine_info.get("walk_forward_degenerate_recovery") is True


class TestReverseDegeneration:
    """When reverse=True, degeneration degrades old bars, not recent ones."""

    @staticmethod
    def _make_data(n: int = 300, seed: int = 42):
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2020-01-01", periods=n)
        returns = pd.Series(rng.normal(0.0002, 0.01, n), index=dates)
        features = pd.DataFrame(
            rng.randn(n, 5),
            index=dates,
            columns=["f0", "f1", "f2", "f3", "f4"],
        )
        prices = pd.Series(
            np.exp(np.cumsum(returns.values)), index=returns.index, name="close"
        )
        return prices, returns, features

    def test_reverse_degeneration_degrades_old_bars(self):
        """Degeneration in reverse mode hits bars near index 0, not near end."""
        prices, returns, features = self._make_data()
        min_train = 50

        # Engine that degenerates after enough bars: first few refits return
        # bear (0), sideways (1), bull (2) rotation; later refits collapse to
        # all-sideways. In reverse mode, "later" = old bars.
        class _ReverseDegenerating:
            n_states = 3
            pca_variance = None
            reverse_classify = True
            default_refit_every = 5
            call_count = 0

            def precompute(self, data):
                return features

            def classify(self, data, prev_means=None):
                self.call_count += 1
                # First 3 refits (most recent bars): balanced rotation
                # After that: all-sideways → bear collapses → degeneration
                if self.call_count <= 3:
                    regime = self.call_count % 3
                else:
                    regime = 1
                return ClassifyResult(
                    regime=regime,
                    means=np.array([[-1.0], [0.0], [1.0]]),
                    posteriors=np.array([0.2, 0.6, 0.2]),
                )

        engine = _ReverseDegenerating()
        result = _hmm_classify_pipeline(
            engine, prices, None, returns,
            min_train=min_train, profile=False,
        )

        assert result.reverse_classify is True
        assert result.engine_info is not None
        assert result.engine_info.get("walk_forward_degenerate_recovery") is True
        # In reverse mode, first refits process recent bars (high t) with
        # diverse regimes, then degeneration fires and old bars (low t)
        # collapse to sideways.
        # Bars 280-299 include the first 3 diverse refits at t=299,294,289.
        recent_regimes = set(result.regimes[280:300])
        old_regimes = result.regimes[50:80]
        # Recent bars should have diverse regimes (bear, sideways, bull)
        assert len(recent_regimes) >= 2, (
            f"Recent bars should have diverse regimes, got {recent_regimes}"
        )
        # Old bars should be predominantly sideways (regime 1)
        sideways_frac = np.mean(old_regimes == 1)
        assert sideways_frac > 0.7, (
            f"Old bars should be mostly sideways, got {sideways_frac:.0%}"
        )


# ===================================================================
# CSV integration tests (slow — real HMM fitting on real data)
# ===================================================================


@pytest.mark.slow
class TestCSVRecoveryIntegration:
    """Verify degenerate recovery on real CSV data.

    All current CSV files trigger the pre-check (full-data 3-state fit
    is degenerate). This is correct — the pre-check from #91 works.
    The mid-stream recovery from #98 handles the subtler case where
    the full-data fit is healthy but walk-forward refits degenerate,
    which is verified by the unit tests above.
    """

    def test_hdb_hmm_precheck_recovery(self, hdb_data):
        """HDB/hmm: degenerate recovery fires (pre-check catches it)."""
        prices, ohlcv = hdb_data
        config = HMMGenericConfig(n_states=3)
        result = pipeline_run(
            prices,
            source="HDB",
            engine_config=config,
            ohlcv=ohlcv,
            min_train=252,
            profile=False,
        )

        info = result.engine_info
        assert info is not None
        # Pre-check recovery fires on HDB
        assert info.get("degenerate_auto_recovered") is True, (
            "HDB/hmm should trigger pre-check degenerate recovery"
        )
        # After recovery, regime distribution should not be degenerate
        counts = result.regime_counts
        total = sum(counts.values())
        for name, count in counts.items():
            if name == "sideways":
                continue  # sideways from warmup, not a regime
            frac = count / total
            assert frac >= 0.05 or total < 200, (
                f"{name} has {frac:.1%} of bars after recovery (should be >= 5%)"
            )

    def test_hdb_stderr_shows_degenerate_warning(self, hdb_data):
        """HDB/hmm: stderr contains '[degenerate]' warning."""
        prices, ohlcv = hdb_data
        config = HMMGenericConfig(n_states=3)

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            pipeline_run(
                prices,
                source="HDB",
                engine_config=config,
                ohlcv=ohlcv,
                min_train=252,
                profile=False,
            )
        finally:
            sys.stderr = old_stderr

        assert "[degenerate]" in captured.getvalue(), (
            "HDB/hmm should produce degenerate warning in stderr"
        )

    def test_hdb_no_degenerate_bear_after_recovery(self, hdb_data):
        """HDB/hmm: after recovery, no regime has < 5% of bars."""
        prices, ohlcv = hdb_data
        config = HMMGenericConfig(n_states=3)
        result = pipeline_run(
            prices,
            source="HDB",
            engine_config=config,
            ohlcv=ohlcv,
            min_train=252,
            profile=False,
        )

        counts = result.regime_counts
        total = sum(counts.values())
        for name, count in counts.items():
            frac = count / total
            assert frac >= 0.05, (
                f"{name} has {frac:.1%} after recovery — should be >= 5%"
            )

    def test_spy_hmm_mid_stream_recovery_fires(self, spy_data):
        """SPY/hmm: mid-stream recovery fires (full-data fit healthy,
        but walk-forward refits degenerate — exactly the #98 scenario)."""
        prices, ohlcv = spy_data
        config = HMMGenericConfig(n_states=3)
        result = pipeline_run(
            prices,
            source="SPY",
            engine_config=config,
            ohlcv=ohlcv,
            min_train=252,
            profile=False,
        )

        info = result.engine_info
        assert info is not None
        # SPY is the classic case from issue #98:
        # full-data 3-state fit is healthy (pre-check passes),
        # but expanding-window refits degenerate through state-remapping.
        assert info.get("walk_forward_degenerate_recovery") is True, (
            "SPY/hmm should trigger mid-stream degenerate recovery"
        )
        assert "degeneration_bar" in info, "SPY/hmm should record degeneration_bar"
        # After recovery, regime distribution should be healthy
        counts = result.regime_counts
        total = sum(counts.values())
        for name, count in counts.items():
            frac = count / total
            assert frac >= 0.05, (
                f"SPY {name} has {frac:.1%} after recovery — should be >= 5%"
            )

    def test_0700hk_hmm_produces_valid_output(self, hk_data):
        """0700.HK/hmm: pipeline completes with valid regime distribution."""
        prices, ohlcv = hk_data
        config = HMMGenericConfig(n_states=3)
        result = pipeline_run(
            prices,
            source="0700.HK",
            engine_config=config,
            ohlcv=ohlcv,
            min_train=252,
            profile=False,
        )

        counts = result.regime_counts
        total = sum(counts.values())
        # After any recovery, all regimes should have >= 5%
        for name, count in counts.items():
            frac = count / total
            assert frac >= 0.05, f"0700.HK {name} has {frac:.1%} — should be >= 5%"
