"""Tests for pipeline profiling — issue #37.

TDD tests for the per-phase timing breakdown of the HMM pipeline.
Each tracer bullet tests one observable behavior through the public
interface: pipeline.run() with profile=True.
"""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.pipeline import run as pipeline_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series for fast deterministic tests."""
    rng = np.random.RandomState(seed)
    returns = rng.randn(n) * 0.02
    prices = 100.0 * np.cumprod(1.0 + returns)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="close")


def _make_ohlcv(prices: pd.Series) -> pd.DataFrame:
    """Synthesize OHLCV from a close-price series."""
    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, len(prices))),
            "high": prices * (1 + np.abs(np.random.randn(len(prices)) * 0.01)),
            "low": prices * (1 - np.abs(np.random.randn(len(prices)) * 0.01)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, len(prices)).astype(float),
        },
        index=prices.index,
    )
    return df


# ---------------------------------------------------------------------------
# Tracer Bullet 1: profile=True returns timing dict (threshold engine)
# ---------------------------------------------------------------------------


class TestProfileThreshold:
    """Threshold engine with profile=True produces a timing dict."""

    def test_profile_returns_timing_dict(self):
        prices = _make_prices(400)
        result = pipeline_run(prices, source="test", engine="threshold", profile=True)

        assert "timing" in result
        timing = result["timing"]
        assert "total_wall_seconds" in timing
        assert timing["total_wall_seconds"] > 0

    def test_timing_has_no_classify_stats_for_threshold(self):
        """Threshold engine doesn't call eng.classify() in a loop."""
        prices = _make_prices(400)
        result = pipeline_run(prices, source="test", engine="threshold", profile=True)

        timing = result["timing"]
        # Threshold has no HMM classify loop, so no classify stats
        assert (
            timing.get("walk_forward_classify_stats") is None
            or timing.get("walk_forward_classify_stats", {}).get("n_calls", 0) == 0
        )

    def test_threshold_timing_has_phases_dict(self):
        """Threshold engine timing includes a phases sub-dict."""
        prices = _make_prices(400)
        result = pipeline_run(prices, source="test", engine="threshold", profile=True)

        timing = result["timing"]
        assert "phases" in timing
        assert isinstance(timing["phases"], dict)


# ---------------------------------------------------------------------------
# Tracer Bullet 2: HMM engine profile populates walk_forward_classify_stats
# ---------------------------------------------------------------------------


class TestProfileHMMClassify:
    """HMM engine with profile=True tracks per-call classify timing."""

    def test_hmm_profile_has_classify_stats(self):
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        result = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states=3,
            min_train=50,
            profile=True,
        )

        assert "timing" in result
        timing = result["timing"]

        # Must have classify stats with actual call counts
        assert "walk_forward_classify_stats" in timing, (
            f"Expected walk_forward_classify_stats in timing keys: {list(timing.keys())}"
        )
        stats = timing["walk_forward_classify_stats"]
        assert stats["n_calls"] > 0, "Expected at least one classify call"
        assert stats["min"] >= 0
        assert stats["median"] >= stats["min"]
        assert stats["p99"] >= stats["median"]

    def test_hmm_classify_stats_monotonically_ordered(self):
        """min <= median <= p99."""
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        result = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states=3,
            min_train=50,
            profile=True,
        )

        stats = result["timing"]["walk_forward_classify_stats"]
        assert stats["min"] <= stats["median"]
        assert stats["median"] <= stats["p99"]


# ---------------------------------------------------------------------------
# Tracer Bullet 3: n_states='auto' produces BIC per-state timing
# ---------------------------------------------------------------------------


class TestProfileBICSelection:
    """BIC state selection with profile=True tracks per-state-count timing."""

    def test_auto_states_has_bic_phase_timing(self):
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        result = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states="auto",
            min_train=50,
            profile=True,
        )

        assert "timing" in result
        timing = result["timing"]
        # BIC selection phase should appear in phases dict
        phases = timing.get("phases", {})
        assert "bic_select_n_states" in phases, (
            f"Expected bic_select_n_states in phases: {list(phases.keys())}"
        )
        assert phases["bic_select_n_states"] > 0

    def test_auto_states_has_bic_detail(self):
        """BIC selection includes per-state-count timing breakdown."""
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        result = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states="auto",
            min_train=50,
            profile=True,
        )

        timing = result["timing"]
        assert "bic_detail" in timing, (
            f"Expected bic_detail in timing keys: {list(timing.keys())}"
        )
        detail = timing["bic_detail"]
        assert isinstance(detail, dict)
        # Should have entries for state counts 2..max_states
        assert len(detail) >= 1
        # Each entry should have timing info
        for key, val in detail.items():
            assert isinstance(val, (int, float, dict)), (
                f"bic_detail[{key!r}] should be numeric or dict, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Tracer Bullet 4: walk_forward_backtest timing for HMM engines
# ---------------------------------------------------------------------------


class TestProfileWalkForwardBacktest:
    """Walk-forward backtest phase timing for HMM engines."""

    def test_hmm_profile_has_wf_backtest_phase(self):
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        result = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states=3,
            min_train=50,
            profile=True,
        )

        phases = result["timing"]["phases"]
        assert "walk_forward_backtest" in phases
        assert phases["walk_forward_backtest"] > 0

    def test_all_hmm_phases_present(self):
        """All major phases should appear in timing.phases dict."""
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        result = pipeline_run(
            prices,
            source="test",
            engine="hmm",
            ohlcv=ohlcv,
            n_states=3,
            min_train=50,
            profile=True,
        )

        phases = result["timing"]["phases"]
        for phase in ("precompute", "walk_forward_classify", "walk_forward_backtest"):
            assert phase in phases, f"Missing phase: {phase}"
            assert phases[phase] >= 0, f"Phase {phase} has negative time"


# ---------------------------------------------------------------------------
# Tracer Bullet 5: All engines produce valid timing structure
# ---------------------------------------------------------------------------


class TestProfileAllEngines:
    """Every engine produces a valid timing structure with profile=True."""

    @pytest.mark.parametrize("engine", ["threshold", "hmm", "messina"])
    def test_engine_timing_structure(self, engine):
        prices = _make_prices(300)
        kwargs = dict(source="test", engine=engine, profile=True, min_train=50)
        if engine != "threshold":
            kwargs["ohlcv"] = _make_ohlcv(prices)
            kwargs["n_states"] = 3

        result = pipeline_run(prices, **kwargs)

        assert "timing" in result
        timing = result["timing"]
        assert timing["total_wall_seconds"] > 0
        # All phases that ran should have non-negative times
        for phase_name, phase_time in timing.get("phases", {}).items():
            assert phase_time >= 0, f"Phase {phase_name} has negative time"
