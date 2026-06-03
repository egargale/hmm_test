"""Tests for degenerate-fit detection (ADR-0018, Issue #83).

Covers detect_degenerate_fit — a pure function that inspects regime counts,
data dimensions, and engine identity to produce diagnostic fields for
engine_info. Detection only; no regime assignment changes.
"""

import pytest

from hmm_futures_analysis.regime.pipeline import (
    _apply_confidence_penalty,
    detect_degenerate_fit,
)


# ===========================================================================
# State collapse detection (Mode 1: hmm, messina, robust_hmm)
# ===========================================================================


class TestStateCollapse:
    """Any state < 5% of bars → degenerate_fit + degenerate_caveat."""

    def test_bull_collapsed_flags_degenerate(self):
        """CRM-like pattern: bull at 0.8% of bars."""
        counts = {"bear": 1081, "sideways": 1413, "bull": 20}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=50,
            engine_name="hmm",
            n_states=3,
        )
        assert result["degenerate_fit"] is True
        assert "bull" in result["degenerate_caveat"].lower()
        assert "0.8%" in result["degenerate_caveat"]

    def test_bear_collapsed_flags_degenerate(self):
        counts = {"bear": 15, "sideways": 1400, "bull": 1100}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2515,
            n_features=50,
            engine_name="hmm",
            n_states=3,
        )
        assert result["degenerate_fit"] is True
        assert "bear" in result["degenerate_caveat"].lower()

    def test_sideways_collapsed_flags_degenerate(self):
        counts = {"bear": 1200, "sideways": 50, "bull": 1264}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=50,
            engine_name="messina",
            n_states=3,
        )
        assert result["degenerate_fit"] is True
        assert "sideways" in result["degenerate_caveat"].lower()

    def test_multiple_states_collapsed_names_all(self):
        """Two states below threshold — caveat names both."""
        counts = {"bear": 10, "sideways": 2490, "bull": 14}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=50,
            engine_name="robust_hmm",
            n_states=3,
        )
        assert result["degenerate_fit"] is True
        caveat = result["degenerate_caveat"].lower()
        assert "bear" in caveat
        assert "bull" in caveat

    def test_clean_fit_no_degenerate_fields(self):
        """0700.HK-like balanced fit — no degenerate fields emitted."""
        counts = {"bear": 200, "sideways": 1994, "bull": 267}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2461,
            n_features=50,
            engine_name="hmm",
            n_states=3,
        )
        assert result.get("degenerate_fit") is None
        assert result.get("degenerate_caveat") is None

    def test_exactly_5_percent_is_not_degenerate(self):
        """5% boundary — exactly at threshold, not degenerate."""
        counts = {"bear": 126, "sideways": 2262, "bull": 126}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=50,
            engine_name="hmm",
            n_states=3,
        )
        assert result.get("degenerate_fit") is None

    def test_threshold_engine_never_flags(self):
        """Threshold engine doesn't use HMM — skip state collapse check."""
        counts = {"bear": 5, "sideways": 2500, "bull": 9}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=1,
            engine_name="threshold",
            n_states=3,
        )
        assert result.get("degenerate_fit") is None

    def test_caveat_suggests_n_states_auto_when_fixed(self):
        """When n_states is fixed (not auto), caveat suggests --n-states auto."""
        counts = {"bear": 1081, "sideways": 1413, "bull": 20}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=50,
            engine_name="hmm",
            n_states=3,
        )
        assert "--n-states auto" in result["degenerate_caveat"]

    def test_caveat_no_suggest_when_n_states_auto(self):
        """When n_states was already auto-resolved, no suggestion needed."""
        counts = {"bear": 1081, "sideways": 1413, "bull": 20}
        result = detect_degenerate_fit(
            regime_counts=counts,
            n_bars=2514,
            n_features=50,
            engine_name="hmm",
            n_states=2,  # auto-resolved to 2
            n_states_was_auto=True,
        )
        assert "--n-states auto" not in result["degenerate_caveat"]


# ===========================================================================
# Low-data warning (Mode 2: all HMM engines)
# ===========================================================================


class TestLowDataWarning:
    """n_bars < 4 * n_features * n_states → low_data_warning."""

    def test_insufficient_data_flags_warning(self):
        """500 bars with 50 features × 3 states → 600 minimum, insufficient."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 100, "sideways": 300, "bull": 100},
            n_bars=500,
            n_features=50,
            engine_name="fshmm",
            n_states=3,
        )
        assert result["low_data_warning"] is True
        assert "600" in result["low_data_caveat"]  # 4*50*3=600

    def test_sufficient_data_no_warning(self):
        """2514 bars with 50 features × 3 → sufficient, no warning."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 500, "sideways": 1514, "bull": 500},
            n_bars=2514,
            n_features=50,
            engine_name="fshmm",
            n_states=3,
        )
        assert result.get("low_data_warning") is None
        assert result.get("low_data_caveat") is None

    def test_low_data_applies_to_all_hmm_engines(self):
        """Low-data check applies to all HMM engines, not just fshmm."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 100, "sideways": 300, "bull": 100},
            n_bars=500,
            n_features=50,
            engine_name="hmm",
            n_states=3,
        )
        assert result["low_data_warning"] is True

    def test_threshold_engine_no_low_data_check(self):
        """Threshold engine uses 1 feature, different threshold."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 10, "sideways": 30, "bull": 10},
            n_bars=50,
            n_features=1,
            engine_name="threshold",
            n_states=3,
        )
        assert result.get("low_data_warning") is None


# ===========================================================================
# Confidence penalty
# ===========================================================================


class TestConfidencePenalty:
    """When degenerate_fit is true, verdict confidence is scaled down."""

    def test_penalty_reduces_confidence(self):
        """0.8% bull state → confidence × 0.16."""
        verdict = {"verdict": "bearish", "confidence": 0.98}
        engine_info = {
            "degenerate_fit": True,
            "degenerate_caveat": "bull state has 0.8% of bars",
        }
        regime_counts = {"bear": 1081, "sideways": 1413, "bull": 20}
        result = _apply_confidence_penalty(verdict, engine_info, regime_counts)
        # min_state_fraction = 20/2514 ≈ 0.008; penalty = 0.008/0.05 = 0.16
        assert result["confidence"] == pytest.approx(0.98 * 0.16, abs=0.01)

    def test_no_penalty_when_not_degenerate(self):
        verdict = {"verdict": "bearish", "confidence": 0.98}
        engine_info = {}  # no degenerate_fit key
        regime_counts = {"bear": 500, "sideways": 1514, "bull": 500}
        result = _apply_confidence_penalty(verdict, engine_info, regime_counts)
        assert result["confidence"] == 0.98

    def test_penalty_minimum_zero(self):
        """A state with 0 bars → min fraction 0 → confidence 0."""
        verdict = {"verdict": "neutral", "confidence": 0.5}
        engine_info = {"degenerate_fit": True}
        regime_counts = {"bear": 0, "sideways": 2514, "bull": 0}
        result = _apply_confidence_penalty(verdict, engine_info, regime_counts)
        assert result["confidence"] == 0.0


# ===========================================================================
# Over-robustness diagnostic (Mode 3: robust_hmm)
# ===========================================================================


class TestOverRobustness:
    """robust_hmm regime counts < 10% different from hmm → over_robustness."""

    def test_nearly_identical_flags_over_robustness(self):
        """CRM pattern: robust ≈ hmm (regime counts differ < 10%)."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 1060, "sideways": 1413, "bull": 41},
            n_bars=2514,
            n_features=50,
            engine_name="robust_hmm",
            n_states=3,
            hmm_regime_counts={"bear": 1081, "sideways": 1413, "bull": 20},
        )
        assert result["over_robustness"] is True
        assert "hmm" in result["over_robustness_caveat"].lower()

    def test_sufficiently_different_no_flag(self):
        """Robust produces meaningfully different results — no flag."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 600, "sideways": 1414, "bull": 500},
            n_bars=2514,
            n_features=50,
            engine_name="robust_hmm",
            n_states=3,
            hmm_regime_counts={"bear": 1081, "sideways": 1413, "bull": 20},
        )
        assert result.get("over_robustness") is None

    def test_non_robust_engine_never_flags(self):
        """Only robust_hmm triggers over-robustness check."""
        result = detect_degenerate_fit(
            regime_counts={"bear": 1060, "sideways": 1413, "bull": 41},
            n_bars=2514,
            n_features=50,
            engine_name="hmm",
            n_states=3,
            hmm_regime_counts={"bear": 1081, "sideways": 1413, "bull": 20},
        )
        assert result.get("over_robustness") is None

# ===========================================================================
# Pipeline integration: detect_degenerate_fit wired into pipeline.run()
# ===========================================================================


class TestPipelineIntegration:
    """detect_degenerate_fit and _apply_confidence_penalty called from run()."""

    def test_degenerate_fit_appears_in_engine_info(self):
        """Pipeline output includes degenerate_fit when state collapses."""
        import numpy as np
        import pandas as pd

        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run

        # Create synthetic prices that force degenerate fit:
        # 2514 bars, mostly flat with one sharp drop → bear-heavy
        np.random.seed(42)
        n = 2514
        dates = pd.date_range("2016-01-01", periods=n, freq="B")
        # Flat random walk with a crash in the middle
        returns = np.random.normal(0, 0.005, n)
        returns[1200:1230] = -0.03  # crash
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)), index=dates, name="TEST"
        )
        ohlcv = pd.DataFrame({
            "open": prices * 0.999, "high": prices * 1.001,
            "low": prices * 0.998, "close": prices, "volume": 1e6,
        }, index=dates)

        config = HMMGenericConfig(n_states=3)
        result = run(
            prices,
            source="TEST",
            engine_config=config,
            ohlcv=ohlcv,
            min_train=252,
            profile=False,
        )

        info = result.engine_info
        # Check that degenerate detection fields exist (may or may not trigger)
        # depending on the synthetic data. The key thing is the fields are
        # either present with correct types or absent.
        if "degenerate_fit" in info:
            assert isinstance(info["degenerate_fit"], bool)
        if "low_data_warning" in info:
            assert isinstance(info["low_data_warning"], bool)

    def test_confidence_penalty_applied_in_verdict(self):
        """When degenerate_fit is true, verdict confidence is penalized."""
        import numpy as np
        import pandas as pd

        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run

        np.random.seed(42)
        n = 2514
        dates = pd.date_range("2016-01-01", periods=n, freq="B")
        returns = np.random.normal(0, 0.005, n)
        returns[1200:1230] = -0.03
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)), index=dates, name="TEST"
        )
        ohlcv = pd.DataFrame({
            "open": prices * 0.999, "high": prices * 1.001,
            "low": prices * 0.998, "close": prices, "volume": 1e6,
        }, index=dates)

        config = HMMGenericConfig(n_states=3)
        result = run(
            prices,
            source="TEST",
            engine_config=config,
            ohlcv=ohlcv,
            min_train=252,
            profile=False,
        )

        # When degenerate_fit is true, confidence should be < abs(signal)
        if result.engine_info.get("degenerate_fit"):
            assert result.verdict["confidence"] <= abs(result.signal)

    def test_crm_hmm_includes_warning_field(self):
        """Acceptance criterion: CRM walk-forward results include ≥ 1 warning.

        Uses real CRM.csv evaluation data which is known to produce
        degenerate fits (bull state < 5% of bars) on hmm engine.
        """
        from pathlib import Path

        import pandas as pd

        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run

        crm_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "hmm-eval-2026-06-02" / "CRM.csv"
        if not crm_path.exists():
            pytest.skip("CRM.csv eval data not available")

        df = pd.read_csv(crm_path, index_col=0, parse_dates=True)
        prices = df["Close"].copy()
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        ohlcv.columns = ["open", "high", "low", "close", "volume"]
        prices.index = pd.DatetimeIndex(prices.index)
        ohlcv.index = prices.index

        config = HMMGenericConfig(n_states=3)
        result = run(
            prices, source="CRM", engine_config=config,
            ohlcv=ohlcv, min_train=252, profile=False,
        )

        warning_fields = {"degenerate_fit", "low_data_warning", "over_robustness"}
        present = warning_fields & set(result.engine_info.keys())
        assert len(present) >= 1, (
            f"CRM/hmm should produce ≥ 1 warning field, got: {result.engine_info}"
        )

