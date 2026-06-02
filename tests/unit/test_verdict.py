"""Tests for verdict computation and dynamic thresholding.

Covers _compute_verdict, _compute_dynamic_threshold, _nan_to_none,
and _probs_to_dict — pure functions in pipeline.py that produce the
consumer-facing verdict ("bullish", "bearish", "neutral",
"transition_bull", "transition_bear").
"""

import math

import numpy as np
import pytest

from hmm_futures_analysis.regime.pipeline import (
    _compute_dynamic_threshold,
    _compute_verdict,
    _nan_to_none,
    _probs_to_dict,
)


# ===========================================================================
# _probs_to_dict
# ===========================================================================


class TestProbsToDict:
    """Convert 3-element probability array to labelled dict."""

    def test_standard_probs(self):
        result = _probs_to_dict(np.array([0.2, 0.3, 0.5]))
        assert result == {"bear": 0.2, "sideways": 0.3, "bull": 0.5}

    def test_all_bear(self):
        result = _probs_to_dict(np.array([1.0, 0.0, 0.0]))
        assert result == {"bear": 1.0, "sideways": 0.0, "bull": 0.0}

    def test_output_values_are_float(self):
        result = _probs_to_dict(np.array([0, 1, 2]))
        assert isinstance(result["bear"], float)
        assert isinstance(result["sideways"], float)
        assert isinstance(result["bull"], float)


# ===========================================================================
# _nan_to_none
# ===========================================================================


class TestNanToNone:
    """NaN and Inf are converted to None; finite floats pass through."""

    def test_nan_becomes_none(self):
        assert _nan_to_none(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _nan_to_none(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _nan_to_none(float("-inf")) is None

    def test_finite_float_passes_through(self):
        assert _nan_to_none(3.14) == 3.14

    def test_zero_passes_through(self):
        assert _nan_to_none(0.0) == 0.0

    def test_negative_passes_through(self):
        assert _nan_to_none(-42.0) == -42.0


# ===========================================================================
# _compute_dynamic_threshold
# ===========================================================================


class TestDynamicThresholdFallbacks:
    """All paths that return base_threshold because duration data is absent/invalid."""

    def test_none_duration_forecast_returns_base(self):
        assert _compute_dynamic_threshold(None) == 0.1
        assert _compute_dynamic_threshold(None, base_threshold=0.2) == 0.2

    def test_missing_days_in_regime_returns_base(self):
        df = {"weibull_scale": 10.0, "weibull_shape": 2.0}
        assert _compute_dynamic_threshold(df) == 0.1

    def test_missing_scale_returns_base(self):
        df = {"days_in_regime": 15, "weibull_shape": 2.0}
        assert _compute_dynamic_threshold(df) == 0.1

    def test_missing_shape_returns_base(self):
        df = {"days_in_regime": 15, "weibull_scale": 10.0}
        assert _compute_dynamic_threshold(df) == 0.1

    def test_shape_zero_returns_base(self):
        df = {"days_in_regime": 15, "weibull_scale": 10.0, "weibull_shape": 0}
        assert _compute_dynamic_threshold(df) == 0.1

    def test_shape_negative_returns_base(self):
        df = {"days_in_regime": 15, "weibull_scale": 10.0, "weibull_shape": -1.5}
        assert _compute_dynamic_threshold(df) == 0.1

    def test_expected_total_zero_returns_base(self):
        """scale=0 → expected_total=0 → division by zero guard."""
        df = {"days_in_regime": 10, "weibull_scale": 0.0, "weibull_shape": 2.0}
        assert _compute_dynamic_threshold(df) == 0.1


class TestDynamicThresholdAgingRamp:
    """Threshold shrinks linearly as regime ages past Weibull expected duration."""

    def test_aging_ratio_at_boundary_one_returns_base(self):
        """Exactly at expected duration: threshold = base_threshold."""
        # shape=2.0, scale=10.0 → expected = 10 * Γ(1+1/2) = 10 * Γ(1.5) ≈ 8.862
        # days_in=8.862 → aging_ratio ≈ 1.0 → threshold = 0.1
        from scipy.special import gamma as _gamma
        scale = 10.0
        shape = 2.0
        expected = scale * _gamma(1.0 + 1.0 / shape)
        df = {"days_in_regime": expected, "weibull_scale": scale, "weibull_shape": shape}
        result = _compute_dynamic_threshold(df)
        assert result == 0.1  # exactly at boundary

    def test_aging_ratio_below_one_returns_base(self):
        """Young regime (< expected): threshold = base_threshold."""
        df = {"days_in_regime": 5.0, "weibull_scale": 10.0, "weibull_shape": 2.0}
        result = _compute_dynamic_threshold(df)
        assert result == 0.1

    def test_aging_ratio_mid_ramp(self):
        """aging_ratio=1.35 → threshold_mult=0.65 → threshold=0.065"""
        from scipy.special import gamma as _gamma
        scale = 10.0
        shape = 2.0
        expected = scale * _gamma(1.0 + 1.0 / shape)
        days_in = expected * 1.35
        df = {"days_in_regime": days_in, "weibull_scale": scale, "weibull_shape": shape}
        result = _compute_dynamic_threshold(df)
        assert result == pytest.approx(0.065)

    def test_aging_ratio_at_floor(self):
        """aging_ratio=1.7 → threshold_mult=0.3 → threshold=0.03"""
        from scipy.special import gamma as _gamma
        scale = 10.0
        shape = 2.0
        expected = scale * _gamma(1.0 + 1.0 / shape)
        days_in = expected * 1.7
        df = {"days_in_regime": days_in, "weibull_scale": scale, "weibull_shape": shape}
        result = _compute_dynamic_threshold(df)
        assert result == pytest.approx(0.03)

    def test_aging_ratio_beyond_floor_stays_at_floor(self):
        """aging_ratio=50 → threshold_mult still 0.3 (floor, no underflow)."""
        from scipy.special import gamma as _gamma
        scale = 10.0
        shape = 2.0
        expected = scale * _gamma(1.0 + 1.0 / shape)
        days_in = expected * 50.0
        df = {"days_in_regime": days_in, "weibull_scale": scale, "weibull_shape": shape}
        result = _compute_dynamic_threshold(df)
        assert result == pytest.approx(0.03)

    def test_custom_base_threshold_respected(self):
        """Base threshold propagates through ramp."""
        from scipy.special import gamma as _gamma
        scale = 10.0
        shape = 2.0
        expected = scale * _gamma(1.0 + 1.0 / shape)
        days_in = expected * 1.35  # threshold_mult = 0.65
        df = {"days_in_regime": days_in, "weibull_scale": scale, "weibull_shape": shape}
        result = _compute_dynamic_threshold(df, base_threshold=0.2)
        assert result == pytest.approx(0.13)  # 0.2 * 0.65

    def test_aging_ratio_near_ramp_end(self):
        """aging_ratio just below 1.7 → threshold_mult just above 0.3"""
        from scipy.special import gamma as _gamma
        scale = 10.0
        shape = 2.0
        expected = scale * _gamma(1.0 + 1.0 / shape)
        days_in = expected * 1.699  # threshold_mult = 2.0 - 1.699 = 0.301
        df = {"days_in_regime": days_in, "weibull_scale": scale, "weibull_shape": shape}
        result = _compute_dynamic_threshold(df)
        assert result < 0.031  # base*0.301 — still above floor by a hair


# ===========================================================================
# _compute_verdict
# ===========================================================================


def _fc(bear=0.0, sideways=0.0, bull=0.0):
    """Shorthand to build a forecast_20 dict."""
    return {"bear": bear, "sideways": sideways, "bull": bull}


class TestVerdictBullRegime:
    """regime=2 (Bull) → bullish if bull forecast > bear forecast, else transition_bear."""

    def test_bull_forecast_dominates_returns_bullish(self):
        result = _compute_verdict(2, 0.7, _fc(bear=0.1, sideways=0.2, bull=0.7))
        assert result["verdict"] == "bullish"
        assert result["confidence"] == 0.7

    def test_bear_forecast_exceeds_bull_returns_transition_bear(self):
        result = _compute_verdict(2, 0.1, _fc(bear=0.6, sideways=0.2, bull=0.2))
        assert result["verdict"] == "transition_bear"

    def test_bull_and_bear_tied_returns_transition_bear(self):
        """When bull forecast == bear forecast, not strictly > → transition."""
        result = _compute_verdict(2, 0.3, _fc(bear=0.4, sideways=0.2, bull=0.4))
        assert result["verdict"] == "transition_bear"


class TestVerdictBearRegime:
    """regime=0 (Bear) → bearish if bear forecast > bull forecast, else transition_bull."""

    def test_bear_forecast_dominates_returns_bearish(self):
        result = _compute_verdict(0, -0.5, _fc(bear=0.7, sideways=0.2, bull=0.1))
        assert result["verdict"] == "bearish"
        assert result["confidence"] == 0.5

    def test_bull_forecast_exceeds_bear_returns_transition_bull(self):
        result = _compute_verdict(0, -0.2, _fc(bear=0.2, sideways=0.2, bull=0.6))
        assert result["verdict"] == "transition_bull"

    def test_bear_and_bull_tied_returns_transition_bull(self):
        """When bear forecast == bull forecast, not strictly > → transition."""
        result = _compute_verdict(0, -0.3, _fc(bear=0.4, sideways=0.2, bull=0.4))
        assert result["verdict"] == "transition_bull"


class TestVerdictSidewaysRegime:
    """regime=1 (Sideways) → signal magnitude vs threshold determines verdict."""

    def test_signal_below_threshold_returns_neutral(self):
        result = _compute_verdict(1, 0.05, _fc(sideways=1.0), sideways_threshold=0.1)
        assert result["verdict"] == "neutral"

    def test_signal_above_threshold_positive_returns_transition_bull(self):
        result = _compute_verdict(1, 0.15, _fc(sideways=1.0), sideways_threshold=0.1)
        assert result["verdict"] == "transition_bull"

    def test_signal_above_threshold_negative_returns_transition_bear(self):
        result = _compute_verdict(1, -0.3, _fc(sideways=1.0), sideways_threshold=0.1)
        assert result["verdict"] == "transition_bear"

    def test_signal_exactly_at_threshold_positive_returns_transition_bull(self):
        """|signal| == threshold: not less than → transition (not neutral)."""
        result = _compute_verdict(1, 0.1, _fc(sideways=1.0), sideways_threshold=0.1)
        assert result["verdict"] == "transition_bull"

    def test_signal_exactly_at_threshold_negative_returns_transition_bear(self):
        result = _compute_verdict(1, -0.1, _fc(sideways=1.0), sideways_threshold=0.1)
        assert result["verdict"] == "transition_bear"


class TestVerdictConfidence:
    """Confidence = abs(signal), rounded to 4 decimal places."""

    def test_confidence_rounded_to_4dp(self):
        result = _compute_verdict(2, 0.123456, _fc(bull=1.0))
        assert result["confidence"] == 0.1235  # rounds up

    def test_confidence_zero_signal(self):
        result = _compute_verdict(1, 0.0, _fc(sideways=1.0), sideways_threshold=0.1)
        assert result["confidence"] == 0.0

    def test_confidence_negative_signal(self):
        result = _compute_verdict(0, -0.8, _fc(bear=0.9, bull=0.1))
        assert result["confidence"] == 0.8


class TestVerdictInvalidRegime:
    """Regime index outside 0-2 treated as Sideways (else fallthrough)."""

    def test_regime_3_treated_as_sideways_neutral(self):
        """regime=3 falls through else → Sideways path; low signal → neutral."""
        result = _compute_verdict(3, 0.05, _fc(bear=0.3, sideways=0.4, bull=0.3))
        assert result["verdict"] == "neutral"

    def test_regime_3_treated_as_sideways_transition(self):
        """regime=3 with high positive signal → transition_bull."""
        result = _compute_verdict(3, 0.5, _fc(bear=0.3, sideways=0.4, bull=0.3))
        assert result["verdict"] == "transition_bull"
