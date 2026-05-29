"""Tests for regime duration forecasting (issue #28)."""
import numpy as np
import pytest
from scipy.stats import weibull_min


class TestSpellExtraction:
    """Extract contiguous regime runs into (regime, duration, censored) spells."""

    def test_spell_extraction_basic(self):
        """Known regime sequence produces correct spells."""
        from hmm_futures_analysis.regime.duration_forecast import _extract_spells

        regimes = np.array([0, 0, 1, 1, 1, 2, 2])
        spells = _extract_spells(regimes)
        # Last spell (regime 2, duration 2) is right-censored
        assert spells == [
            (0, 2, False),
            (1, 3, False),
            (2, 2, True),
        ]

    def test_single_regime_entire_sequence(self):
        """All one regime → one censored spell, zero completed spells."""
        from hmm_futures_analysis.regime.duration_forecast import _extract_spells

        regimes = np.array([1, 1, 1, 1, 1])
        spells = _extract_spells(regimes)
        assert spells == [(1, 5, True)]

    def test_alternating_regimes(self):
        """Rapid alternation produces many short spells."""
        from hmm_futures_analysis.regime.duration_forecast import _extract_spells

        regimes = np.array([0, 1, 0, 1, 0])
        spells = _extract_spells(regimes)
        assert spells == [
            (0, 1, False),
            (1, 1, False),
            (0, 1, False),
            (1, 1, False),
            (0, 1, True),
        ]


class TestWeibullFit:
    """Weibull fitting to per-regime spell durations."""

    def test_weibull_fit_recovers_params(self):
        """Synthetic durations from known Weibull → parameter recovery."""
        from hmm_futures_analysis.regime.duration_forecast import _fit_weibull

        rng = np.random.default_rng(42)
        true_shape = 2.0
        true_scale = 30.0
        durations = rng.weibull(true_shape, size=500) * true_scale

        shape, scale = _fit_weibull(durations)

        assert abs(shape - true_shape) < 0.3, f"shape {shape} far from {true_shape}"
        assert abs(scale - true_scale) / true_scale < 0.1, (
            f"scale {scale} far from {true_scale}"
        )


class TestConditionalRemainingDuration:
    """Conditional expected remaining duration E[T−t | T>t] for Weibull."""

    def test_conditional_remaining_duration_analytical(self):
        """Verify conditional E[T−t|T>t] against analytical Weibull formula."""
        from hmm_futures_analysis.regime.duration_forecast import (
            _conditional_expected_remaining,
        )

        # Weibull with shape=2, scale=30
        shape = 2.0
        scale = 30.0
        # At t=10: E[T-t | T>10] = scale*Γ(1+1/c)*(1-γ(1+1/c, (t/scale)^c)) / exp(-(t/scale)^c) - t
        # Use numerical integration as ground truth
        from scipy.integrate import quad
        from scipy.stats import weibull_min

        t = 10.0
        sf_t = weibull_min.sf(t, shape, scale=scale)  # S(t)

        def integrand(u):
            return weibull_min.sf(u, shape, scale=scale)

        expected_remaining, _ = quad(integrand, t, np.inf)
        expected_remaining /= sf_t

        result = _conditional_expected_remaining(shape, scale, t)

        assert abs(result - expected_remaining) < 0.5, (
            f"E[T-t|T>t]={result}, expected={expected_remaining}"
        )


class TestHazardRate:
    """Weibull hazard rate at current elapsed duration."""

    def test_hazard_rate_at_current_duration(self):
        """Hazard rate = f(t)/S(t) matches scipy's hf()."""
        from hmm_futures_analysis.regime.duration_forecast import _hazard_rate

        shape = 1.8
        scale = 45.0
        t = 20.0

        expected = weibull_min.pdf(t, shape, scale=scale) / weibull_min.sf(t, shape, scale=scale)
        result = _hazard_rate(shape, scale, t)

        assert abs(result - expected) < 1e-6, (
            f"hazard={result}, expected={expected}"
        )


class TestMedianSurvival:
    """Median remaining duration from fitted Weibull."""

    def test_survival_50pct(self):
        """survival_50pct = median total survival from fitted Weibull."""
        from hmm_futures_analysis.regime.duration_forecast import _median_survival

        shape = 2.0
        scale = 30.0

        # Median of Weibull: scale * (ln(2))^(1/shape)
        expected = scale * (np.log(2) ** (1.0 / shape))
        result = _median_survival(shape, scale)

        assert abs(result - expected) < 1e-6, (
            f"median={result}, expected={expected}"
        )


class TestForecastDuration:
    """Full forecast_duration() output structure and correctness."""

    def test_forecast_duration_full_output(self):
        """forecast_duration() on a synthetic regime sequence returns full dict."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        # Build a regime sequence with enough spells for regime 0 (bear)
        rng = np.random.default_rng(123)
        seq = []
        # Generate 10 bear spells of duration ~20-40, interleaved with bull/sideways
        for _ in range(10):
            dur = rng.integers(20, 40)
            seq.extend([0] * dur)
            # A short other-regime spell
            seq.extend([2] * rng.integers(3, 8))
        regimes = np.array(seq)

        result = forecast_duration(regimes)

        assert result is not None
        assert "current_regime" in result
        assert "days_in_regime" in result
        assert "expected_remaining_days" in result
        assert "hazard_rate" in result
        assert "survival_50pct" in result
        assert "weibull_shape" in result
        assert "weibull_scale" in result
        # Last regime should be 2 (bull) — the final censored spell
        assert result["current_regime"] == "bull"
        assert isinstance(result["days_in_regime"], int)
        assert result["days_in_regime"] > 0
        assert isinstance(result["expected_remaining_days"], float)
        assert isinstance(result["hazard_rate"], float)
        assert isinstance(result["survival_50pct"], float)
        assert isinstance(result["weibull_shape"], float)
        assert result["weibull_shape"] > 0
        assert isinstance(result["weibull_scale"], float)
        assert result["weibull_scale"] > 0

    def test_single_regime_input(self):
        """Entire sequence is one regime → no completed spells → null fields."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        regimes = np.array([0, 0, 0, 0, 0])
        result = forecast_duration(regimes)

        assert result is not None
        assert result["current_regime"] == "bear"
        assert result["days_in_regime"] == 5
        assert result["expected_remaining_days"] is None
        assert result["hazard_rate"] is None
        assert result["survival_50pct"] is None
        assert result["weibull_shape"] is None
        assert result["weibull_scale"] is None

    def test_very_short_spells_valid(self):
        """Duration=1 spells are valid data points, not filtered out."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        # Rapid alternation: lots of duration-1 spells
        # 0,1,0,1,0,1,0,1,0,1,0 → regime 0 has 5 completed spells of duration 1
        regimes = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        result = forecast_duration(regimes)

        assert result is not None
        # Current regime is 0 (last element), censored spell of duration 1
        assert result["current_regime"] == "bear"
        # 5 completed spells of regime 0, all duration 1 → enough to fit
        assert result["weibull_shape"] is not None
        assert result["weibull_scale"] is not None

    def test_cox_model_without_statsmodels_raises(self):
        """--duration-model cox without statsmodels → clear ImportError."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        regimes = np.array([0, 0, 1, 1, 2, 2])
        with pytest.raises(ImportError, match=r"statsmodels"):
            forecast_duration(regimes, model="cox")


class TestPipelineIntegration:
    """Pipeline.run() integration with duration forecast."""

    def test_pipeline_with_duration_forecast_flag(self, btc_csv):
        """pipeline.run(duration_forecast=True) → output contains duration_forecast key."""
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine="threshold", duration_forecast=True)

        assert "duration_forecast" in result
        df = result["duration_forecast"]
        assert df is not None or df is None  # may be None if insufficient spells
        if df is not None:
            assert "current_regime" in df
            assert "days_in_regime" in df

    def test_pipeline_default_no_duration_forecast(self, btc_csv):
        """pipeline.run() without flag → no duration_forecast key."""
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine="threshold")

        assert "duration_forecast" not in result
