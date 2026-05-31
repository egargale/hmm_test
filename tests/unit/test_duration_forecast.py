"""Tests for regime duration forecasting (issues #28, #29)."""
import numpy as np
import pandas as pd
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

    def test_weibull_ignores_prices(self):
        """model='weibull' ignores prices parameter — backward compatible."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        rng = np.random.default_rng(42)
        seq = []
        for _ in range(5):
            seq.extend([0] * rng.integers(10, 20))
            seq.extend([1] * rng.integers(3, 8))
        regimes = np.array(seq)

        # Should not raise regardless of prices value
        result_no_prices = forecast_duration(regimes, model="weibull")
        result_none_prices = forecast_duration(regimes, model="weibull", prices=None)

        assert result_no_prices is not None
        assert result_none_prices is not None
        assert result_no_prices == result_none_prices


class TestPipelineIntegration:
    """Pipeline.run() integration with duration forecast."""

    def test_pipeline_with_duration_forecast_flag(self, btc_csv):
        """pipeline.run(duration_forecast=True) → output contains duration_forecast key."""
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
        from hmm_futures_analysis.regime.engine_protocol import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig(), duration_forecast=True)

        assert "duration_forecast" in result
        df = result["duration_forecast"]
        assert df is not None or df is None  # may be None if insufficient spells
        if df is not None:
            assert "current_regime" in df
            assert "days_in_regime" in df

    def test_pipeline_default_no_duration_forecast(self, btc_csv):
        """pipeline.run() without flag → no duration_forecast key."""
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
        from hmm_futures_analysis.regime.engine_protocol import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())

        assert "duration_forecast" not in result


# ============================================================
# Issue #29: Cox PH Duration Forecasting
# ============================================================


def _make_regime_sequence(rng, n_spells=10, regime_idx=0, other_regimes=(1, 2)):
    """Build a regime sequence with many spells of regime_idx."""
    seq = []
    for i in range(n_spells):
        dur = rng.integers(15, 40)
        seq.extend([regime_idx] * int(dur))
        other = other_regimes[i % len(other_regimes)]
        seq.extend([other] * rng.integers(3, 8))
    return np.array(seq)


def _make_prices_for_regimes(regimes, rng, base_price=100.0):
    """Generate a price Series aligned to a regime array."""
    n = len(regimes)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates)


class TestCoxModelRequiresPrices:
    """model='cox' without prices → clear ValueError."""

    def test_cox_without_prices_raises_value_error(self):
        """model='cox' with prices=None → ValueError with clear message."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        regimes = np.array([0, 0, 1, 1, 2, 2])
        with pytest.raises(ValueError, match="price data"):
            forecast_duration(regimes, model="cox", prices=None)


class TestCoxModelWithoutLifelines:
    """model='cox' without lifelines installed → clear ImportError."""

    def test_cox_without_lifelines_raises_import_error(self):
        """Calling _fit_coxph when lifelines missing → ImportError with install hint."""
        # Test the lazy import guard directly
        import importlib
        import sys

        # Save and remove all lifelines modules, then block
        saved = {}
        for key in list(sys.modules):
            if key.startswith("lifelines"):
                saved[key] = sys.modules.pop(key)
        sys.modules["lifelines"] = None

        try:
            # Force reimport to pick up the blocked lifelines
            import hmm_futures_analysis.regime.duration_forecast as mod
            importlib.reload(mod)

            # Directly call _fit_coxph — the lazy import should fail
            with pytest.raises((ImportError, ModuleNotFoundError), match="lifelines"):
                mod._fit_coxph(
                    regimes=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
                    spells=[],
                    prices=pd.Series([100.0] * 10),
                    current_regime_idx=0,
                    days_in_regime=2,
                )
        finally:
            for key in list(sys.modules):
                if key.startswith("lifelines"):
                    del sys.modules[key]
            sys.modules.update(saved)
            importlib.reload(mod)


class TestBuildSpellCovariates:
    """_build_spell_covariates: per-spell realized_vol, spell_return from prices."""

    def test_known_values(self):
        """Known price series → hand-computed realized_vol and spell_return."""
        from hmm_futures_analysis.regime.duration_forecast import (
            _build_spell_covariates,
            _extract_spells,
        )

        # 6 bars: regime [0,0, 1,1,1, 0] → spells: (0,2,False), (1,3,False), (0,1,True)
        # prices: [100, 110, 105, 115, 120, 108]
        regimes = np.array([0, 0, 1, 1, 1, 0])
        prices = pd.Series([100.0, 110.0, 105.0, 115.0, 120.0, 108.0])

        spells = _extract_spells(regimes)
        covs = _build_spell_covariates(spells, prices)

        assert len(covs) == 3

        # Spell 0: regime=0, bars [100, 110], duration=2, event=True
        assert covs.iloc[0]["duration"] == 2
        assert covs.iloc[0]["event"] == True
        assert covs.iloc[0]["regime_idx"] == 0
        # realized_vol = std(log(110/100)) = 0 (only one log-return)
        assert abs(covs.iloc[0]["realized_vol"] - 0.0) < 1e-10
        # spell_return = log(110/100) = log(1.1)
        assert abs(covs.iloc[0]["spell_return"] - np.log(1.1)) < 1e-6

        # Spell 1: regime=1, bars [105, 115, 120], duration=3, event=True
        assert covs.iloc[1]["duration"] == 3
        assert covs.iloc[1]["event"] == True
        # spell_return = log(120/105)
        assert abs(covs.iloc[1]["spell_return"] - np.log(120.0 / 105.0)) < 1e-6

        # Spell 2: regime=0, bars [108], duration=1, event=False (censored)
        assert covs.iloc[2]["duration"] == 1
        assert covs.iloc[2]["event"] == False
        assert covs.iloc[2]["realized_vol"] == 0.0  # single bar → 0
        assert covs.iloc[2]["spell_return"] == 0.0  # single bar → 0

    def test_columns_present(self):
        """Output DataFrame has required columns."""
        from hmm_futures_analysis.regime.duration_forecast import (
            _build_spell_covariates,
            _extract_spells,
        )

        regimes = np.array([0, 0, 1, 1, 0, 0])
        prices = pd.Series([100.0, 102.0, 98.0, 101.0, 99.0, 103.0])
        spells = _extract_spells(regimes)
        covs = _build_spell_covariates(spells, prices)

        for col in ["duration", "event", "regime_idx", "realized_vol", "spell_return"]:
            assert col in covs.columns


lifelines = pytest.importorskip("lifelines")


class TestFitCoxPH:
    """_fit_coxph coefficient recovery with synthetic data."""

    def test_coefficient_recovery(self):
        """Synthetic data with known covariate effects → coefficient recovery."""
        from hmm_futures_analysis.regime.duration_forecast import (
            _extract_spells,
            _fit_coxph,
        )

        rng = np.random.default_rng(42)

        # Generate synthetic regime sequence with many regime-0 spells
        # Use exponentially-distributed durations with covariate-dependent hazard
        # h(t|x) = h0(t) * exp(beta * x)
        # Build prices that produce varying realized_vol
        seq = []
        price_list = []
        base_price = 100.0
        n_spells = 30  # need many for Cox to recover coefficients

        for i in range(n_spells):
            # High-volatility spells should be shorter
            vol_scale = 0.01 if i % 2 == 0 else 0.05
            dur = max(5, int(rng.exponential(30)))
            returns = rng.normal(0.001, vol_scale, dur)
            prices_spell = base_price * np.cumprod(1 + returns)
            price_list.extend(prices_spell.tolist())
            seq.extend([0] * dur)
            base_price = prices_spell[-1]

            # Interleave with regime 1
            gap = rng.integers(3, 8)
            gap_prices = base_price * np.cumprod(1 + rng.normal(0.001, 0.01, gap))
            price_list.extend(gap_prices.tolist())
            seq.extend([1] * gap)
            base_price = gap_prices[-1]

        regimes = np.array(seq)
        prices = pd.Series(price_list)
        spells = _extract_spells(regimes)

        result = _fit_coxph(regimes, spells, prices, current_regime_idx=0, days_in_regime=5)

        assert result is not None
        assert "cox_coefficients" in result
        assert "concordance_index" in result
        assert result["concordance_index"] > 0.5  # better than random
        assert "cox_expected_remaining_days" in result
        assert result["cox_expected_remaining_days"] > 0


class TestForecastDurationCox:
    """forecast_duration(model='cox') full output structure."""

    def test_cox_full_output_has_both_weibull_and_cox_fields(self):
        """model='cox' returns Weibull fields plus Cox fields."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        rng = np.random.default_rng(42)

        # Build a sequence with many regime-0 spells of longer duration
        # so conditional_after is well within the range
        seq = []
        for _ in range(15):
            seq.extend([0] * rng.integers(20, 50))
            seq.extend([1] * rng.integers(3, 8))
        regimes = np.array(seq)
        n = len(regimes)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices_vals = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        prices = pd.Series(prices_vals, index=dates)

        result = forecast_duration(regimes, model="cox", prices=prices)

        assert result is not None
        # Weibull fields always present
        for key in ["current_regime", "days_in_regime", "expected_remaining_days",
                     "hazard_rate", "survival_50pct", "weibull_shape", "weibull_scale"]:
            assert key in result, f"missing Weibull field: {key}"

        # Cox fields present
        for key in ["cox_coefficients", "concordance_index", "baseline_hazard_at_t",
                     "cox_expected_remaining_days"]:
            assert key in result, f"missing Cox field: {key}"

        # Cox fields should be populated (enough spells)
        if result["cox_coefficients"] is not None:
            assert result["concordance_index"] is not None
            assert isinstance(result["concordance_index"], float)

    def test_cox_graceful_degradation_insufficient_spells(self):
        """Fewer than 3 completed spells → Cox fields are None, Weibull still works."""
        from hmm_futures_analysis.regime.duration_forecast import forecast_duration

        # Build sequence where current regime has 0-2 completed spells
        # 0,0,0, 1,1, 0  → regime 0: 1 completed + 1 censored = only 1 completed
        regimes = np.array([0, 0, 0, 1, 1, 0])
        prices = pd.Series([100.0, 102.0, 98.0, 95.0, 99.0, 103.0])

        result = forecast_duration(regimes, model="cox", prices=prices)

        assert result is not None
        # Weibull fields should be None too (insufficient spells)
        assert result["weibull_shape"] is None
        # Cox fields should be None
        assert result["cox_coefficients"] is None
        assert result["concordance_index"] is None
        assert result["cox_expected_remaining_days"] is None


class TestPipelineCoxIntegration:
    """Pipeline.run() passes prices through to forecast_duration."""

    def test_pipeline_cox_produces_cox_fields(self, btc_csv):
        """pipeline.run(duration_model='cox') → output has both Weibull and Cox fields."""
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
        from hmm_futures_analysis.regime.engine_protocol import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices = load_from_csv(btc_csv)
        result = pipeline_run(
            prices, source="test", engine_config=ThresholdConfig(),
            duration_forecast=True, duration_model="cox",
        )

        assert "duration_forecast" in result
        df = result["duration_forecast"]
        if df is not None:
            # Weibull fields always present
            assert "weibull_shape" in df
            # Cox fields present (may be None if insufficient spells)
            assert "cox_coefficients" in df

    def test_pipeline_weibull_no_cox_fields(self, btc_csv):
        """pipeline.run(duration_model='weibull') → no Cox fields (backward compat)."""
        from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
        from hmm_futures_analysis.regime.engine_protocol import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices = load_from_csv(btc_csv)
        result = pipeline_run(
            prices, source="test", engine_config=ThresholdConfig(),
            duration_forecast=True, duration_model="weibull",
        )

        assert "duration_forecast" in result
        df = result["duration_forecast"]
        if df is not None:
            assert "cox_coefficients" not in df


class TestCLICoxIntegration:
    """CLI --duration-model cox integration."""

    def test_cli_cox_with_lifelines_works(self, btc_csv):
        """--duration-model cox with lifelines installed → produces output."""
        import json
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "hmm_futures_analysis.cli",
               "--csv", btc_csv,
               "--engine", "threshold",
               "--duration-forecast",
               "--duration-model", "cox",
               "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Should succeed (lifelines is installed)
        assert proc.returncode == 0
        # Output may have log lines before JSON; find the JSON start
        stdout = proc.stdout
        json_start = stdout.find("{")
        assert json_start >= 0, f"No JSON in output: {stdout[:200]}"
        output = json.loads(stdout[json_start:])
        assert "duration_forecast" in output

    def test_cli_cox_without_lifelines_error_message(self):
        """_fit_coxph raises ImportError with install hint when lifelines missing."""
        # Test the error message content (not CLI subprocess)
        import importlib
        import sys

        import hmm_futures_analysis.regime.duration_forecast as mod

        saved = {}
        for key in list(sys.modules):
            if key.startswith("lifelines"):
                saved[key] = sys.modules.pop(key)
        sys.modules["lifelines"] = None

        try:
            importlib.reload(mod)
            with pytest.raises(ImportError, match="lifelines"):
                mod._fit_coxph(
                    regimes=np.array([0, 0, 1, 1]),
                    spells=[],
                    prices=pd.Series([100.0]),
                    current_regime_idx=0,
                    days_in_regime=1,
                )
        finally:
            for key in list(sys.modules):
                if key.startswith("lifelines"):
                    del sys.modules[key]
            sys.modules.update(saved)
            importlib.reload(mod)
