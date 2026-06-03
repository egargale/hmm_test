"""Per-engine default whipsaw filter recommendations.

Issue #84 — each engine config carries default_dwell_bars and
default_hysteresis_delta reflecting its trade-frequency characteristics.
"""

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.engine_configs import (
    FSHMMConfig,
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)


def _call_parse(parse_fn, value):
    """Call a parser function; convert ArgumentTypeError to SystemExit."""
    try:
        return parse_fn(value)
    except SystemExit:
        raise
    except Exception as exc:
        # argparse wraps ArgumentTypeError into SystemExit during real parsing;
        # in unit tests we catch it directly.
        raise SystemExit(str(exc)) from exc


class TestParseDwellBars:
    """_parse_dwell_bars accepts 'auto' or a non-negative integer."""

    def test_auto_returns_auto(self):
        from hmm_futures_analysis.cli import _parse_dwell_bars

        assert _parse_dwell_bars("auto") == "auto"

    def test_integer_value(self):
        from hmm_futures_analysis.cli import _parse_dwell_bars

        assert _parse_dwell_bars("3") == 3

    def test_zero_value(self):
        from hmm_futures_analysis.cli import _parse_dwell_bars

        assert _parse_dwell_bars("0") == 0

    def test_negative_rejected(self):
        from hmm_futures_analysis.cli import _parse_dwell_bars

        with pytest.raises(SystemExit):
            _call_parse(_parse_dwell_bars, "-1")

    def test_non_numeric_rejected(self):
        from hmm_futures_analysis.cli import _parse_dwell_bars

        with pytest.raises(SystemExit):
            _call_parse(_parse_dwell_bars, "abc")


class TestParseHysteresis:
    """_parse_hysteresis accepts 'auto' or a float in [0, 1)."""

    def test_auto_returns_auto(self):
        from hmm_futures_analysis.cli import _parse_hysteresis

        assert _parse_hysteresis("auto") == "auto"

    def test_float_value(self):
        from hmm_futures_analysis.cli import _parse_hysteresis

        assert _parse_hysteresis("0.1") == 0.1

    def test_zero_value(self):
        from hmm_futures_analysis.cli import _parse_hysteresis

        assert _parse_hysteresis("0.0") == 0.0

    def test_negative_rejected(self):
        from hmm_futures_analysis.cli import _parse_hysteresis

        with pytest.raises(SystemExit):
            _call_parse(_parse_hysteresis, "-0.1")

    def test_one_rejected(self):
        from hmm_futures_analysis.cli import _parse_hysteresis

        with pytest.raises(SystemExit):
            _call_parse(_parse_hysteresis, "1.0")

    def test_non_numeric_rejected(self):
        from hmm_futures_analysis.cli import _parse_hysteresis

        with pytest.raises(SystemExit):
            _call_parse(_parse_hysteresis, "abc")


class TestResolveAutoFilters:
    """pipeline.run() resolves 'auto' to engine config defaults."""

    @pytest.fixture()
    def prices(self):
        """Price series long enough for HMM engine fitting."""
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.standard_normal(400) * 0.5)
        return pd.Series(close, index=dates, name="close")

    @pytest.fixture()
    def ohlcv(self):
        """Synthetic OHLCV data for HMM engines."""
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.standard_normal(400) * 0.5)
        return pd.DataFrame(
            {
                "open": close + rng.standard_normal(400) * 0.3,
                "high": close + np.abs(rng.standard_normal(400) * 0.8),
                "low": close - np.abs(rng.standard_normal(400) * 0.8),
                "close": close,
                "volume": rng.integers(100, 10_000, 400).astype(float),
            },
            index=dates,
        )

    def test_threshold_auto_resolves_dwell(self, prices):
        """--dwell-bars auto with threshold resolves to 3."""
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run

        result_auto = run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
            dwell_bars="auto",
            profile=False,
        )
        result_explicit = run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
            dwell_bars=3,
            profile=False,
        )
        assert (
            result_auto.walk_forward["n_trades"]
            == result_explicit.walk_forward["n_trades"]
        )

    def test_hmm_auto_resolves_hysteresis(self, prices, ohlcv):
        """--hysteresis auto with hmm resolves to 0.1."""
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run

        result_auto = run(
            prices,
            source="test",
            engine_config=HMMGenericConfig(n_states=3),
            ohlcv=ohlcv,
            hysteresis_delta="auto",
            profile=False,
        )
        result_explicit = run(
            prices,
            source="test",
            engine_config=HMMGenericConfig(n_states=3),
            ohlcv=ohlcv,
            hysteresis_delta=0.1,
            profile=False,
        )
        assert (
            result_auto.walk_forward["n_trades"]
            == result_explicit.walk_forward["n_trades"]
        )

    def test_auto_both_filters(self, prices, ohlcv):
        """Both filters as auto resolve to engine defaults."""
        from hmm_futures_analysis.regime.engine_configs import FSHMMConfig
        from hmm_futures_analysis.regime.pipeline import run

        result_auto = run(
            prices,
            source="test",
            engine_config=FSHMMConfig(n_states=3),
            ohlcv=ohlcv,
            dwell_bars="auto",
            hysteresis_delta="auto",
            profile=False,
        )
        result_explicit = run(
            prices,
            source="test",
            engine_config=FSHMMConfig(n_states=3),
            ohlcv=ohlcv,
            dwell_bars=2,
            hysteresis_delta=0.05,
            profile=False,
        )
        assert (
            result_auto.walk_forward["n_trades"]
            == result_explicit.walk_forward["n_trades"]
        )


class TestExplicitValuesPassThrough:
    """Explicit numeric values bypass auto-resolution — backward compat."""

    @pytest.fixture()
    def prices(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.standard_normal(200) * 0.5)
        return pd.Series(close, index=dates, name="close")

    def test_explicit_dwell_unchanged(self, prices):
        """Explicit dwell_bars=5 is passed through, not overridden by auto."""
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run

        result = run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
            dwell_bars=5,
            profile=False,
        )
        assert isinstance(result.walk_forward["n_trades"], int)

    def test_zero_defaults_unchanged(self, prices):
        """dwell_bars=0, hysteresis_delta=0.0 is backward compatible."""
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run

        result = run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
            dwell_bars=0,
            hysteresis_delta=0.0,
            profile=False,
        )
        result_default = run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
            profile=False,
        )
        assert (
            result.walk_forward["n_trades"]
            == result_default.walk_forward["n_trades"]
        )


class TestCLIAutoFlags:
    """CLI --dwell-bars auto and --hysteresis auto produce valid output."""

    def test_cli_dwell_bars_auto(self, btc_csv):
        """--dwell-bars auto produces valid JSON output."""
        from tests.conftest import run_regime
        import json

        result = run_regime("--csv", btc_csv, "--json", "--dwell-bars", "auto")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "walk_forward" in data
        assert isinstance(data["walk_forward"]["n_trades"], int)

    def test_cli_hysteresis_auto(self, btc_csv):
        """--hysteresis auto produces valid JSON output."""
        from tests.conftest import run_regime
        import json

        result = run_regime("--csv", btc_csv, "--json", "--hysteresis", "auto")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "walk_forward" in data

    def test_cli_both_auto(self, btc_csv):
        """Both auto flags produce valid JSON output."""
        from tests.conftest import run_regime
        import json

        result = run_regime(
            "--csv", btc_csv, "--json",
            "--dwell-bars", "auto", "--hysteresis", "auto",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "walk_forward" in data

    def test_cli_auto_matches_explicit_for_threshold(self, btc_csv):
        """--dwell-bars auto with threshold matches --dwell-bars 3."""
        from tests.conftest import run_regime
        import json

        result_auto = run_regime(
            "--csv", btc_csv, "--json", "--engine", "threshold",
            "--dwell-bars", "auto",
        )
        result_explicit = run_regime(
            "--csv", btc_csv, "--json", "--engine", "threshold",
            "--dwell-bars", "3",
        )
        assert result_auto.returncode == 0
        assert result_explicit.returncode == 0
        data_auto = json.loads(result_auto.stdout)
        data_explicit = json.loads(result_explicit.stdout)
        assert data_auto["walk_forward"] == data_explicit["walk_forward"]


class TestEngineFilterDefaults:
    """Every engine config exposes filter defaults with engine-appropriate values."""

    # --- threshold: high frequency, needs dwell filtering ---

    def test_threshold_dwell_bars(self):
        assert ThresholdConfig().default_dwell_bars == 3

    def test_threshold_hysteresis_delta(self):
        assert ThresholdConfig().default_hysteresis_delta == 0.0

    # --- hmm: already sticky, hysteresis on signal strength ---

    def test_hmm_dwell_bars(self):
        assert HMMGenericConfig().default_dwell_bars == 0

    def test_hmm_hysteresis_delta(self):
        assert HMMGenericConfig().default_hysteresis_delta == 0.1

    # --- messina: low frequency, hysteresis only ---

    def test_messina_dwell_bars(self):
        assert HMMMMessinaConfig().default_dwell_bars == 0

    def test_messina_hysteresis_delta(self):
        assert HMMMMessinaConfig().default_hysteresis_delta == 0.1

    # --- robust_hmm: very low frequency, hysteresis only ---

    def test_robust_hmm_dwell_bars(self):
        assert RobustHMMConfig().default_dwell_bars == 0

    def test_robust_hmm_hysteresis_delta(self):
        assert RobustHMMConfig().default_hysteresis_delta == 0.1

    # --- fshmm: low-medium frequency, light dwell + light hysteresis ---

    def test_fshmm_dwell_bars(self):
        assert FSHMMConfig().default_dwell_bars == 2

    def test_fshmm_hysteresis_delta(self):
        assert FSHMMConfig().default_hysteresis_delta == 0.05

    # --- backward compatibility: defaults don't affect construction ---

    @pytest.mark.parametrize(
        "config_cls",
        [ThresholdConfig, HMMGenericConfig, HMMMMessinaConfig, RobustHMMConfig, FSHMMConfig],
    )
    def test_defaults_are_opt_in_no_behavior_change(self, config_cls):
        """Configs construct with same positional args as before — new fields have defaults."""
        cfg = config_cls()
        assert hasattr(cfg, "default_dwell_bars")
        assert hasattr(cfg, "default_hysteresis_delta")
        assert isinstance(cfg.default_dwell_bars, int)
        assert isinstance(cfg.default_hysteresis_delta, float)
