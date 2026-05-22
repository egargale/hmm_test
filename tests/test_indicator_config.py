"""Unit tests for indicator config structure and feature production."""
import numpy as np
import pandas as pd
import pytest

from data_processing.technical_indicators import (
    get_default_indicator_config,
    validate_indicator_config,
)


class TestDefaultConfigStructure:
    EXPECTED_CATEGORIES = {
        "moving_averages",
        "volatility",
        "momentum",
        "volume",
        "trend",
    }

    def test_default_config_nested(self):
        """Config returns nested dict with expected top-level category keys."""
        config = get_default_indicator_config()
        for cat in self.EXPECTED_CATEGORIES:
            assert cat in config, f"Missing category: {cat}"
            assert isinstance(config[cat], dict), f"Category {cat} is not a dict"

    def test_default_config_has_sma_ema(self):
        config = get_default_indicator_config()
        ma = config["moving_averages"]
        assert "sma" in ma
        assert "ema" in ma
        assert "length" in ma["sma"]

    def test_default_config_has_adx(self):
        config = get_default_indicator_config()
        assert "adx" in config["trend"]
        assert "length" in config["trend"]["adx"]

    def test_default_config_adl_vpt_enabled(self):
        """ADL and VPT should have enabled=True after the config fix."""
        config = get_default_indicator_config()
        ev = config["enhanced_volume"]
        assert ev["adl"].get("enabled") is True
        assert ev["vpt"].get("enabled") is True


class TestValidateIndicatorConfig:
    def test_valid_default_config(self):
        config = get_default_indicator_config()
        assert validate_indicator_config(config) is True

    def test_rejects_non_dict(self):
        assert validate_indicator_config("not a dict") is False

    def test_rejects_negative_length(self):
        config = get_default_indicator_config()
        config["moving_averages"]["sma"]["length"] = -1
        assert validate_indicator_config(config) is False

    def test_rejects_macd_fast_ge_slow(self):
        config = get_default_indicator_config()
        config["momentum"]["macd"]["fast"] = 30
        config["momentum"]["macd"]["slow"] = 10
        assert validate_indicator_config(config) is False

    def test_accepts_empty_config(self):
        assert validate_indicator_config({}) is True


class TestFeatureProduction:
    """Verify that add_features() with default config actually produces features."""

    @pytest.fixture
    def ohlcv_300(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        return pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(1, 0.5, n)),
                "low": close - np.abs(np.random.normal(1, 0.5, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n),
            },
            index=dates,
        )

    def test_config_produces_sma(self, ohlcv_300):
        from data_processing.feature_engineering import add_features

        df = add_features(ohlcv_300, min_periods=10)
        sma_cols = [c for c in df.columns if c.startswith("sma_")]
        assert len(sma_cols) > 0, "No SMA columns produced — indicators still skipped"

    def test_config_produces_rsi(self, ohlcv_300):
        from data_processing.feature_engineering import add_features

        df = add_features(ohlcv_300, min_periods=10)
        rsi_cols = [c for c in df.columns if "rsi" in c.lower()]
        assert len(rsi_cols) > 0, "No RSI columns produced"

    def test_config_produces_atr(self, ohlcv_300):
        from data_processing.feature_engineering import add_features

        df = add_features(ohlcv_300, min_periods=10)
        atr_cols = [c for c in df.columns if "atr" in c.lower()]
        assert len(atr_cols) > 0, "No ATR columns produced"

    def test_config_produces_adx(self, ohlcv_300):
        from data_processing.feature_engineering import add_features

        df = add_features(ohlcv_300, min_periods=10)
        adx_cols = [c for c in df.columns if "adx" in c.lower()]
        assert len(adx_cols) > 0, "No ADX columns produced"

    def test_config_produces_adl_vpt(self, ohlcv_300):
        from data_processing.feature_engineering import add_features

        df = add_features(ohlcv_300, min_periods=10)
        assert "adl" in df.columns, "ADL column not produced"
        assert "vpt" in df.columns, "VPT column not produced"
