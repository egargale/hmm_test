"""Unit tests for Messina-specific feature engineering."""
import numpy as np
import pandas as pd
import pytest

from data_processing.messina_features import (
    _calc_vstop,
    _true_range,
    _wilder_smooth,
    add_messina_features,
)


@pytest.fixture
def ohlcv_30():
    """30-row synthetic OHLCV DataFrame with known values."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.arange(100, 100 + n, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1000,
        },
        index=dates,
    )


@pytest.fixture
def ohlcv_250():
    """250-row synthetic OHLCV for tests needing SMA200 warmup."""
    np.random.seed(99)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, n))
    high = close + np.abs(np.random.normal(1.0, 0.5, n))
    low = close - np.abs(np.random.normal(1.0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close + np.random.normal(0, 0.3, n),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(500, 5000, n),
        },
        index=dates,
    )


class TestWilderSmooth:
    def test_wilder_smooth_values(self):
        """Wilder-smoothed series matches hand-computed values."""
        series = pd.Series([10, 12, 11, 14, 13, 15, 16, 14, 17, 18], dtype=float)
        period = 3
        result = _wilder_smooth(series, period)

        # First valid value at index period-1 = simple average of first 3
        assert result.iloc[period - 1] == pytest.approx(
            (10 + 12 + 11) / 3, abs=1e-10
        )

        # Index 3: prev + (val - prev) / period
        prev = (10 + 12 + 11) / 3
        expected_3 = prev + (14 - prev) / period
        assert result.iloc[3] == pytest.approx(expected_3, abs=1e-10)

    def test_wilder_smooth_nan_before_period(self):
        """Values before period-1 should be NaN."""
        series = pd.Series(range(10), dtype=float)
        result = _wilder_smooth(series, 5)
        assert result.iloc[:4].isna().all()
        assert not np.isnan(result.iloc[4])


class TestTrueRange:
    def test_true_range_calculation(self):
        """TR = max(H-L, |H-Cp|, |L-Cp|) for known values."""
        high = pd.Series([105, 110, 108])
        low = pd.Series([100, 103, 102])
        close = pd.Series([103, 107, 105])

        tr = _true_range(high, low, close)

        # Row 0: H-L=5, no prev close → just H-L (others compare to NaN → NaN)
        assert tr.iloc[0] == pytest.approx(5.0, abs=1e-10)
        # Row 1: H-L=7, |H-Cp|=|110-103|=7, |L-Cp|=|103-103|=0 → max=7
        assert tr.iloc[1] == pytest.approx(7.0, abs=1e-10)
        # Row 2: H-L=6, |H-Cp|=|108-107|=1, |L-Cp|=|102-107|=5 → max=6
        assert tr.iloc[2] == pytest.approx(6.0, abs=1e-10)

    def test_true_range_non_negative(self, ohlcv_250):
        tr = _true_range(ohlcv_250["high"], ohlcv_250["low"], ohlcv_250["close"])
        assert (tr.dropna() >= 0).all()


class TestCalcVstop:
    def test_calc_vstop_basic(self):
        """VSTOP tracks trailing stop for a simple uptrend."""
        n = 30
        sma13 = pd.Series(np.arange(100, 100 + n, dtype=float))
        atr20 = pd.Series([2.0] * n)

        vstop, trend = _calc_vstop(sma13, atr20, multiplier=2.0)

        # After warmup, uptrend should persist since prices are monotonically rising
        valid = vstop.dropna()
        trend_valid = trend[valid.index]
        assert (trend_valid == 1).all(), "Monotonically rising SMA13 should stay in uptrend"

    def test_calc_vstop_returns_series(self):
        sma13 = pd.Series([100.0] * 20)
        atr20 = pd.Series([1.0] * 20)
        vstop, trend = _calc_vstop(sma13, atr20)
        assert isinstance(vstop, pd.Series)
        assert isinstance(trend, pd.Series)
        assert len(vstop) == 20
        assert len(trend) == 20


class TestAddMessinaFeatures:
    EXPECTED_COLUMNS = {
        "log_ret",
        "sma_200",
        "sma_13",
        "atr_20",
        "adx_14",
        "di_plus_14",
        "di_minus_14",
        "adx_slope",
        "vstop",
        "vstop_trend",
        "price_sma200_ratio",
        "price_vstop_ratio",
    }

    def test_add_messina_features_columns(self, ohlcv_250):
        result = add_messina_features(ohlcv_250)
        for col in self.EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_add_messina_features_missing_columns(self):
        """Should raise ValueError when OHLCV columns are missing."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing"):
            add_messina_features(df)

    def test_add_messina_features_no_inf(self, ohlcv_250):
        result = add_messina_features(ohlcv_250)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Output contains inf values"

    def test_add_messina_features_no_modify_original(self, ohlcv_250):
        original_cols = set(ohlcv_250.columns)
        original_len = len(ohlcv_250.columns)
        add_messina_features(ohlcv_250)
        assert set(ohlcv_250.columns) == original_cols
        assert len(ohlcv_250.columns) == original_len
