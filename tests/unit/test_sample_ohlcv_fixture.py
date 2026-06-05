"""Tests for the sample_ohlcv fixture contract."""

import pandas as pd

from tests.conftest import ROOT


def test_sample_ohlcv_has_datetime_index(sample_ohlcv):
    """sample_ohlcv fixture must return a DataFrame with a DatetimeIndex."""
    assert isinstance(sample_ohlcv.index, pd.DatetimeIndex)


def test_sample_ohlcv_columns_and_row_count(sample_ohlcv):
    """Fixture must have columns open/high/low/close/volume and ~2513 rows."""
    expected_cols = ["close", "high", "low", "open", "volume"]
    assert list(sample_ohlcv.columns) == expected_cols
    assert 2500 <= len(sample_ohlcv) <= 2600


def test_no_synthetic_csv_remains():
    """Old sample_ohlcv.csv synthetic data file must be removed."""
    assert not (ROOT / "test_data" / "sample_ohlcv.csv").exists()
