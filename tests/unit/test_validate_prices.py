"""Unit tests for pipeline._validate_prices helper."""

import pandas as pd
import pytest

from hmm_futures_analysis.regime.pipeline import _validate_prices


class TestValidatePrices:
    """_validate_prices mirrors the guard clauses that were inline in run()."""

    @staticmethod
    def _make_series(n: int) -> pd.Series:
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.Series([float(i) for i in range(1, n + 1)], index=dates, dtype=float)

    # --- acceptance: valid input passes silently ---

    def test_valid_series_returns_returns(self):
        """A valid prices Series returns a non-empty returns Series."""
        returns = _validate_prices(self._make_series(10))
        assert isinstance(returns, pd.Series)
        assert len(returns) == 9  # pct_change drops first

    # --- mirrors existing run() validation tests ---

    def test_rejects_empty_series(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            _validate_prices(self._make_series(0))

    def test_rejects_single_row(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            _validate_prices(self._make_series(1))

    def test_rejects_zero_price_series(self):
        """[0.0, 0.0] produces only NaN returns — empty after dropna."""
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        with pytest.raises(ValueError, match="at least 2 valid returns"):
            _validate_prices(pd.Series([0.0, 0.0], index=idx))

    def test_rejects_non_numeric_dtype(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        s = pd.Series(["10.0", "20.0", "30.0"], index=idx)
        with pytest.raises(ValueError, match="numeric"):
            _validate_prices(s)

    def test_rejects_non_datetime_index(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _validate_prices(s)

    def test_rejects_dataframe(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match=r"pd\.Series"):
            _validate_prices(df)
