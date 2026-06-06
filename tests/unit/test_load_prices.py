from __future__ import annotations

import pandas as pd
import pytest

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices

from tests.conftest import run_regime


@pytest.fixture
def tiny_csv(tmp_path):
    """Write a minimal CSV with 3 rows and return its path."""
    p = tmp_path / "prices.csv"
    p.write_text("date,close\n2024-01-01,100\n2024-01-02,101\n2024-01-03,102\n")
    return str(p)


class TestLoadPricesArgumentValidation:
    """load_prices rejects invalid argument combinations."""

    def test_raises_when_both_csv_and_ticker_provided(self):
        """Providing both csv and ticker raises ValueError."""
        with pytest.raises(ValueError, match="exactly one of"):
            load_prices(csv="file.csv", ticker="ES=F")

    def test_raises_when_neither_csv_nor_ticker_provided(self):
        """Providing neither csv nor ticker raises ValueError."""
        with pytest.raises(ValueError, match="exactly one of"):
            load_prices()


class TestDunderAllExports:
    """__all__ in data_processing exposes the correct public API."""

    def test_load_prices_is_exported(self):
        import hmm_futures_analysis.data_processing as dp

        assert "load_prices" in dp.__all__

    def test_load_from_yfinance_not_in_all(self):
        import hmm_futures_analysis.data_processing as dp

        assert "load_from_yfinance" not in dp.__all__

    def test_load_price_series_not_in_all(self):
        import hmm_futures_analysis.data_processing as dp

        assert "load_price_series" not in dp.__all__


@pytest.mark.slow
class TestCliLoadPricesIntegration:
    """CLI main() uses load_prices internally — verifies end-to-end."""

    def test_csv_produces_json_output(self, btc_csv):
        """CLI with --csv still produces valid JSON with threshold engine."""
        result = run_regime("--csv", btc_csv, "--json", "--engine", "threshold")
        assert result.returncode == 0
        import json

        output = json.loads(result.stdout)
        assert output.source == btc_csv
        assert "current_regime" in output


class TestLoadPricesYfinancePath:
    """yfinance path downloads, flattens, returns (prices, ohlcv, ticker)."""

    def test_returns_prices_ohlcv_and_ticker_label(self, tmp_path, monkeypatch):
        """yfinance path returns (prices, ohlcv_df, ticker_string)."""
        # Build fake yfinance output with MultiIndex columns
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # yfinance returns MultiIndex columns as (PriceType, Ticker)
        fake_df = pd.DataFrame(
            {
                ("Close", "ES=F"): [104, 105, 106, 107, 108],
                ("High", "ES=F"): [105, 106, 107, 108, 109],
                ("Low", "ES=F"): [99, 100, 101, 102, 103],
                ("Open", "ES=F"): [100, 101, 102, 103, 104],
                ("Volume", "ES=F"): [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )
        fake_df.columns = pd.MultiIndex.from_tuples(fake_df.columns)

        import yfinance as yf_mod

        monkeypatch.setattr(yf_mod, "download", lambda *a, **kw: fake_df)

        prices, ohlcv, label = load_prices(
            ticker="ES=F", cache_dir=str(tmp_path / "cache")
        )

        assert label == "ES=F"
        assert isinstance(prices, pd.Series)
        assert len(prices) == 5
        assert isinstance(ohlcv, pd.DataFrame)
        assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]


class TestLoadPricesMinRowsGuard:
    """load_prices rejects data with fewer than 2 rows."""

    def test_raises_when_csv_has_fewer_than_2_rows(self, tmp_path, monkeypatch):
        """Single-row CSV raises ValueError about needing 2 rows."""
        p = tmp_path / "tiny.csv"
        p.write_text("date,close\n2024-01-01,100\n")
        with pytest.raises(ValueError, match="at least 2 rows"):
            load_prices(csv=str(p))

    def test_raises_when_yfinance_returns_fewer_than_2_rows(self, tmp_path, monkeypatch):
        """Single-row yfinance result raises ValueError about needing 2 rows."""
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        fake_df = pd.DataFrame(
            {
                ("Close", "X"): [100],
                ("High", "X"): [101],
                ("Low", "X"): [99],
                ("Open", "X"): [99],
                ("Volume", "X"): [1000],
            },
            index=dates,
        )
        fake_df.columns = pd.MultiIndex.from_tuples(fake_df.columns)

        import yfinance as yf_mod

        monkeypatch.setattr(yf_mod, "download", lambda *a, **kw: fake_df)

        with pytest.raises(ValueError, match="at least 2 rows"):
            load_prices(ticker="X", cache_dir=str(tmp_path / "cache"))


class TestLoadPricesCsvPath:
    """CSV path delegates to load_from_csv and returns (prices, None, label)."""

    def test_returns_prices_series_none_ohlcv_and_path_label(self, tiny_csv):
        """CSV path returns (prices, None, file_path)."""
        prices, ohlcv, label = load_prices(csv=tiny_csv)
        assert isinstance(prices, pd.Series)
        assert len(prices) == 3
        assert ohlcv is None
        assert label == tiny_csv


class TestLoadPricesCacheIntegration:
    """Ticker path delegates to ticker_cache when cache params provided."""

    def test_uses_cache_and_returns_prices_ohlcv(self, tmp_path, monkeypatch):
        """load_prices with ticker and cache_dir delegates to ticker_cache."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        fake_df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [99, 100, 101, 102, 103],
                "Close": [104, 105, 106, 107, 108],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: fake_df.copy())

        cache_dir = tmp_path / "cache"
        prices, ohlcv, label = load_prices(
            ticker="SPY",
            cache_dir=str(cache_dir),
        )

        assert label == "SPY"
        assert isinstance(prices, pd.Series)
        assert len(prices) == 5
        assert isinstance(ohlcv, pd.DataFrame)
        assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]
        assert (cache_dir / "SPY.csv").exists()
