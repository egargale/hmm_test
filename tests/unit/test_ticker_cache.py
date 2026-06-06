"""Tests for ticker_cache module."""

from __future__ import annotations

import pandas as pd

from hmm_futures_analysis.data_processing.ticker_cache import get_ticker_data


def _make_fake_ohlcv() -> pd.DataFrame:
    """Build a fake yfinance-style OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [104, 105, 106, 107, 108],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )


class TestCacheMiss:
    """When cache file is absent, download and save."""

    def test_downloads_and_saves_on_miss(self, tmp_path, monkeypatch):
        """No cache file → yfinance called with period='max', file written."""
        fake_df = _make_fake_ohlcv()
        calls = []

        def fake_download(ticker, period, progress=False):
            calls.append((ticker, period))
            return fake_df.copy()

        monkeypatch.setattr("yfinance.download", fake_download)

        cache_dir = tmp_path / "cache"
        result = get_ticker_data("SPY", cache_dir=str(cache_dir))

        assert len(calls) == 1
        assert calls[0] == ("SPY", "max")
        pd.testing.assert_frame_equal(result, fake_df)

        expected_path = cache_dir / "SPY.csv"
        assert expected_path.exists()


class TestCacheHit:
    """When cache file exists, read from disk without downloading."""

    def test_reads_from_cache_on_hit(self, tmp_path, monkeypatch):
        """Cache file present → no yfinance call, data returned from disk."""
        fake_df = _make_fake_ohlcv()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "SPY.csv"
        fake_df.to_csv(cache_path)

        calls = []

        def fake_download(ticker, period, progress=False):
            calls.append((ticker, period))
            return fake_df.copy()

        monkeypatch.setattr("yfinance.download", fake_download)

        result = get_ticker_data("SPY", cache_dir=str(cache_dir))

        assert len(calls) == 0
        pd.testing.assert_frame_equal(result, fake_df, check_freq=False)


class TestRefresh:
    """When refresh=True, always re-download and overwrite."""

    def test_re_downloads_and_overwrites(self, tmp_path, monkeypatch):
        """refresh=True → yfinance called even when cache exists."""
        old_df = _make_fake_ohlcv()
        new_df = old_df.copy()
        new_df["Close"] = new_df["Close"] + 10

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "SPY.csv"
        old_df.to_csv(cache_path)

        calls = []

        def fake_download(ticker, period, progress=False):
            calls.append((ticker, period))
            return new_df.copy()

        monkeypatch.setattr("yfinance.download", fake_download)

        result = get_ticker_data("SPY", cache_dir=str(cache_dir), refresh=True)

        assert len(calls) == 1
        assert calls[0] == ("SPY", "max")
        pd.testing.assert_frame_equal(result, new_df, check_freq=False)


class TestNoCache:
    """When no_cache=True, bypass cache entirely."""

    def test_bypasses_cache_does_not_write(self, tmp_path, monkeypatch):
        """no_cache=True → downloads with period='10y', no file written."""
        fake_df = _make_fake_ohlcv()
        calls = []

        def fake_download(ticker, period, progress=False):
            calls.append((ticker, period))
            return fake_df.copy()

        monkeypatch.setattr("yfinance.download", fake_download)

        cache_dir = tmp_path / "cache"
        result = get_ticker_data("SPY", cache_dir=str(cache_dir), no_cache=True)

        assert len(calls) == 1
        assert calls[0] == ("SPY", "10y")
        pd.testing.assert_frame_equal(result, fake_df, check_freq=False)

        expected_path = cache_dir / "SPY.csv"
        assert not expected_path.exists()


class TestFilenameSanitization:
    """Ticker symbols are sanitized for safe filenames."""

    def test_es_f_becomes_es_f_csv(self, tmp_path, monkeypatch):
        """ES=F is written as ES_F.csv."""
        fake_df = _make_fake_ohlcv()
        monkeypatch.setattr("yfinance.download", lambda *a, **kw: fake_df.copy())

        cache_dir = tmp_path / "cache"
        get_ticker_data("ES=F", cache_dir=str(cache_dir))

        assert (cache_dir / "ES_F.csv").exists()

    def test_btc_usd_becomes_btc_usd_csv(self, tmp_path, monkeypatch):
        """BTC-USD is written as BTC_USD.csv."""
        fake_df = _make_fake_ohlcv()
        monkeypatch.setattr("yfinance.download", lambda *a, **kw: fake_df.copy())

        cache_dir = tmp_path / "cache"
        get_ticker_data("BTC-USD", cache_dir=str(cache_dir))

        assert (cache_dir / "BTC_USD.csv").exists()


class TestCorruptedFile:
    """Corrupted cache files are treated as misses."""

    def test_empty_file_triggers_re_download(self, tmp_path, monkeypatch):
        """Empty CSV → re-downloaded and overwritten."""
        fake_df = _make_fake_ohlcv()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "SPY.csv"
        cache_path.write_text("")

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: fake_df.copy())

        result = get_ticker_data("SPY", cache_dir=str(cache_dir))
        pd.testing.assert_frame_equal(result, fake_df, check_freq=False)
        # File should have been overwritten with valid data
        assert "Close" in pd.read_csv(cache_path).columns

    def test_missing_required_columns_triggers_re_download(self, tmp_path, monkeypatch):
        """CSV without OHLC columns → re-downloaded."""
        fake_df = _make_fake_ohlcv()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "SPY.csv"
        cache_path.write_text("date,foo\n2024-01-01,100\n")

        monkeypatch.setattr("yfinance.download", lambda *a, **kw: fake_df.copy())

        result = get_ticker_data("SPY", cache_dir=str(cache_dir))
        pd.testing.assert_frame_equal(result, fake_df, check_freq=False)


class TestDefaultCacheDir:
    """Default cache directory follows XDG convention."""

    def test_uses_xdg_cache_home(self, monkeypatch, tmp_path):
        """When cache_dir is omitted, uses ~/.cache/hmm-regime/tickers/."""
        fake_df = _make_fake_ohlcv()
        monkeypatch.setattr("yfinance.download", lambda *a, **kw: fake_df.copy())

        # Override HOME so we don't pollute the real filesystem
        fake_home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(fake_home))

        get_ticker_data("SPY")

        expected = fake_home / ".cache" / "hmm-regime" / "tickers" / "SPY.csv"
        assert expected.exists()
