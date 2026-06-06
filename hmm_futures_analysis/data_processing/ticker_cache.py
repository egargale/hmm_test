"""Disk cache for yfinance OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "hmm-regime" / "tickers"


def _sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("=", "_").replace("-", "_")


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_sanitize_ticker(ticker)}.csv"


def _save_to_cache(df: pd.DataFrame, path: Path) -> None:
    """Save *df* to *path*, flattening MultiIndex columns if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.to_csv(path)


def _load_from_cache(path: Path) -> pd.DataFrame | None:
    """Try to load a cached CSV.  Returns *None* if corrupted or invalid."""
    try:
        data = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None

    if data.empty:
        return None

    lower_cols = [c.lower().strip() for c in data.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(lower_cols):
        return None

    return data


def get_ticker_data(
    ticker: str,
    cache_dir: str | None = None,
    refresh: bool = False,
    no_cache: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data for *ticker*, with optional disk caching.

    Parameters
    ----------
    ticker : str
        Stock/futures ticker symbol.
    cache_dir : str or None
        Directory for cached CSV files.  Uses ``~/.cache/hmm-regime/tickers/``
        when *None* and caching is active.
    refresh : bool
        Force re-download even if a cached file exists.
    no_cache : bool
        Bypass the cache entirely — do not read or write.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame (same shape as ``yfinance.download`` output).
    """
    import yfinance

    if no_cache:
        data = yfinance.download(ticker, period="max", progress=False)
        if data.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        return data

    cache_dir_path = Path(cache_dir) if cache_dir else _default_cache_dir()
    path = _cache_path(ticker, cache_dir_path)

    if not refresh and path.exists():
        data = _load_from_cache(path)
        if data is not None:
            return data

    data = yfinance.download(ticker, period="max", progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    _save_to_cache(data, path)
    return data
