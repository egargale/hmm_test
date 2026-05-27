"""CSV/yfinance price loader with auto-detection of date and close columns."""

from pathlib import Path

import pandas as pd


def _find_date_column(columns: list[str]) -> str | None:
    candidates = ["date", "time", "timestamp", "datetime"]
    lower = [c.lower().strip() for c in columns]
    for candidate in candidates:
        for i, col in enumerate(lower):
            if candidate in col:
                return columns[i]
    return None


def _find_close_column(df: pd.DataFrame) -> str | None:
    candidates = ["close", "adj close", "price", "last"]
    lower = [c.lower().strip() for c in df.columns]
    for candidate in candidates:
        for i, col in enumerate(lower):
            if candidate in col:
                return df.columns[i]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def load_from_csv(
    path: str,
    date_col: str | None = None,
    close_col: str | None = None,
) -> pd.Series:
    """Load close prices from a CSV file with auto-detection of columns.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    date_col : str or None
        Explicit date column name. Auto-detected if None.
    close_col : str or None
        Explicit close price column name. Auto-detected if None.

    Returns
    -------
    pd.Series
        Close prices with DatetimeIndex.

    Raises
    ------
    ValueError
        If no suitable date or close column can be found.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if date_col is None:
        date_col = _find_date_column(list(df.columns))
    if date_col is None:
        date_col = df.columns[0]

    if close_col is None:
        close_col = _find_close_column(df)
    if close_col is None:
        raise ValueError(
            f"No numeric column found in {path}. "
            f"Columns: {list(df.columns)}"
        )

    dates = pd.to_datetime(df[date_col])
    values = pd.to_numeric(df[close_col], errors="coerce")

    series = pd.Series(values.values, index=dates, name=close_col)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series = series.dropna()

    if series.empty:
        raise ValueError(f"No valid data after parsing {path}")

    return series


def load_from_yfinance(ticker: str, period: str = "10y") -> pd.Series:
    """Load close prices from yfinance.

    Parameters
    ----------
    ticker : str
        Stock/futures ticker symbol (e.g. "ES=F" for E-mini S&P 500).
    period : str
        Data period to download (default "10y").

    Returns
    -------
    pd.Series
        Close prices with DatetimeIndex.

    Raises
    ------
    ValueError
        If ticker returns no data.
    """
    import yfinance

    data = yfinance.download(ticker, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    if close.empty:
        raise ValueError(f"No valid close prices for ticker: {ticker}")

    close.name = ticker
    return close


def _check_min_rows(prices: pd.Series) -> None:
    """Raise ValueError if *prices* has fewer than 2 rows."""
    if len(prices) < 2:
        raise ValueError(
            f"Need at least 2 rows of price data, got {len(prices)}. "
            "Cannot compute returns from a single price point."
        )


def load_prices(
    *,
    csv: str | None = None,
    ticker: str | None = None,
) -> tuple[pd.Series, pd.DataFrame | None, str]:
    """Unified data-loading entry point.

    Exactly one of *csv* or *ticker* must be provided.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame | None, str]
        (prices, ohlcv, source_label)
    """
    if (csv is None) == (ticker is None):
        raise ValueError("Provide exactly one of csv or ticker")

    if csv is not None:
        prices = load_from_csv(csv)
        _check_min_rows(prices)
        return prices, None, csv

    # ticker path
    import yfinance

    ohlcv_raw = yfinance.download(ticker, period="10y", progress=False)
    if ohlcv_raw.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if isinstance(ohlcv_raw.columns, pd.MultiIndex):
        ohlcv_raw.columns = [c[0].lower() for c in ohlcv_raw.columns]
    else:
        ohlcv_raw.columns = [c.lower() for c in ohlcv_raw.columns]

    prices = ohlcv_raw["close"]
    ohlcv = ohlcv_raw[["open", "high", "low", "close", "volume"]]
    _check_min_rows(prices)
    return prices, ohlcv, ticker


def load_price_series(source: str, **kwargs) -> pd.Series:
    """Load a price series from CSV or yfinance.

    If *source* ends with ``.csv`` it is treated as a file path and loaded
    via :func:`load_from_csv`.  Otherwise it is interpreted as a ticker
    symbol and fetched via :func:`load_from_yfinance`.

    Parameters
    ----------
    source : str
        File path (ending in ``.csv``) or ticker symbol.
    **kwargs
        Forwarded to the underlying loader.

    Returns
    -------
    pd.Series
        Close prices with DatetimeIndex.
    """
    if source.lower().endswith(".csv"):
        return load_from_csv(source, **kwargs)
    return load_from_yfinance(source, **kwargs)
