"""
Messina-specific features for HMM regime detection.

Computes the exact indicators used by the Messina Signals framework:
SMA200, SMA13, ATR20 (Wilder's), ADX14/DI, VSTOP, and derived ratios.
"""
import numpy as np
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── helpers ─────────────────────────────────────────────────────────────


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's exponential smoothing (also called modified EMA)."""
    result = series.copy()
    # First value = simple average
    first = series.iloc[:period].mean()
    result.iloc[period - 1] = first
    for i in range(period, len(series)):
        result.iloc[i] = result.iloc[i - 1] + (series.iloc[i] - result.iloc[i - 1]) / period
    # NaN prior to period
    result.iloc[: period - 1] = np.nan
    return result


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _calc_vstop(
    sma13: pd.Series,
    atr20: pd.Series,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Volatility Stop trailing stop.

    Returns (vstop, trend) where trend = 1 for up, -1 for down.
    """
    vstop = pd.Series(np.nan, index=sma13.index)
    trend = pd.Series(0, index=sma13.index)

    max_price = 0.0
    min_price = float("inf")
    last_vstop = 0.0
    current_trend = 1  # 1 = up, -1 = down

    for i in range(len(sma13)):
        sma = sma13.iloc[i]
        atr = atr20.iloc[i]
        if pd.isna(sma) or pd.isna(atr):
            continue

        atr_mult = atr * multiplier

        if last_vstop == 0.0:  # first valid bar
            current_trend = 1
            max_price = sma
            min_price = sma
            last_vstop = sma - atr_mult
            vstop.iloc[i] = last_vstop
            trend.iloc[i] = current_trend
            continue

        if current_trend == 1:  # uptrend
            max_price = max(max_price, sma)
            last_vstop = max(last_vstop, max_price - atr_mult)
            if sma < last_vstop:
                current_trend = -1
                max_price = sma
                min_price = sma
                last_vstop = sma + atr_mult
        else:  # downtrend
            min_price = min(min_price, sma)
            last_vstop = min(last_vstop, min_price + atr_mult)
            if sma > last_vstop:
                current_trend = 1
                max_price = sma
                min_price = sma
                last_vstop = sma - atr_mult

        vstop.iloc[i] = last_vstop
        trend.iloc[i] = current_trend

    return vstop, trend


# ── main entry point ────────────────────────────────────────────────────


def add_messina_features(
    df: pd.DataFrame,
    vstop_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Add Messina-specific features to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
    vstop_multiplier : float
        ATR multiplier for VSTOP (default 2.0).

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        log_ret, sma_200, sma_13, atr_20, adx_14, di_plus_14, di_minus_14,
        adx_slope, vstop, vstop_trend, price_sma200_ratio, price_vstop_ratio
    """
    df = df.copy()
    close = df["close"]

    # Log returns
    df["log_ret"] = np.log(close / close.shift(1))

    # ── SMA ──────────────────────────────────────────────────────────
    df["sma_200"] = close.rolling(window=200, min_periods=200).mean()
    df["sma_13"] = close.rolling(window=13, min_periods=13).mean()

    # ── True Range & ATR20 (Wilder's smoothing) ──────────────────────
    tr = _true_range(df["high"], df["low"], close)
    df["atr_20"] = _wilder_smooth(tr, 20)

    # ── ADX14 / DI+ / DI- (Wilder's smoothing) ───────────────────────
    period = 14
    prev_close = close.shift(1)
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]

    dm_plus = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    dm_plus.iloc[0] = 0.0
    dm_minus.iloc[0] = 0.0

    smooth_tr = _wilder_smooth(tr, period)
    smooth_dmp = _wilder_smooth(dm_plus, period)
    smooth_dmn = _wilder_smooth(dm_minus, period)

    di_plus = 100.0 * smooth_dmp / smooth_tr
    di_minus = 100.0 * smooth_dmn / smooth_tr
    di_plus[smooth_tr == 0] = 0.0
    di_minus[smooth_tr == 0] = 0.0

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    dx[(di_plus + di_minus) == 0] = 0.0

    df["adx_14"] = _wilder_smooth(dx, period)
    df["di_plus_14"] = di_plus
    df["di_minus_14"] = di_minus

    # ADX slope (3-day)
    df["adx_slope"] = df["adx_14"].diff(3)

    # ── VSTOP ─────────────────────────────────────────────────────────
    df["vstop"], trend_series = _calc_vstop(df["sma_13"], df["atr_20"], vstop_multiplier)
    df["vstop_trend"] = trend_series  # 1 = uptrend, -1 = downtrend

    # ── Derived ratios ───────────────────────────────────────────────
    df["price_sma200_ratio"] = close / df["sma_200"]
    df["price_vstop_ratio"] = close / df["vstop"]

    # Replace infinities from division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(
        f"Added Messina features: sma_200, sma_13, atr_20, adx_14, "
        f"di_+/-, adx_slope, vstop, ratios"
    )
    return df
