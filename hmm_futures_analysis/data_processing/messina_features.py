"""
Messina-specific features for HMM regime detection.

Computes the exact indicators used by the Messina Signals framework:
SMA200, SMA13, ATR20 (Wilder's), ADX14/DI, VSTOP, and derived ratios.
"""
import numpy as np
import pandas as pd

from ..utils.logging_config import get_logger

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
        log_ret, sma_200, sma_13, atr_20, adx_14, adx_inflection,
        di_plus_14, di_minus_14, di_spread,
        vstop, vstop_trend, vstop_interaction,
        price_sma200_ratio, price_vstop_ratio,
        price_vstop_gap_atr, sma200_distance_atr,
        volume_ratio, true_range_pct, kdj_j
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_messina_features requires OHLCV columns, missing: {sorted(missing)}")

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

    # ADX inflection (1-day) — V3 rule #1: the last tick matters more than 3-day slope
    df["adx_inflection"] = df["adx_14"].diff(1)

    # DI spread — directional momentum balance; keeps the continuous distribution
    # the HMM needs (di_crossover was too sparse — broke Gaussian clustering)
    df["di_spread"] = di_plus - di_minus

    # ── VSTOP ─────────────────────────────────────────────────────────
    df["vstop"], trend_series = _calc_vstop(df["sma_13"], df["atr_20"], vstop_multiplier)
    df["vstop_trend"] = trend_series  # 1 = uptrend, -1 = downtrend

    # VSTOP interaction — encodes whether VSTOP acts as support or resistance
    # sign(close - vstop) * vstop_trend:
    #   +1 = price above VSTOP in uptrend (support held)
    #   −1 = price below VSTOP in uptrend (retest zone)
    #   +1 = price below VSTOP in downtrend (resistance held)
    #   −1 = price above VSTOP in downtrend (D→B break)
    df["vstop_interaction"] = np.sign(df["close"] - df["vstop"]) * df["vstop_trend"]

    # ── Derived ratios & level gaps ───────────────────────────────────
    df["price_sma200_ratio"] = close / df["sma_200"]
    df["price_vstop_ratio"] = close / df["vstop"]

    # Volatility-normalised gaps from key levels (V3 §4.3 freshness, §2.1 structure)
    df["price_vstop_gap_atr"] = (close - df["vstop"]) / df["atr_20"]
    df["sma200_distance_atr"] = (close - df["sma_200"]) / df["atr_20"]

    # ── Volume & volatility ──────────────────────────────────────────
    # Volume ratio — regime events (VSTOP break, crossover) on high volume
    # are more significant than on low volume
    vol_sma20 = df["volume"].rolling(window=20, min_periods=20).mean()
    df["volume_ratio"] = df["volume"] / vol_sma20

    # True range as percentage of price — volatility regime independent of price level
    df["true_range_pct"] = tr / close

    # ── KDJ oscillator (Wilder's smoothing, period 3) ─────────────────
    # Standard configuration: 9, 3, 3, 3 (RSV period, K smooth, D smooth, J coeff)
    # Uses Wilder's moving average: K = K_prev + (RSV − K_prev) / smooth
    # J-line oscillates beyond [0, 100] for early overbought (>100) / oversold (<0)
    kdj_n = 9
    kdj_smooth = 3
    low_n = df["low"].rolling(window=kdj_n, min_periods=kdj_n).min()
    high_n = df["high"].rolling(window=kdj_n, min_periods=kdj_n).max()
    rsv = 100.0 * (close - low_n) / (high_n - low_n)
    kdj_k = pd.Series(np.nan, index=df.index)
    kdj_d = pd.Series(np.nan, index=df.index)
    first_k = kdj_n - 1  # first valid RSV bar
    if len(df) > first_k:
        kdj_k.iloc[first_k] = rsv.iloc[first_k]  # Wilder's: initialise from first value
        kdj_d.iloc[first_k] = rsv.iloc[first_k]
        for i in range(first_k + 1, len(df)):
            if pd.notna(rsv.iloc[i]):
                kdj_k.iloc[i] = kdj_k.iloc[i - 1] + (rsv.iloc[i] - kdj_k.iloc[i - 1]) / kdj_smooth
                kdj_d.iloc[i] = kdj_d.iloc[i - 1] + (kdj_k.iloc[i] - kdj_d.iloc[i - 1]) / kdj_smooth
    df["kdj_j"] = 3.0 * kdj_k - 2.0 * kdj_d

    # Replace infinities from division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(
        "Added Messina features (18): sma_200, sma_13, atr_20, adx_14, "
        "adx_inflection, di_+/-, di_crossover, vstop, ratios, gaps, volume"
    )
    return df
