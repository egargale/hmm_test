"""
Feature Engineering Module

Computes technical indicators and features for OHLCV data using various libraries
including pandas_ta, with support for memory-efficient processing and configurable
indicator parameters.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import get_logger
from .technical_indicators import (
    get_default_indicator_config,
    get_available_indicators,
    validate_indicator_config,
    validate_ohlcv_columns
)

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


def add_features(
    df: pd.DataFrame,
    indicator_config: Optional[Dict[str, Dict[str, Any]]] = None,
    downcast_floats: bool = True,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Add technical indicators and features to OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        indicator_config: Configuration dictionary for indicators
        downcast_floats: Whether to downcast float64 to float32
        min_periods: Minimum periods for rolling calculations

    Returns:
        pd.DataFrame: Original DataFrame with added features

    Raises:
        ValueError: If required OHLCV columns are missing
    """
    logger.info(f"Adding features to DataFrame with {len(df)} rows")

    # Validate input DataFrame
    validate_ohlcv_columns(df)

    # Use default indicator config if not provided
    if indicator_config is None:
        indicator_config = get_default_indicator_config()

    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not datetime, attempting conversion")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.warning(f"Could not convert index to datetime: {e}")

    # Set default min_periods based on data size
    if min_periods is None:
        min_periods = max(10, len(df) // 100)  # At least 10 or 1% of data

    logger.info(f"Using min_periods={min_periods}")

    # Calculate features
    features_added = []

    # 1. Basic returns
    df = _add_basic_returns(df, min_periods)
    features_added.extend(['log_ret', 'simple_ret'])

    # 2. Moving averages
    df = _add_moving_averages(df, indicator_config.get('moving_averages', {}), min_periods)
    features_added.extend([col for col in df.columns if 'ma_' in col or 'ema_' in col])

    # 3. Volatility indicators
    df = _add_volatility_indicators(df, indicator_config.get('volatility', {}), min_periods)
    features_added.extend([col for col in df.columns if 'atr' in col or 'volatility' in col])

    # 4. Momentum indicators
    df = _add_momentum_indicators(df, indicator_config.get('momentum', {}), min_periods)
    features_added.extend([col for col in df.columns if any(x in col for x in ['rsi', 'roc', 'macd', 'stoch'])])

    # 5. Volume indicators
    df = _add_volume_indicators(df, indicator_config.get('volume', {}), min_periods)
    features_added.extend([col for col in df.columns if any(x in col for x in ['volume_', 'obv', 'vwap'])])

    # 6. Trend indicators
    df = _add_trend_indicators(df, indicator_config.get('trend', {}), min_periods)
    features_added.extend([col for col in df.columns if any(x in col for x in ['adx', 'cci', 'dpo'])])

    # 7. Price patterns
    df = _add_price_patterns(df, indicator_config.get('patterns', {}), min_periods)
    features_added.extend([col for col in df.columns if any(x in col for x in ['bb_', 'kc_', 'dc_'])])

    # 8. Enhanced momentum indicators
    df = _add_enhanced_momentum_indicators(df, indicator_config.get('enhanced_momentum', {}), min_periods)
    enhanced_momentum_features = [col for col in df.columns if any(x in col for x in ['williams_r', 'cci', 'mfi', 'mtm', 'proc'])]
    features_added.extend(enhanced_momentum_features)

    # 9. Enhanced volatility indicators
    df = _add_enhanced_volatility_indicators(df, indicator_config.get('enhanced_volatility', {}), min_periods)
    enhanced_volatility_features = [col for col in df.columns if any(x in col for x in ['chaikin_vol', 'hv_', 'keltner_', 'donchian_'])]
    features_added.extend(enhanced_volatility_features)

    # 10. Enhanced trend indicators
    df = _add_enhanced_trend_indicators(df, indicator_config.get('enhanced_trend', {}), min_periods)
    enhanced_trend_features = [col for col in df.columns if any(x in col for x in ['tma_', 'wma_', 'hma_', 'aroon_', 'di_', 'adx_'])]
    features_added.extend(enhanced_trend_features)

    # 11. Enhanced volume indicators
    df = _add_enhanced_volume_indicators(df, indicator_config.get('enhanced_volume', {}), min_periods)
    enhanced_volume_features = [col for col in df.columns if any(x in col for x in ['adl', 'vpt', 'eom_', 'volume_roc_'])]
    features_added.extend(enhanced_volume_features)

    # 12. Time-based features
    df = _add_time_based_features(df, indicator_config)
    time_features = [col for col in df.columns if any(x in col for x in ['day_of_', 'month', 'quarter', 'hour', 'session', 'is_'])]
    features_added.extend(time_features)

    # 13. Custom indicators from config
    df = _add_custom_indicators(df, indicator_config.get('custom', {}), min_periods)
    custom_features = [col for col in df.columns if col not in df.columns[:len(df.columns) - len(features_added)]]
    features_added.extend(custom_features)

    # Downcast float columns for memory efficiency
    if downcast_floats:
        df = _downcast_float_columns(df)

    # Log feature addition summary
    logger.info(f"Added {len(features_added)} features: {features_added[:15]}...")
    logger.info(f"Total columns: {len(df.columns)}")

    return df






def _add_basic_returns(df: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """Add basic return calculations."""
    # Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # Simple returns
    df['simple_ret'] = df['close'].pct_change()

    return df


def _add_moving_averages(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add moving average indicators."""
    try:
        from ta.trend import EMAIndicator, SMAIndicator
    except ImportError:
        logger.warning("ta library not available for moving averages")
        return df

    # Simple Moving Averages
    sma_config = config.get('sma', {})
    if 'length' in sma_config:
        length = sma_config['length']
        sma = SMAIndicator(df['close'], window=length, fillna=False)
        df[f'sma_{length}'] = sma.sma_indicator()

    # Exponential Moving Averages
    ema_config = config.get('ema', {})
    if 'length' in ema_config:
        length = ema_config['length']
        ema = EMAIndicator(df['close'], window=length, fillna=False)
        df[f'ema_{length}'] = ema.ema_indicator()

    # SMA ratios
    sma_ratio_config = config.get('sma_ratio', {})
    if 'fast' in sma_ratio_config and 'slow' in sma_ratio_config:
        fast = sma_ratio_config['fast']
        slow = sma_ratio_config['slow']

        fast_sma = SMAIndicator(df['close'], window=fast, fillna=False)
        slow_sma = SMAIndicator(df['close'], window=slow, fillna=False)

        fast_ma = fast_sma.sma_indicator()
        slow_ma = slow_sma.sma_indicator()

        df[f'sma_ratio_{fast}_{slow}'] = fast_ma / slow_ma

    return df


def _add_volatility_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add volatility indicators."""
    try:
        from ta.volatility import AverageTrueRange, BollingerBands
    except ImportError:
        logger.warning("ta library not available for volatility indicators")
        return df

    # Average True Range
    atr_config = config.get('atr', {})
    if 'length' in atr_config:
        length = atr_config['length']
        atr = AverageTrueRange(
            df['high'], df['low'], df['close'],
            window=length, fillna=False
        )
        df[f'atr_{length}'] = atr.average_true_range()

    # Bollinger Bands
    bbands_config = config.get('bbands', {})
    if 'length' in bbands_config:
        length = bbands_config['length']
        std = bbands_config.get('std', 2.0)

        bbands = BollingerBands(
            df['close'], window=length, window_dev=int(std), fillna=False
        )

        df[f'bb_upper_{length}'] = bbands.bollinger_hband()
        df[f'bb_middle_{length}'] = bbands.bollinger_mavg()
        df[f'bb_lower_{length}'] = bbands.bollinger_lband()
        df[f'bb_width_{length}'] = bbands.bollinger_wband()
        df[f'bb_position_{length}'] = (df['close'] - bbands.bollinger_lband()) / bbands.bollinger_wband()

    return df


def _add_momentum_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add momentum indicators."""
    try:
        from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator
    except ImportError:
        logger.warning("ta library not available for momentum indicators")
        return df

    # RSI
    rsi_config = config.get('rsi', {})
    if 'length' in rsi_config:
        length = rsi_config['length']
        rsi = RSIIndicator(df['close'], window=length, fillna=False)
        df[f'rsi_{length}'] = rsi.rsi()

    # Rate of Change
    roc_config = config.get('roc', {})
    if 'length' in roc_config:
        length = roc_config['length']
        roc = ROCIndicator(df['close'], window=length, fillna=False)
        df[f'roc_{length}'] = roc.roc()

    # Stochastic Oscillator
    stoch_config = config.get('stochastic', {})
    if 'k' in stoch_config and 'd' in stoch_config:
        k_period = stoch_config['k']
        d_period = stoch_config['d']

        stoch = StochasticOscillator(
            df['high'], df['low'], df['close'],
            window=k_period, smooth_window=d_period, fillna=False
        )

        df[f'stoch_k_{k_period}'] = stoch.stoch()
        df[f'stoch_d_{d_period}'] = stoch.stoch_signal()

    return df


def _add_volume_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add volume-based indicators."""
    # Volume SMA
    volume_sma_config = config.get('volume_sma', {})
    if 'length' in volume_sma_config:
        length = volume_sma_config['length']
        df[f'volume_sma_{length}'] = df['volume'].rolling(window=length, min_periods=min_periods).mean()

    # Volume ratio
    volume_ratio_config = config.get('volume_ratio', {})
    if 'length' in volume_ratio_config:
        length = volume_ratio_config['length']
        volume_ma = df['volume'].rolling(window=length, min_periods=min_periods).mean()
        df[f'volume_ratio_{length}'] = df['volume'] / volume_ma

    # On-Balance Volume (simple implementation)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

    # Volume Weighted Average Price (VWAP)
    if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap'] = vwap

    return df


def _add_trend_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add trend indicators."""
    try:
        from ta.trend import ADXIndicator
    except ImportError:
        logger.warning("ta library not available for trend indicators")
        return df

    # ADX
    adx_config = config.get('adx', {})
    if 'length' in adx_config:
        length = adx_config['length']
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=length, fillna=False)
        df[f'adx_{length}'] = adx.adx()
        df[f'adx_plus_{length}'] = adx.adx_pos()
        df[f'adx_minus_{length}'] = adx.adx_neg()

    return df


def _add_price_patterns(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add price pattern indicators."""

    # Price position within recent range
    for window in [5, 10, 20]:
        if len(df) > window:
            actual_min_periods = min(min_periods, window)
            df[f'price_position_{window}'] = (
                (df['close'] - df['low'].rolling(window, min_periods=actual_min_periods).min()) /
                (df['high'].rolling(window, min_periods=actual_min_periods).max() - df['low'].rolling(window, min_periods=actual_min_periods).min())
            )

    # High-Low ratio
    for window in [5, 10, 20]:
        if len(df) > window:
            actual_min_periods = min(min_periods, window)
            df[f'hl_ratio_{window}'] = (
                df['high'].rolling(window, min_periods=actual_min_periods).max() /
                df['low'].rolling(window, min_periods=actual_min_periods).min()
            )

    return df


def _add_custom_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add custom indicators from configuration."""
    # This can be extended with user-defined indicators
    # For now, it's a placeholder for future custom indicators

    # Example: Add any custom indicators specified in config
    for indicator_name, _params in config.items():
        try:
            # This is where custom indicator logic would go
            logger.debug(f"Custom indicator '{indicator_name}' not implemented yet")
        except Exception as e:
            logger.warning(f"Failed to add custom indicator '{indicator_name}': {e}")

    return df


def _downcast_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 for memory efficiency."""
    float_columns = df.select_dtypes(include=['float64']).columns

    for col in float_columns:
        if df[col].dtype == 'float64':
            # Check if we can safely downcast to float32
            min_val = df[col].min()
            max_val = df[col].max()

            if np.finfo(np.float32).min <= min_val and max_val <= np.finfo(np.float32).max:
                df[col] = df[col].astype('float32')
                logger.debug(f"Downcasted {col} to float32")

    return df






# Enhanced indicator functions
def _add_enhanced_momentum_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add enhanced momentum indicators."""
    try:
        from ta.momentum import (
            MFIIndicator,
            ROCIndicator,
            StochasticOscillator,
            WilliamsRIndicator,
        )
    except ImportError:
        logger.warning("ta library not available for enhanced momentum indicators")
        return df

    # Williams %R
    williams_config = config.get('williams_r', {})
    if 'length' in williams_config:
        length = williams_config['length']
        williams = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=length, fillna=False)
        df[f'williams_r_{length}'] = williams.williams_r()

    # Commodity Channel Index (CCI)
    cci_config = config.get('cci', {})
    if 'length' in cci_config:
        length = cci_config['length']
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=length, min_periods=min_periods).mean()
        mean_deviation = typical_price.rolling(window=length, min_periods=min_periods).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df[f'cci_{length}'] = (typical_price - sma_tp) / (0.015 * mean_deviation)

    # Money Flow Index (MFI)
    mfi_config = config.get('mfi', {})
    if 'length' in mfi_config:
        length = mfi_config['length']
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Positive and negative money flow
        positive_mf = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_mf = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        positive_mf_sum = pd.Series(positive_mf).rolling(window=length, min_periods=min_periods).sum()
        negative_mf_sum = pd.Series(negative_mf).rolling(window=length, min_periods=min_periods).sum()

        money_ratio = positive_mf_sum / negative_mf_sum
        df[f'mfi_{length}'] = 100 - (100 / (1 + money_ratio))

    # Momentum (MTM)
    mtm_config = config.get('momentum', {})
    if 'period' in mtm_config:
        period = mtm_config['period']
        df[f'mtm_{period}'] = df['close'] - df['close'].shift(period)

    # Price Rate of Change (PROC)
    proc_config = config.get('proc', {})
    if 'period' in proc_config:
        period = proc_config['period']
        df[f'proc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    return df


def _add_enhanced_volatility_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add enhanced volatility indicators."""
    # Chaikin Volatility
    chaikin_config = config.get('chaikin_volatility', {})
    if 'ema_period' in chaikin_config and 'roc_period' in chaikin_config:
        ema_period = chaikin_config['ema_period']
        roc_period = chaikin_config['roc_period']

        high_low = df['high'] - df['low']
        ema_hl = high_low.ewm(span=ema_period).mean()
        chaikin_vol = ((ema_hl - ema_hl.shift(roc_period)) / ema_hl.shift(roc_period)) * 100
        df[f'chaikin_vol_{ema_period}_{roc_period}'] = chaikin_vol

    # Historical Volatility
    hv_config = config.get('historical_volatility', {})
    if 'window' in hv_config:
        window = hv_config['window']
        log_returns = np.log(df['close'] / df['close'].shift(1))
        historical_vol = log_returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)
        df[f'hv_{window}'] = historical_vol

    # Keltner Channels
    keltner_config = config.get('keltner_channels', {})
    if all(k in keltner_config for k in ['ema_period', 'atr_period', 'atr_multiplier']):
        ema_period = keltner_config['ema_period']
        atr_period = keltner_config['atr_period']
        atr_multiplier = keltner_config['atr_multiplier']

        # EMA of close prices
        ema_close = df['close'].ewm(span=ema_period).mean()

        # ATR for channel width
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = tr.rolling(window=atr_period, min_periods=min_periods).mean()

        # Keltner Channels
        df[f'keltner_upper_{ema_period}_{atr_period}'] = ema_close + (atr_multiplier * atr)
        df[f'keltner_lower_{ema_period}_{atr_period}'] = ema_close - (atr_multiplier * atr)
        df[f'keltner_middle_{ema_period}_{atr_period}'] = ema_close

    # Donchian Channels
    donchian_config = config.get('donchian_channels', {})
    if 'period' in donchian_config:
        period = donchian_config['period']
        df[f'donchian_upper_{period}'] = df['high'].rolling(window=period, min_periods=min_periods).max()
        df[f'donchian_lower_{period}'] = df['low'].rolling(window=period, min_periods=min_periods).min()
        df[f'donchian_middle_{period}'] = (df[f'donchian_upper_{period}'] + df[f'donchian_lower_{period}']) / 2

    return df


def _add_enhanced_trend_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add enhanced trend indicators."""
    # Triangular Moving Average (TMA)
    tma_config = config.get('tma', {})
    if 'period' in tma_config:
        period = tma_config['period']
        half_period = period // 2
        sma1 = df['close'].rolling(window=half_period, min_periods=min_periods).mean()
        df[f'tma_{period}'] = sma1.rolling(window=half_period, min_periods=min_periods).mean()

    # Weighted Moving Average (WMA)
    wma_config = config.get('wma', {})
    if 'period' in wma_config:
        period = wma_config['period']
        weights = np.arange(1, period + 1)
        def weighted_mean(x):
            return np.dot(x, weights) / weights.sum() if len(x) == period else np.nan

        df[f'wma_{period}'] = df['close'].rolling(window=period, min_periods=min_periods).apply(weighted_mean, raw=True)

    # Hull Moving Average (HMA)
    hma_config = config.get('hma', {})
    if 'period' in hma_config:
        period = hma_config['period']
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))

        # Calculate HMA: WMA(2*WMA(close, half_period) - WMA(close, period), sqrt_period)
        wma_half = df['close'].rolling(window=half_period, min_periods=min_periods).apply(
            lambda x: np.dot(x, np.arange(1, half_period + 1)) / np.arange(1, half_period + 1).sum(), raw=True
        )
        wma_full = df['close'].rolling(window=period, min_periods=min_periods).apply(
            lambda x: np.dot(x, np.arange(1, period + 1)) / np.arange(1, period + 1).sum(), raw=True
        )

        hma_series = (2 * wma_half - wma_full).rolling(window=sqrt_period, min_periods=min_periods).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_period + 1)) / np.arange(1, sqrt_period + 1).sum(), raw=True
        )
        df[f'hma_{period}'] = hma_series

    # Aroon Indicator
    aroon_config = config.get('aroon', {})
    if 'period' in aroon_config:
        period = aroon_config['period']

        # Aroon Up: periods since highest high
        aroon_up = df['high'].rolling(window=period, min_periods=min_periods).apply(
            lambda x: (period - 1 - x.argmax()) / (period - 1) * 100, raw=True
        )

        # Aroon Down: periods since lowest low
        aroon_down = df['low'].rolling(window=period, min_periods=min_periods).apply(
            lambda x: (period - 1 - x.argmin()) / (period - 1) * 100, raw=True
        )

        df[f'aroon_up_{period}'] = aroon_up
        df[f'aroon_down_{period}'] = aroon_down
        df[f'aroon_oscillator_{period}'] = aroon_up - aroon_down

    # Directional Movement Index (DMI)
    dmi_config = config.get('dmi', {})
    if 'period' in dmi_config:
        period = dmi_config['period']

        # Calculate True Range
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )

        # Directional Movement
        dm_plus = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        dm_minus = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                           np.maximum(df['low'].shift(1) - df['low'], 0), 0)

        # Smooth the values
        atr = pd.Series(tr).rolling(window=period, min_periods=min_periods).mean()
        di_plus = pd.Series(dm_plus).rolling(window=period, min_periods=min_periods).mean() / atr * 100
        di_minus = pd.Series(dm_minus).rolling(window=period, min_periods=min_periods).mean() / atr * 100

        # ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = pd.Series(dx).rolling(window=period, min_periods=min_periods).mean()

        df[f'di_plus_{period}'] = di_plus
        df[f'di_minus_{period}'] = di_minus
        df[f'adx_{period}'] = adx

    return df


def _add_enhanced_volume_indicators(df: pd.DataFrame, config: Dict[str, Any], min_periods: int) -> pd.DataFrame:
    """Add enhanced volume indicators."""
    # Accumulation/Distribution Line (ADL)
    adl_config = config.get('adl', {})
    if adl_config.get('enabled', False):
        # Calculate Money Flow Multiplier
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_multiplier = money_flow_multiplier.fillna(0)

        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * df['volume']

        # Calculate A/D Line
        df['adl'] = money_flow_volume.cumsum()

    # Volume Price Trend (VPT)
    vpt_config = config.get('vpt', {})
    if vpt_config.get('enabled', False):
        price_change = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['vpt'] = (price_change * df['volume']).cumsum()

    # Ease of Movement (EOM)
    eom_config = config.get('eom', {})
    if 'period' in eom_config:
        period = eom_config['period']

        # Distance moved
        distance_moved = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)

        # Box height (high - low)
        box_height = df['high'] - df['low']

        # Ease of Movement
        eom = distance_moved / box_height

        # 1-period SMA of EOM
        df[f'eom_{period}'] = eom.rolling(window=period, min_periods=min_periods).mean()

    # Volume Rate of Change
    volume_roc_config = config.get('volume_roc', {})
    if 'period' in volume_roc_config:
        period = volume_roc_config['period']
        df[f'volume_roc_{period}'] = ((df['volume'] - df['volume'].shift(period)) / df['volume'].shift(period)) * 100

    return df


def _add_time_based_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add time-based features."""
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not datetime, skipping time-based features")
        return df

    time_config = config.get('time_features', {})

    # Calendar features
    if time_config.get('calendar_features', True):
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

    # Cyclical features
    if time_config.get('cyclical_features', True):
        # Cyclical encoding for day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Intraday features (if data has intraday frequency)
    if time_config.get('intraday_features', True):
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        # Session detection (simplified)
        def get_session(hour):
            if 0 <= hour < 8:  # Asian session
                return 0
            elif 8 <= hour < 16:  # European session
                return 1
            else:  # US session
                return 2

        df['session'] = df.index.hour.map(get_session)

        # Time of day cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # Weekend/holiday effects
    if time_config.get('weekend_effects', True):
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    return df


class FeatureEngineer:
    """
    Enhanced feature engineering class with configurable indicators.

    This class provides a clean interface for adding features to OHLCV data
    with support for configuration-based indicator selection and validation.
    """

    def __init__(self, config):
        """
        Initialize the feature engineer.

        Args:
            config: FeatureConfig object containing indicator settings
        """
        self.config = config
        self._feature_names = None

        logger.info(f"Initialized FeatureEngineer with config: {type(config).__name__}")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features to dataframe based on configuration.

        Args:
            df: Input OHLCV dataframe

        Returns:
            DataFrame with added features
        """
        # Create indicator config from FeatureConfig
        indicator_config = self._create_indicator_config()

        # Use existing add_features function
        result = add_features(
            df=df,
            indicator_config=indicator_config,
            downcast_floats=True,
            min_periods=None
        )

        # Store feature names for later use
        self._feature_names = self._extract_main_feature_names(result)

        return result

    def _create_indicator_config(self) -> Dict[str, Dict[str, Any]]:
        """Convert FeatureConfig to indicator config format."""
        config = {}

        # Moving averages
        if self.config.enable_sma_ratios:
            config['moving_averages'] = {
                'sma_ratio': {'fast': self.config.sma_window // 2, 'slow': self.config.sma_window}
            }

        # Volatility indicators
        volatility_config = {}
        if self.config.enable_atr:
            volatility_config['atr'] = {'length': self.config.atr_window}
        if self.config.enable_bollinger_bands:
            volatility_config['bbands'] = {
                'length': self.config.bollinger_window,
                'std': self.config.bollinger_std_dev
            }
        if volatility_config:
            config['volatility'] = volatility_config

        # Momentum indicators
        momentum_config = {}
        if self.config.enable_rsi:
            momentum_config['rsi'] = {'length': self.config.rsi_window}
        if self.config.enable_roc:
            momentum_config['roc'] = {'length': self.config.roc_window}
        if self.config.enable_stochastic:
            momentum_config['stochastic'] = {
                'k': self.config.stoch_window,
                'd': self.config.stoch_smooth_window
            }
        if momentum_config:
            config['momentum'] = momentum_config

        # Volume indicators
        volume_config = {}
        if self.config.enable_volume_features:
            volume_config['volume_sma'] = {'length': self.config.volume_window}
            volume_config['volume_ratio'] = {'length': self.config.volume_window}
        if volume_config:
            config['volume'] = volume_config

        # Trend indicators
        if self.config.enable_adx:
            config['trend'] = {'adx': {'length': self.config.adx_window}}

        # Patterns (price position and HL ratio)
        if self.config.enable_price_position:
            config['patterns'] = {}

        # Enhanced indicators (check for new config attributes)
        if hasattr(self.config, 'enable_williams_r') and self.config.enable_williams_r:
            config.setdefault('enhanced_momentum', {})['williams_r'] = {'length': 14}
        if hasattr(self.config, 'enable_cci') and self.config.enable_cci:
            config.setdefault('enhanced_momentum', {})['cci'] = {'length': 20}
        if hasattr(self.config, 'enable_mfi') and self.config.enable_mfi:
            config.setdefault('enhanced_momentum', {})['mfi'] = {'length': 14}
        if hasattr(self.config, 'enable_historical_volatility') and self.config.enable_historical_volatility:
            config.setdefault('enhanced_volatility', {})['historical_volatility'] = {'window': 20}
        if hasattr(self.config, 'enable_keltner_channels') and self.config.enable_keltner_channels:
            config.setdefault('enhanced_volatility', {})['keltner_channels'] = {
                'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0
            }
        if hasattr(self.config, 'enable_aroon') and self.config.enable_aroon:
            config.setdefault('enhanced_trend', {})['aroon'] = {'period': 14}
        if hasattr(self.config, 'enable_wma') and self.config.enable_wma:
            config.setdefault('enhanced_trend', {})['wma'] = {'period': 10}
        if hasattr(self.config, 'enable_adl') and self.config.enable_adl:
            config.setdefault('enhanced_volume', {})['adl'] = {'enabled': True}

        # Time-based features
        if hasattr(self.config, 'enable_time_features') and self.config.enable_time_features:
            config['time_features'] = {
                'calendar_features': True,
                'cyclical_features': True,
                'intraday_features': True,
                'weekend_effects': True
            }

        return config

    def _extract_main_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Extract main feature names for HMM training.

        Returns feature names that closely match the original main.py
        implementation for backward compatibility.
        """
        # Try to match main.py feature names exactly
        main_features = [
            "log_ret", "atr", "roc", "rsi", "bb_width", "bb_position",
            "adx", "stoch", "sma_5_ratio", "hl_ratio", "volume_ratio"
        ]

        available_features = []
        for feature in main_features:
            # Try exact match first
            if feature in df.columns:
                available_features.append(feature)
            else:
                # Try to find similar columns
                similar_cols = [col for col in df.columns if feature in col.lower()]
                if similar_cols:
                    available_features.append(similar_cols[0])  # Take first match

        # If no features found, use all numeric columns except OHLCV
        if not available_features:
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            available_features = [
                col for col in df.columns
                if col not in ohlcv_cols and pd.api.types.is_numeric_dtype(df[col])
            ]

        logger.info(f"Selected {len(available_features)} features for HMM training")
        return available_features

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names.

        Returns:
            List of feature names for model training
        """
        if self._feature_names is None:
            raise ValueError("Features have not been computed yet. Call add_features() first.")
        return self._feature_names

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required features are present in dataframe.

        Args:
            df: DataFrame to validate

        Returns:
            True if all required features are present
        """
        if self._feature_names is None:
            raise ValueError("Features have not been computed yet. Call add_features() first.")

        missing_features = [f for f in self._feature_names if f not in df.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False

        # Check for NaN values
        features_with_nan = [f for f in self._feature_names if df[f].isnull().any()]
        if features_with_nan:
            logger.warning(f"Features contain NaN values: {features_with_nan}")

        return True

    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'log_ret') -> pd.DataFrame:
        """
        Calculate basic feature importance using correlation with target.

        Args:
            df: DataFrame with features
            target_col: Target column for importance calculation

        Returns:
            DataFrame with feature importance scores
        """
        if self._feature_names is None:
            raise ValueError("Features have not been computed yet. Call add_features() first.")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        # Calculate correlation with target
        correlations = []
        for feature in self._feature_names:
            if feature in df.columns and feature != target_col:
                corr = df[feature].corr(df[target_col])
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr) if corr is not None else 0
                })

        importance_df = pd.DataFrame(correlations)
        if not importance_df.empty:
            importance_df = importance_df.sort_values('abs_correlation', ascending=False)

        return importance_df

    def apply_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                                selection_method: str = "correlation", **selection_params) -> pd.DataFrame:
        """
        Apply automated feature selection to reduce dimensionality.

        Args:
            X: Feature matrix
            y: Target variable (optional for unsupervised methods)
            selection_method: Selection method ('correlation', 'variance', 'mutual_info', 'rfe')
            **selection_params: Additional parameters for the selection method

        Returns:
            Selected feature matrix
        """
        from .feature_selection import (
            CorrelationFeatureSelector,
            MutualInformationFeatureSelector,
            RecursiveFeatureEliminationSelector,
            VarianceFeatureSelector,
        )

        # Create selector based on method
        if selection_method == "correlation":
            selector = CorrelationFeatureSelector(**selection_params)
        elif selection_method == "variance":
            selector = VarianceFeatureSelector(**selection_params)
        elif selection_method == "mutual_info":
            selector = MutualInformationFeatureSelector(**selection_params)
        elif selection_method == "rfe":
            selector = RecursiveFeatureEliminationSelector(**selection_params)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        # Apply selection
        X_selected = selector.fit_transform(X, y)

        logger.info(f"Feature selection: {len(X.columns)} -> {len(X_selected.columns)} features")
        return X_selected

    def assess_feature_quality(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                               min_quality_score: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assess feature quality and filter low-quality features.

        Args:
            X: Feature matrix
            y: Target variable (optional)
            min_quality_score: Minimum quality score threshold

        Returns:
            Tuple of (filtered_features, quality_report)
        """
        from .feature_selection import FeatureQualityScorer

        scorer = FeatureQualityScorer()
        filtered_X, quality_report = scorer.filter_by_quality(X, min_quality_score, y)

        logger.info(f"Quality filtering: {len(X.columns)} -> {len(filtered_X.columns)} features")
        return filtered_X, quality_report

    def get_enhanced_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of all features including enhanced indicators.

        Args:
            df: DataFrame with all features

        Returns:
            Dictionary with feature summary statistics
        """
        if self._feature_names is None:
            raise ValueError("Features have not been computed yet. Call add_features() first.")

        # Categorize features
        feature_categories = {
            'basic_returns': [],
            'moving_averages': [],
            'volatility_indicators': [],
            'momentum_indicators': [],
            'volume_indicators': [],
            'trend_indicators': [],
            'enhanced_momentum': [],
            'enhanced_volatility': [],
            'enhanced_trend': [],
            'enhanced_volume': [],
            'time_features': [],
            'custom_features': []
        }

        # Categorize each feature
        for feature in self._feature_names:
            if any(x in feature for x in ['log_ret', 'simple_ret']):
                feature_categories['basic_returns'].append(feature)
            elif any(x in feature for x in ['sma_', 'ema_']):
                feature_categories['moving_averages'].append(feature)
            elif any(x in feature for x in ['atr_', 'bb_']):
                feature_categories['volatility_indicators'].append(feature)
            elif any(x in feature for x in ['rsi_', 'roc_', 'stoch_']):
                feature_categories['momentum_indicators'].append(feature)
            elif any(x in feature for x in ['volume_', 'obv', 'vwap']):
                feature_categories['volume_indicators'].append(feature)
            elif any(x in feature for x in ['adx_']):
                feature_categories['trend_indicators'].append(feature)
            elif any(x in feature for x in ['williams_r', 'cci_', 'mfi_', 'mtm_', 'proc_']):
                feature_categories['enhanced_momentum'].append(feature)
            elif any(x in feature for x in ['chaikin_vol', 'hv_', 'keltner_', 'donchian_']):
                feature_categories['enhanced_volatility'].append(feature)
            elif any(x in feature for x in ['tma_', 'wma_', 'hma_', 'aroon_', 'di_']):
                feature_categories['enhanced_trend'].append(feature)
            elif any(x in feature for x in ['adl', 'vpt', 'eom_', 'volume_roc_']):
                feature_categories['enhanced_volume'].append(feature)
            elif any(x in feature for x in ['day_of_', 'month', 'quarter', 'hour', 'session', 'is_']):
                feature_categories['time_features'].append(feature)
            else:
                feature_categories['custom_features'].append(feature)

        # Calculate statistics
        summary = {
            'total_features': len(self._feature_names),
            'categories': feature_categories,
            'category_counts': {k: len(v) for k, v in feature_categories.items()},
            'feature_statistics': self._calculate_feature_statistics(df[self._feature_names])
        }

        return summary

    def _calculate_feature_statistics(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for features."""
        stats = {
            'mean_nan_ratio': features_df.isnull().mean().mean(),
            'mean_std': features_df.std().mean(),
            'mean_range': (features_df.max() - features_df.min()).mean(),
            'high_variance_features': [],
            'low_variance_features': [],
            'high_missing_features': []
        }

        # Calculate feature-specific statistics
        for feature in features_df.columns:
            nan_ratio = features_df[feature].isnull().mean()
            feature_std = features_df[feature].std()

            if nan_ratio > 0.1:
                stats['high_missing_features'].append(feature)
            if feature_std > features_df.std().quantile(0.9):
                stats['high_variance_features'].append(feature)
            if feature_std < features_df.std().quantile(0.1):
                stats['low_variance_features'].append(feature)

        return stats
