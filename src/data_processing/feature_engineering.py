"""
Feature Engineering Module

Computes technical indicators and features for OHLCV data using various libraries
including pandas_ta, with support for memory-efficient processing and configurable
indicator parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings

from utils import get_logger

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
    _validate_ohlcv_columns(df)

    # Use default indicator config if not provided
    if indicator_config is None:
        indicator_config = _get_default_indicator_config()

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

    # 8. Custom indicators from config
    df = _add_custom_indicators(df, indicator_config.get('custom', {}), min_periods)
    custom_features = [col for col in df.columns if col not in df.columns[:len(df.columns) - len(features_added)]]
    features_added.extend(custom_features)

    # Downcast float columns for memory efficiency
    if downcast_floats:
        df = _downcast_float_columns(df)

    # Log feature addition summary
    logger.info(f"Added {len(features_added)} features: {features_added[:10]}...")
    logger.info(f"Total columns: {len(df.columns)}")

    return df


def _validate_ohlcv_columns(df: pd.DataFrame) -> None:
    """Validate that required OHLCV columns exist."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")


def _get_default_indicator_config() -> Dict[str, Dict[str, Any]]:
    """Get default indicator configuration."""
    return {
        'moving_averages': {
            'sma': {'length': 20},
            'ema': {'length': 20},
            'sma_ratio': {'fast': 10, 'slow': 20}
        },
        'volatility': {
            'atr': {'length': 14},
            'bbands': {'length': 20, 'std': 2.0}
        },
        'momentum': {
            'rsi': {'length': 14},
            'roc': {'length': 10},
            'stochastic': {'k': 14, 'd': 3}
        },
        'volume': {
            'volume_sma': {'length': 20},
            'volume_ratio': {'length': 20}
        },
        'trend': {
            'adx': {'length': 14}
        },
        'patterns': {},
        'custom': {}
    }


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
        from ta.trend import SMAIndicator, EMAIndicator
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
        from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
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
    for indicator_name, params in config.items():
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


def get_available_indicators() -> Dict[str, List[str]]:
    """
    Get list of available indicators by category.

    Returns:
        Dict mapping categories to lists of indicator names
    """
    return {
        'moving_averages': ['sma', 'ema', 'sma_ratio'],
        'volatility': ['atr', 'bbands'],
        'momentum': ['rsi', 'roc', 'stochastic'],
        'volume': ['volume_sma', 'volume_ratio', 'obv', 'vwap'],
        'trend': ['adx'],
        'patterns': ['price_position', 'hl_ratio']
    }


def validate_indicator_config(config: Dict[str, Any]) -> bool:
    """
    Validate indicator configuration.

    Args:
        config: Indicator configuration dictionary

    Returns:
        bool: True if configuration is valid
    """
    available_indicators = get_available_indicators()
    all_available = set()
    for indicators in available_indicators.values():
        all_available.update(indicators)

    # Check if all configured indicators are available
    for category, indicators in config.items():
        if category not in available_indicators:
            logger.warning(f"Unknown indicator category: {category}")
            return False

        for indicator in indicators:
            if indicator not in all_available and indicator != 'custom':
                logger.warning(f"Unknown indicator: {indicator}")
                return False

    return True