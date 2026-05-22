"""
Technical Indicators Module

Provides functions for calculating technical indicators and managing
indicator configurations for financial market data.
"""

from typing import Any, Dict, List

import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


def validate_ohlcv_columns(df: pd.DataFrame) -> None:
    """
    Validate that required OHLCV columns are present in the DataFrame.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    logger.debug(f"OHLCV validation passed for DataFrame with {len(df)} rows")


def get_default_indicator_config() -> Dict[str, Dict[str, Any]]:
    """
    Get default configuration for technical indicators.

    Returns a nested dict matching the category structure expected
    by add_features() in feature_engineering.py.
    """
    return {
        "moving_averages": {
            "sma": {"length": 20},
            "ema": {"length": 20},
            "sma_ratio": {"fast": 50, "slow": 200},
        },
        "volatility": {
            "atr": {"length": 14},
            "bbands": {"length": 20, "std": 2},
        },
        "momentum": {
            "rsi": {"length": 14},
            "roc": {"length": 10},
            "stochastic": {"k": 14, "d": 3},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
        },
        "volume": {
            "volume_sma": {"length": 20},
            "volume_ratio": {"length": 20},
        },
        "trend": {
            "adx": {"length": 14},
        },
        "patterns": {},
        "enhanced_momentum": {
            "williams_r": {"length": 14},
            "cci": {"length": 20},
            "mfi": {"length": 14},
        },
        "enhanced_volatility": {
            "chaikin_vol": {"length": 10},
            "hv": {"length": 20},
        },
        "enhanced_trend": {
            "tma": {"length": 20},
            "aroon": {"length": 25},
        },
        "enhanced_volume": {
            "adl": {},
            "vpt": {},
        },
        "custom": {},
    }


def get_available_indicators() -> Dict[str, List[str]]:
    """
    Get list of available technical indicators by category.

    Returns:
        Dictionary mapping categories to indicator lists
    """
    return {
        "trend": ["sma", "ema", "macd", "adx", "bollinger_bands"],
        "momentum": ["rsi", "roc", "stochastic"],
        "volatility": ["atr", "bollinger_bands"],
        "volume": ["volume_sma", "obv"],
        "price": ["price_patterns", "custom"],
    }


def validate_indicator_config(config: Dict[str, Any]) -> bool:
    """
    Validate indicator configuration parameters.

    Args:
        config: Configuration dictionary for indicators

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(config, dict):
        logger.error("Indicator config must be a dictionary")
        return False

    for indicator, params in config.items():
        if not isinstance(params, dict):
            logger.error(f"Parameters for {indicator} must be a dictionary")
            return False

        # Common validation for length parameters
        if "length" in params:
            if not isinstance(params["length"], int) or params["length"] <= 0:
                logger.error(
                    f"Invalid length parameter for {indicator}: {params['length']}"
                )
                return False

        # Validate MACD parameters
        if indicator == "macd":
            required_params = ["fast", "slow", "signal"]
            for param in required_params:
                if (
                    param not in params
                    or not isinstance(params[param], int)
                    or params[param] <= 0
                ):
                    logger.error(f"Invalid or missing {param} parameter for MACD")
                    return False
            if params["fast"] >= params["slow"]:
                logger.error("MACD fast period must be less than slow period")
                return False

    logger.debug("Indicator configuration validation passed")
    return True
