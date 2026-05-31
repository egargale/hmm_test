"""Data processing for HMM regime detection."""

from .csv_auto_detect import load_from_csv, load_prices

try:
    from .feature_engineering import add_features
except ImportError:
    add_features = None

try:
    from .technical_indicators import (
        get_available_indicators,
        get_default_indicator_config,
        validate_indicator_config,
        validate_ohlcv_columns,
    )
except ImportError:
    get_default_indicator_config = None
    get_available_indicators = None
    validate_indicator_config = None
    validate_ohlcv_columns = None

__all__ = [
    "load_from_csv",
    "load_prices",
]

if add_features is not None:
    __all__.append("add_features")

if get_default_indicator_config is not None:
    __all__.extend(
        [
            "get_default_indicator_config",
            "get_available_indicators",
            "validate_indicator_config",
            "validate_ohlcv_columns",
        ]
    )
