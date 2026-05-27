"""Data processing for HMM regime detection."""

from .csv_parser import process_csv
from .data_validation import validate_data
from .csv_auto_detect import load_from_csv, load_from_yfinance, load_price_series, load_prices

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

try:
    from .csv_format_detector import CSVFormat, CSVFormatDetector, DetectionResult
except ImportError:
    CSVFormatDetector = None
    CSVFormat = None
    DetectionResult = None

__all__ = [
    "process_csv",
    "validate_data",
    "load_from_csv",
    "load_prices",
]

if add_features is not None:
    __all__.append("add_features")

if get_default_indicator_config is not None:
    __all__.extend([
        "get_default_indicator_config",
        "get_available_indicators",
        "validate_indicator_config",
        "validate_ohlcv_columns",
    ])

if CSVFormatDetector is not None:
    __all__.extend(["CSVFormatDetector", "CSVFormat", "DetectionResult"])
