"""
Data Processing Module

This module provides data processing capabilities for the HMM futures analysis project,
including CSV parsing, feature engineering, and data validation.
"""

from .csv_parser import process_csv

try:
    from .feature_engineering import add_features
except ImportError:
    add_features = None

try:
    from .data_validation import validate_data
except ImportError:
    validate_data = None

__all__ = ["process_csv"]

if add_features is not None:
    __all__.append("add_features")

if validate_data is not None:
    __all__.append("validate_data")