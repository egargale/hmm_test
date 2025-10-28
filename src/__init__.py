"""
HMM Futures Analysis Package

A comprehensive package for Hidden Markov Model analysis of futures market data.

This package provides:
- Data processing and feature engineering for futures market data
- HMM model training and inference
- Regime-based backtesting and performance analysis
- Visualization and reporting tools
"""

from .backtesting import backtest_strategy, calculate_performance
from .data_processing import add_features, process_csv
from .utils import ProcessingConfig, get_logger, load_config

__version__ = "0.1.0"
__author__ = "HMM Futures Analysis Team"

# Public API
__all__ = [
    # Core functionality
    "process_csv",
    "add_features",
    "backtest_strategy",
    "calculate_performance",
    # Utilities
    "get_logger",
    "load_config",
    "ProcessingConfig",
    # Metadata
    "__version__",
    "__author__",
]
