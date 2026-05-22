"""
Backtesting Module

Implements regime-based backtesting functionality for HMM-driven trading strategies,
including performance analysis and utility functions.
"""

from .bias_prevention import detect_lookahead_bias, validate_backtest_realism
from .performance_analyzer import analyze_performance
from .performance_metrics import calculate_performance, infer_trading_frequency
from .utils import calculate_transaction_costs, validate_backtest_inputs

__all__ = [
    "analyze_performance",
    "calculate_performance",
    "infer_trading_frequency",
    "detect_lookahead_bias",
    "validate_backtest_realism",
    "validate_backtest_inputs",
    "calculate_transaction_costs",
]

# Version information
__version__ = "1.1.0"
