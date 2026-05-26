"""
Backtesting Module

Implements regime-based backtesting functionality for HMM-driven trading strategies.
"""

from .performance_metrics import calculate_performance, infer_trading_frequency

__all__ = [
    "calculate_performance",
    "infer_trading_frequency",
]
