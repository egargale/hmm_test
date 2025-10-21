"""
Backtesting Module

Implements regime-based backtesting functionality for HMM-driven trading strategies,
including position mapping, trade execution, transaction cost modeling, and performance analysis.
"""

from .strategy_engine import backtest_strategy
from .performance_analyzer import analyze_performance
from .performance_metrics import calculate_performance, infer_trading_frequency
from .bias_prevention import detect_lookahead_bias, validate_backtest_realism
from .utils import validate_backtest_inputs, calculate_transaction_costs

__all__ = [
    'backtest_strategy',
    'analyze_performance',
    'calculate_performance',
    'infer_trading_frequency',
    'detect_lookahead_bias',
    'validate_backtest_realism',
    'validate_backtest_inputs',
    'calculate_transaction_costs'
]

# Version information
__version__ = "1.0.0"