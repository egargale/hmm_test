"""
Data Types Module

Defines core data structures and type hints for the HMM futures analysis project
using Python's dataclasses with support for numpy and pandas types.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class FuturesData:
    """Data class for futures OHLCV data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        """Validate data after initialization."""
        if self.high < max(self.open, self.close):
            raise ValueError("High price cannot be lower than open or close")
        if self.low > min(self.open, self.close):
            raise ValueError("Low price cannot be higher than open or close")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass
class HMMState:
    """Data class for HMM state information."""
    state_id: int
    state_name: Optional[str] = None
    probability: Optional[float] = None
    mean_return: Optional[float] = None
    volatility: Optional[float] = None

    def __post_init__(self):
        """Validate state data after initialization."""
        if self.state_id < 0:
            raise ValueError("State ID must be non-negative")
        if self.probability is not None and not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")


@dataclass
class Trade:
    """Data class for individual trades."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 1.0
    pnl: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0

    def __post_init__(self):
        """Validate trade data after initialization."""
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.size == 0:
            raise ValueError("Trade size cannot be zero")
        if self.exit_price is not None and self.exit_price <= 0:
            raise ValueError("Exit price must be positive")


@dataclass
class BacktestResult:
    """Data class for backtest results."""
    equity_curve: pd.Series
    positions: pd.Series
    trades: List[Trade]
    start_date: datetime
    end_date: datetime

    def __post_init__(self):
        """Validate backtest results after initialization."""
        if len(self.equity_curve) != len(self.positions):
            raise ValueError("Equity curve and positions must have same length")
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if len(self.equity_curve) == 0:
            raise ValueError("Equity curve cannot be empty")


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    loss_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    sortino_ratio: Optional[float] = None

    def __post_init__(self):
        """Validate performance metrics after initialization."""
        if self.annualized_volatility < 0:
            raise ValueError("Volatility cannot be negative")
        if self.max_drawdown > 0:
            raise ValueError("Max drawdown should be negative or zero")
        if self.max_drawdown_duration < 0:
            raise ValueError("Drawdown duration cannot be negative")


@dataclass
class BacktestConfig:
    """Data class for backtest configuration."""
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 0.0
    position_size: float = 1.0
    state_map: Optional[Dict[int, int]] = None

    def __post_init__(self):
        """Validate backtest configuration after initialization."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission_per_trade < 0:
            raise ValueError("Commission cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("Slippage cannot be negative")
        if self.position_size == 0:
            raise ValueError("Position size cannot be zero")

        # Initialize empty state map if not provided
        if self.state_map is None:
            self.state_map = {}


# Type aliases for better readability
PriceData = pd.DataFrame  # Expected columns: open, high, low, close, volume
FeatureMatrix = np.ndarray
StateSequence = np.ndarray
ProbabilityMatrix = np.ndarray