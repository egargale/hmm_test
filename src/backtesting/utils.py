"""
Backtesting Utilities Module

Provides utility functions for backtesting including input validation,
transaction cost calculations, and helper functions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import get_logger
from utils.data_types import BacktestConfig

logger = get_logger(__name__)


def validate_backtest_inputs(
    states: np.ndarray,
    prices: pd.Series,
    config: BacktestConfig
) -> None:
    """
    Validate inputs for backtesting.

    Args:
        states: HMM state sequence
        prices: Price series (typically close prices)
        config: Backtest configuration

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(states, np.ndarray):
        raise ValueError("States must be a numpy array")

    if states.ndim != 1:
        raise ValueError(f"States must be 1D array, got {states.ndim}D")

    if not isinstance(prices, pd.Series):
        raise ValueError("Prices must be a pandas Series")

    if len(states) != len(prices):
        raise ValueError(f"States length ({len(states)}) must match prices length ({len(prices)})")

    if len(states) == 0:
        raise ValueError("Cannot backtest with empty data")

    if not isinstance(config, BacktestConfig):
        raise ValueError("Config must be a BacktestConfig instance")

    if not config.state_map:
        raise ValueError("State map cannot be empty. Configure state mappings in BacktestConfig.")

    # Validate state map contains only valid states
    max_state = np.max(states)
    invalid_states = [state for state in config.state_map.keys() if state > max_state]
    if invalid_states:
        raise ValueError(f"State map contains states not present in data: {invalid_states}")

    logger.debug(f"Validation passed: {len(states)} samples, {len(config.state_map)} state mappings")


def calculate_transaction_costs(
    trade_value: float,
    config: BacktestConfig
) -> Tuple[float, float]:
    """
    Calculate transaction costs for a trade.

    Args:
        trade_value: Gross trade value (price * quantity)
        config: Backtest configuration

    Returns:
        Tuple of (commission, slippage)
    """
    # Commission (fixed per trade)
    commission = config.commission_per_trade

    # Slippage (percentage of trade value, bps = basis points = 1/100 of 1%)
    slippage = trade_value * (config.slippage_bps / 10000)

    return commission, slippage


def create_sample_price_data(
    n_samples: int = 1000,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0001,
    frequency: str = 'D'
) -> pd.Series:
    """
    Create synthetic price data for testing.

    Args:
        n_samples: Number of price observations
        initial_price: Starting price
        volatility: Price volatility (daily)
        drift: Price drift (daily)
        frequency: Pandas frequency string

    Returns:
        Price series with datetime index
    """
    np.random.seed(42)

    # Generate random returns
    returns = np.random.normal(drift, volatility, n_samples)

    # Create price series
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create datetime index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq=frequency)

    return pd.Series(prices, index=dates, name='price')


def create_sample_state_sequence(
    n_samples: int = 1000,
    n_states: int = 3,
    transition_probability: float = 0.05
) -> np.ndarray:
    """
    Create synthetic state sequence for testing.

    Args:
        n_samples: Length of state sequence
        n_states: Number of possible states
        transition_probability: Probability of state transition

    Returns:
        State sequence as numpy array
    """
    np.random.seed(42)

    states = [0]  # Start with state 0
    current_state = 0

    for _ in range(1, n_samples):
        # Decide whether to transition
        if np.random.random() < transition_probability:
            # Transition to a different state
            available_states = [s for s in range(n_states) if s != current_state]
            current_state = np.random.choice(available_states)

        states.append(current_state)

    return np.array(states)


def align_state_and_price_data(
    states: np.ndarray,
    prices: pd.Series,
    alignment_method: str = 'intersection'
) -> Tuple[np.ndarray, pd.Series]:
    """
    Align state and price data.

    Args:
        states: State sequence
        prices: Price series
        alignment_method: How to align ('intersection', 'states_to_prices', 'prices_to_states')

    Returns:
        Tuple of aligned (states, prices)
    """
    if alignment_method == 'intersection':
        # This is the default case where indices should already match
        if len(states) != len(prices):
            raise ValueError(f"Length mismatch: states={len(states)}, prices={len(prices)}")
        return states, prices

    elif alignment_method == 'states_to_prices':
        # Truncate states to match prices length
        min_length = min(len(states), len(prices))
        return states[:min_length], prices.iloc[:min_length]

    elif alignment_method == 'prices_to_states':
        # Truncate prices to match states length
        min_length = min(len(states), len(prices))
        return states[:min_length], prices.iloc[:min_length]

    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")


def calculate_position_returns(
    positions: pd.Series,
    prices: pd.Series,
    position_size: float = 1.0
) -> pd.Series:
    """
    Calculate returns from position series and price changes.

    Args:
        positions: Position series (-1, 0, 1)
        prices: Price series
        position_size: Size of each position

    Returns:
        Returns series
    """
    # Calculate price returns
    price_returns = prices.pct_change().fillna(0)

    # Apply position lag (decisions made before price moves)
    lagged_positions = positions.shift(1).fillna(0)

    # Calculate position returns
    position_returns = lagged_positions * price_returns * position_size

    return position_returns


def analyze_regime_performance(
    states: np.ndarray,
    returns: pd.Series,
    state_names: Optional[List[str]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Analyze performance by regime/state.

    Args:
        states: State sequence
        returns: Returns series
        state_names: Optional names for states

    Returns:
        Dictionary mapping states to performance metrics
    """
    if len(states) != len(returns):
        raise ValueError("States and returns must have same length")

    analysis = {}

    for state in np.unique(states):
        # Find periods where this state was active
        state_mask = states == state

        if not np.any(state_mask):
            continue

        state_returns = returns[state_mask]

        state_analysis = {
            'count': np.sum(state_mask),
            'percentage': np.sum(state_mask) / len(states) * 100,
            'mean_return': state_returns.mean(),
            'std_return': state_returns.std(),
            'sharpe_ratio': state_returns.mean() / state_returns.std() if state_returns.std() > 0 else 0,
            'total_return': (1 + state_returns).prod() - 1,
            'max_return': state_returns.max(),
            'min_return': state_returns.min(),
            'positive_return_periods': np.sum(state_returns > 0),
            'negative_return_periods': np.sum(state_returns < 0),
            'win_rate': np.sum(state_returns > 0) / len(state_returns) if len(state_returns) > 0 else 0
        }

        # Add state name if provided
        if state_names and state < len(state_names):
            state_analysis['state_name'] = state_names[state]

        analysis[state] = state_analysis

    return analysis


def calculate_rolling_regime_metrics(
    states: np.ndarray,
    returns: pd.Series,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling metrics by regime.

    Args:
        states: State sequence
        returns: Returns series
        window: Rolling window size

    Returns:
        DataFrame with rolling regime metrics
    """
    if len(states) != len(returns):
        raise ValueError("States and returns must have same length")

    # Create DataFrame with states and returns
    df = pd.DataFrame({
        'state': states,
        'return': returns
    })

    rolling_metrics = pd.DataFrame(index=returns.index)

    # Calculate rolling metrics for each state
    for state in np.unique(states):
        state_mask = df['state'] == state
        state_returns = df['return'].where(state_mask)

        rolling_metrics[f'state_{state}_count'] = state_mask.rolling(window).sum()
        rolling_metrics[f'state_{state}_mean_return'] = state_returns.rolling(window).mean()
        rolling_metrics[f'state_{state}_volatility'] = state_returns.rolling(window).std()

    return rolling_metrics


def create_trade_log_dataframe(trades: List) -> pd.DataFrame:
    """
    Convert list of Trade objects to pandas DataFrame.

    Args:
        trades: List of Trade objects

    Returns:
        DataFrame with trade information
    """
    if not trades:
        return pd.DataFrame()

    trade_data = []
    for trade in trades:
        trade_data.append({
            'entry_time': trade.entry_time,
            'entry_price': trade.entry_price,
            'exit_time': trade.exit_time,
            'exit_price': trade.exit_price,
            'size': trade.size,
            'pnl': trade.pnl,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'duration_days': (trade.exit_time - trade.entry_time).days if trade.exit_time else None,
            'direction': 'LONG' if trade.size > 0 else 'SHORT' if trade.size < 0 else 'FLAT'
        })

    return pd.DataFrame(trade_data)


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns from equity curve.

    Args:
        equity_curve: Equity curve series

    Returns:
        DataFrame with monthly returns
    """
    # Resample to month end
    monthly_equity = equity_curve.resample('M').last()

    # Calculate monthly returns
    monthly_returns = monthly_equity.pct_change().fillna(0)

    # Create DataFrame with additional information
    monthly_df = pd.DataFrame({
        'equity': monthly_equity,
        'return': monthly_returns,
        'cumulative_return': (monthly_equity / monthly_equity.iloc[0] - 1)
    })

    return monthly_df


def calculate_correlation_matrix(
    returns_dict: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple return series.

    Args:
        returns_dict: Dictionary mapping names to return series

    Returns:
        Correlation matrix DataFrame
    """
    # Align all return series
    aligned_returns = pd.DataFrame(returns_dict)

    # Calculate correlation matrix
    correlation_matrix = aligned_returns.corr()

    return correlation_matrix
