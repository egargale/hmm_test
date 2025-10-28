"""
Strategy Engine Module

Implements regime-based backtesting with HMM state-driven position allocation,
realistic trade execution, transaction cost modeling, and comprehensive trade logging.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import get_logger
from utils.data_types import BacktestConfig, BacktestResult, Trade

logger = get_logger(__name__)


@dataclass
class TradeExecution:
    """Container for trade execution details."""

    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'FLAT'
    price: float
    quantity: float
    commission: float
    slippage: float
    gross_value: float
    net_value: float


def validate_backtest_inputs(
    states: np.ndarray, prices: pd.Series, config: BacktestConfig
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
        raise ValueError(
            f"States length ({len(states)}) must match prices length ({len(prices)})"
        )

    if len(states) == 0:
        raise ValueError("Cannot backtest with empty data")

    if not isinstance(config, BacktestConfig):
        raise ValueError("Config must be a BacktestConfig instance")

    if not config.state_map:
        raise ValueError(
            "State map cannot be empty. Configure state mappings in BacktestConfig."
        )

    # Validate state map contains only valid states
    max_state = np.max(states)
    invalid_states = [state for state in config.state_map.keys() if state > max_state]
    if invalid_states:
        raise ValueError(
            f"State map contains states not present in data: {invalid_states}"
        )

    logger.debug(
        f"Validation passed: {len(states)} samples, {len(config.state_map)} state mappings"
    )


def calculate_transaction_costs(
    trade_value: float, config: BacktestConfig
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


def execute_trade(
    timestamp: datetime,
    action: str,
    price: float,
    quantity: float,
    config: BacktestConfig,
) -> TradeExecution:
    """
    Execute a trade with transaction costs.

    Args:
        timestamp: Trade timestamp
        action: Trade action ('BUY', 'SELL', 'FLAT')
        price: Execution price
        quantity: Trade quantity
        config: Backtest configuration

    Returns:
        TradeExecution: Details of the executed trade
    """
    gross_value = price * abs(quantity)
    commission, slippage = calculate_transaction_costs(gross_value, config)
    net_value = gross_value + commission + slippage

    # Determine direction for net value
    if action == "SELL":
        net_value = -net_value

    return TradeExecution(
        timestamp=timestamp,
        action=action,
        price=price,
        quantity=quantity,
        commission=commission,
        slippage=slippage,
        gross_value=gross_value,
        net_value=net_value,
    )


def backtest_strategy(
    states: np.ndarray,
    prices: pd.Series,
    config: BacktestConfig,
    use_open_prices: bool = False,
    lookahead_bias_prevention: bool = True,
) -> Tuple[pd.Series, List[Trade]]:
    """
    Backtest a regime-based trading strategy.

    This function implements a comprehensive backtesting engine that:
    1. Maps HMM states to target positions using the state_map
    2. Prevents lookahead bias by using lagged states for decisions
    3. Executes trades at realistic prices (open or next bar)
    4. Applies transaction costs (commission and slippage)
    5. Logs all trades with comprehensive details

    Args:
        states: HMM state sequence (should be lagged if bias prevention is desired)
        prices: Price series (close prices by default, open if use_open_prices=True)
        config: Backtest configuration with state mappings and costs
        use_open_prices: Whether to execute trades at open prices (next day)
        lookahead_bias_prevention: Whether to prevent lookahead bias (default True)

    Returns:
        Tuple of (positions_series, trades_list)

    Raises:
        ValueError: If inputs are invalid
    """
    logger.info(
        f"Starting backtest: {len(states)} periods, {len(config.state_map)} state mappings"
    )

    # Validate inputs
    validate_backtest_inputs(states, prices, config)

    # Initialize positions array and tracking variables
    positions = np.zeros(len(states), dtype=int)
    trades: List[Trade] = []
    current_position = 0
    current_trade = None  # Track open trade

    # Apply lookahead bias prevention by shifting states
    if lookahead_bias_prevention:
        # Decision at time t uses state at t-1
        effective_states = np.roll(states, 1)
        effective_states[0] = -1  # No state available for first period
        logger.debug("Applied lookahead bias prevention (1-period lag)")
    else:
        effective_states = states.copy()
        logger.warning(
            "Lookahead bias prevention is DISABLED - results may be unrealistic"
        )

    # Get execution prices
    if use_open_prices:
        # Assume we have access to open prices through a multi-index or separate series
        # For now, we'll use close prices with a warning
        logger.warning(
            "Open price execution not fully implemented - using close prices"
        )
        execution_prices = prices.values
    else:
        execution_prices = prices.values

    logger.debug("Starting main backtesting loop")

    # Main backtesting loop
    for i in range(1, len(states)):  # Start from 1 to use previous state
        current_state = int(effective_states[i])
        current_time = prices.index[i]
        current_price = float(execution_prices[i])

        # Determine target position based on state map
        if current_state >= 0 and current_state in config.state_map:
            target_position = int(config.state_map[current_state])
        else:
            target_position = 0  # Flat position for unmapped states

        # Check if we need to change position
        if target_position != current_position:
            # Close existing trade if we have one
            if current_trade is not None:
                # Close the trade
                close_action = "SELL" if current_position > 0 else "BUY"
                close_quantity = -current_position * config.position_size

                close_execution = execute_trade(
                    timestamp=current_time,
                    action=close_action,
                    price=current_price,
                    quantity=close_quantity,
                    config=config,
                )

                # Calculate P&L
                if current_position > 0:  # Long position
                    pnl = (current_price - current_trade.entry_price) * abs(
                        current_trade.size
                    )
                else:  # Short position
                    pnl = (current_trade.entry_price - current_price) * abs(
                        current_trade.size
                    )

                # Complete the trade object
                current_trade.exit_time = current_time
                current_trade.exit_price = current_price
                current_trade.pnl = pnl
                current_trade.commission += close_execution.commission
                current_trade.slippage += close_execution.slippage

                trades.append(current_trade)
                logger.debug(
                    f"Closed trade: P&L={pnl:.2f}, duration={(current_time - current_trade.entry_time).days} days"
                )

                current_trade = None

            # Open new trade if target position is not flat
            if target_position != 0:
                trade_action = "BUY" if target_position > 0 else "SELL"
                trade_quantity = target_position * config.position_size

                entry_execution = execute_trade(
                    timestamp=current_time,
                    action=trade_action,
                    price=current_price,
                    quantity=trade_quantity,
                    config=config,
                )

                # Create new trade
                current_trade = Trade(
                    entry_time=current_time,
                    entry_price=current_price,
                    size=trade_quantity,
                    commission=entry_execution.commission,
                    slippage=entry_execution.slippage,
                )

                logger.debug(
                    f"Opened trade: {trade_action} {abs(trade_quantity)} @ {current_price}"
                )

            # Update current position
            current_position = target_position

        # Record position
        positions[i] = current_position

    # Handle any open trade at the end
    if current_trade is not None:
        final_time = prices.index[-1]
        final_price = float(execution_prices[-1])

        # Force close final trade
        close_action = "SELL" if current_position > 0 else "BUY"
        close_quantity = -current_position * config.position_size

        close_execution = execute_trade(
            timestamp=final_time,
            action=close_action,
            price=final_price,
            quantity=close_quantity,
            config=config,
        )

        # Calculate final P&L
        if current_position > 0:
            pnl = (final_price - current_trade.entry_price) * abs(current_trade.size)
        else:
            pnl = (current_trade.entry_price - final_price) * abs(current_trade.size)

        current_trade.exit_time = final_time
        current_trade.exit_price = final_price
        current_trade.pnl = pnl
        current_trade.commission += close_execution.commission
        current_trade.slippage += close_execution.slippage

        trades.append(current_trade)
        logger.debug(f"Closed final trade: P&L={pnl:.2f}")

    # Create positions series
    positions_series = pd.Series(positions, index=prices.index, name="position")

    # Log summary statistics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
    losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
    total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
    total_costs = sum(t.commission + t.slippage for t in trades)

    logger.info("Backtest completed:")
    logger.info(f"  - Total trades: {total_trades}")
    logger.info(
        f"  - Winning trades: {winning_trades} ({winning_trades / max(total_trades, 1) * 100:.1f}%)"
    )
    logger.info(
        f"  - Losing trades: {losing_trades} ({losing_trades / max(total_trades, 1) * 100:.1f}%)"
    )
    logger.info(f"  - Total P&L: {total_pnl:.2f}")
    logger.info(f"  - Total costs: {total_costs:.2f}")
    logger.info(f"  - Net P&L: {total_pnl - total_costs:.2f}")

    return positions_series, trades


def create_equity_curve(
    trades: List[Trade], prices: pd.Series, initial_capital: float = 100000.0
) -> pd.Series:
    """
    Create equity curve from trades.

    Args:
        trades: List of executed trades
        prices: Price series for alignment
        initial_capital: Starting capital

    Returns:
        Equity curve as pandas Series
    """
    # Create a series to track equity
    equity = pd.Series(initial_capital, index=prices.index, name="equity")
    current_equity = initial_capital

    # Sort trades by entry time
    sorted_trades = sorted(trades, key=lambda x: x.entry_time)

    # Process each trade
    for trade in sorted_trades:
        if trade.pnl is not None:
            # Find the index position for the trade exit time
            if trade.exit_time in equity.index:
                current_equity += trade.pnl
                equity.loc[trade.exit_time :] = current_equity

    return equity


def analyze_positions(positions: pd.Series) -> Dict[str, Any]:
    """
    Analyze position characteristics.

    Args:
        positions: Position series

    Returns:
        Dictionary with position analysis
    """
    analysis = {}

    # Position distribution
    position_counts = positions.value_counts()
    analysis["position_distribution"] = position_counts.to_dict()
    analysis["long_percentage"] = position_counts.get(1, 0) / len(positions) * 100
    analysis["short_percentage"] = position_counts.get(-1, 0) / len(positions) * 100
    analysis["flat_percentage"] = position_counts.get(0, 0) / len(positions) * 100

    # Position changes
    position_changes = (positions.diff() != 0).sum()
    analysis["total_position_changes"] = int(position_changes)
    analysis["position_change_frequency"] = position_changes / len(positions)

    # Position persistence
    if len(positions) > 0:
        position_durations = []
        current_pos = positions.iloc[0]
        current_duration = 1

        for i in range(1, len(positions)):
            if positions.iloc[i] == current_pos:
                current_duration += 1
            else:
                if current_pos != 0:  # Only count non-flat positions
                    position_durations.append(current_duration)
                current_pos = positions.iloc[i]
                current_duration = 1

        # Add final position duration
        if current_pos != 0:
            position_durations.append(current_duration)

        if position_durations:
            analysis["avg_position_duration"] = np.mean(position_durations)
            analysis["max_position_duration"] = np.max(position_durations)
            analysis["min_position_duration"] = np.min(position_durations)

    return analysis


def backtest_with_analysis(
    states: np.ndarray,
    prices: pd.Series,
    config: BacktestConfig,
    initial_capital: float = 100000.0,
) -> BacktestResult:
    """
    Run backtest with comprehensive analysis.

    Args:
        states: HMM state sequence
        prices: Price series
        config: Backtest configuration
        initial_capital: Starting capital

    Returns:
        BacktestResult with comprehensive analysis
    """
    # Run basic backtest
    positions, trades = backtest_strategy(states, prices, config)

    # Create equity curve
    equity_curve = create_equity_curve(trades, prices, initial_capital)

    # Analyze positions
    position_analysis = analyze_positions(positions)

    # Create result
    result = BacktestResult(
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        start_date=prices.index[0],
        end_date=prices.index[-1],
    )

    logger.info("Backtest analysis completed:")
    logger.info(f"  - Initial capital: {initial_capital:.2f}")
    logger.info(f"  - Final equity: {equity_curve.iloc[-1]:.2f}")
    logger.info(
        f"  - Total return: {(equity_curve.iloc[-1] / initial_capital - 1) * 100:.2f}%"
    )
    logger.info(
        f"  - Position distribution: {position_analysis.get('position_distribution', {})}"
    )

    return result
