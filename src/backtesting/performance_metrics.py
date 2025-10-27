"""
Performance Metrics Module

Implements comprehensive risk-adjusted performance metrics calculation for
strategy evaluation, including returns, volatility, Sharpe ratio, and drawdown analysis.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils import get_logger
from utils.data_types import PerformanceMetrics

logger = get_logger(__name__)


def infer_trading_frequency(equity_curve: pd.Series) -> str:
    """
    Infer the trading frequency from the equity curve index.

    Args:
        equity_curve: Equity curve series with datetime index

    Returns:
        Inferred frequency ('daily', 'weekly', 'monthly', 'hourly', etc.)
    """
    if len(equity_curve) < 2:
        return 'daily'  # Default assumption

    # Calculate time differences
    time_diffs = equity_curve.index.to_series().diff().dropna()

    if len(time_diffs) == 0:
        return 'daily'

    # Get the most common time difference
    median_diff = time_diffs.median()

    # Determine frequency based on time difference
    if median_diff <= pd.Timedelta(hours=1):
        return 'hourly'
    elif median_diff <= pd.Timedelta(days=1):
        return 'daily'
    elif median_diff <= pd.Timedelta(weeks=1):
        return 'weekly'
    elif median_diff <= pd.Timedelta(days=31):
        return 'monthly'
    else:
        return 'quarterly'


def get_annualization_factor(frequency: str) -> float:
    """
    Get the annualization factor for a given trading frequency.

    Args:
        frequency: Trading frequency string

    Returns:
        Annualization factor
    """
    frequency_factors = {
        'hourly': 365 * 24,      # 8760 hours per year
        'daily': 252,           # 252 trading days per year
        'weekly': 52,           # 52 weeks per year
        'monthly': 12,          # 12 months per year
        'quarterly': 4,         # 4 quarters per year
        'annual': 1,            # 1 year per year
    }

    return frequency_factors.get(frequency, 252)  # Default to daily


def calculate_returns(equity_curve: pd.Series, frequency: Optional[str] = None) -> pd.Series:
    """
    Calculate returns from equity curve.

    Args:
        equity_curve: Equity curve series
        frequency: Trading frequency (inferred if None)

    Returns:
        Returns series
    """
    if frequency is None:
        frequency = infer_trading_frequency(equity_curve)
        logger.debug(f"Inferred trading frequency: {frequency}")

    returns = equity_curve.pct_change().fillna(0)

    logger.debug(f"Calculated returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
    return returns


def calculate_annualized_return(
    equity_curve: pd.Series,
    frequency: Optional[str] = None
) -> float:
    """
    Calculate compound annualized return (CAGR).

    Args:
        equity_curve: Equity curve series
        frequency: Trading frequency (inferred if None)

    Returns:
        Annualized return as decimal
    """
    if len(equity_curve) < 2:
        return 0.0

    if frequency is None:
        frequency = infer_trading_frequency(equity_curve)

    # Calculate total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Calculate time period in years
    time_diff = equity_curve.index[-1] - equity_curve.index[0]
    years = time_diff.total_seconds() / (365.25 * 24 * 3600)

    if years <= 0:
        return 0.0

    # Calculate CAGR
    annualized_return = (1 + total_return) ** (1 / years) - 1

    logger.debug(f"CAGR: {annualized_return:.4f} ({years:.2f} years)")
    return annualized_return


def calculate_annualized_volatility(
    returns: pd.Series,
    frequency: Optional[str] = None
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Returns series
        frequency: Trading frequency (inferred if None)

    Returns:
        Annualized volatility as decimal
    """
    if len(returns) == 0:
        return 0.0

    if frequency is None:
        # Try to infer frequency from returns index
        frequency = infer_trading_frequency(returns)

    annualization_factor = get_annualization_factor(frequency)

    # Calculate annualized volatility
    annualized_volatility = returns.std() * np.sqrt(annualization_factor)

    logger.debug(f"Annualized volatility: {annualized_volatility:.4f} (factor: {annualization_factor})")
    return annualized_volatility


def calculate_sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    frequency: Optional[str] = None
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        equity_curve: Equity curve series
        risk_free_rate: Annual risk-free rate
        frequency: Trading frequency (inferred if None)

    Returns:
        Sharpe ratio
    """
    if len(equity_curve) < 2:
        return 0.0

    if frequency is None:
        frequency = infer_trading_frequency(equity_curve)

    # Calculate returns
    returns = calculate_returns(equity_curve, frequency)

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Get annualization factor
    annualization_factor = get_annualization_factor(frequency)

    # Calculate excess returns
    daily_risk_free = risk_free_rate / annualization_factor
    excess_returns = returns - daily_risk_free

    # Calculate Sharpe ratio
    sharpe_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / returns.std()

    logger.debug(f"Sharpe ratio: {sharpe_ratio:.4f}")
    return sharpe_ratio


def calculate_drawdown_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate drawdown metrics.

    Args:
        equity_curve: Equity curve series

    Returns:
        Dictionary with drawdown metrics
    """
    if len(equity_curve) < 2:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'avg_drawdown': 0.0,
            'drawdown_recovery_time': 0.0
        }

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown
    max_drawdown = drawdown.min()

    # Maximum drawdown duration
    in_drawdown = drawdown < 0
    drawdown_duration = pd.Series(0, index=equity_curve.index)
    current_duration = 0

    for i in range(len(drawdown)):
        if in_drawdown.iloc[i]:
            current_duration += 1
        else:
            current_duration = 0
        drawdown_duration.iloc[i] = current_duration

    max_drawdown_duration = drawdown_duration.max()

    # Average drawdown (only for drawdown periods)
    drawdown_periods = drawdown[in_drawdown]
    avg_drawdown = drawdown_periods.mean() if len(drawdown_periods) > 0 else 0.0

    # Average recovery time
    recovery_times = []
    in_drawdown_period = False
    drawdown_start = None

    for i in range(1, len(drawdown)):
        if drawdown.iloc[i-1] < 0 and drawdown.iloc[i] >= 0:
            # End of drawdown period
            if drawdown_start is not None:
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
            in_drawdown_period = False
        elif drawdown.iloc[i] < 0 and not in_drawdown_period:
            # Start of drawdown period
            drawdown_start = i
            in_drawdown_period = True

    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0

    drawdown_metrics = {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': int(max_drawdown_duration),
        'avg_drawdown': avg_drawdown,
        'avg_recovery_time': avg_recovery_time
    }

    logger.debug(f"Drawdown metrics: max={max_drawdown:.4f}, duration={max_drawdown_duration}")
    return drawdown_metrics


def calculate_calmar_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Calmar ratio (annualized return / absolute max drawdown).

    Args:
        equity_curve: Equity curve series
        risk_free_rate: Annual risk-free rate

    Returns:
        Calmar ratio
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate annualized return
    annualized_return = calculate_annualized_return(equity_curve)

    # Calculate maximum drawdown
    drawdown_metrics = calculate_drawdown_metrics(equity_curve)
    max_drawdown = abs(drawdown_metrics['max_drawdown'])

    if max_drawdown == 0:
        return 0.0 if annualized_return == 0 else float('inf')

    calmar_ratio = annualized_return / max_drawdown

    logger.debug(f"Calmar ratio: {calmar_ratio:.4f}")
    return calmar_ratio


def calculate_performance(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    benchmark_curve: Optional[pd.Series] = None,
    frequency: Optional[str] = None
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Equity curve series
        risk_free_rate: Annual risk-free rate
        benchmark_curve: Optional benchmark equity curve
        frequency: Trading frequency (inferred if None)

    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    logger.info("Starting comprehensive performance analysis")

    if len(equity_curve) == 0:
        logger.warning("Empty equity curve provided")
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0
        )

    # Infer frequency if not provided
    if frequency is None:
        frequency = infer_trading_frequency(equity_curve)
        logger.info(f"Inferred trading frequency: {frequency}")

    # Calculate basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annualized_return = calculate_annualized_return(equity_curve, frequency)

    # Calculate returns and volatility
    returns = calculate_returns(equity_curve, frequency)
    annualized_volatility = calculate_annualized_volatility(returns, frequency)

    # Calculate risk-adjusted metrics
    sharpe_ratio = calculate_sharpe_ratio(equity_curve, risk_free_rate, frequency)

    # Calculate drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(equity_curve)
    max_drawdown = drawdown_metrics['max_drawdown']
    max_drawdown_duration = drawdown_metrics['max_drawdown_duration']

    # Calculate Calmar ratio
    calmar_ratio = calculate_calmar_ratio(equity_curve, risk_free_rate)

    # Initialize metrics with basic values
    metrics = PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        calmar_ratio=calmar_ratio
    )

    logger.info("Core performance metrics calculated:")
    logger.info(f"  - Total return: {total_return:.2%}")
    logger.info(f"  - Annualized return: {annualized_return:.2%}")
    logger.info(f"  - Annualized volatility: {annualized_volatility:.2%}")
    logger.info(f"  - Sharpe ratio: {sharpe_ratio:.2f}")
    logger.info(f"  - Max drawdown: {max_drawdown:.2%}")
    logger.info(f"  - Calmar ratio: {calmar_ratio:.2f}")

    return metrics


def validate_performance_metrics(metrics: PerformanceMetrics) -> Dict[str, Any]:
    """
    Validate calculated performance metrics for reasonableness.

    Args:
        metrics: Performance metrics to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }

    # Check for impossible values
    if metrics.annualized_volatility < 0:
        validation_results['errors'].append("Annualized volatility cannot be negative")
        validation_results['valid'] = False

    if metrics.max_drawdown > 0:
        validation_results['errors'].append("Maximum drawdown should be negative or zero")
        validation_results['valid'] = False

    if metrics.max_drawdown_duration < 0:
        validation_results['errors'].append("Maximum drawdown duration cannot be negative")
        validation_results['valid'] = False

    # Check for suspicious values
    if abs(metrics.annualized_return) > 10:  # > 1000% annual return
        validation_results['warnings'].append("Extremely high annualized return detected")

    if metrics.annualized_volatility > 5:  # > 500% annual volatility
        validation_results['warnings'].append("Extremely high volatility detected")

    if abs(metrics.sharpe_ratio) > 10:
        validation_results['warnings'].append("Extremely high Sharpe ratio detected")

    # Check for reasonable relationships
    if metrics.calmar_ratio != 0 and metrics.annualized_return != 0:
        expected_calmar = metrics.annualized_return / abs(metrics.max_drawdown)
        if abs(metrics.calmar_ratio - expected_calmar) > 0.1:
            validation_results['warnings'].append("Calmar ratio inconsistency detected")

    return validation_results


def create_performance_summary(metrics: PerformanceMetrics) -> str:
    """
    Create a human-readable performance summary.

    Args:
        metrics: Performance metrics

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=== Performance Summary ===")
    summary.append(f"Total Return:          {metrics.total_return:.2%}")
    summary.append(f"Annualized Return:     {metrics.annualized_return:.2%}")
    summary.append(f"Annualized Volatility: {metrics.annualized_volatility:.2%}")
    summary.append(f"Sharpe Ratio:          {metrics.sharpe_ratio:.2f}")
    summary.append(f"Maximum Drawdown:      {metrics.max_drawdown:.2%}")
    summary.append(f"Max Drawdown Duration: {metrics.max_drawdown_duration} periods")

    if metrics.calmar_ratio is not None:
        summary.append(f"Calmar Ratio:          {metrics.calmar_ratio:.2f}")

    if metrics.win_rate is not None:
        summary.append(f"Win Rate:              {metrics.win_rate:.2%}")

    if metrics.profit_factor is not None:
        summary.append(f"Profit Factor:         {metrics.profit_factor:.2f}")

    if metrics.sortino_ratio is not None:
        summary.append(f"Sortino Ratio:         {metrics.sortino_ratio:.2f}")

    return "\n".join(summary)
