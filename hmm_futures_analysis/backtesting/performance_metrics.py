"""
Performance Metrics Module

Implements comprehensive risk-adjusted performance metrics calculation for
strategy evaluation, including returns, volatility, Sharpe ratio, and drawdown analysis.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def _infer_frequency(series: pd.Series) -> str:
    """Infer trading frequency from a Series' datetime index."""
    if len(series) < 2:
        return "daily"
    time_diffs = series.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return "daily"
    median_diff = time_diffs.median()
    if median_diff <= pd.Timedelta(hours=1):
        return "hourly"
    if median_diff <= pd.Timedelta(days=1):
        return "daily"
    if median_diff <= pd.Timedelta(weeks=1):
        return "weekly"
    if median_diff <= pd.Timedelta(days=31):
        return "monthly"
    return "quarterly"


def get_annualization_factor(frequency: str) -> float:
    """
    Get the annualization factor for a given trading frequency.

    Args:
        frequency: Trading frequency string

    Returns:
        Annualization factor
    """
    frequency_factors = {
        "hourly": 365 * 24,  # 8760 hours per year
        "daily": 252,  # 252 trading days per year
        "weekly": 52,  # 52 weeks per year
        "monthly": 12,  # 12 months per year
        "quarterly": 4,  # 4 quarters per year
        "annual": 1,  # 1 year per year
    }

    return frequency_factors.get(frequency, 252)  # Default to daily


def calculate_returns(
    equity_curve: pd.Series, frequency: Optional[str] = None
) -> pd.Series:
    """
    Calculate returns from equity curve.

    Args:
        equity_curve: Equity curve series
        frequency: Trading frequency (inferred if None)

    Returns:
        Returns series
    """
    if frequency is None:
        frequency = _infer_frequency(equity_curve)
        logger.debug(f"Inferred trading frequency: {frequency}")

    returns = equity_curve.pct_change().fillna(0)

    logger.debug(
        f"Calculated returns: mean={returns.mean():.6f}, std={returns.std():.6f}"
    )
    return returns


def calculate_annualized_return(
    equity_curve: pd.Series, frequency: Optional[str] = None
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
        frequency = _infer_frequency(equity_curve)

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
    returns: pd.Series, frequency: Optional[str] = None
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
        frequency = _infer_frequency(returns)

    annualization_factor = get_annualization_factor(frequency)

    # Calculate annualized volatility
    annualized_volatility = returns.std() * np.sqrt(annualization_factor)

    logger.debug(
        f"Annualized volatility: {annualized_volatility:.4f} (factor: {annualization_factor})"
    )
    return annualized_volatility


def calculate_sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    frequency: Optional[str] = None,
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
        frequency = _infer_frequency(equity_curve)

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
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "avg_drawdown": 0.0,
            "drawdown_recovery_time": 0.0,
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
        if drawdown.iloc[i - 1] < 0 and drawdown.iloc[i] >= 0:
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
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": int(max_drawdown_duration),
        "avg_drawdown": avg_drawdown,
        "avg_recovery_time": avg_recovery_time,
    }

    logger.debug(
        f"Drawdown metrics: max={max_drawdown:.4f}, duration={max_drawdown_duration}"
    )
    return drawdown_metrics


def calculate_calmar_ratio(
    equity_curve: pd.Series, risk_free_rate: float = 0.02
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
    max_drawdown = abs(drawdown_metrics["max_drawdown"])

    if max_drawdown == 0:
        return 0.0 if annualized_return == 0 else float("inf")

    calmar_ratio = annualized_return / max_drawdown

    logger.debug(f"Calmar ratio: {calmar_ratio:.4f}")
    return calmar_ratio
