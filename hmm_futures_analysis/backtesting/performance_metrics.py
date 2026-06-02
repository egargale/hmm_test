"""
Performance Metrics Module

Risk-adjusted performance metrics: Sharpe ratio and drawdown analysis.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)

_ANN_FACTORS = {
    "hourly": 365 * 24,
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
}


def _infer_ann_factor(series: pd.Series) -> float:
    """Infer trading frequency from a Series' datetime index and return
    the annualization factor directly."""
    if len(series) < 2:
        return 252
    time_diffs = series.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return 252
    median_diff = time_diffs.median()
    if median_diff <= pd.Timedelta(hours=1):
        freq = "hourly"
    elif median_diff <= pd.Timedelta(days=1):
        freq = "daily"
    elif median_diff <= pd.Timedelta(weeks=1):
        freq = "weekly"
    elif median_diff <= pd.Timedelta(days=31):
        freq = "monthly"
    else:
        freq = "quarterly"
    return _ANN_FACTORS[freq]


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

    returns = equity_curve.pct_change().fillna(0)

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    if frequency is None:
        ann_factor = _infer_ann_factor(equity_curve)
    else:
        ann_factor = _ANN_FACTORS.get(frequency, 252)

    daily_risk_free = risk_free_rate / ann_factor
    excess_returns = returns - daily_risk_free

    sharpe_ratio = np.sqrt(ann_factor) * excess_returns.mean() / returns.std()

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
