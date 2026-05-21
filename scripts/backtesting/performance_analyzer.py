"""
Performance Analyzer Module

Implements comprehensive performance analysis for backtesting results,
including risk metrics, drawdown analysis, and statistical evaluation.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from utils import get_logger
from utils.data_types import BacktestResult, PerformanceMetrics

from .performance_metrics import calculate_performance, calculate_returns


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown from equity curve.

    Args:
        equity_curve: Equity curve series

    Returns:
        Drawdown series (negative values represent drawdown)
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown


logger = get_logger(__name__)


def analyze_performance(
    result: BacktestResult, risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    Analyze backtest performance and calculate comprehensive metrics.

    Args:
        result: Backtest results
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation

    Returns:
        PerformanceMetrics: Comprehensive performance analysis
    """
    logger.info("Starting comprehensive performance analysis")

    # Use the new core performance metrics calculation
    core_metrics = calculate_performance(
        equity_curve=result.equity_curve, risk_free_rate=risk_free_rate
    )

    # Extract trade-based metrics
    trades = result.trades
    if trades:
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0
        loss_rate = len(losing_trades) / len(trades) if trades else 0

        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Sortino ratio (downside deviation)
        returns = calculate_returns(result.equity_curve)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        excess_returns = returns - risk_free_rate / 252
        sortino_ratio = (
            excess_returns.mean() * np.sqrt(252) / downside_deviation
            if downside_deviation > 0
            else 0
        )
    else:
        win_rate = loss_rate = profit_factor = sortino_ratio = 0

    # Create complete metrics with trade-based information
    complete_metrics = PerformanceMetrics(
        total_return=core_metrics.total_return,
        annualized_return=core_metrics.annualized_return,
        annualized_volatility=core_metrics.annualized_volatility,
        sharpe_ratio=core_metrics.sharpe_ratio,
        max_drawdown=core_metrics.max_drawdown,
        max_drawdown_duration=core_metrics.max_drawdown_duration,
        calmar_ratio=core_metrics.calmar_ratio,
        win_rate=win_rate,
        loss_rate=loss_rate,
        profit_factor=profit_factor,
        sortino_ratio=sortino_ratio,
    )

    logger.info("Comprehensive performance analysis completed:")
    logger.info(f"  - Total return: {complete_metrics.total_return:.2%}")
    logger.info(f"  - Annualized return: {complete_metrics.annualized_return:.2%}")
    logger.info(f"  - Sharpe ratio: {complete_metrics.sharpe_ratio:.2f}")
    logger.info(f"  - Sortino ratio: {complete_metrics.sortino_ratio:.2f}")
    logger.info(f"  - Max drawdown: {complete_metrics.max_drawdown:.2%}")
    logger.info(f"  - Calmar ratio: {complete_metrics.calmar_ratio:.2f}")
    logger.info(f"  - Win rate: {complete_metrics.win_rate:.2%}")
    logger.info(f"  - Profit factor: {complete_metrics.profit_factor:.2f}")

    return complete_metrics


def calculate_rolling_metrics(
    equity_curve: pd.Series, window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        equity_curve: Equity curve series
        window: Rolling window size (default 252 trading days)

    Returns:
        DataFrame with rolling metrics
    """
    returns = calculate_returns(equity_curve)

    rolling_metrics = pd.DataFrame(index=equity_curve.index)

    # Rolling returns
    rolling_metrics["rolling_return"] = equity_curve.pct_change(window)

    # Rolling volatility
    rolling_metrics["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(252)

    # Rolling Sharpe ratio
    rolling_excess_returns = returns - 0.02 / 252  # Assuming 2% risk-free rate
    rolling_metrics["rolling_sharpe"] = (
        rolling_excess_returns.rolling(window).mean()
        / returns.rolling(window).std()
        * np.sqrt(252)
    )

    # Rolling max drawdown
    rolling_max = equity_curve.rolling(window).max()
    rolling_drawdown = (equity_curve - rolling_max) / rolling_max
    rolling_metrics["rolling_max_drawdown"] = rolling_drawdown.rolling(window).min()

    return rolling_metrics


def analyze_trade_distribution(trades: list) -> Dict[str, Any]:
    """
    Analyze the distribution of trade outcomes.

    Args:
        trades: List of Trade objects

    Returns:
        Dictionary with trade distribution analysis
    """
    if not trades:
        return {"error": "No trades to analyze"}

    trade_pnls = [t.pnl for t in trades if t.pnl is not None]

    if not trade_pnls:
        return {"error": "No trades with P&L data"}

    analysis = {}

    # Basic statistics
    analysis["total_trades"] = len(trades)
    analysis["positive_trades"] = len([pnl for pnl in trade_pnls if pnl > 0])
    analysis["negative_trades"] = len([pnl for pnl in trade_pnls if pnl < 0])
    analysis["breakeven_trades"] = len([pnl for pnl in trade_pnls if pnl == 0])

    # P&L statistics
    analysis["total_pnl"] = sum(trade_pnls)
    analysis["avg_pnl"] = np.mean(trade_pnls)
    analysis["median_pnl"] = np.median(trade_pnls)
    analysis["std_pnl"] = np.std(trade_pnls)
    analysis["max_profit"] = max(trade_pnls)
    analysis["max_loss"] = min(trade_pnls)

    # Profit/loss analysis
    positive_pnls = [pnl for pnl in trade_pnls if pnl > 0]
    negative_pnls = [pnl for pnl in trade_pnls if pnl < 0]

    if positive_pnls:
        analysis["avg_profit"] = np.mean(positive_pnls)
        analysis["median_profit"] = np.median(positive_pnls)
        analysis["max_profit"] = max(positive_pnls)

    if negative_pnls:
        analysis["avg_loss"] = np.mean(negative_pnls)
        analysis["median_loss"] = np.median(negative_pnls)
        analysis["max_loss"] = min(negative_pnls)

    # Risk-reward ratios
    if positive_pnls and negative_pnls:
        analysis["profit_factor"] = sum(positive_pnls) / abs(sum(negative_pnls))
        analysis["avg_win_loss_ratio"] = analysis["avg_profit"] / abs(
            analysis["avg_loss"]
        )

    # Trade duration analysis
    trade_durations = []
    for trade in trades:
        if trade.entry_time and trade.exit_time:
            duration = (trade.exit_time - trade.entry_time).days
            trade_durations.append(duration)

    if trade_durations:
        analysis["avg_trade_duration_days"] = np.mean(trade_durations)
        analysis["median_trade_duration_days"] = np.median(trade_durations)
        analysis["max_trade_duration_days"] = max(trade_durations)
        analysis["min_trade_duration_days"] = min(trade_durations)

    return analysis


def create_performance_report(
    result: BacktestResult, metrics: PerformanceMetrics
) -> Dict[str, Any]:
    """
    Create a comprehensive performance report.

    Args:
        result: Backtest results
        metrics: Performance metrics

    Returns:
        Dictionary with comprehensive performance report
    """
    report = {
        "summary": {
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "trading_days": len(result.equity_curve),
            "total_trades": len(result.trades),
            "initial_capital": result.equity_curve.iloc[0],
            "final_equity": result.equity_curve.iloc[-1],
        },
        "returns": {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "total_return_pct": f"{metrics.total_return:.2%}",
            "annualized_return_pct": f"{metrics.annualized_return:.2%}",
        },
        "risk_metrics": {
            "annualized_volatility": metrics.annualized_volatility,
            "volatility_pct": f"{metrics.annualized_volatility:.2%}",
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_pct": f"{metrics.max_drawdown:.2%}",
            "max_drawdown_duration": metrics.max_drawdown_duration,
            "calmar_ratio": metrics.calmar_ratio,
        },
        "trade_analysis": analyze_trade_distribution(result.trades),
    }

    # Position analysis
    position_counts = result.positions.value_counts()
    report["position_analysis"] = {
        "long_periods": position_counts.get(1, 0),
        "short_periods": position_counts.get(-1, 0),
        "flat_periods": position_counts.get(0, 0),
        "long_percentage": position_counts.get(1, 0) / len(result.positions) * 100,
        "short_percentage": position_counts.get(-1, 0) / len(result.positions) * 100,
        "flat_percentage": position_counts.get(0, 0) / len(result.positions) * 100,
    }

    return report


def benchmark_comparison(
    result: BacktestResult,
    benchmark_returns: pd.Series,
    benchmark_name: str = "Benchmark",
) -> Dict[str, Any]:
    """
    Compare strategy performance against a benchmark.

    Args:
        result: Backtest results
        benchmark_returns: Benchmark returns series (same index as strategy)
        benchmark_name: Name of the benchmark

    Returns:
        Dictionary with benchmark comparison
    """
    strategy_returns = calculate_returns(result.equity_curve)

    # Align series
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_aligned = strategy_returns.loc[common_index]
    benchmark_aligned = benchmark_returns.loc[common_index]

    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_aligned).cumprod()
    benchmark_cumulative = (1 + benchmark_aligned).cumprod()

    # Calculate correlation
    correlation = strategy_aligned.corr(benchmark_aligned)

    # Calculate beta
    covariance = strategy_aligned.cov(benchmark_aligned)
    benchmark_variance = benchmark_aligned.var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

    # Calculate alpha (using CAPM)
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    benchmark_return = benchmark_cumulative.iloc[-1] - 1
    strategy_return = strategy_cumulative.iloc[-1] - 1

    expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
    alpha = strategy_return - expected_return

    # Calculate tracking error
    tracking_error = (strategy_aligned - benchmark_aligned).std() * np.sqrt(252)

    # Calculate information ratio
    excess_return = strategy_aligned - benchmark_aligned
    information_ratio = (
        excess_return.mean() / excess_return.std() * np.sqrt(252)
        if excess_return.std() > 0
        else 0
    )

    comparison = {
        "benchmark_name": benchmark_name,
        "correlation": correlation,
        "beta": beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "total_return_strategy": strategy_return,
        "total_return_benchmark": benchmark_return,
        "outperformance": strategy_return - benchmark_return,
        "outperformance_pct": f"{(strategy_return - benchmark_return) * 100:.2f}%",
    }

    return comparison
