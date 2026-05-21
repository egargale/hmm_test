"""No-lookahead walk-forward backtest using Markov regime transitions."""

import numpy as np
import pandas as pd

from backtesting.performance_metrics import calculate_drawdown_metrics
from backtesting.performance_metrics import calculate_sharpe_ratio
from regime.markov_chain import build_transition_matrix
from regime.markov_chain import classify_regimes
from regime.markov_chain import compute_signal


def walk_forward_backtest(
    prices: pd.Series,
    window: int = 20,
    threshold: float = 0.05,
    min_train: int = 252,
) -> dict:
    """No-lookahead walk-forward backtest.

    For each bar *t* from ``min_train`` to end:
    1. Compute returns using only data ``[0:t]``.
    2. Classify regimes using ``classify_regimes`` on those returns.
    3. Build transition matrix from the regime sequence.
    4. Current regime = regime at ``t-1``.
    5. Next-state probabilities = ``transition_matrix[current_regime]``.
    6. Signal = ``compute_signal(next_state_probs)``.
    7. Position = signal clipped to ``[-1, 1]``.
    8. P&L at *t* = position * return at *t*.

    Parameters
    ----------
    prices : pd.Series
        Price series with DatetimeIndex.
    window : int
        Rolling window for regime classification.
    threshold : float
        Return threshold for bull/bear classification.
    min_train : int
        Minimum number of bars before trading starts.

    Returns
    -------
    dict
        ``{"sharpe": float, "max_drawdown": float, "n_trades": int}``
    """
    if len(prices) < min_train + 1:
        return {"sharpe": np.nan, "max_drawdown": np.nan, "n_trades": 0}

    returns = prices.pct_change().dropna()
    n = len(returns)

    if n < min_train:
        return {"sharpe": np.nan, "max_drawdown": np.nan, "n_trades": 0}

    positions = np.zeros(n)
    prev_position = 0.0

    for t in range(min_train, n):
        hist_returns = returns.iloc[:t]
        regimes = classify_regimes(hist_returns, window=window, threshold=threshold)
        transmat = build_transition_matrix(regimes)

        current_regime = int(regimes[-1])
        next_state_probs = transmat[current_regime]
        signal = compute_signal(next_state_probs)

        positions[t] = np.clip(signal, -1.0, 1.0)
        prev_position = positions[t]

    pnl = positions * returns.values
    pnl_series = pd.Series(pnl, index=returns.index)

    equity = (1.0 + pnl_series).cumprod()
    equity.iloc[:min_train] = np.nan
    equity = equity.dropna()

    if len(equity) < 2:
        return {"sharpe": np.nan, "max_drawdown": np.nan, "n_trades": 0}

    sharpe = calculate_sharpe_ratio(equity)
    dd_metrics = calculate_drawdown_metrics(equity)
    max_dd = dd_metrics["max_drawdown"]

    position_changes = np.diff(positions)
    n_trades = int(np.sum(position_changes != 0))

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": n_trades,
    }
