"""No-lookahead walk-forward backtest with discrete trade model.

At each bar *t* from ``min_train`` to end:
1. Replay pre-computed regime labels via ``_replay_regimes``.
2. Map regime to discrete position via ``{0: -1, 1: 0, 2: 1}``.
3. Apply position at bar *t*, trade in/out on regime changes.

Produces trade-level analytics: Sharpe, max drawdown, trade count,
win rate, profit factor, total return.

ADR-0017: engine param removed; regimes always pre-computed by pipeline.
ADR-0022: regime replay extracted from _hmm_pipeline into this module.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..backtesting.performance_metrics import calculate_drawdown_metrics
from ..backtesting.performance_metrics import calculate_sharpe_ratio

_STATE_MAP = {0: -1, 1: 0, 2: 1}  # bear=short, sideways=flat, bull=long


def _empty_result() -> dict:
    return {
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "n_trades": 0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "total_return": float("nan"),
    }


def _compute_trade_stats(
    positions: np.ndarray, equity: np.ndarray
) -> tuple[int, float, float]:
    n = len(positions)
    trade_pnls: list[float] = []
    in_trade = False
    entry_equity = 0.0

    for t in range(1, n):
        if positions[t] != 0 and positions[t - 1] == 0:
            entry_equity = float(equity[t])
            in_trade = True
        elif positions[t] == 0 and in_trade:
            trade_pnl = float(equity[t]) / entry_equity - 1.0
            trade_pnls.append(trade_pnl)
            in_trade = False

    if in_trade:
        trade_pnl = float(equity[-1]) / entry_equity - 1.0
        trade_pnls.append(trade_pnl)

    n_trades = len(trade_pnls)
    if n_trades == 0:
        return 0, float("nan"), float("nan")

    winning = [p for p in trade_pnls if p > 0]
    losing = [p for p in trade_pnls if p < 0]

    win_rate = len(winning) / n_trades
    profit_factor = sum(winning) / abs(sum(losing)) if losing else float("inf")

    return n_trades, float(win_rate), float(profit_factor)


def _replay_regimes(
    regimes: np.ndarray,
    min_train: int = 252,
    n_bars: int | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield (bar_index, regime) for each bar from min_train onward.

    This is the regime-replay generator — the walk-forward backtest's
    mode-1 iterator that replays pre-computed regime labels without
    calling any engine.  Formerly part of
    ``_hmm_pipeline._walk_forward_classify(mode=1)``.

    Parameters
    ----------
    regimes : np.ndarray
        Pre-computed regime labels, one per bar.
    min_train : int
        First bar to yield.
    n_bars : int | None
        Number of bars to iterate.  Defaults to ``len(regimes)``.
        When the caller's data has a different length (e.g. returns
        is one shorter than prices/regimes), pass ``len(returns)``
        to stay aligned.

    Yields
    ------
    (int, int)
        ``(t, regime)`` for every bar t in [min_train, n_bars).
    """
    limit = n_bars if n_bars is not None else len(regimes)
    for t in range(min_train, limit):
        yield t, int(regimes[t])


def _walk_forward_positions(
    returns: pd.Series,
    *,
    regimes: np.ndarray,
    posteriors: np.ndarray | None = None,
    min_train: int = 252,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
) -> np.ndarray:
    """Build position array from pre-computed regime labels.

    Replays regimes via the local _replay_regimes generator,
    applies dwell/hysteresis filters, and maps regimes to discrete
    positions via _STATE_MAP.
    """
    n = len(returns)
    positions = np.zeros(n, dtype=int)
    current_regime: int = 1
    consecutive_count = 0
    current_posteriors: np.ndarray | None = None

    for t, regime in _replay_regimes(regimes, min_train=min_train, n_bars=n):
        new_posteriors_arr: np.ndarray | None = posteriors[t] if posteriors is not None else None

        should_switch, consecutive_count = _apply_filters(
            regime,
            current_regime,
            new_posteriors_arr,
            current_posteriors,
            consecutive_count,
            dwell_bars,
            hysteresis_delta,
        )
        if should_switch:
            current_regime = regime
            current_posteriors = new_posteriors_arr

        positions[t] = _STATE_MAP.get(current_regime, 0)

    return positions


def walk_forward_backtest(
    prices: pd.Series,
    *,
    regimes: np.ndarray,
    posteriors: np.ndarray | None = None,
    min_train: int = 252,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
) -> dict:
    """No-lookahead walk-forward backtest with discrete position sizing.

    Parameters
    ----------
    prices : pd.Series
        Close prices with DatetimeIndex.
    regimes : np.ndarray
        Pre-computed regime label per bar (0=Bear, 1=Sideways, 2=Bull).
    posteriors : np.ndarray | None
        Pre-computed (n_bars, 3) posterior probabilities, used by the
        hysteresis filter. Hysteresis is a no-op when posteriors is None.
    min_train : int
        Minimum number of bars before trading starts.
    dwell_bars : int
        Minimum consecutive bars with same regime before switching position.
        0 disables the filter (default).
    hysteresis_delta : float
        Minimum posterior probability margin required to switch regimes.
        0.0 disables the filter (default). No-op when posteriors are None.

    Returns
    -------
    dict
        ``{sharpe, max_drawdown, n_trades, win_rate, profit_factor, total_return}``
    """

    if len(prices) < min_train + 1:
        return _empty_result()

    returns = prices.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n < min_train:
        return _empty_result()

    positions = _walk_forward_positions(
        returns,
        regimes=regimes,
        posteriors=posteriors,
        min_train=min_train,
        dwell_bars=dwell_bars,
        hysteresis_delta=hysteresis_delta,
    )

    # --- Daily P&L from lagged discrete positions ---
    pnl = np.zeros(n, dtype=float)
    pnl[1:] = positions[:-1].astype(float) * returns.iloc[1:].values
    equity = np.cumprod(1.0 + pnl)

    # Mark training period as NaN for metrics
    equity[:min_train] = np.nan
    valid_equity = equity[~np.isnan(equity)]

    if len(valid_equity) < 2:
        return _empty_result()

    equity_series = pd.Series(equity, index=returns.index).dropna()

    sharpe = calculate_sharpe_ratio(equity_series)
    dd_metrics = calculate_drawdown_metrics(equity_series)
    max_dd = dd_metrics["max_drawdown"]
    total_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0)

    n_trades, win_rate, profit_factor = _compute_trade_stats(positions, equity)

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return": total_return,
    }


def _apply_filters(
    new_regime: int,
    current_regime: int,
    posteriors: np.ndarray | None,
    current_posteriors: np.ndarray | None,
    consecutive_count: int,
    dwell_bars: int,
    hysteresis_delta: float,
) -> tuple[bool, int]:
    """Decide whether to switch from current_regime to new_regime.

    Returns (should_switch, updated_consecutive_count).
    AND logic: both dwell and hysteresis must agree to switch.
    """
    if new_regime == current_regime:
        # Same regime, no switch needed, reset counter
        return False, 0

    # Dwell-time: increment counter for this new regime
    new_count = consecutive_count + 1
    dwell_pass = (dwell_bars <= 0) or (new_count >= dwell_bars)

    # Hysteresis: check posterior margin
    hyst_pass = True
    if (
        hysteresis_delta > 0.0
        and posteriors is not None
        and current_posteriors is not None
    ):
        if new_regime < len(posteriors) and current_regime < len(current_posteriors):
            margin = posteriors[new_regime] - current_posteriors[current_regime]
            hyst_pass = margin > hysteresis_delta

    if dwell_pass and hyst_pass:
        return True, 0  # switch, reset counter
    return False, new_count  # hold, keep counting
