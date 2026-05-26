"""No-lookahead walk-forward backtest with discrete trade model.

At each bar *t* from ``min_train`` to end:
1. Classify regime using only data ``[0:t]``.
2. Map regime to discrete position via ``{0: -1, 1: 0, 2: 1}``.
3. Apply position at bar *t*, trade in/out on regime changes.

Produces trade-level analytics: Sharpe, max drawdown, trade count,
win rate, profit factor, total return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..backtesting.performance_metrics import calculate_drawdown_metrics
from ..backtesting.performance_metrics import calculate_sharpe_ratio
from .engine_protocol import ENGINE_REGISTRY

if TYPE_CHECKING:
    from .engine_protocol import RegimeEngine

_STATE_MAP = {0: -1, 1: 0, 2: 1}  # bear=short, sideways=flat, bull=long
_VALID_ENGINES = frozenset(ENGINE_REGISTRY.keys())
_HMM_ENGINES = frozenset({"messina", "hmm"})


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
    profit_factor = (
        sum(winning) / abs(sum(losing))
        if losing
        else float("inf")
    )

    return n_trades, float(win_rate), float(profit_factor)


def _resolve_engine(
    engine: str | RegimeEngine,
    window: int,
    threshold: float,
    n_states: int,
    ohlcv: pd.DataFrame | None,
) -> RegimeEngine:
    if isinstance(engine, str):
        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"engine must be one of {sorted(_VALID_ENGINES)}, got {engine!r}"
            )
        if engine in _HMM_ENGINES and ohlcv is None:
            raise ValueError(
                f"engine {engine!r} requires OHLCV data "
                "(open/high/low/close/volume). Pass ohlcv= DataFrame."
            )
        cls = ENGINE_REGISTRY[engine]
        if engine == "threshold":
            return cls(window=window, threshold=threshold)
        return cls(n_states=n_states)
    return engine


def walk_forward_backtest(
    prices: pd.Series,
    *,
    engine: str | RegimeEngine = "threshold",
    window: int = 20,
    threshold: float = 0.05,
    min_train: int = 252,
    ohlcv: pd.DataFrame | None = None,
    n_states: int = 3,
) -> dict:
    """No-lookahead walk-forward backtest with discrete position sizing.

    Parameters
    ----------
    prices : pd.Series
        Close prices with DatetimeIndex.
    engine : str | RegimeEngine
        Engine name (``"threshold"``, ``"messina"``, or ``"hmm"``) or a
        ``RegimeEngine`` instance.
    window : int
        Rolling window for threshold-based regime classification.
    threshold : float
        Return threshold for bull/bear classification.
    min_train : int
        Minimum number of bars before trading starts.
    ohlcv : pd.DataFrame | None
        OHLCV data required for messina/hmm engines.
    n_states : int
        Number of HMM states (ignored by threshold engine).

    Returns
    -------
    dict
        ``{sharpe, max_drawdown, n_trades, win_rate, profit_factor, total_return}``
    """
    eng = _resolve_engine(engine, window, threshold, n_states, ohlcv)

    if len(prices) < min_train + 1:
        return _empty_result()

    returns = prices.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n < min_train:
        return _empty_result()

    # Precompute features (returns None for threshold engine)
    precomputed = None
    if ohlcv is not None:
        try:
            precomputed = eng.precompute(ohlcv)
        except (ValueError, RuntimeError, KeyError):
            precomputed = None

    positions = np.zeros(n, dtype=int)

    if precomputed is not None:
        positions = _walk_forward_precomputed(
            eng, precomputed, returns, min_train, n_states,
        )
    else:
        positions = _walk_forward_raw(eng, returns, min_train, window, threshold)

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


def _walk_forward_raw(
    eng: RegimeEngine,
    returns: pd.Series,
    min_train: int,
    window: int,
    threshold: float,
) -> np.ndarray:
    n = len(returns)
    positions = np.zeros(n, dtype=int)

    for t in range(min_train, n):
        result = eng.classify(returns.iloc[:t])
        positions[t] = _STATE_MAP.get(result.regime, 0)

    return positions


def _walk_forward_precomputed(
    eng: RegimeEngine,
    features: pd.DataFrame,
    returns: pd.Series,
    min_train: int,
    n_states: int,
) -> np.ndarray:
    n = len(returns)
    positions = np.zeros(n, dtype=int)
    prev_means: np.ndarray | None = None
    last_regime: int = 1

    refit_every = max(1, (n - min_train) // 100)
    refit_every = min(refit_every, 20)

    for t in range(min_train, n):
        refit_now = (t == min_train) or ((t - min_train) % refit_every == 0)

        if refit_now:
            features_slice = features.iloc[:t]
            try:
                result = eng.classify(features_slice, prev_means=prev_means)
                last_regime = result.regime
                prev_means = result.means
            except (ValueError, RuntimeError):
                pass

        positions[t] = _STATE_MAP.get(last_regime, 0)

    return positions
