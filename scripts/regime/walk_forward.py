"""No-lookahead walk-forward backtest with discrete trade model.

At each bar *t* from ``min_train`` to end:
1. Classify regime using only data ``[0:t]``.
2. Map regime to discrete position via ``{0: -1, 1: 0, 2: 1}``.
3. Apply position at bar *t*, trade in/out on regime changes.

Produces trade-level analytics: Sharpe, max drawdown, trade count,
win rate, profit factor, total return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.performance_metrics import calculate_drawdown_metrics
from backtesting.performance_metrics import calculate_sharpe_ratio
from regime.hmm_adapter import hmm_state_from_slice
from regime.markov_chain import classify_regimes

_VALID_ENGINES = frozenset({"threshold", "messina", "hmm"})
_STATE_MAP = {0: -1, 1: 0, 2: 1}  # bear=short, sideways=flat, bull=long
_HMM_ENGINES = frozenset({"messina", "hmm"})


def _empty_result() -> dict:
    """Return NaN-filled result for insufficient data."""
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
    """Extract trade-level statistics from position transitions.

    Returns (n_trades, win_rate, profit_factor).
    """
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


def walk_forward_backtest(
    prices: pd.Series,
    *,
    engine: str = "threshold",
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
    engine : str
        Which engine to use: ``"threshold"``, ``"messina"``, or ``"hmm"``.
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
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(_VALID_ENGINES)}, got {engine!r}"
        )

    # HMM engines require OHLCV
    if engine in _HMM_ENGINES and ohlcv is None:
        raise ValueError(
            f"engine {engine!r} requires OHLCV data "
            "(open/high/low/close/volume). Pass ohlcv= DataFrame."
        )

    if len(prices) < min_train + 1:
        return _empty_result()

    returns = prices.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n < min_train:
        return _empty_result()

    # --- Dispatch to engine ---
    if engine == "threshold":
        positions = _walk_forward_threshold(returns, window, threshold, min_train)
    elif engine in _HMM_ENGINES:
        use_messina = engine == "messina"
        positions = _walk_forward_hmm(
            ohlcv,  # type: ignore[arg-type]  # guard above ensures not None
            returns,
            min_train,
            n_states=n_states,
            use_messina=use_messina,
        )
    else:
        return _empty_result()  # pragma: no cover

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


def _walk_forward_threshold(
    returns: pd.Series,
    window: int,
    threshold: float,
    min_train: int,
) -> np.ndarray:
    """Threshold engine: classify regimes on expanding window, map to positions.

    Returns an integer array of positions (-1, 0, 1) parallel to ``returns``.
    """
    n = len(returns)
    positions = np.zeros(n, dtype=int)

    for t in range(min_train, n):
        hist_returns = returns.iloc[:t]
        regimes = classify_regimes(hist_returns, window=window, threshold=threshold)
        regime = int(regimes[-1])
        positions[t] = _STATE_MAP[regime]

    return positions


def _walk_forward_hmm(
    ohlcv: pd.DataFrame,
    returns: pd.Series,
    min_train: int,
    n_states: int,
    use_messina: bool,
) -> np.ndarray:
    """HMM engine: refit HMM on expanding feature window, map to positions.

    Precomputes features once on the full dataset, then slices per bar for
    walk-forward training.  Uses parameter matching to preserve label
    continuity across consecutive fits.

    Returns an integer array of positions (-1, 0, 1) parallel to ``returns``.
    """
    n = len(returns)

    # Precompute features once (bias-free: all indicators are backward-looking)
    try:
        precomputed = hmm_state_from_slice(
            ohlcv, n_states=n_states, use_messina=use_messina, return_features=True
        )
        features: pd.DataFrame = precomputed["features"]
    except (ValueError, RuntimeError, KeyError):
        # Feature engineering or HMM fitting failed — no signal
        return np.zeros(n, dtype=int)

    positions = np.zeros(n, dtype=int)
    prev_means: np.ndarray | None = None
    last_regime: int = 1  # default sideways

    # Skip-N: refit HMM every N bars, hold regime between refits.
    # Balances accuracy vs performance for real tickers (2000+ bars × 44 features).
    # refit_every is capped at 20 bars (refit at least every 20 bars).
    refit_every = max(1, (n - min_train) // 100)  # target ~100 fits total
    refit_every = min(refit_every, 20)              # cap interval at 20 bars

    for t in range(min_train, n):
        refit_now = (t == min_train) or ((t - min_train) % refit_every == 0)

        if refit_now:
            # Slice features up to bar t (no lookahead)
            features_slice = features.iloc[:t]

            try:
                result = hmm_state_from_slice(
                    features_slice,
                    n_states=n_states,
                    use_messina=False,  # features already computed
                    return_features=False,
                    prev_means=prev_means,
                    precomputed=True,
                )
                last_regime = result["regime"]
                prev_means = result["means"]
            except (ValueError, RuntimeError):
                # HMM fit failure — hold previous position
                pass

        positions[t] = _STATE_MAP.get(last_regime, 0)

    return positions
