"""No-lookahead walk-forward backtest with discrete or signal-sized positions.

At each bar *t* from ``min_train`` to end:
1. Replay pre-computed regime labels via ``_replay_regimes``.
2. Apply the dwell-time / hysteresis filters to pick the allowed direction.
3. Write the position:
   - ``binary`` (default): ``{0: -1, 1: 0, 2: 1}``.
   - ``signal`` (opt-in): direction-gated projection of
     ``clip(P(bull) - P(bear), -1, 1)`` onto the filtered direction, with an
     optional ``signal_floor`` that flattens small positions.

Produces trade-level analytics: Sharpe, max drawdown, trade count,
win rate, profit factor, total return, and the applied ``sizing_mode``.

Signal sizing requires per-bar posteriors, so the threshold engine (which has
no posteriors) is unsupported and raises ``ValueError``.

ADR-0017: engine param removed; regimes always pre-computed by pipeline.
ADR-0022: regime replay extracted from _hmm_pipeline into this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import numpy as np
import pandas as pd

from ..backtesting.performance_metrics import calculate_drawdown_metrics
from ..backtesting.performance_metrics import calculate_sharpe_ratio

_STATE_MAP = {0: -1, 1: 0, 2: 1}  # bear=short, sideways=flat, bull=long
_SIZING_MODES = ("binary", "signal")
_SIGNAL_UNSUPPORTED = (
    "signal sizing requires an engine with posterior probabilities "
    "(hmm, messina, robust_hmm, fshmm); the threshold engine is unsupported"
)


def _validate_sizing_args(sizing_mode: str, signal_floor: float) -> float:
    """Validate sizing arguments and return the normalised floor as a float."""
    if sizing_mode not in _SIZING_MODES:
        raise ValueError(
            f"sizing_mode must be 'binary' or 'signal', got {sizing_mode!r}"
        )
    try:
        floor = float(signal_floor)
    except (TypeError, ValueError):
        raise ValueError(
            f"signal_floor must be a finite number in [0, 1], got {signal_floor!r}"
        ) from None
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError(
            f"signal_floor must be a finite number in [0, 1], got {signal_floor!r}"
        )
    if sizing_mode == "binary" and floor != 0.0:
        raise ValueError("signal_floor requires sizing_mode='signal'")
    return floor


def _empty_result(sizing_mode: str = "binary") -> dict:
    return {
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "n_trades": 0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "total_return": float("nan"),
        "sizing_mode": sizing_mode,
    }


def _sized_position(
    posterior_row: np.ndarray,
    allowed: int,
    signal_floor: float,
) -> float:
    """Return the direction-gated signal-sized position for one bar.

    ``allowed`` is the filtered regime direction (-1, 0, +1).  The raw signal
    is ``P(bull) - P(bear)`` clipped to [-1, 1].  If the signal opposes the
    allowed direction (or the regime is sideways), the position is flat.
    """
    if allowed == 0:
        return 0.0
    raw = float(posterior_row[2]) - float(posterior_row[0])
    if not math.isfinite(raw):
        return 0.0
    clipped = float(np.clip(raw, -1.0, 1.0))
    directional = allowed * clipped
    if directional <= 0.0:
        return 0.0
    position = float(allowed * directional)
    if abs(position) < signal_floor:
        return 0.0
    return position


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
    sizing_mode: Literal["binary", "signal"] = "binary",
    signal_floor: float = 0.0,
) -> np.ndarray:
    """Build the position array from pre-computed regime labels.

    Replays regimes via the local _replay_regimes generator, applies
    dwell/hysteresis filters, and writes either discrete positions via
    _STATE_MAP (``binary``) or direction-gated signal-sized positions
    (``signal``).

    In signal mode, ``posteriors`` is required and must have shape
    ``(>= len(returns), 3)``; the per-bar signal is
    ``P(bull) - P(bear)`` clipped to [-1, 1].
    """
    signal_floor = _validate_sizing_args(sizing_mode, signal_floor)
    n = len(returns)

    post_arr: np.ndarray | None = None
    if sizing_mode == "signal":
        if posteriors is None:
            raise ValueError(_SIGNAL_UNSUPPORTED)
        post_arr = np.asarray(posteriors, dtype=float)
        if post_arr.ndim != 2 or post_arr.shape[1] < 3 or len(post_arr) < n:
            raise ValueError(
                "signal sizing requires posteriors with shape (>= n, 3), "
                f"got {post_arr.shape}"
            )

    positions = np.zeros(n, dtype=float if sizing_mode == "signal" else int)
    current_regime: int = 1
    consecutive_count = 0
    current_posteriors: np.ndarray | None = None

    for t, regime in _replay_regimes(regimes, min_train=min_train, n_bars=n):
        new_posteriors_arr: np.ndarray | None = (
            posteriors[t] if posteriors is not None else None
        )

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

        if sizing_mode == "signal":
            positions[t] = _sized_position(
                post_arr[t],  # type: ignore[index]
                _STATE_MAP.get(current_regime, 0),
                signal_floor,
            )
        else:
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
    sizing_mode: Literal["binary", "signal"] = "binary",
    signal_floor: float = 0.0,
) -> dict:
    """No-lookahead walk-forward backtest with binary or signal sizing.

    Parameters
    ----------
    prices : pd.Series
        Close prices with DatetimeIndex.
    regimes : np.ndarray
        Pre-computed regime label per bar (0=Bear, 1=Sideways, 2=Bull).
    posteriors : np.ndarray | None
        Pre-computed (n_bars, 3) posterior probabilities, used by the
        hysteresis filter and, in signal mode, for position sizing.
        Hysteresis is a no-op when posteriors is None.
    min_train : int
        Minimum number of bars before trading starts.
    dwell_bars : int
        Minimum consecutive bars with same regime before switching position.
        0 disables the filter (default).
    hysteresis_delta : float
        Minimum posterior probability margin required to switch regimes.
        0.0 disables the filter (default). No-op when posteriors are None.
    sizing_mode : {"binary", "signal"}
        ``binary`` (default) maps the filtered regime to -1/0/+1.
        ``signal`` sizes the position by ``clip(P(bull) - P(bear), -1, 1)``
        within the filtered direction; it requires ``posteriors`` and an
        HMM-family engine (the threshold engine is unsupported).
    signal_floor : float
        In signal mode, positions with ``abs(position) < signal_floor`` are
        flattened.  Must be in [0, 1]; default 0.0 (disabled).

    Returns
    -------
    dict
        ``{sharpe, max_drawdown, n_trades, win_rate, profit_factor,
        total_return, sizing_mode}`` (plus internal reliability metadata on
        non-empty results).
    """
    signal_floor = _validate_sizing_args(sizing_mode, signal_floor)
    if sizing_mode == "signal" and posteriors is None:
        raise ValueError(_SIGNAL_UNSUPPORTED)

    if len(prices) < min_train + 1:
        return _empty_result(sizing_mode)

    returns = prices.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n < min_train:
        return _empty_result(sizing_mode)

    positions = _walk_forward_positions(
        returns,
        regimes=regimes,
        posteriors=posteriors,
        min_train=min_train,
        dwell_bars=dwell_bars,
        hysteresis_delta=hysteresis_delta,
        sizing_mode=sizing_mode,
        signal_floor=signal_floor,
    )

    # --- Daily P&L from lagged positions ---
    pnl = np.zeros(n, dtype=float)
    pnl[1:] = positions[:-1].astype(float) * returns.iloc[1:].values
    equity = np.cumprod(1.0 + pnl)

    # Mark training period as NaN for metrics
    equity[:min_train] = np.nan
    valid_equity = equity[~np.isnan(equity)]

    if len(valid_equity) < 2:
        return _empty_result(sizing_mode)

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
        "sizing_mode": sizing_mode,
        "metrics_reliable": n_trades >= 10,
        "metrics_caveat": (
            f"n_trades={n_trades}; Sharpe, win-rate, and profit-factor "
            f"are statistically meaningless with < 10 independent trades"
            if n_trades < 10
            else None
        ),
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
