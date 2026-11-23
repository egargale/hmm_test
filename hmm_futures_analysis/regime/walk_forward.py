"""No-lookahead walk-forward backtest with discrete trade model.

At each bar *t* from ``min_train`` to end:
1. Replay pre-computed regime labels via ``_replay_regimes``.
2. Map regime to position: binary mode via ``{0: -1, 1: 0, 2: 1}``, or
   signal-scaled mode via ``clip(signal, -1, 1)`` where ``signal`` is the
   per-bar P(bull)−P(bear) (Issue #85).
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


def _bar_signal(
    t: int,
    posteriors: np.ndarray | None,
    returns: pd.Series,
    threshold_window: int | None,
    threshold_value: float | None,
) -> float:
    """Per-bar directional signal in ``[-1, 1]`` (Issue #85).

    HMM engines (posteriors present): ``signal = P(bull) − P(bear)``, i.e.
    ``posteriors[t][2] − posteriors[t][0]``. Reuses the
    ``markov_chain.compute_signal`` formula so there is one definition in
    the codebase.

    Threshold engine (posteriors is None): synthesize a pseudo-signal from
    the rolling return that the threshold engine itself uses for
    classification: ``pseudo_signal = clip(rolling_ret[t] / threshold, -1, 1)``
    where ``rolling_ret = returns.rolling(window).sum()``. A return exactly
    at ``+threshold`` → ``+1``; at ``−threshold`` → ``−1``; beyond saturates.
    This is no-lookahead: the rolling window is backward-looking, and the
    position is applied at bar ``t`` trading on ``returns[t+1]`` via the
    existing walk-forward lag.

    Defensive fallback: if sizing mode is on but the engine provides neither
    posteriors nor threshold params, raise ``ValueError``. Unreachable with
    the current engine set, but guards future engines that lack a signal
    source.
    """
    if posteriors is not None:
        row = posteriors[t]
        return float(row[2] - row[0])

    if threshold_window is not None and threshold_value is not None:
        if threshold_value == 0:
            raise ValueError("signal-sizing threshold_value must be non-zero")
        rolling_ret = returns.rolling(window=threshold_window).sum()
        val = rolling_ret.iloc[t]
        if val != val:  # NaN check (rolling window not yet filled)
            return 0.0
        return float(np.clip(val / threshold_value, -1.0, 1.0))

    raise ValueError(
        "signal-sizing requires posteriors or threshold window/threshold "
        "params; engine provides neither"
    )


def _empty_result() -> dict:
    return {
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "n_trades": 0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "total_return": float("nan"),
        "sizing_mode": "binary",
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
    signal_sizing: bool = False,
    signal_floor: float = 0.0,
    threshold_window: int | None = None,
    threshold_value: float | None = None,
) -> np.ndarray:
    """Build position array from pre-computed regime labels.

    Replays regimes via the local _replay_regimes generator,
    applies dwell/hysteresis filters, and maps regimes to discrete
    positions via _STATE_MAP (binary mode) or to signal-scaled float
    positions (signal-sizing mode, Issue #85).

    In signal-sizing mode the filters still gate *direction* (which regime
    label is active via ``current_regime``); the position *magnitude* is
    ``clip(signal[t], -1, 1)`` where ``signal[t]`` is the per-bar
    P(bull)−P(bear) (HMM engines) or the threshold pseudo-signal. When
    ``signal_floor > 0`` and ``abs(signal) < signal_floor`` the position is
    forced to 0.0 (flat). Position dtype is ``int`` in binary mode
    (bit-identical to pre-feature output) and ``float`` in sizing mode.
    """
    n = len(returns)
    positions = np.zeros(n, dtype=float if signal_sizing else int)
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

        if signal_sizing:
            signal_t = _bar_signal(
                t, posteriors, returns, threshold_window, threshold_value
            )
            mag = float(np.clip(signal_t, -1.0, 1.0))
            if signal_floor > 0.0 and abs(mag) < signal_floor:
                positions[t] = 0.0
            else:
                positions[t] = mag
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
    signal_sizing: bool = False,
    signal_floor: float = 0.0,
    threshold_window: int | None = None,
    threshold_value: float | None = None,
) -> dict:
    """No-lookahead walk-forward backtest with discrete or signal-scaled sizing.

    Parameters
    ----------
    prices : pd.Series
        Close prices with DatetimeIndex.
    regimes : np.ndarray
        Pre-computed regime label per bar (0=Bear, 1=Sideways, 2=Bull).
    posteriors : np.ndarray | None
        Pre-computed (n_bars, 3) posterior probabilities, used by the
        hysteresis filter and (in sizing mode) as the per-bar signal source.
        Hysteresis is a no-op when posteriors is None.
    min_train : int
        Minimum number of bars before trading starts.
    dwell_bars : int
        Minimum consecutive bars with same regime before switching position.
        0 disables the filter (default).
    hysteresis_delta : float
        Minimum posterior probability margin required to switch regimes.
        0.0 disables the filter (default). No-op when posteriors are None.
    signal_sizing : bool
        When True, use signal-scaled positions ``clip(signal, -1, 1)``
        instead of binary ±1/0 (Issue #85). Default False (binary).
    signal_floor : float
        In sizing mode, go flat when ``abs(signal) < signal_floor``.
        Default 0.0 (disabled).
    threshold_window : int | None
        Threshold-engine rolling window, used to synthesize a pseudo-signal
        when ``signal_sizing=True`` and ``posteriors is None``. Read off the
        engine config in ``pipeline.run()``.
    threshold_value : float | None
        Threshold-engine return threshold, used with ``threshold_window``.

    Returns
    -------
    dict
        ``{sharpe, max_drawdown, n_trades, win_rate, profit_factor,
        total_return, sizing_mode}`` where ``sizing_mode`` is ``"binary"``
        (default) or ``"signal"``.
    """

    sizing_mode = "signal" if signal_sizing else "binary"

    if len(prices) < min_train + 1:
        result = _empty_result()
        result["sizing_mode"] = sizing_mode
        return result

    returns = prices.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n < min_train:
        result = _empty_result()
        result["sizing_mode"] = sizing_mode
        return result

    positions = _walk_forward_positions(
        returns,
        regimes=regimes,
        posteriors=posteriors,
        min_train=min_train,
        dwell_bars=dwell_bars,
        hysteresis_delta=hysteresis_delta,
        signal_sizing=signal_sizing,
        signal_floor=signal_floor,
        threshold_window=threshold_window,
        threshold_value=threshold_value,
    )

    # --- Daily P&L from lagged positions ---
    # positions are int in binary mode, float in sizing mode; the .astype(float)
    # cast is a no-op for sizing mode and keeps the binary path bit-identical.
    pnl = np.zeros(n, dtype=float)
    pnl[1:] = positions[:-1].astype(float) * returns.iloc[1:].values
    equity = np.cumprod(1.0 + pnl)

    # Mark training period as NaN for metrics
    equity[:min_train] = np.nan
    valid_equity = equity[~np.isnan(equity)]

    if len(valid_equity) < 2:
        result = _empty_result()
        result["sizing_mode"] = sizing_mode
        return result

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
        "metrics_reliable": n_trades >= 10,
        "metrics_caveat": (
            f"n_trades={n_trades}; Sharpe, win-rate, and profit-factor "
            f"are statistically meaningless with < 10 independent trades"
            if n_trades < 10
            else None
        ),
        "sizing_mode": sizing_mode,
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
