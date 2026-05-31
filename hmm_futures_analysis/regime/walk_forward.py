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
from .engine_protocol import ENGINE_REGISTRY, HMM_ENGINES

if TYPE_CHECKING:
    from .engine_protocol import RegimeEngine

_STATE_MAP = {0: -1, 1: 0, 2: 1}  # bear=short, sideways=flat, bull=long
_VALID_ENGINES = frozenset(ENGINE_REGISTRY.keys())


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


def _resolve_engine(
    engine: str | RegimeEngine,
    window: int,
    threshold: float,
    n_states: int,
    ohlcv: pd.DataFrame | None,
    pca_variance: float | None = None,
    robust_method: str = "huber",
    saliency_threshold: float = 0.5,
) -> RegimeEngine:
    if isinstance(engine, str):
        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"engine must be one of {sorted(_VALID_ENGINES)}, got {engine!r}"
            )
        if engine in HMM_ENGINES and ohlcv is None:
            raise ValueError(
                f"engine {engine!r} requires OHLCV data "
                "(open/high/low/close/volume). Pass ohlcv= DataFrame."
            )
        cls = ENGINE_REGISTRY[engine][0]
        if engine == "threshold":
            return cls(window=window, threshold=threshold)
        kwargs: dict = {"n_states": n_states, "pca_variance": pca_variance}
        if engine == "robust_hmm":
            kwargs["robust_method"] = robust_method
        if engine == "fshmm":
            kwargs["saliency_threshold"] = saliency_threshold
        return cls(**kwargs)
    return engine


def _walk_forward_from_arrays(
    regimes: np.ndarray,
    posteriors: np.ndarray | None = None,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
) -> np.ndarray:
    """Apply dwell/hysteresis filters to pre-computed regime labels."""
    n = len(regimes)
    positions = np.zeros(n, dtype=int)
    current_regime: int = 1
    consecutive_count = 0
    current_posteriors: np.ndarray | None = None

    for t in range(n):
        new_regime = int(regimes[t])
        new_posteriors = posteriors[t] if posteriors is not None else None

        should_switch, consecutive_count = _apply_filters(
            new_regime,
            current_regime,
            new_posteriors,
            current_posteriors,
            consecutive_count,
            dwell_bars,
            hysteresis_delta,
        )
        if should_switch:
            current_regime = new_regime
            current_posteriors = new_posteriors

        positions[t] = _STATE_MAP.get(current_regime, 0)

    return positions


def walk_forward_backtest(
    prices: pd.Series,
    *,
    engine: str | RegimeEngine = "threshold",
    window: int = 20,
    threshold: float = 0.05,
    min_train: int = 252,
    ohlcv: pd.DataFrame | None = None,
    n_states: int = 3,
    pca_variance: float | None = None,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
    robust_method: str = "huber",
    saliency_threshold: float = 0.5,
    regimes: np.ndarray | None = None,
    posteriors: np.ndarray | None = None,
) -> dict:
    """No-lookahead walk-forward backtest with discrete position sizing.

    Parameters
    ----------
    prices : pd.Series
        Close prices with DatetimeIndex.
    engine : str | RegimeEngine
        Engine name (``"threshold"``, ``"messina"``, or ``"hmm"``) or a
        ``RegimeEngine`` instance.  Ignored when ``regimes`` is provided.
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
    pca_variance : float | None
        Optional PCA whitening threshold for HMM engines.
    dwell_bars : int
        Minimum consecutive bars with same regime before switching position.
        0 disables the filter (default).
    hysteresis_delta : float
        Minimum posterior probability margin required to switch regimes.
        0.0 disables the filter (default). No-op when posteriors are None.
    regimes : np.ndarray | None
        Pre-computed regime label per bar (0=Bear, 1=Sideways, 2=Bull).
        When provided, skips engine-based classification and applies
        dwell/hysteresis filters to these labels.
    posteriors : np.ndarray | None
        Pre-computed (n_bars, 3) posterior probabilities, used by the
        hysteresis filter. Ignored unless ``regimes`` is also provided.

    Returns
    -------
    dict
        ``{sharpe, max_drawdown, n_trades, win_rate, profit_factor, total_return}``
    """
    eng = _resolve_engine(
        engine,
        window,
        threshold,
        n_states,
        ohlcv,
        pca_variance,
        robust_method,
        saliency_threshold,
    )

    if len(prices) < min_train + 1:
        return _empty_result()

    returns = prices.pct_change(fill_method=None).dropna()
    n = len(returns)
    if n < min_train:
        return _empty_result()

    positions = np.zeros(n, dtype=int)

    if regimes is not None:
        positions = _walk_forward_from_arrays(
            regimes,
            posteriors=posteriors,
            dwell_bars=dwell_bars,
            hysteresis_delta=hysteresis_delta,
        )
    else:
        # Precompute features (returns None for threshold engine)
        precomputed = None
        if ohlcv is not None:
            try:
                precomputed = eng.precompute(ohlcv)
            except (ValueError, RuntimeError, KeyError):
                precomputed = None

        if precomputed is not None:
            positions = _walk_forward_precomputed(
                eng,
                precomputed,
                returns,
                min_train,
                n_states,
                dwell_bars=dwell_bars,
                hysteresis_delta=hysteresis_delta,
            )
        else:
            positions = _walk_forward_raw(
                eng,
                returns,
                min_train,
                window,
                threshold,
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


def _walk_forward_raw(
    eng: RegimeEngine,
    returns: pd.Series,
    min_train: int,
    window: int,
    threshold: float,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
) -> np.ndarray:
    n = len(returns)
    positions = np.zeros(n, dtype=int)
    current_regime = 1  # sideways default
    consecutive_count = 0
    current_posteriors: np.ndarray | None = None

    for t in range(min_train, n):
        result = eng.classify(returns.iloc[:t])
        new_regime = result.regime

        should_switch, consecutive_count = _apply_filters(
            new_regime,
            current_regime,
            result.posteriors,
            current_posteriors,
            consecutive_count,
            dwell_bars,
            hysteresis_delta,
        )
        if should_switch:
            current_regime = new_regime
            current_posteriors = result.posteriors

        positions[t] = _STATE_MAP.get(current_regime, 0)

    return positions


def _walk_forward_precomputed(
    eng: RegimeEngine,
    features: pd.DataFrame,
    returns: pd.Series,
    min_train: int,
    n_states: int,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
) -> np.ndarray:
    n = len(returns)
    positions = np.zeros(n, dtype=int)
    prev_means: np.ndarray | None = None
    current_regime: int = 1
    consecutive_count = 0
    current_posteriors: np.ndarray | None = None

    refit_every = max(1, (n - min_train) // 100)
    refit_every = min(refit_every, 20)

    for t in range(min_train, n):
        refit_now = (t == min_train) or ((t - min_train) % refit_every == 0)

        if refit_now:
            features_slice = features.iloc[:t]
            try:
                result = eng.classify(features_slice, prev_means=prev_means)
                new_regime = result.regime
                prev_means = result.means

                should_switch, consecutive_count = _apply_filters(
                    new_regime,
                    current_regime,
                    result.posteriors,
                    current_posteriors,
                    consecutive_count,
                    dwell_bars,
                    hysteresis_delta,
                )
                if should_switch:
                    current_regime = new_regime
                    current_posteriors = result.posteriors

            except (ValueError, RuntimeError):
                pass

        positions[t] = _STATE_MAP.get(current_regime, 0)

    return positions
