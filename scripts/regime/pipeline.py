"""Regime detection pipeline — callable from Python or CLI.

This module owns the full analysis pipeline: threshold-based regime
classification, transition matrix, statistics, forecasts, and walk-forward
backtest.  Supports three engines: threshold (fast, close-only), messina
(HMM + 12 Messina features), and hmm (HMM + ~44 generic features).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from regime.markov_chain import (
    build_transition_matrix,
    classify_regimes,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
    forecast_n_steps,
)
from regime.walk_forward import walk_forward_backtest

_STATE_NAMES = ("bear", "sideways", "bull")
_VALID_ENGINES = frozenset({"threshold", "messina", "hmm"})
_FRAMEWORK_VERSION = "hmm_test v0.2.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)

# Engine feature labels
_ENGINE_FEATURES: dict[str, str] = {
    "threshold": "returns",
    "messina": "messina",
    "hmm": "generic",
}


def _probs_to_dict(probs: np.ndarray) -> dict[str, float]:
    """Convert a 3-element probability array to {bear, sideways, bull} dict."""
    return {
        "bear": float(probs[0]),
        "sideways": float(probs[1]),
        "bull": float(probs[2]),
    }


def _nan_to_none(value: float) -> float | None:
    """Replace NaN and Infinity with None for JSON serialisation."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def run(
    prices: pd.Series,
    *,
    source: str,
    engine: str = "threshold",
    window: int = 20,
    threshold: float = 0.05,
    min_train: int = 252,
    ohlcv: pd.DataFrame | None = None,
    n_states: int = 3,
) -> dict:
    """Run the full regime-detection pipeline and return a JSON-compatible dict.

    Parameters
    ----------
    prices : pd.Series
        Close price series with a DatetimeIndex.
    source : str
        Label for the data source (ticker symbol or filename).
    engine : str
        Which engine to use: ``"threshold"``, ``"messina"``, or ``"hmm"``.
    window : int
        Rolling window for threshold-based regime classification.
    threshold : float
        Return threshold for bull/bear classification.
    min_train : int
        Minimum bars before walk-forward trading starts.
    ohlcv : pd.DataFrame | None
        OHLCV data (required for messina/hmm engines).
    n_states : int
        Number of HMM states (ignored by threshold engine).

    Returns
    -------
    dict
        JSON-compatible output conforming to the hmm-regime-detection contract.
    """
    # --- Engine validation ---
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(_VALID_ENGINES)}, got {engine!r}"
        )

    # --- Input validation ---
    if not isinstance(prices, pd.Series):
        raise ValueError(f"prices must be a pd.Series, got {type(prices).__name__}")
    if not pd.api.types.is_numeric_dtype(prices):
        raise ValueError(f"prices must be numeric, got dtype {prices.dtype}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if len(prices) < 2:
        raise ValueError(f"prices must have at least 2 rows, got {len(prices)}")

    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError(
            f"prices must yield at least 2 valid returns, got {len(returns)}"
        )

    # --- Threshold-based regime classification (always runs for stats) ---
    regimes = classify_regimes(returns, window=window, threshold=threshold)
    transmat = build_transition_matrix(regimes)
    stationary = compute_stationary_distribution(transmat)
    persistence = compute_persistence_diagonal(transmat)

    last_regime = int(regimes[-1])
    current_probs = transmat[last_regime]
    signal = compute_signal(current_probs)

    # Regime counts
    unique, counts = np.unique(regimes, return_counts=True)
    regime_counts_map: dict[str, int] = {name: 0 for name in _STATE_NAMES}
    for s, c in zip(unique, counts):
        regime_counts_map[_STATE_NAMES[s]] = int(c)

    # Date boundaries
    date_start: str = ""
    date_end: str = ""
    try:
        date_start = str(prices.index[0].date())
        date_end = str(prices.index[-1].date())
    except (AttributeError, IndexError, TypeError):
        pass

    # Forecasts
    forecast_1 = _probs_to_dict(forecast_n_steps(transmat, current_probs, 1))
    forecast_5 = _probs_to_dict(forecast_n_steps(transmat, current_probs, 5))
    forecast_20 = _probs_to_dict(forecast_n_steps(transmat, current_probs, 20))

    # Walk-forward backtest
    wf = walk_forward_backtest(
        prices,
        engine=engine,
        window=window,
        threshold=threshold,
        min_train=min_train,
        ohlcv=ohlcv,
        n_states=n_states,
    )
    walk_forward = {
        "sharpe": _nan_to_none(wf["sharpe"]),
        "max_drawdown": _nan_to_none(wf["max_drawdown"]),
        "n_trades": wf["n_trades"],
        "win_rate": _nan_to_none(wf["win_rate"]),
        "profit_factor": _nan_to_none(wf["profit_factor"]),
        "total_return": _nan_to_none(wf["total_return"]),
    }

    # Engine info
    engine_info: dict[str, object] = {
        "method": engine,
        "features": _ENGINE_FEATURES.get(engine, engine),
        "n_states": n_states,
    }
    if engine in ("messina", "hmm"):
        engine_info["caveat"] = (
            "HMM states sorted by mean return; labels may swap on re-fit"
        )

    return {
        "source": source,
        "engine": engine,
        "dates": {
            "start": date_start,
            "end": date_end,
        },
        "current_regime": {
            "name": _STATE_NAMES[last_regime],
            "index": last_regime,
        },
        "signal": signal,
        "next_state_probabilities": _probs_to_dict(current_probs),
        "transition_matrix": transmat.tolist(),
        "stationary_distribution": _probs_to_dict(stationary),
        "persistence_diagonal": persistence,
        "regime_counts": regime_counts_map,
        "walk_forward": walk_forward,
        "forecast": {
            "1_step": forecast_1,
            "5_step": forecast_5,
            "20_step": forecast_20,
        },
        "engine_info": engine_info,
        "framework": _FRAMEWORK_VERSION,
        "disclaimer": _DISCLAIMER,
    }
