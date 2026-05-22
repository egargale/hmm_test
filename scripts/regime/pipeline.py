"""Regime detection pipeline — callable from Python or CLI.

This module owns the full analysis pipeline: threshold-based regime
classification, transition matrix, statistics, forecasts, walk-forward
backtest, and optional HMM analysis.  It returns a plain dict that
conforms to the regime-detection JSON contract.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from regime.hmm_adapter import run_hmm_regime
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
_FRAMEWORK_VERSION = "hmm_test v0.1.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)


def _probs_to_dict(probs: np.ndarray) -> dict[str, float]:
    """Convert a 3-element probability array to {bear, sideways, bull} dict."""
    return {
        "bear": float(probs[0]),
        "sideways": float(probs[1]),
        "bull": float(probs[2]),
    }


def _nan_to_none(value: float) -> float | None:
    """Replace NaN with None for JSON serialisation."""
    return None if isinstance(value, float) and math.isnan(value) else value


def run(
    prices: pd.Series,
    *,
    source: str,
    window: int = 20,
    threshold: float = 0.05,
    min_train: int = 252,
    use_hmm: bool = True,
    n_states: int = 3,
) -> dict:
    """Run the full regime-detection pipeline and return a JSON-compatible dict.

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex.
    source : str
        Label for the data source (ticker symbol or filename).
    window : int
        Rolling window for threshold-based regime classification.
    threshold : float
        Return threshold for bull/bear classification.
    min_train : int
        Minimum bars before walk-forward trading starts.
    use_hmm : bool
        Whether to run optional HMM analysis.
    n_states : int
        Number of HMM states.

    Returns
    -------
    dict
        JSON-compatible output conforming to the regime-detection contract.
    """
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

    # --- Threshold-based regime classification ---
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
        prices, window=window, threshold=threshold, min_train=min_train
    )
    walk_forward = {
        "sharpe": _nan_to_none(wf["sharpe"]),
        "max_drawdown": _nan_to_none(wf["max_drawdown"]),
        "n_trades": wf["n_trades"],
    }

    # HMM (optional)
    hmm_result: dict
    if use_hmm:
        try:
            hmm_result = run_hmm_regime(prices, n_states=n_states)
        except (ValueError, RuntimeError) as exc:
            hmm_result = {"available": False, "reason": str(exc)}
    else:
        hmm_result = {"available": False, "reason": "HMM disabled via --no-hmm"}

    return {
        "source": source,
        "rows": len(prices),
        "date_start": date_start,
        "date_end": date_end,
        "params": {
            "window": window,
            "threshold": threshold,
            "method": "threshold",
        },
        "states": [
            {"name": "bear", "index": 0},
            {"name": "sideways", "index": 1},
            {"name": "bull", "index": 2},
        ],
        "current_regime": {
            "name": _STATE_NAMES[last_regime],
            "index": last_regime,
        },
        "next_state_probabilities": _probs_to_dict(current_probs),
        "signal": signal,
        "transition_matrix": transmat.tolist(),
        "persistence_diagonal": persistence,
        "stationary_distribution": _probs_to_dict(stationary),
        "walk_forward": walk_forward,
        "hmm": hmm_result,
        "hmm_test_extras": {
            "n_states": n_states,
            "method": "threshold",
            "data_points": len(prices),
            "regime_counts": regime_counts_map,
        },
        "forecast": {
            "1_step": forecast_1,
            "5_step": forecast_5,
            "20_step": forecast_20,
        },
        "framework": _FRAMEWORK_VERSION,
        "disclaimer": _DISCLAIMER,
    }
