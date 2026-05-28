"""Regime detection pipeline — callable from Python or CLI.

This module owns the full analysis pipeline: threshold-based regime
classification, transition matrix, statistics, forecasts, and walk-forward
backtest.  Supports three engines: threshold (fast, close-only), messina
(HMM + 19 Messina features), and hmm (HMM + ~50 generic features).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine_protocol import ENGINE_REGISTRY
from .engines._hmm_shared import select_n_states
from .markov_chain import (
    build_transition_matrix,
    classify_regimes,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
    forecast_n_steps,
)
from .walk_forward import walk_forward_backtest

_HMM_ENGINES = frozenset({"messina", "hmm"})

_STATE_NAMES = ("bear", "sideways", "bull")
_FRAMEWORK_VERSION = "hmm_test v0.2.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)

_ENGINE_FEATURES: dict[str, str] = {
    "threshold": "returns",
    "messina": "messina",
    "hmm": "generic",
}


def _probs_to_dict(probs: np.ndarray) -> dict[str, float]:
    return {
        "bear": float(probs[0]),
        "sideways": float(probs[1]),
        "bull": float(probs[2]),
    }


def _nan_to_none(value: float) -> float | None:
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
    n_states: int | str = 3,
    pca_variance: float | None = None,
) -> dict:
    """Run the full regime-detection pipeline and return a JSON-compatible dict."""
    if engine not in ENGINE_REGISTRY:
        raise ValueError(
            f"engine must be one of {sorted(ENGINE_REGISTRY.keys())}, got {engine!r}"
        )

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

    # --- Resolve n_states='auto' for HMM engines ---
    resolved_n_states: int = 3  # default for threshold
    if isinstance(n_states, str) and n_states == "auto":
        if engine in _HMM_ENGINES:
            # Need to precompute features first for BIC evaluation
            pass  # resolved below after precompute
        else:
            resolved_n_states = 3  # threshold ignores n_states
    elif isinstance(n_states, int):
        resolved_n_states = n_states
    else:
        raise ValueError(f"n_states must be int or 'auto', got {n_states!r}")

    # --- Engine-specific regime classification ---
    warmup_bars: int | None = None

    if engine == "threshold":
        regimes = classify_regimes(returns, window=window, threshold=threshold)
    else:
        if ohlcv is None:
            raise ValueError(
                f"engine {engine!r} requires OHLCV data "
                "(open/high/low/close/volume). Pass ohlcv= DataFrame."
            )
        eng_cls = ENGINE_REGISTRY[engine]

        # Precompute features (needed for BIC and for classification)
        eng_temp = eng_cls(n_states=3)  # n_states doesn't affect precompute
        precomputed = None
        try:
            precomputed = eng_temp.precompute(ohlcv)
        except (ValueError, RuntimeError, KeyError):
            precomputed = None
        if precomputed is None:
            raise ValueError(
                f"engine {engine!r} failed to precompute features from OHLCV data"
            )

        # Resolve 'auto' now that we have features
        if isinstance(n_states, str) and n_states == "auto":
            resolved_n_states = select_n_states(
                precomputed.dropna().to_numpy(dtype=np.float64),
                max_states=6,
            )

        eng = eng_cls(n_states=resolved_n_states, pca_variance=pca_variance)

        n = len(returns)
        regimes = np.ones(n, dtype=int)
        prev_means: np.ndarray | None = None
        last_regime = 1

        refit_every = max(1, (n - min_train) // 100)
        refit_every = min(refit_every, 20)

        for t in range(min_train, n):
            refit_now = (t == min_train) or ((t - min_train) % refit_every == 0)
            if refit_now:
                features_slice = precomputed.iloc[:t]
                try:
                    result = eng.classify(
                        features_slice, prev_means=prev_means
                    )
                    last_regime = result.regime
                    prev_means = result.means
                except (ValueError, RuntimeError):
                    pass
            regimes[t] = last_regime

        warmup_bars = min_train

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
        n_states=resolved_n_states,
        pca_variance=pca_variance,
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
        "n_states": resolved_n_states,
    }
    if engine in ("messina", "hmm"):
        engine_info["caveat"] = (
            "HMM states sorted by mean return; labels may swap on re-fit"
        )
        if warmup_bars is not None:
            engine_info["warmup_bars"] = warmup_bars

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
