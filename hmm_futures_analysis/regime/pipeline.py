"""Regime detection pipeline — callable from Python or CLI.

This module owns the full analysis pipeline: threshold-based regime
classification, transition matrix, statistics, forecasts, and walk-forward
backtest.  Supports three engines: threshold (fast, close-only), messina
(HMM + 19 Messina features), and hmm (HMM + ~50 generic features).
"""

from __future__ import annotations

import math
import time

import dataclasses

import numpy as np
import pandas as pd

from .engine_protocol import ENGINE_REGISTRY, resolve_engine
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


_STATE_NAMES = ("bear", "sideways", "bull")
_FRAMEWORK_VERSION = "hmm_test v0.2.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)


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
    engine_config: object = None,
    min_train: int = 252,
    ohlcv: pd.DataFrame | None = None,
    dwell_bars: int = 0,
    hysteresis_delta: float = 0.0,
    duration_forecast: bool = False,
    duration_model: str = "weibull",
    profile: bool = False,
) -> dict:
    """Run the full regime-detection pipeline and return a JSON-compatible dict."""
    t_start = time.monotonic() if profile else 0.0
    _phases: dict[str, float] = {}
    _classify_times: list[float] = []

    if engine_config is None:
        raise ValueError("engine_config is required")

    engine: str = getattr(engine_config, "name", None)
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
    config = engine_config  # shorthand
    raw_n_states = getattr(config, "n_states", 3)
    features_label: str = getattr(config, "features", engine)
    resolved_n_states: int = 3  # default for threshold

    # --- Engine-specific regime classification ---
    warmup_bars: int | None = None
    eng: object | None = None  # set for HMM engines

    if engine == "threshold":
        window = getattr(config, "window", 20)
        threshold = getattr(config, "threshold", 0.05)
        regimes = classify_regimes(returns, window=window, threshold=threshold)
        eng = resolve_engine(config)
    else:
        if ohlcv is None:
            raise ValueError(
                f"engine {engine!r} requires OHLCV data "
                "(open/high/low/close/volume). Pass ohlcv= DataFrame."
            )
        eng_cls = ENGINE_REGISTRY[engine][0]

        # Precompute features (needed for BIC and for classification)
        eng_temp = eng_cls(n_states=3)  # n_states doesn't affect precompute
        precomputed = None
        t_pc = time.monotonic()
        try:
            precomputed = eng_temp.precompute(ohlcv)
        except (ValueError, RuntimeError, KeyError):
            precomputed = None
        if profile:
            _phases["precompute"] = float(round(time.monotonic() - t_pc, 6))
        if precomputed is None:
            raise ValueError(
                f"engine {engine!r} failed to precompute features from OHLCV data"
            )

        # Resolve 'auto' now that we have features
        if isinstance(raw_n_states, str) and raw_n_states == "auto":
            t_bic = time.monotonic()
            resolved_n_states = select_n_states(
                precomputed.dropna().to_numpy(dtype=np.float64),
                max_states=6,
                profile=_phases if profile else False,
            )
            if profile:
                _phases["bic_select_n_states"] = float(
                    round(time.monotonic() - t_bic, 6)
                )
            config = dataclasses.replace(config, n_states=resolved_n_states)
        elif isinstance(raw_n_states, int):
            resolved_n_states = raw_n_states
        else:
            raise ValueError(f"n_states must be int or 'auto', got {raw_n_states!r}")

        eng = resolve_engine(config)

        n = len(returns)
        regimes = np.ones(n, dtype=int)
        posteriors_all = np.zeros((n, 3), dtype=float)
        prev_means: np.ndarray | None = None
        last_regime = 1
        last_post: np.ndarray | None = None

        refit_every = max(1, (n - min_train) // 100)
        refit_every = min(refit_every, 20)

        t_wf = time.monotonic()
        for t in range(min_train, n):
            refit_now = (t == min_train) or ((t - min_train) % refit_every == 0)
            if refit_now:
                features_slice = precomputed.iloc[:t]
                try:
                    t_cls_start = time.monotonic() if profile else 0.0
                    result = eng.classify(features_slice, prev_means=prev_means)
                    if profile:
                        _classify_times.append(time.monotonic() - t_cls_start)
                    last_regime = result.regime
                    prev_means = result.means
                    last_post = result.posteriors
                except (ValueError, RuntimeError):
                    pass
            regimes[t] = last_regime
            if last_post is not None:
                posteriors_all[t] = last_post
        if profile:
            _phases["walk_forward_classify"] = float(round(time.monotonic() - t_wf, 6))

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

    # Walk-forward backtest (reuse pre-computed regime labels for HMM engines)
    t_wfb = time.monotonic()
    wf_kwargs: dict = {
        "engine": eng,
        "min_train": min_train,
        "dwell_bars": dwell_bars,
        "hysteresis_delta": hysteresis_delta,
    }
    # HMM engines: reuse pre-computed regime labels
    if config.is_hmm:
        wf_kwargs["regimes"] = regimes
        wf_kwargs["posteriors"] = posteriors_all
    wf = walk_forward_backtest(prices, **wf_kwargs)
    walk_forward = {
        "sharpe": _nan_to_none(wf["sharpe"]),
        "max_drawdown": _nan_to_none(wf["max_drawdown"]),
        "n_trades": wf["n_trades"],
        "win_rate": _nan_to_none(wf["win_rate"]),
        "profit_factor": _nan_to_none(wf["profit_factor"]),
        "total_return": _nan_to_none(wf["total_return"]),
    }
    if profile:
        _phases["walk_forward_backtest"] = float(round(time.monotonic() - t_wfb, 6))

    # Engine info
    engine_info: dict[str, object] = {
        "method": engine,
        "features": features_label,
        "n_states": resolved_n_states,
    }
    engine_info.update(config.engine_info_extras(warmup_bars=warmup_bars, eng=eng))

    result = {
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

    # --- Duration forecast (optional post-processing) ---
    if duration_forecast:
        from .duration_forecast import forecast_duration

        result["duration_forecast"] = forecast_duration(
            regimes, model=duration_model, prices=prices
        )

    # --- Profiling ---
    if profile:
        # Extract bic_detail from _phases if present (put there by select_n_states)
        bic_detail = _phases.pop("bic_detail", None)

        timing: dict = {
            "total_wall_seconds": float(round(time.monotonic() - t_start, 6)),
            "phases": _phases,
        }
        if bic_detail is not None:
            timing["bic_detail"] = bic_detail
        if _classify_times:
            sorted_times = sorted(_classify_times)
            n_cls = len(sorted_times)
            median = sorted_times[n_cls // 2]
            if n_cls % 2 == 0:
                median = (sorted_times[n_cls // 2 - 1] + median) / 2.0
            p99_idx = int(n_cls * 0.99)
            if p99_idx >= n_cls:
                p99_idx = n_cls - 1
            timing["walk_forward_classify_stats"] = {
                "min": float(round(sorted_times[0], 6)),
                "median": float(round(median, 6)),
                "p99": float(round(sorted_times[p99_idx], 6)),
                "n_calls": n_cls,
            }
        result["timing"] = timing

    return result
