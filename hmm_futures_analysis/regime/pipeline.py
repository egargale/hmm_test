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
from typing import NamedTuple

import numpy as np
import pandas as pd

from .engine_protocol import (
    ENGINE_REGISTRY,
    resolve_engine,
)
from .markov_chain import (
    build_transition_matrix,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
    forecast_n_steps,
)
from .regime_transitions import extract_transitions
from .walk_forward import walk_forward_backtest


_STATE_NAMES = ("bear", "sideways", "bull")
_FRAMEWORK_VERSION = "hmm_test v0.2.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)


@dataclasses.dataclass
class MarkovStats:
    """Computed Markov chain statistics."""

    transmat: np.ndarray
    stationary: np.ndarray
    persistence: dict
    signal: float
    current_regime: int
    current_probs: np.ndarray
    regime_counts: dict[str, int]
    dates: dict[str, str]


class PipelineResult(NamedTuple):
    """Immutable result from :func:`run`.

    Call ``._asdict()`` to get a JSON-compatible dict for serialization.
    """

    source: str
    engine: str
    dates: dict[str, str]
    current_regime: dict[str, str | int]
    signal: float
    next_state_probabilities: dict[str, float]
    transition_matrix: list
    stationary_distribution: dict[str, float]
    persistence_diagonal: dict[str, float]
    regime_counts: dict[str, int]
    walk_forward: dict
    forecast: dict
    engine_info: dict
    framework: str
    disclaimer: str
    verdict: dict
    duration_forecast: dict | None = None
    regime_transitions: list | None = None
    timing: dict | None = None


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


def _compute_dynamic_threshold(
    duration_forecast: dict | None,
    base_threshold: float = 0.1,
) -> float:
    """Compute a regime-aging-adjusted threshold for Sideways verdict.

    Uses the Weibull expected duration to detect when the current regime
    has outlasted its historical norm. When ``days_in_regime >
    expected_total``, the threshold shrinks linearly from 1.0x at exactly
    expected up to 0.3x at 1.7x expected.

    Returns ``base_threshold`` when duration data is unavailable or the
    regime is within its expected life.
    """
    if duration_forecast is None:
        return base_threshold

    days_in = duration_forecast.get("days_in_regime")
    scale = duration_forecast.get("weibull_scale")
    shape = duration_forecast.get("weibull_shape")

    if days_in is None or scale is None or shape is None or shape <= 0:
        return base_threshold

    from scipy.special import gamma as _gamma  # type: ignore[unused-ignore]

    _days = float(days_in)
    _scale = float(scale)
    _shape = float(shape)
    expected_total = _scale * _gamma(1.0 + 1.0 / _shape)
    if expected_total <= 0:
        return base_threshold

    aging_ratio = _days / expected_total

    if aging_ratio <= 1.0:
        return base_threshold

    # Linear ramp: 1.0x at aging_ratio=1, 0.3x at aging_ratio >= 1.7
    threshold_mult = max(0.3, 2.0 - aging_ratio)
    return base_threshold * threshold_mult


def _compute_verdict(
    current_regime: int,
    signal: float,
    forecast_20: dict[str, float],
    sideways_threshold: float = 0.1,
) -> dict[str, object]:
    """Synthesize regime + forecasts into a single actionable verdict.

    Parameters
    ----------
    sideways_threshold :
        Signal magnitude below which the verdict stays ``"neutral"``
        when in a Sideways regime.  Use :func:`_compute_dynamic_threshold`
        to generate a regime-aging-adjusted value.

    Returns a dict with ``verdict`` (one of ``"bullish"``, ``"bearish"``,
    ``"neutral"``, ``"transition_bull"``, ``"transition_bear"``) and
    ``confidence`` (abs(signal), range 0-1).
    """
    if current_regime == 2:  # Bull
        # Bull forecast > bear forecast: bull dominance continues
        if forecast_20["bull"] > forecast_20.get("bear", 0):
            verdict = "bullish"
        else:
            verdict = "transition_bear"
    elif current_regime == 0:  # Bear
        # Bear forecast > bull forecast: bear dominance continues
        if forecast_20["bear"] > forecast_20.get("bull", 0):
            verdict = "bearish"
        else:
            verdict = "transition_bull"
    else:  # Sideways
        if abs(signal) < sideways_threshold:
            verdict = "neutral"
        elif signal > 0:
            verdict = "transition_bull"
        else:
            verdict = "transition_bear"

    return {
        "verdict": verdict,
        "confidence": round(float(abs(signal)), 4),
    }


def _validate_prices(prices: pd.Series) -> pd.Series:
    """Validate prices Series and return computed returns.

    Checks type, dtype, DatetimeIndex, and length.
    Returns the pct_change returns series (NaN dropped).
    """
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
    return returns


def _build_engine_info(
    engine_config: object,
    resolved_n_states: int,
    eng: object | None,
    *,
    warmup_bars: int | None = None,
) -> dict[str, object]:
    """Build engine_info dict from config and optional engine instance.

    Constructs base info (method, features, n_states) from config,
    then enriches via duck-typed ``enrich_info()`` if the engine
    provides it.
    """
    engine = getattr(engine_config, "name", None)
    features_label: str = getattr(engine_config, "features", engine)

    info: dict[str, object] = {
        "method": engine,
        "features": features_label,
        "n_states": resolved_n_states,
    }
    if hasattr(eng, "enrich_info"):
        ctx = {"warmup_bars": warmup_bars} if warmup_bars is not None else {}
        info.update(eng.enrich_info(ctx))
    return info


def _build_markov_stats(
    regimes: np.ndarray, price_index: pd.DatetimeIndex
) -> MarkovStats:
    """Compute Markov chain statistics from a regimes array.

    Pure function: given regimes and a price index, produces
    transition matrix, stationary distribution, persistence,
    signal, regime counts, current regime/probs, and dates.
    """
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
        date_start = str(price_index[0].date())
        date_end = str(price_index[-1].date())
    except (AttributeError, IndexError, TypeError):
        pass

    return MarkovStats(
        transmat=transmat,
        stationary=stationary,
        persistence=persistence,
        signal=signal,
        current_regime=last_regime,
        current_probs=current_probs,
        regime_counts=regime_counts_map,
        dates={"start": date_start, "end": date_end},
    )


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
    profile: bool = True,
) -> dict:
    """Run the full regime-detection pipeline and return a JSON-compatible dict.

    Profiling is enabled by default (low overhead).
    """
    t_start = time.monotonic() if profile else 0.0
    _phases: dict[str, float] = {}
    _classify_times: list[float] = []

    if engine_config is None:
        raise ValueError("engine_config is required")

    engine_name: str = getattr(engine_config, "name", None)
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(
            f"engine must be one of {sorted(ENGINE_REGISTRY.keys())}, got {engine_name!r}"
        )

    returns = _validate_prices(prices)

    config = engine_config
    resolved_n_states: int = getattr(config, "n_states", 3)
    if isinstance(resolved_n_states, str):
        resolved_n_states = 3  # auto will be resolved inside classify_pipeline

    # --- Regime classification (uniform across all engines, ADR-0017) ---
    eng = resolve_engine(config)
    classify_out = eng.run_classify(
        prices,
        ohlcv,
        returns,
        min_train,
        profile=profile,
        _phases=_phases,
        _classify_times=_classify_times,
    )

    # If engine resolved n_states internally (HMM engines set it), use that
    if classify_out.n_states is not None:
        resolved_n_states = classify_out.n_states

    markov = _build_markov_stats(classify_out.regimes, prices.index)
    forecast_1 = _probs_to_dict(
        forecast_n_steps(markov.transmat, markov.current_probs, 1)
    )
    forecast_5 = _probs_to_dict(
        forecast_n_steps(markov.transmat, markov.current_probs, 5)
    )
    forecast_20 = _probs_to_dict(
        forecast_n_steps(markov.transmat, markov.current_probs, 20)
    )

    # Walk-forward backtest (reuse pre-computed regime labels, ADR-0017)
    t_wfb = time.monotonic()
    wf_kwargs: dict = {
        "regimes": classify_out.regimes,
        "posteriors": classify_out.posteriors,
        "min_train": min_train,
        "dwell_bars": dwell_bars,
        "hysteresis_delta": hysteresis_delta,
    }
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

    engine_info = _build_engine_info(
        config,
        resolved_n_states,
        eng,
        warmup_bars=classify_out.warmup_bars,
    )

    # --- Duration forecast (compute before result assembly) ---
    df_result: dict | None = None
    if duration_forecast:
        from .duration_forecast import forecast_duration

        df_result = forecast_duration(
            classify_out.regimes, model=duration_model, prices=prices
        )

    # --- Regime transitions (always computed, issue #63) ---
    transitions = [
        ev._asdict() for ev in extract_transitions(classify_out.regimes, prices.index)
    ]

    # --- Synthesized verdict ---
    sideways_threshold = _compute_dynamic_threshold(df_result)
    verdict_out = _compute_verdict(
        markov.current_regime,
        markov.signal,
        forecast_20,
        sideways_threshold=sideways_threshold,
    )

    # --- Profiling ---
    timing: dict | None = None
    if profile:
        bic_detail = _phases.pop("bic_detail", None)
        timing = {
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

    return PipelineResult(
        source=source,
        engine=engine_name,
        dates={
            "start": markov.dates["start"],
            "end": markov.dates["end"],
        },
        current_regime={
            "name": _STATE_NAMES[markov.current_regime],
            "index": markov.current_regime,
        },
        signal=markov.signal,
        next_state_probabilities=_probs_to_dict(markov.current_probs),
        transition_matrix=markov.transmat.tolist(),
        stationary_distribution=_probs_to_dict(markov.stationary),
        persistence_diagonal=markov.persistence,
        regime_counts=markov.regime_counts,
        walk_forward=walk_forward,
        forecast={
            "1_step": forecast_1,
            "5_step": forecast_5,
            "20_step": forecast_20,
        },
        engine_info=engine_info,
        framework=_FRAMEWORK_VERSION,
        disclaimer=_DISCLAIMER,
        verdict=verdict_out,
        duration_forecast=df_result,
        regime_transitions=transitions,
        timing=timing,
    )
