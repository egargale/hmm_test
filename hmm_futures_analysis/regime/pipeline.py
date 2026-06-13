"""Regime detection pipeline — callable from Python or CLI.

This module owns the full analysis pipeline: threshold-based regime
classification, transition matrix, statistics, forecasts, and walk-forward
backtest.  Supports five engines: threshold (fast, close-only), hmm
(HMM + ~50 generic features), messina (HMM + 19 Messina features),
robust_hmm (HMM + outlier-resistant emissions), and fshmm (HMM +
feature saliency).
"""

from __future__ import annotations

import time

import dataclasses

import numpy as np
import pandas as pd

from ._result_assembly import (  # re-exported for backward compat
    PipelineResult,
    _FRAMEWORK_VERSION,
    _STATE_NAMES,
    _assemble_result,
    _compute_dynamic_threshold,
    _compute_verdict,
    _nan_to_none,
    _probs_to_dict,
)
from .engine_protocol import (
    ENGINE_REGISTRY,
    resolve_engine,
)
from .markov_chain import (
    build_transition_matrix,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
)
from .regime_transitions import extract_transitions
from .walk_forward import walk_forward_backtest

__all__ = [
    "PipelineResult",
    "_nan_to_none",
    "_probs_to_dict",
    "_compute_verdict",
    "_compute_dynamic_threshold",
    "_assemble_result",
    "_STATE_NAMES",
    "_FRAMEWORK_VERSION",
]


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


def detect_degenerate_fit(
    regime_counts: dict[str, int],
    n_bars: int,
    n_features: int,
    engine_name: str,
    n_states: int,
    *,
    n_states_was_auto: bool = False,
    hmm_regime_counts: dict[str, int] | None = None,
    degenerate_auto_recovered: bool = False,
    original_n_states: int | None = None,
) -> dict[str, object]:
    """Detect degenerate HMM fits per ADR-0018.

    Pure function: inspects regime balance, data dimensions, and engine
    identity. Returns diagnostic fields to merge into engine_info.
    Detection only — no regime assignment changes.

    When ``degenerate_auto_recovered`` is True, the model has already been
    auto-downgraded to a healthy state. In that case audit fields are emitted
    (``degenerate_auto_recovered``, ``original_n_states``, ``recovery_method``)
    instead of ``degenerate_fit: true`` — unless the recovered model is *still*
    degenerate (defensive edge case).
    """
    diagnostics: dict[str, object] = {}
    hmm_with_collapse = {"hmm", "messina", "robust_hmm"}
    hmm_all = hmm_with_collapse | {"fshmm"}

    # --- Mode 2: Low-data warning ---
    if engine_name in hmm_all:
        min_bars = 4 * n_features * n_states
        if n_bars < min_bars:
            diagnostics["low_data_warning"] = True
            diagnostics["low_data_caveat"] = (
                f"{n_bars} bars with {n_features} features × {n_states} states; "
                f"recommend minimum {min_bars} bars"
            )

    # --- Auto-recovery audit (Issue #95) ---
    if degenerate_auto_recovered:
        diagnostics["degenerate_auto_recovered"] = True
        diagnostics["original_n_states"] = original_n_states
        diagnostics["recovery_method"] = f"auto-downgrade to n_states={n_states}"

    # --- Mode 1: State collapse (hmm, messina, robust_hmm) ---
    if engine_name in hmm_with_collapse:
        total = sum(regime_counts.values())
        if total > 0:
            min_fraction = 0.05
            collapsed = [
                (name, count)
                for name, count in regime_counts.items()
                if count / total < min_fraction
            ]
            if collapsed:
                # If auto-recovery already happened, only flag degenerate_fit
                # if the recovered model is STILL degenerate (defensive).
                if not degenerate_auto_recovered:
                    diagnostics["degenerate_fit"] = True
                    parts = []
                    for name, count in collapsed:
                        pct = count / total * 100
                        parts.append(
                            f"{name} state has {pct:.1f}% of bars ({count}/{total})"
                        )
                    caveat = "; ".join(parts) + "; model is effectively "
                    n_effective = n_states - len(collapsed)
                    caveat += f"{n_effective}-regime"
                    if not n_states_was_auto:
                        caveat += ". Consider --n-states auto"
                    diagnostics["degenerate_caveat"] = caveat
                else:
                    # Auto-recovered but still degenerate — defensive warning
                    diagnostics["degenerate_fit"] = True
                    parts = []
                    for name, count in collapsed:
                        pct = count / total * 100
                        parts.append(
                            f"{name} state has {pct:.1f}% of bars ({count}/{total})"
                        )
                    caveat = (
                        "; ".join(parts) + " (post-recovery); model is effectively "
                    )
                    n_effective = n_states - len(collapsed)
                    caveat += f"{n_effective}-regime"
                    diagnostics["degenerate_caveat"] = caveat

    # --- Mode 3: Over-robustness (robust_hmm vs hmm) ---
    if engine_name == "robust_hmm" and hmm_regime_counts is not None:
        total_robust = sum(regime_counts.values())
        total_hmm = sum(hmm_regime_counts.values())
        if total_robust > 0 and total_hmm > 0:
            all_below_10pct = all(
                abs(
                    regime_counts.get(k, 0) / total_robust
                    - hmm_regime_counts.get(k, 0) / total_hmm
                )
                < 0.10
                for k in ("bear", "sideways", "bull")
            )
            if all_below_10pct:
                diagnostics["over_robustness"] = True
                diagnostics["over_robustness_caveat"] = (
                    "regime counts differ from hmm by <10% on all states; "
                    "robust correction not adding value"
                )

    return diagnostics


def _build_engine_info(
    engine_config: object,
    resolved_n_states: int,
    classify_out: object | None = None,
) -> dict[str, object]:
    """Build engine_info dict from config and classify output.

    Constructs base info (method, features, n_states) from config,
    then merges engine metadata from ``classify_out.engine_info``
    if the engine populated it.
    """
    engine = getattr(engine_config, "name", None)
    features_label: str = getattr(engine_config, "features", engine)

    info: dict[str, object] = {
        "method": engine,
        "features": features_label,
        "n_states": resolved_n_states,
    }
    engine_meta = getattr(classify_out, "engine_info", None)
    if engine_meta:
        info.update(engine_meta)
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
    dwell_bars: int | str = 0,
    hysteresis_delta: float | str = 0.0,
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

    # Resolve 'auto' filter params to engine config defaults
    if dwell_bars == "auto":
        dwell_bars = getattr(engine_config, "default_dwell_bars", 0)
    if hysteresis_delta == "auto":
        hysteresis_delta = getattr(engine_config, "default_hysteresis_delta", 0.0)

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
        classify_out,
    )

    # --- Lookahead bias warning for reverse-classify (Issue #102) ---
    reverse_classify = getattr(engine_config, "reverse_classify", False)
    if reverse_classify and classify_out.reverse_classify:
        engine_info["reverse_classify"] = True
        engine_info["lookahead_bias_warning"] = True
        engine_info["lookahead_bias_caveat"] = (
            "Walk-forward backtest contains lookahead bias: regime at bar t "
            "is partially informed by data from t+1 … n. "
            "Use for display only, not backtest-driven decisions."
        )

    # --- Degenerate-fit detection (ADR-0018, Issue #95 audit) ---
    # Feature count is read off the engine's precomputed output (FeatureSet
    # source of truth) rather than a magic-number lookup table.
    n_features = classify_out.n_features if classify_out.n_features is not None else 1
    n_states_was_auto = isinstance(getattr(config, "n_states", 3), str)
    deg_diagnostics = detect_degenerate_fit(
        regime_counts=markov.regime_counts,
        n_bars=len(classify_out.regimes),
        n_features=n_features,
        engine_name=engine_name,
        n_states=resolved_n_states,
        n_states_was_auto=n_states_was_auto,
        degenerate_auto_recovered=bool(
            engine_info.get("degenerate_auto_recovered", False)
        ),
        original_n_states=engine_info.get("original_n_states"),  # type: ignore[arg-type]
    )
    engine_info.update(deg_diagnostics)

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

    return _assemble_result(
        source=source,
        engine_name=engine_name,
        markov=markov,
        walk_forward=walk_forward,
        engine_info=engine_info,
        timing=timing,
        duration_forecast_result=df_result,
        regime_transitions=transitions,
    )
