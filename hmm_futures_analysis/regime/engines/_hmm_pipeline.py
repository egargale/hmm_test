"""Pipeline-level HMM orchestration helpers.

These build on the engine-level building blocks in _hmm_engine.py to compose
the full HMM classify pipeline: BIC state selection, walk-forward classify
loop, state remapping across ticks, and dwell/hysteresis filtering.

Imported by _hmm_engine.py and pipeline.run().
ADR-0022: walk_forward.py no longer imports this module — regime replay
now lives in walk_forward._replay_regimes().
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ._hmm_engine import _fit_hmm_on_slice
from ..engine_protocol import ClassifyOutput, ClassifyResult, RegimeEngine


def select_n_states(
    features: np.ndarray,
    max_states: int = 6,
    random_state: int = 42,
    n_restarts: int = 3,
    pca_variance: float | None = None,
    profile: bool | dict = False,
) -> int:
    """Select optimal number of HMM states via Bayesian Information Criterion.

    Fits GaussianHMM for each candidate n_states in [2, max_states] with
    multiple random restarts and returns the count with the lowest BIC.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features).
    max_states : int
        Maximum number of states to evaluate.
    random_state : int
        Base random seed for reproducibility.
    n_restarts : int
        Number of random restarts per candidate state count.
    profile : bool | dict
        If a dict is provided, populates it with per-state-count timing
        under key ``"bic_detail"`` — ``{k: {"total_s": float, "restarts": int}}``.

    Returns
    -------
    int
        Optimal number of states by BIC.
    """
    import time as _time

    n = len(features)

    # Guard: cap max_states to avoid degenerate fits on short data
    effective_max = min(max_states, max(2, n // 10))

    best_bic = float("inf")
    best_k = 2

    bic_detail: dict[int, dict] = {}

    for k in range(2, effective_max + 1):
        k_start = _time.monotonic()
        for restart in range(n_restarts):
            seed = random_state + restart * 1000 + k
            model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
                features,
                n_states=k,
                random_state=seed,
                pca_variance=pca_variance,
            )
            X_norm = ((features - center) / scale).astype(np.float64)
            if pca_transform is not None:
                X_norm = pca_transform.transform(X_norm).astype(np.float64)
            log_likelihood = model.score(X_norm)
            # Effective dimensionality (reduced by PCA when active)
            d_eff = model.means_.shape[1]
            # Free params: means (k*d) + diag covariances (k*d) + transition (k*(k-1))
            n_params = k * d_eff + k * d_eff + k * (k - 1)
            bic = -2.0 * log_likelihood + n_params * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        bic_detail[k] = {
            "total_s": float(round(_time.monotonic() - k_start, 6)),
            "restarts": n_restarts,
        }

    if isinstance(profile, dict):
        profile["bic_detail"] = bic_detail

    return best_k


def _check_degenerate(
    regimes: np.ndarray, n_states: int, threshold: float = 0.05
) -> bool:
    """Check if regime distribution has a collapsed state.

    Returns True when any regime index in [0, n_states) has fewer
    than *threshold* fraction of total bars AND there are at least
    3 states to collapse.  Missing states (count=0) are degenerate.

    Pure function — no side effects.
    """
    if n_states < 3:
        return False
    total = len(regimes)
    if total == 0:
        return False
    unique, counts = np.unique(regimes, return_counts=True)
    # Build a full count array indexed by regime label
    for state_id in range(n_states):
        idx = np.searchsorted(unique, state_id)
        if idx >= len(unique) or unique[idx] != state_id:
            # State is missing entirely — 0 bars → degenerate
            return True
        if counts[idx] / total < threshold:
            return True
    return False


def _hmm_classify_pipeline(
    engine: Any,
    prices: pd.Series,
    ohlcv: pd.DataFrame | None,
    returns: pd.Series,
    min_train: int = 252,
    *,
    profile: bool = True,
    _phases: dict[str, float] | None = None,
    _classify_times: list[float] | None = None,
) -> ClassifyOutput:
    """Shared classify_pipeline implementation for all HMM engines.

    Handles: precompute → BIC auto-resolve → walk-forward classify.
    Reads ``reverse_classify`` from the engine object (set via config)
    instead of accepting a separate parameter — ADR-0023.
    """
    import time as _time

    _phases = _phases if _phases is not None else {}
    _classify_times = _classify_times if _classify_times is not None else []

    reverse = getattr(engine, 'reverse_classify', False)

    # 1. Precompute features
    precomputed = None
    t_pc = _time.monotonic()
    try:
        precomputed = engine.precompute(ohlcv)
    except (ValueError, RuntimeError, KeyError):
        precomputed = None
    if profile:
        _phases["precompute"] = float(round(_time.monotonic() - t_pc, 6))
    if precomputed is None:
        raise ValueError(
            f"{type(engine).__name__} failed to precompute features from OHLCV data"
        )

    # 2. Resolve n_states='auto' via BIC
    if isinstance(engine.n_states, str) and engine.n_states == "auto":
        t_bic = _time.monotonic()
        resolved_n_states = select_n_states(
            precomputed.dropna().to_numpy(dtype=np.float64),
            max_states=6,
            profile=_phases if profile else False,
        )
        if profile:
            _phases["bic_select_n_states"] = float(round(_time.monotonic() - t_bic, 6))
    elif isinstance(engine.n_states, int):
        resolved_n_states = engine.n_states
    else:
        raise ValueError(f"n_states must be int or 'auto', got {engine.n_states!r}")

    # Store resolved n_states for classify() to use without mutating n_states
    engine._n_states_resolved = resolved_n_states  # type: ignore[attr-defined]

    # 2b. Degenerate-fit pre-check: auto-downgrade to n_states=2
    original_n_states = resolved_n_states
    degenerate_auto_recovered = False
    if resolved_n_states >= 3:
        features_full = precomputed.dropna().to_numpy(dtype=np.float64)
        try:
            model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
                features_full,
                n_states=resolved_n_states,
                random_state=42,
                pca_variance=getattr(engine, "pca_variance", None),
            )
            X_norm = ((features_full - center) / scale).astype(np.float64)
            if pca_transform is not None:
                X_norm = pca_transform.transform(X_norm).astype(np.float64)
            precheck_labels = model.predict(X_norm)
            if _check_degenerate(precheck_labels, n_states=resolved_n_states):
                resolved_n_states = 2
                degenerate_auto_recovered = True
                print(
                    "  [degenerate] 3-state fit collapsed."
                    " Auto-downgrading to n_states=2.",
                    file=sys.stderr,
                )
        except (ValueError, RuntimeError):
            pass  # if pre-check fit fails, proceed with original n_states

    # 3. Walk-forward classify
    n = len(returns)
    regimes = np.ones(n, dtype=int)
    posteriors_all = np.zeros((n, 3), dtype=float)

    # Side-channel for mid-stream degeneration recovery metadata
    wf_recovery: dict[str, object] = {}

    t_wf = _time.monotonic()
    for t, result in _walk_forward_classify(
        returns,
        eng=engine,
        precomputed=precomputed,
        min_train=min_train,
        profile=profile,
        _phases=_phases,
        _classify_times=_classify_times,
        _wf_recovery=wf_recovery,
        reverse=reverse,
        n_states_active=resolved_n_states,
    ):
        regimes[t] = result.regime
        if result.posteriors is not None:
            posteriors_all[t] = result.posteriors
    if profile:
        _phases["walk_forward_classify"] = float(round(_time.monotonic() - t_wf, 6))

    engine_info = {}
    if degenerate_auto_recovered:
        engine_info["degenerate_auto_recovered"] = True
        engine_info["original_n_states"] = original_n_states
    if wf_recovery.get("mid_stream_recovery"):
        engine_info["walk_forward_degenerate_recovery"] = True
        engine_info["degeneration_bar"] = wf_recovery["degeneration_bar"]

    return ClassifyOutput(
        regimes=regimes,
        posteriors=posteriors_all,
        last_regime=int(regimes[-1]),
        warmup_bars=min_train,
        n_states=resolved_n_states,
        engine_info=engine_info if engine_info else None,
        reverse_classify=reverse,
    )


def _walk_forward_classify(
    returns: pd.Series,
    *,
    eng: RegimeEngine | None = None,
    precomputed: pd.DataFrame | None = None,
    regimes: np.ndarray | None = None,
    min_train: int = 252,
    profile: bool = True,
    _phases: dict[str, float] | None = None,
    _classify_times: list[float] | None = None,
    _wf_recovery: dict[str, object] | None = None,
    reverse: bool = False,
    n_states_active: int = 3,
) -> Iterator[tuple[int, ClassifyResult]]:
    """Walk-forward classify generator, yielding (t, ClassifyResult) per bar.

    Two mutually-exclusive input modes determined by which kwargs are set:

    - **regimes** is not None: replay pre-computed regime labels without
      calling ``eng.classify``.  Yields a ``ClassifyResult(regime=…)`` with
      ``means=None, posteriors=None``.
      .. deprecated:: ADR-0022
         Production code now uses ``walk_forward._replay_regimes()`` for
         mode-1 replay.  This path is retained for test compatibility.
    - **precomputed** is not None: adaptive skip-N refit calling
      ``eng.classify(precomputed.iloc[:t], prev_means=…)``.  Carries the
      last result forward on non-refit bars.

    When *reverse* is True, the iteration direction flips: ``t`` runs from
    ``n-1`` down to ``min_train`` and the feature slice becomes
    ``precomputed.iloc[t:]`` (backward-expanding window).  The regime
    array is filled by bar index so no caller flip is needed.

    Yields
    ------
    (t, ClassifyResult) for every bar t in [min_train, len(returns)).
    """
    _phases = _phases if _phases is not None else {}
    _classify_times = _classify_times if _classify_times is not None else []

    n = len(returns)

    if regimes is not None:
        # Mode 1: pre-computed regime labels — replay without classifying.
        for t in range(min_train, n):
            yield t, ClassifyResult(regime=int(regimes[t]))
        return

    # Reverse mode: no-op for precomputed-regime replay (threshold engine).
    # When precomputed features are provided, reverse changes iteration
    # direction and feature slicing.

    # Mode 2: precomputed features, adaptive skip-N refit (requires engine).
    if eng is None:
        raise ValueError("eng is required when regimes is not set")
    if precomputed is None:
        raise ValueError("precomputed is required when regimes is not set")

    _wf_recovery = _wf_recovery if _wf_recovery is not None else {}

    prev_means: np.ndarray | None = None
    last_result = ClassifyResult(regime=1)

    refit_every = eng.default_refit_every

    n_refits = 0
    t_start_wf = time.monotonic()
    mid_stream_degenerate = False
    # Track yielded regimes for cumulative degeneration check
    yielded_regimes: list[int] = []

    if reverse:
        bar_range = range(n - 1, min_train - 1, -1)
        first_t = n - 1
        last_t = min_train
    else:
        bar_range = range(min_train, n)
        first_t = min_train
        last_t = n - 1

    for t in bar_range:
        refit_now = (t == first_t) or (abs(t - first_t) % refit_every == 0) or (t == last_t)
        if refit_now:
            n_refits += 1
            features_slice = precomputed.iloc[t:] if reverse else precomputed.iloc[:t]
            try:
                t_cls_start = time.monotonic() if profile else 0.0
                result = eng.classify(features_slice, prev_means=prev_means)
                if profile:
                    _classify_times.append(time.monotonic() - t_cls_start)
                prev_means = result.means
                last_result = result
            except (ValueError, RuntimeError):
                pass

            # Mid-stream degeneration check (Issue #98, ADR-0018)
            # After each refit, check if cumulative regime distribution
            # has collapsed. Only check when n_states_active >= 3 and we have
            # enough classified bars to make a meaningful assessment.
            if (
                not mid_stream_degenerate
                and n_states_active >= 3
                and len(yielded_regimes) >= 20
            ):
                cumulative = np.array(yielded_regimes)
                if _check_degenerate(cumulative, n_states=n_states_active):
                    mid_stream_degenerate = True
                    n_states_active = 2
                    print(
                        f"[walk-forward] mid-stream degeneration detected"
                        f" at bar {t}. Downgrading remaining refits"
                        f" to n_states=2.",
                        file=sys.stderr,
                    )
                    _wf_recovery["mid_stream_recovery"] = True
                    _wf_recovery["degeneration_bar"] = t

            # Progress: log every 20 refits and on the final refit
            total_refits = 1 + (n - 1 - min_train) // refit_every
            if n_refits % 20 == 0 or n_refits == total_refits:
                elapsed = time.monotonic() - t_start_wf
                print(
                    f"  [walk-forward] refit {n_refits}/{total_refits}"
                    f"  bar {t}/{n}  elapsed {elapsed:.1f}s",
                    file=sys.stderr,
                )
        yielded_regimes.append(last_result.regime)
        yield t, last_result
