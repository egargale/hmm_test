"""Pipeline-level HMM orchestration helpers.

These build on the engine-level building blocks in _hmm_engine.py to compose
the full HMM classify pipeline: BIC state selection, walk-forward classify
loop, state remapping across ticks, and dwell/hysteresis filtering.

Imported by pipeline.run() and walk_forward.py.
"""

from __future__ import annotations

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
    """
    import time as _time

    _phases = _phases if _phases is not None else {}
    _classify_times = _classify_times if _classify_times is not None else []

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
        resolved = select_n_states(
            precomputed.dropna().to_numpy(dtype=np.float64),
            max_states=6,
            profile=_phases if profile else False,
        )
        if profile:
            _phases["bic_select_n_states"] = float(round(_time.monotonic() - t_bic, 6))
        engine.n_states = resolved
    elif isinstance(engine.n_states, int):
        pass  # already resolved
    else:
        raise ValueError(f"n_states must be int or 'auto', got {engine.n_states!r}")

    # 3. Walk-forward classify
    n = len(returns)
    regimes = np.ones(n, dtype=int)
    posteriors_all = np.zeros((n, 3), dtype=float)

    t_wf = _time.monotonic()
    for t, result in _walk_forward_classify(
        returns,
        eng=engine,
        precomputed=precomputed,
        min_train=min_train,
        profile=profile,
        _phases=_phases,
        _classify_times=_classify_times,
    ):
        regimes[t] = result.regime
        if result.posteriors is not None:
            posteriors_all[t] = result.posteriors
    if profile:
        _phases["walk_forward_classify"] = float(round(_time.monotonic() - t_wf, 6))

    return ClassifyOutput(
        regimes=regimes,
        posteriors=posteriors_all,
        last_regime=int(regimes[-1]),
        warmup_bars=min_train,
        engine_instance=engine,
        n_states=engine.n_states,
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
) -> Iterator[tuple[int, ClassifyResult]]:
    """Walk-forward classify generator, yielding (t, ClassifyResult) per bar.

    Two mutually-exclusive input modes determined by which kwargs are set:

    - **regimes** is not None: replay pre-computed regime labels without
      calling ``eng.classify``.  Yields a ``ClassifyResult(regime=…)`` with
      ``means=None, posteriors=None``.
    - **precomputed** is not None: adaptive skip-N refit calling
      ``eng.classify(precomputed.iloc[:t], prev_means=…)``.  Carries the
      last result forward on non-refit bars.

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

    # Mode 2: precomputed features, adaptive skip-N refit (requires engine).
    if eng is None:
        raise ValueError("eng is required when regimes is not set")
    if precomputed is None:
        raise ValueError("precomputed is required when regimes is not set")

    prev_means: np.ndarray | None = None
    last_result = ClassifyResult(regime=1)

    refit_every = max(1, (n - min_train) // 100)
    refit_every = min(refit_every, 20)

    for t in range(min_train, n):
        refit_now = (t == min_train) or ((t - min_train) % refit_every == 0)
        if refit_now:
            features_slice = precomputed.iloc[:t]
            try:
                t_cls_start = time.monotonic() if profile else 0.0
                result = eng.classify(features_slice, prev_means=prev_means)
                if profile:
                    _classify_times.append(time.monotonic() - t_cls_start)
                prev_means = result.means
                last_result = result
            except (ValueError, RuntimeError):
                pass
        yield t, last_result
