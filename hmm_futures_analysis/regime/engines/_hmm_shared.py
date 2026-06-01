"""Shared HMM utilities used by both HMM engine classes."""

from __future__ import annotations

import os
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hmmlearn import hmm

from ...data_processing.feature_engineering import add_features
from ...data_processing.messina_features import MESSINA_FEATURE_COLUMNS
from ...data_processing.messina_features import add_messina_features
from ..engine_protocol import ClassifyOutput, ClassifyResult, RegimeEngine

if TYPE_CHECKING:
    from sklearn.decomposition import PCA


def engineer_features(data: pd.DataFrame, use_messina: bool) -> pd.DataFrame:
    if use_messina:
        df = add_messina_features(data)
        cols = [c for c in MESSINA_FEATURE_COLUMNS if c in df.columns]
    else:
        df = add_features(data, min_periods=10)
        df = df.dropna(axis=1, how="all")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ohlcv_set = {"open", "high", "low", "close", "volume"}
        cols = [c for c in numeric_cols if c not in ohlcv_set]

    if not cols:
        raise ValueError("No numeric features after engineering")

    return df[cols]


def _fit_hmm_on_slice(
    features: np.ndarray,
    n_states: int = 3,
    random_state: int = 42,
    pca_variance: float | None = None,
) -> tuple[hmm.GaussianHMM, np.ndarray, np.ndarray, int | None, PCA | None]:
    """Fit GaussianHMM on a feature slice, with optional PCA whitening.

    Pipeline: raw features → z-score → (optional PCA) → fit.

    Returns (model, center, scale, pca_n_components, pca_transform).
    When ``pca_variance`` is None, returns (model, center, scale, None, None)
    preserving backward compatibility in structure.
    """
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0) + 1e-8
    X = ((features - center) / scale).astype(np.float64)

    # Optional PCA whitening (model layer, per ADR-0005)
    pca_n_components_used: int | None = None
    pca_transform: PCA | None = None
    if pca_variance is not None:
        from sklearn.decomposition import PCA as _PCA

        pca_transform = _PCA(
            n_components=pca_variance,
            svd_solver="full",
            random_state=random_state,
        )
        X = pca_transform.fit_transform(X).astype(np.float64)
        pca_n_components_used = int(pca_transform.n_components_)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=30,
                    tol=1e-4,
                    random_state=random_state,
                    params="stmc",
                    init_params="stmc",
                    verbose=False,
                )
                model.fit(X)

    return model, center, scale, pca_n_components_used, pca_transform


def _huber_correction(
    model: hmm.GaussianHMM,
    X: np.ndarray,
    posteriors: np.ndarray,
    k: float = 1.345,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> None:
    """Post-hoc Huber IRLS correction of emission parameters."""
    for s in range(model.n_components):
        resp = posteriors[:, s]
        mu = model.means_[s].copy()
        var = np.diag(model.covars_[s]).copy()

        for _ in range(max_iter):
            prev_mu = mu.copy()
            diff = X - mu
            mahal = np.sqrt(np.sum(diff**2 / (var + 1e-8), axis=1))
            w = np.ones(len(X))
            mask = mahal > k
            w[mask] = k / (mahal[mask] + 1e-8)
            combined = resp * w
            total = combined.sum() + 1e-8
            mu = (combined[:, np.newaxis] * X).sum(axis=0) / total
            diff = X - mu
            var = (combined[:, np.newaxis] * diff**2).sum(axis=0) / total
            if np.max(np.abs(mu - prev_mu)) < tol:
                break

        model.means_[s] = mu
        model.covars_[s] = np.diag(var)


_MCD_MAX_POINTS = 200


def _mcd_correction(
    model: hmm.GaussianHMM,
    X: np.ndarray,
    posteriors: np.ndarray,
    random_state: int = 0,
) -> None:
    """Post-hoc MinCovDet correction of emission parameters."""
    from sklearn.covariance import MinCovDet

    rng = np.random.RandomState(random_state)
    for s in range(model.n_components):
        mask = posteriors[:, s] > 0.3
        n_pts = mask.sum()
        n_features = X.shape[1]
        if n_pts < max(model.n_components + 1, n_features + 1, 5):
            continue
        try:
            X_state = X[mask]
            if n_pts > _MCD_MAX_POINTS:
                idx = rng.choice(n_pts, size=_MCD_MAX_POINTS, replace=False)
                X_state = X_state[idx]
            mcd = MinCovDet(random_state=random_state)
            mcd.fit(X_state)
            model.means_[s] = mcd.location_
            model.covars_[s] = np.diag(mcd.covariance_)
        except (ValueError, np.linalg.LinAlgError):
            continue


def robust_fit_gaussian_hmm(
    features: np.ndarray,
    n_states: int = 3,
    random_state: int = 42,
    pca_variance: float | None = None,
    robust_method: str = "huber",
) -> tuple[hmm.GaussianHMM, np.ndarray, np.ndarray, int | None, PCA | None]:
    """Fit GaussianHMM with post-hoc robust emission correction.

    Same return shape as _fit_hmm_on_slice for drop-in compatibility.
    """
    model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
        features,
        n_states=n_states,
        random_state=random_state,
        pca_variance=pca_variance,
    )

    X = ((features - center) / scale).astype(np.float64)
    if pca_transform is not None:
        X = pca_transform.transform(X).astype(np.float64)

    posteriors = model.predict_proba(X)

    if robust_method == "huber":
        _huber_correction(model, X, posteriors)
    elif robust_method == "mcd":
        _mcd_correction(model, X, posteriors, random_state=random_state)

    return model, center, scale, pca_n, pca_transform


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


def _match_states(
    new_means: np.ndarray,
    prev_means: np.ndarray,
) -> dict[int, int]:
    assignment: dict[int, int] = {}
    used: set[int] = set()

    for old_idx in range(len(prev_means)):
        best_new = -1
        best_dist = float("inf")
        for new_idx in range(len(new_means)):
            if new_idx in used:
                continue
            dist = np.linalg.norm(new_means[new_idx] - prev_means[old_idx])
            if dist < best_dist:
                best_dist = dist
                best_new = new_idx
        if best_new >= 0:
            assignment[best_new] = old_idx
            used.add(best_new)

    return assignment


def _classify_hmm_slice(
    model: hmm.GaussianHMM,
    X_last: np.ndarray,
    n_states: int,
    prev_means: np.ndarray | None = None,
) -> ClassifyResult:
    """Shared post-fit classify pipeline for HMM engines.

    Pipeline: model.predict(X_last) → label map (sort means ascending,
    collapse to 3 regimes if n_states > 3) → posteriors reorder/aggregate
    → _remap_to_prev_states if prev_means given.

    Parameters
    ----------
    model : pre-fit GaussianHMM
    X_last : already-normalized last observation, shape (1, n_features)
        Caller is responsible for z-scoring and optional PCA transform.
    n_states : number of HMM states
    prev_means : previous-cycle means for state remapping, or None
    """
    means = model.means_

    raw_state = model.predict(X_last.astype(np.float64))[0]
    posteriors = model.predict_proba(X_last.astype(np.float64))[-1]

    # Map HMM states to regime indices (0=bear, 1=sideways, 2=bull)
    # based on ascending mean return order, collapsed to 3 buckets
    state_means = means[:, 0]
    order = np.argsort(state_means)
    if n_states <= 3:
        label_map = {int(order[i]): i for i in range(len(order))}
    else:
        n = len(order)
        label_map = {}
        for i, state_idx in enumerate(order):
            regime = min(2, i * 3 // n)
            label_map[int(state_idx)] = regime

    regime = label_map.get(int(raw_state), 1)

    # Reorder posteriors to match regime labels (0=bear, 1=sideways, 2=bull)
    if n_states <= 3:
        reordered = np.zeros(n_states)
        for state_idx in range(n_states):
            reordered[label_map[state_idx]] = posteriors[state_idx]
        posteriors = reordered
    else:
        # Aggregate posteriors by regime bucket
        agg = np.zeros(3)
        for state_idx in range(n_states):
            agg[label_map[state_idx]] += posteriors[state_idx]
        posteriors = agg

    if prev_means is not None:
        regime = _remap_to_prev_states(means, raw_state, prev_means, default=regime)

    return ClassifyResult(regime=int(regime), means=means, posteriors=posteriors)


def _remap_to_prev_states(
    means: np.ndarray, raw_state: int, prev_means: np.ndarray, *, default: int = 0
) -> int:
    """Remap a raw HMM state to the regime index from the previous cycle.

    Sorts prev_means by column 0, builds a label map (identity for ≤3 states,
    collapsed to 3 regimes for >3), then maps raw_state through _match_states
    and the prev label map.

    Args:
        means: Current-cycle HMM means (n_states × n_features).
        raw_state: Predicted raw latent state index.
        prev_means: Previous-cycle HMM means (prev_n × n_features).
        default: Fallback regime when raw_state has no match.

    Returns:
        Remapped regime index (0=bear, 1=sideways, 2=bull).
    """
    prev_order = np.argsort(prev_means[:, 0])
    prev_n = len(prev_order)

    if prev_n <= 3:
        prev_label_map = {int(prev_order[i]): i for i in range(prev_n)}
    else:
        prev_label_map = {}
        for i, si in enumerate(prev_order):
            prev_label_map[int(si)] = min(2, i * 3 // prev_n)

    assignment = _match_states(means, prev_means)
    old_state = assignment.get(int(raw_state))
    if old_state is not None:
        return prev_label_map.get(old_state, default)

    return default


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

    Three mutually-exclusive input modes determined by which kwargs are set:

    - **regimes** is not None: replay pre-computed regime labels without
      calling ``eng.classify``.  Yields a ``ClassifyResult(regime=…)`` with
      ``means=None, posteriors=None``.
    - **precomputed** is not None: adaptive skip-N refit calling
      ``eng.classify(precomputed.iloc[:t], prev_means=…)``.  Carries the
      last result forward on non-refit bars.
    - Neither: per-bar ``eng.classify(returns.iloc[:t])``.

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

    # Modes 2 & 3: require an engine.
    if eng is None:
        raise ValueError("eng is required when regimes is not set")

    if precomputed is not None:
        # Mode 2: precomputed features, adaptive skip-N refit.
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
    else:
        # Mode 3: raw returns, per-bar classify.
        last_result = ClassifyResult(regime=1)
        for t in range(min_train, n):
            try:
                result = eng.classify(returns.iloc[:t])
                last_result = result
            except (ValueError, RuntimeError):
                pass
            yield t, last_result
