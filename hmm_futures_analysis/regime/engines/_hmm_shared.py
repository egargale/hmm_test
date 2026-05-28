"""Shared HMM utilities used by both HMM engine classes."""
from __future__ import annotations

import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hmmlearn import hmm

from ...data_processing.feature_engineering import add_features
from ...data_processing.messina_features import add_messina_features

if TYPE_CHECKING:
    from sklearn.decomposition import PCA

_MESSINA_COLS = [
    "log_ret", "sma_200", "sma_13", "atr_20",
    "adx_14", "adx_inflection",
    "di_plus_14", "di_minus_14", "di_spread",
    "vstop", "vstop_trend", "vstop_interaction",
    "price_sma200_ratio", "price_vstop_ratio",
    "price_vstop_gap_atr", "sma200_distance_atr",
    "volume_ratio", "true_range_pct", "kdj_j",
]


def engineer_features(
    data: pd.DataFrame, use_messina: bool
) -> pd.DataFrame:
    if use_messina:
        df = add_messina_features(data)
        cols = [c for c in _MESSINA_COLS if c in df.columns]
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
            n_components=pca_variance, svd_solver="full", random_state=random_state,
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


def select_n_states(
    features: np.ndarray,
    max_states: int = 6,
    random_state: int = 42,
    n_restarts: int = 3,
    pca_variance: float | None = None,
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

    Returns
    -------
    int
        Optimal number of states by BIC.
    """
    n = len(features)
    d = features.shape[1]

    # Guard: cap max_states to avoid degenerate fits on short data
    effective_max = min(max_states, max(2, n // 10))

    best_bic = float("inf")
    best_k = 2

    for k in range(2, effective_max + 1):
        for restart in range(n_restarts):
            seed = random_state + restart * 1000 + k
            model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
                features, n_states=k, random_state=seed,
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

    return best_k


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
