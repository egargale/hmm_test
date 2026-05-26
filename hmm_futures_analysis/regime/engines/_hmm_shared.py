"""Shared HMM utilities used by both HMM engine classes."""
from __future__ import annotations

import os
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd
from hmmlearn import hmm

from ...data_processing.feature_engineering import add_features
from ...data_processing.messina_features import add_messina_features

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
) -> tuple[hmm.GaussianHMM, np.ndarray, np.ndarray]:
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0) + 1e-8
    X = ((features - center) / scale).astype(np.float64)

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

    return model, center, scale


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
