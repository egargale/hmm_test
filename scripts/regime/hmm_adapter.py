"""HMM adapter — model fitting, state prediction, regime labeling.

Provides functions for training Hidden Markov Models on financial data,
labeling latent states as market regimes (bear/sideways/bull), and
walk-forward-compatible per-slice fitting with parameter matching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm

from data_processing.feature_engineering import add_features
from data_processing.messina_features import add_messina_features

# Feature column names for Messina mode
_MESSINA_COLS = [
    "log_ret", "sma_200", "sma_13", "atr_20",
    "adx_14", "adx_inflection",
    "di_plus_14", "di_minus_14", "di_spread",
    "vstop", "vstop_trend", "vstop_interaction",
    "price_sma200_ratio", "price_vstop_ratio",
    "price_vstop_gap_atr", "sma200_distance_atr",
    "volume_ratio", "true_range_pct", "kdj_j",
]


def _engineer_features(
    data: pd.DataFrame, use_messina: bool
) -> pd.DataFrame:
    """Apply feature engineering and return clean numeric feature DataFrame."""
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
    """Fit a GaussianHMM on a feature slice, return model and standardization stats.

    Returns (model, center, scale) where center and scale were used to
    standardize the features before fitting.
    """
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0) + 1e-8
    X = (features - center) / scale

    # Ensure float64
    X = X.astype(np.float64)

    import os
    import warnings
    from contextlib import redirect_stderr, redirect_stdout

    # Suppress hmmlearn convergence noise during walk-forward fitting
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
    """Match new HMM state indices to previous state indices.

    Uses nearest-neighbor Euclidean distance in mean space to preserve
    label continuity across consecutive fits.

    Returns mapping: {new_state_index: old_state_index}.
    """
    assignment: dict[int, int] = {}
    used: set[int] = set()

    # Greedy assignment: for each old state, find closest unmatched new state
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


def hmm_state_from_slice(
    data: pd.DataFrame,
    *,
    n_states: int = 3,
    use_messina: bool = False,
    return_features: bool = False,
    prev_means: np.ndarray | None = None,
    precomputed: bool = False,
) -> dict:
    """HMM regime detection for a data slice or precomputed feature set.

    Two modes:

    1. ``return_features=True`` (precompute pass):
       Compute features on the full ``data`` DataFrame and return the
       feature DataFrame keyed as ``"features"``.

    2. ``return_features=False`` (per-bar fit):
       Fit a GaussianHMM on ``data`` (which should be a feature slice
       if ``precomputed=True``, or raw OHLCV otherwise), predict the
       state at the last bar, and return:

       - ``"regime"``: int (0=bear, 1=sideways, 2=bull)
       - ``"means"``: np.ndarray shape (n_states, n_features) in
          standardized space — for parameter matching across fits.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data (if precomputed=False) or feature slice (if precomputed=True).
    n_states : int
        Number of HMM latent states.
    use_messina : bool
        If True, use Messina features (ignored when precomputed=True).
    return_features : bool
        If True, return the feature DataFrame instead of fitting.
    prev_means : np.ndarray | None
        Previous fit's mean vectors for parameter matching.
    precomputed : bool
        If True, ``data`` is already a feature DataFrame — skip engineering.

    Returns
    -------
    dict
        See mode descriptions above.
    """
    if return_features:
        features = _engineer_features(data, use_messina=use_messina)
        return {"features": features}

    # Feature engineering (if needed)
    if not precomputed:
        features_df = _engineer_features(data, use_messina=use_messina)
    else:
        features_df = data

    # Clean: drop NaN rows
    features_clean = features_df.dropna()
    if len(features_clean) < n_states + 1:
        raise ValueError(
            f"Not enough clean rows ({len(features_clean)}) "
            f"for {n_states} HMM states"
        )

    features_arr = features_clean.to_numpy(dtype=np.float64)

    # Fit HMM on cleaned slice
    model, center, scale = _fit_hmm_on_slice(features_arr, n_states=n_states)

    # Get means in standardized space
    means = model.means_  # shape (n_states, n_features)

    # Predict state at last bar
    last_features = (features_arr[-1:] - center) / scale
    raw_state = model.predict(last_features.astype(np.float64))[0]

    # Determine regime mapping
    if prev_means is not None:
        # Parameter matching: map new states → old states
        assignment = _match_states(means, prev_means)
        matched_state = assignment.get(int(raw_state), 1)  # default sideways
        # matched_state is the old regime index (0=bear, 1=sideways, 2=bull)
        regime = matched_state
    else:
        # First fit: sort by log_ret mean in standardized space
        # (division by positive scale preserves ordering)
        state_means = means[:, 0]  # log_ret is always column 0
        order = np.argsort(state_means)  # ascending: bear, sideways, bull
        label_map = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
        regime = label_map.get(int(raw_state), 1)

    return {
        "regime": regime,
        "means": means,
    }
