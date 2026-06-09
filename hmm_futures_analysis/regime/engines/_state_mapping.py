"""HMM latent state → regime mapping.

Maps arbitrary HMM state indices to the canonical regime buckets
(Bear=0, Sideways=1, Bull=2) by sorting the emission means on the
return-signal dimension.  Also provides walk-forward label continuity
via nearest-neighbor state matching across consecutive refits.

Extracted from ``_hmm_engine.py`` per ADR-0009 (shared HMM utilities).
"""

from __future__ import annotations

import numpy as np


def build_label_map(
    means: np.ndarray,
    sort_col: int,
) -> dict[int, int]:
    """Build a mapping from HMM latent state index → regime bucket.

    Sorts states by ``means[:, sort_col]`` ascending (the return-signal
    dimension) and collapses into 3 regime buckets regardless of the
    actual number of states:

    * 1 state  → Sideways (1)
    * 2 states → Bear (0), Bull (2)
    * 3 states → Bear (0), Sideways (1), Bull (2)
    * N > 3    → evenly distributed across {0, 1, 2}

    Parameters
    ----------
    means : ndarray, shape (n_states, n_features)
        Emission means from a fitted GaussianHMM.
    sort_col : int
        Column index to sort by (0 = log_ret without PCA, or the
        PCA component most correlated with returns).

    Returns
    -------
    dict[int, int]
        ``{raw_state_index: regime_bucket}`` where regime_bucket ∈ {0, 1, 2}.
    """
    order = np.argsort(means[:, sort_col])
    n = len(order)

    if n == 1:
        return {int(order[0]): 1}
    if n == 2:
        return {int(order[0]): 0, int(order[1]): 2}
    if n == 3:
        return {int(order[i]): i for i in range(3)}

    return {int(si): min(2, i * 3 // n) for i, si in enumerate(order)}


def _match_states(
    new_means: np.ndarray,
    prev_means: np.ndarray,
) -> dict[int, int]:
    """Greedy nearest-neighbor state assignment across refit cycles.

    For each previous-state mean, finds the closest current-state mean
    (by Euclidean distance) and records the assignment
    ``{new_state_idx: old_state_idx}``.

    Parameters
    ----------
    new_means : ndarray, shape (n_new, n_features)
        Current-cycle emission means.
    prev_means : ndarray, shape (n_prev, n_features)
        Previous-cycle emission means.

    Returns
    -------
    dict[int, int]
        ``{new_state_index: prev_state_index}`` mapping.
    """
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


def _remap_to_prev_states(
    means: np.ndarray,
    raw_state: int,
    prev_means: np.ndarray,
    *,
    default: int = 0,
    return_component: int | None = None,
) -> int:
    """Remap a raw HMM state to the regime index from the previous cycle.

    Uses :func:`build_label_map` on *prev_means* to get the previous
    regime mapping, then routes *raw_state* through
    :func:`_match_states` to find its corresponding previous state.

    Parameters
    ----------
    means : ndarray, shape (n_states, n_features)
        Current-cycle emission means.
    raw_state : int
        Predicted raw latent state index from the current cycle.
    prev_means : ndarray, shape (n_prev, n_features)
        Previous-cycle emission means.
    default : int
        Fallback regime when raw_state has no match.
    return_component : int or None
        PCA component index for the return-signal dimension.
        When None, column 0 is used.

    Returns
    -------
    int
        Remapped regime index (0=Bear, 1=Sideways, 2=Bull).
    """
    sort_col = 0 if return_component is None else return_component
    # PCA may reduce n_components below the original feature index
    if sort_col >= prev_means.shape[1]:
        sort_col = 0

    prev_label_map = build_label_map(prev_means, sort_col)

    assignment = _match_states(means, prev_means)
    old_state = assignment.get(int(raw_state))
    if old_state is not None:
        return prev_label_map.get(old_state, default)

    return default
