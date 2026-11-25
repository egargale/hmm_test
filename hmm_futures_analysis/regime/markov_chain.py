import numpy as np
import pandas as pd


def classify_regimes(
    returns: pd.Series, window: int = 20, threshold: float = 0.05
) -> np.ndarray:
    """Classify each bar as Bear(0), Sideways(1), or Bull(2) based on rolling return."""
    if len(returns) < window:
        return np.ones(len(returns), dtype=int)

    rolling_ret = returns.rolling(window=window).sum()
    regimes = np.ones(len(returns), dtype=int)

    first_valid = rolling_ret.first_valid_index()
    if first_valid is None:
        # All NaN in rolling result — default to sideways
        return regimes
    start_loc = rolling_ret.index.get_loc(first_valid)
    regimes[start_loc:] = np.where(
        rolling_ret.iloc[start_loc:] > threshold,
        2,
        np.where(rolling_ret.iloc[start_loc:] < -threshold, 0, 1),
    )
    return regimes


def build_transition_matrix(
    regimes: np.ndarray, n_states: int = 3
) -> np.ndarray:
    """Build n_states x n_states transition matrix from regime sequence."""
    trans = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(len(regimes) - 1):
        s_from = int(regimes[i])
        s_to = int(regimes[i + 1])
        if 0 <= s_from < n_states and 0 <= s_to < n_states:
            trans[s_from, s_to] += 1

    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans /= row_sums
    trans += 1e-10
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def compute_stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute stationary distribution (left eigenvector with eigenvalue 1)."""
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.abs(eigenvectors[:, idx].real)
    stationary /= stationary.sum()
    return stationary


def compute_persistence_diagonal(transmat: np.ndarray) -> dict:
    """Extract persistence (stickiness) for each state.

    Keys are positional and capped at the matrix size: the standard 3×3
    regime-space matrix returns bear/sideways/bull exactly as before, while a
    smaller matrix (e.g. a 2×2 post-degenerate-recovery shape, issue #112)
    returns one key per available diagonal element instead of raising
    IndexError.
    """
    diag = np.diag(transmat)
    keys = ("bear", "sideways", "bull")
    return {
        name: float(diag[i]) for i, name in enumerate(keys) if i < len(diag)
    }


def compute_signal(next_state_probs: np.ndarray) -> float:
    """Compute directional signal: P(last) - P(first) in [-1, 1].

    For the standard 3-state vector this is P(bull) - P(bear); reading the
    last/first entries keeps the 3×3 value identical and stays defined for a
    2-state vector (issue #112).
    """
    return float(next_state_probs[-1] - next_state_probs[0])


def forecast_n_steps(
    transmat: np.ndarray, current_probs: np.ndarray, n: int
) -> np.ndarray:
    """Forecast regime probabilities n steps ahead via matrix power."""
    result = transmat.copy()
    power = n
    accum = np.eye(transmat.shape[0], dtype=np.float64)
    while power > 0:
        if power % 2 == 1:
            accum = accum @ result
        result = result @ result
        power //= 2
    return current_probs @ accum
