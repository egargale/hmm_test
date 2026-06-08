"""HMM engine building blocks: feature engineering, fitting, state matching.

Provides the abstract :class:`HMMEngineBase` shared by the four HMM engine
classes (hmm_generic, hmm_messina, robust_hmm, fshmm), standalone helper
functions used by those engines, and pipeline-level orchestration helpers
in ``_hmm_pipeline.py``.
"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hmmlearn import hmm

from ...data_processing.feature_engineering import add_features
from ...data_processing.messina_features import MESSINA_FEATURE_COLUMNS
from ...data_processing.messina_features import add_messina_features
from ..engine_protocol import ClassifyOutput, ClassifyResult

if TYPE_CHECKING:
    from sklearn.decomposition import PCA


class HMMEngineBase(ABC):
    """Abstract base for HMM-backed regime engines.

    Provides default ``__init__``, ``precompute``, ``_build_engine_info``, and
    ``run_classify``.  Concrete engines override ``classify()`` (and
    optionally ``__init__``, ``precompute``, ``_build_engine_info``) to inject
    their differentiated logic.

    Parameters
    ----------
    n_states : int or 'auto'
        Number of hidden Markov states (or ``'auto'`` for BIC selection).
    pca_variance : float or None
        Fraction of variance to retain via PCA whitening, or ``None`` to
        skip PCA.
    """

    # Subclasses set True if they use Messina features.
    use_messina: bool = False

    def __init__(
        self,
        n_states: int = 3,
        pca_variance: float | None = None,
        reverse_classify: bool = False,
        default_refit_every: int = 50,
    ) -> None:
        self.n_states = n_states
        self.pca_variance = pca_variance
        self.reverse_classify = reverse_classify
        self.default_refit_every = default_refit_every
        self._pca_n_components: int | None = None
        self._pca_return_component: int | None = None

    # ------------------------------------------------------------------
    # Protected hook: subclasses override to inject custom fitting.
    # ------------------------------------------------------------------

    def _fit_on_slice(
        self, features_arr: np.ndarray, n_states: int
    ) -> tuple[hmm.GaussianHMM, np.ndarray, np.ndarray, int | None, PCA | None]:
        """Fit HMM on a feature slice. Subclasses override for custom fitting.

        The default implementation delegates to :func:`_fit_hmm_on_slice`
        with this engine's ``pca_variance``.  RobustHMMEngine overrides this
        to call :func:`robust_fit_gaussian_hmm` instead.
        """
        return _fit_hmm_on_slice(
            features_arr,
            n_states=n_states,
            pca_variance=self.pca_variance,
        )

    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """Engineer features from OHLCV data.

        Uses :func:`engineer_features` with the subclass' ``use_messina`` flag.
        """
        if data is None:
            raise ValueError(
                f"{type(self).__name__} requires OHLCV data for feature engineering"
            )
        return engineer_features(data, use_messina=self.use_messina)

    def _build_engine_info(self, warmup_bars: int | None = None) -> dict:
        """Return engine-specific metadata for ClassifyOutput.engine_info.

        Base implementation adds the standard HMM caveat and warmup_bars
        when available. Subclasses override to add extra keys.
        """
        info: dict[str, object] = {
            "caveat": ("HMM states sorted by mean return; labels may swap on re-fit"),
        }
        if warmup_bars is not None:
            info["warmup_bars"] = warmup_bars
        return info

    def run_classify(
        self,
        prices: pd.Series,
        ohlcv: pd.DataFrame | None,
        returns: pd.Series,
        min_train: int,
        **kwargs,
    ) -> ClassifyOutput:
        """Delegate to the shared HMM walk-forward pipeline."""
        from ._hmm_pipeline import _hmm_classify_pipeline

        result = _hmm_classify_pipeline(
            self, prices, ohlcv, returns, min_train, **kwargs
        )
        result.engine_info = {
            **(result.engine_info or {}),
            **self._build_engine_info(warmup_bars=result.warmup_bars),
        }
        return result

    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult:
        """Fit HMM on *data* and classify the last bar.

        Shared implementation for engines that follow the standard
        z-score → (PCA) → fit → classify pipeline.  Engines that need
        fundamentally different logic (e.g. FSHMM with its custom EM
        and saliency metadata) override this method entirely.
        """
        n_states = getattr(self, "_n_states_resolved", self.n_states)
        features_clean = data.bfill().dropna()
        if len(features_clean) < n_states + 1:
            raise ValueError(
                f"Not enough clean rows ({len(features_clean)}) "
                f"for {n_states} HMM states"
            )
        features_arr = features_clean.to_numpy(dtype=np.float64)

        model, center, scale, pca_n, pca_transform = self._fit_on_slice(
            features_arr,
            n_states=n_states,
        )

        # Sticky component count: first refit determines, subsequent reuse
        if pca_n is not None and self._pca_n_components is None:
            self._pca_n_components = pca_n

        # PCA-aware return component: when PCA is active, determine which
        # component is most correlated with log_ret (original column 0).
        return_component: int | None = None
        if pca_transform is not None:
            return_component = int(np.argmax(np.abs(pca_transform.components_[:, 0])))
            if self._pca_return_component is None:
                self._pca_return_component = return_component

        # Normalize last bar for prediction
        X_last = (features_arr[-1:] - center) / scale
        if pca_transform is not None:
            X_last = pca_transform.transform(X_last)

        return _classify_hmm_slice(
            model,
            X_last,
            n_states,
            prev_means,
            return_component=return_component,
        )


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
    return_component: int | None = None,
) -> ClassifyResult:
    """Shared post-fit classify pipeline for HMM engines.

    Pipeline: model.predict(X_last) → label map (sort means ascending,
    collapse to 3 regimes if n_states > 3) → posteriors reorder/aggregate
    → _remap_to_prev_states if prev_means given.

    Sorting dimension
    -----------------
    When *return_component* is None (the default), HMM states are sorted
    by column 0 of the means matrix.  Without PCA, column 0 is ``log_ret``
    -- the return-signal dimension -- and ascending order yields a
    reliable bear→sideways→bull mapping.

    When *return_component* is an int, states are sorted by
    ``means[:, return_component]`` instead.  This is the PCA-aware path:
    the caller determines the component whose loadings are most correlated
    with the original ``log_ret`` feature (by inspecting
    ``pca_transform.components_[:, 0]``) and passes the index here.

    Parameters
    ----------
    model : pre-fit GaussianHMM
    X_last : already-normalized last observation, shape (1, n_features)
        Caller is responsible for z-scoring and optional PCA transform.
    n_states : number of HMM states
    prev_means : previous-cycle means for state remapping, or None
    return_component : optional int index of the PCA component most
        correlated with the return signal.  When None, column 0 is used
        (backward-compatible, correct without PCA).
    """
    means = model.means_

    raw_state = model.predict(X_last.astype(np.float64))[0]
    posteriors = model.predict_proba(X_last.astype(np.float64))[-1]

    # Map HMM states to regime indices (0=bear, 1=sideways, 2=bull)
    # based on ascending mean return order, collapsed to 3 buckets.
    # Always produce 3 regime buckets regardless of n_states so that
    # downstream posteriors arrays have consistent shape.
    sort_col = 0 if return_component is None else return_component
    # PCA may reduce n_components below the original feature index
    if sort_col >= means.shape[1]:
        sort_col = 0
    state_means = means[:, sort_col]
    order = np.argsort(state_means)
    n_actual = len(order)  # may differ from n_states if model converged
    #    with fewer states than requested

    if n_actual == 1:
        # Degenerate: single state → sideways (regime 1)
        label_map = {int(order[0]): 1}
    elif n_actual == 2:
        # Two states: map to bear (lowest mean) and bull (highest mean).
        # No sideways regime — the two states span the full range.
        label_map = {
            int(order[0]): 0,  # bear
            int(order[1]): 2,  # bull
        }
    elif n_actual == 3:
        label_map = {int(order[i]): i for i in range(3)}
    else:
        label_map = {}
        for i, state_idx in enumerate(order):
            label_map[int(state_idx)] = min(2, i * 3 // n_actual)

    regime = label_map.get(int(raw_state), 1)

    # Always aggregate posteriors into 3-element regime-bucket array
    agg = np.zeros(3)
    for state_idx in range(n_actual):
        bucket = label_map.get(int(state_idx), 1)
        agg[bucket] += posteriors[state_idx]
    posteriors = agg

    if prev_means is not None:
        regime = _remap_to_prev_states(
            means,
            raw_state,
            prev_means,
            default=regime,
            return_component=return_component,
        )

    return ClassifyResult(regime=int(regime), means=means, posteriors=posteriors)


def _remap_to_prev_states(
    means: np.ndarray,
    raw_state: int,
    prev_means: np.ndarray,
    *,
    default: int = 0,
    return_component: int | None = None,
) -> int:
    """Remap a raw HMM state to the regime index from the previous cycle.

    Sorts prev_means by the return-signal dimension (column 0 when
    *return_component* is None, or the specified PCA component otherwise).
    Builds a label map (identity for ≤3 states, collapsed to 3 regimes
    for >3), then maps raw_state through _match_states and the prev label
    map.

    Args:
        means: Current-cycle HMM means (n_states × n_features).
        raw_state: Predicted raw latent state index.
        prev_means: Previous-cycle HMM means (prev_n × n_features).
        default: Fallback regime when raw_state has no match.
        return_component: Optional int index of the PCA component most
            correlated with returns. When None, column 0 is used.

    Returns:
        Remapped regime index (0=bear, 1=sideways, 2=bull).
    """
    sort_col = 0 if return_component is None else return_component
    # PCA may reduce n_components below the original feature index
    if sort_col >= prev_means.shape[1]:
        sort_col = 0
    prev_order = np.argsort(prev_means[:, sort_col])
    prev_n = len(prev_order)

    if prev_n == 1:
        prev_label_map = {int(prev_order[0]): 1}
    elif prev_n == 2:
        prev_label_map = {
            int(prev_order[0]): 0,
            int(prev_order[1]): 2,
        }
    elif prev_n == 3:
        prev_label_map = {int(prev_order[i]): i for i in range(3)}
    else:
        prev_label_map = {}
        for i, si in enumerate(prev_order):
            prev_label_map[int(si)] = min(2, i * 3 // prev_n)

    assignment = _match_states(means, prev_means)
    old_state = assignment.get(int(raw_state))
    if old_state is not None:
        return prev_label_map.get(old_state, default)

    return default
