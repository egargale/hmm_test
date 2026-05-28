"""HMM Messina-feature regime classification engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine_protocol import ClassifyResult
from ._hmm_shared import _fit_hmm_on_slice, _match_states, engineer_features


class HMMMMessinaEngine:
    def __init__(
        self,
        n_states: int = 3,
        pca_variance: float | None = None,
    ) -> None:
        self.n_states = n_states
        self.pca_variance = pca_variance
        self._pca_n_components: int | None = None

    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None:
        return engineer_features(data, use_messina=True)

    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult:
        features_clean = data.dropna()
        if len(features_clean) < self.n_states + 1:
            raise ValueError(
                f"Not enough clean rows ({len(features_clean)}) "
                f"for {self.n_states} HMM states"
            )
        features_arr = features_clean.to_numpy(dtype=np.float64)

        model, center, scale, pca_n, pca_transform = _fit_hmm_on_slice(
            features_arr, n_states=self.n_states, pca_variance=self.pca_variance,
        )

        # Sticky component count
        if pca_n is not None and self._pca_n_components is None:
            self._pca_n_components = pca_n

        means = model.means_

        # Transform last bar through z-score → (optional PCA) for prediction
        last_features = (features_arr[-1:] - center) / scale
        if pca_transform is not None:
            last_features = pca_transform.transform(last_features)
        raw_state = model.predict(last_features.astype(np.float64))[0]

        # Compute posteriors for the last observation
        posteriors = model.predict_proba(last_features.astype(np.float64))[-1]

        # Map HMM states to regime indices (0=bear, 1=sideways, 2=bull)
        # based on ascending mean return order, collapsed to 3 buckets
        state_means = means[:, 0]
        order = np.argsort(state_means)
        if self.n_states <= 3:
            label_map = {int(order[i]): i for i in range(len(order))}
        else:
            # Collapse n_states into 3 regimes: low/middle/high terciles
            n = len(order)
            label_map = {}
            for i, state_idx in enumerate(order):
                regime = min(2, i * 3 // n)
                label_map[int(state_idx)] = regime

        regime = label_map.get(int(raw_state), 1)

        # Reorder posteriors to match regime labels (0=bear, 1=sideways, 2=bull)
        if self.n_states <= 3:
            reordered = np.zeros(self.n_states)
            for state_idx in range(self.n_states):
                reordered[label_map[state_idx]] = posteriors[state_idx]
            posteriors = reordered
        else:
            agg = np.zeros(3)
            for state_idx in range(self.n_states):
                agg[label_map[state_idx]] += posteriors[state_idx]
            posteriors = agg

        if prev_means is not None:
            # Map through: raw_state -> matched old state -> old regime
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
                regime = prev_label_map.get(old_state, regime)

        return ClassifyResult(regime=int(regime), means=means, posteriors=posteriors)
