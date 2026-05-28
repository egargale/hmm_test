"""HMM generic-feature regime classification engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine_protocol import ClassifyResult
from ._hmm_shared import _fit_hmm_on_slice, _match_states, engineer_features


class HMMGenericEngine:
    def __init__(self, n_states: int = 3) -> None:
        self.n_states = n_states

    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None:
        return engineer_features(data, use_messina=False)

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
        model, center, scale = _fit_hmm_on_slice(features_arr, n_states=self.n_states)
        means = model.means_
        last_features = (features_arr[-1:] - center) / scale
        raw_state = model.predict(last_features.astype(np.float64))[0]

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
                # Bottom third → bear(0), middle → sideways(1), top → bull(2)
                regime = min(2, i * 3 // n)
                label_map[int(state_idx)] = regime

        regime = label_map.get(int(raw_state), 1)

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

        return ClassifyResult(regime=int(regime), means=means)
