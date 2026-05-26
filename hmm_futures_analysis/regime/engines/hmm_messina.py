"""HMM Messina-feature regime classification engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine_protocol import ClassifyResult
from ._hmm_shared import _fit_hmm_on_slice, _match_states, engineer_features


class HMMMMessinaEngine:
    def __init__(self, n_states: int = 3) -> None:
        self.n_states = n_states

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
        model, center, scale = _fit_hmm_on_slice(features_arr, n_states=self.n_states)
        means = model.means_
        last_features = (features_arr[-1:] - center) / scale
        raw_state = model.predict(last_features.astype(np.float64))[0]

        if prev_means is not None:
            assignment = _match_states(means, prev_means)
            regime = assignment.get(int(raw_state), 1)
        else:
            state_means = means[:, 0]
            order = np.argsort(state_means)
            label_map = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
            regime = label_map.get(int(raw_state), 1)

        return ClassifyResult(regime=int(regime), means=means)
