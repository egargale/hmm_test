"""Robust HMM regime classification engine with outlier-resistant emission estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine_protocol import ClassifyResult
from ._hmm_engine import _classify_hmm_slice, engineer_features, robust_fit_gaussian_hmm


class RobustHMMEngine:
    def __init__(
        self,
        n_states: int = 3,
        pca_variance: float | None = None,
        robust_method: str = "huber",
    ) -> None:
        self.n_states = n_states
        self.pca_variance = pca_variance
        self.robust_method = robust_method
        self._pca_n_components: int | None = None

    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None:
        if data is None:
            raise ValueError(
                "RobustHMMEngine requires OHLCV data for feature engineering"
            )
        return engineer_features(data, use_messina=False)

    def enrich_info(self, info: dict) -> dict:
        result = {**info}
        result["caveat"] = "HMM states sorted by mean return; labels may swap on re-fit"
        result["robust_method"] = self.robust_method
        return result

    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult:
        features_clean = data.bfill().dropna()
        if len(features_clean) < self.n_states + 1:
            raise ValueError(
                f"Not enough clean rows ({len(features_clean)}) "
                f"for {self.n_states} HMM states"
            )
        features_arr = features_clean.to_numpy(dtype=np.float64)

        model, center, scale, pca_n, pca_transform = robust_fit_gaussian_hmm(
            features_arr,
            n_states=self.n_states,
            pca_variance=self.pca_variance,
            robust_method=self.robust_method,
        )

        if pca_n is not None and self._pca_n_components is None:
            self._pca_n_components = pca_n

        # Normalize last bar for prediction
        X_last = (features_arr[-1:] - center) / scale
        if pca_transform is not None:
            X_last = pca_transform.transform(X_last)

        return _classify_hmm_slice(
            model,
            X_last,
            self.n_states,
            prev_means,
        )
