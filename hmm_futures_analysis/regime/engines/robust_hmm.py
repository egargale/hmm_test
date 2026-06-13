"""Robust HMM regime classification engine with outlier-resistant emission estimation."""

from __future__ import annotations

import numpy as np

from ._hmm_engine import HMMEngineBase, robust_fit_gaussian_hmm
from ._feature_set import GenericFeatureSet


class RobustHMMEngine(HMMEngineBase):
    """HMM engine with Huber or MCD robust emission correction."""

    featureset = GenericFeatureSet()
    use_messina = False

    def __init__(
        self,
        n_states: int = 3,
        pca_variance: float | None = None,
        reverse_classify: bool = False,
        robust_method: str = "huber",
        default_refit_every: int = 100,
    ) -> None:
        super().__init__(
            n_states=n_states,
            pca_variance=pca_variance,
            reverse_classify=reverse_classify,
            default_refit_every=default_refit_every,
        )
        self.robust_method = robust_method

    def _build_engine_info(self, warmup_bars: int | None = None) -> dict:
        result = super()._build_engine_info(warmup_bars=warmup_bars)
        result["robust_method"] = self.robust_method
        return result

    def _fit_on_slice(self, features_arr, n_states):
        """Fit HMM with post-hoc robust emission correction."""
        return robust_fit_gaussian_hmm(
            features_arr,
            n_states=n_states,
            pca_variance=self.pca_variance,
            robust_method=self.robust_method,
        )
