"""Threshold-based regime classification engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine_protocol import ClassifyOutput, ClassifyResult
from ...regime.markov_chain import classify_regimes as _classify_regimes


class ThresholdEngine:
    def __init__(self, window: int = 20, threshold: float = 0.05) -> None:
        self.window = window
        self.threshold = threshold

    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None:
        return None

    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult:
        regimes = _classify_regimes(data, window=self.window, threshold=self.threshold)
        return ClassifyResult(regime=int(regimes[-1]))

    def run_classify(
        self,
        prices: pd.Series,
        ohlcv: pd.DataFrame | None,
        returns: pd.Series,
        min_train: int,
        **kwargs,
    ) -> ClassifyOutput:
        """One-shot vectorized regime classification (no walk-forward)."""
        regimes = _classify_regimes(
            returns, window=self.window, threshold=self.threshold
        )
        return ClassifyOutput(
            regimes=regimes,
            posteriors=None,
            last_regime=int(regimes[-1]),
            warmup_bars=None,
            n_states=3,
        )
