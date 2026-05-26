"""Threshold-based regime classification engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine_protocol import ClassifyResult
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
