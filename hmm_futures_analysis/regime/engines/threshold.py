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

    def classify_pipeline(
        self,
        prices: pd.Series,
        ohlcv: pd.DataFrame | None,
        returns: pd.Series,
        min_train: int = 252,
        *,
        profile: bool = True,
        _phases: dict[str, float] | None = None,
        _classify_times: list[float] | None = None,
    ) -> ClassifyOutput:
        regimes = _classify_regimes(
            returns, window=self.window, threshold=self.threshold
        )
        return ClassifyOutput(
            regimes=regimes,
            posteriors=None,
            last_regime=int(regimes[-1]),
            warmup_bars=None,
            engine_instance=self,
        )
