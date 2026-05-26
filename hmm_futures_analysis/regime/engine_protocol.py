"""RegimeEngine protocol, ClassifyResult, and ENGINE_REGISTRY."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class ClassifyResult:
    regime: int  # 0=bear, 1=sideways, 2=bull
    means: np.ndarray | None = None


@runtime_checkable
class RegimeEngine(Protocol):
    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None: ...
    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult: ...


def _build_registry() -> dict[str, type]:
    from .engines.hmm_generic import HMMGenericEngine
    from .engines.hmm_messina import HMMMMessinaEngine
    from .engines.threshold import ThresholdEngine

    return {
        "threshold": ThresholdEngine,
        "hmm": HMMGenericEngine,
        "messina": HMMMMessinaEngine,
    }


ENGINE_REGISTRY: dict[str, type] = _build_registry()
