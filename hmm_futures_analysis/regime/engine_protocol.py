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
    posteriors: np.ndarray | None = None


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


class _LazyRegistry(dict):
    """Dict subclass that lazily builds on first access."""

    def __init__(self) -> None:
        super().__init__()
        self._built = False

    def _ensure_built(self) -> None:
        if not self._built:
            self._built = True
            self.update(_build_registry())

    def __contains__(self, key: object) -> bool:
        self._ensure_built()
        return super().__contains__(key)

    def __getitem__(self, key: str) -> type:
        self._ensure_built()
        return super().__getitem__(key)

    def __iter__(self):
        self._ensure_built()
        return super().__iter__()

    def keys(self):
        self._ensure_built()
        return super().keys()

    def values(self):
        self._ensure_built()
        return super().values()

    def items(self):
        self._ensure_built()
        return super().items()

    def __len__(self) -> int:
        self._ensure_built()
        return super().__len__()


ENGINE_REGISTRY: dict[str, type] = _LazyRegistry()
