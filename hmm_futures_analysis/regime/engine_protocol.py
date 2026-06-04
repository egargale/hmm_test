"""RegimeEngine protocol, result types, ENGINE_REGISTRY, and engine factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .engine_configs import (
    FSHMMConfig,
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)


@dataclass
class ClassifyResult:
    regime: int  # 0=bear, 1=sideways, 2=bull
    means: np.ndarray | None = None
    posteriors: np.ndarray | None = None
    feature_saliency: np.ndarray | None = None
    selected_features: list[str] | None = None


@dataclass
class ClassifyOutput:
    """Intermediate state from the classify phase."""

    regimes: np.ndarray
    posteriors: np.ndarray | None = None
    last_regime: int = 1
    warmup_bars: int | None = None
    n_states: int = 3
    engine_info: dict | None = None


@runtime_checkable
class RegimeEngine(Protocol):
    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None: ...
    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult: ...
    def run_classify(
        self,
        prices: pd.Series,
        ohlcv: pd.DataFrame | None,
        returns: pd.Series,
        min_train: int,
        **kwargs,
    ) -> ClassifyOutput: ...


# ---------------------------------------------------------------------------
# Engine registry — lazy to avoid circular imports
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, tuple[type, type]]:
    from .engines.fshmm import FSHMMEngine
    from .engines.hmm_generic import HMMGenericEngine
    from .engines.hmm_messina import HMMMMessinaEngine
    from .engines.robust_hmm import RobustHMMEngine
    from .engines.threshold import ThresholdEngine

    return {
        "threshold": (ThresholdEngine, ThresholdConfig),
        "hmm": (HMMGenericEngine, HMMGenericConfig),
        "messina": (HMMMMessinaEngine, HMMMMessinaConfig),
        "robust_hmm": (RobustHMMEngine, RobustHMMConfig),
        "fshmm": (FSHMMEngine, FSHMMConfig),
    }


class _LazyRegistry:
    """Lazily-built engine registry. Resolves on first access.

    Exists to avoid a circular import: configs import from
    ``.engine_configs``, engines import from here, and the registry
    references both.  Building on first access defers the import
    until after all modules are loaded.
    """

    _cache: dict | None = None

    @classmethod
    def _data(cls) -> dict:
        if cls._cache is None:
            cls._cache = _build_registry()
        return cls._cache

    def __contains__(self, key: object) -> bool:
        return key in self._data()

    def __getitem__(self, key: str) -> tuple:
        return self._data()[key]

    def keys(self):
        return self._data().keys()

    def items(self):
        return self._data().items()

    def __len__(self) -> int:
        return len(self._data())


ENGINE_REGISTRY: dict[str, tuple[type, type]] = _LazyRegistry()  # type: ignore[assignment]


def resolve_engine(config) -> RegimeEngine:
    """Construct an engine from a config dataclass."""
    import dataclasses

    name = getattr(config, "name", None)
    if name not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown engine name {name!r}. Available: {sorted(ENGINE_REGISTRY.keys())}"
        )
    engine_cls, _ = ENGINE_REGISTRY[name]
    # Strip name/features/defaults — they're config metadata, not constructor params
    kwargs = {
        k: v
        for k, v in dataclasses.asdict(config).items()
        if k
        not in (
            "name",
            "features",
            "default_dwell_bars",
            "default_hysteresis_delta",
        )
    }
    return engine_cls(**kwargs)


# Engine sets derived from the registry.
# HMM_ENGINES: engines that fit a GaussianHMM and provide posteriors.
HMM_ENGINES: frozenset[str] = frozenset({"messina", "hmm", "robust_hmm", "fshmm"})
