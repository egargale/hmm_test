"""RegimeEngine protocol, ClassifyResult, config dataclasses, and ENGINE_REGISTRY."""

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
    feature_saliency: np.ndarray | None = None
    selected_features: list[str] | None = None


@runtime_checkable
class RegimeEngine(Protocol):
    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None: ...
    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult: ...


# --- Config dataclasses (ADR-0004) ---


@dataclass
class ThresholdConfig:
    name: str = "threshold"
    features: str = "returns"
    window: int = 20
    threshold: float = 0.05

    @property
    def is_hmm(self) -> bool:
        return False

    def walk_forward_kwargs(self) -> dict:
        return {"window": self.window, "threshold": self.threshold}


@dataclass
class HMMGenericConfig:
    name: str = "hmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None

    @property
    def is_hmm(self) -> bool:
        return True

    def walk_forward_kwargs(self, n_states: int) -> dict:
        kwargs: dict = {"n_states": n_states}
        if self.pca_variance is not None:
            kwargs["pca_variance"] = self.pca_variance
        return kwargs


@dataclass
class HMMMMessinaConfig:
    name: str = "messina"
    features: str = "messina"
    n_states: int | str = 3
    pca_variance: float | None = None

    @property
    def is_hmm(self) -> bool:
        return True

    def walk_forward_kwargs(self, n_states: int) -> dict:
        kwargs: dict = {"n_states": n_states}
        if self.pca_variance is not None:
            kwargs["pca_variance"] = self.pca_variance
        return kwargs


@dataclass
class RobustHMMConfig:
    name: str = "robust_hmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None
    robust_method: str = "huber"

    @property
    def is_hmm(self) -> bool:
        return True

    def walk_forward_kwargs(self, n_states: int) -> dict:
        kwargs: dict = {"n_states": n_states}
        if self.pca_variance is not None:
            kwargs["pca_variance"] = self.pca_variance
        kwargs["robust_method"] = self.robust_method
        return kwargs


@dataclass
class FSHMMConfig:
    name: str = "fshmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None
    saliency_threshold: float = 0.5

    @property
    def is_hmm(self) -> bool:
        return True

    def walk_forward_kwargs(self, n_states: int) -> dict:
        kwargs: dict = {"n_states": n_states}
        if self.pca_variance is not None:
            kwargs["pca_variance"] = self.pca_variance
        kwargs["saliency_threshold"] = self.saliency_threshold
        return kwargs


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


def resolve_engine(config) -> RegimeEngine:
    """Construct an engine from a config dataclass."""
    import dataclasses

    name = getattr(config, "name", None)
    if name not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown engine name {name!r}. Available: {sorted(ENGINE_REGISTRY.keys())}"
        )
    engine_cls, _ = ENGINE_REGISTRY[name]
    # Strip name/features — they're config metadata, not constructor params
    kwargs = {
        k: v
        for k, v in dataclasses.asdict(config).items()
        if k not in ("name", "features")
    }
    return engine_cls(**kwargs)


ENGINE_REGISTRY: dict[str, tuple[type, type]] = _LazyRegistry()

# Engine sets derived from the registry.
# HMM_ENGINES: engines that fit a GaussianHMM and provide posteriors.
HMM_ENGINES: frozenset[str] = frozenset({"messina", "hmm", "robust_hmm", "fshmm"})
