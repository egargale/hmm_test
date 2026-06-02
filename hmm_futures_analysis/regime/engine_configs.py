"""Engine config dataclasses — pure data, no framework dependencies.

Each config encapsulates the constructor parameters for one engine.
The ``name`` and ``features`` fields are metadata (not constructor args)
used by the registry and pipeline respectively.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    name: str = "threshold"
    features: str = "returns"
    window: int = 20
    threshold: float = 0.05

    @property
    def is_hmm(self) -> bool:
        return False


@dataclass
class HMMGenericConfig:
    name: str = "hmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None

    @property
    def is_hmm(self) -> bool:
        return True


@dataclass
class HMMMMessinaConfig:
    name: str = "messina"
    features: str = "messina"
    n_states: int | str = 3
    pca_variance: float | None = None

    @property
    def is_hmm(self) -> bool:
        return True


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
