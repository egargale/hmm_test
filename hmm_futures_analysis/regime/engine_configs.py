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
    default_dwell_bars: int = 3
    default_hysteresis_delta: float = 0.0

    @property
    def is_hmm(self) -> bool:
        return False


@dataclass
class HMMGenericConfig:
    name: str = "hmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None
    reverse_classify: bool = False
    default_refit_every: int = 50
    default_dwell_bars: int = 0
    default_hysteresis_delta: float = 0.0

    @property
    def is_hmm(self) -> bool:
        return True


@dataclass
class HMMMMessinaConfig:
    name: str = "messina"
    features: str = "messina"
    n_states: int | str = 3
    pca_variance: float | None = 0.95
    reverse_classify: bool = False
    default_refit_every: int = 50
    default_dwell_bars: int = 0
    default_hysteresis_delta: float = 0.0

    @property
    def is_hmm(self) -> bool:
        return True


@dataclass
class RobustHMMConfig:
    name: str = "robust_hmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = 0.90
    reverse_classify: bool = False
    robust_method: str = "huber"
    default_refit_every: int = 100
    default_dwell_bars: int = 0
    default_hysteresis_delta: float = 0.0

    @property
    def is_hmm(self) -> bool:
        return True


@dataclass
class FSHMMConfig:
    name: str = "fshmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None
    reverse_classify: bool = False
    saliency_threshold: float = 0.5
    default_refit_every: int = 100
    default_dwell_bars: int = 2
    default_hysteresis_delta: float = 0.05

    @property
    def is_hmm(self) -> bool:
        return True
