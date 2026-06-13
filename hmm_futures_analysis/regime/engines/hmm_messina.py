"""HMM Messina-feature regime classification engine."""

from __future__ import annotations

from ._hmm_engine import HMMEngineBase
from ._feature_set import MessinaFeatureSet


class HMMMMessinaEngine(HMMEngineBase):
    """HMM engine using the 19 Messina features."""

    featureset = MessinaFeatureSet()
    use_messina = True
