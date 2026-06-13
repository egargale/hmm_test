"""HMM generic-feature regime classification engine."""

from __future__ import annotations

from ._hmm_engine import HMMEngineBase
from ._feature_set import GenericFeatureSet


class HMMGenericEngine(HMMEngineBase):
    """HMM engine using generic (~50) engineered features."""

    featureset = GenericFeatureSet()
    use_messina = False
