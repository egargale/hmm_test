"""HMM generic-feature regime classification engine."""

from __future__ import annotations

from ._hmm_engine import HMMEngineBase


class HMMGenericEngine(HMMEngineBase):
    """HMM engine using generic (~50) engineered features."""

    use_messina = False
