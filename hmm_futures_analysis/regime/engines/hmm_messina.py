"""HMM Messina-feature regime classification engine."""

from __future__ import annotations

from ._hmm_engine import HMMEngineBase


class HMMMMessinaEngine(HMMEngineBase):
    """HMM engine using the 19 Messina features."""

    use_messina = True
